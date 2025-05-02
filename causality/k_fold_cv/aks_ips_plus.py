#%%
import os 
import sys
import torch
import argparse
import subprocess
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.model import NCF,  NCF_AKBIPS_ExpPlus, LinearCF_AKBIPS_ExpPlus
from module.dataset import load_data, generate_total_sample
from module.utils import set_device, set_seed

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%%
parser = argparse.ArgumentParser()

parser.add_argument("--lr1", type=float, default=0.01)
parser.add_argument("--lamb1", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="original")
parser.add_argument("--G", type=int, default=1)
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr2", type=float, default=0.05)
parser.add_argument("--lr3", type=float, default=0.05)
parser.add_argument("--lamb2", type=float, default=0.)
parser.add_argument("--lamb3", type=float, default=0.)
parser.add_argument("--J", type=int, default=3)
parser.add_argument("--gamma", type=float, default=1.)
parser.add_argument("--C", type=float, default=1e-5)
parser.add_argument("--num-w-epo", type=int, default=3)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../data")
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--base-model", type=str, default="ncf") # ["ncf","linearcf"]
parser.add_argument("--device", type=str, default="none")

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

embedding_k = args.embedding_k
lr1 = args.lr1
lr2 = args.lr2
lr3 = args.lr3
lamb1 = args.lamb1
lamb2 = args.lamb2
lamb3 = args.lamb3
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name
gamma = args.gamma
G = args.G
C = args.C
num_w_epo = args.num_w_epo
J = args.J
base_model = args.base_model
device = args.device


expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device(device)

x_train, x_test = load_data(data_dir, dataset_name)
x_train_cv, y_train_cv, t_train_cv, ps_train_cv = x_train[:,:2].astype(int), x_train[:,2:3], x_train[:,3:4], x_train[:,4:]

num_users = int(x_train_cv[:,0].max())+1
num_items = int(x_train[:,1].max())+1
print(f"# user: {num_users}, # item: {num_items}")

kf = KFold(n_splits=4, shuffle=True, random_state=random_seed)
for cv_num, (train_idx, test_idx) in enumerate(kf.split(x_train)):

    if cv_num > 1:
        continue

    configs = vars(args)
    configs["device"] = device
    configs["cv_num"] = cv_num
    wandb_var = wandb.init(project="no_ips", config=configs)
    wandb.run.name = f"cv_akb_ips_plus_causality_{expt_num}"


    x_train = x_train_cv[train_idx]
    y_train = y_train_cv[train_idx]
    t_train = t_train_cv[train_idx]
    ps_train = ps_train_cv[train_idx]

    x_test = x_train_cv[test_idx]
    y_test = y_train_cv[test_idx]
    t_test = t_train_cv[test_idx]
    ps_test = ps_train_cv[test_idx]

    x_all = generate_total_sample(num_users, num_items)

    y1_train = y_train[t_train==1]
    x1_train = x_train[np.squeeze(t_train==1)]
    ps1_train = ps_train[t_train==1]
    y1_test = y_test[t_test==1]
    x1_test = x_test[np.squeeze(t_test==1)]

    obs1 = sps.csr_matrix((np.ones(len(y1_train)), (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    y1_entire = sps.csr_matrix((y1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    ps1_entire = sps.csr_matrix((ps1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)

    y0_train = y_train[t_train==0]
    x0_train = x_train[np.squeeze(t_train==0)]
    ps0_train = 1-ps_train[t_train==0]
    y0_test = y_test[t_test==0]
    x0_test = x_test[np.squeeze(t_test==0)]

    obs0 = sps.csr_matrix((np.ones(len(y0_train)), (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    y0_entire = sps.csr_matrix((y0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    ps0_entire = sps.csr_matrix((ps0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)

    num_samples = len(x1_train)
    total_batch = num_samples // batch_size

    x1_test_tensor = torch.LongTensor(x1_test).to(device)
    x0_test_tensor = torch.LongTensor(x0_test).to(device)

    ps_model = NCF(num_users, num_items, embedding_k)
    ps_model = ps_model.to(device)
    optimizer = torch.optim.Adam(ps_model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fcn = torch.nn.BCELoss()

    for epoch in range(1, num_epochs+1):
        ul_idxs = np.arange(x_all.shape[0]) # all
        np.random.shuffle(ul_idxs)
        ps_model.train()

        epoch_select_loss = 0.

        for idx in range(total_batch):

            selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(device)
            sub_t = obs1[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

            pred, user_embed, item_embed = ps_model(sub_x)

            select_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_t)
            epoch_select_loss += select_loss

            optimizer.zero_grad()
            select_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch:>4d} Train Propensity Loss] select: {epoch_select_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_select_loss': float(epoch_select_loss.item()),
        }

        wandb_var.log(loss_dict)


    if base_model == "ncf":
        model = NCF_AKBIPS_ExpPlus(num_users, num_items, embedding_k)
    elif base_model == "linearcf":
        model = LinearCF_AKBIPS_ExpPlus(num_users, num_items, embedding_k)

    model = model.to(device)

    optimizer_prediction = torch.optim.Adam(
        model.prediction_model.parameters(), lr=lr1, weight_decay=lamb1)
    optimizer_weight = torch.optim.Adam(
        model.weight_model.parameters(), lr=lr2, weight_decay=lamb2)
    optimizer_epo = torch.optim.Adam(
        [model.epsilon], lr=lr3, weight_decay=lamb3)

    ps_model.eval()
    loss_epsilon_last = 0
    for epoch in range(1, num_epochs+1):
        all_idx = np.arange(num_samples)
        np.random.shuffle(all_idx)
        model.train()

        ul_idxs = np.arange(x_all.shape[0])

        epoch_pred_y1_loss = 0.
        epoch_pred_y0_loss = 0.
        epoch_weight_model_loss = 0.
        epoch_total_loss = 0.

        for idx in range(total_batch):

            sampled_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sampled_x1 = x1_train[sampled_idx]
            sampled_x1 = torch.LongTensor(sampled_x1).to(device)
            sampled_y1 = y1_train[sampled_idx]
            sampled_y1 = torch.Tensor(sampled_y1).unsqueeze(-1).to(device)

            selected_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(device)

            sub_t = obs1[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

            sub_y1 = y1_entire[selected_idx]
            sub_y1 = torch.Tensor(sub_y1).unsqueeze(-1).to(device)
            sub_y0 = y0_entire[selected_idx]
            sub_y0 = torch.Tensor(sub_y0).unsqueeze(-1).to(device)

            prop, _, __ = ps_model(sampled_x1)
            inv_prop = 1 / nn.Sigmoid()(prop).detach()

            pred, _ = model.prediction_model.forward(sampled_x1)
            e_loss = F.binary_cross_entropy(nn.Sigmoid()(pred), sampled_y1, reduction='none')

            feature_emd_o = model.get_embedding(sampled_x1)
            feature_emd_d = model.get_embedding(sub_x)

            loss_epsilon = (((e_loss - torch.mm(model.epsilon, model.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
            if (loss_epsilon_last - loss_epsilon)/(loss_epsilon_last + 1e-10) < 0.2:
                for i in range(num_w_epo):
                    loss_epsilon = (((e_loss - torch.mm(model.epsilon, model.exp_kernel(feature_emd_o,feature_emd_d)).squeeze()) ** 2) * inv_prop).sum()
                    optimizer_epo.zero_grad()
                    loss_epsilon.backward(retain_graph=True)
                    optimizer_epo.step()
            loss_epsilon_last =  loss_epsilon  

            topj_values, topj_indices = torch.topk(torch.abs(model.epsilon), k=J)
            topj_indices = topj_indices.cpu().numpy().tolist()

            w_, _, __ = model.weight_model(sampled_x1)
            w = nn.Sigmoid()(w_).detach() * inv_prop
                    
            feature_emd_j = model.get_embedding(sub_x[topj_indices[0]])                       

            h_all = torch.mean(model.exp_kernel(feature_emd_j, feature_emd_d), axis =0)
            w_tran = (w.unsqueeze(1)).T
            h_obs = 1/len(w) * (torch.mm(w_tran[0], model.exp_kernel(feature_emd_j, feature_emd_o))).squeeze()

            zero = torch.zeros(1).to(device)
            loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
            weight_model_loss =  1/len(w) * torch.dot(w[0], torch.log(w)[0]) + gamma * loss_ker_bal

            optimizer_weight.zero_grad()
            weight_model_loss.backward()
            optimizer_weight.step()

            prop, _, __ = ps_model(sub_x)
            w_, _, __ = model.weight_model(sub_x)
            pred_y1, pred_y0 = model.prediction_model(sub_x)

            inv_prop = 1/nn.Sigmoid()(prop).detach()
            w1 = nn.Sigmoid()(w_).detach() * inv_prop
            inv_prop = 1/(1-nn.Sigmoid()(prop).detach())
            w0 = (1-nn.Sigmoid()(w_).detach()) * inv_prop

            pred_y1_loss = F.binary_cross_entropy(nn.Sigmoid()(pred_y1), sub_y1, weight=w1, reduction='none')
            pred_y1_loss = (pred_y1_loss * sub_t).mean()
            pred_y0_loss = F.binary_cross_entropy(nn.Sigmoid()(pred_y0), sub_y0, weight=w0, reduction='none')
            pred_y0_loss = (pred_y0_loss * (1-sub_t)).mean()
            
            pred_loss = pred_y1_loss + pred_y0_loss

            optimizer_prediction.zero_grad()
            pred_loss.backward()
            optimizer_prediction.step()

            total_loss = weight_model_loss + pred_y1_loss + pred_y0_loss

            epoch_weight_model_loss += weight_model_loss
            epoch_pred_y1_loss += pred_y1_loss
            epoch_pred_y0_loss += pred_y0_loss
            epoch_total_loss += total_loss

        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_y1_loss': float(epoch_pred_y1_loss.item()),
            'epoch_y0_loss': float(epoch_pred_y0_loss.item()),
            'epoch_weight_model_loss': float(epoch_weight_model_loss.item()),
            'epoch_total_loss': float(epoch_total_loss.item()),
        }

        wandb_var.log(loss_dict)

        if epoch % evaluate_interval == 0:
            model.eval()

            pred_y1, _ = model.prediction_model(x1_test_tensor)
            _, pred_y0 = model.prediction_model(x0_test_tensor)

            nll_y1 = nn.BCELoss()(nn.Sigmoid()(pred_y1), torch.Tensor(y1_test).unsqueeze(-1).to(device))
            nll_y1 = nll_y1.detach().cpu().item()
            pred_y1 = pred_y1.detach().cpu().numpy()
            auc_y1 = roc_auc_score(y1_test, pred_y1)

            nll_y0 = nn.BCELoss()(nn.Sigmoid()(pred_y0), torch.Tensor(y0_test).unsqueeze(-1).to(device))
            nll_y0 = nll_y0.detach().cpu().item()
            pred_y0 = pred_y0.detach().cpu().numpy()
            auc_y0 = roc_auc_score(y0_test, pred_y0)

            wandb_var.log({
                "auc_y1": auc_y1,
                "auc_y0": auc_y0,
                "nll_y1": nll_y1,
                "nll_y0": nll_y0,
                })

    print(f"AUC_y1: {auc_y1}")
    print(f"AUC_y0: {auc_y0}")
    print(f"NLL_y1: {nll_y1}")
    print(f"NLL_y0: {nll_y0}")

    wandb.finish()

# %%
