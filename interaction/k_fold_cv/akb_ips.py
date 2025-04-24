#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.model import NCF_AKBIPS_Exp, NCF, MF_AKBIPS_Exp
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import set_seed, set_device, estimate_ips_bayes
from module.dataset import binarize, generate_total_sample, load_data

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


#%%
# SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--lr1", type=float, default=0.01)
parser.add_argument("--lamb1", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="coat")
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
parser.add_argument("--base-model", type=str, default="ncf") # ["ncf","mf"]

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

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


# DATA LOADER
x_train, _ = load_data(data_dir, dataset_name)
x_train_cv, y_train = x_train[:,:-1], x_train[:,-1]
y_train_cv = binarize(y_train)

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print(f"# user: {num_users}, # item: {num_items}")

kf = KFold(n_splits=4, shuffle=True, random_state=random_seed)
for cv_num, (train_idx, test_idx) in enumerate(kf.split(x_train)):

    if cv_num > 1: #need to delete
        continue

    configs = vars(args)
    configs["device"] = device
    configs["cv_num"] = cv_num
    wandb_var = wandb.init(project="no_ips", config=configs)
    wandb.run.name = f"cv_akb_ips_interaction_{expt_num}"


    x_train = x_train_cv[train_idx]
    y_train = y_train_cv[train_idx]
    x_test = x_train_cv[test_idx]
    y_test = y_train_cv[test_idx]

    # TRAIN ps model
    obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    x_all = generate_total_sample(num_users, num_items)

    num_samples = len(x_all)
    total_batch = num_samples // batch_size

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
            sub_t = obs[selected_idx]
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


    #train abkips
    num_sample = len(x_train)
    total_batch = num_sample // batch_size

    if base_model == "ncf":
        model = NCF_AKBIPS_Exp(num_users, num_items, embedding_k, dataset_name)
    elif base_model == "mf":
        model = MF_AKBIPS_Exp(num_users, num_items, embedding_k, dataset_name)

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
        all_idx = np.arange(num_sample)
        np.random.shuffle(all_idx)
        model.train()

        ul_idxs = np.arange(x_all.shape[0]) # all
        np.random.shuffle(ul_idxs)

        epoch_pred_loss = 0.
        epoch_weight_model_loss = 0.
        epoch_total_loss = 0.

        for idx in range(total_batch):

            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_train[selected_idx]
            sub_x = torch.LongTensor(sub_x - 1).to(device)
            sub_y = y_train[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

            x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
            x_sampled = x_all[x_all_idx]
            x_sampled = torch.LongTensor(x_sampled).to(device)
            sub_y_entire = y_entire[x_all_idx]
            sub_y_entire = torch.Tensor(sub_y_entire).unsqueeze(-1).to(device)
            sub_t = obs[x_all_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

            prop, _, __ = ps_model(sub_x)
            inv_prop = 1/nn.Sigmoid()(prop).detach()

            pred, _, __ = model.prediction_model.forward(sub_x)
            e_loss = F.binary_cross_entropy(nn.Sigmoid()(pred), sub_y, reduction='none')

            feature_emd_o = model.get_embedding(sub_x)
            feature_emd_d = model.get_embedding(x_sampled)

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

            # weight model 
            w_, _, __ = model.weight_model(sub_x)
            w = nn.Sigmoid()(w_).detach() * inv_prop
                    
            feature_emd_j = model.get_embedding(x_sampled[topj_indices[0]])                       

            h_all = torch.mean(model.exp_kernel(feature_emd_j, feature_emd_d), axis =0)
            w_tran = (w.unsqueeze(1)).T
            h_obs = 1/len(w) * (torch.mm(w_tran[0], model.exp_kernel(feature_emd_j, feature_emd_o))).squeeze()

            zero = torch.zeros(1).to(device)
            loss_ker_bal =  torch.sum(max(zero, torch.sum(h_all - h_obs) - C)) + torch.sum(max(zero, torch.sum(h_obs - h_all) - C))
            weight_model_loss =  1/len(w) * torch.dot(w[0], torch.log(w)[0]) + gamma * loss_ker_bal

            optimizer_weight.zero_grad()
            weight_model_loss.backward()
            optimizer_weight.step()

            prop, _, __ = ps_model(x_sampled)
            inv_prop = 1/nn.Sigmoid()(prop).detach()
            w_, _, __ = model.weight_model(x_sampled)
            w = nn.Sigmoid()(w_).detach() * inv_prop

            pred, _, __ = model.prediction_model(x_sampled)
            pred_loss = F.binary_cross_entropy(nn.Sigmoid()(pred), sub_y_entire, weight=w, reduction='none')
            pred_loss = (pred_loss * sub_t).mean()

            optimizer_prediction.zero_grad()
            pred_loss.backward()
            optimizer_prediction.step()

            total_loss = weight_model_loss + pred_loss

            epoch_weight_model_loss += weight_model_loss
            epoch_pred_loss += pred_loss
            epoch_total_loss += total_loss

        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_prop_loss': float(epoch_pred_loss.item()),
            'epoch_pred_all_loss': float(epoch_weight_model_loss.item()),
            'epoch_total_loss': float(epoch_total_loss.item()),
        }

        wandb_var.log(loss_dict)

        if epoch % evaluate_interval == 0:
            model.eval()
            x_test_tensor = torch.LongTensor(x_test-1).to(device)
            pred_, _, __ = model.prediction_model(x_test_tensor)
            pred = pred_.flatten().cpu().detach().numpy()

            ndcg_res = ndcg_func(pred, x_test, y_test, top_k_list)
            ndcg_dict: dict = {}
            for top_k in top_k_list:
                ndcg_dict[f"ndcg_{top_k}"] = np.mean(ndcg_res[f"ndcg_{top_k}"])

            recall_res = recall_func(pred, x_test, y_test, top_k_list)
            recall_dict: dict = {}
            for top_k in top_k_list:
                recall_dict[f"recall_{top_k}"] = np.mean(recall_res[f"recall_{top_k}"])

            ap_res = ap_func(pred, x_test, y_test, top_k_list)
            ap_dict: dict = {}
            for top_k in top_k_list:
                ap_dict[f"ap_{top_k}"] = np.mean(ap_res[f"ap_{top_k}"])

            auc = roc_auc_score(y_test, pred)

            wandb_var.log(ndcg_dict)
            wandb_var.log(recall_dict)
            wandb_var.log(ap_dict)
            wandb_var.log({"auc": auc})

    print(f"NDCG: {ndcg_dict}")
    print(f"Recall: {recall_dict}")
    print(f"AP: {ap_dict}")
    print(f"AUC: {auc}")

    wandb.finish()

# %%
