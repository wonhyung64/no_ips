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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.model import SharedNCFPlus, SharedLinearCFPlus
from module.dataset import load_data, generate_total_sample
from module.utils import set_device, set_seed

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="original")#[original, personalized]

parser.add_argument("--loss-type", type=str, default="naive")#[naive, ips]
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10,100])
parser.add_argument("--data-dir", type=str, default="../data")
parser.add_argument("--alpha", type=float, default=1.) # [2., 1., 0.1, 0.01, 0.001]
parser.add_argument("--beta", type=float, default=1.) # [2., 1., 0.1, 0.01, 0.001]
parser.add_argument("--propensity", type=str, default="pred")#[pred,true]
parser.add_argument("--base-model", type=str, default="ncf")#[ncf, linearcf]
parser.add_argument("--device", type=str, default="none")
parser.add_argument("--omega", type=float, default=9999.) #[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

embedding_k = args.embedding_k
lr = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name
loss_type = args.loss_type
propensity = args.propensity
alpha = args.alpha
beta = args.beta
base_model = args.base_model
device = args.device
omega = args.omega

if omega < 9999.:
    omega1 = 1/omega
    omega0 = 1/(1-omega)
else:
    omega1 = 1.
    omega0 = 1.


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
    wandb.run.name = f"cv_escm2_plus_{loss_type}_causality_{expt_num}"


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

    num_samples = len(x_all)
    total_batch = num_samples // batch_size

    x1_test_tensor = torch.LongTensor(x1_test).to(device)
    x0_test_tensor = torch.LongTensor(x0_test).to(device)

    if base_model == "ncf":
        model = SharedNCFPlus(num_users, num_items, embedding_k)
    elif base_model == "linearcf":
        model = SharedLinearCFPlus(num_users, num_items, embedding_k)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    inv_prop = torch.tensor([1.]).to(device)

    for epoch in range(1, num_epochs+1):
        all_idx = np.arange(num_samples)
        np.random.shuffle(all_idx)
        model.train()

        epoch_total_loss = 0.
        epoch_y1_loss = 0.
        epoch_y0_loss = 0.
        epoch_t_loss = 0.
        epoch_y1_ctcvr_loss = 0.
        epoch_y0_ctcvr_loss = 0.

        for idx in range(total_batch):
            # mini-batch training
            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(device)

            sub_y = y1_entire[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
            sub_t = obs1[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
            sub_ps = ps1_entire[selected_idx]
            sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

            pred_y1, pred_y0, ctr = model(sub_x)

            if loss_type == "ips":
                if propensity == "true":
                    inv_prop = 1/(sub_ps+1e-9)
                elif propensity == "pred":
                    inv_prop = 1 / nn.Sigmoid()(ctr).detach()

            rec_loss = nn.functional.binary_cross_entropy(
                nn.Sigmoid()(pred_y1), sub_y, weight=inv_prop, reduction="none")
            y1_loss = torch.mean(rec_loss * sub_t) * omega1

            ctr_loss = nn.functional.binary_cross_entropy(nn.Sigmoid()(ctr), sub_t) * alpha

            ctcvr = nn.Sigmoid()(pred_y1) * nn.Sigmoid()(ctr)
            y1_ctcvr_loss = nn.functional.binary_cross_entropy(ctcvr, sub_y) * beta

            sub_y = y0_entire[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
            sub_ps = ps0_entire[selected_idx]
            sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

            if loss_type == "ips":
                if propensity == "true":
                    inv_prop = 1/(sub_ps+1e-9)
                elif propensity == "pred":
                    inv_prop = 1 / (1-nn.Sigmoid()(ctr).detach())

            rec_loss = nn.functional.binary_cross_entropy(
                nn.Sigmoid()(pred_y0), sub_y, weight=inv_prop, reduction="none")
            y0_loss = torch.mean(rec_loss * sub_t) * omega0

            ctcvr = nn.Sigmoid()(pred_y0) * (1-nn.Sigmoid()(ctr))
            y0_ctcvr_loss = nn.functional.binary_cross_entropy(ctcvr, sub_y) * beta

            total_loss = y1_loss + y0_loss + ctr_loss + y1_ctcvr_loss + y0_ctcvr_loss

            epoch_y1_loss += y1_loss
            epoch_y0_loss += y0_loss
            epoch_y1_ctcvr_loss += y1_ctcvr_loss
            epoch_y0_ctcvr_loss += y0_ctcvr_loss
            epoch_t_loss += ctr_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch:>4d} Train Loss] y1: {epoch_y1_loss.item():.4f} / y0: {epoch_y0_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_y1_loss': float(epoch_y1_loss.item()),
            'epoch_y0_loss': float(epoch_y0_loss.item()),
            'epoch_y1_ctcvr_loss': float(epoch_y1_ctcvr_loss.item()),
            'epoch_y0_ctcvr_loss': float(epoch_y0_ctcvr_loss.item()),
            'epoch_total_loss': float(epoch_total_loss.item()),
            'epoch_t_loss': float(epoch_t_loss.item()),
        }

        wandb_var.log(loss_dict)

        if epoch % evaluate_interval == 0:
            model.eval()

            pred_y1, _, __ = model(x1_test_tensor)
            _, pred_y0, __ = model(x0_test_tensor)

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
                "nll_y" : nll_y1*0.15267 + nll_y0*(1-0.15267),
                })

    print(f"AUC_y1: {auc_y1}")
    print(f"AUC_y0: {auc_y0}")
    print(f"NLL_y1: {nll_y1}")
    print(f"NLL_y0: {nll_y0}")

    wandb.finish()
