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

from module.model import SharedNCF, SharedMF
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import set_seed, set_device
from module.dataset import binarize, generate_total_sample, load_data

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


# SETTINGS
parser = argparse.ArgumentParser()

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="coat")
parser.add_argument("--loss-type", type=str, default="naive")

parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="../data")

parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--G", type=int, default=1)

parser.add_argument("--base-model", type=str, default="ncf") # ["ncf", "mf"]

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
alpha = args.alpha
beta = args.beta
G = args.G
loss_type = args.loss_type
base_model = args.base_model

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


# DATA LOADER
x_train, x_test = load_data(data_dir, dataset_name)
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
    wandb.run.name = f"cv_escm2_{loss_type}_interaction_{expt_num}"

    x_train = x_train_cv[train_idx]
    y_train = y_train_cv[train_idx]
    x_test = x_train_cv[test_idx]
    y_test = y_train_cv[test_idx]

    obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    x_all = generate_total_sample(num_users, num_items)

    num_samples = len(x_all)
    total_batch = num_samples // batch_size

    # TRAIN
    if base_model == "ncf":
        model = SharedNCF(num_users, num_items, embedding_k)
    elif base_model == "mf":
        model = SharedMF(num_users, num_items, embedding_k)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = torch.nn.BCELoss()
    x_test_tensor = torch.LongTensor(x_test-1).to(device)
    inv_prop = torch.tensor([1.]).to(device)

    for epoch in range(1, num_epochs+1):
        ul_idxs = np.arange(x_all.shape[0])
        np.random.shuffle(ul_idxs)
        model.train()

        epoch_ctr_loss = 0.
        epoch_cvr_loss = 0.
        epoch_ctcvr_loss = 0.
        epoch_total_loss = 0.

        for idx in range(total_batch):

            selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(device)
            sub_y = y_entire[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
            sub_t = obs[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

            pred_cvr, pred_ctr, pred_ctcvr = model(sub_x)
            if loss_type == "ips":
                inv_prop = 1/torch.nn.Sigmoid()(pred_ctr).detach()
            ctr_loss = loss_fcn(nn.Sigmoid()(pred_ctr), sub_t) * alpha
            ctcvr_loss = loss_fcn(pred_ctcvr, sub_y) * beta
            cvr_loss = F.binary_cross_entropy(nn.Sigmoid()(pred_cvr), sub_y, inv_prop, reduction="none")
            cvr_loss = (cvr_loss * sub_t).mean()
            total_loss = ctr_loss + ctcvr_loss + cvr_loss

            epoch_ctr_loss += ctr_loss
            epoch_cvr_loss += cvr_loss
            epoch_ctcvr_loss += ctcvr_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_ctr_loss': float(epoch_ctr_loss.item()),
            'epoch_cvr_loss': float(epoch_ctr_loss.item()),
            'epoch_ctcvr_loss': float(epoch_ctcvr_loss.item()),
            'epoch_total_loss': float(epoch_total_loss.item()),
        }

        wandb_var.log(loss_dict)

        if epoch % evaluate_interval == 0:
            model.eval()
            x_test_tensor = torch.LongTensor(x_test-1).to(device)
            pred_, _, __ = model(x_test_tensor)
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
