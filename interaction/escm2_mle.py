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

from module.model import ESCM2Ips
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import set_seed, set_device
from module.dataset import binarize, generate_total_sample, load_data

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


# for seed in range(10):
# SETTINGS
parser = argparse.ArgumentParser()

"""coat"""
# parser.add_argument("--embedding-k", type=int, default=64)
# parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--weight-decay", type=float, default=1e-6)
# parser.add_argument("--batch-size", type=int, default=4096)
# parser.add_argument("--dataset-name", type=str, default="coat")

"""yahoo"""
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--dataset-name", type=str, default="yahoo_r3")

parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=seed)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")

parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--G", type=int, default=1)

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

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


# WANDB
configs = vars(args)
configs["device"] = device
wandb_var = wandb.init(project="no_ips", config=configs)
wandb.run.name = f"escm2_mle_{expt_num}"


# DATA LOADER
x_train, x_test = load_data(data_dir, dataset_name)

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

y_train = binarize(y_train)
y_test = binarize(y_test)

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
num_sample = len(x_train)
print(f"# user: {num_users}, # item: {num_items}")

total_batch = num_sample // batch_size


# TRAIN
model = ESCM2Ips(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss()

obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
x_all = generate_total_sample(num_users, num_items)

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    ul_idxs = np.arange(x_all.shape[0]) # all
    np.random.shuffle(ul_idxs)

    epoch_ctr_loss = 0.
    epoch_cvr_loss = 0.
    epoch_ctcvr_loss = 0.
    epoch_total_loss = 0.

    for idx in range(total_batch):

        x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
        x_sampled = x_all[x_all_idx]
        x_sampled = torch.LongTensor(x_sampled).to(device)
        sub_obs = torch.Tensor(obs[x_all_idx]).unsqueeze(-1).to(device)
        sub_entire_y = torch.Tensor(y_entire[x_all_idx]).unsqueeze(-1).to(device)

        pred_cvr, pred_ctr, pred_ctcvr = model(x_sampled)
        ctr_loss = loss_fcn(nn.Sigmoid()(pred_ctr), sub_obs)
        ctcvr_loss = loss_fcn(pred_ctcvr, sub_entire_y) * alpha
        cvr_loss = loss_fcn(nn.Sigmoid()(pred_cvr), sub_entire_y) * beta
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
