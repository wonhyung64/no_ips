#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import roc_auc_score

from module.model import IpsV2
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

"""coat"""
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="coat")

"""yahoo"""
# parser.add_argument("--lr", type=float, default=5e-2)
# parser.add_argument("--weight-decay", type=float, default=1e-4)
# parser.add_argument("--batch-size", type=int, default=8192)
# parser.add_argument("--dataset-name", type=str, default="yahoo_r3")

parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--alpha", type=float, default=1.)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--eta", type=float, default=1.)
parser.add_argument("--gamma", type=float, default=0.1)
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
eta = args.eta
gamma = args.gamma
G = args.G

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


# WANDB
configs = vars(args)
configs["device"] = device
wandb_var = wandb.init(project="no_ips", config=configs)
wandb.run.name = f"ips_v2_{expt_num}"


# DATA LOADER
x_train, x_test = load_data(data_dir, dataset_name)

x_train, y_train = x_train[:,:-1], x_train[:,-1]
x_test, y_test = x_test[:, :-1], x_test[:,-1]

y_train = binarize(y_train)
y_test = binarize(y_test)

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print(f"# user: {num_users}, # item: {num_items}")

# TRAIN
model = IpsV2(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
x_all = generate_total_sample(num_users, num_items)
num_sample = len(x_all)
total_batch = num_sample // batch_size

for epoch in range(1, num_epochs+1):
    ul_idxs = np.arange(x_all.shape[0]) # all
    np.random.shuffle(ul_idxs)
    model.train()

    epoch_prop_loss = 0.
    epoch_pred_all_loss = 0.
    epoch_ips_loss = 0.
    epoch_bmse_loss = 0.
    epoch_total_loss = 0.

    for idx in range(total_batch):

        x_all_idx = ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]
        x_sampled = x_all[x_all_idx]
        x_sampled = torch.LongTensor(x_sampled).to(device)
        sub_obs = torch.Tensor(obs[x_all_idx]).unsqueeze(-1).to(device)
        sub_entire_y = torch.Tensor(y_entire[x_all_idx]).unsqueeze(-1).to(device)

        prop_pred_all, _, __ =  model.propensity_model(x_sampled)
        inv_prop_all = 1/torch.clip(nn.Sigmoid()(prop_pred_all), gamma, 1)
        prop_loss = F.binary_cross_entropy(1/inv_prop_all, sub_obs) * alpha

        pred, _, __ = model.prediction_model(x_sampled)
        pred = nn.Sigmoid()(pred)
        ips_loss = -torch.mean((sub_entire_y * torch.log(pred + 1e-6) + (1-sub_entire_y) * torch.log(1 - pred + 1e-6)) * inv_prop_all * sub_obs)

        pred_all, _, __ = model.prediction_model(x_sampled)
        pred_all_loss = F.binary_cross_entropy(1/inv_prop_all * nn.Sigmoid()(pred_all), sub_entire_y) * beta

        ones_all = torch.ones(len(inv_prop_all)).unsqueeze(-1).to(device)
        w_all = torch.divide(sub_obs,1/inv_prop_all) - torch.divide((ones_all-sub_obs),(ones_all-(1/inv_prop_all)))
        bmse_loss = (torch.mean(w_all * pred_all))**2 * eta

        total_loss = prop_loss + pred_all_loss + ips_loss + bmse_loss

        epoch_prop_loss += prop_loss
        epoch_pred_all_loss += pred_all_loss
        epoch_ips_loss += ips_loss
        epoch_bmse_loss += bmse_loss
        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_prop_loss': float(epoch_prop_loss.item()),
        'epoch_pred_all_loss': float(epoch_pred_all_loss.item()),
        'epoch_ips_loss': float(epoch_ips_loss.item()),
        'epoch_bmse_loss': float(epoch_bmse_loss.item()),
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
