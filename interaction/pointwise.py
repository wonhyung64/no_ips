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

from module.dataset import binarize, load_data, generate_total_sample
from module.utils import set_device, set_seed


class NCF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear_2 = nn.Linear(self.embedding_k, self.embedding_k//2)
        self.linear_3 = nn.Linear(self.embedding_k//2, 1, bias=False)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)

        h1 = self.linear_1(z_embed)
        h1 = torch.nn.ReLU()(h1)
        h2 = self.linear_2(h1)
        h2 = torch.nn.ReLU()(h2)
        out = self.linear_3(h2)

        return out, user_embed, item_embed


#%%
# SETTINGS
model_dir = f"./assets/selection_model"
parser = argparse.ArgumentParser()

"""coat"""
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--dataset-name", type=str, default="coat")

"""yahoo"""
# parser.add_argument("--embedding-k", type=int, default=64)
# parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--weight-decay", type=float, default=1e-6)
# parser.add_argument("--batch-size", type=int, default=8192)
# parser.add_argument("--dataset-name", type=str, default="yahoo_r3")

parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
parser.add_argument("--data-dir", type=str, default="./data")
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

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()

# DATA LOADER
x_train, _ = load_data(data_dir, dataset_name)
x_train, y_train = x_train[:,:-1], x_train[:,-1]
y_train = binarize(y_train)

num_users = x_train[:,0].max()
num_items = x_train[:,1].max()
print(f"# user: {num_users}, # item: {num_items}")

obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
x_all = generate_total_sample(num_users, num_items)

num_samples = len(x_all)
total_batch = num_samples // batch_size

# TRAIN
model = NCF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fcn = torch.nn.BCELoss(reduction="none")

for epoch in range(1, num_epochs+1):
    ul_idxs = np.arange(x_all.shape[0])
    np.random.shuffle(ul_idxs)
    model.train()

    epoch_total_loss = 0.
    epoch_point_loss = 0.

    for idx in range(total_batch):

        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_all[selected_idx]
        sub_x = torch.LongTensor(sub_x).to(device)
        sub_t = obs[selected_idx]
        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

        pred, user_embed, item_embed = model(sub_x)

        point_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_t).mean()
        epoch_point_loss += point_loss

        total_loss = point_loss
        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

torch.save({"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()}, f"{model_dir}/pointwise.pth")

# %%
# checkpoint = torch.load(f"{model_dir}/pointwise.pth")
# model.load_state_dict(checkpoint["model_state_dict"])
