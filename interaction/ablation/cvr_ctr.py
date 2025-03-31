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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.model import MF
from module.metric import ndcg_func, recall_func, ap_func
from module.dataset import binarize, load_data, generate_total_sample
from module.utils import set_device, set_seed

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


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
        self.ctr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )
        self.cvr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        ctr = self.ctr(z_embed)
        cvr = self.cvr(z_embed)
        return ctr, cvr


#%%
# SETTINGS
for seed in range(10):
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss-type", type=str, default="naive")

    """coat"""
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dataset-name", type=str, default="coat")

    """yahoo"""
    # parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--weight-decay", type=float, default=1e-6)
    # parser.add_argument("--batch-size", type=int, default=8192)
    # parser.add_argument("--dataset-name", type=str, default="yahoo_r3")

    parser.add_argument("--embedding-k", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=seed)
    parser.add_argument("--evaluate-interval", type=int, default=50)
    parser.add_argument("--top-k-list", type=list, default=[1,3,5,7,10])
    parser.add_argument("--data-dir", type=str, default="../data")
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
    alpha = args.alpha

    expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
    set_seed(random_seed)
    device = set_device()


    # WANDB
    configs = vars(args)
    configs["device"] = device
    wandb_var = wandb.init(project="no_ips", config=configs)
    wandb.run.name = f"ctr_{loss_type}_{expt_num}"

    # DATA LOADER
    x_train, x_test = load_data(data_dir, dataset_name)

    x_train, y_train = x_train[:,:-1], x_train[:,-1]
    x_test, y_test = x_test[:, :-1], x_test[:,-1]

    y_train = binarize(y_train)
    y_test = binarize(y_test)

    num_users = x_train[:,0].max()
    num_items = x_train[:,1].max()
    print(f"# user: {num_users}, # item: {num_items}")

    obs = sps.csr_matrix((np.ones(len(y_train)), (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    y_entire = sps.csr_matrix((y_train, (x_train[:, 0]-1, x_train[:, 1]-1)), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
    x_all = generate_total_sample(num_users, num_items)

    num_samples = len(x_all)
    total_batch = num_samples // batch_size

    # TRAIN
    model = NCF(num_users, num_items, embedding_k)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = lambda x, y, z=None: F.binary_cross_entropy(x, y, z, reduction="none")
    inv_prop = torch.ones([1]).to(device)

    for epoch in range(1, num_epochs+1):
        ul_idxs = np.arange(x_all.shape[0]) # all
        np.random.shuffle(ul_idxs)
        model.train()

        epoch_total_loss = 0.
        epoch_rec_loss = 0.
        epoch_ctr_loss = 0.

        for idx in range(total_batch):

            selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_all[selected_idx]
            sub_x = torch.LongTensor(sub_x).to(device)
            sub_y = y_entire[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
            sub_t = obs[selected_idx]
            sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

            ps_pred, pred = model(sub_x)

            if loss_type == "ips":
                inv_prop = 1/torch.nn.Sigmoid()(ps_pred).detach()

            ctr_loss = loss_fcn(nn.Sigmoid()(ps_pred), sub_t).mean()

            rec_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y, inv_prop) 
            rec_loss = (rec_loss * sub_t).mean()
            epoch_rec_loss += rec_loss
            epoch_ctr_loss += ctr_loss

            total_loss = rec_loss + ctr_loss * alpha
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_rec_loss': float(epoch_rec_loss.item()),
            'epoch_ctr_loss': float(epoch_ctr_loss.item()),
            'epoch_total_loss': float(epoch_total_loss.item()),
        }

        wandb_var.log(loss_dict)

        if epoch % evaluate_interval == 0:
            model.eval()
            x_test_tensor = torch.LongTensor(x_test-1).to(device)
            _, pred_ = model(x_test_tensor)
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

    os.makedirs("./ctr", exist_ok=True)
    torch.save(model.state_dict(), f"./ctr/data={dataset_name}_loss={loss_type}_wd={'%.0e'%weight_decay}_lr={'%.0e'%lr}_seed={random_seed}.pt")

    wandb.finish()


# %%
