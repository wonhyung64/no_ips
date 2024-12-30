#%%
import os
import sys
import torch
import argparse
import subprocess
import numpy as np
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from module.model import MF
from module.metric import ndcg_func, recall_func, ap_func
from module.utils import set_seed, estimate_ips_bayes, set_device
from module.dataset import binarize, load_data

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


# SETTINGS
for seed in range(10):
    parser = argparse.ArgumentParser()

    """coat"""
    parser.add_argument("--embedding-k", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dataset-name", type=str, default="coat")

    """yahoo"""
    # parser.add_argument("--embedding-k", type=int, default=128)
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--weight-decay", type=float, default=1e-4)
    # parser.add_argument("--batch-size", type=int, default=8192)
    # parser.add_argument("--dataset-name", type=str, default="yahoo_r3")

    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=seed)
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


    # WANDB
    configs = vars(args)
    configs["device"] = device
    wandb_var = wandb.init(project="no_ips", config=configs)
    wandb.run.name = f"ips_{expt_num}"


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
    model = MF(num_users, num_items, embedding_k)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fcn = lambda x, y, z: F.binary_cross_entropy(x, y, z)

    ips_idxs = np.arange(len(y_test))
    np.random.shuffle(ips_idxs)
    y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

    one_over_zl = estimate_ips_bayes(x_train, y_train, y_ips)

    for epoch in range(1, num_epochs+1):
        all_idx = np.arange(num_sample)
        np.random.shuffle(all_idx)
        model.train()

        epoch_total_loss = 0.
        epoch_ips_loss = 0.

        for idx in range(total_batch):

            selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
            sub_x = x_train[selected_idx]
            sub_x = torch.LongTensor(sub_x - 1).to(device)
            sub_y = y_train[selected_idx]
            sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

            pred, user_embed, item_embed = model(sub_x)
            inv_prop = one_over_zl[selected_idx].unsqueeze(-1).to(device)

            ips_loss = loss_fcn(torch.nn.Sigmoid()(pred), sub_y, inv_prop)

            epoch_ips_loss += ips_loss

            total_loss = ips_loss
            epoch_total_loss += total_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch:>4d} Train Loss] rec: {epoch_total_loss.item():.4f}")

        loss_dict: dict = {
            'epoch_ips_loss': float(epoch_ips_loss.item()),
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
