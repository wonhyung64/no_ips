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

from module.model import NCFPlus
from module.metric import cdcg_func, car_func, cp_func, ncdcg_func
from module.dataset import load_data, generate_total_sample
from module.utils import set_device, set_seed

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


parser = argparse.ArgumentParser()



"""original""" #end
parser.add_argument("--dataset-name", type=str, default="original")#[original, personalized]
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=1.)

"""personalized""" #end
# parser.add_argument("--dataset-name", type=str, default="personalized")#[original, personalized]
# parser.add_argument("--lr", type=float, default=1e-4)
# parser.add_argument("--weight-decay", type=float, default=1e-4)
# parser.add_argument("--gamma", type=float, default=1.)

parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[10, 30, 100, 1372])
parser.add_argument("--data-dir", type=str, default="./data")
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

lr = args.lr
weight_decay = args.weight_decay

embedding_k = args.embedding_k
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name
gamma = args.gamma

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


configs = vars(args)
configs["device"] = device
wandb_var = wandb.init(project="no_ips", config=configs)
wandb.run.name = f"sen_single_naive_plus_causality_{expt_num}"


x_train, x_test = load_data(data_dir, dataset_name)
x_train, y_train, t_train, ps_train = x_train[:,:2].astype(int), x_train[:,2:3], x_train[:,3:4], x_train[:,4:]
x_test, cate_test = x_test[:,:2].astype(int), x_test[:,2]

num_users = x_train[:,0].max()+1
num_items = x_train[:,1].max()+1
print(f"# user: {num_users}, # item: {num_items}")


x_all = generate_total_sample(num_users, num_items)

y1_train = y_train[t_train==1]
x1_train = x_train[np.squeeze(t_train==1)]
ps1_train = ps_train[t_train==1]

obs1 = sps.csr_matrix((np.ones(len(y1_train)), (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
y1_entire = sps.csr_matrix((y1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
ps1_entire = sps.csr_matrix((ps1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)

y0_train = y_train[t_train==0]
x0_train = x_train[np.squeeze(t_train==0)]
ps0_train = 1-ps_train[t_train==0]

obs0 = sps.csr_matrix((np.ones(len(y0_train)), (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
y0_entire = sps.csr_matrix((y0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)
ps0_entire = sps.csr_matrix((ps0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(num_users, num_items), dtype=np.float32).toarray().reshape(-1)

num_samples = len(x_all)
total_batch = num_samples // batch_size

x_test_tensor = torch.LongTensor(x_test).to(device)

# conditional outcome modeling
model = NCFPlus(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_samples)
    np.random.shuffle(all_idx)
    model.train()

    epoch_total_loss = 0.
    epoch_y1_loss = 0.
    epoch_y0_loss = 0.

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

        pred_y1, pred_y0 = model(sub_x)

        rec_loss = nn.functional.binary_cross_entropy(
            nn.Sigmoid()(pred_y1), sub_y, reduction="none")
        y1_loss = torch.mean(rec_loss * sub_t) * gamma
        epoch_y1_loss += y1_loss

        sub_y = y0_entire[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
        sub_t = obs0[selected_idx]
        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
        sub_ps = ps0_entire[selected_idx]
        sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

        rec_loss = nn.functional.binary_cross_entropy(
            nn.Sigmoid()(pred_y0), sub_y, reduction="none")
        y0_loss = torch.mean(rec_loss * sub_t)
        epoch_y0_loss += y0_loss

        total_loss = y1_loss + y0_loss
        epoch_total_loss += total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch:>4d} Train Loss] y1: {epoch_y1_loss.item():.4f} / y0: {epoch_y0_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_y1_loss': float(epoch_y1_loss.item()),
        'epoch_y0_loss': float(epoch_y0_loss.item()),
        'epoch_total_loss': float(epoch_total_loss.item()),
    }

    wandb_var.log(loss_dict)

    if epoch % evaluate_interval == 0:
        model.eval()

        pred_y1, pred_y0 = model(x_test_tensor)
        pred_y1 = nn.Sigmoid()(pred_y1).detach().cpu().numpy()
        pred_y0 = nn.Sigmoid()(pred_y0).detach().cpu().numpy()
        pred = (pred_y1 - pred_y0).squeeze()

        ncdcg_res = ncdcg_func(pred, x_test, cate_test, top_k_list)
        ncdcg_dict: dict = {}
        for top_k in top_k_list:
            ncdcg_dict[f"ncdcg_{top_k}"] = np.mean(ncdcg_res[f"ncdcg_{top_k}"])

        cdcg_res = cdcg_func(pred, x_test, cate_test, top_k_list)
        cdcg_dict: dict = {}
        for top_k in top_k_list:
            cdcg_dict[f"cdcg_{top_k}"] = np.mean(cdcg_res[f"cdcg_{top_k}"])

        cp_res = cp_func(pred, x_test, cate_test, top_k_list)
        cp_dict: dict = {}
        for top_k in top_k_list:
            cp_dict[f"cp_{top_k}"] = np.mean(cp_res[f"cp_{top_k}"])

        car_res = car_func(pred, x_test, cate_test, top_k_list)
        car_dict: dict = {}
        for top_k in top_k_list:
            car_dict[f"car_{top_k}"] = np.mean(car_res[f"car_{top_k}"])

        mse = np.square(cate_test - pred).mean()

        wandb_var.log({"mse":mse})
        wandb_var.log(cdcg_dict)
        wandb_var.log(cp_dict)
        wandb_var.log(car_dict)

print(f"ncDCG: {ncdcg_dict}")
print(f"cDCG: {cdcg_dict}")
print(f"cP: {cp_dict}")
print(f"cAR: {car_dict}")

cdcg_res = cdcg_func(cate_test, x_test, cate_test, top_k_list)
cdcg_dict: dict = {}
for top_k in top_k_list:
    cdcg_dict[f"true_cdcg_{top_k}"] = np.mean(cdcg_res[f"cdcg_{top_k}"])

cp_res = cp_func(cate_test, x_test, cate_test, top_k_list)
cp_dict: dict = {}
for top_k in top_k_list:
    cp_dict[f"true_cp_{top_k}"] = np.mean(cp_res[f"cp_{top_k}"])

car_res = car_func(cate_test, x_test, cate_test, top_k_list)
car_dict: dict = {}
for top_k in top_k_list:
    car_dict[f"true_car_{top_k}"] = np.mean(car_res[f"car_{top_k}"])


wandb_var.log(cdcg_dict)
wandb_var.log(cp_dict)
wandb_var.log(car_dict)

wandb.finish()

torch.save(model.state_dict(), f"./weights/naive_plus_{dataset_name[:3]}_{random_seed}.pth")
# %%
