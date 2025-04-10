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

from module.model import NCF, SharedNCF
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
parser.add_argument("--lr1", type=float, default=1e-2)
parser.add_argument("--weight-decay1", type=float, default=1e-4)
parser.add_argument("--lr0", type=float, default=1e-3)
parser.add_argument("--weight-decay0", type=float, default=1e-4)


"""personalized""" #end
# parser.add_argument("--dataset-name", type=str, default="personalized")#[original, personalized]
# parser.add_argument("--lr1", type=float, default=1e-2)
# parser.add_argument("--lr0", type=float, default=1e-4)
# parser.add_argument("--weight-decay1", type=float, default=1e-5)
# parser.add_argument("--weight-decay0", type=float, default=1e-4)

parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[10, 30, 100, 1372])
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--propensity", type=str, default="pred")#[pred,true]
parser.add_argument("--ps-model-name", type=str, default="multi") #[escm2, multi]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

lr1 = args.lr1
lr0 = args.lr0
weight_decay1 = args.weight_decay1
weight_decay0 = args.weight_decay0

embedding_k = args.embedding_k
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name
propensity = args.propensity
ps_model_name = args.ps_model_name

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()


configs = vars(args)
configs["device"] = device
wandb_var = wandb.init(project="no_ips", config=configs)
wandb.run.name = f"single_ips_bestps_causality_{expt_num}"


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

ps_model_y1 = SharedNCF(num_users, num_items, embedding_k)
ps_model_y1 = ps_model_y1.to(device)
weight_dir = f"./weights/{ps_model_name}_ips_y1{'_ori' if dataset_name=='original' else '_per'}_seed{random_seed}.pth"
ps_model_y1.load_state_dict(torch.load(weight_dir, map_location=device))

ps_model_y0 = SharedNCF(num_users, num_items, embedding_k)
ps_model_y0 = ps_model_y0.to(device)
weight_dir = f"./weights/{ps_model_name}_ips_y0{'_ori' if dataset_name=='original' else '_per'}_seed{random_seed}.pth"
ps_model_y0.load_state_dict(torch.load(weight_dir, map_location=device))


x_test_tensor = torch.LongTensor(x_test).to(device)

# conditional outcome modeling
model_y1 = NCF(num_users, num_items, embedding_k)
model_y1 = model_y1.to(device)
optimizer_y1 = torch.optim.Adam(model_y1.parameters(), lr=lr1, weight_decay=weight_decay1)

model_y0 = NCF(num_users, num_items, embedding_k)
model_y0 = model_y0.to(device)
optimizer_y0 = torch.optim.Adam(model_y0.parameters(), lr=lr0, weight_decay=weight_decay0)

for epoch in range(1, num_epochs+1):
    all_idx = np.arange(num_samples)
    np.random.shuffle(all_idx)
    model_y1.train()
    model_y0.train()
    ps_model_y1.eval()
    ps_model_y0.eval()

    epoch_total_y1_loss = 0.
    epoch_y1_loss = 0.
    epoch_total_y0_loss = 0.
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

        _, ps_pred, __ = ps_model_y1(sub_x)
        pred, _, __ = model_y1(sub_x)

        if propensity == "true":
            inv_prop = 1/(sub_ps+1e-9)
        elif propensity == "clip":
            inv_prop = 1/(sub_ps.clip(0.0025, 0.9975))
        elif propensity == "pred":
            inv_prop = 1 / nn.Sigmoid()(ps_pred).detach()

        rec_loss = nn.functional.binary_cross_entropy(
            nn.Sigmoid()(pred), sub_y, weight=inv_prop, reduction="none")
        rec_loss = torch.mean(rec_loss * sub_t)
        total_loss = rec_loss

        epoch_y1_loss += rec_loss
        epoch_total_y1_loss += total_loss

        optimizer_y1.zero_grad()
        total_loss.backward()
        optimizer_y1.step()

        sub_y = y0_entire[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
        sub_t = obs0[selected_idx]
        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
        sub_ps = ps0_entire[selected_idx]
        sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

        _, ps_pred, __ = ps_model_y0(sub_x)
        pred, _, __ = model_y0(sub_x)

        if propensity == "true":
            inv_prop = 1/(1-sub_ps+1e-9)
        elif propensity == "clip":
            inv_prop = 1/(1-sub_ps.clip(0.0025, 0.9975))
        elif propensity == "pred":
            inv_prop = 1 / (1-nn.Sigmoid()(ps_pred).detach())

        rec_loss = nn.functional.binary_cross_entropy(
            nn.Sigmoid()(pred), sub_y, weight=inv_prop, reduction="none")
        rec_loss = torch.mean(rec_loss * sub_t)
        total_loss = rec_loss

        epoch_y0_loss += rec_loss
        epoch_total_y0_loss += total_loss

        optimizer_y0.zero_grad()
        total_loss.backward()
        optimizer_y0.step()

    print(f"[Epoch {epoch:>4d} Train Loss] y1: {epoch_total_y1_loss.item():.4f} / y0: {epoch_total_y0_loss.item():.4f}")

    loss_dict: dict = {
        'epoch_y1_loss': float(epoch_y1_loss.item()),
        'epoch_y0_loss': float(epoch_y0_loss.item()),
        'epoch_total_y1_loss': float(epoch_total_y1_loss.item()),
        'epoch_total_y0_loss': float(epoch_total_y0_loss.item()),
    }

    wandb_var.log(loss_dict)

    if epoch % evaluate_interval == 0:
        model_y1.eval()
        model_y0.eval()

        pred_y1, _, __ = model_y1(x_test_tensor)
        pred_y0, _, __ = model_y0(x_test_tensor)
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

torch.save(model_y1.state_dict(), f"./weights/ips_y1_{dataset_name[:3]}_{ps_model_name}_{random_seed}.pth")
torch.save(model_y0.state_dict(), f"./weights/ips_y0_{dataset_name[:3]}_{ps_model_name}_{random_seed}.pth")