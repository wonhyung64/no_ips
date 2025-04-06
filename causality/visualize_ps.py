#%%
import os 
import sys
import torch
import argparse
import subprocess
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime

from module.model import NCF, SharedNCF, SharedNCFPlus
from module.metric import cdcg_func, car_func, cp_func, ncdcg_func
from module.dataset import load_data, generate_total_sample
from module.utils import set_device, set_seed

try:
    import wandb
except: 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def logit(x):
    return np.log(x/(1-x))
#%%
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default="original")#[original, personalized]
# parser.add_argument("--dataset-name", type=str, default="personalized")#[original, personalized]
parser.add_argument("--batch-size", type=int, default=4096)
parser.add_argument("--embedding-k", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--evaluate-interval", type=int, default=50)
parser.add_argument("--top-k-list", type=list, default=[10, 30, 100, 1372])
parser.add_argument("--data-dir", type=str, default="./data")
# parser.add_argument("--ps-model-name", type=str, default="multi") #[escm2, multi]
parser.add_argument("--ps-model-name", type=str, default="escm2") #[escm2, multi]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


embedding_k = args.embedding_k
batch_size = args.batch_size
num_epochs = args.num_epochs
random_seed = args.random_seed
evaluate_interval = args.evaluate_interval
top_k_list = args.top_k_list
data_dir = args.data_dir
dataset_name = args.dataset_name
ps_model_name = args.ps_model_name

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()

x_train, x_test = load_data(data_dir, dataset_name)
x_train, y_train, t_train, ps_train = x_train[:,:2].astype(int), x_train[:,2:3], x_train[:,3:4], x_train[:,4:]
x_test, cate_test = x_test[:,:2].astype(int), x_test[:,2]

num_users = x_train[:,0].max()+1
num_items = x_train[:,1].max()+1
print(f"# user: {num_users}, # item: {num_items}")

ps_model_y1 = SharedNCF(num_users, num_items, embedding_k)
ps_model_y1 = ps_model_y1.to(device)
weight_dir = f"./weights/{ps_model_name}_ips_y1{'_ori' if dataset_name=='original' else '_per'}_seed{random_seed}.pth"
ps_model_y1.load_state_dict(torch.load(weight_dir, map_location=device))

ps_model_y0 = SharedNCF(num_users, num_items, embedding_k)
ps_model_y0 = ps_model_y0.to(device)
weight_dir = f"./weights/{ps_model_name}_ips_y0{'_ori' if dataset_name=='original' else '_per'}_seed{random_seed}.pth"
ps_model_y0.load_state_dict(torch.load(weight_dir, map_location=device))

ps_model_plus = SharedNCFPlus(num_users, num_items, embedding_k)
ps_model_plus = ps_model_plus.to(device)
weight_dir = f"./weights/{ps_model_name}_plus_ips{'_ori' if dataset_name=='original' else '_per'}_seed{random_seed}.pth"
ps_model_plus.load_state_dict(torch.load(weight_dir, map_location=device))

x_test_tensor = torch.LongTensor(x_test).to(device)

ps_model_y1.eval()
ps_model_y0.eval()
ps_model_plus.eval()

_, ps_pred_y1, __ = ps_model_y1(x_test_tensor)
_, ps_pred_y0, __ = ps_model_y0(x_test_tensor)
_, ps_pred_plus, __ = ps_model_plus(x_test_tensor)

ps_pred_y1 = ps_pred_y1.cpu().detach().numpy().squeeze()
ps_pred_y0 = ps_pred_y0.cpu().detach().numpy().squeeze()
ps_pred_plus = ps_pred_plus.cpu().detach().numpy().squeeze()
ps_true = logit(ps_train).squeeze()

data_list = [ps_pred_y1, ps_pred_y0, ps_pred_plus, ps_true]
# labels = ["Multi-Y1", "Multi-Y0", "Multi-Plus", "True"]
labels = ["ESCM2-Y1", "ESCM2-Y0", "ESCM2-Plus", "True"]
colors = ["blue", "green", "orange", "red"]

plt.figure(figsize=(10, 6))

for data, label, color in zip(data_list, labels, colors):
    kde = gaussian_kde(data)
    x_range = np.linspace(0, 1, 1000)
    plt.plot(x_range, kde(x_range), label=label, color=color)

plt.title('Distribution Comparison of Predictions and True Values')
plt.xlabel('Prediction / True Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
# %%
