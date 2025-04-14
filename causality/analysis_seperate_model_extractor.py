#%%
import os
import re
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from module.model import NCF, NCFPlus, SharedNCF, SharedNCFPlus
from module.dataset import load_data
from module.utils import set_device, set_seed, sigmoid


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
parser.add_argument("--weights-dir", type=str, default="./weights")


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
weights_dir = args.weights_dir
dataset_name = args.dataset_name

expt_num = f'{datetime.now().strftime("%y%m%d_%H%M%S_%f")}'
set_seed(random_seed)
device = set_device()

x_train, x_test = load_data(data_dir, dataset_name)
x_train, y_train, t_train, ps_train = x_train[:,:2].astype(int), x_train[:,2:3], x_train[:,3:4], x_train[:,4:]
x_test, cate_test = x_test[:,:2].astype(int), x_test[:,2]

num_users = x_train[:,0].max()+1
num_items = x_train[:,1].max()+1
print(f"# user: {num_users}, # item: {num_items}")

x_test_tensor = torch.LongTensor(x_test).to(device)
true_propensity = logit(ps_train).squeeze()


#%%
model_name_generator = lambda method, arch: f"multi_{method}_{arch}_{dataset_name[:3]}_seed{random_seed}"
# model_name_generator = lambda method, arch: f"escm2_{method}_{arch}_{dataset_name[:3]}_seed{random_seed}"

# for i, model_name_generator in enumerate(model_name_generator_list):
naive_y1_name = model_name_generator("naive", "y1")
naive_y0_name = model_name_generator("naive", "y0")

naive_y1 = SharedNCF(num_users, num_items, embedding_k)
naive_y1 = naive_y1.to(device)
naive_y1.load_state_dict(torch.load(f"{weights_dir}/{naive_y1_name}.pth", map_location=device))

naive_y0 = SharedNCF(num_users, num_items, embedding_k)
naive_y0 = naive_y0.to(device)
naive_y0.load_state_dict(torch.load(f"{weights_dir}/{naive_y0_name}.pth", map_location=device))


ips_y1_name = model_name_generator("ips", "y1")
ips_y0_name = model_name_generator("ips", "y0")

ips_y1 = SharedNCF(num_users, num_items, embedding_k)
ips_y1 = ips_y1.to(device)
ips_y1.load_state_dict(torch.load(f"{weights_dir}/{ips_y1_name}.pth", map_location=device))

ips_y0 = SharedNCF(num_users, num_items, embedding_k)
ips_y0 = ips_y0.to(device)
ips_y0.load_state_dict(torch.load(f"{weights_dir}/{ips_y0_name}.pth", map_location=device))


naive_y1.eval()
naive_y0.eval()
preds_naive_y1 = naive_y1(x_test_tensor)
preds_naive_y0 = naive_y0(x_test_tensor)

ips_y1.eval()
ips_y0.eval()
preds_ips_y1 = ips_y1(x_test_tensor)
preds_ips_y0 = ips_y0(x_test_tensor)

pred_naive_y1 = preds_naive_y1[0].cpu().detach().numpy().squeeze()
pred_naive_y0 = preds_naive_y0[0].cpu().detach().numpy().squeeze()

pred_ips_y1 = preds_ips_y1[0].cpu().detach().numpy().squeeze()
pred_ips_y0 = preds_ips_y0[0].cpu().detach().numpy().squeeze()

# np.save(f"{save_dir}/pred_t_{model_name_y1}.npy", pred_model_y1, allow_pickle=True)
# np.save(f"{save_dir}/pred_t_{model_name_y0}.npy", pred_model_y0, allow_pickle=True)

# %%
data_arr = np.array([pred_naive_y1, pred_naive_y0, pred_ips_y1, pred_ips_y0])
x_min_max = (data_arr.min(), data_arr.max())
x_range = np.linspace(x_min_max[0], x_min_max[1], 1000)

density_file_name = f"{data_dir}/density_yt_multi_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy"
# density_file_name = f"{data_dir}/density_yt_escm2_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy"
if os.path.exists(density_file_name):
    density_arr = np.load(density_file_name)
        
else:
    density = []
    for data in data_arr:
        kde = gaussian_kde(data)
        density.append(kde(x_range))
    density_arr = np.array(density)
    np.save(density_file_name, density_arr, allow_pickle=True)

np.save(f"{data_dir}/pred_yt_multi_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy", data_arr, allow_pickle=True)
# np.save(f"{data_dir}/pred_yt_escm2_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy", data_arr, allow_pickle=True)
