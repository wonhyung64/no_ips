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
from module.utils import set_device, set_seed


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

# save_dir = "./visualize_data"
# os.makedirs(save_dir, exist_ok=True)
# np.save(f"{save_dir}/true_propensity_{dataset_name[:3]}", true_propensity, allow_pickle=True)


#%%
# model_name_generator_list = [
#     lambda method, arch: f"multi_{method}_{arch}_{dataset_name[:3]}_seed{random_seed}",
#     lambda method, arch: f"escm2_{method}_{arch}_{dataset_name[:3]}_seed{random_seed}",
# ]
model_name_generator = lambda method, arch: f"multi_{method}_{arch}_{dataset_name[:3]}_seed{random_seed}"

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

file_name = f"{data_dir}/density_yt_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy"
if os.path.exists(file_name):
    density_arr = np.load(file_name)
        
else:
    density = []
    for data in data_arr:
        kde = gaussian_kde(data)
        density.append(kde(x_range))
    density_arr = np.array(density)
    np.save(file_name, density_arr, allow_pickle=True)


# %%
# VISUALIZATION OPTIONS
font_size=16
# font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

# fontprop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = fontprop.get_name()
# plt.rcParams['font.weight'] = 'bold'

#%%
labels = ["naive_y1", "naive_y0", "ips_y1", "ips_y0"]
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 4))
for j, (density, label, color) in enumerate(zip(density_arr, labels, colors)):
    line = ax.plot(x_range, density, label=label, color=color)

    ax.set_ylim(0, density_arr.max())
    ax.set_xlim(np.floor(x_min_max[0]), np.ceil(x_min_max[1]))

    ax.tick_params(axis='x', which='both', top=False, labelsize=font_size)
    ax.tick_params(axis='y', which='both', labelsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('')
    ax.set_ylabel('Density', fontsize=font_size, fontweight="bold")

    ax.set_title(f'Multi', fontsize=font_size, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

# if dataset_name == "personalized":
#     legend = fig.legend(handles, labels, loc='center', ncol=4, fontsize=font_size, frameon=True, bbox_to_anchor=(0.54, -0.1))
#     legend.get_frame().set_edgecolor('black')

plt.tight_layout()
plt.show()


# %%
# if dataset_name == "original":
#     fig.savefig("./kde_original.pdf", bbox_inches='tight')
# elif dataset_name == "personalized":
#     fig.savefig("./kde_personalized.pdf", bbox_inches="tight")

# %%

