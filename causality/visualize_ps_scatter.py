#%%
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm

from module.dataset import load_data
from module.utils import set_device, set_seed


def logit(x):
    return np.log(x/(1-x))


#%%
parser = argparse.ArgumentParser()

# parser.add_argument("--dataset-name", type=str, default="original")#[original, personalized]
parser.add_argument("--dataset-name", type=str, default="personalized")#[original, personalized]
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


embedding_k = args.embedding_k
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

#%%
x_train, x_test = load_data(data_dir, dataset_name)
x_train, y_train, t_train, ps_train = x_train[:,:2].astype(int), x_train[:,2:3], x_train[:,3:4], x_train[:,4:]
x_test, cate_test = x_test[:,:2].astype(int), x_test[:,2]

num_users = x_train[:,0].max()+1
num_items = x_train[:,1].max()+1
print(f"# user: {num_users}, # item: {num_items}")

x_test_tensor = torch.LongTensor(x_test).to(device)


#%%
# VISUALIZATION OPTIONS
data_dir = f"./visualize_data"
font_size=13
font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['font.weight'] = 'bold'
np.random.seed(random_seed)

#%%
ps_model_name_list = ["multi", "escm2"]
subtitle_list = ["Multi-IPS", "ESCM2"]
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8, 4))
for i, (ax, ps_model_name) in enumerate(zip(axes, ps_model_name_list)):
    ps_pred_plus = np.load(f"{data_dir}/pred_t_{ps_model_name}_ips_plus_{dataset_name[:3]}_seed{random_seed}.npy")
    ps_true = np.load(f"{data_dir}/true_propensity_{dataset_name[:3]}.npy")
    sample_idx = np.random.choice(np.arange(0, len(ps_true)), 200)

    scatter = ax.scatter(ps_true[sample_idx], ps_pred_plus[sample_idx], alpha=0.6, edgecolors="black", c="#1f77b4")

    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', which='both', labelsize=font_size)

    if i == 0:
        ax.set_ylabel('Pred', fontsize=font_size, fontweight="bold")
    ax.set_xlabel('')

    ax.grid(True, linestyle='--', alpha=0.5)
    fig.text(0.5, 0, 'True', ha='center', fontsize=12)

    ax.set_title(f'{subtitle_list[i]}', fontsize=font_size, fontweight="bold")

plt.tight_layout()
plt.show()

#%%
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

if dataset_name == "original":
    legend = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=font_size, frameon=True, bbox_to_anchor=(0.54, 1.25))
    legend.get_frame().set_edgecolor('black')



# %%
if dataset_name == "original":
    fig.savefig("./kde_original.pdf", bbox_inches='tight')
elif dataset_name == "personalized":
    fig.savefig("./kde_personalized.pdf", bbox_inches="tight")

# %%