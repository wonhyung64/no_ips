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

from module.model import SharedNCF, SharedNCFPlus
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

data_dir = f"./visualize_data"


#%%
ps_label_name_list = ["Multi-IPS", "ESCM2-IPS"]
ps_model_name_list = ["multi", "escm2"]
labels_list = [["Y(1)", "Y(0)", "Plus", "True"], ["Y(1)", "Y(0)", "Y(1) + Y(0)", "True"]]
density_list = []

for ps_model_name in ps_model_name_list:
    ps_pred_y1 = np.load(f"{data_dir}/pred_t_{ps_model_name}_ips_y1_{dataset_name[:3]}_seed{random_seed}.npy")
    ps_pred_y0 = np.load(f"{data_dir}/pred_t_{ps_model_name}_ips_y0_{dataset_name[:3]}_seed{random_seed}.npy")
    ps_pred_plus = np.load(f"{data_dir}/pred_t_{ps_model_name}_ips_plus_{dataset_name[:3]}_seed{random_seed}.npy")
    ps_true = np.load(f"{data_dir}/true_propensity_{dataset_name[:3]}.npy")
    
    data_arr = np.array([ps_pred_y1, ps_pred_y0, ps_pred_plus, ps_true])
    x_min_max = (data_arr.min(), data_arr.max())

    if dataset_name == "original":
        x_range = np.linspace(x_min_max[0], x_min_max[1], 1000)
    elif dataset_name == "personalized":
        x_range = np.linspace(-8, 14, 1000)

    file_name = f"{data_dir}/density_t_{ps_model_name}_ips_{dataset_name[:3]}_seed{random_seed}.npy"
    if os.path.exists(file_name):
        density = np.load(file_name)
        
    else:
        density = []
        for data in data_arr:
            kde = gaussian_kde(data)
            density.append(kde(x_range))
        np.save(f"{data_dir}/density_t_{ps_model_name}_ips_{dataset_name[:3]}_seed{random_seed}.npy", np.array(density), allow_pickle=True)

    density_list.append(density)


# %%
# VISUALIZATION OPTIONS
font_size=16
font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['font.weight'] = 'bold'

#%%
handles, labels = [], []
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(8, 2))
for i, (densities, labels, ax) in enumerate(zip(density_list, labels_list, axes)):

    for j, (density, label, color) in enumerate(zip(densities, labels, colors)):
        # if j != 3:
        #     continue
        line = ax.plot(x_range, density, label=label, color=color)
        if i == 0:
            handles.append(line[0])
            labels.append(label)

        ax.set_ylim(0, 0.5)
        if dataset_name == "original": 
            ax.set_xlim(np.floor(x_min_max[0]), 4)
        else:
            ax.set_xlim(-8, 14)

        ax.tick_params(axis='x', which='both', top=False, labelsize=font_size)
        ax.tick_params(axis='y', which='both', labelsize=font_size)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('')
        if i == 0:
            ax.set_ylabel('Density', fontsize=font_size, fontweight="bold")

        ax.set_title(f'{ps_label_name_list[i]}', fontsize=font_size, fontweight="bold")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

if dataset_name == "personalized":
    legend = fig.legend(handles, labels, loc='center', ncol=4, fontsize=font_size, frameon=True, bbox_to_anchor=(0.54, -0.1))
    legend.get_frame().set_edgecolor('black')

plt.tight_layout()
plt.show()


# %%
if dataset_name == "original":
    fig.savefig("./kde_original.pdf", bbox_inches='tight')
elif dataset_name == "personalized":
    fig.savefig("./kde_personalized.pdf", bbox_inches="tight")

# %%

