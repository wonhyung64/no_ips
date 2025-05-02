#%%
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime

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


# %%
density_file_name = f"{data_dir}/density_yt_multi_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy"
multi_density_arr = np.load(density_file_name)
multi_data_arr = np.load(f"{data_dir}/pred_yt_multi_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy")

density_file_name = f"{data_dir}/density_yt_escm2_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy"
escm2_density_arr = np.load(density_file_name)
escm2_data_arr = np.load(f"{data_dir}/pred_yt_escm2_naive_ips_{dataset_name[:3]}_seed{random_seed}.npy")

# %%
# VISUALIZATION OPTIONS
font_size=16
font_path = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['font.weight'] = 'bold'


#%%
def mean_mse(true, pred):
    return np.mean(np.square(true-pred))

def min_mse(true, pred):
    return np.min(np.square(true-pred))


multi_mean_results = []
multi_min_results = []
topk_list = [10, 100, num_items]
for u in range(num_users):
    user_multi_mean_results = []
    user_multi_min_results = []
    for topk in topk_list:
        user_idx = x_test[:,0] == u
        user_true = cate_test[user_idx]

        user_naive_y1 = sigmoid(multi_data_arr[0][user_idx])
        user_naive_y0 = sigmoid(multi_data_arr[1][user_idx])
        user_naive_pred = user_naive_y1 - user_naive_y0

        user_ips_y1 = sigmoid(multi_data_arr[2][user_idx])
        user_ips_y0 = sigmoid(multi_data_arr[3][user_idx])
        user_ips_pred = user_ips_y1 - user_ips_y0

        user_naive_true_topk = user_true[np.argsort(-user_naive_pred)][:topk]
        user_ips_true_topk = user_true[np.argsort(-user_ips_pred)][:topk]

        user_naive_pred_topk = user_naive_pred[np.argsort(-user_naive_pred)][:topk]
        user_ips_pred_topk = user_ips_pred[np.argsort(-user_ips_pred)][:topk]

        user_multi_mean_results.append(mean_mse(user_naive_true_topk, user_naive_pred_topk))
        user_multi_mean_results.append(mean_mse(user_ips_true_topk, user_ips_pred_topk))
        user_multi_min_results.append(min_mse(user_naive_true_topk, user_naive_pred_topk))
        user_multi_min_results.append(min_mse(user_ips_true_topk, user_ips_pred_topk))
    multi_mean_results.append(user_multi_mean_results)
    multi_min_results.append(user_multi_min_results)

multi_mean_arr = np.array(multi_mean_results)
multi_min_arr = np.array(multi_min_results)


escm2_mean_results = []
escm2_min_results = []
for u in range(num_users):
    user_escm2_mean_results = []
    user_escm2_min_results = []
    for topk in topk_list:
        user_idx = x_test[:,0] == u
        user_true = cate_test[user_idx]

        user_naive_y1 = sigmoid(escm2_data_arr[0][user_idx])
        user_naive_y0 = sigmoid(escm2_data_arr[1][user_idx])
        user_naive_pred = user_naive_y1 - user_naive_y0

        user_ips_y1 = sigmoid(escm2_data_arr[2][user_idx])
        user_ips_y0 = sigmoid(escm2_data_arr[3][user_idx])
        user_ips_pred = user_ips_y1 - user_ips_y0

        user_naive_true_topk = user_true[np.argsort(-user_naive_pred)][:topk]
        user_ips_true_topk = user_true[np.argsort(-user_ips_pred)][:topk]

        user_naive_pred_topk = user_naive_pred[np.argsort(-user_naive_pred)][:topk]
        user_ips_pred_topk = user_ips_pred[np.argsort(-user_ips_pred)][:topk]

        user_escm2_mean_results.append(mean_mse(user_naive_true_topk, user_naive_pred_topk))
        user_escm2_mean_results.append(mean_mse(user_ips_true_topk, user_ips_pred_topk))

        user_escm2_min_results.append(min_mse(user_naive_true_topk, user_naive_pred_topk))
        user_escm2_min_results.append(min_mse(user_ips_true_topk, user_ips_pred_topk))

    escm2_mean_results.append(user_escm2_mean_results)
    escm2_min_results.append(user_escm2_min_results)

escm2_mean_arr = np.array(escm2_mean_results)
escm2_min_arr = np.array(escm2_min_results)


#%%

#%%
fig, axes = plt.subplots(2, 4, figsize=(9, 3))
for i, (mean_arr, min_arr) in enumerate([[multi_mean_arr, multi_min_arr], [escm2_mean_arr, escm2_min_arr]]):
    for j, k in enumerate([0,2]):
        all_preds = np.concatenate([mean_arr[k], mean_arr[k+1]])
        bins = np.linspace(all_preds.min(), all_preds.max(), 20)

        axes[i][j].hist(mean_arr[:,k], bins=bins, alpha=0.7, label="naive", density=True, color="#1f77b4", edgecolor="black")
        axes[i][j].hist(mean_arr[:,k+1], bins=bins, alpha=0.7, label="ips", density=True, color="#d62728", edgecolor="black")

        all_preds = np.concatenate([min_arr[k], min_arr[k+1]])
        bins = np.linspace(all_preds.min(), all_preds.max(), 10)

        axes[i][j+2].hist(min_arr[:,k], bins=bins, alpha=0.7, label="naive", density=True, color="#1f77b4", edgecolor="black")
        axes[i][j+2].hist(min_arr[:,k+1], bins=bins, alpha=0.7, label="ips", density=True, color="#d62728", edgecolor="black")

        if j==0:
            if i==0:
                axes[i][j].set_ylabel('Multi', fontsize=font_size, weight="bold")
            if i==1:
                axes[i][j].set_ylabel('ESCM2', fontsize=font_size, weight="bold")
        axes[i][j].grid(True, alpha=0.5, linestyle="--")
        axes[i][j+2].grid(True, alpha=0.5, linestyle="--")

        axes[i][j].tick_params(axis='x', which='both', top=False, labelsize=font_size-3.5)
        axes[i][j].tick_params(axis='y', which='both', labelsize=font_size-3.5)

        axes[i][j+2].tick_params(axis='x', which='both', top=False, labelsize=font_size-3.5)
        axes[i][j+2].tick_params(axis='y', which='both', labelsize=font_size-3.5)

    if i==0:
        axes[i, 0].set_title("Top 10", fontsize=font_size, pad=8, weight="bold")
        axes[i, 1].set_title("Top 100", fontsize=font_size, pad=8, weight="bold")
        axes[i, 2].set_title("Top 10", fontsize=font_size, pad=8, weight="bold")
        axes[i, 3].set_title("Top 100", fontsize=font_size, pad=8, weight="bold")


fig.text(0.28, -0.03, "Mean", ha='center', fontsize=font_size, weight='bold')
fig.text(0.78, -0.03, "Min", ha='center', fontsize=font_size, weight='bold')

legend = fig.legend(['Naive', 'IPS'],
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.53, 1.15),
    frameon=True,
    fontsize=font_size-2,
    )
legend.get_frame().set_edgecolor('black')


    # 출력
plt.tight_layout()
plt.show()

fig.savefig(f"./{dataset_name}_seperate_mse.pdf", bbox_inches='tight')
#%%
np.random.seed(random_seed)
sample_idx = np.random.choice(multi_data_arr.shape[1], 500)

fig, axes = plt.subplots(1, 4, figsize=(9, 2))

axes[0].scatter(multi_data_arr[0, sample_idx], multi_data_arr[2, sample_idx], edgecolors="black", alpha=0.8, color="#1f77b4", s=25)
axes[0].set_ylabel('IPS', fontsize=font_size, weight="bold")

axes[1].scatter(multi_data_arr[1, sample_idx], multi_data_arr[3, sample_idx], edgecolors="black", alpha=0.8, color="#2ca02c", s=25)

axes[2].scatter(escm2_data_arr[0, sample_idx], escm2_data_arr[2, sample_idx], edgecolors="black", alpha=0.8, color="#1f77b4", s=25)

axes[3].scatter(escm2_data_arr[1, sample_idx], escm2_data_arr[3, sample_idx], edgecolors="black", alpha=0.8, color="#2ca02c", s=25)

fig.text(0.54, -0.05, "Naive", ha='center', fontsize=font_size, weight='bold')
fig.text(0.3, 1., "Multi", ha='center', fontsize=font_size, weight='bold')
fig.text(0.77, 1., "ESCM2", ha='center', fontsize=font_size, weight='bold')

for ax in axes:
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.tick_params(axis='x', which='both', top=False, labelsize=font_size)
    ax.tick_params(axis='y', which='both', labelsize=font_size)

legend = fig.legend(['Y(1)', 'Y(0)'],
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.53, -0.35),
    fontsize=font_size-2,
    frameon=True)
legend.get_frame().set_edgecolor('black')

plt.tight_layout()
plt.show()


fig.savefig(f"./{dataset_name}_seperate_outcome.pdf", bbox_inches='tight')
# %%
