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

from module.model import NCF, NCFPlus, SharedNCF, SharedNCFPlus
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

save_dir = "./visualize_data"
os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/true_propensity_{dataset_name[:3]}", true_propensity, allow_pickle=True)


#%%
model_name_generator_list = [
    lambda x: f"escm2_ips_{x}_{dataset_name[:3]}_seed{random_seed}",
    lambda x: f"multi_ips_{x}_{dataset_name[:3]}_seed{random_seed}",
    lambda x: f"ips_{x}_{dataset_name[:3]}_escm2_{random_seed}",
    lambda x: f"ips_{x}_{dataset_name[:3]}_multi_{random_seed}",
]


for i, model_name_generator in enumerate(model_name_generator_list):
    model_name_y1 = model_name_generator("y1")
    model_name_y0 = model_name_generator("y0")
    model_name_plus = model_name_generator("plus")

    if re.match(r"^ips*", model_name_generator('tmp')):
        model_y1 = NCF(num_users, num_items, embedding_k)
        model_y1 = model_y1.to(device)
        model_y1.load_state_dict(torch.load(f"{weights_dir}/{model_name_y1}.pth", map_location=device))

        model_y0 = NCF(num_users, num_items, embedding_k)
        model_y0 = model_y0.to(device)
        model_y0.load_state_dict(torch.load(f"{weights_dir}/{model_name_y0}.pth", map_location=device))

        model_plus = NCFPlus(num_users, num_items, embedding_k)
        model_plus = model_plus.to(device)
        model_plus.load_state_dict(torch.load(f"{weights_dir}/{model_name_plus}.pth", map_location=device))

    else:
        model_y1 = SharedNCF(num_users, num_items, embedding_k)
        model_y1 = model_y1.to(device)
        model_y1.load_state_dict(torch.load(f"{weights_dir}/{model_name_y1}.pth", map_location=device))

        model_y0 = SharedNCF(num_users, num_items, embedding_k)
        model_y0 = model_y0.to(device)
        model_y0.load_state_dict(torch.load(f"{weights_dir}/{model_name_y0}.pth", map_location=device))

        model_plus = SharedNCFPlus(num_users, num_items, embedding_k)
        model_plus = model_plus.to(device)
        model_plus.load_state_dict(torch.load(f"{weights_dir}/{model_name_plus}.pth", map_location=device))


    model_y1.eval()
    model_y0.eval()
    model_plus.eval()

    preds_model_y1 = model_y1(x_test_tensor)
    preds_model_y0 = model_y0(x_test_tensor)
    preds_model_plus = model_plus(x_test_tensor)

    if re.match(r"^ips*", model_name_generator('tmp')):
        pred_model_y1 = preds_model_y1[0].cpu().detach().numpy().squeeze()
        pred_model_y0 = preds_model_y0[0].cpu().detach().numpy().squeeze()
        pred_model_plus_y1 = preds_model_plus[0].cpu().detach().numpy().squeeze()
        pred_model_plus_y0 = preds_model_plus[1].cpu().detach().numpy().squeeze()

        np.save(f"{save_dir}/pred_y1_{model_name_y1}.npy", pred_model_y1, allow_pickle=True)
        np.save(f"{save_dir}/pred_y0_{model_name_y0}.npy", pred_model_y0, allow_pickle=True)
        np.save(f"{save_dir}/pred_y1_{model_name_plus}.npy", pred_model_plus_y1, allow_pickle=True)
        np.save(f"{save_dir}/pred_y0_{model_name_plus}.npy", pred_model_plus_y0, allow_pickle=True)

    else:
        pred_model_y1 = preds_model_y1[1].cpu().detach().numpy().squeeze()
        pred_model_y0 = preds_model_y0[1].cpu().detach().numpy().squeeze()
        pred_model_plus = preds_model_plus[2].cpu().detach().numpy().squeeze()

        np.save(f"{save_dir}/pred_t_{model_name_y1}.npy", pred_model_y1, allow_pickle=True)
        np.save(f"{save_dir}/pred_t_{model_name_y0}.npy", pred_model_y0, allow_pickle=True)
        np.save(f"{save_dir}/pred_t_{model_name_plus}.npy", pred_model_plus, allow_pickle=True)

# %%
