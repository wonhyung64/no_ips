#%%
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.dataset import generate_total_sample
from module.model import MF
from module.utils import set_seed, set_device, sigmoid
from module.metric import cdcg_func, cp_func, car_func

#%%
n_factors_list = [4,8,16]
n_items_list = [60] #[20, 60]
n_samples_list = [1000] # [100,1000]
treat_bias = 0.
lr = 1e-2
repeat_num = 30
num_epochs = 500
batch_size = 512
device = set_device()

mle = torch.nn.BCELoss(reduction="none")
ipw = lambda x, y, z: F.binary_cross_entropy(x, y, z, reduction="none")

for n_samples in n_samples_list:
    for n_items in n_items_list:
        for n_factors in n_factors_list:
            top_k = n_items
            T_list = []
            naive_cdcg_list, naive_cp_list, naive_car_list = [], [], []
            ips_cdcg_list, ips_cp_list, ips_car_list = [], [], []

            for random_seed in range(1, repeat_num+1):
                set_seed(random_seed)

                Z_treat = np.random.normal(0, 1, (n_items, n_factors))
                Lambda_treat = np.random.uniform(0., 1., (n_samples, n_factors))

                logit_t_real = Lambda_treat @ Z_treat.T  + treat_bias
                prob_t_real = sigmoid(logit_t_real)
                T_real = np.random.binomial(1, prob_t_real)
                T_list.append(T_real.mean())

                Z_effect = np.random.normal(0, 1, (n_items, n_factors))
                Lambda_effect = np.random.uniform(0., 1., (n_samples, n_factors))
                treatment_effect = Lambda_effect @ Z_effect.T

                Z_interact = np.random.normal(0, 1, (n_items, n_factors))
                Lambda_interact = np.random.uniform(0., 1., (n_samples, n_factors))
                prob_y1 = sigmoid(Lambda_interact @ Z_interact.T + nn.ReLU()(torch.tensor(treatment_effect)).numpy())
                prob_y0 = sigmoid(Lambda_interact @ Z_interact.T)

                Y1 = np.random.binomial(1, prob_y1)
                Y0 = np.random.binomial(1, prob_y0)
                Y = Y1 * T_real + Y0 * (1-T_real)

                true_cate = prob_y1 - prob_y0

                x_all = generate_total_sample(n_samples, n_items)

                Y1_train = Y[T_real==1]
                user_idx, item_idx = np.where(T_real==1)
                x1_train = np.concatenate([[user_idx],[item_idx]]).T
                ps1_train = prob_t_real[T_real==1]

                obs1 = sps.csr_matrix((np.ones(len(Y1_train)), (x1_train[:, 0], x1_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
                y1_entire = sps.csr_matrix((Y1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
                ps1_entire = sps.csr_matrix((ps1_train, (x1_train[:, 0], x1_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)

                Y0_train = Y[T_real==0]
                user_idx, item_idx = np.where(T_real==0)
                x0_train = np.concatenate([[user_idx],[item_idx]]).T
                ps0_train = 1 - prob_t_real[T_real==0]

                obs0 = sps.csr_matrix((np.ones(len(Y0_train)), (x0_train[:, 0], x0_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
                y0_entire = sps.csr_matrix((Y0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
                ps0_entire = sps.csr_matrix((ps0_train, (x0_train[:, 0], x0_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)

                cate_test = true_cate.reshape(-1)
                user_idx, item_idx = np.meshgrid(np.arange(n_samples), np.arange(n_items))

                num_samples = len(x_all)
                total_batch = num_samples // batch_size

                """naive simulation"""
                model_y1 = MF(n_samples, n_items, n_factors)
                model_y1 = model_y1.to(device)
                optimizer = torch.optim.Adam(model_y1.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    ul_idxs = np.arange(x_all.shape[0]) # all
                    np.random.shuffle(ul_idxs)
                    model_y1.train()

                    for idx in range(total_batch):
                        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_all[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = y1_entire[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                        sub_t = obs1[selected_idx]
                        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model_y1(sub_x)

                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        rec_loss = (rec_loss * sub_t).mean()
                        total_loss = rec_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model_y0 = MF(n_samples, n_items, n_factors)
                model_y0 = model_y0.to(device)
                optimizer = torch.optim.Adam(model_y0.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    ul_idxs = np.arange(x_all.shape[0]) # all
                    np.random.shuffle(ul_idxs)
                    model_y0.train()

                    for idx in range(total_batch):
                        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_all[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = y0_entire[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                        sub_t = obs0[selected_idx]
                        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model_y0(sub_x)

                        rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                        rec_loss = (rec_loss * sub_t).mean()
                        total_loss = rec_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model_y1.eval()
                model_y0.eval()

                sub_x = torch.LongTensor(x_all).to(device)
                pred_y1, _, __ = model_y1(sub_x)
                pred_y0, _, __ = model_y0(sub_x)
                pred = (nn.Sigmoid()(pred_y1) - nn.Sigmoid()(pred_y0)).detach().cpu().numpy().squeeze()

                cdcg_dict = cdcg_func(pred, x_all, cate_test, top_k_list=[top_k])
                cdcg_res = np.mean(cdcg_dict[f"cdcg_{top_k}"])

                cp_dict = cp_func(pred, x_all, cate_test, top_k_list=[top_k])
                cp_res = np.mean(cp_dict[f"cp_{top_k}"])

                car_dict = car_func(pred, x_all, cate_test, top_k_list=[top_k])
                car_res = np.mean(car_dict[f"car_{top_k}"])

                naive_cdcg_list.append(cdcg_res)
                naive_cp_list.append(cp_res)
                naive_car_list.append(car_res)

                """ipw simulation"""
                model_y1 = MF(n_samples, n_items, n_factors)
                model_y1 = model_y1.to(device)
                optimizer = torch.optim.Adam(model_y1.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    ul_idxs = np.arange(x_all.shape[0]) # all
                    np.random.shuffle(ul_idxs)
                    model_y1.train()

                    for idx in range(total_batch):
                        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_all[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = y1_entire[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                        sub_t = obs1[selected_idx]
                        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
                        sub_ps = ps1_entire[selected_idx]
                        sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model_y1(sub_x)

                        rec_loss = ipw(torch.nn.Sigmoid()(pred), sub_y, 1/(sub_ps+1e-9))
                        rec_loss = (rec_loss * sub_t).mean()
                        total_loss = rec_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model_y0 = MF(n_samples, n_items, n_factors)
                model_y0 = model_y0.to(device)
                optimizer = torch.optim.Adam(model_y0.parameters(), lr=lr)

                for epoch in range(1, num_epochs+1):
                    ul_idxs = np.arange(x_all.shape[0]) # all
                    np.random.shuffle(ul_idxs)
                    model_y0.train()

                    for idx in range(total_batch):
                        selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                        sub_x = x_all[selected_idx]
                        sub_x = torch.LongTensor(sub_x).to(device)
                        sub_y = y0_entire[selected_idx]
                        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                        sub_t = obs0[selected_idx]
                        sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
                        sub_ps = ps0_entire[selected_idx]
                        sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

                        pred, user_embed, item_embed = model_y0(sub_x)

                        rec_loss = ipw(torch.nn.Sigmoid()(pred), sub_y, 1/(sub_ps+1e-9))
                        rec_loss = (rec_loss * sub_t).mean()
                        total_loss = rec_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                model_y1.eval()
                model_y0.eval()

                sub_x = torch.LongTensor(x_all).to(device)
                pred_y1, _, __ = model_y1(sub_x)
                pred_y0, _, __ = model_y0(sub_x)
                pred = (nn.Sigmoid()(pred_y1) - nn.Sigmoid()(pred_y0)).detach().cpu().numpy().squeeze()

                cdcg_dict = cdcg_func(pred, x_all, cate_test, top_k_list=[top_k])
                cdcg_res = np.mean(cdcg_dict[f"cdcg_{top_k}"])

                cp_dict = cp_func(pred, x_all, cate_test, top_k_list=[top_k])
                cp_res = np.mean(cp_dict[f"cp_{top_k}"])

                car_dict = car_func(pred, x_all, cate_test, top_k_list=[top_k])
                car_res = np.mean(car_dict[f"car_{top_k}"])

                ips_cdcg_list.append(cdcg_res)
                ips_cp_list.append(cp_res)
                ips_car_list.append(car_res)

                cdcg_dict = cdcg_func(cate_test, x_all, cate_test, top_k_list=[n_items])

            print(f"{n_samples} users, {n_items} items, {n_factors} factors")
            print(f"T_bar : {np.mean(T_list)}")
            print("cdcg")
            print(np.mean(naive_cdcg_list))
            print(np.std(naive_cdcg_list))
            print(np.mean(ips_cdcg_list))
            print(np.std(ips_cdcg_list))

            print("cp")
            print(np.mean(naive_cp_list))
            print(np.std(naive_cp_list))
            print(np.mean(ips_cp_list))
            print(np.std(ips_cp_list))

            print("car")
            print(np.mean(naive_car_list))
            print(np.std(naive_car_list))
            print(np.mean(ips_car_list))
            print(np.std(ips_car_list))

            print()
