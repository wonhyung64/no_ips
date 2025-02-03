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


#%%
n_factors = 8
n_items = 20
n_samples = 100
treat_bias_list = [-1.9, -3.6, -4.8, -6.5]
lr = 1e-2
repeat_num = 30
num_epochs = 500
batch_size = 512
device = set_device()

mle = torch.nn.BCELoss(reduction="none")
ipw = lambda x, y, z: F.binary_cross_entropy(x, y, z, reduction="none")

for treat_bias in treat_bias_list:
    T_list, mle_auc_list, ipw_auc_list = [], [], []
    for random_seed in range(1, repeat_num+1):
        set_seed(random_seed)

        Z_treat = np.random.normal(0, 1, (n_items, n_factors))
        Lambda_treat = np.random.uniform(0., 1., (n_samples, n_factors))

        logit_t_real = Lambda_treat @ Z_treat.T + treat_bias
        prob_t_real = sigmoid(logit_t_real)
        T_real = np.random.binomial(1, prob_t_real)
        T_list.append(T_real.mean())

        Z_effect = np.random.normal(0, 1, (n_items, n_factors))
        Lambda_effect = np.random.uniform(0., 1., (n_samples, n_factors))
        treatment_effect = Lambda_effect @ Z_effect.T

        Z_interact = np.random.normal(0, 1, (n_items, n_factors))
        Lambda_interact = np.random.uniform(0., 1., (n_samples, n_factors))
        prob_y1 = sigmoid(Lambda_interact @ Z_interact.T  + nn.ReLU()(torch.tensor(treatment_effect)).numpy())
        prob_y0 = sigmoid(Lambda_interact @ Z_interact.T)

        Y1 = np.random.binomial(1, prob_y1)
        Y0 = np.random.binomial(1, prob_y0)
        Y = Y1 * T_real + Y0 * (1-T_real)

        Y_train = Y[T_real==1]
        user_idx, item_idx = np.where(T_real==1)
        x_train = np.concatenate([[user_idx],[item_idx]]).T
        ps_train = prob_t_real[T_real==1]

        Y_test = Y1.reshape(-1)
        user_idx, item_idx = np.meshgrid(np.arange(n_samples), np.arange(n_items))
        x_test = np.concatenate([[user_idx, item_idx]]).T.reshape(-1, 2)

        obs = sps.csr_matrix((np.ones(len(Y_train)), (x_train[:, 0], x_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
        y_entire = sps.csr_matrix((Y_train, (x_train[:, 0], x_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
        ps_entire = sps.csr_matrix((ps_train, (x_train[:, 0], x_train[:, 1])), shape=(n_samples, n_items), dtype=np.float32).toarray().reshape(-1)
        x_all = generate_total_sample(n_samples, n_items)

        num_samples = len(x_all)
        total_batch = num_samples // batch_size

        """mle simulation"""
        model = MF(n_samples, n_items, n_factors)
        model = model.to("mps")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs+1):
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)
            model.train()

            epoch_total_loss = 0.
            for idx in range(total_batch):
                selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_all[selected_idx]
                sub_x = torch.LongTensor(sub_x).to(device)
                sub_y = y_entire[selected_idx]
                sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                sub_t = obs[selected_idx]
                sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)

                pred, user_embed, item_embed = model(sub_x)

                rec_loss = mle(torch.nn.Sigmoid()(pred), sub_y)
                rec_loss = (rec_loss * sub_t).mean()
                total_loss = rec_loss
                epoch_total_loss += total_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        model.eval()
        sub_x = torch.LongTensor(x_test).to(device)
        pred_, _, __ = model(sub_x)
        pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

        fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
        mle_auc = auc(fpr, tpr)
        mle_auc_list.append(mle_auc)


        """ipw simulation"""
        model = MF(n_samples, n_items, n_factors)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, num_epochs+1):
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)
            model.train()

            epoch_total_loss = 0.
            for idx in range(total_batch):
                selected_idx = ul_idxs[batch_size*idx:(idx+1)*batch_size]
                sub_x = x_all[selected_idx]
                sub_x = torch.LongTensor(sub_x).to(device)
                sub_y = y_entire[selected_idx]
                sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)
                sub_t = obs[selected_idx]
                sub_t = torch.Tensor(sub_t).unsqueeze(-1).to(device)
                sub_ps = ps_entire[selected_idx]
                sub_ps = torch.Tensor(sub_ps).unsqueeze(-1).to(device)

                pred, user_embed, item_embed = model(sub_x)
                rec_loss = ipw(torch.nn.Sigmoid()(pred), sub_y, 1/(sub_ps+1e-9))
                rec_loss = (rec_loss * sub_t).mean()
                total_loss = rec_loss
                epoch_total_loss += total_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        model.eval()
        sub_x = torch.LongTensor(x_test).to(device)
        pred_, _, __ = model(sub_x)
        pred = nn.Sigmoid()(pred_).detach().cpu().numpy()

        fpr, tpr, thresholds = roc_curve(Y_test, pred, pos_label=1)
        ipw_auc = auc(fpr, tpr)
        ipw_auc_list.append(ipw_auc)

    print(treat_bias)
    print(f"{n_samples} users, {n_items} items, {n_factors} factors")
    print(f"T_bar : {np.mean(T_list)}")
    print(np.mean(mle_auc_list))
    print(np.std(mle_auc_list))
    print(np.mean(ipw_auc_list))
    print(np.std(ipw_auc_list))
    print()
