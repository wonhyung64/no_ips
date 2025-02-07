#%%
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy.random.mtrand import RandomState


class MF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        torch.nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        torch.nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)

        return user_embed, item_embed


def func_sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.exp(x) / (1.0 + np.exp(x))


def print_result(result_list, top_k_list):
    for k, top_k in enumerate(top_k_list):
        print(f"Top {top_k}")
        print(f"Avg. : {np.mean(np.array(result_list)[:, k]).round(6)}, Std. : {np.std(np.array(result_list)[:, k]).round(6)}")
        print()


def sample_excluding(A, B, n):
    A_set = set(A)
    B_set = set(B)
    excluded_set = A_set - B_set
    excluded_list = list(excluded_set)
    sampled_values = np.random.choice(excluded_list, size=n, replace=True)
    
    return sampled_values


#%% options
rng = RandomState(seed=None)
capping_T = 0.1
capping_C = 0.1
with_IPS = True
lr = 0.003
embedding_k = 200
num_epochs = 500
top_k_list = [10, 100]
batch_size = 512


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cpu"


# %% Category / Original
data_dir = "/Users/wonhyung64/Github/causal/UnbiasedLearningCausal/data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210"

train_df = pd.read_csv(f"{data_dir}/data_train.csv")
test_df = pd.read_csv(f"{data_dir}/data_test.csv")
valid_df = pd.read_csv(f"{data_dir}/data_vali.csv")

num_items = test_df["idx_item"].max() + 1
num_users = test_df["idx_user"].max() + 1
top_k_list.append(num_items)

df_train = train_df.loc[train_df.loc[:, "outcome"] > 0, :]

bool_cap = np.logical_and(df_train.loc[:, "propensity"] < capping_T, df_train.loc[:, "treated"] == 1)
if np.sum(bool_cap) > 0:
    df_train.loc[bool_cap, "propensity"] = capping_T

bool_cap = np.logical_and(df_train.loc[:, "propensity"] > 1 - capping_C, df_train.loc[:, "treated"] == 0)
if np.sum(bool_cap) > 0:
    df_train.loc[bool_cap, "propensity"] = 1 - capping_C

if with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
    df_train.loc[:, 'ITE'] =  df_train.loc[:, "treated"] * df_train.loc[:, "outcome"]/df_train.loc[:, "propensity"] - \
                                (1 - df_train.loc[:, "treated"]) * df_train.loc[:, "outcome"]/(1 - df_train.loc[:, "propensity"])
else:
    df_train.loc[:, 'ITE'] =  df_train.loc[:, "treated"] * df_train.loc[:, "outcome"]  - \
                                (1 - df_train.loc[:, "treated"]) * df_train.loc[:, "outcome"]


#%% Train
df_train = df_train.sample(frac=1)
x_train = df_train.loc[:, ["idx_user", "idx_item"]].values
y_train = df_train.loc[:, 'ITE'].values

num_sample = len(df_train)
total_batch = num_sample // batch_size

model = MF(num_users, num_items, embedding_k)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

all_items = np.arange(num_items)

for epoch in tqdm(range(1, num_epochs+1)):
    all_idx = np.arange(num_sample)
    np.random.shuffle(all_idx)
    model.train()

    for idx in range(total_batch):

        # mini-batch training
        selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
        sub_x = x_train[selected_idx]
        x_item_neg = sample_excluding(all_items, sub_x[:,1], batch_size)
        sub_x_neg = np.array([sub_x[:,0], x_item_neg]).T
        sub_x = torch.LongTensor(sub_x).to(device)
        sub_x_neg = torch.LongTensor(sub_x_neg).to(device)
        sub_y = y_train[selected_idx]
        sub_y = torch.Tensor(sub_y).unsqueeze(-1).to(device)

        user_embed, item_embed = model(sub_x)
        user_embed, neg_item_embed = model(sub_x)

        diff_rating = nn.Sigmoid()((user_embed * (item_embed - neg_item_embed)).sum(-1)).unsqueeze(-1)
        total_loss = (sub_y * diff_rating).mean()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()



#%%
test_ate = test_df.groupby(["idx_user", "idx_item"])["causal_effect"].mean().reset_index()
users = test_ate["idx_user"].values
items = test_ate["idx_item"].values
true = test_ate["causal_effect"].values
x_test = test_ate[["idx_user", "idx_item"]].values
x_test = torch.LongTensor(x_test).to(device)

model.eval()
user_embed, item_embed = model(x_test)
pred = nn.Sigmoid()((user_embed * item_embed).sum(-1)).detach().cpu().numpy()

user_dcg_list, user_precision_list, user_ar_list = [], [], []
for u in tqdm(range(num_users)):

    user_idx = users==u
    user_pred = (pred[user_idx])
    user_pred = (user_pred - user_pred.min()) / (user_pred.max() - user_pred.min())
    user_true = true[user_idx]

    dcg_k_list, precision_k_list, ar_k_list = [], [], []
    for top_k in top_k_list:

        """ndcg@k"""
        log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
        pred_top_k_rel = user_true[np.argsort(-user_pred)][:top_k]
        dcg_k = (pred_top_k_rel / log2_iplus1).sum()
        dcg_k_list.append(dcg_k)

        """precision@k"""
        p_k = pred_top_k_rel.sum()
        precision_k_list.append(p_k)

        """average rank@k"""
        ar_k = np.sum(-(np.arange(1,top_k+1)) * pred_top_k_rel)
        ar_k_list.append(ar_k)

    user_dcg_list.append(dcg_k_list)
    user_precision_list.append(precision_k_list)
    user_ar_list.append(ar_k_list)

print("CDCG")
print_result(user_dcg_list, top_k_list)

print("CP")
print_result(user_precision_list, top_k_list)

print("CAR")
print_result(user_ar_list, top_k_list)


# %%
