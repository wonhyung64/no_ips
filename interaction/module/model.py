import torch
import torch.nn as nn


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

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        out = torch.sum(user_embed.mul(item_embed), 1).unsqueeze(-1)

        return out, user_embed, item_embed


class IpsV2(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.prediction_model = MF(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)       
        self.propensity_model = NCF1(
            num_users = self.num_users, num_items = self.num_items, embedding_k=self.embedding_k, *args, **kwargs)


class NCF1(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(NCF1, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = nn.Linear(self.embedding_k*2, 1, bias=True)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out = self.linear_1(z_embed)

        return out, user_embed, item_embed


class ESMM(nn.Module):
    """ESMM"""
    def __init__(self, num_users, num_items, embedding_k):
        super(ESMM, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.ctr = nn.Sequential(
            nn.Linear(in_features=self.embedding_k*2, out_features=360),
            nn.ReLU(),
            nn.Linear(in_features=360, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1),
        )
        self.cvr = nn.Sequential(
            nn.Linear(in_features=self.embedding_k*2, out_features=360),
            nn.ReLU(),
            nn.Linear(in_features=360, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out_cvr = self.cvr(z_embed)
        out_ctr = self.ctr(z_embed)
        out_ctcvr = torch.mul(nn.Sigmoid()(out_ctr), nn.Sigmoid()(out_cvr))

        return out_cvr, out_ctr, out_ctcvr


class MultiIps(nn.Module):
    """Multi-task IPS"""
    def __init__(self, num_users, num_items, embedding_k):
        super(MultiIps, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.ctr = nn.Sequential(
            nn.Linear(in_features=self.embedding_k*2, out_features=360),
            nn.ReLU(),
            nn.Linear(in_features=360, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1),
        )
        self.cvr = nn.Sequential(
            nn.Linear(in_features=self.embedding_k*2, out_features=360),
            nn.ReLU(),
            nn.Linear(in_features=360, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=2),
            nn.ReLU(),
            nn.Linear(in_features=2, out_features=1),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        out_cvr = self.cvr(z_embed)
        out_ctr = self.ctr(z_embed)

        return out_cvr, out_ctr
