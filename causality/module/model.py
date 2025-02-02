import torch
import torch.nn as nn


class MF(nn.Module):
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


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.linear1 = nn.Linear(self.embedding_k*2, self.embedding_k)
        self.linear2 = nn.Linear(self.embedding_k, 1, bias=False)

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        h1 = nn.ReLU()(self.linear1(z_embed))
        out = self.linear2(h1)
        return out, user_embed, item_embed


class LinearCF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(LinearCF, self).__init__()
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


class SharedNCF(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(SharedNCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.ctr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )
        self.cvr = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        ctr = self.ctr(z_embed)
        cvr = self.cvr(z_embed)
        ctcvr = torch.mul(nn.Sigmoid()(ctr), nn.Sigmoid()(cvr))
        return cvr, ctr, ctcvr
