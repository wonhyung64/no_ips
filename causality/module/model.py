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


class NCFPlus(nn.Module):
    def __init__(self, num_users, num_items, embedding_k):
        super(NCFPlus, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.y1 = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )
        self.y0 = nn.Sequential(
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
        y1_out = self.y1(z_embed)
        y0_out = self.y0(z_embed)
        return y1_out, y0_out


class LinearCFPlus(nn.Module):
    def __init__(self, num_users, num_items, embedding_k):
        super(LinearCFPlus, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_k)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_k)
        self.y1 = nn.Sequential(
            nn.Linear(self.embedding_k*2, 1, bias=False),
        )
        self.y0 = nn.Sequential(
            nn.Linear(self.embedding_k*2, 1, bias=False),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        y1_out = self.y1(z_embed)
        y0_out = self.y0(z_embed)
        return y1_out, y0_out


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


class SharedNCFPlus(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(SharedNCFPlus, self).__init__()
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
        self.y1 = nn.Sequential(
            nn.Linear(self.embedding_k*2, self.embedding_k),
            nn.ReLU(),
            nn.Linear(self.embedding_k, 1, bias=False),
        )
        self.y0 = nn.Sequential(
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
        y1 = self.y1(z_embed)
        y0 = self.y0(z_embed)
        return y1, y0, ctr


class SharedLinearCFPlus(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k):
        super(SharedLinearCFPlus, self).__init__()
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
        self.y1 = nn.Sequential(
            nn.Linear(self.embedding_k*2, 1, bias=False),
        )
        self.y0 = nn.Sequential(
            nn.Linear(self.embedding_k*2, 1, bias=False),
        )

    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        user_embed = self.user_embedding(user_idx)
        item_embed = self.item_embedding(item_idx)
        z_embed = torch.cat([user_embed, item_embed], axis=1)
        ctr = self.ctr(z_embed)
        y1 = self.y1(z_embed)
        y0 = self.y0(z_embed)
        return y1, y0, ctr


class NCF_AKBIPS_ExpPlus(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = NCFPlus(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = nn.Parameter(torch.rand(1,4096)) 

    def get_embedding(self,x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]
        return feature

    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)


class LinearCF_AKBIPS_ExpPlus(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = nn.Embedding(self.num_users, self.embedding_k)
        self.H = nn.Embedding(self.num_items, self.embedding_k)
        self.prediction_model = LinearCFPlus(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.weight_model = MF(
            num_users = self.num_users, num_items = self.num_items,embedding_k=self.embedding_k, *args, **kwargs)
        self.epsilon = nn.Parameter(torch.rand(1,4096)) 

    def get_embedding(self,x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)
        feature = torch.cat([U_emb ,V_emb ],dim=1)  
        f_min = torch.min(feature)
        f_max = torch.max(feature)
        feature = feature - f_min / (f_max - f_min)                
        feature = feature/feature.shape[1]
        return feature

    def exp_kernel(self,X,Y,gamma = 0.1):
        Euclidean_distances = abs(torch.cdist(Y,X))
        return torch.exp(-Euclidean_distances * gamma)

