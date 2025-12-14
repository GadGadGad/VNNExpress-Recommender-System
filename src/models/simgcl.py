"""
SimGCL: Simple Graph Contrastive Learning (SIGIR 2022)
Key: Uses random noise instead of graph augmentation for contrastive views.
Simpler and more effective than SGL.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class SimGCLConv(MessagePassing):
    """LightGCN-style convolution."""
    
    def __init__(self):
        super().__init__(aggr='mean')
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j


class SimGCL(nn.Module):
    """Simple Graph Contrastive Learning with noise-based augmentation."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3,
                 eps=0.1, ssl_weight=0.5, temp=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.eps = eps  # Noise magnitude
        self.ssl_weight = ssl_weight
        self.temp = temp
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.convs = nn.ModuleList([SimGCLConv() for _ in range(n_layers)])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
    
    def add_noise(self, emb):
        """Add random uniform noise for contrastive augmentation."""
        noise = torch.rand_like(emb).sign() * F.normalize(torch.rand_like(emb), dim=1) * self.eps
        return emb + noise
    
    def forward(self, edge_index, perturb=False):
        x = self.get_ego_embeddings()
        all_embs = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            if perturb:
                x = self.add_noise(x)
            all_embs.append(x)
        
        all_embs = torch.stack(all_embs, dim=1)
        final_emb = torch.mean(all_embs, dim=1)
        
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def ssl_loss(self, emb1, emb2, nodes):
        """InfoNCE contrastive loss."""
        emb1 = F.normalize(emb1[nodes], dim=1)
        emb2 = F.normalize(emb2[nodes], dim=1)
        
        pos = (emb1 * emb2).sum(dim=1) / self.temp
        neg = torch.mm(emb1, emb2.t()) / self.temp
        
        return (-pos + torch.logsumexp(neg, dim=1)).mean()
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        """BPR loss with noise-based SSL."""
        # Main view (no noise)
        user_emb, item_emb = self.forward(edge_index, perturb=False)
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Perturbed views for SSL
        user_emb1, item_emb1 = self.forward(edge_index, perturb=True)
        user_emb2, item_emb2 = self.forward(edge_index, perturb=True)
        
        ssl_user = self.ssl_loss(user_emb1, user_emb2, users)
        ssl_item = self.ssl_loss(item_emb1, item_emb2, pos_items)
        ssl = (ssl_user + ssl_item) / 2
        
        # Regularization
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return bpr + self.ssl_weight * ssl, reg
