"""
NGCF: Neural Graph Collaborative Filtering (WWW 2019)
Key: Feature transformation at each GNN layer (unlike LightGCN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class NGCFConv(MessagePassing):
    """Single NGCF propagation layer."""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_i, x_j, norm):
        # NGCF message: W1*x_j + W2*(x_i * x_j)
        msg = self.W1(x_j) + self.W2(x_i * x_j)
        return norm.view(-1, 1) * msg
    
    def update(self, aggr_out):
        return self.dropout(F.leaky_relu(aggr_out))


class NGCF(nn.Module):
    """Neural Graph Collaborative Filtering model."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.convs = nn.ModuleList([
            NGCFConv(embedding_dim, embedding_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        all_embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embs.append(x)
        
        # Concatenate all layer outputs
        out = torch.cat(all_embs, dim=1)
        
        user_emb = out[:self.n_users]
        item_emb = out[self.n_users:]
        
        return user_emb, item_emb
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        user_emb, item_emb = self.forward(edge_index)
        
        u = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]
        
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / users.shape[0]
        
        return bpr_loss, reg_loss
