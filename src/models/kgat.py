"""
KGAT: Knowledge Graph Attention Network (KDD 2019)
Key: Attention-based aggregation over knowledge graph (user-item + item-category).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class KGATConv(MessagePassing):
    """Knowledge-aware attention convolution."""
    
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__(aggr='add')
        self.heads = heads
        self.head_dim = out_dim // heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(1, heads, self.head_dim * 2))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, edge_index):
        x = self.W(x)
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j, index, size_i):
        # Reshape for multi-head attention
        x_i = x_i.view(-1, self.heads, self.head_dim)
        x_j = x_j.view(-1, self.heads, self.head_dim)
        
        # Attention scores
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.a).sum(dim=-1)
        alpha = self.leaky_relu(alpha)
        alpha = F.softmax(alpha, dim=0)
        
        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.head_dim)


class KGAT(nn.Module):
    """Knowledge Graph Attention Network for recommendation."""
    
    def __init__(self, n_users, n_items, n_categories=15, embedding_dim=64, 
                 n_layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        
        self.convs = nn.ModuleList([
            KGATConv(embedding_dim, embedding_dim, heads) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
    
    def get_all_embeddings(self):
        return torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight,
            self.category_embedding.weight
        ], dim=0)
    
    def forward(self, edge_index):
        x = self.get_all_embeddings()
        all_embs = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            all_embs.append(x)
        
        all_embs = torch.stack(all_embs, dim=1)
        final_emb = torch.mean(all_embs, dim=1)
        
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:self.n_users + self.n_items]
        
        return user_emb, item_emb
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        """BPR loss with attention-based propagation."""
        user_emb, item_emb = self.forward(edge_index)
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return bpr, reg
