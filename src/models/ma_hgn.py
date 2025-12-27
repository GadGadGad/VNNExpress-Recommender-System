"""
Multi-Aspect Heterogeneous GNN (MA-HGN)
Explicitly models Behavioral (user-item) and Social (user-user) aspects 
using a Heterogeneous Attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from .base_gcl import BaseGCL

class MAHGN(nn.Module):
    """
    MA-HGN: Multi-Aspect Heterogeneous GNN.
    Models 'Social' (user-user) and 'Behavioral' (user-article) signals 
    with different semantic weights.
    """
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, dropout=0.2, n_categories=0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        if n_categories > 0:
            self.category_emb = nn.Embedding(n_categories, embedding_dim)
        
        # Heterogeneous Convolution layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            # We explicitly define convs for each relationship aspect
            conv_dict = {
                ('user', 'comments', 'article'): GATConv((embedding_dim, embedding_dim), embedding_dim // 4, heads=4, add_self_loops=False),
                ('article', 'rev_comments', 'user'): GATConv((embedding_dim, embedding_dim), embedding_dim // 4, heads=4, add_self_loops=False),
                ('user', 'replied_to', 'user'): SAGEConv(embedding_dim, embedding_dim),
                ('user', 'interacts_with', 'user'): SAGEConv(embedding_dim, embedding_dim),
                ('article', 'belongs_to', 'category'): SAGEConv(embedding_dim, embedding_dim),
                ('category', 'has_article', 'article'): SAGEConv(embedding_dim, embedding_dim),
                ('user', 'interested_in', 'category'): SAGEConv(embedding_dim, embedding_dim),
                ('category', 'attracts', 'user'): SAGEConv(embedding_dim, embedding_dim)
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
        self.dropout = nn.Dropout(dropout)
        
        # Fusion Attention: Learning relative importance of aspects
        # 1: User-Article flow, 2: User-User social flow
        self.aspect_attention = nn.Parameter(torch.ones(2)) 
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        if hasattr(self, 'category_emb'):
            nn.init.normal_(self.category_emb.weight, std=0.1)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with multi-aspect feature aggregation.
        """
        if x_dict is None:
            x_dict = {
                'user': self.user_emb.weight,
                'article': self.item_emb.weight
            }
            if self.n_categories > 0:
                x_dict['category'] = self.category_emb.weight
            
        h_dict = x_dict
        all_user_embs = [h_dict['user']]
        all_item_embs = [h_dict['article']]
        
        for conv in self.convs:
            # Ensure edge_index_dict is on the same device as embeddings
            device = h_dict['user'].device
            edge_index_dict_gpu = {
                key: ei.to(device) if hasattr(ei, 'to') else ei 
                for key, ei in edge_index_dict.items()
            }
            # Message Passing
            h_dict = conv(h_dict, edge_index_dict_gpu)
            
            # Activation & Dropout
            h_dict = {k: F.leaky_relu(v) for k, v in h_dict.items()}
            h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
            
            all_user_embs.append(h_dict['user'])
            all_item_embs.append(h_dict['article'])
            
        # Final embedding: average across layers (LightGCN style)
        user_final = torch.stack(all_user_embs, dim=1).mean(dim=1)
        item_final = torch.stack(all_item_embs, dim=1).mean(dim=1)
        
        return user_final, item_final

    def calculate_loss(self, edge_index_dict, users, pos_items, neg_items):
        """Standard Bayesian Personalized Ranking loss."""
        user_final, item_final = self.forward(None, edge_index_dict)
        
        u_emb = user_final[users]
        pos_emb = item_final[pos_items]
        neg_emb = item_final[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 Regularization
        reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return mf_loss + 1e-4 * reg_loss, mf_loss, torch.tensor(0.0).to(mf_loss.device), reg_loss

    def predict(self, users, items, edge_index_dict):
        """Prediction for inference."""
        user_final, item_final = self.forward(None, edge_index_dict)
        return (user_final[users] * item_final[items]).sum(dim=1)
