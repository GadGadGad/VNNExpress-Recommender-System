"""
Multi-Aspect Heterogeneous GNN (MA-HGN)
Explicitly models Behavioral (user-item) and Social (user-user) aspects 
using a Heterogeneous Attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GCNConv, TransformerConv
from .base_gcl import BaseGCL

class MAHGN(nn.Module):
    """
    MA-HGN: Multi-Aspect Heterogeneous GNN.
    Models 'Social' (user-user) and 'Behavioral' (user-article) signals 
    with different semantic weights.
    """
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, dropout=0.2, 
                 n_categories=0, gnn_type='gat', cl_weight=0.0, temp=0.2, 
                 pretrained_item_emb=None): # <--- Thêm tham số này
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type.lower()
        self.cl_weight = 0.1
        self.temp = 0.2
        self.eps = 0.1
        
        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        if n_categories > 0:
            self.category_emb = nn.Embedding(n_categories, embedding_dim)
        
        # Heterogeneous Convolution layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            # Dynamic GNN selection
            def get_gnn_layer(in_channels, out_channels):
                if self.gnn_type == 'gat':
                    return GATConv((in_channels, in_channels), out_channels // 4, heads=4, add_self_loops=False)
                elif self.gnn_type == 'transformer':
                    return TransformerConv((in_channels, in_channels), out_channels // 4, heads=4)
                elif self.gnn_type == 'gcn':
                    return GCNConv(in_channels, out_channels, add_self_loops=False)
                elif self.gnn_type == 'sage':
                    return SAGEConv(in_channels, out_channels)
                else:
                    return GATConv((in_channels, in_channels), out_channels // 4, heads=4, add_self_loops=False)

            # Explicitly define convs for each relationship aspect
            conv_dict = {
                ('user', 'comments', 'article'): get_gnn_layer(embedding_dim, embedding_dim),
                ('article', 'rev_comments', 'user'): get_gnn_layer(embedding_dim, embedding_dim),
                ('user', 'replied_to', 'user'): get_gnn_layer(embedding_dim, embedding_dim),
                ('user', 'interacts_with', 'user'): get_gnn_layer(embedding_dim, embedding_dim),
                ('article', 'belongs_to', 'category'): get_gnn_layer(embedding_dim, embedding_dim),
                ('category', 'has_article', 'article'): get_gnn_layer(embedding_dim, embedding_dim),
                ('user', 'interested_in', 'category'): get_gnn_layer(embedding_dim, embedding_dim),
                ('category', 'attracts', 'user'): get_gnn_layer(embedding_dim, embedding_dim)
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
        self.dropout = nn.Dropout(dropout)
        
        # Fusion Attention: Learning relative importance of aspects
        self.aspect_attention = nn.Parameter(torch.ones(2)) 
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        if hasattr(self, 'category_emb'):
            nn.init.normal_(self.category_emb.weight, std=0.1)

    def forward(self, x_dict, edge_index_dict, perturbed=False):
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
        
        # Perturbation for SimGCL
        if perturbed:
            x_dict = {k: v + torch.sign(v) * F.normalize(torch.randn_like(v), dim=1) * self.eps 
                      for k, v in x_dict.items()}
            
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
            # Store original embeddings to prevent KeyError if some node types don't receive messages
            h_dict_prev = {k: v.clone() for k, v in h_dict.items()}
            
            # Message Passing
            h_dict_new = conv(h_dict, edge_index_dict_gpu)
            
            # Update: keep previous embedding if no new message received
            h_dict = {k: h_dict_new.get(k, h_dict_prev[k]) for k in h_dict_prev.keys()}
            
            # Activation & Dropout
            h_dict = {k: F.leaky_relu(v) for k, v in h_dict.items()}
            h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
            
            # Diagnostic: ensure user/article always exist
            if 'user' not in h_dict: h_dict['user'] = h_dict_prev['user']
            if 'article' not in h_dict: h_dict['article'] = h_dict_prev['article']
            
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
        
        # Contrastive Learning Loss (SimGCL)
        cl_loss = torch.tensor(0.0, device=mf_loss.device)
        if self.cl_weight > 0:
            user_view1, item_view1 = self.forward(None, edge_index_dict, perturbed=True)
            user_view2, item_view2 = self.forward(None, edge_index_dict, perturbed=True)
            cl_loss = self.cal_cl_loss([users], user_view1, user_view2) + \
                      self.cal_cl_loss([pos_items], item_view1, item_view2)
            cl_loss *= self.cl_weight

        return mf_loss + 1e-4 * reg_loss + cl_loss, mf_loss, cl_loss, reg_loss

    def cal_cl_loss(self, idx, view1, view2):
        """SimGCL InfoNCE Loss"""
        u_idx = torch.unique(torch.cat(idx))
        view1_norm = F.normalize(view1[u_idx], dim=1)
        view2_norm = F.normalize(view2[u_idx], dim=1)
        
        pos_score = (view1_norm * view2_norm).sum(dim=1) / self.temp
        exp_pos = torch.exp(pos_score)
        
        # Approximate denominator with batch negatives for efficiency
        # Or full negatives if memory allows (SimGCL uses full)
        # Follows standard SimGCL implementation: full matrix multiplication
        # Batching might be needed for huge graphs 
        # Using the standard SimGCL formula: -log( exp(sim(u,u')) / sum(exp(sim(u, v'))) )
        
        # Simplified implementation for speed (Batch-wise contrastive)
        ttl_score = torch.matmul(view1_norm, view2_norm.transpose(0, 1)) / self.temp
        exp_ttl = torch.exp(ttl_score).sum(dim=1)
        
        cl_loss = -torch.log(exp_pos / exp_ttl).mean()
        return cl_loss

    def predict(self, users, items, edge_index_dict):
        """Prediction for inference."""
        user_final, item_final = self.forward(None, edge_index_dict)
        return (user_final[users] * item_final[items]).sum(dim=1)
