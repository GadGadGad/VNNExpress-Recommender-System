"""
HetGNN: Heterogeneous Graph Neural Network (KDD 2019)
Key: Handles heterogeneous graphs with different node/edge types.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv


class HetGNN(nn.Module):
    """Heterogeneous Graph Neural Network for recommendation."""
    
    def __init__(self, n_users, n_items, n_categories=15, embedding_dim=64,
                 n_layers=2, heads=4, dropout=0.1, aggregation='attention'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.n_layers = n_layers
        self.aggregation = aggregation
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        
        # Heterogeneous convolutions for each edge type
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {
                ('user', 'interacts', 'item'): SAGEConv(embedding_dim, embedding_dim),
                ('item', 'rev_interacts', 'user'): SAGEConv(embedding_dim, embedding_dim),
                ('item', 'belongs_to', 'category'): SAGEConv(embedding_dim, embedding_dim),
                ('category', 'rev_belongs_to', 'item'): SAGEConv(embedding_dim, embedding_dim),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        self.dropout = nn.Dropout(dropout)
        
        # Type-specific transformation
        self.type_transform = nn.ModuleDict({
            'user': nn.Linear(embedding_dim, embedding_dim),
            'item': nn.Linear(embedding_dim, embedding_dim),
            'category': nn.Linear(embedding_dim, embedding_dim),
        })
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
    
    def forward(self, x_dict, edge_index_dict):
        """Forward pass with heterogeneous message passing."""
        # Initialize embeddings if not provided
        if x_dict is None:
            x_dict = {
                'user': self.user_embedding.weight,
                'item': self.item_embedding.weight,
                'category': self.category_embedding.weight,
            }
        
        # Handle raw tensor input (fallback from bipartite graph)
        if isinstance(edge_index_dict, torch.Tensor):
            device = x_dict['user'].device
            edge_index_dict = {
                ('user', 'interacts', 'item'): edge_index_dict.to(device),
                ('item', 'rev_interacts', 'user'): torch.stack([edge_index_dict[1], edge_index_dict[0]]).to(device),
                # Empty category edges (required by HeteroConv but won't be used)
                ('item', 'belongs_to', 'category'): torch.zeros((2, 0), dtype=torch.long, device=device),
                ('category', 'rev_belongs_to', 'item'): torch.zeros((2, 0), dtype=torch.long, device=device),
            }
        
        all_embs = {k: [v] for k, v in x_dict.items()}
        
        for conv in self.convs:
            # Store original embeddings to prevent KeyError
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}
            
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Update: keep previous embedding if no new message received
            x_dict = {k: x_dict_new.get(k, x_dict_prev[k]) for k in x_dict_prev.keys()}
            
            x_dict = {k: F.relu(self.dropout(v)) for k, v in x_dict.items()}
            for k, v in x_dict.items():
                all_embs[k].append(v)
        
        # Mean pooling across layers
        final_embs = {}
        for k, embs in all_embs.items():
            stacked = torch.stack(embs, dim=1)
            final_embs[k] = torch.mean(stacked, dim=1)
            final_embs[k] = self.type_transform[k](final_embs[k])
        
        return final_embs['user'], final_embs['item']
    
    def bpr_loss(self, users, pos_items, neg_items, x_dict=None, edge_index_dict=None):
        """BPR loss for heterogeneous graph."""
        user_emb, item_emb = self.forward(x_dict, edge_index_dict)
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return bpr, reg
    
    def predict(self, users, items, x_dict=None, edge_index_dict=None):
        """Predict scores for user-item pairs."""
        user_emb, item_emb = self.forward(x_dict, edge_index_dict)
        return (user_emb[users] * item_emb[items]).sum(dim=1)
