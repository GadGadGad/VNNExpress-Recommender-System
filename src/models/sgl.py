"""
SGL: Self-supervised Graph Learning (SIGIR 2021)
Key: Graph augmentation via edge dropout, node dropout, or random walk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge


class SGLConv(MessagePassing):
    """LightGCN-style convolution for SGL."""
    
    def __init__(self):
        super().__init__(aggr='mean')
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j


class SGL(nn.Module):
    """Self-supervised Graph Learning with contrastive augmentation."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3,
                 dropout=0.1, ssl_weight=0.1, temp=0.2, aug_type='ed'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.dropout = dropout
        self.ssl_weight = ssl_weight
        self.temp = temp
        self.aug_type = aug_type  # 'ed' (edge dropout), 'nd' (node dropout), 'rw' (random walk)
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self.convs = nn.ModuleList([SGLConv() for _ in range(n_layers)])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
    
    def graph_augment(self, edge_index, p=None):
        """Augment graph via edge dropout."""
        if p is None:
            p = self.dropout
        if self.aug_type == 'ed':
            aug_edge_index, _ = dropout_edge(edge_index, p=p, training=self.training)
            return aug_edge_index
        else:
            return edge_index
    
    def forward(self, edge_index):
        x = self.get_ego_embeddings()
        all_embs = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embs.append(x)
        
        all_embs = torch.stack(all_embs, dim=1)
        final_emb = torch.mean(all_embs, dim=1)
        
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def forward_augmented(self, edge_index):
        """Forward pass with augmented graph."""
        aug_edge_index = self.graph_augment(edge_index)
        return self.forward(aug_edge_index)
    
    def ssl_loss(self, user_emb1, user_emb2, item_emb1, item_emb2, users, items):
        """InfoNCE contrastive loss between two augmented views."""
        user_emb1 = F.normalize(user_emb1[users], dim=1)
        user_emb2 = F.normalize(user_emb2[users], dim=1)
        item_emb1 = F.normalize(item_emb1[items], dim=1)
        item_emb2 = F.normalize(item_emb2[items], dim=1)
        
        # User contrastive loss
        pos_user = (user_emb1 * user_emb2).sum(dim=1) / self.temp
        neg_user = torch.mm(user_emb1, user_emb2.t()) / self.temp
        user_loss = -pos_user + torch.logsumexp(neg_user, dim=1)
        
        # Item contrastive loss
        pos_item = (item_emb1 * item_emb2).sum(dim=1) / self.temp
        neg_item = torch.mm(item_emb1, item_emb2.t()) / self.temp
        item_loss = -pos_item + torch.logsumexp(neg_item, dim=1)
        
        return (user_loss.mean() + item_loss.mean()) / 2
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        """BPR loss with SSL regularization."""
        # Main view
        user_emb, item_emb = self.forward(edge_index)
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Augmented views for SSL
        user_emb1, item_emb1 = self.forward_augmented(edge_index)
        user_emb2, item_emb2 = self.forward_augmented(edge_index)
        
        ssl = self.ssl_loss(user_emb1, user_emb2, item_emb1, item_emb2, users, pos_items)
        
        # Regularization
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return bpr + self.ssl_weight * ssl, reg
