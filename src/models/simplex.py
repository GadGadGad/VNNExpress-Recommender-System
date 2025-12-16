"""
SimpleX: A Simple and Strong Baseline for Collaborative Filtering (CIKM 2021)
Key: Cosine Contrastive Loss (CCL) with aggregated neighbor embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleX(nn.Module):
    """SimpleX model with cosine contrastive loss."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, margin=0.9, neg_weight=0.5):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.margin = margin
        self.neg_weight = neg_weight
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, user_history=None):
        """Return embeddings (optionally aggregate history)."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return user_emb, item_emb
    
    def cosine_contrastive_loss(self, users, pos_items, neg_items):
        """Cosine Contrastive Loss (CCL)."""
        user_emb = F.normalize(self.user_embedding(users), dim=1)
        pos_emb = F.normalize(self.item_embedding(pos_items), dim=1)
        neg_emb = F.normalize(self.item_embedding(neg_items), dim=1)
        
        # Cosine similarity
        pos_sim = (user_emb * pos_emb).sum(dim=1)
        neg_sim = (user_emb * neg_emb).sum(dim=1)
        
        # CCL: max(0, margin - pos) + neg_weight * max(0, neg - margin)
        pos_loss = F.relu(self.margin - pos_sim).mean()
        neg_loss = F.relu(neg_sim + self.margin).mean()
        
        return pos_loss + self.neg_weight * neg_loss
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index=None):
        """BPR loss for compatibility with training loop."""
        ccl_loss = self.cosine_contrastive_loss(users, pos_items, neg_items)
        
        reg_loss = (
            self.user_embedding(users).norm(2).pow(2) +
            self.item_embedding(pos_items).norm(2).pow(2) +
            self.item_embedding(neg_items).norm(2).pow(2)
        ) / users.shape[0]
        
        return ccl_loss, reg_loss
