"""
DirectAU: Direct Alignment and Uniformity (KDD 2022)
Key: Explicit optimization of alignment (matched pairs) and uniformity (spread).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAU(nn.Module):
    """DirectAU with alignment and uniformity losses."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, gamma=1.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.gamma = gamma  # Balance between alignment and uniformity
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, edge_index=None):
        return self.user_embedding.weight, self.item_embedding.weight
    
    def alignment_loss(self, user_emb, item_emb):
        """Alignment: matched pairs should be close."""
        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)
        return (user_emb - item_emb).norm(dim=1).pow(2).mean()
    
    def uniformity_loss(self, emb, t=2.0):
        """Uniformity: embeddings should be spread out."""
        emb = F.normalize(emb, dim=1)
        sq_dist = torch.pdist(emb, p=2).pow(2)
        return sq_dist.mul(-t).exp().mean().log()
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index=None):
        """Combined alignment + uniformity loss."""
        user_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        
        # Alignment loss (matched pairs should be close)
        align_loss = self.alignment_loss(user_emb, pos_emb)
        
        # Uniformity loss (all embeddings spread out)
        # Sample subset for efficiency
        sample_size = min(1024, self.n_users, self.n_items)
        user_sample = self.user_embedding.weight[:sample_size]
        item_sample = self.item_embedding.weight[:sample_size]
        
        uniform_user = self.uniformity_loss(user_sample)
        uniform_item = self.uniformity_loss(item_sample)
        uniform_loss = (uniform_user + uniform_item) / 2
        
        total_loss = align_loss + self.gamma * uniform_loss
        
        reg_loss = (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2)) / users.shape[0]
        
        return total_loss, reg_loss
