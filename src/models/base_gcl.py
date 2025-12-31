"""
BaseGCL - Base class for Graph Contrastive Learning models

Shared functionality:
- LightGCN-style propagation
- BPR loss
- Regularization
- Prediction

Subclasses (SimGCL, XSimGCL, LightGCL) implement their own:
- Contrastive augmentation strategy
- SSL loss computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BaseGCL(nn.Module):
    """
    Base class for Graph Contrastive Learning models.
    
    Provides:
    - Embedding initialization
    - LightGCN-style forward propagation
    - BPR loss
    - L2 regularization
    - Prediction method
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
        reg_weight: float = 1e-4,
        ssl_weight: float = 0.2,
        temp: float = 0.2
    ):
        super(BaseGCL, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.reg_weight = reg_weight
        self.ssl_weight = ssl_weight
        self.temp = temp
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def _propagate(self, adj_norm: torch.sparse.Tensor, 
                   all_embeddings: torch.Tensor,
                   perturb: bool = False) -> torch.Tensor:
        """
        LightGCN-style graph propagation.
        Override this method to add custom augmentation.
        """
        embeddings_list = [all_embeddings]
        
        for layer in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)
            
            # Subclasses can override _perturb_layer() for custom augmentation
            if perturb:
                all_embeddings = self._perturb_layer(all_embeddings, layer)
            
            if self.dropout > 0 and self.training:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout)
                
            embeddings_list.append(all_embeddings)
        
        # Layer combination (mean pooling)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        return all_embeddings
    
    def _perturb_layer(self, embeddings: torch.Tensor, layer: int) -> torch.Tensor:
        """Override in subclasses for custom perturbation strategy."""
        return embeddings
    
    def forward(self, adj_norm: torch.sparse.Tensor, 
                perturb: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        all_embeddings = self._propagate(adj_norm, all_embeddings, perturb)
        
        user_emb = all_embeddings[:self.n_users]
        item_emb = all_embeddings[self.n_users:]
        
        return user_emb, item_emb
    
    def bpr_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                 users: torch.Tensor, pos_items: torch.Tensor, 
                 neg_items: torch.Tensor) -> torch.Tensor:
        """BPR (Bayesian Personalized Ranking) Loss."""
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss
    
    def reg_loss(self, users: torch.Tensor, pos_items: torch.Tensor,
                 neg_items: torch.Tensor) -> torch.Tensor:
        """L2 Regularization on initial embeddings."""
        user_e = self.user_embedding.weight[users]
        pos_e = self.item_embedding.weight[pos_items]
        neg_e = self.item_embedding.weight[neg_items]
        
        reg = (user_e.norm(2).pow(2) + 
               pos_e.norm(2).pow(2) + 
               neg_e.norm(2).pow(2)) / (2 * len(users))
        
        return reg
    
    def contrastive_loss(self, 
                         user_emb1: torch.Tensor, item_emb1: torch.Tensor,
                         user_emb2: torch.Tensor, item_emb2: torch.Tensor,
                         users: torch.Tensor, pos_items: torch.Tensor) -> torch.Tensor:
        """InfoNCE Contrastive Loss between two perturbed views."""
        user_e1 = F.normalize(user_emb1[users], dim=1)
        user_e2 = F.normalize(user_emb2[users], dim=1)
        item_e1 = F.normalize(item_emb1[pos_items], dim=1)
        item_e2 = F.normalize(item_emb2[pos_items], dim=1)
        
        # User contrastive loss
        pos_score_user = torch.sum(user_e1 * user_e2, dim=1) / self.temp
        neg_score_user = user_e1 @ user_e2.T / self.temp
        ssl_loss_user = -torch.mean(
            pos_score_user - torch.logsumexp(neg_score_user, dim=1)
        )
        
        # Item contrastive loss
        pos_score_item = torch.sum(item_e1 * item_e2, dim=1) / self.temp
        neg_score_item = item_e1 @ item_e2.T / self.temp
        ssl_loss_item = -torch.mean(
            pos_score_item - torch.logsumexp(neg_score_item, dim=1)
        )
        
        return ssl_loss_user + ssl_loss_item
    
    def calculate_loss(self, adj_norm: torch.sparse.Tensor,
                       users: torch.Tensor, pos_items: torch.Tensor,
                       neg_items: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Calculate total loss = BPR + Reg + SSL."""
        # Forward without perturbation (for BPR)
        user_emb, item_emb = self.forward(adj_norm, perturb=False)
        
        # BPR loss
        bpr = self.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
        
        # Regularization
        reg = self.reg_loss(users, pos_items, neg_items)
        
        # Contrastive loss with two perturbed views
        user_emb1, item_emb1 = self.forward(adj_norm, perturb=True)
        user_emb2, item_emb2 = self.forward(adj_norm, perturb=True)
        ssl = self.contrastive_loss(user_emb1, item_emb1, user_emb2, item_emb2,
                                    users, pos_items)
        
        # Total
        total = bpr + self.reg_weight * reg + self.ssl_weight * ssl
        
        return total, bpr, reg, ssl
    
    def predict(self, adj_norm: torch.sparse.Tensor, 
                users: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict scores for all items."""
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_norm, perturb=False)
            
            if users is not None:
                user_e = user_emb[users]
            else:
                user_e = user_emb
                
            scores = user_e @ item_emb.T
            
        return scores
    
    def get_embeddings(self, adj_norm: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final embeddings."""
        self.eval()
        with torch.no_grad():
            return self.forward(adj_norm, perturb=False)
