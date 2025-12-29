from .base_gcl import BaseGCL
import torch
from typing import Tuple

class LightGCN(BaseGCL):
    """
    LightGCN: Simplified GCN for Recommendation
    Paper: https://arxiv.org/abs/2002.02126
    
    Inherits from BaseGCL but removes Contrastive Learning (SSL) overhead.
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 dropout=0.0, reg_weight=1e-4):
        # Initialize BaseGCL with ssl_weight=0 and temp=1 (dummy)
        super().__init__(
            n_users=n_users, 
            n_items=n_items, 
            embedding_dim=embedding_dim, 
            n_layers=n_layers, 
            dropout=dropout, 
            reg_weight=reg_weight,
            ssl_weight=0.0,
            temp=1.0
        )
        
    def calculate_loss(self, adj_norm: torch.sparse.Tensor,
                       users: torch.Tensor, pos_items: torch.Tensor,
                       neg_items: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Calculate BPR + Reg only (No SSL).
        """
        # Forward pass (no perturbation)
        user_emb, item_emb = self.forward(adj_norm, perturb=False)
        
        # BPR loss
        bpr = self.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
        
        # Regularization
        reg = self.reg_loss(users, pos_items, neg_items)
        
        # Total
        total = bpr + self.reg_weight * reg
        
        # Return 0 for ssl loss
        return total, bpr, reg, torch.tensor(0.0, device=total.device)
