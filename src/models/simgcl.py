"""
SimGCL - Simple Graph Contrastive Learning for Recommendation

Paper: "Are Graph Augmentations Necessary? Simple Graph Contrastive 
        Learning for Recommendation" (SIGIR 2022)

Key idea: Use random noise perturbation instead of graph augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .base_gcl import BaseGCL


class SimGCL(BaseGCL):
    """
    SimGCL Model - inherits from BaseGCL
    
    Difference from base:
    - Adds uniform noise to embeddings for contrastive learning
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        eps: float = 0.1,           # Noise magnitude
        dropout: float = 0.0,
        reg_weight: float = 1e-4,
        ssl_weight: float = 0.2,
        temp: float = 0.2
    ):
        super(SimGCL, self).__init__(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            dropout=dropout,
            reg_weight=reg_weight,
            ssl_weight=ssl_weight,
            temp=temp
        )
        
        self.eps = eps
        
        print(f"\nSimGCL Model initialized:")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Layers: {n_layers}, Noise eps: {eps}")
        print(f"  SSL weight: {ssl_weight}, Temp: {temp}")
    
    def _perturb_layer(self, embeddings: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Add uniform noise to embeddings for contrastive augmentation
        
        e' = e + eps * sign(e) * random_noise
        
        where random_noise ~ Uniform(0, 1)
        """
        noise = torch.rand_like(embeddings)
        noise = F.normalize(noise, dim=-1)
        return embeddings + torch.sign(embeddings) * self.eps * noise
