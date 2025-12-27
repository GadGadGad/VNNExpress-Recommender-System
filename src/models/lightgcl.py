"""
LightGCL: Light Graph Contrastive Learning for Recommendation
Paper: https://arxiv.org/abs/2302.08191
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LightGCL(nn.Module):
    """LightGCL with per-layer SVD propagation and SUM aggregation."""
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        svd_q: int = 20,
        dropout: float = 0.0,
        reg_weight: float = 1e-4,
        ssl_weight: float = 0.1,
        temp: float = 0.2
    ):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.svd_q = svd_q
        self.dropout = dropout
        self.reg_weight = reg_weight
        self.ssl_weight = ssl_weight
        self.temp = temp
        
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, embedding_dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_items, embedding_dim)))
        
        self.u_mul_s = None
        self.v_mul_s = None
        self.ut = None
        self.vt = None
        self.E_u = None
        self.E_i = None
        
    def compute_svd(self, adj_norm: torch.sparse.Tensor):
        """Compute SVD of normalized interaction matrix."""
        svd_q = min(self.svd_q, min(adj_norm.shape) - 1)
        svd_u, s, svd_v = torch.svd_lowrank(adj_norm, q=svd_q)
        
        self.u_mul_s = svd_u @ torch.diag(s)
        self.v_mul_s = svd_v @ torch.diag(s)
        self.ut = svd_u.T
        self.vt = svd_v.T
        
    def to_device(self, device: torch.device):
        """Move model and buffers to device."""
        self.to(device)
        if self.u_mul_s is not None:
            self.u_mul_s = self.u_mul_s.to(device)
            self.v_mul_s = self.v_mul_s.to(device)
            self.ut = self.ut.to(device)
            self.vt = self.vt.to(device)
    
    def _sparse_dropout(self, adj: torch.sparse.Tensor, p: float) -> torch.sparse.Tensor:
        if p == 0 or not self.training:
            return adj
        indices = adj.indices()
        values = adj.values()
        mask = torch.rand(values.size(0), device=values.device) > p
        new_values = values * mask.float() / (1 - p)
        return torch.sparse_coo_tensor(indices, new_values, adj.size()).coalesce()
            
    def forward(self, adj_norm: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with per-layer SVD propagation."""
        E_u_list = [self.E_u_0]
        E_i_list = [self.E_i_0]
        G_u_list = [self.E_u_0]
        G_i_list = [self.E_i_0]
        
        for _ in range(self.n_layers):
            adj = self._sparse_dropout(adj_norm, self.dropout)
            
            Z_u = torch.sparse.mm(adj, E_i_list[-1])
            Z_i = torch.sparse.mm(adj.t(), E_u_list[-1])
            
            vt_ei = self.vt @ E_i_list[-1]
            G_u = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ E_u_list[-1]
            G_i = self.v_mul_s @ ut_eu
            
            E_u_list.append(Z_u)
            E_i_list.append(Z_i)
            G_u_list.append(G_u)
            G_i_list.append(G_i)
        
        E_u = sum(E_u_list)
        E_i = sum(E_i_list)
        G_u = sum(G_u_list)
        G_i = sum(G_i_list)
        
        self.E_u = E_u
        self.E_i = E_i
        
        return E_u, E_i, G_u, G_i
    
    def contrastive_loss(self, E_u: torch.Tensor, E_i: torch.Tensor,
                         G_u: torch.Tensor, G_i: torch.Tensor,
                         users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        pos_score_u = torch.clamp((G_u[users] * E_u[users]).sum(dim=1) / self.temp, -5.0, 5.0).mean()
        neg_score_u = torch.log(torch.exp(G_u[users] @ E_u.T / self.temp).sum(dim=1) + 1e-8).mean()
        
        pos_score_i = torch.clamp((G_i[items] * E_i[items]).sum(dim=1) / self.temp, -5.0, 5.0).mean()
        neg_score_i = torch.log(torch.exp(G_i[items] @ E_i.T / self.temp).sum(dim=1) + 1e-8).mean()
        
        return -pos_score_u + neg_score_u + (-pos_score_i + neg_score_i)
    
    def bpr_loss(self, E_u: torch.Tensor, E_i: torch.Tensor,
                 users: torch.Tensor, pos_items: torch.Tensor, 
                 neg_items: torch.Tensor) -> torch.Tensor:
        """BPR loss."""
        u_emb = E_u[users]
        pos_emb = E_i[pos_items]
        neg_emb = E_i[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        
        return -(pos_scores - neg_scores).sigmoid().log().mean()
    
    def reg_loss(self) -> torch.Tensor:
        """L2 regularization."""
        reg = 0
        for param in self.parameters():
            reg += param.norm(2).square()
        return reg
    
    def calculate_loss(self, adj_norm: torch.sparse.Tensor,
                       users: torch.Tensor, pos_items: torch.Tensor,
                       neg_items: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Calculate total loss = BPR + SSL + Reg."""
        E_u, E_i, G_u, G_i = self.forward(adj_norm)
        
        bpr = self.bpr_loss(E_u, E_i, users, pos_items, neg_items)
        ssl = self.contrastive_loss(E_u, E_i, G_u, G_i, users, pos_items)
        reg = self.reg_loss()
        
        total = bpr + self.ssl_weight * ssl + self.reg_weight * reg
        
        return total, bpr, reg, ssl
    
    def predict(self, adj_norm: torch.sparse.Tensor, 
                users: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict scores for all items."""
        self.eval()
        with torch.no_grad():
            E_u, E_i, _, _ = self.forward(adj_norm)
            if users is not None:
                return E_u[users] @ E_i.T
            return E_u @ E_i.T
    
    def get_embeddings(self, adj_norm: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final embeddings for evaluation."""
        self.eval()
        with torch.no_grad():
            E_u, E_i, _, _ = self.forward(adj_norm)
            return E_u, E_i