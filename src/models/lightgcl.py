"""
LightGCL: Light Graph Contrastive Learning for Recommendation
==============================================================

Paper: https://arxiv.org/abs/2302.08191
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from typing import Tuple, Optional


class LightGCL(nn.Module):
    """
    LightGCL Model
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        svd_q: int = 5,
        dropout: float = 0.0,
        reg_weight: float = 1e-4,
        ssl_weight: float = 0.1,
        temp: float = 0.2
    ):
        super(LightGCL, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.svd_q = svd_q
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
        
        # SVD components (set by compute_svd)
        self.u_mul_s = None
        self.v_mul_s = None
        self.ut = None
        self.vt = None
        
        print(f"\nLightGCL Model initialized:")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Layers: {n_layers}, SVD-q: {svd_q}")
        print(f"  SSL weight: {ssl_weight}, Temp: {temp}")
        
    def compute_svd(self, interaction_matrix: sp.spmatrix):
        """
        Compute SVD of interaction matrix for contrastive augmentation
        """
        print("Computing SVD for contrastive augmentation...")
        
        # Convert to dense array
        if sp.issparse(interaction_matrix):
            R = interaction_matrix.toarray().astype(np.float32)
        else:
            R = np.array(interaction_matrix, dtype=np.float32)
            
        # Truncated SVD using scipy.sparse.linalg.svds
        try:
            from scipy.sparse.linalg import svds
            R_sparse = sp.csr_matrix(R)
            k = min(self.svd_q, min(R.shape) - 1)
            U, S, Vt = svds(R_sparse, k=k)
            
            # Sort by singular values (descending) - svds returns in ascending order
            idx = np.argsort(S)[::-1]
            
            # Create contiguous copies to avoid negative stride issues
            U = np.array(U[:, idx], order='C', dtype=np.float32)
            S = np.array(S[idx], order='C', dtype=np.float32)
            Vt = np.array(Vt[idx, :], order='C', dtype=np.float32)
            
        except Exception as e:
            print(f"  Warning: sparse SVD failed ({e}), using dense SVD")
            U, S, Vt = np.linalg.svd(R, full_matrices=False)
            
            # Truncate and make contiguous
            U = np.array(U[:, :self.svd_q], order='C', dtype=np.float32)
            S = np.array(S[:self.svd_q], order='C', dtype=np.float32)
            Vt = np.array(Vt[:self.svd_q, :], order='C', dtype=np.float32)
        
        # Compute components for contrastive learning
        S_sqrt = np.sqrt(S)
        
        # Create contiguous arrays explicitly
        u_mul_s = np.array(U * S_sqrt, order='C', dtype=np.float32)  # (n_users, q)
        v_mul_s = np.array(Vt.T * S_sqrt, order='C', dtype=np.float32)  # (n_items, q)
        ut = np.array(U.T, order='C', dtype=np.float32)  # (q, n_users)
        vt = np.array(Vt, order='C', dtype=np.float32)  # (q, n_items)
        
        # Convert to tensors
        self.u_mul_s = torch.from_numpy(u_mul_s)
        self.v_mul_s = torch.from_numpy(v_mul_s)
        self.ut = torch.from_numpy(ut)
        self.vt = torch.from_numpy(vt)
        
        print(f"  SVD completed: q={len(S)}")
        print(f"  U shape: {self.u_mul_s.shape}, V shape: {self.v_mul_s.shape}")
        
    def to_device(self, device: torch.device):
        """Move model and buffers to device"""
        self.to(device)
        if self.u_mul_s is not None:
            self.u_mul_s = self.u_mul_s.to(device)
            self.v_mul_s = self.v_mul_s.to(device)
            self.ut = self.ut.to(device)
            self.vt = self.vt.to(device)
            
    def forward(self, adj_norm: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with LightGCN backbone
        """
        # Concatenate embeddings
        all_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        embeddings_list = [all_embeddings]
        
        # Multi-layer graph convolution
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)
            
            if self.dropout > 0 and self.training:
                all_embeddings = F.dropout(all_embeddings, p=self.dropout)
                
            embeddings_list.append(all_embeddings)
            
        # Layer combination (mean pooling)
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split
        user_emb = all_embeddings[:self.n_users]
        item_emb = all_embeddings[self.n_users:]
        
        return user_emb, item_emb
    
    def get_svd_embeddings(self, user_emb: torch.Tensor, 
                           item_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create SVD-augmented embeddings for contrastive learning
        """
        # User SVD embedding: u_mul_s @ vt @ item_emb
        user_svd_emb = self.u_mul_s @ (self.vt @ item_emb)
        
        # Item SVD embedding: v_mul_s @ ut @ user_emb
        item_svd_emb = self.v_mul_s @ (self.ut @ user_emb)
        
        return user_svd_emb, item_svd_emb
    
    def contrastive_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                         user_svd_emb: torch.Tensor, item_svd_emb: torch.Tensor,
                         users: torch.Tensor, pos_items: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE Contrastive Loss
        """
        # Get batch embeddings and normalize
        user_e = F.normalize(user_emb[users], dim=1)
        item_e = F.normalize(item_emb[pos_items], dim=1)
        user_svd_e = F.normalize(user_svd_emb[users], dim=1)
        item_svd_e = F.normalize(item_svd_emb[pos_items], dim=1)
        
        # User contrastive loss
        pos_score_user = torch.sum(user_e * user_svd_e, dim=1) / self.temp
        neg_score_user = user_e @ user_svd_e.T / self.temp
        ssl_loss_user = -torch.mean(
            pos_score_user - torch.logsumexp(neg_score_user, dim=1)
        )
        
        # Item contrastive loss
        pos_score_item = torch.sum(item_e * item_svd_e, dim=1) / self.temp
        neg_score_item = item_e @ item_svd_e.T / self.temp
        ssl_loss_item = -torch.mean(
            pos_score_item - torch.logsumexp(neg_score_item, dim=1)
        )
        
        return ssl_loss_user + ssl_loss_item
    
    def bpr_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                 users: torch.Tensor, pos_items: torch.Tensor, 
                 neg_items: torch.Tensor) -> torch.Tensor:
        """
        BPR (Bayesian Personalized Ranking) Loss
        """
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss
    
    def reg_loss(self, users: torch.Tensor, pos_items: torch.Tensor,
                 neg_items: torch.Tensor) -> torch.Tensor:
        """
        L2 Regularization on initial embeddings
        """
        user_e = self.user_embedding.weight[users]
        pos_e = self.item_embedding.weight[pos_items]
        neg_e = self.item_embedding.weight[neg_items]
        
        reg = (user_e.norm(2).pow(2) + 
               pos_e.norm(2).pow(2) + 
               neg_e.norm(2).pow(2)) / (2 * len(users))
        
        return reg
    
    def calculate_loss(self, adj_norm: torch.sparse.Tensor,
                       users: torch.Tensor, pos_items: torch.Tensor,
                       neg_items: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Calculate total loss = BPR + Reg + SSL
        """
        # Forward
        user_emb, item_emb = self.forward(adj_norm)
        
        # BPR loss
        bpr = self.bpr_loss(user_emb, item_emb, users, pos_items, neg_items)
        
        # Regularization
        reg = self.reg_loss(users, pos_items, neg_items)
        
        # Contrastive loss
        user_svd_emb, item_svd_emb = self.get_svd_embeddings(user_emb, item_emb)
        ssl = self.contrastive_loss(user_emb, item_emb, user_svd_emb, item_svd_emb,
                                    users, pos_items)
        
        # Total
        total = bpr + self.reg_weight * reg + self.ssl_weight * ssl
        
        return total, bpr, reg, ssl
    
    def predict(self, adj_norm: torch.sparse.Tensor, 
                users: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict scores for all items
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_norm)
            
            if users is not None:
                user_e = user_emb[users]
            else:
                user_e = user_emb
                
            scores = user_e @ item_emb.T
            
        return scores
    
    def get_embeddings(self, adj_norm: torch.sparse.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get final embeddings"""
        self.eval()
        with torch.no_grad():
            return self.forward(adj_norm)