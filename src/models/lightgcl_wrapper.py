#!/usr/bin/env python3
"""
Wrapper for LightGCL that handles its special requirements (SVD, adj_norm).
"""
import numpy as np
import torch
import scipy.sparse as sp

from src.models import LightGCL


class LightGCLWrapper:
    """Wrapper that handles LightGCL's special requirements (SVD, adj_norm)."""
    
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, device='cpu',
                 svd_q=20, ssl_weight=0.1, temp=0.2):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.model = LightGCL(n_users, n_items, embed_dim, n_layers, 
                              svd_q=svd_q, ssl_weight=ssl_weight, temp=temp)
        self.adj_norm = None
        
    def setup(self, train_pairs, augmented_pairs=None):
        """Create normalized interaction matrix and compute SVD."""
        all_pairs = list(train_pairs)
        if augmented_pairs:
            print(f"  Injecting {len(augmented_pairs)} synthetic interactions from LLMRec...")
            all_pairs.extend(augmented_pairs)
            
        row = np.array([u for u, i in all_pairs])
        col = np.array([i for u, i in all_pairs])
        data = np.ones(len(all_pairs), dtype=np.float32)
        R = sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        
        rowD = np.array(R.sum(1)).squeeze()
        colD = np.array(R.sum(0)).squeeze()
        rowD[rowD == 0] = 1
        colD[colD == 0] = 1
        
        R_coo = R.tocoo()
        normalized_data = np.zeros_like(R_coo.data)
        for i in range(len(R_coo.data)):
            normalized_data[i] = R_coo.data[i] / np.sqrt(rowD[R_coo.row[i]] * colD[R_coo.col[i]])
        
        indices = torch.LongTensor(np.array([R_coo.row, R_coo.col]))
        values = torch.FloatTensor(normalized_data)
        shape = torch.Size(R_coo.shape)
        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(self.device)
        
        self.model.to_device(self.device)
        self.model.compute_svd(self.adj_norm)
        
    def to(self, device):
        self.device = device
        return self
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()
        
    def parameters(self):
        return self.model.parameters()
        
    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    def forward(self, edge_index=None):
        E_u, E_i, _, _ = self.model.forward(self.adj_norm)
        return E_u, E_i
    
    def __call__(self, edge_index=None):
        return self.forward(edge_index)
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        total_loss, bpr, reg, ssl = self.model.calculate_loss(
            self.adj_norm, users, pos_items, neg_items
        )
        return total_loss, torch.tensor(0.0, device=total_loss.device)
