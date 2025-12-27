"""
SimMAHGN: MA-HGN with Contrastive Learning (SimGCL style)
Adds noise-based perturbation and InfoNCE loss to the Heterogeneous GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ma_hgn import MAHGN

class SimMAHGN(MAHGN):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, dropout=0.2, 
                 eps=0.1, ssl_weight=0.1, temp=0.2, lambda_reg=1e-4, n_categories=0):
        super().__init__(n_users, n_items, embedding_dim, n_layers, dropout, n_categories)
        self.eps = eps
        self.ssl_weight = ssl_weight
        self.temp = temp
        self.lambda_reg = lambda_reg
        
    def forward(self, x_dict, edge_index_dict, perturb=False):
        """
        Forward pass with optional perturbation for CL.
        """
        if x_dict is None:
            x_dict = {
                'user': self.user_emb.weight,
                'article': self.item_emb.weight
            }
            if self.n_categories > 0:
                x_dict['category'] = self.category_emb.weight
            
        # Perturbation (Noise Injection) for CL View
        if perturb:
            x_dict = {
                key: emb + torch.randn_like(emb) * self.eps 
                for key, emb in x_dict.items()
            }
            
        h_dict = x_dict
        all_user_embs = [h_dict['user']]
        all_item_embs = [h_dict['article']]
        
        for conv in self.convs:
            # Ensure edge_index_dict is on the same device as embeddings
            device = h_dict['user'].device
            edge_index_dict_gpu = {
                key: ei.to(device) if hasattr(ei, 'to') else ei
                for key, ei in edge_index_dict.items()
            }
            # Message Passing
            h_dict = conv(h_dict, edge_index_dict_gpu)
            
            # Activation & Dropout
            h_dict = {k: F.leaky_relu(v) for k, v in h_dict.items()}
            h_dict = {k: self.dropout(v) for k, v in h_dict.items()}
            
            all_user_embs.append(h_dict['user'])
            all_item_embs.append(h_dict['article'])
            
        # Final embedding: average across layers
        user_final = torch.stack(all_user_embs, dim=1).mean(dim=1)
        item_final = torch.stack(all_item_embs, dim=1).mean(dim=1)
        
        return user_final, item_final

    def calculate_loss(self, edge_index_dict, users, pos_items, neg_items, **kwargs):
        """
        Computes (BPR Loss + InfoNCE Loss + Reg Loss).
        """
        # 1. Main View (Forward)
        user_main, item_main = self.forward(None, edge_index_dict, perturb=False)
        
        u_emb = user_main[users]
        pos_emb = item_main[pos_items]
        neg_emb = item_main[neg_items]
        
        # BPR Loss
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # 2. Perturbed View (Forward for CL)
        user_cl, item_cl = self.forward(None, edge_index_dict, perturb=True)
        
        # Contrastive Loss (InfoNCE)
        # We perform CL on the specific batch users and items to save memory/time
        # User Side CL
        u_main_norm = F.normalize(u_emb, p=2, dim=1)
        u_cl_norm = F.normalize(user_cl[users], p=2, dim=1)
        
        # Positive pair: same user in Main vs Perturbed view
        pos_score_u = torch.exp((u_main_norm * u_cl_norm).sum(dim=1) / self.temp)
        # Denominator: user vs all other users in batch (approximate softmax)
        ttl_score_u = torch.exp(torch.mm(u_main_norm, u_cl_norm.t()) / self.temp).sum(dim=1)
        ssl_loss_u = -torch.log(pos_score_u / ttl_score_u).mean()
        
        # Item Side CL (on Positive Items only)
        # NOTE: We only do CL on positive items to ensure we have good representations for interacting items
        i_main_norm = F.normalize(pos_emb, p=2, dim=1)
        i_cl_norm = F.normalize(item_cl[pos_items], p=2, dim=1)
        
        pos_score_i = torch.exp((i_main_norm * i_cl_norm).sum(dim=1) / self.temp)
        ttl_score_i = torch.exp(torch.mm(i_main_norm, i_cl_norm.t()) / self.temp).sum(dim=1)
        ssl_loss_i = -torch.log(pos_score_i / ttl_score_i).mean()
        
        ssl_loss = ssl_loss_u + ssl_loss_i
        
        # 3. Reg Loss
        reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        # Total Loss
        total_loss = bpr_loss + self.ssl_weight * ssl_loss + self.lambda_reg * reg_loss
        
        return total_loss, bpr_loss, ssl_loss, reg_loss
