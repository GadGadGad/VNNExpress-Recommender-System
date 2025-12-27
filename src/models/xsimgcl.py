import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class XSimGCL(nn.Module):
    """
    XSimGCL: eXtremely Simple Graph Contrastive Learning for Recommendation
    Paper: https://arxiv.org/abs/2112.08679
    """
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 eps=0.1, ssl_weight=0.1, temp=0.2, lambda_reg=1e-4):
        super(XSimGCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.eps = eps
        self.ssl_weight = ssl_weight
        self.temp = temp
        self.lambda_reg = lambda_reg

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Optional Semantic ID layer
        self.semantic_layer = None
        self.semantic_alpha = nn.Parameter(torch.tensor(0.1)) # Learnable scaling
        
        # Optional User Prior layer
        self.user_prior_layer = None
        self.user_alpha = nn.Parameter(torch.tensor(0.1)) # Learnable scaling

    def forward(self, adj_norm, perturb=False, semantic_ids=None, user_priors=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        if semantic_ids is not None and self.semantic_layer is not None:
            s_emb = self.semantic_layer(semantic_ids)
            i_emb = i_emb + self.semantic_alpha * F.normalize(s_emb, p=2, dim=1)
            
        if user_priors is not None and self.user_prior_layer is not None:
            up_emb = self.user_prior_layer(user_priors)
            u_emb = u_emb + self.user_alpha * F.normalize(up_emb, p=2, dim=1)
            
        all_emb = torch.cat([u_emb, i_emb], dim=0)
        
        embs_list = [all_emb]
        for i in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_norm, all_emb)
            if perturb:
                # Noise-based embedding augmentation (XSimGCL core)
                noise = torch.randn_like(all_emb) * self.eps
                all_emb = all_emb + noise
            embs_list.append(all_emb)
            
        final_embs = torch.stack(embs_list, dim=1).mean(dim=1)
        user_all, item_all = torch.split(final_embs, [self.n_users, self.n_items])
        return user_all, item_all

    def calculate_loss(self, adj_norm, users, pos_items, neg_items, semantic_ids=None, user_priors=None, return_per_sample=False):
        # Original view
        user_all, item_all = self.forward(adj_norm, perturb=False, semantic_ids=semantic_ids, user_priors=user_priors)
        
        # Perturbed view for SSL
        user_all_p, item_all_p = self.forward(adj_norm, perturb=True, semantic_ids=semantic_ids, user_priors=user_priors)
        
        u_emb = user_all[users]
        pos_emb = item_all[pos_items]
        neg_emb = item_all[neg_items]
        
        # BPR Loss
        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        bpr_sample = -F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = bpr_sample.mean()
        
        # SSL Loss (InfoNCE)
        u_emb_p = user_all_p[users]
        pos_emb_p = item_all_p[pos_items]
        
        # User SSL
        u_norm = F.normalize(u_emb, p=2, dim=1)
        u_p_norm = F.normalize(u_emb_p, p=2, dim=1)
        pos_score_u = torch.exp((u_norm * u_p_norm).sum(dim=1) / self.temp)
        ttl_score_u = torch.exp(torch.mm(u_norm, u_p_norm.t()) / self.temp).sum(dim=1)
        ssl_loss_u = -torch.log(pos_score_u / ttl_score_u).mean()
        
        # Item SSL
        i_norm = F.normalize(pos_emb, p=2, dim=1)
        i_p_norm = F.normalize(pos_emb_p, p=2, dim=1)
        pos_score_i = torch.exp((i_norm * i_p_norm).sum(dim=1) / self.temp)
        ttl_score_i = torch.exp(torch.mm(i_norm, i_p_norm.t()) / self.temp).sum(dim=1)
        ssl_loss_i = -torch.log(pos_score_i / ttl_score_i).mean()
        
        ssl_loss = ssl_loss_u + ssl_loss_i
        
        # Reg Loss
        reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.size(0)
        
        total_loss = bpr_loss + self.ssl_weight * ssl_loss + self.lambda_reg * reg_loss
        
        if return_per_sample:
            return total_loss, bpr_sample, ssl_loss, reg_loss
        return total_loss, bpr_loss, ssl_loss, reg_loss

    def predict(self, adj_norm, users=None, semantic_ids=None, user_priors=None):
        user_all, item_all = self.forward(adj_norm, semantic_ids=semantic_ids, user_priors=user_priors)
        if users is not None:
            return torch.mm(user_all[users], item_all.t())
        return torch.mm(user_all, item_all.t())
