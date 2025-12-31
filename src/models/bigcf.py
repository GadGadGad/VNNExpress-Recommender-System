import torch
import torch.nn as nn
import torch.nn.functional as F

class BIGCF(nn.Module):
    """
    Bilateral Intent-guided Graph Collaborative Filtering (BIGCF)
    Simplified version for Vietnamese News recommendation.
    Focuses on dual-intent (Individual vs. Collective) modeling.
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_intents=4, n_layers=3):
        super(BIGCF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_intents = n_intents
        self.n_layers = n_layers
        
        # 1. User/Item Embeddings (Individual preferences)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 2. Collective Intents (Latent prototypes shared across all users/items)
        # Represents global trends like "popularity" or "bandwagon effect"
        self.user_collective_intents = nn.Parameter(torch.randn(n_intents, embedding_dim))
        self.item_collective_intents = nn.Parameter(torch.randn(n_intents, embedding_dim))
        
        # 3. Intent Attention (To map individual preferences to collective intents)
        self.user_intent_attn = nn.Linear(embedding_dim, n_intents)
        self.item_intent_attn = nn.Linear(embedding_dim, n_intents)
        
        self.alpha = 0.5 # Fusion factor
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.user_collective_intents)
        nn.init.xavier_uniform_(self.item_collective_intents)
        
    def get_user_intents(self, users_emb):
        # Calculate attention weights over collective intents
        attn_weights = F.softmax(self.user_intent_attn(users_emb), dim=-1)
        # Combine collective intents based on attention
        collective_view = torch.mm(attn_weights, self.user_collective_intents)
        return collective_view
        
    def get_item_intents(self, items_emb):
        attn_weights = F.softmax(self.item_intent_attn(items_emb), dim=-1)
        collective_view = torch.mm(attn_weights, self.item_collective_intents)
        return collective_view

    def forward(self, adj_norm):
        # 1. Base Embeddings
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # 2. Intent Fusion (Bilateral)
        # Compute collective views
        u_collective = self.get_user_intents(u_emb)
        i_collective = self.get_item_intents(i_emb)
        
        # Fuse with individual preferences
        u_fused = u_emb + self.alpha * u_collective
        i_fused = i_emb + self.alpha * i_collective
        
        all_embeddings = torch.cat([u_fused, i_fused], dim=0).float()
        embs_list = [all_embeddings]
        
        # 3. Graph Convolution
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)
            embs_list.append(all_embeddings)
            
        final_embs = torch.stack(embs_list, dim=1).mean(dim=1)
        user_all, item_all = torch.split(final_embs, [self.n_users, self.n_items])
        
        return user_all, item_all

    def calculate_loss(self, adj_norm, users, pos_items, neg_items):
        # 1. Forward pass
        user_all, item_all = self.forward(adj_norm)
        
        u_g = user_all[users]
        pos_g = item_all[pos_items]
        neg_g = item_all[neg_items]
        
        # 2. BPR Loss
        pos_scores = torch.sum(u_g * pos_g, dim=-1)
        neg_scores = torch.sum(u_g * neg_g, dim=-1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # 3. Contrastive Loss (Dual-Space Regularization)
        # Align individual embedding with collective view
        u_ind = self.user_embedding(users)
        i_ind = self.item_embedding(pos_items)
        
        u_coll = self.get_user_intents(u_ind)
        i_coll = self.get_item_intents(i_ind)
        
        # Simple alignment loss
        cl_loss = (F.mse_loss(u_ind, u_coll) + F.mse_loss(i_ind, i_coll)) / 2.0
        
        # 4. Reg Loss
        reg_loss = (u_g.norm(2).pow(2) + pos_g.norm(2).pow(2) + neg_g.norm(2).pow(2)) / users.shape[0]
        
        total_loss = bpr_loss + 0.1 * cl_loss + 1e-4 * reg_loss
        return total_loss, bpr_loss, cl_loss, reg_loss
