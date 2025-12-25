import torch
import torch.nn as nn
import torch.nn.functional as F

class CGRC(nn.Module):
    """
    Content-based Graph Reconstruction for Collaborative Filtering (CGRC)
    Simplified version focusing on bridging Content (TF-IDF/PhoBERT) and Graph (GCN).
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, content_dim=64, n_layers=3):
        super(CGRC, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # 1. ID Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 2. Content Projection (to align multimodal content with ID space)
        self.content_projection = nn.Linear(content_dim, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.content_projection.weight)
        
    def forward(self, adj_norm, item_content=None):
        """
        Forward pass with content injection.
        
        Args:
            adj_norm: Sparse normalized symmetric adjacency matrix (U+I)x(U+I)
            item_content: Pretrained item embeddings (N_items, content_dim)
        """
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # Inject content information into item representations
        if item_content is not None:
            c_emb = self.content_projection(item_content)
            i_emb = i_emb + c_emb # Residual-style fusion
            
        all_embeddings = torch.cat([u_emb, i_emb], dim=0)
        embs_list = [all_embeddings]
        
        # Graph Convolution (LightGCN style)
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_norm, all_embeddings)
            embs_list.append(all_embeddings)
            
        # Aggregate layers
        light_out = torch.stack(embs_list, dim=1)
        final_embeddings = torch.mean(light_out, dim=1)
        
        user_all, item_all = torch.split(final_embeddings, [self.n_users, self.n_items])
        return user_all, item_all
    
    def calculate_loss(self, adj_norm, users, pos_items, neg_items, item_content=None):
        """
        Combined BPR Loss + Reconstruction Loss
        """
        # 1. Compute Embeddings
        user_emb, item_emb = self.forward(adj_norm, item_content)
        
        u_g = user_emb[users]
        pos_g = item_emb[pos_items]
        neg_g = item_emb[neg_items]
        
        # 2. BPR Loss
        pos_scores = torch.sum(u_g * pos_g, dim=1)
        neg_scores = torch.sum(u_g * neg_g, dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # 3. Reconstruction Loss (MGAE style)
        # Minimize the distance between predicted scores and true interactions
        # We can also add a feature reconstruction loss:
        # ||content_projection(item_content) - item_embedding.weight||^2
        recon_loss = 0.0
        if item_content is not None:
            projected_content = self.content_projection(item_content)
            # Alignment between ID embedding and Content embedding
            recon_loss = F.mse_loss(projected_content, self.item_embedding.weight)
        
        # 4. Total Loss
        reg_loss = (u_g.norm(2).pow(2) + pos_g.norm(2).pow(2) + neg_g.norm(2).pow(2)) / users.shape[0]
        
        # Weighting factors (defaults)
        lambda_recon = 0.1
        lambda_reg = 1e-4
        
        total_loss = bpr_loss + lambda_recon * recon_loss + lambda_reg * reg_loss
        
        return total_loss, bpr_loss, recon_loss, reg_loss
