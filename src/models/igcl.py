import torch
import torch.nn as nn
import torch.nn.functional as F

class IGCL(nn.Module):
    """
    Information-Controllable Graph Contrastive Learning (IGCL)
    Addresses 'augmented representation collapse' by controlling shared information.
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 temp=0.2, ssl_weight=0.1, info_penalty=0.01):
        super(IGCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.temp = temp
        self.ssl_weight = ssl_weight
        self.info_penalty = info_penalty # Power of information control
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def graph_step(self, adj_norm, embeddings):
        return torch.sparse.mm(adj_norm.float(), embeddings.float())
        
    def forward(self, adj_norm, perturb=False):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        all_embeddings = torch.cat([u_emb, i_emb], dim=0).float()
        
        # Optional: Noise perturbation for contrastive views
        if perturb:
            random_noise = torch.randn_like(all_embeddings) * 0.1
            all_embeddings = all_embeddings + random_noise
            
        embs_list = [all_embeddings]
        for _ in range(self.n_layers):
            # print(f"DEBUG: adj_norm device: {adj_norm.device}, embeddings device: {all_embeddings.device}")
            # print(f"DEBUG: adj_norm shape: {adj_norm.shape}, embeddings shape: {all_embeddings.shape}")
            all_embeddings = torch.sparse.mm(adj_norm.float(), all_embeddings.float())
            embs_list.append(all_embeddings)
            
        final_embs = torch.stack(embs_list, dim=1).mean(dim=1)
        user_all, item_all = torch.split(final_embs, [self.n_users, self.n_items])
        return user_all, item_all

    def info_control_loss(self, z1, z2):
        """
        Information-Controllable Inhibition Term.
        Ensures that z1 and z2 don't share REDUNDANT information.
        Inspired by the principle of minimizing off-diagonal cross-correlation.
        """
        # Normalize along batch dimension
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        
        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / z1.shape[0]
        
        # Loss: Off-diagonal elements should be 0 (decoupling dimensions)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        
        return on_diag + off_diag # Hybrid loss to maintain useful info but decouple views

    def contrastive_loss(self, z1, z2, users):
        # InfoNCE
        u_z1 = F.normalize(z1[users], p=2, dim=1)
        u_z2 = F.normalize(z2[users], p=2, dim=1)
        
        pos_score = torch.exp(torch.sum(u_z1 * u_z2, dim=1) / self.temp)
        all_score = torch.exp(torch.mm(u_z1, u_z2.t()) / self.temp).sum(dim=1)
        
        cl_loss = -torch.log(pos_score / all_score).mean()
        return cl_loss

    def calculate_loss(self, adj_norm, users, pos_items, neg_items):
        # 1. Main View (BPR)
        user_all, item_all = self.forward(adj_norm)
        u_g = user_all[users]
        pos_g = item_all[pos_items]
        neg_g = item_all[neg_items]
        
        bpr_loss = -F.logsigmoid(torch.sum(u_g * pos_g, -1) - torch.sum(u_g * neg_g, -1)).mean()
        
        # 2. Contrastive Views with Information Control
        # Generate two jittered views
        z1_u, z1_i = self.forward(adj_norm, perturb=True)
        z2_u, z2_i = self.forward(adj_norm, perturb=True)
        
        # Standard SSL loss (InfoNCE)
        ssl_loss = self.contrastive_loss(z1_u, z2_u, users)
        
        # Information Inhibition Term (The "IGCL" core)
        inhibition = self.info_control_loss(z1_u[users], z2_u[users])
        
        # 3. Reg Loss
        reg_loss = (u_g.norm(2).pow(2) + pos_g.norm(2).pow(2) + neg_g.norm(2).pow(2)) / users.shape[0]
        
        total_loss = bpr_loss + self.ssl_weight * ssl_loss + self.info_penalty * inhibition + 1e-4 * reg_loss
        return total_loss, bpr_loss, ssl_loss, reg_loss
