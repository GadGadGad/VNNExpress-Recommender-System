"""
NCL: Neighborhood-enriched Contrastive Learning (WWW 2022)
Key: Uses prototypical contrastive learning with cluster-based neighbors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class NCLConv(MessagePassing):
    """LightGCN-style convolution."""
    
    def __init__(self):
        super().__init__(aggr='mean')
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j


class NCL(nn.Module):
    """Neighborhood-enriched Contrastive Learning."""
    
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3,
                 n_clusters=100, ssl_weight=0.1, proto_weight=0.1, temp=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.ssl_weight = ssl_weight
        self.proto_weight = proto_weight
        self.temp = temp
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Prototype embeddings (cluster centers)
        self.user_prototypes = nn.Parameter(torch.randn(n_clusters, embedding_dim))
        self.item_prototypes = nn.Parameter(torch.randn(n_clusters, embedding_dim))
        
        self.convs = nn.ModuleList([NCLConv() for _ in range(n_layers)])
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.user_prototypes)
        nn.init.xavier_uniform_(self.item_prototypes)
    
    def get_ego_embeddings(self):
        return torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
    
    def forward(self, edge_index):
        x = self.get_ego_embeddings()
        all_embs = [x]
        
        for conv in self.convs:
            x = conv(x, edge_index)
            all_embs.append(x)
        
        all_embs = torch.stack(all_embs, dim=1)
        final_emb = torch.mean(all_embs, dim=1)
        
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def proto_loss(self, user_emb, item_emb, users, items):
        """Prototypical contrastive loss."""
        u_emb = F.normalize(user_emb[users], dim=1)
        i_emb = F.normalize(item_emb[items], dim=1)
        
        u_proto = F.normalize(self.user_prototypes, dim=1)
        i_proto = F.normalize(self.item_prototypes, dim=1)
        
        # User-prototype similarity
        u_sim = torch.mm(u_emb, u_proto.t()) / self.temp
        u_loss = -torch.logsumexp(u_sim, dim=1).mean()
        
        # Item-prototype similarity
        i_sim = torch.mm(i_emb, i_proto.t()) / self.temp
        i_loss = -torch.logsumexp(i_sim, dim=1).mean()
        
        return (u_loss + i_loss) / 2
    
    def structural_loss(self, user_emb, item_emb, edge_index, users, items):
        """Structural neighborhood contrastive loss."""
        u_emb = F.normalize(user_emb[users], dim=1)
        i_emb = F.normalize(item_emb[items], dim=1)
        
        # Positive: user-item pairs
        pos = (u_emb * i_emb).sum(dim=1) / self.temp
        
        # Negative: random pairs
        neg_items = items[torch.randperm(items.size(0))]
        neg_emb = F.normalize(item_emb[neg_items], dim=1)
        neg = (u_emb * neg_emb).sum(dim=1) / self.temp
        
        return -F.logsigmoid(pos - neg).mean()
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        """BPR loss with NCL regularization."""
        user_emb, item_emb = self.forward(edge_index)
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Prototypical contrastive loss
        proto = self.proto_loss(user_emb, item_emb, users, pos_items)
        
        # Structural contrastive loss
        struct = self.structural_loss(user_emb, item_emb, edge_index, users, pos_items)
        
        # Regularization
        reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        return bpr + self.proto_weight * proto + self.ssl_weight * struct, reg
