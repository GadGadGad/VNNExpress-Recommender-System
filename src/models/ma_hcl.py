"""
MA-HCL: Multi-Aspect Heterogeneous Contrastive Learning
Uses LightGCN-style propagation (no attention, simpler) with contrastive learning.
Alternative to MA-HGN that replaces GATv2/GraphSAGE with simpler message passing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class LightGCNConv(MessagePassing):
    """Simple LightGCN-style message passing (no learnable params)."""
    def __init__(self):
        super().__init__(aggr='add')
    
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class MAHCL(nn.Module):
    """
    MA-HCL: Multi-Aspect Heterogeneous Contrastive Learning.
    Uses LightGCN propagation + SimCL contrastive loss.
    """
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3, 
                 ssl_weight=0.1, temp=0.2, eps=0.1, n_categories=0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.ssl_weight = ssl_weight
        self.temp = temp
        self.eps = eps
        
        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        if n_categories > 0:
            self.category_emb = nn.Embedding(n_categories, embedding_dim)
        
        # Simple LightGCN convolutions (no learnable params per layer)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
        
        # Multi-Aspect fusion weights (Interest vs Social)
        self.aspect_weight = nn.Parameter(torch.tensor([0.7, 0.3]))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        if hasattr(self, 'category_emb'):
            nn.init.xavier_uniform_(self.category_emb.weight)
    
    def _build_bipartite_graph(self, edge_index_dict):
        """Build bipartite user-item graph from hetero edge_index_dict or raw tensor."""
        # Handle raw tensor input (fallback from bipartite graph)
        if isinstance(edge_index_dict, torch.Tensor):
            # Assume it's already a bipartite edge_index [2, E] with user-item edges
            ua_edges = edge_index_dict
            src = ua_edges[0]
            dst = ua_edges[1] + self.n_users
            edge_index = torch.stack([
                torch.cat([src, dst]),
                torch.cat([dst, src])
            ], dim=0)
            return edge_index
        
        # Convert to regular dict if needed
        if hasattr(edge_index_dict, 'keys'):
            edge_dict = dict(edge_index_dict)
        else:
            edge_dict = edge_index_dict
            
        # Get user-article edges
        ua_edges = None
        for key in edge_dict.keys():
            if isinstance(key, tuple) and len(key) == 3:
                if 'user' in str(key[0]).lower() and ('article' in str(key[2]).lower() or 'item' in str(key[2]).lower()):
                    ua_edges = edge_dict[key]
                    break
        
        if ua_edges is None:
            return None
        
        # Build symmetric bipartite graph
        src = ua_edges[0]
        dst = ua_edges[1] + self.n_users
        
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)
        
        return edge_index
    
    def _build_social_graph(self, edge_index_dict):
        """Build user-user social graph."""
        # Handle raw tensor input (fallback from bipartite graph) - no social edges
        if isinstance(edge_index_dict, torch.Tensor):
            return None
        
        # Convert to regular dict if needed
        if hasattr(edge_index_dict, 'keys'):
            edge_dict = dict(edge_index_dict)
        else:
            edge_dict = edge_index_dict
            
        social_edges = None
        for key in edge_dict.keys():
            if isinstance(key, tuple) and len(key) == 3:
                if 'user' in str(key[0]).lower() and 'user' in str(key[2]).lower():
                    if 'replied' in str(key[1]).lower() or 'interact' in str(key[1]).lower():
                        if social_edges is None:
                            social_edges = edge_dict[key]
                        else:
                            social_edges = torch.cat([social_edges, edge_dict[key]], dim=1)
        
        if social_edges is None:
            return None
        
        # Make symmetric
        edge_index = torch.stack([
            torch.cat([social_edges[0], social_edges[1]]),
            torch.cat([social_edges[1], social_edges[0]])
        ], dim=0)
        
        return edge_index
    
    def forward(self, edge_index_dict=None, perturb=False):
        """
        Forward pass with multi-aspect propagation.
        """
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        
        if perturb:
            # Add noise for contrastive learning
            user_emb = user_emb + torch.randn_like(user_emb) * self.eps
            item_emb = item_emb + torch.randn_like(item_emb) * self.eps
        
        # Combined embedding for bipartite propagation
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Build graphs and ensure correct device
        device = user_emb.device
        if edge_index_dict is not None:
            bipartite_edge = self._build_bipartite_graph(edge_index_dict)
            social_edge = self._build_social_graph(edge_index_dict)
            if bipartite_edge is not None:
                bipartite_edge = bipartite_edge.to(device)
            if social_edge is not None:
                social_edge = social_edge.to(device)
        else:
            bipartite_edge = None
            social_edge = None
        
        # Interest aspect (user-item propagation)
        all_embs_interest = [x]
        h = x
        if bipartite_edge is not None:
            for conv in self.convs:
                h = conv(h, bipartite_edge)
                all_embs_interest.append(h)
        
        interest_emb = torch.stack(all_embs_interest, dim=1).mean(dim=1)
        
        # Social aspect (user-user propagation)
        if social_edge is not None:
            all_embs_social = [user_emb]
            h_social = user_emb
            for conv in self.convs:
                h_social = conv(h_social, social_edge)
                all_embs_social.append(h_social)
            social_emb = torch.stack(all_embs_social, dim=1).mean(dim=1)
        else:
            social_emb = user_emb
        
        # Split interest embeddings
        user_interest = interest_emb[:self.n_users]
        item_final = interest_emb[self.n_users:]
        
        # Multi-Aspect Fusion for users
        alpha = F.softmax(self.aspect_weight, dim=0)
        user_final = alpha[0] * user_interest + alpha[1] * social_emb
        
        return user_final, item_final
    
    def calculate_loss(self, edge_index_dict, users, pos_items, neg_items):
        """BPR loss + SimCL contrastive loss."""
        # Clean forward
        user_emb, item_emb = self.forward(edge_index_dict, perturb=False)
        
        # Perturbed forwards for contrastive learning
        user_emb1, item_emb1 = self.forward(edge_index_dict, perturb=True)
        user_emb2, item_emb2 = self.forward(edge_index_dict, perturb=True)
        
        # BPR Loss
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Contrastive Loss (InfoNCE)
        user_cl_loss = self._infonce_loss(user_emb1[users], user_emb2[users])
        item_cl_loss = self._infonce_loss(item_emb1[pos_items], item_emb2[pos_items])
        cl_loss = (user_cl_loss + item_cl_loss) / 2
        
        # L2 Regularization
        reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / users.shape[0]
        
        total_loss = bpr_loss + self.ssl_weight * cl_loss + 1e-4 * reg_loss
        
        return total_loss, bpr_loss, cl_loss, reg_loss
    
    def _infonce_loss(self, view1, view2):
        """InfoNCE contrastive loss."""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        pos_score = (view1 * view2).sum(dim=1) / self.temp
        neg_score = torch.mm(view1, view2.t()) / self.temp
        
        loss = -pos_score + torch.logsumexp(neg_score, dim=1)
        return loss.mean()
    
    def predict(self, users, items, edge_index_dict):
        """Prediction for inference."""
        user_final, item_final = self.forward(edge_index_dict, perturb=False)
        return (user_final[users] * item_final[items]).sum(dim=1)
