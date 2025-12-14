#!/usr/bin/env python3
"""
Train Collaborative Filtering / Contrastive Learning Models
Supports: NGCF, SimpleX, DirectAU, SGL, SimGCL, NCL, LightGCL
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ngcf import NGCF
from src.models.simplex import SimpleX
from src.models.directau import DirectAU
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.models.ncl import NCL
from src.models.lightgcl import LightGCL
import scipy.sparse as sp


class LightGCLWrapper:
    """Wrapper that handles LightGCL's special requirements (SVD, adj_norm)."""
    
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, device='cpu'):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.model = LightGCL(n_users, n_items, embed_dim, n_layers)
        self.adj_norm = None
        
    def setup(self, train_pairs):
        """Create adjacency matrix and compute SVD."""
        # Create interaction matrix
        row = np.array([u for u, i in train_pairs])
        col = np.array([i for u, i in train_pairs])
        data = np.ones(len(train_pairs))
        R = sp.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        
        # Compute SVD
        self.model.compute_svd(R)
        
        # Create bipartite adjacency matrix
        adj_size = self.n_users + self.n_items
        adj_mat = sp.dok_matrix((adj_size, adj_size), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.tocsr()
        
        # Symmetric normalization
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        
        # Convert to sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        
        # Move model to device
        self.model.to_device(self.device)
        
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
        return self.model.forward(self.adj_norm)
    
    def __call__(self, edge_index=None):
        return self.forward(edge_index)
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        """Wrapper for LightGCL's calculate_loss."""
        total_loss, bpr, reg, ssl = self.model.calculate_loss(
            self.adj_norm, users, pos_items, neg_items
        )
        return total_loss, reg


def load_data(data_path, min_interactions=2):
    """Load and process data for CF models."""
    import pandas as pd
    from torch_geometric.data import HeteroData
    
    # Check for cached data
    cache_path = Path(data_path) / 'cf_cache.pt'
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}...")
        return torch.load(cache_path, weights_only=False)
    
    print("Processing data...")
    replies = pd.read_csv(Path(data_path).parent / 'raw' / 'replies.csv')
    replies = replies[replies['parent_user_id'] != 'NO_COMMENT'].copy()
    
    def clean_id(val):
        try:
            return str(int(float(val))) if pd.notna(val) else None
        except:
            return str(val)
    
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Filter by min interactions
    prev_len = 0
    while len(replies) != prev_len:
        prev_len = len(replies)
        user_counts = replies['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        article_counts = replies['article_url'].value_counts()
        valid_articles = article_counts[article_counts >= min_interactions].index
        replies = replies[
            (replies['user_id'].isin(valid_users)) &
            (replies['article_url'].isin(valid_articles))
        ].copy()
    
    user_ids = replies['user_id'].unique()
    article_urls = replies['article_url'].unique()
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    article_map = {url: idx for idx, url in enumerate(article_urls)}
    
    n_users = len(user_map)
    n_items = len(article_map)
    
    replies['user_idx'] = replies['user_id'].map(user_map)
    replies['item_idx'] = replies['article_url'].map(article_map)
    
    # Create edge index (user -> item)
    users = torch.tensor(replies['user_idx'].values, dtype=torch.long)
    items = torch.tensor(replies['item_idx'].values, dtype=torch.long)
    
    # Bipartite graph edge index
    edge_index = torch.stack([
        torch.cat([users, items + n_users]),
        torch.cat([items + n_users, users])
    ], dim=0)
    
    # Split train/test
    interactions = list(zip(users.numpy(), items.numpy()))
    interactions = list(set(interactions))
    np.random.shuffle(interactions)
    
    split_idx = int(len(interactions) * 0.8)
    train_pairs = interactions[:split_idx]
    test_pairs = interactions[split_idx:]
    
    train_dict = {}
    for u, i in train_pairs:
        if u not in train_dict:
            train_dict[u] = set()
        train_dict[u].add(i)
    
    test_dict = {}
    for u, i in test_pairs:
        if u not in test_dict:
            test_dict[u] = set()
        test_dict[u].add(i)
    
    data = {
        'n_users': n_users,
        'n_items': n_items,
        'edge_index': edge_index,
        'train_pairs': train_pairs,
        'train_dict': train_dict,
        'test_dict': test_dict,
    }
    
    torch.save(data, cache_path)
    return data


def sample_batch(train_pairs, train_dict, n_items, batch_size):
    """Sample a batch with negative items."""
    indices = np.random.choice(len(train_pairs), min(batch_size, len(train_pairs)), replace=False)
    users, pos_items, neg_items = [], [], []
    
    for idx in indices:
        u, pos = train_pairs[idx]
        users.append(u)
        pos_items.append(pos)
        
        # Sample negative
        neg = np.random.randint(0, n_items)
        while neg in train_dict.get(u, set()):
            neg = np.random.randint(0, n_items)
        neg_items.append(neg)
    
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long)
    )


def evaluate(model, test_dict, train_dict, n_items, edge_index, k_list=[1, 5, 10], device='cpu'):
    """Evaluate model with Recall, NDCG, HitRate at multiple k values."""
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'forward'):
            args = [edge_index.to(device)] if 'edge_index' in str(model.forward.__code__.co_varnames) else []
            user_emb, item_emb = model(*args) if args else model()
        else:
            user_emb = model.user_embedding.weight
            item_emb = model.item_embedding.weight
    
    max_k = max(k_list)
    results = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg', 'hitrate'] for k in k_list}
    results['mrr'] = []
    
    for user, test_items in test_dict.items():
        if user >= user_emb.size(0):
            continue
            
        train_items = train_dict.get(user, set())
        u_emb = user_emb[user].unsqueeze(0)
        scores = torch.mm(u_emb, item_emb.t()).squeeze()
        
        # Mask train items
        for item in train_items:
            if item < scores.size(0):
                scores[item] = -float('inf')
        
        _, topk = torch.topk(scores, max_k)
        topk_list = topk.cpu().numpy().tolist()
        
        # MRR (computed once, uses first hit)
        mrr = 0.0
        for i, item in enumerate(topk_list):
            if item in test_items:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'].append(mrr)
        
        # Compute metrics at each k
        for k in k_list:
            topk_k = set(topk_list[:k])
            hits = len(topk_k & test_items)
            
            # Recall@K
            results[f'recall@{k}'].append(hits / min(k, len(test_items)))
            
            # HitRate@K (at least one hit)
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
            
            # NDCG@K
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(topk_list[:k]) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
            results[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0)
    
    # Average all metrics
    return {k: np.mean(v) for k, v in results.items()}


def train_model(model, data, args, device):
    """Train a CF model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    edge_index = data['edge_index'].to(device)
    train_pairs = data['train_pairs']
    train_dict = data['train_dict']
    test_dict = data['test_dict']
    n_items = data['n_items']
    
    best_recall = 0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = len(train_pairs) // args.batch_size + 1
        
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_batch(train_pairs, train_dict, n_items, args.batch_size)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            
            # Different models have different loss signatures
            if hasattr(model, 'bpr_loss'):
                loss, reg = model.bpr_loss(users, pos_items, neg_items, edge_index)
                loss = loss + args.weight_decay * reg
            else:
                user_emb, item_emb = model(edge_index)
                pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
                neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = evaluate(model, test_dict, train_dict, n_items, edge_index, device=device)
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | "
                  f"R@1: {metrics.get('recall@1', 0):.3f} | R@5: {metrics.get('recall@5', 0):.3f} | R@10: {metrics.get('recall@10', 0):.3f}")
            
            recall_10 = metrics.get('recall@10', 0)
            if recall_10 > best_recall:
                best_recall = recall_10
                best_metrics = metrics.copy()
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description='Train CF/CL Models')
    parser.add_argument('--model', '-m', choices=['ngcf', 'simplex', 'directau', 'sgl', 'simgcl', 'ncl', 'lightgcl'],
                        default='ngcf', help='Model to train')
    parser.add_argument('--data-path', default='data/processed', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print(f"Training {args.model.upper()}")
    print("=" * 60)
    
    # Load data
    data = load_data(args.data_path)
    n_users, n_items = data['n_users'], data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(data['train_pairs'])}, Test users: {len(data['test_dict'])}")
    
    # Create model
    models_map = {
        'ngcf': lambda: NGCF(n_users, n_items, args.hidden_dim, args.n_layers),
        'simplex': lambda: SimpleX(n_users, n_items, args.hidden_dim),
        'directau': lambda: DirectAU(n_users, n_items, args.hidden_dim),
        'sgl': lambda: SGL(n_users, n_items, args.hidden_dim, args.n_layers),
        'simgcl': lambda: SimGCL(n_users, n_items, args.hidden_dim, args.n_layers),
        'ncl': lambda: NCL(n_users, n_items, args.hidden_dim, args.n_layers),
        'lightgcl': lambda: LightGCLWrapper(n_users, n_items, args.hidden_dim, args.n_layers, device=args.device),
    }
    
    model = models_map[args.model]()
    
    # Special setup for LightGCL (SVD computation + adj_norm creation)
    if args.model == 'lightgcl':
        model.setup(data['train_pairs'])
    else:
        model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    best_metrics = train_model(model, data, args, device)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"models/{args.model}_{timestamp}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items,
        'config': vars(args),
        'metrics': best_metrics
    }, save_path)
    print(f"\nModel saved: {save_path}")
    print(f"\nBest Metrics:")
    print(f"  Recall@1:   {best_metrics.get('recall@1', 0):.4f}")
    print(f"  Recall@5:   {best_metrics.get('recall@5', 0):.4f}")
    print(f"  Recall@10:  {best_metrics.get('recall@10', 0):.4f}")
    print(f"  NDCG@1:     {best_metrics.get('ndcg@1', 0):.4f}")
    print(f"  NDCG@5:     {best_metrics.get('ndcg@5', 0):.4f}")
    print(f"  NDCG@10:    {best_metrics.get('ndcg@10', 0):.4f}")
    print(f"  HitRate@10: {best_metrics.get('hitrate@10', 0):.4f}")
    print(f"  MRR:        {best_metrics.get('mrr', 0):.4f}")


if __name__ == '__main__':
    main()
