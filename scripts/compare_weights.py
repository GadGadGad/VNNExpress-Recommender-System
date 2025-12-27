#!/usr/bin/env python3
"""
Quick A/B Test: G2 (Complex Weights) vs G2' (Simple Weights)
============================================================
Compares recommendation performance between weighted and unweighted graphs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SimGCL
import scipy.sparse as sp


def compute_adj(n_users, n_items, train_pairs, edge_weights=None):
    """Create normalized adjacency matrix."""
    row = np.array([u for u, i in train_pairs])
    col = np.array([i for u, i in train_pairs])
    
    if edge_weights is not None:
        data = edge_weights
    else:
        data = np.ones(len(train_pairs), dtype=np.float32)
    
    R = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32).tolil()
    R = R.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.tocoo()
    
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocoo()
    
    indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).coalesce()


def evaluate(model, adj, n_users, n_items, test_pairs, k=10):
    """Quick ranking evaluation."""
    model.eval()
    with torch.no_grad():
        E_u, E_i = model(adj)
    
    # Sample users for evaluation
    test_users = list(set([u for u, _ in test_pairs]))[:500]
    
    # Build ground truth
    gt = {}
    for u, i in test_pairs:
        if u not in gt:
            gt[u] = set()
        gt[u].add(i)
    
    recall_sum = 0
    ndcg_sum = 0
    count = 0
    
    for u in test_users:
        if u not in gt:
            continue
        
        scores = torch.matmul(E_u[u], E_i.T)
        _, top_k = torch.topk(scores, k)
        top_k = top_k.cpu().numpy()
        
        hits = len(set(top_k) & gt[u])
        recall_sum += hits / min(len(gt[u]), k)
        
        # NDCG
        dcg = 0
        for rank, item in enumerate(top_k):
            if item in gt[u]:
                dcg += 1 / np.log2(rank + 2)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gt[u]), k)))
        ndcg_sum += dcg / idcg if idcg > 0 else 0
        count += 1
    
    return recall_sum / count if count > 0 else 0, ndcg_sum / count if count > 0 else 0


def run_experiment(graph_path, name, epochs=30):
    """Train and evaluate on a graph."""
    print(f"\n{'='*60}")
    print(f"Training on: {name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(graph_path, weights_only=False)
    
    # Get dimensions
    n_users = data['user'].x.shape[0]
    n_items = data['article'].x.shape[0]
    
    # Get edges
    edge_index = data['user', 'comments', 'article'].edge_index
    edge_weight = data['user', 'comments', 'article'].get('edge_weight', None)
    if edge_weight is not None:
        edge_weight = edge_weight.numpy()
    
    # Split train/test
    n_edges = edge_index.shape[1]
    perm = np.random.permutation(n_edges)
    train_idx = perm[:int(0.8 * n_edges)]
    test_idx = perm[int(0.8 * n_edges):]
    
    train_pairs = list(zip(edge_index[0, train_idx].numpy(), edge_index[1, train_idx].numpy()))
    test_pairs = list(zip(edge_index[0, test_idx].numpy(), edge_index[1, test_idx].numpy()))
    
    train_weights = edge_weight[train_idx] if edge_weight is not None else None
    
    # Build adjacency
    adj = compute_adj(n_users, n_items, train_pairs, train_weights).to(device)
    
    # Model
    model = SimGCL(n_users, n_items, embedding_dim=64, n_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Sample batch
        batch_idx = np.random.choice(len(train_pairs), min(2048, len(train_pairs)), replace=False)
        users = torch.tensor([train_pairs[i][0] for i in batch_idx], device=device)
        pos_items = torch.tensor([train_pairs[i][1] for i in batch_idx], device=device)
        neg_items = torch.randint(0, n_items, (len(batch_idx),), device=device)
        
        optimizer.zero_grad()
        
        # Forward pass first
        E_u, E_i = model(adj)
        
        # BPR loss with embeddings
        loss = model.bpr_loss(E_u, E_i, users, pos_items, neg_items)
        reg_loss = model.reg_weight * (E_u[users].norm(2).pow(2) + E_i[pos_items].norm(2).pow(2))
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            recall, ndcg = evaluate(model, adj, n_users, n_items, test_pairs)
            print(f"Epoch {epoch+1}: Loss={total_loss.item():.4f}, R@10={recall:.4f}, NDCG@10={ndcg:.4f}")
    
    # Final evaluation
    recall, ndcg = evaluate(model, adj, n_users, n_items, test_pairs)
    print(f"\n[FINAL] {name}: R@10={recall:.4f}, NDCG@10={ndcg:.4f}")
    
    return recall, ndcg


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Compare G2' (simple) vs G2 (complex if exists)
    results = {}
    
    # G2' - Simple binary weights (same structure as G2)
    if Path('data/processed/full_hetero_simple_strict.pt').exists():
        r, n = run_experiment('data/processed/full_hetero_simple_strict.pt', "G2' (w=1)")
        results["G2' (Simple w=1)"] = {'R@10': r, 'NDCG@10': n}
    
    # G2 - Complex weights (full_hetero_graph.pt)
    if Path('data/processed/full_hetero_graph.pt').exists():
        r, n = run_experiment('data/processed/full_hetero_graph.pt', "G2 (Complex)")
        results["G2 (Complex)"] = {'R@10': r, 'NDCG@10': n}
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name}: R@10={metrics['R@10']:.4f}, NDCG@10={metrics['NDCG@10']:.4f}")
