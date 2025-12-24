#!/usr/bin/env python3
"""
Compare CF Models on Alternative Graph Types
=============================================
Trains CF models on different graph types and compares performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from datetime import datetime
import argparse
from tabulate import tabulate

# Import models
import sys
sys.path.insert(0, '.')
from src.models.ngcf import NGCF
from src.models.simplex import SimpleX
from src.models.directau import DirectAU


def load_graph_data(graph_path):
    """Load graph data and prepare for training."""
    data = torch.load(graph_path, weights_only=False)
    
    # Extract common fields
    n_users = data.get('n_users', 0)
    n_items = data.get('n_items', data.get('n_articles', data.get('n_categories', 0)))
    train_pairs = data.get('train_pairs', [])
    train_dict = data.get('train_dict', {})
    
    # Get edge index for GNN models
    if 'edge_index' in data:
        edge_index = data['edge_index']
    elif 'user_category_edge_index' in data:
        edge_index = data['user_category_edge_index']
    elif 'article_edge_index' in data:
        edge_index = data['article_edge_index']
    else:
        # Create from train pairs
        if train_pairs:
            src = [p[0] for p in train_pairs]
            dst = [p[1] for p in train_pairs]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    return {
        'n_users': n_users,
        'n_items': n_items,
        'train_pairs': train_pairs,
        'train_dict': train_dict,
        'edge_index': edge_index,
        'graph_type': data.get('graph_type', 'unknown')
    }


def sample_batch(train_pairs, train_dict, n_items, batch_size, neg_ratio=4):
    """Sample batch with negative items."""
    if len(train_pairs) == 0:
        return None, None, None
    >
    indices = np.random.choice(len(train_pairs), min(batch_size, len(train_pairs)), replace=False)
    users, pos_items, neg_items = [], [], []
    
    for idx in indices:
        u, pos = train_pairs[idx]
        for _ in range(neg_ratio):
            users.append(u)
            pos_items.append(pos)
            neg = np.random.randint(0, n_items)
            while neg in train_dict.get(u, set()):
                neg = np.random.randint(0, n_items)
            neg_items.append(neg)
    
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long)
    )


def create_test_split(train_pairs, train_dict, test_ratio=0.2):
    """Split data into train/test."""
    # Group by user
    user_items = defaultdict(list)
    for u, i in train_pairs:
        user_items[u].append(i)
    
    new_train_pairs = []
    new_train_dict = defaultdict(set)
    test_dict = {}
    
    for user, items in user_items.items():
        if len(items) >= 2:
            np.random.shuffle(items)
            n_test = max(1, int(len(items) * test_ratio))
            test_items = items[:n_test]
            train_items = items[n_test:]
            
            test_dict[user] = test_items
            for i in train_items:
                new_train_pairs.append((user, i))
                new_train_dict[user].add(i)
        else:
            # Keep all in train
            for i in items:
                new_train_pairs.append((user, i))
                new_train_dict[user].add(i)
    
    return new_train_pairs, dict(new_train_dict), test_dict


def evaluate(model, test_dict, train_dict, n_items, edge_index, k_list=[5, 10], device='cpu'):
    """Evaluate model."""
    model.eval()
    
    with torch.no_grad():
        try:
            user_emb, item_emb = model(edge_index.to(device))
        except:
            # Some models don't need edge_index
            user_emb, item_emb = model(None)
    
    user_emb = user_emb.detach().cpu().numpy()
    item_emb = item_emb.detach().cpu().numpy()
    
    results = defaultdict(list)
    
    for user, test_items in test_dict.items():
        if user >= len(user_emb):
            continue
        
        scores = user_emb[user] @ item_emb.T
        
        # Mask training items
        for item in train_dict.get(user, set()):
            if item < len(scores):
                scores[item] = -np.inf
        
        for k in k_list:
            top_k = np.argsort(scores)[::-1][:k]
            hits = len(set(top_k) & set(test_items))
            
            # Recall
            results[f'recall@{k}'].append(hits / len(test_items))
            
            # NDCG
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            results[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0)
            
            # Hit Rate
            results[f'hr@{k}'].append(1.0 if hits > 0 else 0.0)
    
    return {k: np.mean(v) for k, v in results.items()}


def train_model(model, data, epochs=30, batch_size=512, lr=0.001, neg_ratio=1, device='cpu'):
    """Train a CF model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_pairs, train_dict, test_dict = create_test_split(data['train_pairs'], data['train_dict'])
    edge_index = data['edge_index'].to(device)
    n_items = data['n_items']
    
    if len(train_pairs) == 0:
        return {'recall@5': 0, 'recall@10': 0, 'ndcg@10': 0, 'hr@10': 0}
    
    best_recall = 0
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = max(1, len(train_pairs) // batch_size)
        
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_batch(train_pairs, train_dict, n_items, batch_size, neg_ratio)
            if users is None:
                continue
            
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'bpr_loss'):
                loss, reg = model.bpr_loss(users, pos_items, neg_items, edge_index)
                loss = loss + 1e-4 * reg
            else:
                user_emb, item_emb = model(edge_index)
                pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
                neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(model, test_dict, train_dict, n_items, edge_index, device=device)
            if metrics.get('recall@10', 0) > best_recall:
                best_recall = metrics['recall@10']
                best_metrics = metrics.copy()
    
    return best_metrics


def compare_graphs(graph_paths, models=['ngcf', 'simplex', 'directau'], epochs=30, neg_ratio=1, device='cpu'):
    """Compare CF models on different graphs."""
    results = []
    
    for graph_path in graph_paths:
        print(f"\n{'='*60}")
        print(f"Graph: {graph_path}")
        print('='*60)
        
        try:
            data = load_graph_data(graph_path)
            print(f"  Users: {data['n_users']}, Items: {data['n_items']}")
            print(f"  Train pairs: {len(data['train_pairs'])}")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        for model_name in models:
            print(f"\n  Training {model_name.upper()}...")
            
            try:
                n_users = data['n_users']
                n_items = data['n_items']
                
                if model_name == 'ngcf':
                    model = NGCF(n_users, n_items, 64, 3).to(device)
                elif model_name == 'simplex':
                    model = SimpleX(n_users, n_items, 64).to(device)
                elif model_name == 'directau':
                    model = DirectAU(n_users, n_items, 64).to(device)
                
                metrics = train_model(model, data, epochs=epochs, neg_ratio=neg_ratio, device=device)
                
                results.append({
                    'Graph': graph_path.split('/')[-1].replace('.pt', ''),
                    'Model': model_name.upper(),
                    'R@5': f"{metrics.get('recall@5', 0):.4f}",
                    'R@10': f"{metrics.get('recall@10', 0):.4f}",
                    'N@10': f"{metrics.get('ndcg@10', 0):.4f}",
                    'HR@10': f"{metrics.get('hr@10', 0):.4f}",
                })
                
                print(f"    R@10: {metrics.get('recall@10', 0):.4f}, NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
                
            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    'Graph': graph_path.split('/')[-1].replace('.pt', ''),
                    'Model': model_name.upper(),
                    'R@5': 'ERR',
                    'R@10': 'ERR',
                    'N@10': 'ERR',
                    'HR@10': 'ERR',
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare CF models on alternative graphs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--neg-ratio', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPARING CF MODELS ON ALTERNATIVE GRAPH TYPES")
    print("=" * 70)
    print(f"Epochs: {args.epochs} | Neg Ratio: {args.neg_ratio} | Device: {args.device}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    graph_paths = [
        'data/processed_graphs/user_category_graph.pt',
        'data/processed_graphs/reaction_weighted_graph.pt',
        'data/processed_graphs/no_comment_filtered_graph.pt',
    ]
    
    # Check which graphs exist
    import os
    graph_paths = [p for p in graph_paths if os.path.exists(p)]
    
    results = compare_graphs(
        graph_paths,
        models=['ngcf', 'simplex', 'directau'],
        epochs=args.epochs,
        neg_ratio=args.neg_ratio,
        device=args.device
    )
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(tabulate(results, headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    main()
