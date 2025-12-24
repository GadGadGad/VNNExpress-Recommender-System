#!/usr/bin/env python3
"""
Evaluate CF Models with User Group Segmentation
================================================
Groups users by interaction count and evaluates separately.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from datetime import datetime
import argparse
from tabulate import tabulate

import sys
sys.path.insert(0, '.')
from src.models.ngcf import NGCF
from src.models.simplex import SimpleX
from src.models.directau import DirectAU


# User groups by interaction count
USER_GROUPS = {
    'Cold': (1, 1),      # =1
    'Low': (2, 3),       # 2-3
    'Medium': (4, 10),   # 4-10
    'Active': (11, 999), # >10
}


def load_graph_data(graph_path):
    """Load graph data."""
    data = torch.load(graph_path, weights_only=False)
    
    n_users = data.get('n_users', 0)
    n_items = data.get('n_items', data.get('n_articles', 0))
    train_pairs = data.get('train_pairs', [])
    train_dict = data.get('train_dict', {})
    
    if 'edge_index' in data:
        edge_index = data['edge_index']
    else:
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
    }


def segment_users_by_group(train_dict):
    """Segment users into groups based on interaction count."""
    user_groups = {name: [] for name in USER_GROUPS.keys()}
    
    for user, items in train_dict.items():
        n_items = len(items)
        for group_name, (min_int, max_int) in USER_GROUPS.items():
            if min_int <= n_items <= max_int:
                user_groups[group_name].append(user)
                break
    
    return user_groups


def sample_batch(train_pairs, train_dict, n_items, batch_size, neg_ratio=1):
    """Sample batch with negatives."""
    if len(train_pairs) == 0:
        return None, None, None
    
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
    """Split data into train/test, preserving user groups."""
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
            for i in items:
                new_train_pairs.append((user, i))
                new_train_dict[user].add(i)
    
    return new_train_pairs, dict(new_train_dict), test_dict


def evaluate_by_group(model, test_dict, train_dict, n_items, edge_index, 
                      user_groups, k_list=[5, 10], device='cpu'):
    """Evaluate model with user group segmentation."""
    model.eval()
    
    with torch.no_grad():
        try:
            user_emb, item_emb = model(edge_index.to(device))
        except:
            user_emb, item_emb = model(None)
    
    user_emb = user_emb.detach().cpu().numpy()
    item_emb = item_emb.detach().cpu().numpy()
    
    # Results per group
    group_results = {group: defaultdict(list) for group in USER_GROUPS.keys()}
    group_results['Overall'] = defaultdict(list)
    
    # Map users to groups
    user_to_group = {}
    for group_name, users in user_groups.items():
        for u in users:
            user_to_group[u] = group_name
    
    for user, test_items in test_dict.items():
        if user >= len(user_emb):
            continue
        
        scores = user_emb[user] @ item_emb.T
        
        # Mask training items
        for item in train_dict.get(user, set()):
            if item < len(scores):
                scores[item] = -np.inf
        
        group = user_to_group.get(user, 'Cold')
        
        for k in k_list:
            top_k = np.argsort(scores)[::-1][:k]
            hits = len(set(top_k) & set(test_items))
            
            recall = hits / len(test_items) if len(test_items) > 0 else 0
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            hr = 1.0 if hits > 0 else 0.0
            
            group_results[group][f'recall@{k}'].append(recall)
            group_results[group][f'ndcg@{k}'].append(ndcg)
            group_results[group][f'hr@{k}'].append(hr)
            
            group_results['Overall'][f'recall@{k}'].append(recall)
            group_results['Overall'][f'ndcg@{k}'].append(ndcg)
            group_results['Overall'][f'hr@{k}'].append(hr)
    
    # Aggregate
    final_results = {}
    for group_name in list(USER_GROUPS.keys()) + ['Overall']:
        final_results[group_name] = {
            'n_users': len(group_results[group_name].get('recall@10', [])),
            'metrics': {k: np.mean(v) if v else 0 for k, v in group_results[group_name].items()}
        }
    
    return final_results


def train_model(model, data, epochs=30, batch_size=512, lr=0.001, neg_ratio=1, device='cpu'):
    """Train a CF model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_pairs, train_dict, test_dict = create_test_split(data['train_pairs'], data['train_dict'])
    edge_index = data['edge_index'].to(device)
    n_items = data['n_items']
    
    if len(train_pairs) == 0:
        return None, None, None
    
    # Segment users
    user_groups = segment_users_by_group(train_dict)
    
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
    
    # Evaluate by group
    group_results = evaluate_by_group(model, test_dict, train_dict, n_items, edge_index, user_groups, device=device)
    
    return group_results, user_groups, train_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate with user group segmentation')
    parser.add_argument('--graph', default='data/processed_graphs/reaction_weighted_graph.pt')
    parser.add_argument('--model', choices=['ngcf', 'simplex', 'directau'], default='simplex')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--neg-ratio', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("USER GROUP SEGMENTED EVALUATION")
    print("=" * 70)
    print(f"Graph: {args.graph}")
    print(f"Model: {args.model.upper()} | Epochs: {args.epochs} | Neg Ratio: {args.neg_ratio}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data = load_graph_data(args.graph)
    print(f"\nData: {data['n_users']} users, {data['n_items']} items, {len(data['train_pairs'])} pairs")
    
    # Show user distribution
    user_groups = segment_users_by_group(data['train_dict'])
    print("\nUser Group Distribution:")
    for group, users in user_groups.items():
        pct = 100 * len(users) / data['n_users'] if data['n_users'] > 0 else 0
        print(f"  {group}: {len(users)} users ({pct:.1f}%)")
    
    # Create model
    n_users, n_items = data['n_users'], data['n_items']
    if args.model == 'ngcf':
        model = NGCF(n_users, n_items, 64, 3).to(args.device)
    elif args.model == 'simplex':
        model = SimpleX(n_users, n_items, 64).to(args.device)
    elif args.model == 'directau':
        model = DirectAU(n_users, n_items, 64).to(args.device)
    
    print(f"\nTraining {args.model.upper()}...")
    group_results, _, _ = train_model(model, data, epochs=args.epochs, neg_ratio=args.neg_ratio, device=args.device)
    
    if group_results is None:
        print("Error: No training data")
        return
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS BY USER GROUP")
    print("=" * 70)
    
    results_table = []
    for group in ['Cold', 'Low', 'Medium', 'Active', 'Overall']:
        r = group_results.get(group, {})
        m = r.get('metrics', {})
        results_table.append({
            'Group': group,
            '#Users': r.get('n_users', 0),
            'R@5': f"{m.get('recall@5', 0):.4f}",
            'R@10': f"{m.get('recall@10', 0):.4f}",
            'NDCG@10': f"{m.get('ndcg@10', 0):.4f}",
            'HR@10': f"{m.get('hr@10', 0):.4f}",
        })
    
    print(tabulate(results_table, headers='keys', tablefmt='grid'))


if __name__ == '__main__':
    main()
