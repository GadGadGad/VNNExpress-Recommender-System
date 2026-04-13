#!/usr/bin/env python3
"""
Evaluation utilities for Collaborative Filtering models.
Computes Recall, NDCG, HitRate, Precision, MAP, F1, MRR, and Entropy.
"""
import numpy as np
import torch
import pandas as pd


def load_item_categories(idx2item, csv_path='data/raw/articles.csv'):
    """Map item indices to their categories."""
    df = pd.read_csv(csv_path)
    url_to_cat = dict(zip(df['url'], df['source_category']))
    unique_cats = sorted(df['source_category'].unique().tolist())
    cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
    
    categories = []
    for i in range(len(idx2item)):
        url = idx2item[i]
        cat = url_to_cat.get(url, 'Other')
        categories.append(cat_to_id.get(cat, 0))
    return np.array(categories), len(unique_cats)

def compute_entropy(item_indices, item_categories, n_categories):
    """Compute Shannon Entropy of categorical distribution."""
    cats = item_categories[item_indices]
    counts = np.bincount(cats, minlength=n_categories)
    probs = counts / (len(item_indices) + 1e-9)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0


def evaluate(model, test_dict, train_dict, n_items, edge_index, k_list=[1, 5, 10, 50], device='cpu', adj_norm=None, 
             re_ranker=None, rerank_strategy='none', eval_protocol='full', cold_users=None, edge_index_dict=None):
    """
    Evaluate model with multiple protocols.
    
    eval_protocol:
        - 'full': Full ranking over all items (hardest, most realistic)
        - 'loo100': Leave-one-out + 100 random negatives (common in papers)
        - 'cold': Evaluate only on cold-start users
    """
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'forward'):
            forward_args = model.forward.__code__.co_varnames
            
            if 'edge_index_dict' in forward_args and edge_index_dict is not None:
                user_emb, item_emb = model(None, edge_index_dict)
            elif 'adj_norm' in forward_args and adj_norm is not None:
                kwargs = {}
                if 'item_content' in forward_args:
                    kwargs['item_content'] = getattr(model, 'item_content', None)
                if 'semantic_ids' in forward_args:
                    kwargs['semantic_ids'] = getattr(model, 'semantic_ids', None)
                user_emb, item_emb = model(adj_norm, **kwargs)
            elif 'edge_index' in forward_args:
                user_emb, item_emb = model(edge_index.to(device))
            elif hasattr(model, 'adj_norm'):
                user_emb, item_emb = model()
            else:
                user_emb, item_emb = model()
        else:
            user_emb = model.user_embedding.weight
            item_emb = model.item_embedding.weight
    
    max_k = max(k_list)
    results = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg', 'hitrate', 'precision', 'map', 'f1'] for k in k_list}
    results['mrr'] = []
    
    # Choose which users to evaluate based on protocol
    if eval_protocol == 'cold' and cold_users is not None:
        eval_users = {u: items for u, items in test_dict.items() if u in cold_users}
        print(f"  Cold-start eval: {len(eval_users)} users")
    else:
        eval_users = test_dict
    
    for user, test_items in eval_users.items():
        # Ensure test_items is a set for set operations
        test_items = set(test_items) if isinstance(test_items, list) else test_items
        
        if user >= user_emb.size(0):
            continue
            
        train_items = train_dict.get(user, set())
        u_emb = user_emb[user].unsqueeze(0)
        
        if eval_protocol == 'loo100':
            # Leave-One-Out + 100 negatives: sample 100 neg items + all positive items
            all_items = set(range(n_items)) - train_items - test_items
            neg_samples = np.random.choice(list(all_items), min(100, len(all_items)), replace=False)
            candidate_items = list(test_items) + list(neg_samples)
            
            # Score only these candidates
            candidate_emb = item_emb[candidate_items]
            scores = torch.mm(u_emb, candidate_emb.t()).squeeze()
            
            # Map back to original indices
            _, topk_local = torch.topk(scores, min(max_k, len(candidate_items)))
            topk_candidates = [candidate_items[i] for i in topk_local.cpu().numpy()]
        else:
            # Full ranking
            scores = torch.mm(u_emb, item_emb.t()).squeeze()
            
            # Mask train items
            for item in train_items:
                if item < scores.size(0):
                    scores[item] = -float('inf')
            
            _, topk = torch.topk(scores, 100)
            topk_candidates = topk.cpu().numpy().tolist()
        
        # Apply re-ranking if specified
        if rerank_strategy == 'mmr' and re_ranker is not None:
            topk_list = re_ranker.mmr_rerank(item_emb, scores if eval_protocol == 'full' else None, top_k=max_k)
        elif rerank_strategy == 'calib' and re_ranker is not None:
            user_history = list(train_items) if train_items else []
            score_arr = scores.cpu().numpy() if eval_protocol == 'full' else np.zeros(n_items)
            topk_list = re_ranker.calibrate(score_arr, user_history, top_k=max_k)
        else:
            topk_list = topk_candidates[:max_k]
        
        # MRR
        mrr = 0.0
        for i, item in enumerate(topk_list):
            if item in test_items:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'].append(mrr)
        
        # Metrics at each k
        for k in k_list:
            topk_k = set(topk_list[:k])
            hits = len(topk_k & test_items)
            
            prec = hits / k
            rec = hits / len(test_items)
            results[f'recall@{k}'].append(rec)
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
            results[f'precision@{k}'].append(prec)
            
            # F1@k
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            results[f'f1@{k}'].append(f1)
            
            # NDCG@k
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(topk_list[:k]) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
            results[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0)
            
            # mAP@k (Average Precision): sum of precision at each hit / min(k, num_positives)
            ap = 0.0
            n_hits = 0
            for i, item in enumerate(topk_list[:k]):
                if item in test_items:
                    n_hits += 1
                    ap += n_hits / (i + 1)
            ap = ap / min(k, len(test_items)) if len(test_items) > 0 else 0
            results[f'map@{k}'].append(ap)
        
        if re_ranker is not None:
             ent = compute_entropy(topk_list, re_ranker.item_categories, re_ranker.n_categories)
             results.setdefault('entropy', []).append(ent)
    
    return {k: np.mean(v) for k, v in results.items()}
