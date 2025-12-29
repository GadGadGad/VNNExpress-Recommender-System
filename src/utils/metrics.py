"""
Evaluation Metrics for Recommendation
=====================================
Recall@K, NDCG@K, HR@K, Precision@K
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Union
import torch


def compute_metrics(predictions: Union[Dict, np.ndarray, torch.Tensor],
                    test_data: Dict[int, List[int]],
                    train_dict: Dict[int, Set[int]],
                    k_list: List[int] = [10, 20, 50]) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, HR@K, Precision@K
    
    Args:
        predictions: Score matrix or dict {user: scores}
        test_data: Dict {user: [test_items]}
        train_dict: Dict {user: set(train_items)} - for masking
        k_list: List of K values
        
    Returns:
        Dict with metrics
    """
    results = defaultdict(list)
    max_k = max(k_list)
    
    for user in test_data:
        if user not in train_dict:
            continue
            
        # Ground truth
        gt_items = set(test_data[user])
        if len(gt_items) == 0:
            continue
            
        # Get scores
        if isinstance(predictions, dict):
            scores = predictions[user]
        elif isinstance(predictions, torch.Tensor):
            scores = predictions[user].cpu().numpy()
        else:
            scores = predictions[user]
            
        # Copy and mask training items
        scores_masked = np.array(scores).copy()
        for item in train_dict[user]:
            if item < len(scores_masked):
                scores_masked[item] = -np.inf
                
        # Get top-k items
        top_items = np.argsort(scores_masked)[::-1][:max_k]
        
        for k in k_list:
            top_k = top_items[:k]
            hits = len(set(top_k) & gt_items)
            
            # Recall@K
            recall = hits / len(gt_items)
            results[f'Recall@{k}'].append(recall)
            
            # NDCG@K
            dcg = sum([1.0 / np.log2(i + 2) 
                      for i, item in enumerate(top_k) if item in gt_items])
            idcg = sum([1.0 / np.log2(i + 2) 
                       for i in range(min(len(gt_items), k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            results[f'NDCG@{k}'].append(ndcg)
            
            # HR@K (Hit Rate)
            hr = 1.0 if hits > 0 else 0.0
            results[f'HR@{k}'].append(hr)
            
            # Precision@K
            precision = hits / k
            results[f'Precision@{k}'].append(precision)

            # MRR (Mean Reciprocal Rank) @ K
            # Find the rank of the first relevant item
            try:
                # np.where returns tuple of arrays, we want first index of first match
                first_hit_rank = np.where(np.isin(top_k, list(gt_items)))[0][0]
                mrr = 1.0 / (first_hit_rank + 1)
            except IndexError:
                # No hits in top_k
                mrr = 0.0
            results[f'MRR@{k}'].append(mrr)
    
    # Average
    avg_results = {}
    for key, values in results.items():
        avg_results[key] = np.mean(values) if len(values) > 0 else 0.0
        
    return avg_results


def print_metrics(metrics: Dict[str, float], epoch: int = None):
    """Pretty print metrics"""
    if epoch is not None:
        print(f"\n{'='*50}")
        print(f"Epoch {epoch} Evaluation")
        print('='*50)
    
    # Group by type
    recalls = sorted([(k, v) for k, v in metrics.items() if 'Recall' in k])
    ndcgs = sorted([(k, v) for k, v in metrics.items() if 'NDCG' in k])
    hrs = sorted([(k, v) for k, v in metrics.items() if 'HR' in k])
    
    if recalls:
        print("  Recall:  " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in recalls]))
    if ndcgs:
        print("  NDCG:    " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in ndcgs]))
    if hrs:
        print("  HR:      " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in hrs]))