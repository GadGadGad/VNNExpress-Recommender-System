import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.content_based import ContentBasedRecommender

def get_metrics(topk_items, ground_truth, k_list=[1, 5, 10, 50]):
    """
    Calculate Recall@K, NDCG@K, Precision@K, mAP@K for multiple k values.
    topk_items: List of item IDs derived from TopK (should be at least max(k_list))
    ground_truth: Set of relevant item IDs
    """
    results = {}
    
    for k in k_list:
        topk_k = topk_items[:k]
        hits = sum(1 for item in topk_k if item in ground_truth)
        
        # Recall@k
        results[f'recall@{k}'] = hits / len(ground_truth) if len(ground_truth) > 0 else 0
        
        # Precision@k
        results[f'precision@{k}'] = hits / k
        
        # NDCG@k
        dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(topk_k) if item in ground_truth)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
        results[f'ndcg@{k}'] = dcg / idcg if idcg > 0 else 0
        
        # mAP@k (Average Precision)
        ap = 0.0
        n_hits = 0
        for i, item in enumerate(topk_k):
            if item in ground_truth:
                n_hits += 1
                ap += n_hits / (i + 1)
        results[f'map@{k}'] = ap / min(k, len(ground_truth)) if len(ground_truth) > 0 else 0
        
        # HitRate@k
        results[f'hitrate@{k}'] = 1.0 if hits > 0 else 0.0
    
    return results

def load_pretrained_embeddings(embedding_type, device='cpu'):
    """Load embeddings from checkpoints"""
    base_path = Path("checkpoints")
    path_map = {
        'tfidf': base_path / 'tfidf_article_embeddings.pt',
        'phobert': base_path / 'phobert_article_embeddings.pt',
        'vndoc': base_path / 'vndoc_article_embeddings.pt',
        'bge-m3': base_path / 'bge-m3_article_embeddings.pt',
        'vn-sbert': base_path / 'vietnamese-sbert_article_embeddings.pt',
        'e5-base': base_path / 'e5-base_article_embeddings.pt',
        'e5-large': base_path / 'e5-large_article_embeddings.pt'
    }
    
    if embedding_type == 'random':
        return None
        
    path = path_map.get(embedding_type)
    if path and path.exists():
        print(f"  Loading {embedding_type} from {path}...")
        return torch.load(path, map_location=device)
    else:
        print(f"  Warning: Embedding {embedding_type} not found at {path}")
        return None

def load_split_data(data_path):
    print(f"Loading split data from {data_path}...")
    split_path = Path(data_path) / 'graph_with_negatives.pt'
    data = torch.load(split_path, weights_only=False)
    print(f"  Debug: Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"  Debug: Keys: {list(data.keys())}")
    else:
        print(f"  Debug: Dir: {dir(data)[:20]}") # First 20 attrs

    
    train_dict = {}
    test_dict = {}
    n_users = 0
    n_items = 0
    
    # Logic to extract train/test pairs (handling wrapper dicts)
    if isinstance(data, dict) and 'splits' in data:
        print("  Debug: Found 'splits' key.")
        splits_wrapper = data['splits']
        if isinstance(splits_wrapper, dict):
             print(f"  Debug: Splits keys: {list(splits_wrapper.keys())}")
        
        num_users = data.get('num_users', 0)
        num_items = data.get('num_articles', 0)
        
        if hasattr(data, 'num_users') and isinstance(data.num_users, int): n_users = data.num_users
        elif 'num_users' in data: n_users = data['num_users']
        
        if hasattr(data, 'num_articles') and isinstance(data.num_articles, int): n_items = data.num_articles
        elif 'num_articles' in data: n_items = data['num_articles']
        
        # Test Split
        if 'test' in splits_wrapper:
            t = splits_wrapper['test']
            
            edges = None
            if isinstance(t, dict) and 'edge_index' in t: edges = t['edge_index']
            elif hasattr(t, 'edge_index'): edges = t.edge_index
            
            if edges is not None:
                for i in range(edges.shape[1]):
                    u, v = edges[0, i].item(), edges[1, i].item()
                    if u not in test_dict: test_dict[u] = set()
                    test_dict[u].add(v)
            elif isinstance(t, dict) and 'pos_users' in t and 'pos_articles' in t:
                # Handle Tensor Format
                us = t['pos_users']
                vs = t['pos_articles']
                for i in range(len(us)):
                     u, v = us[i].item(), vs[i].item()
                     if u not in test_dict: test_dict[u] = set()
                     test_dict[u].add(v)
        
        # Train Split (for history)
        if 'train' in splits_wrapper:
             t = splits_wrapper['train']
             edges = None
             if isinstance(t, dict) and 'edge_index' in t: edges = t['edge_index']
             elif hasattr(t, 'edge_index'): edges = t.edge_index
             elif isinstance(t, dict) and 'pos_edge_label_index' in t: edges = t['pos_edge_label_index']
                 
             if edges is not None:
                for i in range(edges.shape[1]):
                    u, v = edges[0, i].item(), edges[1, i].item()
                    if u not in train_dict: train_dict[u] = []
                    train_dict[u].append(v)
             elif isinstance(t, dict) and 'pos_users' in t and 'pos_articles' in t:
                # Handle Tensor Format
                us = t['pos_users']
                vs = t['pos_articles']
                for i in range(len(us)):
                     u, v = us[i].item(), vs[i].item()
                     if u not in train_dict: train_dict[u] = []
                     train_dict[u].append(v)
    
    # Check for HeteroData (PyG)
    # If splits failed, try graph masks
    if not train_dict and isinstance(data, dict) and 'graph' in data:
        print("  Debug: checking 'graph' object...")
        target_data = data['graph']
        if hasattr(target_data, 'edge_index_dict') or (isinstance(target_data, dict) and 'edge_index_dict' in target_data):
            edge_index_dict = target_data.edge_index_dict if hasattr(target_data, 'edge_index_dict') else target_data.get('edge_index_dict')
            if edge_index_dict and ('user', 'comments', 'article') in edge_index_dict:
                 store = target_data['user', 'comments', 'article']
                 all_edges = store.edge_index
                 # ... same logic as before ...
                 if hasattr(store, 'train_mask') and store.train_mask is not None:
                     print("  Debug: Using Train Mask from Graph")
                     train_edges = all_edges[:, store.train_mask]
                     for i in range(train_edges.shape[1]):
                        u, v = train_edges[0, i].item(), train_edges[1, i].item()
                        if u not in train_dict: train_dict[u] = []
                        train_dict[u].append(v)

                 if hasattr(store, 'test_mask') and store.test_mask is not None:
                     print("  Debug: Using Test Mask from Graph")
                     test_edges = all_edges[:, store.test_mask]
                     for i in range(test_edges.shape[1]):
                        u, v = test_edges[0, i].item(), test_edges[1, i].item()
                        if u not in test_dict: test_dict[u] = set()
                        test_dict[u].add(v)
                        
    # Refine n_users/n_items if 0
    if n_users == 0 and train_dict:
        n_users = max(train_dict.keys()) + 1
    
    print(f"  Loaded {len(train_dict)} train users, {len(test_dict)} test users.")
    return train_dict, test_dict, n_users, n_items
        
    print(f"  Loaded {len(train_dict)} train users, {len(test_dict)} test users.")
    return train_dict, test_dict, n_users, n_items

def benchmark(embedding_type, data_path, eval_protocol='full', cold_users=None, device='cuda'):
    print(f"\n--- Benchmarking {embedding_type} (Protocol: {eval_protocol}) ---")
    
    # 1. Load Data
    train_dict, test_dict, n_cfg_users, n_cfg_items = load_split_data(data_path)
    
    if not test_dict:
        print("  Error: No test data found.")
        return 0, 0, 0
        
    # 2. Load Embeddings
    item_emb = load_pretrained_embeddings(embedding_type, device)
    
    # Handle Random Case
    embedding_dim = 64
    if item_emb is not None:
        embedding_dim = item_emb.shape[1]
        n_items = item_emb.shape[0]
        # Ensure we cover all items
        if n_cfg_items > n_items:
             n_items = n_cfg_items
    else:
        # Random
        n_items = n_cfg_items if n_cfg_items > 0 else 3645 # Fallback
        item_emb = torch.randn(n_items, embedding_dim).to(device)
    
    item_emb = item_emb.to(device)

    # 3. Setup Recommender
    print("  Computing User Profiles...")
    user_profiles = {}
    for u, history in train_dict.items():
        if not history: 
            user_profiles[u] = torch.zeros(embedding_dim).to(device)
            continue
            
        hist_tensor = torch.tensor(history).to(device)
        hist_tensor = hist_tensor[hist_tensor < item_emb.shape[0]] 
        if len(hist_tensor) == 0:
             user_profiles[u] = torch.zeros(embedding_dim).to(device)
             continue
             
        embs = item_emb[hist_tensor]
        user_profiles[u] = torch.mean(embs, dim=0)

    # 4. Evaluate
    k_list = [1, 5, 10, 50]
    max_k = max(k_list)
    all_results = {f'{m}@{k}': [] for m in ['recall', 'ndcg', 'precision', 'map', 'hitrate'] for k in k_list}
    aucs = []
    
    # Choose which users to evaluate based on protocol
    if eval_protocol == 'cold' and cold_users is not None:
        eval_users = {u: items for u, items in test_dict.items() if u in cold_users}
        print(f"  Cold-start eval: {len(eval_users)} users")
    else:
        eval_users = test_dict
    
    print(f"  Evaluating ({eval_protocol})...")
    
    # Batchified Evaluation? Or Per-User loop?
    # Per-user loop on GPU is fast enough for 3k users
    
    for u, test_items in tqdm(eval_users.items()):
        if u not in user_profiles: continue
            
        u_prof = user_profiles[u].unsqueeze(0) # [1, dim]
        train_items = set(train_dict.get(u, []))
        
        if eval_protocol == 'loo100':
            # Leave-One-Out (Sampled) Protocol
            target_item = list(test_items)[0]
            all_negatives = set(range(item_emb.shape[0])) - train_items - set(test_items)
            neg_samples = np.random.choice(list(all_negatives), min(100, len(all_negatives)), replace=False)
            candidate_items = [target_item] + list(neg_samples)
            
            candidate_emb = item_emb[candidate_items]
            scores = torch.mm(u_prof, candidate_emb.t()).squeeze()
            
            _, topk_local = torch.topk(scores, min(max_k, len(candidate_items)))
            topk_list = [candidate_items[i] for i in topk_local.cpu().numpy()]
            metrics = get_metrics(topk_list, {target_item}, k_list)
        else:
            # Full Ranking Protocol
            scores = torch.mm(u_prof, item_emb.t()).squeeze()
            for item in train_items:
                if item < scores.shape[0]: scores[item] = -float('inf')
            
            if len(test_items) == 0: continue
            _, topk = torch.topk(scores, min(max_k, scores.shape[0]))
            topk_list = topk.cpu().tolist()
            metrics = get_metrics(topk_list, test_items, k_list)
            
            # AUC (Only for full ranking)
            valid_indices = scores > -float('inf')
            y_true = torch.zeros(scores.shape[0], device=device)
            test_tensor = torch.tensor(list(test_items), device=device)
            test_tensor = test_tensor[test_tensor < scores.shape[0]]
            y_true[test_tensor] = 1.0
            
            try:
                from sklearn.metrics import roc_auc_score
                y_true_np = y_true[valid_indices].cpu().numpy()
                y_score_np = scores[valid_indices].cpu().numpy()
                if np.sum(y_true_np) > 0:
                    aucs.append(roc_auc_score(y_true_np, y_score_np))
            except: pass
        
        # Accumulate metrics
        for key, val in metrics.items():
            all_results[key].append(val)
        
    # Compute averages
    avg_results = {k: np.mean(v) if v else 0.0 for k, v in all_results.items()}
    avg_results['auc'] = np.mean(aucs) if aucs else 0.0
    
    print(f"  Result for {embedding_type}:")
    for k in k_list:
        print(f"  @{k}: R={avg_results[f'recall@{k}']:.4f} N={avg_results[f'ndcg@{k}']:.4f} H={avg_results[f'hitrate@{k}']:.4f} P={avg_results[f'precision@{k}']:.4f} mAP={avg_results[f'map@{k}']:.4f}")
    print(f"  AUC: {avg_results['auc']:.4f}")
    
    return avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/strict_g2")
    parser.add_argument("--model", choices=['random', 'tfidf', 'phobert', 'vn-sbert', 'bge-m3', 'e5-base', 'all'], default='all')
    parser.add_argument("--eval-protocol", choices=['full', 'loo100', 'cold'], default='full')
    parser.add_argument("--output", type=str, default=None, help="Save metrics to JSON")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Identification of cold-start users for 'cold' protocol
    cold_users = None
    if args.eval_protocol == 'cold':
        train_dict, _, _, _ = load_split_data(args.data_path)
        cold_users = {u for u, items in train_dict.items() if len(items) <= 3}
    
    if args.model == 'all':
        embeddings_to_test = ['random', 'tfidf', 'phobert', 'vn-sbert', 'bge-m3', 'e5-base']
    else:
        embeddings_to_test = [args.model]
    
    print(f"Benchmarking Content-Based Filtering on {args.data_path} (Protocol: {args.eval_protocol})")
    
    all_results = {}
    for emb in embeddings_to_test:
        metrics = benchmark(emb, args.data_path, args.eval_protocol, cold_users, device)
        metrics['protocol'] = args.eval_protocol
        all_results[emb] = metrics
    
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # If running a single model, save its metrics directly to match CF format
        if len(embeddings_to_test) == 1:
            save_data = all_results[embeddings_to_test[0]]
        else:
            save_data = all_results
            
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Saved results to {output_path}")

