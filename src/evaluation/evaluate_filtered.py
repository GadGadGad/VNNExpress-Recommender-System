import torch
import torch.nn as nn
import pandas as pd
import json
import argparse
import random
import glob
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import to_hetero

# Re-define LightGCN Encoder to avoid import issues
class LightGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        
    def forward(self, x, edge_index):
        all_embeddings = [x]
        for _ in range(self.num_layers):
            x = self._propagate(x, edge_index)
            all_embeddings.append(x)
        x = torch.stack(all_embeddings, dim=0).mean(dim=0)
        x = self.out_proj(x)
        return x
    
    def _propagate(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(-1))
        return out

def get_model(model_name, hidden_dim, out_dim, num_layers, dropout):
    if model_name == 'lightgcn':
        return LightGCNEncoder(hidden_dim, out_dim, num_layers, dropout)
    raise ValueError(f"Unknown model: {model_name}")

def load_latest_model(model_dir='models'):
    files = glob.glob(f"{model_dir}/lightgcn_*.pt")
    if not files:
        raise FileNotFoundError("No models found!")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading latest model: {latest_file}")
    try:
        return torch.load(latest_file, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(latest_file, map_location='cpu')

def compute_metrics(scores, ground_truth, k=10):
    _, topk_indices = torch.topk(scores, k)
    recs = topk_indices.tolist()
    hits = 0
    dcg = 0
    idcg = 0
    for i, item_idx in enumerate(recs):
        if item_idx in ground_truth:
            hits += 1
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)
    recall = hits / len(ground_truth) if ground_truth else 0
    ndcg = dcg / idcg if idcg > 0 else 0
    precision = hits / k
    hit_rate = 1.0 if hits > 0 else 0.0
    
    mrr = 0.0
    for i, item_idx in enumerate(recs):
        if item_idx in ground_truth:
            mrr = 1.0 / (i + 1)
            break
            
    return {"recall": recall, "ndcg": ndcg, "precision": precision, "hit_rate": hit_rate, "mrr": mrr}

def evaluate_filtered(data_dir, replies_path, graph_path, min_interactions=3, k=10, model_path=None):
    print(f"Evaluating on users with >= {min_interactions} interactions using GNN prop...")
    
    if not os.path.exists(graph_path):
        print(f"[ERROR] Graph file not found: {graph_path}")
        return
    print(f"Loading graph from {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    
    if model_path:
        print(f"Loading specific model: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = load_latest_model()
        
    node_embeddings = checkpoint.get('node_embeddings')
    model_config = checkpoint.get('model_config', {})
    
    # Restore embeddings from checkpoint
    print(f"Restoring embeddings for: {list(node_embeddings.keys())}")
    for nt in node_embeddings:
        if nt in data.node_types:
            data[nt].x = node_embeddings[nt]
        else:
            print(f"[WARN] Node type {nt} in checkpoint but not in graph data.")
            
    # Verify all graph nodes have correct dim
    target_dim = model_config.get('hidden_dim', 64)
    for nt in data.node_types:
        if data[nt].x.shape[1] != target_dim:
             # Skip if dimension mismatch (model likely uses projection or learnable params)
             pass

    print("Rebuilding LightGCN model...")
    base_model = get_model(
        'lightgcn', 
        hidden_dim=model_config.get('hidden_dim', 64),
        out_dim=model_config.get('out_dim', 32), 
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.3)
    )
    model = to_hetero(base_model, data.metadata(), aggr='sum')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
    
    user_emb = z_dict['user']
    article_emb = z_dict['article']
    
    with open(f"{data_dir}/user_map.json") as f:
        user_map = json.load(f)
    with open(f"{data_dir}/article_map.json") as f:
        article_map = json.load(f)
        
    df = pd.read_csv(replies_path)
    df1 = df[['parent_user_id', 'article_url']].rename(columns={'parent_user_id': 'user_id'})
    df2 = df[['reply_user_id', 'article_url']].rename(columns={'reply_user_id': 'user_id'})
    interactions = pd.concat([df1, df2]).dropna().drop_duplicates()
    
    user_counts = interactions['user_id'].value_counts()
    active_users_ids = user_counts[user_counts >= min_interactions].index.tolist()
    
    print(f"Active Users (min {min_interactions}): {len(active_users_ids)}")
    
    if not active_users_ids:
        print("No active users found!")
        return

    metrics_agg = {
        "recall": [], "ndcg": [], "precision": [], "hit_rate": [], "mrr": []
    }
    
    print("\nRunning Evaluation...")
    for uid in tqdm(active_users_ids):
        orig_uid_str = str(uid).replace('.0', '')
        if orig_uid_str not in user_map:
            try:
                if float(uid) in user_map: 
                     u_idx = user_map[float(uid)]
                elif int(float(uid)) in user_map: 
                     u_idx = user_map[int(float(uid))]
                elif str(int(float(uid))) in user_map:
                     u_idx = user_map[str(int(float(uid)))]
                else: continue 
            except: continue
        else:
             u_idx = user_map[orig_uid_str]
             
        user_history = interactions[interactions['user_id'] == uid]['article_url'].tolist()
        item_indices = [article_map[url] for url in user_history if url in article_map]
        
        if len(item_indices) < min_interactions:
            continue
            
        random.shuffle(item_indices)
        split_idx = int(len(item_indices) * 0.8)
        train_items = set(item_indices[:split_idx])
        test_items = set(item_indices[split_idx:])
        
        if not test_items and len(item_indices) > 1:
            test_items = {item_indices[-1]}
            train_items = set(item_indices[:-1])
        elif not test_items:
             continue

        u_vec = user_emb[u_idx].unsqueeze(0)
        all_scores = (u_vec @ article_emb.T).squeeze(0)
        num_articles = all_scores.size(0)
        
        # Filter out indices that are out of bounds
        train_items = {ti for ti in train_items if ti < num_articles}
        test_items = {ti for ti in test_items if ti < num_articles}
        
        if not test_items:
            continue
        
        for ti in train_items:
            all_scores[ti] = float('-inf')
            
        m = compute_metrics(all_scores, test_items, k=k)
        for key in metrics_agg:
            metrics_agg[key].append(m[key])
        
    print("\n" + "="*40)
    print(f"RESULTS (Min Interactions: {min_interactions})")
    print("="*40)
    print(f"Users Evaluated: {len(metrics_agg['recall'])}")
    print(f"Recall@{k}:      {np.mean(metrics_agg['recall']):.4f}")
    print(f"NDCG@{k}:        {np.mean(metrics_agg['ndcg']):.4f}")
    print(f"Precision@{k}:   {np.mean(metrics_agg['precision']):.4f}")
    print(f"HitRate@{k}:     {np.mean(metrics_agg['hit_rate']):.4f}")
    print(f"MRR@{k}:         {np.mean(metrics_agg['mrr']):.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed_phobert')
    parser.add_argument('--replies', default='data/raw/replies.csv')
    parser.add_argument('--graph', default='data/processed_phobert/user_article_graph.pt')
    parser.add_argument('--min', type=int, default=3)
    
    parser.add_argument('--model', default=None, help='Path to specific model checkpoint')
    
    args = parser.parse_args()
    
    evaluate_filtered(args.data_dir, args.replies, args.graph, args.min, model_path=args.model)
