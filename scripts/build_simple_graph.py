#!/usr/bin/env python3
"""
Build Simplified Graph Variants (STRICT FILTERING)
===================================================
Creates G2' with simple binary weights (w=1) AND strict k-core filtering (min=2).
This version results in a smaller, denser graph for comparison.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json

def build_strict_simple_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Build G2' - Heterogeneous graph with SIMPLE binary weights (w=1) 
    AND strict iterative k-core filtering.
    """
    print("=" * 60)
    print(f"Building G2' - Strict Filtered (min={min_interactions})")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)
    
    # Extract IDs
    def clean_id(val):
        try:
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                return None
            return str(int(float(val)))
        except:
            return str(val)
    
    interactions = []
    for _, row in replies.iterrows():
        p_id = clean_id(row['parent_user_id'])
        r_id = clean_id(row['reply_user_id'])
        url = row['article_url']
        if p_id: interactions.append({'user_id': p_id, 'article_url': url})
        if r_id: interactions.append({'user_id': r_id, 'article_url': url})
    
    int_df = pd.DataFrame(interactions).dropna()
    
    # Iterative K-core
    prev_len = 0
    while len(int_df) != prev_len:
        prev_len = len(int_df)
        u_counts = int_df['user_id'].value_counts()
        a_counts = int_df['article_url'].value_counts()
        valid_u = u_counts[u_counts >= min_interactions].index
        valid_a = a_counts[a_counts >= min_interactions].index
        int_df = int_df[int_df['user_id'].isin(valid_u) & int_df['article_url'].isin(valid_a)].copy()
    
    # Mappings
    user_map = {u: i for i, u in enumerate(int_df['user_id'].unique())}
    article_map = {a: i for i, a in enumerate(int_df['article_url'].unique())}
    
    # Build data
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
    
    data = HeteroData()
    data['user'].x = torch.randn(len(user_map), 64)
    data['article'].x = torch.randn(len(article_map), 64)
    
    src = torch.tensor(int_df['user_id'].map(user_map).values, dtype=torch.long)
    dst = torch.tensor(int_df['article_url'].map(article_map).values, dtype=torch.long)
    data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
    data['user', 'comments', 'article'].edge_weight = torch.ones(len(src), dtype=torch.float32)
    
    data = T.ToUndirected()(data)
    
    save_path = output_dir / 'full_hetero_simple_strict.pt'
    torch.save(data, save_path)
    
    print(f"[OK] Saved Strict Graph: {save_path}")
    print(f"Stats: {len(user_map)} users, {len(article_map)} articles, {len(int_df)} edges")
    return data

if __name__ == "__main__":
    build_strict_simple_graph('data/raw/articles.csv', 'data/raw/replies.csv', 'data/processed')
