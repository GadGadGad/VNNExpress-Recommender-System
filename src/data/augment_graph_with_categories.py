import torch
import pandas as pd
import json
import os
from torch_geometric.data import HeteroData
from pathlib import Path
from tqdm import tqdm
import argparse

def augment_with_categories():
    parser = argparse.ArgumentParser(description="G4: Category-Augmented Heterogeneous Graph Generator")
    parser.add_argument("--input-dir", default="data/processed/enhanced_v1", help="Input processed directory")
    parser.add_argument("--output-dir", default="data/processed/enhanced_v1", help="Output processed directory")
    args = parser.parse_args()

    print("=" * 60)
    print("G4: Category-Augmented Heterogeneous Graph Generator")
    print("=" * 60)
    
    # Paths
    raw_dir = Path('data/raw')
    processed_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not (processed_dir / 'user_map.json').exists():
        print(f"Error: Mappings not found in {processed_dir}")
        return

    # 1. Load mappings to ensure consistency
    print(f"Loading mappings and metadata from {processed_dir}...")
    with open(processed_dir / 'user_map.json', 'r') as f:
        user_map = json.load(f)
    with open(processed_dir / 'article_map.json', 'r') as f:
        article_map = json.load(f)
        
    # 2. Load Article Metadata
    articles_df = pd.read_csv(raw_dir / 'articles.csv')
    # Filter to articles in our graph
    articles_df = articles_df[articles_df['url'].isin(article_map.keys())]
    print(f"   → Found metadata for {len(articles_df):,} articles in the graph.")
    
    # 3. Process Categories
    unique_categories = articles_df['source_category'].dropna().unique().tolist()
    cat_map = {cat: i for i, cat in enumerate(unique_categories)}
    num_categories = len(cat_map)
    print(f"   → Found {num_categories} unique categories.")
    
    # 4. Load Base Graph
    # Prefer hetero graph if it exists, else use user-article
    base_graph_path = processed_dir / 'full_hetero_graph.pt'
    if not base_graph_path.exists():
        base_graph_path = processed_dir / 'user_article_graph.pt'
        
    print(f"Loading base graph from {base_graph_path}...")
    data = torch.load(base_graph_path, weights_only=False)
    
    # 5. Add Category Nodes
    # Using random init for category nodes, same dim as users
    hidden_dim = data['user'].x.shape[1]
    torch.manual_seed(42)
    data['category'].x = torch.randn(num_categories, hidden_dim)
    print(f"   → Added 'category' nodes with feature dim {hidden_dim}.")
    
    # 6. Create Article -> Category Edges (belongs_to)
    print("Creating Article-Category edges...")
    a_indices = []
    c_indices = []
    for _, row in articles_df.iterrows():
        url = row['url']
        cat = row['source_category']
        if pd.isna(cat): continue
        
        a_idx = article_map[url]
        c_idx = cat_map[cat]
        a_indices.append(a_idx)
        c_indices.append(c_idx)
        
    data['article', 'belongs_to', 'category'].edge_index = torch.tensor([a_indices, c_indices], dtype=torch.long)
    
    # 7. Create User -> Category Edges (interested_in)
    print("Inferring User-Category interest edges (using only training split)...")
    # User -> Article (comments) -> Category
    ua_edge_index = data['user', 'comments', 'article'].edge_index
    interactions = ua_edge_index.t().tolist()
    
    # The input graph (base_graph) ALREADY contains only training edges (filtered in convert_to_gnn.py).
    # So we can use ALL of them to infer user interests without leakage.
    train_interactions = interactions
    
    # Map article_idx to category_idx
    art_to_cat = {article_map[url]: cat_map[cat] 
                  for url, cat in zip(articles_df['url'], articles_df['source_category']) 
                  if not pd.isna(cat)}
    
    user_cat_interactions = []
    for u_idx, a_idx in train_interactions:
        if a_idx in art_to_cat:
            user_cat_interactions.append((u_idx, art_to_cat[a_idx]))
            
    # Deduplicate User-Category edges
    user_cat_edges = sorted(list(set(user_cat_interactions)))
    if user_cat_edges:
        u_src, c_dst = zip(*user_cat_edges)
        data['user', 'interested_in', 'category'].edge_index = torch.tensor([u_src, c_dst], dtype=torch.long)
    
    # 8. Add Reverse Edges for message passing
    print("Adding reverse edges...")
    # category -> article
    data['category', 'has_article', 'article'].edge_index = data['article', 'belongs_to', 'category'].edge_index.flip(0)
    # category -> user
    if user_cat_edges:
        data['category', 'attracts', 'user'].edge_index = data['user', 'interested_in', 'category'].edge_index.flip(0)
    
    # 9. Save
    output_path = output_dir / 'category_graph.pt'
    torch.save(data, output_path)
    print(f"\n[SUCCESS] Saved category-augmented graph to {output_path}")
    print(f"Final Graph Stats:")
    print(f"  Nodes: {data['user'].num_nodes} Users, {data['article'].num_nodes} Articles, {num_categories} Categories")
    
    return data

if __name__ == "__main__":
    augment_with_categories()
