import torch
import os
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData

def get_graph_stats(path, name):
    print(f"\nAnalyzing {name} at {path}...")
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        return None
        
    data = torch.load(path, weights_only=False)
    
    # Handle dict vs direct objects
    if isinstance(data, dict):
        splits = data.get('splits', {})
        graph = data.get('graph')
        n_users = data.get('n_users', 0)
        n_items = data.get('n_items', 0)
        n_categories = data.get('n_categories', 0)
    else:
        graph = data
        n_users = graph['user'].num_nodes if 'user' in graph.node_types else 0
        n_items = graph['article'].num_nodes if 'article' in graph.node_types else 0
        n_categories = graph['category'].num_nodes if 'category' in graph.node_types else 0
        splits = {}

    stats = {
        'Tier': name,
        'Users': n_users,
        'Items': n_items,
        'Categories': n_categories,
        'Edges': {},
        'Density': 0,
        'Avg Degree': 0
    }

    if isinstance(graph, HeteroData):
        for edge_type, edge_index in graph.edge_index_dict.items():
            # Skip reverse edges for counting unique interactions
            if 'rev_' in edge_type[1] or 'rev-' in edge_type[1]:
                continue
            if edge_type[1].endswith('_to') and edge_type[0] == 'article':
                continue
            stats['Edges'][str(edge_type)] = edge_index.shape[1]
    
    # Fallback for dict-based G3
    if not stats['Edges'] and isinstance(data, dict):
        if 'user_category_edge_index' in data:
             stats['Edges']['(user, interested_in, category)'] = data['user_category_edge_index'].shape[1]
        elif 'edge_index' in data:
             stats['Edges']['Total'] = data['edge_index'].shape[1]

    # Calculate density for the main relation
    main_edges = 0
    if "('user', 'comments', 'article')" in stats['Edges']:
        main_edges = stats['Edges']["('user', 'comments', 'article')"]
    elif "(user, interested_in, category)" in stats['Edges']:
        main_edges = stats['Edges']["(user, interested_in, category)"]
    elif 'Total' in stats['Edges']:
        main_edges = stats['Edges']['Total']

    if n_users > 0:
        if n_items > 0:
            denom = n_users * n_items
            stats['Density'] = (main_edges / denom) * 100
            stats['Avg Degree'] = main_edges / n_users
        elif n_categories > 0:
            denom = n_users * n_categories
            stats['Density'] = (main_edges / denom) * 100
            stats['Avg Degree'] = main_edges / n_users

    return stats

def main():
    base_dir = "data/processed"
    graphs = [
        ("strict_g1/user_article_graph.pt", "G1 (Bipartite)"),
        ("strict_g2/full_hetero_graph.pt", "G2 (Full Hetero)"),
        ("strict_g3/category_graph.pt", "G3 (User-Category)")
    ]
    
    results = []
    for rel_path, name in graphs:
        full_path = os.path.join(base_dir, rel_path)
        res = get_graph_stats(full_path, name)
        if res:
            results.append(res)
            
    print("\n" + "="*50)
    print("BENCHMARK EDA SUMMARY")
    print("="*50)
    for r in results:
        print(f"\n{r['Tier']}:")
        print(f"  Nodes: {r['Users']} Users, {r['Items']} Articles, {r['Categories']} Categories")
        print(f"  Edges:")
        for et, count in r['Edges'].items():
            print(f"    - {et}: {count:,}")
        print(f"  Density: {r['Density']:.4f}%")
        print(f"  Avg Degree: {r['Avg Degree']:.2f}")

if __name__ == "__main__":
    main()
