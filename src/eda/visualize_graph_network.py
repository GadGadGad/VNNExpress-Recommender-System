import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def visualize_subgraph(data_path, output_path, num_users=15, max_articles=20):
    print(f"Loading graph from {data_path}...")
    try:
        data = torch.load(data_path, weights_only=False)
    except TypeError:
        # Fallback for older torch versions
        data = torch.load(data_path)
    
    G = nx.Graph()
    
    # 1. Sample users with high interactions
    # Correct edge type for interaction is ('user', 'comments', 'article')
    rel_type = ('user', 'comments', 'article')
    if rel_type not in data.edge_types:
        # Fallback to check other possible names
        print(f"Available edge types: {data.edge_types}")
        return

    edge_index = data[rel_type].edge_index
    user_indices, counts = torch.unique(edge_index[0], return_counts=True)
    top_user_indices = user_indices[torch.argsort(counts, descending=True)[:num_users]].tolist()
    
    # 2. Add User nodes
    for u_idx in top_user_indices:
        G.add_node(f"User_{u_idx}", type='user', color='#1f77b4')
    
    # 3. Add Interaction edges and Article nodes
    articles_found = set()
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i].item()
        a = edge_index[1, i].item()
        if u in top_user_indices and len(articles_found) < max_articles:
            if a not in articles_found:
                articles_found.add(a)
                G.add_node(f"Art_{a}", type='article', color='#2ca02c')
            G.add_edge(f"User_{u}", f"Art_{a}", weight=1, type='interaction')
            
    # 4. Add Social edges (Replied_to)
    social_rel = ('user', 'replied_to', 'user')
    if social_rel in data.edge_types:
        social_edge_index = data[social_rel].edge_index
        for i in range(social_edge_index.shape[1]):
            u1 = social_edge_index[0, i].item()
            u2 = social_edge_index[1, i].item()
            if u1 in top_user_indices and u2 in top_user_indices:
                G.add_edge(f"User_{u1}", f"User_{u2}", weight=2, type='social')

    # 5. Visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Extract node attributes
    node_colors = [data['color'] for node, data in G.nodes(data=True)]
    node_sizes = [800 if data['type'] == 'user' else 600 for node, data in G.nodes(data=True)]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    
    # Draw edges with different styles
    interaction_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'interaction']
    social_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'social']
    
    nx.draw_networkx_edges(G, pos, edgelist=interaction_edges, width=1.0, alpha=0.4, edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=social_edges, width=2.5, alpha=0.7, edge_color='#d62728', style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', font_weight='bold')
    
    plt.title("Heterogeneous News-Social Graph (Subset)", fontsize=15, pad=20)
    
    # Legend with larger fonts
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='User', markerfacecolor='#1f77b4', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Article', markerfacecolor='#2ca02c', markersize=15),
        Line2D([0], [0], color='gray', lw=2, label='Read Interaction', alpha=0.6),
        Line2D([0], [0], color='#d62728', lw=3, label='Social Reply', ls='--')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=14, markerscale=1.2)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Graph visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="plots/graph_network_topology.png")
    args = parser.parse_args()
    
    visualize_subgraph(args.data_path, args.output_path)
