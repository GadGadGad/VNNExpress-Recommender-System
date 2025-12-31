#!/usr/bin/env python3
"""
Analyze Embedding Structure (Over-smoothing & Over-uniformity)
==============================================================

Calculates Mean Average Distance (MAD) and visualizes t-SNE.
Supports two modes:
1. Compare multiple models (default)
2. Analyze layers of a single model (to check over-smoothing)

Usage:
    # Mode 1: Compare Models
    python scripts/analyze_embeddings.py --models MA-HGN=path/to/model.pt LightGCN=path/to/model.pt

    # Mode 2: Layer Analysis
    python scripts/analyze_embeddings.py --mode layers --model-path path/to/model.pt --data-path data/processed/strict_g2 --layers 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import scipy.sparse as sp

def compute_mad(embeddings):
    """
    Compute Mean Average Distance (MAD) efficiently.
    MAD = mean(1 - cosine_similarity) over all pairs.
    """
    # Normalize
    emb = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    
    # Pairwise cosine distance = 1 - cosine_similarity
    # To save memory on large N, we can sample or loop, but here N=2000 is small.
    dists = cosine_distances(emb)
    
    # Mask diagonal (distance 0)
    np.fill_diagonal(dists, np.nan)
    
    # Mean of off-diagonal elements
    mad = np.nanmean(dists)
    return mad

def visualize_tsne(embeddings_dict, output_path="tsne_comparison.png", n_samples=2000, title_prefix=""):
    """
    Visualize t-SNE for multiple models/layers side-by-side.
    """
    n_models = len(embeddings_dict)
    cols = min(n_models, 4)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()
    
    print(f"Generating t-SNE for {n_models} embeddings (samples={n_samples})...")
    
    for i, (name, emb) in enumerate(embeddings_dict.items()):
        ax = axes[i]
        
        # Random sample
        N = emb.shape[0]
        if N > n_samples:
            indices = np.random.choice(N, n_samples, replace=False)
            emb_subset = emb[indices]
        else:
            emb_subset = emb
            
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        emb_2d = tsne.fit_transform(emb_subset.cpu().numpy())
        
        # Plot
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, alpha=0.6, edgecolors='none')
        ax.set_title(f"{title_prefix}{name}", fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    # Hide unused axes
    for i in range(n_models, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def load_base_embedding(ckpt_path):
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Extract user embeddings
    if 'user_embedding.weight' in state_dict:
        return state_dict['user_embedding.weight']
    elif 'E_u_0' in state_dict: # LightGCL / LightGCN
        return state_dict['E_u_0']
    elif 'gu_0' in state_dict: # MA-HGN
        return state_dict['gu_0']
    else:
        n_users = checkpoint['n_users']
        for k, v in state_dict.items():
            if v.shape[0] == n_users and v.ndim == 2:
                print(f"  Guessing {k} is user embedding.")
                return v
        raise ValueError(f"Could not locate user embeddings in {ckpt_path}")

def compute_normalized_adjacency(n_users, n_items, train_pairs):
    """Simplified adjacency computation for LightGCN propagation."""
    print(f"Constructing Adjacency Matrix ({n_users} users, {n_items} items)...")
    row = np.array([u for u, i in train_pairs])
    col = np.array([i for u, i in train_pairs])
    data = np.ones(len(train_pairs), dtype=np.float32)
        
    R = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))
    
    # Construct (N+M)x(N+M) matrix
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    
    adj_mat = adj_mat.tocoo()
    
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def load_graph_data(data_path):
    print(f"Loading graph data from {data_path}...")
    path = Path(data_path) / 'user_article_graph.pt'
    if not path.exists():
        raise FileNotFoundError(f"Graph data not found at {path}")
    
    data = torch.load(path, weights_only=False)
    # Check if 'train_pairs' is in data (for dict or HeteroData wrapper)
    if isinstance(data, dict):
        return data['n_users'], data['n_items'], data['train_pairs']
    else:
        # Fallback for simple object (assumes structure)
        return data.n_users, data.n_items, data.train_pairs

def propagate_layers(E_u_0, E_i_0, adj_norm, n_layers):
    """
    Simulate LightGCN propagation: E(k+1) = D^-0.5 A D^-0.5 E(k)
    Returns list of user embeddings at each layer [E0, E1, E2...]
    """
    print(f"Propagating {n_layers} layers...")
    all_embeddings = torch.cat([E_u_0, E_i_0], dim=0)
    layer_embeddings = [E_u_0] # Layer 0
    
    current_emb = all_embeddings
    n_users = E_u_0.shape[0]
    
    for k in range(1, n_layers + 1):
        if adj_norm.device != current_emb.device:
            adj_norm = adj_norm.to(current_emb.device)
            
        # Sparse MM
        current_emb = torch.sparse.mm(adj_norm, current_emb)
        E_u_k = current_emb[:n_users]
        layer_embeddings.append(E_u_k)
        print(f"  Captured Layer {k}")
        
    return layer_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='List of model paths in format name=path (Mode 1)')
    parser.add_argument('--mode', choices=['compare', 'layers'], default='compare', help='Analysis mode')
    
    # Layer Analysis Args
    parser.add_argument('--model-path', help='Path to single model checkpoint (Mode 2)')
    parser.add_argument('--data-path', help='Path to processed graph data (Mode 2)')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers to propagate (Mode 2)')
    
    parser.add_argument('--output', default='results/analysis.png')
    parser.add_argument('--csv-output', default='results/mad_stats.csv')
    args = parser.parse_args()
    
    embeddings_dict = {}
    
    if args.mode == 'compare':
        if not args.models:
            print("Error: --models required for compare mode.")
            return
            
        for item in args.models:
            name, path = item.split('=')
            try:
                emb = load_base_embedding(path)
                embeddings_dict[name] = emb
            except Exception as e:
                print(f"Error loading {name}: {e}")
                
    elif args.mode == 'layers':
        if not args.model_path or not args.data_path:
            print("Error: --model-path and --data-path required for layers mode.")
            return
            
        # 1. Load Base Embeddings
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        sd = checkpoint['model_state_dict']
        
        # Try to find E_i_0 as well
        if 'item_embedding.weight' in sd:
            E_i_0 = sd['item_embedding.weight']
        elif 'E_i_0' in sd:
            E_i_0 = sd['E_i_0']
        else:
            print("Warning: Could not find item embedding (E_i_0). Cannot propagate.")
            return

        E_u_0 = load_base_embedding(args.model_path)
        
        # 2. Load Graph & Compute Adjacency
        n_users, n_items, train_pairs = load_graph_data(args.data_path)
        adj_norm = compute_normalized_adjacency(n_users, n_items, train_pairs)
        
        # 3. Propagate
        layers = propagate_layers(E_u_0, E_i_0, adj_norm, args.layers)
        
        for k, emb in enumerate(layers):
            embeddings_dict[f'Layer_{k}'] = emb
            
    # Compute MAD
    mad_results = []
    print("\nCalculating MAD...")
    for name, emb in embeddings_dict.items():
        mad = compute_mad(emb)
        mad_results.append({'Model/Layer': name, 'MAD': mad})
        print(f"  {name}: MAD = {mad:.4f}")
        
    # Save MAD results
    if mad_results:
        df = pd.DataFrame(mad_results)
        df.to_csv(args.csv_output, index=False)
        print(f"Saved MAD stats to {args.csv_output}")
        print("\nMAD Table:")
        print(df)
        
    # Plot
    if embeddings_dict:
        visualize_tsne(embeddings_dict, args.output, title_prefix="Embeddings: " if args.mode == 'layers' else "")

if __name__ == '__main__':
    main()
