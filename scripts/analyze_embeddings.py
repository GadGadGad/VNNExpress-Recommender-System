#!/usr/bin/env python3
"""
Analyze Embedding Structure (Over-smoothing & Over-uniformity)
==============================================================

Calculates Mean Average Distance (MAD) and visualizes t-SNE 
of user/item embeddings to assess the balance between:
1. Community structure (avoiding over-uniformity)
2. Dispersion (avoiding over-smoothing)

Usage:
    python scripts/analyze_embeddings.py \
        --ma-hgn models/ma_hgn_strict_g2_xxxx.pt \
        --lightgcn models/lightgcn_strict_g2_xxxx.pt \
        --simgcl models/simgcl_strict_g2_xxxx.pt \
        --output results/analysis_rq3.png
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

def visualize_tsne(embeddings_dict, output_path="tsne_comparison.png", n_samples=2000):
    """
    Visualize t-SNE for multiple models side-by-side.
    """
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    print(f"Generating t-SNE for {n_models} models (samples={n_samples})...")
    
    for i, (model_name, emb) in enumerate(embeddings_dict.items()):
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
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def load_embedding(ckpt_path):
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Extract user embeddings
    # Common keys: 'user_embedding.weight', 'E_u_0' (LightGCL), 'gu_0' (MA-HGN/HetGNN base)
    if 'user_embedding.weight' in state_dict:
        return state_dict['user_embedding.weight']
    elif 'E_u_0' in state_dict: # LightGCL / LightGCN
        return state_dict['E_u_0']
    elif 'gu_0' in state_dict: # MA-HGN
        return state_dict['gu_0']
    else:
        # Try finding any weight with shape [n_users, dim]
        n_users = checkpoint['n_users']
        for k, v in state_dict.items():
            if v.shape[0] == n_users and v.ndim == 2:
                print(f"  Guessing {k} is user embedding.")
                return v
        raise ValueError(f"Could not locate user embeddings in {ckpt_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='List of model paths in format name=path', required=True)
    parser.add_argument('--output', default='results/rq3_analysis.png')
    parser.add_argument('--csv-output', default='results/rq3_mad.csv')
    args = parser.parse_args()
    
    embeddings_dict = {}
    mad_results = []
    
    for item in args.models:
        name, path = item.split('=')
        try:
            emb = load_embedding(path)
            embeddings_dict[name] = emb
            
            # Compute MAD
            mad = compute_mad(emb)
            mad_results.append({'Model': name, 'MAD': mad})
            print(f"  {name}: MAD = {mad:.4f}")
            
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    # Save MAD results
    if mad_results:
        df = pd.DataFrame(mad_results)
        df.to_csv(args.csv_output, index=False)
        print(f"Saved MAD stats to {args.csv_output}")
        print("\nFinal MAD Table:")
        print(df)
        
    # Plot
    if embeddings_dict:
        visualize_tsne(embeddings_dict, args.output)

if __name__ == '__main__':
    main()
