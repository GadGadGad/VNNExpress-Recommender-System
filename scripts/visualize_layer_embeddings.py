# %% [markdown]
# # Layer-wise Embedding Visualization
# This script visualizes the embeddings of different GCL models at specific layers (Layer 1).
# Run this in a Jupyter Notebook environment or as a script.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.models.lightgcn import LightGCN
from src.models.lightgcl import LightGCL
from src.models.xsimgcl import XSimGCL
from src.models.simgcl import SimGCL
from src.models.ma_hgn import MAHGN

# Fix random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Data Loader Setup
# Using patched LightGCLDataLoader to handle read-only issues if on Kaggle.

# %%
from src.data.dataloader_lightgcl import LightGCLDataLoader

def load_dataset(data_path='data'):
    print(f"Loading data from {data_path}...")
    loader = LightGCLDataLoader(data_path)
    
    # Try loading processed
    result = loader.load_processed()
    if result is None:
        print("Processed data not found. Processing from raw...")
        loader.load_raw_data()
        loader.preprocess(min_user_interactions=2, min_article_interactions=2)
        loader.create_mappings()
        interactions = loader.create_interactions()
        train_data, train_dict, test_data = loader.train_test_split(interactions)
        loader.save_processed(train_data, train_dict, test_data)
        result = (train_data, train_dict, test_data)
        
    train_data, train_dict, test_data = result
    
    # Build Graph
    print("Building adjacency matrix...")
    
    def build_sparse_graph(n_users, n_items, train_pairs):
        import scipy.sparse as sp
        row = np.array([u for u, i in train_pairs])
        col = np.array([i for u, i in train_pairs])
        data = np.ones(len(train_pairs), dtype=np.float32)
        
        R = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))
        
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

    adj_norm = build_sparse_graph(loader.n_users, loader.n_items, train_data)
    adj_norm = adj_norm.to(device)
    
    # For MA-HGN, we need heterogenous graph structure (simulated from adj_norm for now)
    # Ideally, we should load 'full_hetero_graph.pt' but for visualization we construct a basic one
    edge_index = adj_norm.indices()
    edge_index_dict = {
        ('user', 'interacts', 'article'): edge_index,
        ('article', 'interacted_by', 'user'): torch.stack([edge_index[1], edge_index[0]]),
        # Dummy social/category edges for L1 visualization if full graph missing
        ('user', 'social', 'user'): torch.stack([edge_index[0], edge_index[0]]), 
    }
    
    return loader, adj_norm, edge_index_dict

# %% [markdown]
# ## 2. Layer Extraction Logic
# We define helper functions to extract embeddings specifically at Layer 1
# by replicating the model's forward pass logic up to that point.

# %%
def extract_layer1_lightgcn(model, adj, **kwargs):
    """Effect of L=1 on LightGCN: E1 = A * E0"""
    with torch.no_grad():
        # LightGCN inherits from BaseGCL which uses nn.Embedding
        all_emb = torch.cat([model.user_embedding.weight, model.item_embedding.weight], dim=0)
        layer1_emb = torch.sparse.mm(adj, all_emb)
        u, i = torch.split(layer1_emb, [model.n_users, model.n_items])
        return u, i

import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='Path to data directory')
parser.add_argument('--dummy', action='store_true', help='Use dummy data')
args = parser.parse_args()

def extract_layer1_lightgcl(model, adj, **kwargs):
    """Effect of L=1 on LightGCL: Uses SVD q-rank + Norm"""
    with torch.no_grad():
        # Re-compute or use stored SVD components
        if model.u_mul_s is None:
            # Note: SVD on full adj might be heavy, typically done on R (N x M)
            # But here adj is (N+M) x (N+M). 
            # Ideally we extract R, but for simplicity we rely on model's internal handling 
            # or just assume the user passes a proper SVD-ready matrix if needed.
            # For visualization of *Contrastive* part, we need G_u, G_i.
            # But the *Graph* part (Z_u, Z_i) is just GCN.
            pass
            
        E_u_list = [model.E_u_0]
        E_i_list = [model.E_i_0]
        
        # Layer 1 Prop (Standard GCN part of LightGCL)
        # Fix dimension mismatch: adj is (N+M, N+M), E is (N/M, D)
        # We must concat E_u and E_i to propagate with adj
        all_emb = torch.cat([model.E_u_0, model.E_i_0], dim=0)
        res = torch.sparse.mm(adj, all_emb)
        
        Z_u = res[:model.n_users]
        Z_i = res[model.n_users:]
        
        # We perform SVD part if possible, but for "Graph Effect" visualization, 
        # Z (the aggregated neighbor info) is the direct equivalent of LightGCN's output.
        # So we return Z_u, Z_i.
        return Z_u, Z_i

def extract_layer1_xsimgcl(model, adj, **kwargs):
    """Effect of L=1 on XSimGCL: A * E0 (plus noise during train, but clean during eval)"""
    with torch.no_grad():
        u_emb = model.user_embedding.weight
        i_emb = model.item_embedding.weight
        all_emb = torch.cat([u_emb, i_emb], dim=0)
        
        # Layer 1
        layer1_emb = torch.sparse.mm(adj, all_emb)
        u, i = torch.split(layer1_emb, [model.n_users, model.n_items])
        return u, i

def extract_layer1_simgcl(model, adj, **kwargs):
    """Effect of L=1 on SimGCL: A * E0"""
    with torch.no_grad():
        u_emb = model.user_embedding.weight
        i_emb = model.item_embedding.weight
        all_emb = torch.cat([u_emb, i_emb], dim=0)
        
        # Layer 1
        layer1_emb = torch.sparse.mm(adj, all_emb)
        u, i = torch.split(layer1_emb, [model.n_users, model.n_items])
        return u, i

def extract_layer1_mahgn(model, adj, edge_index_dict=None):
    """Effect of L=1 on MA-HGN: HeteroConv"""
    with torch.no_grad():
        if edge_index_dict is None:
            raise ValueError("Edge Index Dict required for MA-HGN")
            
        x_dict = {
            'user': model.user_emb.weight,
            'article': model.item_emb.weight
        }
        if model.category_emb is not None:
             x_dict['category'] = model.category_emb.weight

        # Move edges to device
        device = x_dict['user'].device
        edge_index_dict_gpu = {k: v.to(device) for k, v in edge_index_dict.items()}

        # Layer 1 Conv (HeteroConv)
        # We only take the first conv layer from the ModuleList
        conv1 = model.convs[0]
        
        # Message Passing
        h_dict_new = conv1(x_dict, edge_index_dict_gpu)
        
        # In MA-HGN code: 
        # h_dict = {k: h_dict_new.get(k, h_dict_prev[k]) ...} (Residual/Keep old if no update)
        # For L1 strictly, we look at the new embeddings. 
        # If 'user' updated, use it. If not (isolated), use original.
        
        u_emb = h_dict_new.get('user', x_dict['user'])
        i_emb = h_dict_new.get('article', x_dict['article'])
        
        return u_emb, i_emb

# %% [markdown]
# ## 3. Visualization Function

# %%
def calculate_mad(emb):
    """
    Calculate Mean Average Distance (MAD) to measure smoothness.
    MAD = Mean( PairwiseCosineDistance ) or Mean( EuclideanDistanceToCentroid )
    Here we use Mean Euclidean Distance to Centroid for efficiency and robustness.
    """
    if torch.is_tensor(emb):
        emb = emb.detach().cpu().numpy()
    
    # Method 1: Mean Euclidean Distance to Centroid (Fast)
    centroid = np.mean(emb, axis=0)
    dists = np.linalg.norm(emb - centroid, axis=1)
    mad_value = np.mean(dists)
    return mad_value

def visualize_layer1(embeddings_dict, title="Layer 1 Embeddings", n_samples=1000):
    """
    Plots Grid of t-SNE embeddings with MAD scores.
    embeddings_dict: {'ModelName': (user_emb, item_emb)}
    """
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    print(f"Generating t-SNE and MAD scores for {n_models} models (Samples={n_samples})...")
    
    for idx, (model_name, (u_emb, _)) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
        
        # Data prep
        if len(u_emb) > n_samples:
            indices = np.random.choice(len(u_emb), n_samples, replace=False)
            u_sample = u_emb[indices].detach().cpu().numpy()
        else:
            u_sample = u_emb.detach().cpu().numpy()
            
        # 1. Calculate MAD (Quality Metric) on Sample
        mad_score = calculate_mad(u_sample)
        
        # 2. t-SNE (Visual Structure)
        # Init t-SNE (perplexity=30 is standard)
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        u_tsne = tsne.fit_transform(u_sample)
        
        # Plot
        ax.scatter(u_tsne[:, 0], u_tsne[:, 1], s=5, alpha=0.6)
        ax.set_title(f"{model_name}\nMAD: {mad_score:.4f}", fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.suptitle(f"{title} (User Embeddings)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('layer1_comparison.png', bbox_inches='tight')
    plt.show()

# Main Execution
try:
    if args.dummy:
        print("Forced dummy mode.")
        raise ValueError("Dummy mode requested")
    loader, adj, edge_index_dict = load_dataset(args.data_path)
    n_users, n_items = loader.n_users, loader.n_items
except Exception as e:
    print(f"Could not load real data: {e}")
    print("Using dummy data for demonstration...")
    n_users, n_items = 1000, 500
    # Dummy Bipartite Adjacency
    adj = torch.rand(n_users + n_items, n_users + n_items).to_sparse().coalesce().to(device)
    edge_index = adj.indices()
    
    # For MA-HGN, split indices roughly for demo
    # Note: This is random structure, purely for checking code flow
    edge_index = adj.indices()
    
    # Randomly assign edges to types for heterogenous simulation
    mask = torch.rand(edge_index.size(1)) < 0.5
    edge_index_ui = edge_index[:, mask]
    edge_index_social = edge_index[:, ~mask]
    
    edge_index_dict = {
        ('user', 'interacts', 'article'): edge_index_ui,
        ('article', 'interacted_by', 'user'): torch.stack([edge_index_ui[1], edge_index_ui[0]]),
        ('user', 'social', 'user'): edge_index_social
    }

# Initialize Models
print("Initializing models...")
models = {
    'LightGCN': LightGCN(n_users, n_items).to(device),
    'LightGCL': LightGCL(n_users, n_items).to(device),
    'XSimGCL': XSimGCL(n_users, n_items).to(device),
    'SimGCL': SimGCL(n_users, n_items).to(device),
    'MA-HGN': MAHGN(n_users, n_items, n_categories=6).to(device)
}

# Extract Embeddings
print("Extracting Layer 1 embeddings...")
results = {}

results['LightGCN'] = extract_layer1_lightgcn(models['LightGCN'], adj)
results['LightGCL'] = extract_layer1_lightgcl(models['LightGCL'], adj)
results['XSimGCL'] = extract_layer1_xsimgcl(models['XSimGCL'], adj)
results['SimGCL'] = extract_layer1_simgcl(models['SimGCL'], adj)
results['MA-HGN'] = extract_layer1_mahgn(models['MA-HGN'], adj, edge_index_dict=edge_index_dict)

# Visualize
print("Visualizing...")
visualize_layer1(results, title="Layer 1 Distribution")
print("Done! Saved to layer1_comparison.png")
