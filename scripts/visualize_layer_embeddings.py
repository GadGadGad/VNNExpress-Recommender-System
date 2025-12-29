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
    from src.utils.graph_utils import build_sparse_graph
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
def visualize_layer1(embeddings_dict, title="Layer 1 Embeddings", n_samples=500):
    """
    Plots PCA of embeddings.
    embeddings_dict: {'ModelName': (user_emb, item_emb)}
    """
    plt.figure(figsize=(15, 6))
    
    # 1. User Embeddings Comparison
    plt.subplot(1, 2, 1)
    for model_name, (u_emb, _) in embeddings_dict.items():
        # Sample random users
        indices = np.random.choice(len(u_emb), n_samples, replace=False)
        u_sample = u_emb[indices].detach().cpu().numpy()
        
        pca = PCA(n_components=2)
        u_pca = pca.fit_transform(u_sample)
        
        # Normalize for easier overlay comparison
        u_pca = (u_pca - u_pca.mean(0)) / u_pca.std(0)
        
        plt.scatter(u_pca[:, 0], u_pca[:, 1], label=model_name, alpha=0.5, s=10)
        
    plt.title(f"User Embeddings (PCA) - {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Item Embeddings Comparison
    plt.subplot(1, 2, 2)
    for model_name, (_, i_emb) in embeddings_dict.items():
        # Sample random items
        indices = np.random.choice(len(i_emb), n_samples, replace=False)
        i_sample = i_emb[indices].detach().cpu().numpy()
        
        pca = PCA(n_components=2)
        i_pca = pca.fit_transform(i_sample)
        
        # Normalize
        i_pca = (i_pca - i_pca.mean(0)) / i_pca.std(0)
        
        plt.scatter(i_pca[:, 0], i_pca[:, 1], label=model_name, alpha=0.5, s=10)
        
    plt.title(f"Item Embeddings (PCA) - {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layer1_comparison.png')
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
    adj = torch.rand(n_users + n_items, n_users + n_items).to_sparse().to(device)
    edge_index = adj.indices()
    edge_index_dict = {
        ('user', 'interacts', 'article'): edge_index,
        ('article', 'interacted_by', 'user'): torch.stack([edge_index[1], edge_index[0]]),
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
