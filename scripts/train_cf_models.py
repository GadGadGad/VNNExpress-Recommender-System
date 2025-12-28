#!/usr/bin/env python3
"""
Train Collaborative Filtering / Contrastive Learning Models
Supports: NGCF, SimpleX, DirectAU, SGL, SimGCL, NCL, LightGCL
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.simplex import SimpleX
from src.models.directau import DirectAU
from src.models.sgl import SGL
from src.models import NGCF, LightGCL, SimGCL, XSimGCL, MAHGN, SimMAHGN, HetGNN
from src.models.ncl import NCL
from src.models.cgrc import CGRC
from src.models.bigcf import BIGCF
from src.models.igcl import IGCL
from src.models.semantic_id import generate_semantic_ids
from src.inference.re_ranker import CalibratedReRanker
import scipy.sparse as sp


class SemanticEmbeddingLayer(nn.Module):
    """
    Combines hierarchical semantic IDs into a single embedding.
    """
    def __init__(self, n_codebooks, codebook_size, embedding_dim):
        super(SemanticEmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(n_codebooks)
        ])
        
    def forward(self, semantic_ids):
        # semantic_ids: (n_items, n_codebooks)
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(semantic_ids[:, i])
        return out

class UserPriorLayer(nn.Module):
    """
    Projects dense user priors to matching embedding dimension.
    """
    def __init__(self, input_dim, output_dim):
        super(UserPriorLayer, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, user_priors):
        return self.proj(user_priors)

def compute_normalized_adjacency(n_users, n_items, train_pairs, device, 
                                 item_item_edges=None, user_user_edges=None, 
                                 edge_weights=None, social_weight=1.0):
    """Compute normalized adjacency matrix for GCN-based models."""
    row = np.array([u for u, i in train_pairs])
    col = np.array([i for u, i in train_pairs])
    
    if edge_weights is not None:
        data = edge_weights
        if len(data) != len(train_pairs):
            print(f"  Warning: edge_weight size ({len(data)}) != train_pairs ({len(train_pairs)}). Clipping.")
            data = data[:len(train_pairs)]
    else:
        data = np.ones(len(train_pairs), dtype=np.float32)
        
    R = sp.coo_matrix((data, (row, col)), shape=(n_users, n_items))
    
    # Construct (N+M)x(N+M) matrix
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()
    
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    
    # Add Item-Item edges (LLMRec/Semantic)
    if item_item_edges is not None:
        print(f"  Injecting item-item semantic edges into adjacency...")
        s_row = item_item_edges[0].numpy() if torch.is_tensor(item_item_edges) else item_item_edges[0]
        s_col = item_item_edges[1].numpy() if torch.is_tensor(item_item_edges) else item_item_edges[1]
        for u, v in zip(s_row, s_col):
            adj_mat[n_users + u, n_users + v] = 1.0
            adj_mat[n_users + v, n_users + u] = 1.0

    # Add User-User edges (Social/Latent)
    if user_user_edges is not None:
        print(f"  Injecting user-user social edges into adjacency (weight={social_weight})...")
        u_row = user_user_edges[0].numpy() if torch.is_tensor(user_user_edges) else user_user_edges[0]
        u_col = user_user_edges[1].numpy() if torch.is_tensor(user_user_edges) else user_user_edges[1]
        for u, v in zip(u_row, u_col):
            adj_mat[u, v] = social_weight
            adj_mat[v, u] = social_weight
            
    adj_mat = adj_mat.tocoo()
    
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)


class LightGCLWrapper:
    """Wrapper that handles LightGCL's special requirements (SVD, adj_norm)."""
    
    def __init__(self, n_users, n_items, embed_dim=64, n_layers=3, device='cpu',
                 svd_q=20, ssl_weight=0.1, temp=0.2):
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.model = LightGCL(n_users, n_items, embed_dim, n_layers, 
                              svd_q=svd_q, ssl_weight=ssl_weight, temp=temp)
        self.adj_norm = None
        
    def setup(self, train_pairs, augmented_pairs=None):
        """Create normalized interaction matrix and compute SVD."""
        all_pairs = list(train_pairs)
        if augmented_pairs:
            print(f"  Injecting {len(augmented_pairs)} synthetic interactions from LLMRec...")
            all_pairs.extend(augmented_pairs)
            
        row = np.array([u for u, i in all_pairs])
        col = np.array([i for u, i in all_pairs])
        data = np.ones(len(all_pairs), dtype=np.float32)
        R = sp.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        
        rowD = np.array(R.sum(1)).squeeze()
        colD = np.array(R.sum(0)).squeeze()
        rowD[rowD == 0] = 1
        colD[colD == 0] = 1
        
        R_coo = R.tocoo()
        normalized_data = np.zeros_like(R_coo.data)
        for i in range(len(R_coo.data)):
            normalized_data[i] = R_coo.data[i] / np.sqrt(rowD[R_coo.row[i]] * colD[R_coo.col[i]])
        
        indices = torch.LongTensor(np.array([R_coo.row, R_coo.col]))
        values = torch.FloatTensor(normalized_data)
        shape = torch.Size(R_coo.shape)
        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(self.device)
        
        self.model.to_device(self.device)
        self.model.compute_svd(self.adj_norm)
        
    def to(self, device):
        self.device = device
        return self
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()
        
    def parameters(self):
        return self.model.parameters()
        
    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    def forward(self, edge_index=None):
        E_u, E_i, _, _ = self.model.forward(self.adj_norm)
        return E_u, E_i
    
    def __call__(self, edge_index=None):
        return self.forward(edge_index)
    
    def bpr_loss(self, users, pos_items, neg_items, edge_index):
        total_loss, bpr, reg, ssl = self.model.calculate_loss(
            self.adj_norm, users, pos_items, neg_items
        )
        return total_loss, torch.tensor(0.0, device=total_loss.device)


def load_data(data_path, min_interactions=2):
    """Load and process data for CF models."""
    import pandas as pd
    from torch_geometric.data import HeteroData
    
    # Check if data_path is already a file or a directory
    p = Path(data_path)
    if p.is_file():
        cache_path = p
    else:
        cache_path = p / 'cf_cache.pt'
        if not cache_path.exists():
            cache_path = p / 'graph_with_negatives.pt'
            if not cache_path.exists():
                cache_path = p / 'user_article_graph.pt'
        
    if cache_path.exists():
        data = torch.load(cache_path, weights_only=False)
        
        # Normalization for different graph types
        if 'n_users' not in data:
            data['n_users'] = data.get('num_users') or data.get('n_users') or 0
        if 'n_items' not in data:
            data['n_items'] = data.get('num_articles') or data.get('num_items') or data.get('n_articles') or 0
        
        if 'item_map' in data and 'article_map' not in data:
            data['article_map'] = data['item_map']
            
        # Handle PyG Data/HeteroData objects or dicts missing standard CF keys
        from torch_geometric.data import Data, HeteroData
        required_keys = ['train_pairs', 'train_dict', 'test_dict', 'edge_index']
        
        if isinstance(data, (Data, HeteroData)) or any(k not in data for k in required_keys):
            print("  Ensuring interaction data and splits...")
            
            # ATTEMPT TO FIND MASTER SPLITS in the same directory to ensure consistency
            master_split_path = cache_path.parent / 'graph_with_negatives.pt'
            master_splits = None
            if master_split_path.exists() and master_split_path != cache_path:
                print(f"  Synchronizing splits with master: {master_split_path}")
                master_data = torch.load(master_split_path, weights_only=False)
                if isinstance(master_data, dict) and 'splits' in master_data:
                    master_splits = master_data['splits']
            
            edge_index = None
            if isinstance(data, HeteroData):
                if ('user', 'comments', 'article') in data.edge_types:
                    edge_index = data['user', 'comments', 'article'].edge_index
                else:
                    # Fallback to first user-item like relation
                    edge_index = next(iter(data.edge_index_dict.values()))
                n_users = data['user'].num_nodes
                n_items = data['article'].num_nodes
                if 'category' in data.node_types:
                    n_categories = data['category'].num_nodes
            elif isinstance(data, Data):
                edge_index = data.edge_index
                n_users = data.num_nodes # Approximate if not specified
                n_items = data.num_nodes
            elif isinstance(data, dict):
                # Try to get interactions from either edge_index or existing train_pairs
                if 'edge_index' in data:
                    edge_index = data['edge_index']
                elif 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                    # HeteroGraph in dict
                    edge_index = data['graph']['user', 'comments', 'article'].edge_index
                elif 'train_pairs' in data:
                    interactions = list(set(data['train_pairs']))
                elif 'splits' in data:
                    # Will be handled below in pre-defined splits block
                    pass
                else:
                    return data
            
            if edge_index is not None:
                interactions = list(set(zip(edge_index[0].tolist(), edge_index[1].tolist())))
            
            # Get n_users/n_items from dictionary if not set by graph extraction
            if 'n_users' not in locals(): 
                n_users = data.get('n_users') or data.get('num_users') or 0
            if 'n_items' not in locals(): 
                n_items = data.get('n_items') or data.get('num_articles') or data.get('num_items') or 0

            if master_splits is not None:
                print("  Using pre-defined splits from master (graph_with_negatives.pt)...")
                splits = master_splits
                train_pairs = list(zip(splits['train']['pos_users'].tolist(), splits['train']['pos_articles'].tolist()))
                test_dict = {}
                for u, i in zip(splits['test']['pos_users'].tolist(), splits['test']['pos_articles'].tolist()):
                    if u not in test_dict: test_dict[u] = set()
                    test_dict[u].add(i)
                train_dict = {}
                for u, i in train_pairs:
                    if u not in train_dict: train_dict[u] = set()
                    train_dict[u].add(i)
                
                data_dict = {
                    'n_users': n_users,
                    'n_items': n_items,
                    'n_categories': n_categories if 'n_categories' in locals() else 0,
                    'train_pairs': train_pairs,
                    'test_dict': test_dict,
                    'train_dict': train_dict,
                    'edge_index': torch.tensor(train_pairs, dtype=torch.long).t(),
                    'graph': data # Store PyG object if available
                }
                return data_dict

            if edge_index is not None:
                np.random.seed(42)
                np.random.shuffle(interactions)
                split = int(len(interactions) * 0.8)
                
                data_dict = {
                    'n_users': n_users,
                    'n_items': n_items,
                    'n_categories': n_categories if 'n_categories' in locals() else 0,
                    'train_pairs': interactions[:split],
                    'test_dict': {}, 
                    'train_dict': {}
                }
                # Fill dicts
                for u, i in interactions[:split]:
                    if u not in data_dict['train_dict']: data_dict['train_dict'][u] = set()
                    data_dict['train_dict'][u].add(i)
                # test dict
                for u, i in interactions[split:]:
                    if u not in data_dict['test_dict']: data_dict['test_dict'][u] = set()
                    data_dict['test_dict'][u].add(i)
            
            if 'edge_index' not in data_dict or data_dict['edge_index'] is None:
                # Bipartite graph edge index
                train_users = torch.tensor([u for u, i in data_dict['train_pairs']], dtype=torch.long)
                train_items = torch.tensor([i for u, i in data_dict['train_pairs']], dtype=torch.long)
                data_dict['edge_index'] = torch.stack([
                    torch.cat([train_users, train_items + n_users]),
                    torch.cat([train_items + n_users, train_users])
                ], dim=0)

            # Keep original fields if it was a dict
            if isinstance(data, dict):
                data.update(data_dict)
                return data
            
            # For ma_hgn, preserve edge_index_dict if present in original data
            if isinstance(data, HeteroData):
                data_dict['edge_index_dict'] = data.edge_index_dict
                
            return data_dict

        return data
    
    print("Processing data...")
    raw_replies = Path('data/raw/replies.csv')
    if not raw_replies.exists():
        raw_replies = Path(data_path).parent / 'raw' / 'replies.csv'
        
    replies = pd.read_csv(raw_replies)
    replies = replies[replies['parent_user_id'] != 'NO_COMMENT'].copy()
    
    def clean_id(val):
        try:
            return str(int(float(val))) if pd.notna(val) else None
        except:
            return str(val)
    
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Filter by min interactions
    prev_len = 0
    while len(replies) != prev_len:
        prev_len = len(replies)
        user_counts = replies['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        article_counts = replies['article_url'].value_counts()
        valid_articles = article_counts[article_counts >= min_interactions].index
        replies = replies[
            (replies['user_id'].isin(valid_users)) &
            (replies['article_url'].isin(valid_articles))
        ].copy()
    
    user_ids = replies['user_id'].unique()
    article_urls = replies['article_url'].unique()
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    article_map = {url: idx for idx, url in enumerate(article_urls)}
    
    n_users = len(user_map)
    n_items = len(article_map)
    
    replies['user_idx'] = replies['user_id'].map(user_map)
    replies['item_idx'] = replies['article_url'].map(article_map)
    
    # Split train/test FIRST, then create edge_index from train only
    interactions = list(zip(replies['user_idx'].values, replies['item_idx'].values))
    interactions = list(set(interactions))
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(interactions)
    
    split_idx = int(len(interactions) * 0.8)
    train_pairs = interactions[:split_idx]
    test_pairs = interactions[split_idx:]
    
    train_dict = {}
    for u, i in train_pairs:
        if u not in train_dict:
            train_dict[u] = set()
        train_dict[u].add(i)
    
    test_dict = {}
    for u, i in test_pairs:
        if u not in test_dict:
            test_dict[u] = set()
        test_dict[u].add(i)
    
    # Create edge index from TRAIN PAIRS ONLY (no data leakage)
    train_users = torch.tensor([u for u, i in train_pairs], dtype=torch.long)
    train_items = torch.tensor([i for u, i in train_pairs], dtype=torch.long)
    
    # Bipartite graph edge index (train only)
    edge_index = torch.stack([
        torch.cat([train_users, train_items + n_users]),
        torch.cat([train_items + n_users, train_users])
    ], dim=0)
    
    data = {
        'n_users': n_users,
        'n_items': n_items,
        'user_map': user_map,
        'article_map': article_map,
        'edge_index': edge_index,
        'train_pairs': train_pairs,
        'train_dict': train_dict,
        'test_dict': test_dict,
    }
    
    # Ensure directory exists
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)

    return data


def sample_batch(train_pairs, train_dict, n_items, batch_size, neg_ratio=1, sampling='random', item_probs=None):
    """Sample a batch with negative items."""
    indices = np.random.choice(len(train_pairs), min(batch_size, len(train_pairs)), replace=False)
    users, pos_items, neg_items = [], [], []
    
    # Pre-sample negatives if popular strategy is used
    if sampling == 'popular' and item_probs is not None:
        popular_items = np.arange(n_items)
    
    for idx in indices:
        u, pos = train_pairs[idx]
        
        for _ in range(neg_ratio):
            users.append(u)
            pos_items.append(pos)
            
            # Sample negative
            if sampling == 'popular' and item_probs is not None:
                neg = np.random.choice(popular_items, p=item_probs)
                while neg in train_dict.get(u, set()):
                    neg = np.random.choice(popular_items, p=item_probs)
            else:
                neg = np.random.randint(0, n_items)
                while neg in train_dict.get(u, set()):
                    neg = np.random.randint(0, n_items)
            neg_items.append(neg)
    
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long)
    )


def load_pretrained_embeddings(embedding_type, n_items, target_dim, device='cpu'):
    """Load pretrained embeddings and project to target dimension."""
    if embedding_type == 'random':
        return None
        
    print(f"\nLoading {embedding_type} embeddings...")
    embeddings = None
    
    if embedding_type == 'phobert':
        path = 'checkpoints/phobert_article_embeddings.pt'
        if os.path.exists(path):
            embeddings = torch.load(path, map_location='cpu')
            print(f"  Loaded PhoBERT embeddings: {embeddings.shape}")
        else:
            print(f"  Warning: {path} not found. Using random initialization.")
            return None
            
    elif embedding_type == 'vndoc':
        path = 'checkpoints/vndoc_article_embeddings.pt'
        if os.path.exists(path):
            embeddings = torch.load(path, map_location='cpu')
            print(f"  Loaded VnDoc embeddings: {embeddings.shape}")
        else:
            print(f"  Warning: {path} not found. Using random initialization.")
            return None
    
    elif embedding_type in ['bge-m3', 'gte', 'e5-large', 'e5-base', 'vn-sbert']:
        # Map shortnames to full paths
        path_map = {
            'bge-m3': 'checkpoints/bge-m3_article_embeddings.pt',
            'gte': 'checkpoints/gte-multilingual_article_embeddings.pt',
            'e5-large': 'checkpoints/e5-large_article_embeddings.pt',
            'e5-base': 'checkpoints/e5-base_article_embeddings.pt',
            'vn-sbert': 'checkpoints/vietnamese-sbert_article_embeddings.pt'
        }
        path = path_map[embedding_type]
        if os.path.exists(path):
            embeddings = torch.load(path, map_location='cpu')
            print(f"  Loaded {embedding_type} embeddings: {embeddings.shape}")
        else:
            print(f"  Warning: {path} not found. Using random initialization.")
            return None
            
    elif embedding_type == 'tfidf':
        print("  Computing TF-IDF embeddings (LSA)...")
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        # Load articles to get text
        try:
            pass
            
            # Re-implementation to be safe: Load the full dataloader to get mapping
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.data.dataloader_lightgcl import LightGCLDataLoader
            
            # Initialize loader with root 'data' path (assuming project structure)
            loader = LightGCLDataLoader('data')
            
            # Load processed data which populates idx2item
            if loader.load_processed() is None:
                print("  Error: Could not load processed LightGCL data for mapping. Run training first.")
                return None
                
            idx2item = loader.idx2item
            print(f"  Loaded mapping for {len(idx2item)} items")
            
            # Create text list in index order
            article_map = dict(zip(articles['url'], zip(articles['title'], articles['short_description'])))
            
            for idx in range(n_items):
                url = idx2item.get(idx, None)
                if url and url in article_map:
                    title, desc = article_map[url]
                    # Handle NaNs
                    title = str(title) if pd.notna(title) else ""
                    desc = str(desc) if pd.notna(desc) else ""
                    texts.append(f"{title} {desc}")
                else:
                    texts.append("")
            
            print(f"  Collected {len(texts)} texts. Computing TF-IDF...")
            
            # Compute TF-IDF
            # Use stricter max_features to avoid noise, but SVD handles dimension reduction anyway
            vectorizer = TfidfVectorizer(max_features=10000, stop_words=None) 
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimension with SVD (LSA)
            print(f"  Reducing dimension {tfidf_matrix.shape[1]} -> {target_dim} with SVD...")
            svd = TruncatedSVD(n_components=target_dim, random_state=42)
            embeddings = torch.tensor(svd.fit_transform(tfidf_matrix), dtype=torch.float32)
            
            # Normalize calculated embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            print(f"  Computed LSA embeddings: {embeddings.shape}")
            return embeddings.to(device) # Return directly, skipping external normalization block to avoid double norm
            
        except Exception as e:
            print(f"  Error computing TF-IDF: {e}. Using random.")
            import traceback
            traceback.print_exc()
            return None
        
    if embeddings is not None:
        # Match item count (truncate or pad if needed)
        # Note: Ideally n_items matches exactly. If valid items < n_items (due to padding), we pad.
        curr_items, curr_dim = embeddings.shape
        
        if curr_items != n_items:
            print(f"  Warning: Embedding items ({curr_items}) != Dataset items ({n_items})")
            # If we have fewer embeddings, pad with random
            if curr_items < n_items:
                padding = torch.randn(n_items - curr_items, curr_dim)
                embeddings = torch.cat([embeddings, padding], dim=0)
            else:
                embeddings = embeddings[:n_items]
        
        # Project dimension if needed
        if curr_dim != target_dim:
            print(f"  Projecting dimension: {curr_dim} -> {target_dim}")
            # Use random projection matrix
            projection = torch.randn(curr_dim, target_dim) / np.sqrt(curr_dim)
            embeddings = torch.matmul(embeddings.float(), projection.float())
            
        # Normalize embeddings to prevent numerical instability (NaN loss)
        # LightGCL uses Xavier init which produces small values. Large pretrained norms cause exp() explosion.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Check for NaNs
        if torch.isnan(embeddings).any():
             print("  Warning: Embeddings contain NaN! Replacing with random.")
             return None
             
        # Clamp to avoid extreme values just in case
        embeddings = torch.clamp(embeddings, -1.0, 1.0)
        
        # Scale down slightly to ensure stability (Xavier often < 1)
        embeddings = embeddings * 0.1
        
        print(f"  Final embeddings (mean={embeddings.mean():.4f}, std={embeddings.std():.4f}, max={embeddings.abs().max():.4f})")
            
        return embeddings.to(device)
            
    return None


def evaluate(model, test_dict, train_dict, n_items, edge_index, k_list=[1, 5, 10, 50], device='cpu', adj_norm=None, 
             re_ranker=None, rerank_strategy='none', eval_protocol='full', cold_users=None, edge_index_dict=None):
    """
    Evaluate model with multiple protocols.
    
    eval_protocol:
        - 'full': Full ranking over all items (hardest, most realistic)
        - 'loo100': Leave-one-out + 100 random negatives (common in papers)
        - 'cold': Evaluate only on cold-start users
    """
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, 'forward'):
            forward_args = model.forward.__code__.co_varnames
            
            if 'edge_index_dict' in forward_args and edge_index_dict is not None:
                user_emb, item_emb = model(None, edge_index_dict)
            elif 'adj_norm' in forward_args and adj_norm is not None:
                kwargs = {}
                if 'item_content' in forward_args:
                    kwargs['item_content'] = getattr(model, 'item_content', None)
                if 'semantic_ids' in forward_args:
                    kwargs['semantic_ids'] = getattr(model, 'semantic_ids', None)
                user_emb, item_emb = model(adj_norm, **kwargs)
            elif 'edge_index' in forward_args:
                user_emb, item_emb = model(edge_index.to(device))
            elif hasattr(model, 'adj_norm'):
                user_emb, item_emb = model()
            else:
                user_emb, item_emb = model()
        else:
            user_emb = model.user_embedding.weight
            item_emb = model.item_embedding.weight
    
    max_k = max(k_list)
    results = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg', 'hitrate', 'precision', 'map'] for k in k_list}
    results['mrr'] = []
    
    # Choose which users to evaluate based on protocol
    if eval_protocol == 'cold' and cold_users is not None:
        eval_users = {u: items for u, items in test_dict.items() if u in cold_users}
        print(f"  Cold-start eval: {len(eval_users)} users")
    else:
        eval_users = test_dict
    
    for user, test_items in eval_users.items():
        # Ensure test_items is a set for set operations
        test_items = set(test_items) if isinstance(test_items, list) else test_items
        
        if user >= user_emb.size(0):
            continue
            
        train_items = train_dict.get(user, set())
        u_emb = user_emb[user].unsqueeze(0)
        
        if eval_protocol == 'loo100':
            # Leave-One-Out + 100 negatives: sample 100 neg items + all positive items
            all_items = set(range(n_items)) - train_items - test_items
            neg_samples = np.random.choice(list(all_items), min(100, len(all_items)), replace=False)
            candidate_items = list(test_items) + list(neg_samples)
            
            # Score only these candidates
            candidate_emb = item_emb[candidate_items]
            scores = torch.mm(u_emb, candidate_emb.t()).squeeze()
            
            # Map back to original indices
            _, topk_local = torch.topk(scores, min(max_k, len(candidate_items)))
            topk_candidates = [candidate_items[i] for i in topk_local.cpu().numpy()]
        else:
            # Full ranking
            scores = torch.mm(u_emb, item_emb.t()).squeeze()
            
            # Mask train items
            for item in train_items:
                if item < scores.size(0):
                    scores[item] = -float('inf')
            
            _, topk = torch.topk(scores, 100)
            topk_candidates = topk.cpu().numpy().tolist()
        
        # Apply re-ranking if specified
        if rerank_strategy == 'mmr' and re_ranker is not None:
            topk_list = re_ranker.mmr_rerank(item_emb, scores if eval_protocol == 'full' else None, top_k=max_k)
        elif rerank_strategy == 'calib' and re_ranker is not None:
            user_history = list(train_items) if train_items else []
            score_arr = scores.cpu().numpy() if eval_protocol == 'full' else np.zeros(n_items)
            topk_list = re_ranker.calibrate(score_arr, user_history, top_k=max_k)
        else:
            topk_list = topk_candidates[:max_k]
        
        # MRR
        mrr = 0.0
        for i, item in enumerate(topk_list):
            if item in test_items:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'].append(mrr)
        
        # Metrics at each k
        for k in k_list:
            topk_k = set(topk_list[:k])
            hits = len(topk_k & test_items)
            
            results[f'recall@{k}'].append(hits / len(test_items))
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
            
            # Precision@k: hits / k
            results[f'precision@{k}'].append(hits / k)
            
            # NDCG@k
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(topk_list[:k]) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
            results[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0)
            
            # mAP@k (Average Precision): sum of precision at each hit / min(k, num_positives)
            ap = 0.0
            n_hits = 0
            for i, item in enumerate(topk_list[:k]):
                if item in test_items:
                    n_hits += 1
                    ap += n_hits / (i + 1)
            ap = ap / min(k, len(test_items)) if len(test_items) > 0 else 0
            results[f'map@{k}'].append(ap)
        
        if re_ranker is not None:
             ent = compute_entropy(topk_list, re_ranker.item_categories, re_ranker.n_categories)
             results.setdefault('entropy', []).append(ent)
    
    return {k: np.mean(v) for k, v in results.items()}



def load_item_categories(idx2item, csv_path='data/raw/articles.csv'):
    """Map item indices to their categories."""
    df = pd.read_csv(csv_path)
    url_to_cat = dict(zip(df['url'], df['source_category']))
    unique_cats = sorted(df['source_category'].unique().tolist())
    cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
    
    categories = []
    for i in range(len(idx2item)):
        url = idx2item[i]
        cat = url_to_cat.get(url, 'Other')
        categories.append(cat_to_id.get(cat, 0))
    return np.array(categories), len(unique_cats)

def compute_entropy(item_indices, item_categories, n_categories):
    """Compute Shannon Entropy of categorical distribution."""
    cats = item_categories[item_indices]
    counts = np.bincount(cats, minlength=n_categories)
    probs = counts / (len(item_indices) + 1e-9)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0


def train_model(model, data, args, device, item_content=None, semantic_ids=None, user_priors=None, 
                re_ranker=None, cold_users=None):

    """Train a CF model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    edge_index = data['edge_index'].to(device)
    edge_index_dict = data.get('edge_index_dict')
    if edge_index_dict is None and isinstance(data, dict):
        if 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
            edge_index_dict = data['graph'].edge_index_dict
    train_pairs = data['train_pairs']
    train_dict = data['train_dict']
    test_dict = data['test_dict']
    n_items = data['n_items']
    
    # --- STRUCTURAL LEAKAGE CHECK ---
    # Check if the graph used for message passing contains more interactions than the training set.
    # This happens if test edges were not removed from the graph index during data conversion.
    graph_to_check = edge_index_dict if edge_index_dict is not None else edge_index
    if graph_to_check is not None:
        if isinstance(graph_to_check, dict):
            # Hetero graph: Look for user->article or user->item edges
            ua_edges = None
            for key, val in graph_to_check.items():
                if isinstance(key, tuple) and len(key) == 3:
                    if 'user' in str(key[0]).lower() and ('article' in str(key[2]).lower() or 'item' in str(key[2]).lower()):
                        ua_edges = val
                        break
        else:
            # Bipartite graph: It's usually symmetric [2, 2*E] or directed [2, E]
            # Since we offset items, we check src < n_users and dst >= n_users
            ua_edges = graph_to_check
            
        if ua_edges is not None and hasattr(ua_edges, 'size'):
            # For symmetric bipartite graphs in MA-HCL etc, edge_index has 2*E edges.
            # We only care about unique interactions.
            n_graph_edges = ua_edges.size(1)
            # If symmetric [2, 2*E], we divide by 2
            if not isinstance(graph_to_check, dict):
                src, dst = ua_edges[0], ua_edges[1]
                # Filter to only directed user->item edges if it's symmetric
                is_user_item = (src < data['n_users']) & (dst >= data['n_users'])
                n_graph_interactions = is_user_item.sum().item()
            else:
                n_graph_interactions = n_graph_edges
                
            n_train_interactions = len(train_pairs)
            
            if n_graph_interactions > n_train_interactions:
                print(f"\n" + "!"*60)
                print(f"⚠️  CRITICAL LEAKAGE DETECTED!")
                print(f"   Message Passing Graph has {n_graph_interactions:,} user-item interactions.")
                print(f"   Training Set has only {n_train_interactions:,} interactions.")
                print(f"   Leakage: {n_graph_interactions - n_train_interactions:,} test edges are visible to the model!")
                print(f"   REASON: Graph was likely built with '--min-user-interactions' on ALL data.")
                print(f"   FIX: Regenerate graph using leakage-fixed converter.")
                print("!"*60 + "\n")
    
    best_recall = 0
    best_metrics = {}
    best_state = None
    patience_counter = 0
    
    pbar = tqdm(range(args.epochs), desc=f"Training {args.model.upper()}", ncols=100)
    
    if args.epochs == 0:
        print("Epochs set to 0. Exiting after data preparation.")
        return {'status': 'consolidated'}
        
    item_interaction_counts = np.zeros(n_items)
    for _, item in train_pairs:
        item_interaction_counts[item] += 1
    item_probs = (item_interaction_counts + 1e-6) / (item_interaction_counts.sum() + 1e-6 * n_items)
    sampling_strategy = 'popular' if args.denoise_ratio > 0 else 'random' # Default to popular if denoising
    
    for epoch in pbar:
        model.train()
        total_loss = 0
        n_batches = len(train_pairs) // args.batch_size + 1
        
        for _ in range(n_batches):
            users, pos_items, neg_items = sample_batch(train_pairs, train_dict, n_items, args.batch_size, args.neg_ratio, 
                                                      sampling=sampling_strategy, item_probs=item_probs)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            
            # Different models have different loss signatures
            if isinstance(model, HetGNN):
                # HetGNN needs x_dict and edge_index_dict
                graph_structure = getattr(data, 'edge_index_dict', None)
                if graph_structure is None and isinstance(data, dict):
                    graph_structure = data.get('edge_index_dict')
                if graph_structure is None:
                        # Fallback to bipartite edge_index if no hetero data
                        graph_structure = edge_index
                loss, reg = model.bpr_loss(users, pos_items, neg_items, x_dict=None, edge_index_dict=graph_structure)
                loss = loss + args.weight_decay * reg

            elif hasattr(model, 'calculate_loss'):
                # Graph-based models need `adj_norm`
                if args.model in ['simgcl', 'cgrc', 'bigcf', 'igcl', 'xsimgcl', 'sgl', 'ncl', 'sim-mahgn']:
                    graph_structure = data.get('adj_norm')
                    if graph_structure is None:
                        # Fallback if not computed (shouldn't happen if setup is correct)
                        print("Warning: adj_norm missing in train loop, falling back to edge_index")
                        graph_structure = edge_index
                elif args.model == 'lightgcl':
                     # LightGCL internal model manages its structure usually, likely stored inside Wrapper
                     # But if we access `model.calculate_loss`, we might need to pass something.
                     # Our LightGCLWrapper (if used) handles it. 
                     # If `model` is the inner model, we might need adj. 
                     # Let's check LightGCLWrapper usage. 
                     # The `model` passed here is likely the Wrapper instance or the internal Module.
                     # In main(), `model` is assigned `LightGCLWrapper(...).model`? Or the wrapper itself?
                     # Let's assume wrapper extracts what it needs.
                     graph_structure = data.get('adj_norm') # Pass it anyway
                elif args.model == 'ma_hgn':
                    # MA-HGN needs the full heterogeneous edge dictionary
                    graph_structure = getattr(data, 'edge_index_dict', None)
                    if graph_structure is None and isinstance(data, dict):
                        graph_structure = data.get('edge_index_dict')
                    if graph_structure is None:
                         # Fallback to bipartite edge_index if no hetero data
                         graph_structure = edge_index
                else:
                    graph_structure = edge_index
                
                
                if isinstance(model, CGRC):
                    loss, bpr, recon, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items, item_content)
                elif isinstance(model, XSimGCL):
                    if args.denoise_ratio > 0 and epoch >= 5: # Burn-in 5 epochs
                        loss, bpr_sample, ssl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items, 
                                                                         semantic_ids=semantic_ids, user_priors=user_priors,
                                                                         return_per_sample=True)
                        # Truncated Loss Denoising
                        n_to_prune = int(len(bpr_sample) * args.denoise_ratio)
                        if n_to_prune > 0:
                            _, indices = torch.topk(bpr_sample, k=n_to_prune, largest=True)
                            mask = torch.ones_like(bpr_sample)
                            mask[indices] = 0
                            bpr = (bpr_sample * mask).sum() / mask.sum()
                            loss = bpr + model.ssl_weight * ssl + model.lambda_reg * reg
                        else:
                            bpr = bpr_sample.mean()
                    else:
                        loss, bpr, ssl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items, 
                                                                  semantic_ids=semantic_ids, user_priors=user_priors)
                elif isinstance(model, BIGCF):
                    loss, bpr, cl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items)
                elif isinstance(model, IGCL):
                    loss, bpr, ssl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items)
                elif isinstance(model, SimMAHGN) or args.model in ['ma-hcl', 'ma-hcl-v2']:
                    # SimMAHGN and MA-HCL need the full heterogeneous edge dictionary
                    hetero_graph_structure = getattr(data, 'edge_index_dict', None)
                    if hetero_graph_structure is None and isinstance(data, dict):
                        # Nested in 'graph' attribute from load_data
                        if 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                             hetero_graph_structure = data['graph'].edge_index_dict
                        else:
                             hetero_graph_structure = data.get('edge_index_dict')
                             
                    if hetero_graph_structure is None:
                         # Fallback to bipartite edge_index if no hetero data
                         hetero_graph_structure = edge_index
                    loss, bpr, cl, reg = model.calculate_loss(hetero_graph_structure, users, pos_items, neg_items)
                else:
                    loss, bpr, reg, ssl = model.calculate_loss(graph_structure, users, pos_items, neg_items)
            elif hasattr(model, 'bpr_loss'):
                # NGCF/NCL style
                loss, reg = model.bpr_loss(users, pos_items, neg_items, edge_index)
                loss = loss + args.weight_decay * reg
            else:
                if isinstance(model, (SimGCL, CGRC, BIGCF, IGCL)):
                    if isinstance(model, CGRC):
                        user_emb, item_emb = model(data['adj_norm'], item_content)
                    else:
                        user_emb, item_emb = model(data['adj_norm'])
                else:
                    user_emb, item_emb = model(edge_index)
                
                pos_scores = (user_emb[users] * item_emb[pos_items]).sum(dim=1)
                neg_scores = (user_emb[users] * item_emb[neg_items]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        else:
            pbar.set_postfix({'loss': f"{total_loss:.4f}"})
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            edge_index_dict = None
            if isinstance(data, dict):
                edge_index_dict = data.get('edge_index_dict')
                if edge_index_dict is None and 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                    edge_index_dict = data['graph'].edge_index_dict
            elif hasattr(data, 'edge_index_dict'):
                edge_index_dict = data.edge_index_dict
            
            adj_norm = data.get('adj_norm') if isinstance(data, dict) else getattr(data, 'adj_norm', None)
            
            metrics = evaluate(model, test_dict, train_dict, n_items, edge_index, device=device, adj_norm=adj_norm,
                               re_ranker=re_ranker, rerank_strategy=args.rerank, eval_protocol=args.eval_protocol,
                               cold_users=cold_users, edge_index_dict=edge_index_dict)


            
            pbar.set_postfix({
                'loss': f"{total_loss:.4f}", 
                'R@10': f"{metrics.get('recall@10', 0):.4f}"
            })
            
            tqdm.write(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f} | "
                  f"R@10: {metrics.get('recall@10', 0):.4f} | NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
            
            recall_10 = metrics.get('recall@10', 0)
            if recall_10 > best_recall:
                best_recall = recall_10
                best_metrics = metrics.copy()
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics

def main():
    parser = argparse.ArgumentParser(description='Train CF/CL Models')
    parser.add_argument('--model', '-m', choices=['ngcf', 'simplex', 'directau', 'sgl', 'simgcl', 'ncl', 'lightgcl', 'cgrc', 'bigcf', 'igcl', 'xsimgcl', 'ma_hgn', 'sim-mahgn', 'ma-hcl', 'ma-hcl-v2', 'hetgnn'],
                        default='ngcf', help='Model to train')
    parser.add_argument('--data-path', default='data/processed', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # LightGCL-specific parameters
    parser.add_argument('--svd-q', type=int, default=20, help='LightGCL: SVD components (default: 20)')
    parser.add_argument('--ssl-weight', type=float, default=0.1, help='LightGCL: SSL loss weight (default: 0.1)')
    parser.add_argument('--temp', type=float, default=0.2, help='LightGCL: Temperature for contrastive loss (default: 0.2)')
    
    # Embedding arguments
    parser.add_argument('--embedding', choices=['random', 'phobert', 'tfidf', 'vndoc', 'bge-m3', 'gte', 'e5-large', 'e5-base', 'vn-sbert'], default='random',
                        help='Embedding initialization')
    parser.add_argument('--augment', choices=['none', 'llmrec'], default='none',
                        help='Graph augmentation strategy')
    parser.add_argument('--semantic-id-bits', type=int, default=0,
                        help='Number of semantic ID codebooks (0 to disable)')
    parser.add_argument('--denoise-ratio', type=float, default=0.0,
                        help='Ratio of noisy interactions to prune per batch (0.0 to disable)')
    parser.add_argument('--rerank', choices=['none', 'mmr', 'calib'], default='none',
                        help='Post-processing re-ranking strategy')
    parser.add_argument('--eval-protocol', choices=['full', 'loo100', 'cold'], default='full',
                        help='Evaluation protocol: full (all items), loo100 (leave-one-out + 100 neg), cold (cold-start users)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save metrics in JSON format')
    parser.add_argument('--neg-ratio', type=int, default=1,
                        help='Number of negative samples per positive sample (default: 1)')
    parser.add_argument('--cold-start', type=int, default=10,
                        help='Number of cold-start users to evaluate (default: 10)')
    parser.add_argument('--social-weight', type=float, default=1.0,
                        help='Weight for social reply edges in adjacency matrix (default: 1.0)')
    parser.add_argument('--graph-type', choices=['bipartite', 'hetero', 'article', 'category'], default='bipartite',
                        help='Graph type: bipartite, hetero, article (Article-Augmented), or category (Category Hubs)')
    # SimMAHGN specific arguments
    parser.add_argument('--emb-dim', type=int, default=64, help='Embedding dimension for SimMAHGN')
    parser.add_argument('--cl-rate', type=float, default=0.1, help='Contrastive loss rate for SimMAHGN')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for SimMAHGN')

    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print(f"Training {args.model.upper()}")
    print(f"Embedding Initialization: {args.embedding.upper()}")
    print(f"Graph Type: {args.graph_type.upper()}")
    print(f"Data Path: {args.data_path}")
    print("=" * 60)
    
    # Switch data path based on graph type
    if args.graph_type == 'hetero':
        hetero_path = Path(args.data_path) / 'full_hetero_graph.pt'
        if not hetero_path.exists():
            hetero_path = Path(args.data_path) / 'all_graphs' / 'full_hetero_graph.pt'
            
        if hetero_path.exists():
            print(f"  Loading Heterogeneous Graph from: {hetero_path}")
            data = load_data(str(hetero_path))
        else:
            print(f"  Warning: {hetero_path} not found. Falling back to default bipartite graph.")
            data = load_data(args.data_path)
    elif args.graph_type == 'category':
        cat_path = Path(args.data_path) / 'all_graphs' / 'category_graph.pt'
        if not cat_path.exists():
             cat_path = Path(args.data_path) / 'category_graph.pt'
             
        if cat_path.exists():
            print(f"  Loading Category-Augmented Graph from: {cat_path}")
            data = load_data(str(cat_path))
        else:
            print(f"  Warning: {cat_path} not found. Falling back.")
            data = load_data(args.data_path)

    elif args.graph_type == 'article':
        # Load full hetero graph as base to ensure consistent dimensions
        hetero_path = Path(args.data_path) / 'all_graphs' / 'full_hetero_graph.pt'
        if hetero_path.exists():
            print(f"  Loading Full Base Graph from: {hetero_path}")
            data = load_data(str(hetero_path))
            
            # REMOVE Social Edges to ensure purely Article-Augmented experiment
            if isinstance(data, dict):
                 if 'edge_index_dict' in data and ('user', 'replied_to', 'user') in data['edge_index_dict']:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['edge_index_dict'][('user', 'replied_to', 'user')]
            elif hasattr(data, 'edge_index_dict'):
                 # PyG HeteroData
                 if ('user', 'replied_to', 'user') in data.edge_types:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['user', 'replied_to', 'user']

        else:
             print("Full graph not found, falling back (might fail dimensions)")
             data = load_data(args.data_path)
        
        # Load auxiliary Article-Article graph
        article_path = Path(args.data_path) / 'all_graphs' / 'article_article_graph_users.pt'
        if article_path.exists():
            print(f"  Loading Article-Article Edges from: {article_path}")
            article_data = torch.load(article_path, weights_only=False)
            data['article_edge_index'] = article_data.edge_index
        else:
            print(f"  Warning: {article_path} not found. Using Bipartite only.")
    else:
        # Load data (default bipartite)
        data = load_data(args.data_path)
    n_users, n_items = data['n_users'], data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(data['train_pairs'])}, Test users: {len(data['test_dict'])}")
    
    # Identify cold-start users (users with <= 3 training interactions)
    train_dict = data['train_dict']
    cold_threshold = 3
    cold_users = {u for u, items in train_dict.items() if len(items) <= cold_threshold}
    print(f"Cold-start users (≤{cold_threshold} interactions): {len(cold_users)}")
    

    # Precompute adj_norm for SimGCL / CGRC / BIGCF / IGCL / XSimGCL
    # Precompute adj_norm for graph-based models
    graph_models = ['simgcl', 'cgrc', 'bigcf', 'igcl', 'xsimgcl', 'sgl', 'ncl', 'lightgcl', 'directau', 'sim-mahgn', 'ma-hcl', 'ma-hcl-v2']
    if args.model in graph_models:
        print(f"\nPrecomputing normalized adjacency for {args.model.upper()}...")
        
        augmented_pairs = None
        # Fix ambiguous boolean evaluation for Tensors
        item_item_edges = data.get('article_edge_index')
        if item_item_edges is None:
            item_item_edges = data.get('user_category_edge_index')
        if item_item_edges is None:
            item_item_edges = data.get('cat_article_edges')
        
        if args.augment == 'llmrec':
            augment_path = Path('data/processed/augmented_edges.pt')
            if augment_path.exists():
                print(f"  Loading augmented interactions from {augment_path}...")
                aug_data = torch.load(augment_path, weights_only=False)
                augmented_pairs = aug_data.get('augmented_pairs', None)
                item_item_edges = aug_data.get('item_item_edges', item_item_edges) 
            else:
                print(f"  Warning: {augment_path} not found. Running without augmentation.")
        
        # Check for weights in the data
        edge_weights = data.get('edge_weight')
        if edge_weights is not None:
            if isinstance(edge_weights, torch.Tensor):
                edge_weights = edge_weights.numpy()
            print(f"  Found {len(edge_weights)} edge weights. Using weighted adjacency.")
        
        # Check for social edges
        user_user_edges = None
        social_graph_path = Path(args.data_path) / 'full_hetero_graph.pt'
        if social_graph_path.exists():
            print(f"  Loading social signals from {social_graph_path}...")
            social_data = torch.load(social_graph_path, weights_only=False)
            if isinstance(social_data, dict) and 'edge_index_dict' in social_data:
                # If saved as dict
                edges_dict = social_data['edge_index_dict']
                user_user_edges = edges_dict.get(('user', 'replied_to', 'user'))
            elif hasattr(social_data, 'edge_index_dict'):
                # If PyG HeteroData
                user_user_edges = social_data['user', 'replied_to', 'user'].edge_index
        
        data['adj_norm'] = compute_normalized_adjacency(n_users, n_items, data['train_pairs'], device, 
                                                       item_item_edges, user_user_edges, 
                                                       edge_weights=edge_weights, social_weight=args.social_weight)
        data['augmented_pairs'] = augmented_pairs # Store for LightGCL
    
    # Load pretrained embeddings if requested
    
    # Load pretrained embeddings if requested
    pretrained_emb = load_pretrained_embeddings(args.embedding, n_items, args.hidden_dim, device)
    
    # Generate Semantic IDs if requested
    semantic_ids = None
    if args.semantic_id_bits > 0:
        if pretrained_emb is not None:
            print(f"\nGenerating Semantic IDs ({args.semantic_id_bits} stages)...")
            semantic_ids, _ = generate_semantic_ids(pretrained_emb, n_codebooks=args.semantic_id_bits, codebook_size=32)
            semantic_ids = semantic_ids.to(device)
            print(f"  Generated IDs shape: {semantic_ids.shape}")
        else:
            print("\n  Warning: Cannot generate Semantic IDs without pretrained embeddings. Skipping.")
            
    # Load User Priors if requested
    user_priors = None
    if args.model == 'xsimgcl':
        prior_path = Path('data/processed/user_priors.pt')
        if prior_path.exists():
            print(f"\nLoading User Priors from {prior_path}...")
            priors = torch.load(prior_path, weights_only=False).to(device)
            # Alignment: priors might have fewer users than GNN
            if priors.shape[0] < n_users:
                print(f"  Padding User Priors: {priors.shape[0]} -> {n_users}")
                user_priors = torch.zeros((n_users, priors.shape[1]), device=device)
                user_priors[:priors.shape[0]] = priors
            else:
                user_priors = priors
            print(f"  Final Priors shape: {user_priors.shape}")
        else:
            print(f"\n  Warning: User Priors not found at {prior_path}. Running without priors.")
            
    # Load Re-ranker if requested
    re_ranker = None
    if args.rerank != 'none':
        print("\nInitializing Re-ranker...")
        # Get idx2item from pkl
        with open('data/processed/lightgcl_data.pkl', 'rb') as f:
            idx_data = pickle.load(f)
            idx2item = idx_data['idx2item']
        
        categories, n_cats = load_item_categories(idx2item)
        re_ranker = CalibratedReRanker(categories, alpha=0.5, lambda_mmr=0.5)
        print(f"  Loaded {n_cats} categories for {len(categories)} items.")
    
    # Create model
    if args.model == 'ngcf':
        model = NGCF(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to NGCF")
             
    elif args.model == 'simplex':
        model = SimpleX(n_users, n_items, embedding_dim=args.hidden_dim).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to SimpleX")
             
    elif args.model == 'directau':
        model = DirectAU(n_users, n_items, embedding_dim=args.hidden_dim).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to DirectAU")
             
    elif args.model == 'lightgcl':
        model = LightGCLWrapper(n_users, n_items, embed_dim=args.hidden_dim, n_layers=args.n_layers, 
                                device=args.device, svd_q=args.svd_q, ssl_weight=args.ssl_weight, temp=args.temp)
        model.setup(data['train_pairs'], data.get('augmented_pairs', None)) # Computes SVD
        
        if pretrained_emb is not None:
            # For LightGCL, items are initialized in self.model.E_i_0
            model.model.E_i_0.data.copy_(pretrained_emb)
            print("  Transferred embeddings to LightGCL (E_i_0)")
            
    elif args.model == 'cgrc':
        model = CGRC(n_users, n_items, embedding_dim=args.hidden_dim, 
                     content_dim=pretrained_emb.shape[1] if pretrained_emb is not None else args.hidden_dim,
                     n_layers=args.n_layers).to(device)
        if pretrained_emb is not None:
             # Initialize item embeddings from pretrained
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to CGRC")
             
    elif args.model == 'bigcf':
        model = BIGCF(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to BIGCF")
             
    elif args.model == 'sim-mahgn':
        model = SimMAHGN(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.emb_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            ssl_weight=args.cl_rate,
            eps=args.eps,
            temp=args.temp,
            n_categories=data.get('n_categories', 0)
        ).to(device)
        if pretrained_emb is not None:
             model.item_emb.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to SimMAHGN")
             
    elif args.model == 'ma-hcl':
        from src.models.ma_hcl import MAHCL
        model = MAHCL(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.emb_dim,
            n_layers=args.n_layers,
            ssl_weight=args.cl_rate,
            eps=args.eps,
            temp=args.temp,
            n_categories=data.get('n_categories', 0)
        ).to(device)
        if pretrained_emb is not None:
             model.item_emb.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to MA-HCL")
             
    elif args.model == 'ma-hcl-v2':
        from src.models.ma_hcl_v2 import MAHCLV2
        model = MAHCLV2(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.emb_dim,
            n_layers=args.n_layers,
            ssl_weight=args.cl_rate,
            eps=args.eps,
            temp=args.temp,
            n_authors=data.get('n_authors', 0)
        ).to(device)
        if pretrained_emb is not None:
             model.item_emb.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to MA-HCL-V2")
    elif args.model == 'igcl':
        model = IGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers,
                     ssl_weight=args.ssl_weight, temp=args.temp).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to IGCL")
             
    elif args.model == 'ma_hgn':
        model = MAHGN(n_users, n_items, args.hidden_dim, args.n_layers, args.dropout).to(device)

    elif args.model == 'hetgnn':
        model = HetGNN(
            n_users=n_users,
            n_items=n_items,
            n_categories=data.get('n_categories', 15), # Default fallback
            embedding_dim=args.hidden_dim,
            n_layers=args.n_layers
            # Use defaults for others: heads=4, dropout=0.1
        ).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to HetGNN")
        
    elif args.model == 'xsimgcl':
        model = XSimGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers,
                         ssl_weight=args.ssl_weight, temp=args.temp).to(device)
        if semantic_ids is not None:
            model.semantic_layer = SemanticEmbeddingLayer(args.semantic_id_bits, 32, args.hidden_dim).to(device)
            model.semantic_ids = semantic_ids
            print("  Initialized XSimGCL with Semantic ID Fusion (Pillar 1)")
            
        if user_priors is not None:
            model.user_prior_layer = UserPriorLayer(user_priors.shape[1], args.hidden_dim).to(device)
            model.user_priors = user_priors
            print("  Initialized XSimGCL with User Prior Fusion (Pillar 2)")
            
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to XSimGCL")
             
    # Add other models as needed...
    else:
        # Combined logic for models with standard item_embedding
        if args.model == 'sgl':
            model = SGL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        elif args.model == 'simgcl':
            model = SimGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        elif args.model == 'ncl':
             model = NCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        
        # Inject embeddings
        if pretrained_emb is not None:
            model.item_embedding.weight.data.copy_(pretrained_emb)
            print(f"  Transferred embeddings to {args.model.upper()}")

    # Ensure model is on the correct device if not handled internally (e.g., LightGCL)
    if args.model != 'lightgcl':
        model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    if hasattr(model, 'forward') and pretrained_emb is not None:
        model.item_content = pretrained_emb # Attach for evaluate
        
    # Run training
    best_metrics = train_model(model, data, args, device, item_content=pretrained_emb, 
                               semantic_ids=semantic_ids, user_priors=user_priors, 
                               re_ranker=re_ranker, cold_users=cold_users)

    p = Path(args.data_path)
    # Priority: If graph is in a variant folder (strict_g2, etc.), use that folder name
    # Otherwise fallback to the filename stem
    if p.parent.name in ["strict_g1", "strict_g2", "strict_g3", "regular_g2", "enhanced_v1", "enhanced_v2"]:
        graph_name = p.parent.name
    else:
        graph_name = p.stem
        
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_path = f"models/{args.model}_{graph_name}_{timestamp}.pt"
    
    # Ensure directory exists
    Path("models").mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items,
        'config': vars(args),
        'metrics': best_metrics
    }, save_path)

    print(f"\nModel saved: {save_path}")
    print(f"\nBest Metrics:")
    print(f"  Recall@1:   {best_metrics.get('recall@1', 0):.4f}")
    print(f"  Recall@5:   {best_metrics.get('recall@5', 0):.4f}")
    print(f"  Recall@10:  {best_metrics.get('recall@10', 0):.4f}")
    print(f"  NDCG@1:     {best_metrics.get('ndcg@1', 0):.4f}")
    print(f"  NDCG@5:     {best_metrics.get('ndcg@5', 0):.4f}")
    print(f"  NDCG@10:    {best_metrics.get('ndcg@10', 0):.4f}")
    print(f"  HitRate@1:  {best_metrics.get('hitrate@1', 0):.4f}")
    print(f"  HitRate@5:  {best_metrics.get('hitrate@5', 0):.4f}")
    print(f"  HitRate@10: {best_metrics.get('hitrate@10', 0):.4f}")
    print(f"  MRR:        {best_metrics.get('mrr', 0):.4f}")
    if 'entropy' in best_metrics:
        print(f"  Entropy:    {best_metrics['entropy']:.4f}")
    
    # Save results json if requested
    if args.save_results:
        import json
        
        # Ensure results go to results/ folder
        results_path = Path(args.save_results)
        if not results_path.parent.exists() or results_path.parent == Path('.'):
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / results_path.name
        
        # Helper to serializable
        def convert(o):
            if isinstance(o, np.float32): return float(o)
            return o
            
        with open(results_path, 'w') as f:
            json.dump(best_metrics, f, default=convert)
        print(f"Saved metrics to {results_path}")


if __name__ == '__main__':
    main()
