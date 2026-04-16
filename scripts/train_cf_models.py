#!/usr/bin/env python3
"""
Train Collaborative Filtering / Contrastive Learning Models
Supports: SimpleX, DirectAU, SGL, SimGCL, NCL, LightGCL
"""
import os
import sys
import argparse
import re
import torch
import numpy as np
import pickle
import random
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
try:
    from tqdm.auto import tqdm
except Exception:
    # Fallback keeps training loop functional when tqdm is unavailable.
    def tqdm(iterable, *args, **kwargs):
        return iterable

from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cf_data_loader import load_data, compute_normalized_adjacency
from src.data.embedding_loader import load_pretrained_embeddings, SemanticEmbeddingLayer, UserPriorLayer
from src.training.cf_evaluator import load_item_categories
from src.training.cf_trainer import train_model
from src.models.lightgcl_wrapper import LightGCLWrapper

from src.models import LightGCL, SimGCL, XSimGCL, MAHGN, LightGCN
from src.models.ma_hcl import MAHCL
from src.models.bigcf import BIGCF
from src.models.semantic_id import generate_semantic_ids
from src.inference.re_ranker import CalibratedReRanker

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # Đảm bảo thuật toán convolution/GNN chạy cố định
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f">>> Global seed set to: {seed}")

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


def load_data(data_path, min_interactions=2, split_strategy='random'):
    """Load and process data for CF models."""
    import pandas as pd
    from torch_geometric.data import HeteroData

    def writable_cache_path(source_path, filename):
        source_path = Path(source_path)
        if str(source_path).startswith('/kaggle/input'):
            try:
                relative_source = source_path.relative_to('/kaggle/input')
            except ValueError:
                relative_source = Path(source_path.name)
            return Path('/kaggle/working') / 'cf_cache' / relative_source / filename
        if source_path.is_file():
            return source_path.parent / filename
        return source_path / filename
    
    # Check if data_path is already a file or a directory
    p = Path(data_path)
    cache_filename = f'cf_cache_{split_strategy}.pt'
    if p.is_file():
        cache_path = p
    else:
        cache_path = p / cache_filename
        if not cache_path.exists() and split_strategy == 'random':
            if (p / 'graph_with_negatives.pt').exists():
                cache_path = p / 'graph_with_negatives.pt'
            elif (p / 'user_article_graph.pt').exists():
                cache_path = p / 'user_article_graph.pt'
            elif (p / 'full_hetero_graph.pt').exists():
                cache_path = p / 'full_hetero_graph.pt'
            elif (p / 'category_graph.pt').exists():
                cache_path = p / 'category_graph.pt'
        
    if cache_path.exists():
        print(f"  Loading cached data from {cache_path} (Strategy: {split_strategy})...")
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
            
            # ATTEMPT TO FIND MASTER SPLITS
            master_splits = None
            if isinstance(data, dict) and 'splits' in data:
                # The loaded file IS the master split file
                master_splits = data['splits']
            else:
                # Look for it in the directory
                master_split_path = cache_path.parent / 'graph_with_negatives.pt'
                if master_split_path.exists():
                    print(f"  Synchronizing splits with master: {master_split_path}")
                    master_data = torch.load(master_split_path, weights_only=False)
                    if isinstance(master_data, dict) and 'splits' in master_data:
                        master_splits = master_data['splits']
            
            edge_index = None
            n_users = 0
            n_items = 0
            n_categories = 0
            if isinstance(data, HeteroData):
                if ('user', 'comments', 'article') in data.edge_types:
                    edge_index = data['user', 'comments', 'article'].edge_index
                else:
                    # Fallback to first user-item relation
                    edge_index = next(iter(data.edge_index_dict.values()))
                n_users = data['user'].num_nodes
                n_items = data['article'].num_nodes
                if 'category' in data.node_types:
                    n_categories = data['category'].num_nodes
            elif isinstance(data, Data):
                edge_index = data.edge_index
                # Search for n_users if encoded in edge_index or specified in data
                n_users = data.get('n_users') or data.get('num_users') or 0
                n_items = data.get('n_items') or data.get('num_articles') or data.get('num_items') or 0
            elif isinstance(data, dict):
                # Get interactions from either edge_index or existing train_pairs
                if 'edge_index' in data:
                    edge_index = data['edge_index']
                elif 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                    # HeteroGraph in dict
                    if ('user', 'comments', 'article') in data['graph'].edge_types:
                        edge_index = data['graph']['user', 'comments', 'article'].edge_index
                    else:
                        edge_index = next(iter(data['graph'].edge_index_dict.values()))
                elif 'splits' in data:
                    pass
                else:
                    return data
            
            # Get n_users/n_items from dictionary if not set by graph extraction
            if n_users == 0: 
                n_users = data.get('n_users') or data.get('num_users') or 0
                if isinstance(data, dict) and 'num_users' in data: n_users = data['num_users']
            if n_items == 0: 
                n_items = data.get('n_items') or data.get('num_articles') or data.get('num_items') or 0
                if isinstance(data, dict) and 'num_articles' in data: n_items = data['num_articles']

            if master_splits is not None:
                print("  Using pre-defined splits from format (splits dictionary)...")
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
                
                # IMPORTANT: Synchronize edge_index with train_pairs to prevent leakage
                # Message passing should ONLY use training edges.
                new_edge_index = torch.stack([
                    torch.tensor([u for u, i in train_pairs], dtype=torch.long),
                    torch.tensor([i for u, i in train_pairs], dtype=torch.long)
                ], dim=0)

                data_dict = {
                    'n_users': n_users,
                    'n_items': n_items,
                    'n_categories': n_categories if 'n_categories' in locals() else (data.get('n_categories') or 0),
                    'train_pairs': train_pairs,
                    'test_dict': test_dict,
                    'train_dict': train_dict,
                    'edge_index': new_edge_index,
                    'graph': data
                }
                
                # If hetero graph, also prune internal edge_index_dict
                if isinstance(data, HeteroData) or (isinstance(data, dict) and 'graph' in data and isinstance(data['graph'], HeteroData)):
                    target_graph = data if isinstance(data, HeteroData) else data.get('graph')
                    if target_graph is None:
                        return data_dict
                
                # Update both directions for symmetry and proper message passing
                    ua_key = ('user', 'comments', 'article')
                    rev_ua_key = ('article', 'rev_comments', 'user')
                    
                    if ua_key in target_graph.edge_types:
                         target_graph[ua_key].edge_index = new_edge_index
                    
                    if rev_ua_key in target_graph.edge_types:
                         target_graph[rev_ua_key].edge_index = torch.stack([new_edge_index[1], new_edge_index[0]], dim=0)
                    
                    # Also check for 'interacts' naming convention
                    i_key = ('user', 'interacts', 'item')
                    rev_i_key = ('item', 'rev_interacts', 'user')
                    if i_key in target_graph.edge_types:
                         target_graph[i_key].edge_index = new_edge_index
                    if rev_i_key in target_graph.edge_types:
                         target_graph[rev_i_key].edge_index = torch.stack([new_edge_index[1], new_edge_index[0]], dim=0)
                         
                    data_dict['edge_index_dict'] = target_graph.edge_index_dict

                return data_dict

            if edge_index is not None:
                interactions = list(set(zip(edge_index[0].tolist(), edge_index[1].tolist())))
            
            # Get n_users/n_items from dictionary if not set by graph extraction
            if n_users == 0: 
                n_users = data.get('n_users') or data.get('num_users') or 0
            if n_items == 0: 
                n_items = data.get('n_items') or data.get('num_articles') or data.get('num_items') or 0

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
            
            # Filter edge_index_dict to only contain train edges
            # The original edge_index_dict contains ALL edges (train + test).
            if isinstance(data, HeteroData):
                train_edge_index = torch.stack([
                    torch.tensor([u for u, i in data_dict['train_pairs']], dtype=torch.long),
                    torch.tensor([i for u, i in data_dict['train_pairs']], dtype=torch.long)
                ], dim=0)
                
                # Update ALL naming conventions for user-item interaction edges
                edge_type_pairs = [
                    (('user', 'comments', 'article'), ('article', 'rev_comments', 'user')),
                    (('user', 'interacts', 'item'), ('item', 'rev_interacts', 'user')),
                    (('user', 'interacts', 'article'), ('article', 'rev_interacts', 'user')),
                ]
                
                for ua_key, rev_ua_key in edge_type_pairs:
                    if ua_key in data.edge_types:
                        data[ua_key].edge_index = train_edge_index
                    if rev_ua_key in data.edge_types:
                        data[rev_ua_key].edge_index = torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
                
                data_dict['edge_index_dict'] = data.edge_index_dict

                
            return data_dict



        return data
    
    print(f"Processing data from raw CSVs (Strategy: {split_strategy})...")
    raw_replies = None
    reply_candidates = [
        Path('data/raw/replies.csv'),
        Path(data_path) / 'replies.csv',
        Path(data_path).parent / 'raw' / 'replies.csv',
        Path('/kaggle/input/vnexpress-data-v2/replies.csv'),
    ]

    for candidate in reply_candidates:
        if candidate.exists():
            raw_replies = candidate
            break

    if raw_replies is None:
        kaggle_input_root = Path('/kaggle/input')
        if kaggle_input_root.exists():
            found = list(kaggle_input_root.rglob('replies.csv'))
            if found:
                raw_replies = found[0]

    if raw_replies is None:
        raise FileNotFoundError(
            "Could not locate replies.csv. Please attach vnexpress-data-v2 dataset on Kaggle."
        )

    print(f"  Using replies file: {raw_replies}")
    replies = pd.read_csv(raw_replies)
    def extract_vnexpress_id(url):
        # Trích xuất số ID trước đuôi .html (ví dụ: 4987772)
        match = re.search(r'-(\d+)\.html', str(url))
        return int(match.group(1)) if match else 0

    print("   [INFO] Extracting Article IDs as temporal proxy...")
    replies['proxy_time'] = replies['article_url'].apply(extract_vnexpress_id)
    
    # Sắp xếp toàn bộ dữ liệu theo ID bài báo trước khi làm bất cứ việc gì khác
    # Điều này cực kỳ quan trọng để split 'time' không bị rò rỉ dữ liệu
    replies = replies.sort_values('proxy_time').reset_index(drop=True)
    print(f"   [SUCCESS] Data sorted by Article ID. Range: {replies['proxy_time'].min()} to {replies['proxy_time'].max()}")
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
    
    if split_strategy == 'time':
        print("   [INFO] Strategy: TIME-BASED SPLIT")
        print("   Preserving CSV order")
        
        # 1. Deduplicate giữ nguyên thứ tự (Keep First)
        replies = replies.drop_duplicates(subset=['user_idx', 'item_idx'], keep='first')
        
        # 2. Tạo list interactions theo dòng chảy thời gian
        interactions = list(zip(replies['user_idx'].values, replies['item_idx'].values))
        
        # 3. Cắt theo mốc thời gian (80% đầu là train, 20% sau là test)
        split_idx = int(len(interactions) * 0.8)
        train_pairs = interactions[:split_idx]
        test_pairs = interactions[split_idx:]
        
        print(f"   -> Split Point: Interaction #{split_idx}")
        print(f"   -> Train: {len(train_pairs)} (Past) | Test: {len(test_pairs)} (Future)")
        
    else: # split_strategy == 'random'
        print("   [INFO] Strategy: RANDOM SPLIT (Legacy)")
        print("   -> Shuffling interactions...")
        
        # Logic cũ: Dùng set (mất thứ tự) và shuffle
        interactions = list(zip(replies['user_idx'].values, replies['item_idx'].values))
        interactions = list(set(interactions))
        np.random.seed(42)
        np.random.shuffle(interactions)
        
        split_idx = int(len(interactions) * 0.8)
        train_pairs = interactions[:split_idx]
        test_pairs = interactions[split_idx:]
        print(f"   -> Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    
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
    
    if 'split_idx' in locals():
        train_replies_df = replies.iloc[:split_idx].copy()
    else:
        train_replies_df = replies.iloc[:int(len(replies)*0.8)].copy()

    train_replies_df['parent_idx'] = train_replies_df['parent_user_id'].map(user_map)
    valid_social = train_replies_df.dropna(subset=['user_idx', 'parent_idx'])
    
    if len(valid_social) > 0:
        social_src = torch.tensor(valid_social['user_idx'].values.astype('int64'), dtype=torch.long)
        social_dst = torch.tensor(valid_social['parent_idx'].values.astype('int64'), dtype=torch.long)
        user_user_edges = torch.stack([
            torch.cat([social_src, social_dst]),
            torch.cat([social_dst, social_src])
        ], dim=0)
        user_user_edges = torch.unique(user_user_edges, dim=1)
        print(f"   -> Created {user_user_edges.shape[1]} safe social edges from past interactions.")
    else:
        user_user_edges = None
        print("   -> No valid social edges found in training set.")
    
    # Create edge index from TRAIN PAIRS ONLY (no data leakage)
    train_users = torch.tensor([u for u, i in train_pairs], dtype=torch.long)
    train_items = torch.tensor([i for u, i in train_pairs], dtype=torch.long)
    
    # Bipartite graph edge index with offset
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
        'user_user_edges': user_user_edges,
    }
    
    # Save cache to a writable location (Kaggle input is read-only).
    save_path = writable_cache_path(p, cache_filename)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    print(f"   -> Processed data saved to {save_path}")

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


def load_pretrained_embeddings(embedding_type, n_items, target_dim, device='cpu', train_item_indices=None, data_path=None, articles_path=None):
    """Load pretrained embeddings and project to target dimension."""
    if embedding_type == 'random':
        return None
        
    print(f"\nLoading {embedding_type} embeddings...")
    if articles_path:
        articles_path = Path(articles_path)
    embeddings = None

    # Helper to find file in multiple locations
    def resolve_path(filename, search_paths):
        for p in search_paths:
            if p is None: continue
            candidate = Path(p) / filename
            if candidate.exists():
                return candidate
        return None

    # Common locations to search
    search_dirs = [
        'checkpoints', 
        'data/raw', 
        'data',
        '/kaggle/working/checkpoints',
        '/kaggle/input/vnexpress-news-dataset'
    ]
    if data_path:
        search_dirs.insert(0, Path(data_path).parent) # e.g., /kaggle/input/vnexpress-graph-processed
        search_dirs.insert(0, Path(data_path))
    
    # 1. Resolve Embedding Path
    emb_filename = None
    if embedding_type == 'phobert':
        emb_filename = 'phobert_article_embeddings.pt'
    elif embedding_type == 'vndoc':
        emb_filename = 'vndoc_article_embeddings.pt'
    elif embedding_type in ['bge-m3', 'gte', 'e5-large', 'e5-base', 'vn-sbert', 'vndoc']:
        name_map = {
            'bge-m3': 'bge-m3_article_embeddings.pt',
            'gte': 'gte-multilingual_article_embeddings.pt',
            'e5-large': 'e5-large_article_embeddings.pt',
            'e5-base': 'e5-base_article_embeddings.pt',
            'vn-sbert': 'vietnamese-sbert_article_embeddings.pt',
            'vndoc': 'vietnamese-document-embedding.pt'
        }
        emb_filename = name_map[embedding_type]
    
    if emb_filename:
        # Special check for checkpoints folder
        path = resolve_path(emb_filename, ['checkpoints'] + search_dirs)
        
        if path and path.exists():
             embeddings = torch.load(path, map_location='cpu')
             print(f"  Loaded {embedding_type} embeddings from {path}: {embeddings.shape}")
        else:
             print(f"  Warning: {emb_filename} not found locally.")
             
             # Fallback: Auto-Download & Encode
             print("  Attempting to download and encode on-the-fly (this may take time)...")
             try:
                 from sentence_transformers import SentenceTransformer
                 import pandas as pd
                 
                 # HF Model Names
                 hf_map = {
                     'bge-m3': 'BAAI/bge-m3',
                     'gte': 'thenlper/gte-large', 
                     'e5-large': 'intfloat/multilingual-e5-large',
                     'e5-base': 'intfloat/multilingual-e5-base',
                     'vn-sbert': 'keepitreal/vietnamese-sbert',
                     'vndoc': 'dangvantuan/vietnamese-document-embedding'
                 }
                 
                 model_name = hf_map.get(embedding_type)
                 if not model_name:
                     print(f"  No HF model mapping for {embedding_type}. Fallback to Random.")
                     return None
                     
                 # Load Articles
                 # Priority: Explicit Path > Resolved Path > Default Fallback
                 if articles_path:
                     if Path(articles_path).exists():
                         articles_path = Path(articles_path)
                         print(f"  Using explicit articles file: {articles_path}")
                     else:
                         print(f"  Warning: Explicit articles path not found: {articles_path}")
                     print(f"  ... Falling back to auto-search in: {search_dirs}")
                     if not articles_path:
                         articles_path = resolve_path('articles.csv', search_dirs)
                     if (not articles_path) and data_path: articles_path = Path(data_path).parent / 'articles.csv'
                     if not articles_path: articles_path = Path('data/raw/articles.csv')
                 else:
                     if not articles_path:
                         articles_path = resolve_path('articles.csv', search_dirs)
                     if (not articles_path) and data_path: articles_path = Path(data_path).parent / 'articles.csv'
                     if not articles_path: articles_path = Path('data/raw/articles.csv')

                 if not articles_path or not articles_path.exists():
                     print("  Standard paths failed. Attempting deep search for 'articles.csv'...")
                     potential_roots = ['/kaggle/input', 'data']
                     found = False
                     for root in potential_roots:
                         if not os.path.exists(root): continue
                         for r, d, f in os.walk(root):
                             if 'articles.csv' in f:
                                 articles_path = Path(r) / 'articles.csv'
                                 print(f"  Found articles.csv via deep search: {articles_path}")
                                 found = True
                                 break
                         if found: break

                 
                 if not articles_path or not articles_path.exists():
                     print(f"  Articles file not found. Cannot encode. Fallback to Random.")
                     return None
                     
                 df = pd.read_csv(articles_path)
                 print(f"  Loaded {len(df)} articles. Encoding with {model_name}...")
                 
                 # Prepare Text
                 text_col = 'abstract' if 'abstract' in df.columns else 'short_description'
                 if text_col not in df.columns:
                     print(f"  neither 'abstract' nor 'short_description' found. Using title only.")
                     df['text'] = df['title'].fillna('')
                 else:
                     df['text'] = df['title'].fillna('') + ' ' + df[text_col].fillna('')
                 
                 texts = df['text'].tolist()
                 
                 # Encode
                 model = SentenceTransformer(model_name, device=device)
                 emb_matrix = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
                 embeddings = emb_matrix.cpu()
                 
                 # Save for future
                 save_dir = Path('checkpoints')
                 save_dir.mkdir(exist_ok=True)
                 save_path = save_dir / emb_filename
                 torch.save(embeddings, save_path)
                 print(f"  Saved cached embeddings to {save_path}")
                 
             except ImportError:
                 print("  `sentence_transformers` not installed. Cannot auto-encode. Fallback to Random.")
                 return None
             except Exception as e:
                 print(f"  Auto-encoding failed: {e}. Fallback to Random.")
                 return None

    elif embedding_type == 'tfidf':
        print("   Computing TF-IDF embeddings (LSA)...")
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        try:
            texts = []
            
            if not articles_path:
                articles_path = resolve_path('articles.csv', search_dirs)
            
            if not articles_path:
                 articles_path = Path('data/raw/articles.csv')
            
            if not articles_path.exists():
                print(f"      Error: Articles file not found at {articles_path}")
                return None
                
            articles = pd.read_csv(articles_path)


            # Load mapping idx2item
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.data.dataloader_lightgcl import LightGCLDataLoader
            loader = LightGCLDataLoader(data_path if data_path else 'data')
            if loader.load_processed() is None:
                print("      Error: Could not load processed LightGCL data for mapping.")
                return None
            
            idx2item = loader.idx2item
            print(f"      Loaded mapping for {len(idx2item)} items")
            
            # Flatten loop for speedr
            article_map = dict(zip(articles['url'], zip(articles['title'], articles['short_description'])))
            
            for idx in range(n_items):
                url = idx2item.get(idx, None)
                if url and url in article_map:
                    title, desc = article_map[url]
                    title = str(title) if pd.notna(title) else ""
                    desc = str(desc) if pd.notna(desc) else ""
                    texts.append(f"{title} {desc}")
                else:
                    texts.append("")
            
            print(f"      Collected {len(texts)} texts. Computing TF-IDF...")
            
            vectorizer = TfidfVectorizer(max_features=10000, stop_words=None) 
            
            if train_item_indices is not None:
                print("      -> Fitting TF-IDF only on TRAIN items to prevent leakage...")
                train_texts = [texts[i] for i in train_item_indices if i < len(texts)]
                vectorizer.fit(train_texts)
                tfidf_matrix = vectorizer.transform(texts)
            else:
                print("      -> [WARNING] Fitting on ALL items (Data Snooping).")
                tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimension with SVD (LSA)
            print(f"      Reducing dimension {tfidf_matrix.shape[1]} -> {target_dim} with SVD...")
            svd = TruncatedSVD(n_components=target_dim, random_state=42)
            embeddings = torch.tensor(svd.fit_transform(tfidf_matrix), dtype=torch.float32)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            print(f"      Computed LSA embeddings: {embeddings.shape}")
            return embeddings.to(device)

        except Exception as e:
            print(f"      Error computing TF-IDF: {e}. Using random.")
            import traceback
            traceback.print_exc()
            return None
        
    if embeddings is not None:
        # Match item count (truncate or pad if needed)
        # Check item count
        curr_items, curr_dim = embeddings.shape
        
        if curr_items != n_items:
            print(f"  Warning: Embedding items ({curr_items}) != Dataset items ({n_items})")
            # If fewer embeddings, pad with random
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
             
        # Clamp values
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
    results = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg', 'hitrate', 'precision', 'map', 'f1'] for k in k_list}
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
            
            prec = hits / k
            rec = hits / len(test_items)
            results[f'recall@{k}'].append(rec)
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
            results[f'precision@{k}'].append(prec)
            
            # F1@k
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            results[f'f1@{k}'].append(f1)
            
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

    is_hetero_model = args.model in ['ma_hgn', 'sim-mahgn']
    graph_to_check = edge_index_dict if (is_hetero_model and edge_index_dict is not None) else edge_index
    
    if graph_to_check is not None:
        if isinstance(graph_to_check, dict):
            # Hetero graph: Look for user-article or user-item edges
            ua_edges = None
            for key, val in graph_to_check.items():
                if isinstance(key, tuple) and len(key) == 3:
                    if 'user' in str(key[0]).lower() and ('article' in str(key[2]).lower() or 'item' in str(key[2]).lower()):
                        ua_edges = val
                        break
            if ua_edges is not None:
                n_graph_interactions = ua_edges.size(1)
        else:
            # Bipartite graph: Count edges (might be symmetric 2*N or directed N)
            n_edges = graph_to_check.size(1)
            
            # If edges == 2 * train_pairs, it's symmetric (count half)
            if n_edges == len(train_pairs) * 2:
                n_graph_interactions = n_edges // 2
            elif n_edges == len(train_pairs):
                n_graph_interactions = n_edges
            else:
                # Unknown format: use unique pairs
                src, dst = graph_to_check[0].cpu(), graph_to_check[1].cpu()
                pairs = set(zip(src.tolist(), dst.tolist()))
                n_graph_interactions = len(pairs)
                
        n_train_interactions = len(train_pairs)
        
        if n_graph_interactions > n_train_interactions:
            print(f"\n" + "!"*60)
            print(f"CRITICAL LEAKAGE DETECTED!")
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
            if hasattr(model, 'calculate_loss'):
                if args.model in ['simgcl', 'bigcf', 'igcl', 'xsimgcl', 'lightgcn']:
                    graph_structure = data.get('adj_norm')
                    if graph_structure is None:
                        print("Warning: adj_norm missing in train loop, falling back to edge_index")
                        graph_structure = edge_index
                elif args.model == 'lightgcl':
                     graph_structure = data.get('adj_norm')
                elif args.model in ['ma_hgn', 'ma-hcl']:
                    graph_structure = getattr(data, 'edge_index_dict', None)
                    if graph_structure is None and isinstance(data, dict):
                        graph_structure = data.get('edge_index_dict')
                    
                    if graph_structure is None:
                         src, dst = edge_index
                         n_users_limit = data['n_users']
                         
                         mask = (src < n_users_limit) & (dst >= n_users_limit)
                         u_i_src = src[mask]
                         u_i_dst = dst[mask] - n_users_limit # Remove offset
                         
                         u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                         i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                         
                         graph_structure = {
                             ('user', 'interacts', 'item'): u_i_edges,
                             ('item', 'rev_interacts', 'user'): i_u_edges
                         }
                else:
                    graph_structure = edge_index
                
                
                if isinstance(model, XSimGCL):
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

                elif isinstance(model, MAHGN):
                    # MAHGN logic for heterogeneous graph structure
                    hetero_graph_structure = getattr(data, 'edge_index_dict', None)
                    if hetero_graph_structure is None and isinstance(data, dict):
                        if 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                             hetero_graph_structure = data['graph'].edge_index_dict
                        else:
                             hetero_graph_structure = data.get('edge_index_dict')
                             
                    # If still None, construct from bipartite edge_index
                    if hetero_graph_structure is None and edge_index is not None:
                         src, dst = edge_index
                         n_users_limit = data['n_users']
                         
                         # Filter user - item edges (src < n_users, dst >= n_users)
                         mask = (src < n_users_limit) & (dst >= n_users_limit)
                         u_i_src = src[mask]
                         u_i_dst = dst[mask] - n_users_limit
                         
                         u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                         i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                         
                         hetero_graph_structure = {
                             ('user', 'interacts', 'item'): u_i_edges,
                             ('item', 'rev_interacts', 'user'): i_u_edges
                         }
                    
                    loss, bpr, cl, reg = model.calculate_loss(hetero_graph_structure, users, pos_items, neg_items)

                elif isinstance(model, MAHCL):
                     # MAHCL uses graph_structure computed above
                     loss, bpr, cl, reg = model.calculate_loss(graph_structure, users, pos_items, neg_items)
                else:
                    loss, bpr, reg, ssl = model.calculate_loss(graph_structure, users, pos_items, neg_items)
            elif hasattr(model, 'bpr_loss'):
                # NGCF/NCL style
                loss, reg = model.bpr_loss(users, pos_items, neg_items, edge_index)
                loss = loss + args.weight_decay * reg
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
        
        # Evaluate only at the final epoch
        if epoch == args.epochs - 1:
            edge_index_dict = None
            if isinstance(data, dict):
                edge_index_dict = data.get('edge_index_dict')
                if edge_index_dict is None and 'graph' in data and hasattr(data['graph'], 'edge_index_dict'):
                    edge_index_dict = data['graph'].edge_index_dict
            elif hasattr(data, 'edge_index_dict'):
                edge_index_dict = data.edge_index_dict
            
            adj_norm = data.get('adj_norm') if isinstance(data, dict) else getattr(data, 'adj_norm', None)
            
            # Fallback for Hetero models on Homogeneous Data
            if edge_index_dict is None and args.model in ['ma_hgn', 'sim-mahgn', 'ma-hcl']:
                 if edge_index is not None:
                     src, dst = edge_index
                     n_users_limit = data['n_users']
                     
                     mask = (src < n_users_limit) & (dst >= n_users_limit)
                     u_i_src = src[mask]
                     u_i_dst = dst[mask] - n_users_limit # Remove offset
                     
                     u_i_edges = torch.stack([u_i_src, u_i_dst], dim=0)
                     i_u_edges = torch.stack([u_i_dst, u_i_src], dim=0)
                     
                     edge_index_dict = {
                         ('user', 'interacts', 'item'): u_i_edges,
                         ('item', 'rev_interacts', 'user'): i_u_edges
                     }

            metrics = evaluate(model, test_dict, train_dict, n_items, edge_index, device=device, adj_norm=adj_norm,
                               re_ranker=re_ranker, rerank_strategy=args.rerank, eval_protocol=args.eval_protocol,
                               cold_users=cold_users, edge_index_dict=edge_index_dict)

            best_metrics = metrics.copy()
            best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_metrics

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train CF/CL Models')
    parser.add_argument('--model', '-m', choices=['simgcl', 'xsimgcl', 'lightgcl', 'ma-hcl', 'ma_hgn', 'bigcf', 'lightgcn'], 
                        default='simgcl', help='Model to train')
    parser.add_argument('--data-path', default='data/processed/strict_g1', help='Path to graph data')
    parser.add_argument('--articles-path', default=None, help='Explicit path to articles.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--re-rank', action='store_true', help='Use Semantic Re-ranking')
    parser.add_argument('--gnn-type', choices=['gat', 'sage', 'gcn', 'transformer'], default='gat', help='GNN type for MA-HGN (gat, sage, gcn, transformer)')
    parser.add_argument('--cold-p', type=float, default=0.2, help='Percentage of cold users for evaluation')
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
    parser.add_argument('--graph-type', choices=['bipartite', 'hetero', 'article', 'category',
                                                  'author', 'temporal', 'reaction', 'crosscat', 'tenure'], 
                        default='bipartite',
                        help='Graph type: bipartite, hetero, article, category, '
                             'author (G4), temporal (G5), reaction (G6), crosscat (G7), tenure (G8)')
    parser.add_argument('--split-strategy', choices=['random', 'time'], default='random',
                        help='Data splitting strategy: "random" (shuffle) or "time" (chronological)')

    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for MA-HCL (default: 0.1)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print(f"Training {args.model.upper()}")
    print(f"Embedding Initialization: {args.embedding.upper()}")
    print(f"Graph Type: {args.graph_type.upper()}")
    print(f"Data Path: {args.data_path}")
    print("=" * 60)
    
    # --- Load Data ---
    data = _load_graph_data(args)
    n_users, n_items = data['n_users'], data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(data['train_pairs'])}, Test users: {len(data['test_dict'])}")
    
    # Identify cold-start users (users with <= 3 training interactions)
    train_dict = data['train_dict']
    cold_threshold = 3
    cold_users = {u for u, items in train_dict.items() if len(items) <= cold_threshold}
    print(f"Cold-start users (≤{cold_threshold} interactions): {len(cold_users)}")
    
    # --- Precompute Adjacency ---
    _precompute_adjacency(args, data, device)

    # --- Load Embeddings ---
    train_item_indices = list(set([i for u, i in data['train_pairs']]))
    pretrained_emb = load_pretrained_embeddings(
        args.embedding, n_items, args.hidden_dim, device, 
        train_item_indices=train_item_indices,
        data_path=args.data_path, articles_path=args.articles_path
    )
    
    # --- Semantic IDs & User Priors ---
    semantic_ids = _load_semantic_ids(args, pretrained_emb, device)
    user_priors = _load_user_priors(args, n_users, device)
    
    # --- Re-ranker ---
    re_ranker = _load_reranker(args)

    # --- Build Model ---
    model = _build_model(args, n_users, n_items, data, device, pretrained_emb, semantic_ids, user_priors)
    
    # Ensure model is on the correct device if not handled internally (e.g., LightGCL)
    if args.model != 'lightgcl':
        model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- Train ---
    if hasattr(model, 'forward') and pretrained_emb is not None:
        model.item_content = pretrained_emb # Attach for evaluate
        
    best_metrics = train_model(model, data, args, device, item_content=pretrained_emb, 
                               semantic_ids=semantic_ids, user_priors=user_priors, 
                               re_ranker=re_ranker, cold_users=cold_users)

    # --- Save & Print Results ---
    _save_and_print_results(args, model, n_users, n_items, best_metrics)


def _load_graph_data(args):
    """Load graph data based on graph type argument."""
    if args.graph_type == 'hetero':
        base_path = Path(args.data_path)
        hetero_path = None
        hetero_candidates = [
            base_path / 'full_hetero_graph.pt',
            base_path / 'all_graphs' / 'full_hetero_graph.pt',
        ]

        for candidate in hetero_candidates:
            if candidate.exists():
                hetero_path = candidate
                break

        if hetero_path is None:
            kaggle_input_root = Path('/kaggle/input')
            if kaggle_input_root.exists():
                matches = list(kaggle_input_root.rglob('full_hetero_graph.pt'))
                preferred = [m for m in matches if m.parent.name == base_path.name]
                if preferred:
                    hetero_path = preferred[0]
                elif matches:
                    hetero_path = matches[0]

        if hetero_path is not None:
            print(f"  Loading Heterogeneous Graph from: {hetero_path}")
            return load_data(str(hetero_path), split_strategy=args.split_strategy)
        else:
            print("  Warning: full_hetero_graph.pt not found. Falling back to default bipartite graph.")
            data = load_data(args.data_path, split_strategy=args.split_strategy)
    elif args.graph_type == 'category':
        cat_path = Path(args.data_path) / 'all_graphs' / 'category_graph.pt'
        if not cat_path.exists():
             cat_path = Path(args.data_path) / 'category_graph.pt'
             
        if cat_path.exists():
            print(f"  Loading Category-Augmented Graph from: {cat_path}")
            return load_data(str(cat_path), split_strategy=args.split_strategy)
        else:
            print(f"  Warning: {cat_path} not found. Falling back.")
            return load_data(args.data_path, split_strategy=args.split_strategy)

    elif args.graph_type == 'article':
        hetero_path = Path(args.data_path) / 'all_graphs' / 'full_hetero_graph.pt'
        if hetero_path.exists():
            print(f"  Loading Full Base Graph from: {hetero_path}")
            data = load_data(args.data_path, split_strategy=args.split_strategy)
            
            # REMOVE Social Edges to ensure purely Article-Augmented experiment
            if isinstance(data, dict):
                 if 'edge_index_dict' in data and ('user', 'replied_to', 'user') in data['edge_index_dict']:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['edge_index_dict'][('user', 'replied_to', 'user')]
            elif hasattr(data, 'edge_index_dict'):
                 if ('user', 'replied_to', 'user') in data.edge_types:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['user', 'replied_to', 'user']
        else:
             print("Full graph not found, falling back (might fail dimensions)")
             data = load_data(args.data_path, split_strategy=args.split_strategy)
        
        # Load auxiliary Article-Article graph
        article_path = Path(args.data_path) / 'all_graphs' / 'article_article_graph_users.pt'
        if article_path.exists():
            print(f"  Loading Article-Article Edges from: {article_path}")
            article_data = torch.load(article_path, weights_only=False)
            data['article_edge_index'] = article_data.edge_index
        else:
            print(f"  Warning: {article_path} not found. Using Bipartite only.")
        return data
    
    else:
        # Load data (default bipartite)
        data = load_data(args.data_path, split_strategy=args.split_strategy)
    n_users, n_items = data['n_users'], data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(data['train_pairs'])}, Test users: {len(data['test_dict'])}")
    
    # Identify cold-start users (users with <= 3 training interactions)
    train_dict = data['train_dict']
    cold_threshold = 3
    cold_users = {u for u, items in train_dict.items() if len(items) <= cold_threshold}
    print(f"Cold-start users (≤{cold_threshold} interactions): {len(cold_users)}")
    


def _precompute_adjacency(args, data, device):
    """Precompute normalized adjacency for graph-based models."""
    graph_models = ['simgcl', 'bigcf', 'igcl', 'xsimgcl', 'lightgcl', 'ma-hcl', 'ma_hgn', 'lightgcn']
    if args.model not in graph_models:
        return
    
    n_users, n_items = data['n_users'], data['n_items']
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
    
    user_user_edges = data.get('user_user_edges') 
    
    if user_user_edges is not None:
         print(f"  Using {user_user_edges.shape[1]} pre-filtered social edges from data dictionary.")
    else:
         # Legacy Fallback — load and FILTER social edges to train-only users
         user_user_edges = None
         social_graph_path = Path(args.data_path) / 'full_hetero_graph.pt'
         if social_graph_path.exists():
            print(f"  Loading social signals from {social_graph_path}...")
            social_data = torch.load(social_graph_path, weights_only=False)
            raw_social = None
            if isinstance(social_data, dict) and 'edge_index_dict' in social_data:
                edges_dict = social_data['edge_index_dict']
                raw_social = edges_dict.get(('user', 'replied_to', 'user'))
            elif hasattr(social_data, 'edge_index_dict'):
                if ('user', 'replied_to', 'user') in social_data.edge_types:
                    raw_social = social_data['user', 'replied_to', 'user'].edge_index
            
            if raw_social is not None:
                # Filter: only keep edges where BOTH users appear in training interactions
                train_users = set(u for u, _ in data['train_pairs'])
                src, dst = raw_social[0], raw_social[1]
                mask = torch.tensor([s.item() in train_users and d.item() in train_users 
                                     for s, d in zip(src, dst)], dtype=torch.bool)
                user_user_edges = raw_social[:, mask]
                print(f"  Filtered social edges: {raw_social.size(1)} -> {user_user_edges.size(1)} (train-only users)")
    
    data['adj_norm'] = compute_normalized_adjacency(n_users, n_items, data['train_pairs'], device, 
                                                    item_item_edges, user_user_edges, 
                                                    edge_weights=edge_weights, social_weight=args.social_weight)
    data['augmented_pairs'] = augmented_pairs


def _load_semantic_ids(args, pretrained_emb, device):
    """Generate Semantic IDs if requested."""
    if args.semantic_id_bits <= 0:
        return None
    if pretrained_emb is not None:
        print(f"\nGenerating Semantic IDs ({args.semantic_id_bits} stages)...")
        semantic_ids, _ = generate_semantic_ids(pretrained_emb, n_codebooks=args.semantic_id_bits, codebook_size=32)
        semantic_ids = semantic_ids.to(device)
        print(f"  Generated IDs shape: {semantic_ids.shape}")
        return semantic_ids
    else:
        print("\n  Warning: Cannot generate Semantic IDs without pretrained embeddings. Skipping.")
        return None


def _load_user_priors(args, n_users, device):
    """Load User Priors if requested."""
    if args.model != 'xsimgcl':
        return None
    prior_path = Path('data/processed/user_priors.pt')
    if prior_path.exists():
        print(f"\nLoading User Priors from {prior_path}...")
        priors = torch.load(prior_path, weights_only=False).to(device)
        if priors.shape[0] < n_users:
            print(f"  Padding User Priors: {priors.shape[0]} -> {n_users}")
            user_priors = torch.zeros((n_users, priors.shape[1]), device=device)
            user_priors[:priors.shape[0]] = priors
        else:
            user_priors = priors
        print(f"  Final Priors shape: {user_priors.shape}")
        return user_priors
    else:
        print(f"\n  Warning: User Priors not found at {prior_path}. Running without priors.")
        return None


def _load_reranker(args):
    """Load Re-ranker if requested."""
    if args.rerank == 'none':
        return None
    print("\nInitializing Re-ranker...")
    with open('data/processed/lightgcl_data.pkl', 'rb') as f:
        idx_data = pickle.load(f)
        idx2item = idx_data['idx2item']
    
    categories, n_cats = load_item_categories(idx2item)
    re_ranker = CalibratedReRanker(categories, alpha=0.5, lambda_mmr=0.5)
    print(f"  Loaded {n_cats} categories for {len(categories)} items.")
    return re_ranker


def _build_model(args, n_users, n_items, data, device, pretrained_emb, semantic_ids, user_priors):
    """Instantiate the chosen model and inject pretrained embeddings."""
    if args.model == 'lightgcl':
        model = LightGCLWrapper(n_users, n_items, embed_dim=args.hidden_dim, n_layers=args.n_layers, 
                                device=args.device, svd_q=args.svd_q, ssl_weight=args.ssl_weight, temp=args.temp)
        model.setup(data['train_pairs'], data.get('augmented_pairs', None))
        if pretrained_emb is not None:
            model.model.E_i_0.data.copy_(pretrained_emb)
            print("  Transferred embeddings to LightGCL (E_i_0)")
            
    elif args.model == 'lightgcn':
        model = LightGCN(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers, 
                         dropout=args.dropout).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to LightGCN")
             
    elif args.model == 'bigcf':
        model = BIGCF(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to BIGCF")

    elif args.model == 'ma-hcl':
        model = MAHCL(
            n_users=n_users, n_items=n_items,
            embedding_dim=args.hidden_dim, n_layers=args.n_layers,
            ssl_weight=args.ssl_weight, eps=args.eps, temp=args.temp,
            n_categories=data.get('n_categories', 0)
        ).to(device)
        if pretrained_emb is not None:
             model.item_emb.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to MA-HCL")
             
    elif args.model == 'ma_hgn':
        num_cats = 0
        if isinstance(data, dict):
            num_cats = data.get('n_categories', 0)
        elif hasattr(data, 'n_categories'):
            num_cats = data.n_categories
        
        model = MAHGN(n_users, n_items, args.hidden_dim, n_layers=args.n_layers, 
                      n_categories=num_cats, gnn_type=args.gnn_type,
                      cl_weight=args.ssl_weight, temp=args.temp).to(device)

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

    else:
        # Default: SimGCL
        if args.model == 'simgcl':
            model = SimGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        
        if pretrained_emb is not None:
            model.item_embedding.weight.data.copy_(pretrained_emb)
            print(f"  Transferred embeddings to {args.model.upper()}")
    
    return model


def _save_and_print_results(args, model, n_users, n_items, best_metrics):
    """Save model checkpoint and print final metrics."""
    p = Path(args.data_path)
    if p.parent.name in ["strict_g1", "strict_g2", "strict_g3", "strict_g4", "strict_g5", 
                         "strict_g6", "strict_g7", "strict_g8", "regular_g2", "enhanced_v1", "enhanced_v2"]:
        graph_name = p.parent.name
    else:
        graph_name = p.stem
        
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_path = f"models/{args.model}_{graph_name}_{timestamp}.pt"
    
    Path("models").mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items,
        'config': vars(args),
        'metrics': best_metrics
    }, save_path)

    print(f"\nModel saved: {save_path}")
    print(f"\nBest Metrics (Final Evaluation):")
    metrics_to_print = ['recall', 'ndcg', 'precision', 'f1', 'hitrate', 'map']
    k_list = [1, 5, 10, 50]
    
    header = f"{'Metric':<12} | " + " | ".join([f"K={k:<8}" for k in k_list])
    print(header)
    print("-" * len(header))
    
    for m in metrics_to_print:
        row = f"{m.upper():<12} | "
        values = []
        for k in k_list:
            val = best_metrics.get(f'{m}@{k}', 0)
            values.append(f"{val:.6f}")
        print(row + " | ".join(values))
    
    print("-" * len(header))
    print(f"MRR: {best_metrics.get('mrr', 0):.6f}")
    if 'entropy' in best_metrics:
        print(f"Entropy: {best_metrics['entropy']:.6f}")
    
    if args.save_results:
        import json
        
        results_path = Path(args.save_results)
        if not results_path.parent.exists() or results_path.parent == Path('.'):
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / results_path.name
        
        def convert(o):
            if isinstance(o, np.float32): return float(o)
            return o
            
        with open(results_path, 'w') as f:
            json.dump(best_metrics, f, default=convert)
        print(f"Saved metrics to {results_path}")


if __name__ == '__main__':
    main()