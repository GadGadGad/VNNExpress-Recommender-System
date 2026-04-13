#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for Collaborative Filtering models.
Handles graph loading, split management, adjacency normalization, and batch sampling.
"""
import os
import sys
import numpy as np
import torch
import torch.sparse as sp
import pandas as pd
from pathlib import Path

import scipy.sparse as sp


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


def load_data(data_path, min_interactions=2, split_strategy='random'):
    """Load and process data for CF models."""
    import pandas as pd
    from torch_geometric.data import HeteroData
    
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
                    target_graph = data if isinstance(data, HeteroData) else data['graph']
                
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
    
    # Save cache with strategy name
    # Nếu là file path, ta lưu vào parent directory với tên mới
    if p.is_file():
        save_path = p.parent / cache_filename
    else:
        save_path = p / cache_filename
        
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
