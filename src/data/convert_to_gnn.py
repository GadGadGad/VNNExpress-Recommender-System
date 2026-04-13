"""
Convert Crawled VnExpress Data to GNN-Ready Format
===================================================
This script provides comprehensive graph construction options for GNN training:

1. User-Article Bipartite Graph (for recommendations)
2. User-User Graph (based on shared article interactions)
3. Article-Article Graph (based on shared commenters)
4. Full Heterogeneous Graph (all node types and relations)

Supports:
- PyTorch Geometric (HeteroData / Data)
- DGL (heterogeneous and homogeneous)
- NetworkX export

Usage:
    python src/convert_to_gnn.py --graph-type hetero --output data/processed
    python src/convert_to_gnn.py --graph-type user-article --add-text-features
"""

import argparse
import json
import os
import re
import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class GNNDataConverter:
    """
    Comprehensive converter for crawled data to GNN-ready format.
    """
    
    def __init__(
        self,
        articles_path: str = 'data/raw/articles.csv',
        replies_path: str = 'data/raw/replies.csv',
        users_path: str = 'data/raw/user_profiles.csv',
        output_dir: str = 'data/processed',
        hidden_dim: int = 64,
        add_text_features: bool = False, use_phobert: bool = False,
        text_max_features: int = 1000,
        min_user_interactions: int = 0,
        min_article_interactions: int = 0,
        split_strategy: str = 'random'
    ):
        self.articles_path = articles_path
        self.replies_path = replies_path
        self.users_path = users_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hidden_dim = hidden_dim
        self.add_text_features = add_text_features
        self.use_phobert = use_phobert
        self.text_max_features = text_max_features
        self.min_user_interactions = min_user_interactions
        self.min_article_interactions = min_article_interactions
        self.split_strategy = split_strategy
        
        self.articles = None
        self.replies = None
        self.users = None
        
        self.user_map = {}
        self.article_map = {}

        
        self._load_data()
        
    def _parse_vn_date(self, date_str):
        """Parse VnExpress date strings to datetime objects."""
        if not isinstance(date_str, str) or not date_str or date_str.lower() == 'nan':
            return None
        
        now = datetime.datetime.now()
        
        # Relative: "Xh trước"
        match_h = re.search(r'(\d+)h trước', date_str)
        if match_h:
            return now - datetime.timedelta(hours=int(match_h.group(1)))
            
        # Relative: "Xp trước"
        match_p = re.search(r'(\d+)p trước', date_str)
        if match_p:
            return now - datetime.timedelta(minutes=int(match_p.group(1)))
            
        # Absolute: "HH:MM DD/MM"
        match_dt = re.search(r'(\d{1,2}):(\d{1,2})\s+(\d{1,2})/(\d{1,2})', date_str)
        if match_dt:
            h, m, d, mon = map(int, match_dt.groups())
            year = now.year
            try:
                dt = datetime.datetime(year, mon, d, h, m)
                if dt > now: dt = dt.replace(year=year-1)
                return dt
            except: return None
        return None

    def _load_data(self):
        """Load and preprocess all data sources with Temporal & Social support."""
        print("=" * 60)
        print("GNN Data Converter - Loading Data (Temporal + Social)")
        print("=" * 60)
        
        print(f"\nLoading articles from {self.articles_path}...")
        self.articles = pd.read_csv(self.articles_path)
        
        # Deduplicate articles: keep first category for multi-category articles
        id_col = 'article_id' if 'article_id' in self.articles.columns else 'id'
        initial_count = len(self.articles)
        self.articles = self.articles.drop_duplicates(subset=[id_col], keep='first')
        if len(self.articles) < initial_count:
            print(f"   Deduplicated articles: {initial_count} -> {len(self.articles)} (Removed {initial_count - len(self.articles)} duplicates)")
        
        print(f"   {len(self.articles):,} articles loaded")
        
        print(f"Loading replies from {self.replies_path}...")
        raw_replies = pd.read_csv(self.replies_path)
        
        def clean_id(val):
            try:
                if pd.isna(val) or val == '' or str(val).lower() == 'nan': return None
                return str(int(float(val)))
            except: return str(val)

        # Process ALL interactions (Parents and Replies)
        interactions = []
        social_edges = []
        
        for _, row in tqdm(raw_replies.iterrows(), total=len(raw_replies), desc="   Parsing Interactions"):
            p_id = clean_id(row['parent_user_id'])
            r_id = clean_id(row['reply_user_id'])
            url = row['article_url']
            
            if p_id:
                p_date = self._parse_vn_date(row['parent_date'])
                interactions.append({
                    'user_id': p_id, 'article_url': url, 
                    'date': p_date, 'reactions': row.get('parent_reactions', 0)
                })
            
            if r_id:
                r_date = self._parse_vn_date(row['reply_date'])
                interactions.append({
                    'user_id': r_id, 'article_url': url, 
                    'date': r_date, 'reactions': row.get('reply_reactions', 0)
                })
                
                if p_id:
                    social_edges.append({'from': r_id, 'to': p_id, 'type': 'reply', 'article_url': url})

        self.replies = pd.DataFrame(interactions).dropna(subset=['user_id', 'article_url'])
        

        initial_len = len(self.replies)
        

        # Parse dates but keep NaT for interactions that don't have them
        self.replies['date'] = pd.to_datetime(self.replies['date'], errors='coerce')
        
        # Sort and deduplicate. Pandas puts NaT at the end.
        self.replies = self.replies.sort_values('date', na_position='last').drop_duplicates(
            subset=['user_id', 'article_url'], keep='first'
        )
        print(f"   Deduplicated interactions: {initial_len:,} -> {len(self.replies):,} (Removed {initial_len - len(self.replies):,})")

        self.social_df = pd.DataFrame(social_edges)
        
        print(f"   {len(self.replies):,} initial interactions captured")
        
        # Time Decay REMOVED as per user request
        self.replies['time_decay'] = 1.0

        # Iterative K-Core filtering
        prev_len = 0
        iteration = 0
        while len(self.replies) != prev_len:
            prev_len = len(self.replies)
            iteration += 1
            user_counts = self.replies['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            article_counts = self.replies['article_url'].value_counts()
            valid_articles = article_counts[article_counts >= self.min_article_interactions].index
            
            self.replies = self.replies[
                (self.replies['user_id'].isin(valid_users)) &
                (self.replies['article_url'].isin(valid_articles))
            ].copy()
            
        print(f"   After {iteration} k-core iterations: {len(self.replies):,} interactions")
        
        valid_article_urls = self.replies['article_url'].unique()
        self.articles = self.articles[self.articles['url'].isin(valid_article_urls)].copy()
        
        density = len(self.replies) / (self.replies['user_id'].nunique() * len(self.articles)) * 100
        print(f"   Cleaned density: {density:.4f}%")
        
        self._create_mappings()
        
    def _create_mappings(self):
        """Create ID to index mappings."""
        unique_users = self.replies['user_id'].unique()
        unique_articles = self.articles['url'].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.article_map = {a: i for i, a in enumerate(unique_articles)}
        
        self.replies['user_idx'] = self.replies['user_id'].map(self.user_map)
        self.replies['article_idx'] = self.replies['article_url'].map(self.article_map)
        
        
        print(f"\nGraph Statistics:")
        print(f"   Users: {len(self.user_map):,}")
        print(f"   Articles: {len(self.article_map):,}")
        print(f"   Edges (interactions): {len(self.replies):,}")
        
    def _save_mappings(self):
        """Save ID mappings for later use."""
        with open(self.output_dir / 'user_map.json', 'w') as f:
            json.dump(self.user_map, f)
        with open(self.output_dir / 'article_map.json', 'w') as f:
            json.dump(self.article_map, f)
        print(f"   [OK] Saved mappings to {self.output_dir}")
            
    def _create_user_features(self) -> torch.Tensor:
        """Create user node features."""
        num_users = len(self.user_map)
        
        features = torch.randn(num_users, self.hidden_dim)
        
        
        return features
    
    def _create_article_features(self) -> torch.Tensor:
        """Create article node features."""
        if self.use_phobert:
            print("   Using PhoBERT embeddings (Projected to hidden_dim)...")
            emb_path = self.output_dir / 'phobert_embeddings.pt'
            if not emb_path.exists():
                raise FileNotFoundError(f"Run generate_embeddings.py first!")
            emb_dict = torch.load(emb_path)
            
            features = []
            zeros = 0
            for url in self.articles['url']:
                if url in emb_dict:
                    features.append(emb_dict[url])
                else:
                    features.append(torch.randn(768)*0.1)
                    zeros += 1
            if zeros > 0: print(f"      [WARN] {zeros} missing embeddings")
            
            full_emb = torch.stack(features)
            torch.manual_seed(42)
            proj = torch.nn.Linear(768, self.hidden_dim)
            return proj(full_emb).detach()

        num_articles = len(self.article_map)
        
        if self.add_text_features:
            print("   Adding text features (TF-IDF)...")
            titles = self.articles['title'].fillna('').tolist()
            tfidf = TfidfVectorizer(max_features=self.text_max_features)
            text_features = tfidf.fit_transform(titles).toarray()
            text_tensor = torch.tensor(text_features, dtype=torch.float32)
            
            projector = torch.nn.Linear(text_tensor.shape[1], self.hidden_dim)
            features = projector(text_tensor).detach()
        else:
            # Use random initialization (no category available)
            features = torch.randn(num_articles, self.hidden_dim)
        
        return features

    
    def _get_split_indices(self, seed=42):
        """Get indices split by Time (Chronological) OR Randomly."""
        import numpy as np
        
        filename = f'split_indices_{self.split_strategy}.pt' 
        split_file = self.output_dir / filename
        num_positives = len(self.replies)
        
        if split_file.exists():
            indices = torch.load(split_file, weights_only=False)
            if len(indices) == num_positives:
                return indices.numpy() if torch.is_tensor(indices) else indices
        

        if self.split_strategy == 'time': # Or default if you want to enforce it
            print("   Splitting Chronologically (Past -> Future)...")
            # Since replies are sorted by date, use sequential indices
            indices = np.arange(num_positives) 
        else:
            # Random split (Legacy code)
            print("   Splitting Randomly (Shuffled)...")
            np.random.seed(seed)
            indices = np.random.permutation(num_positives)


        torch.save(indices, split_file)
        return indices

    def _get_train_mask(self, seed=42, train_ratio=0.8):
        """Get mask for training interactions."""
        import numpy as np
        indices = self._get_split_indices(seed)
        num_positives = len(indices)
        
        train_end = int(num_positives * train_ratio)
        train_idx = indices[:train_end]
        
        mask = np.zeros(num_positives, dtype=bool)
        mask[train_idx] = True
        return mask

    def build_user_article_graph(self) -> 'HeteroData':
        """
        Build User-Article bipartite graph (PyTorch Geometric HeteroData).
        
        Nodes:
            - user: Users who comment
            - article: News articles
        
        Edges:
            - (user, comments, article): User commented on article
            - (article, rev_comments, user): Reverse edge for message passing
        """
        print("\nBuilding User-Article Bipartite Graph...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        data = HeteroData()
        
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        

        print("   Splitting Train/Test to prevent leakage...")
        
        # 1. Get mask for Train set (80%)
        # The _get_train_mask function is available in your class
        train_mask = self._get_train_mask(train_ratio=0.8) 
        
        # 2. Only use Train data to build edges for the graph (Message Passing)
        train_replies = self.replies[train_mask]
        
        src_train = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst_train = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        
        # Edge Index: Only contains Train edges. Model does not see future interactions.
        data['user', 'comments', 'article'].edge_index = torch.stack([src_train, dst_train])
        
        # Label Index: Used for Loss calculation in Trainer
        data['user', 'comments', 'article'].edge_label_index = torch.stack([src_train, dst_train])
        
        # 3. Calculate weights (Reactions + Time Decay) only on Train set
        reactions = pd.to_numeric(train_replies['reactions'], errors='coerce').fillna(0).values
        reactions = np.clip(reactions, 0, None)
        
        edge_weights = (1.0 + np.log1p(reactions)) * train_replies['time_decay'].values
        data['user', 'comments', 'article'].edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        # 4. (Important) Save Test edges separately for later Evaluation
        test_replies = self.replies[~train_mask]
        src_test = torch.tensor(test_replies['user_idx'].values, dtype=torch.long)
        dst_test = torch.tensor(test_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].test_edge_index = torch.stack([src_test, dst_test])
        
        print(f"     Train edges: {len(train_replies)} | Test edges: {len(test_replies)}")

        
        data = T.ToUndirected()(data)
        
        # Consistent metadata for train_cf_models
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'user_article_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Edges: {data['user', 'comments', 'article'].edge_index.shape}")
        print(f"   [OK] Saved to: {save_path}")
        
        return result

    def build_full_hetero_graph(self) -> 'HeteroData':
        """
        Build full heterogeneous graph with all node types and relations.
        """
        print("\nBuilding Full Heterogeneous Graph...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        data = HeteroData()
        
        # ===== NODE FEATURES =====
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # ===== EDGES =====
        print("   Filtering interactions for User-Article edges...")
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]

        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # User - User (Social Reply Network)
        if hasattr(self, 'social_df') and not self.social_df.empty:
            self.social_df['from_idx'] = self.social_df['from'].map(self.user_map)
            self.social_df['to_idx'] = self.social_df['to'].map(self.user_map)
            valid_social = self.social_df.dropna(subset=['from_idx', 'to_idx'])
            
            print("   Filtering social reply edges...")
            train_articles = set(train_replies['article_url'].unique())
            valid_social = valid_social[valid_social['article_url'].isin(train_articles)]
            
            if not valid_social.empty:
                data['user', 'replied_to', 'user'].edge_index = torch.tensor(
                    valid_social[['from_idx', 'to_idx']].values.T, dtype=torch.long
                )
        
        # User - User (Shared Interest)
        if not (hasattr(self, 'no_aux_edges') and self.no_aux_edges):
            print("   Filtering interactions for Shared Interest edges...")
            user_articles = train_replies.groupby('user_idx')['article_idx'].apply(set).to_dict()
            shared_edges = []
            user_ids = list(user_articles.keys())
            
            print(f"   Computing latent user-user edges (shared >= 2)...")
            for i in range(len(user_ids)):
                u1 = user_ids[i]
                for j in range(i + 1, min(i + 500, len(user_ids))):
                    u2 = user_ids[j]
                    if len(user_articles[u1] & user_articles[u2]) >= 2:
                        shared_edges.append([u1, u2])
                        shared_edges.append([u2, u1])
            
            if shared_edges:
                data['user', 'interacts_with', 'user'].edge_index = torch.tensor(
                    shared_edges, dtype=torch.long
                ).T
        
        data = T.ToUndirected()(data)
        
        # Consistent metadata for train_cf_models
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        filename = 'full_hetero_graph_no_aux.pt' if (hasattr(self, 'no_aux_edges') and self.no_aux_edges) else 'full_hetero_graph.pt'
        save_path = self.output_dir / filename
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Saved to: {save_path}")
        
        return result
    
    def build_user_category_graph(self) -> Dict:
        """
        Build User-Category bipartite graph (G3).
        Users connected to categories based on their article interactions.
        Uses ONLY training interactions to prevent leakage.
        """
        print("\nBuilding User-Category Bipartite Graph (G3)...")
        
        # Merge interactions with article categories
        self.articles['article_url'] = self.articles['url']
        
        # Only use training interactions for category weights
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        merged = train_replies.merge(
            self.articles[['article_url', 'source_category']], 
            on='article_url', 
            how='inner'
        )
        
        # Create mappings
        categories = merged['source_category'].unique()
        cat_map = {c: i for i, c in enumerate(categories)}
        
        # Build edge weights (user - category interaction count)
        user_cat_counts = merged.groupby(['user_idx', 'source_category']).size().reset_index(name='count')
        
        src = torch.tensor(user_cat_counts['user_idx'].values, dtype=torch.long)
        dst = torch.tensor([cat_map[c] for c in user_cat_counts['source_category']], dtype=torch.long)
        weights = torch.tensor(user_cat_counts['count'].values, dtype=torch.float32)
        
        # Prepare splits for train_cf_models consistency
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        data = {
            'user_category_edge_index': torch.stack([src, dst]),
            'user_category_edge_weight': weights,
            'cat_map': cat_map,
            'n_users': len(self.user_map),
            'n_categories': len(cat_map),
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'category_graph.pt'
        torch.save(data, save_path)
        print(f"   [OK] Saved to: {save_path}")
        return data

    def build_author_mediated_graph(self) -> Dict:
        """
        Build Author-Mediated Heterogeneous Graph (G4).
        Adds 'author' as a node type: article --written_by--> author.
        Users who like articles by the same author get connected through the author node.
        Uses ONLY training interactions to prevent leakage.
        """
        print("\nBuilding Author-Mediated Graph (G4)...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric")
        
        data = HeteroData()
        
        # Node features
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # Author mapping
        valid_articles = self.articles.dropna(subset=['author'])
        unique_authors = sorted(valid_articles['author'].unique())
        author_map = {a: i for i, a in enumerate(unique_authors)}
        n_authors = len(author_map)
        print(f"   Authors: {n_authors} (from {len(valid_articles)} articles with known authors)")
        
        # Author node features (random init, same dim as others)
        data['author'].x = torch.randn(n_authors, self.hidden_dim)
        
        # Article --written_by--> Author edges
        art_src, auth_dst = [], []
        for _, row in valid_articles.iterrows():
            url = row['url']
            if url in self.article_map and row['author'] in author_map:
                art_src.append(self.article_map[url])
                auth_dst.append(author_map[row['author']])
        
        if art_src:
            data['article', 'written_by', 'author'].edge_index = torch.tensor(
                [art_src, auth_dst], dtype=torch.long
            )
            print(f"   Article-Author edges: {len(art_src)}")
        
        # User-Article edges (train only)
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        data = T.ToUndirected()(data)
        
        # Splits
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'n_authors': n_authors,
            'author_map': author_map,
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'author_mediated_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] Nodes: user={data['user'].x.shape[0]}, article={data['article'].x.shape[0]}, author={n_authors}")
        print(f"   [OK] Saved to: {save_path}")
        return result

    def build_temporal_cooccurrence_graph(self, day_window=0) -> Dict:
        """
        Build Temporal Co-Occurrence Graph (G5).
        Connects articles published within the same day (or ±day_window).
        Uses ONLY training interactions for user-article edges.
        """
        print(f"\nBuilding Temporal Co-Occurrence Graph (G5, window=±{day_window} days)...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric")
        
        data = HeteroData()
        
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # Parse article publish dates
        dates = self.articles['published_at'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0]
        self.articles['pub_date'] = pd.to_datetime(dates, format='%d/%m/%Y', errors='coerce')
        
        valid_dated = self.articles.dropna(subset=['pub_date'])
        print(f"   Articles with valid dates: {len(valid_dated)}/{len(self.articles)}")
        
        # Group by date and create co-occurrence edges
        art_src, art_dst = [], []
        date_groups = valid_dated.groupby(valid_dated['pub_date'].dt.date)
        
        if day_window == 0:
            # Same-day co-occurrence
            for date, group in date_groups:
                urls = group['url'].tolist()
                idxs = [self.article_map[u] for u in urls if u in self.article_map]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        # Only connect within same category to reduce density
                        cat_i = group[group['url'].map(self.article_map) == idxs[i]]['source_category'].iloc[0] if len(group[group['url'].map(lambda x: self.article_map.get(x)) == idxs[i]]) > 0 else None
                        art_src.append(idxs[i])
                        art_dst.append(idxs[j])
        else:
            # Window-based: ±day_window days
            sorted_articles = valid_dated.sort_values('pub_date')
            article_dates = sorted_articles[['url', 'pub_date']].values
            window = pd.Timedelta(days=day_window)
            
            for i in range(len(article_dates)):
                url_i, date_i = article_dates[i]
                if url_i not in self.article_map:
                    continue
                idx_i = self.article_map[url_i]
                for j in range(i + 1, len(article_dates)):
                    url_j, date_j = article_dates[j]
                    if date_j - date_i > window:
                        break
                    if url_j not in self.article_map:
                        continue
                    art_src.append(idx_i)
                    art_dst.append(self.article_map[url_j])
        
        if art_src:
            data['article', 'co_published', 'article'].edge_index = torch.tensor(
                [art_src, art_dst], dtype=torch.long
            )
            print(f"   Temporal co-occurrence edges: {len(art_src)}")
        
        # User-Article edges (train only)
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        data = T.ToUndirected()(data)
        
        # Splits
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'temporal_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] Saved to: {save_path}")
        return result

    def build_reaction_weighted_graph(self) -> Dict:
        """
        Build Reaction-Weighted Graph (G6).
        Same topology as G1, but edge weights = 1 + log1p(reactions).
        Uses ONLY training interactions.
        """
        print("\nBuilding Reaction-Weighted Graph (G6)...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric")
        
        data = HeteroData()
        
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # Train split
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # Compute reaction-based weights
        reactions = pd.to_numeric(train_replies['reactions'], errors='coerce').fillna(0).values
        reactions = np.clip(reactions, 0, None)
        edge_weights = (1.0 + np.log1p(reactions)).astype(np.float32)
        
        data['user', 'comments', 'article'].edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        print(f"   Edge weights: mean={edge_weights.mean():.2f}, max={edge_weights.max():.2f}, "
              f"non-unit={np.sum(edge_weights > 1.01)}/{len(edge_weights)} ({np.sum(edge_weights > 1.01)/len(edge_weights)*100:.1f}%)")
        
        data = T.ToUndirected()(data)
        
        # Splits
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'edge_weight': torch.tensor(edge_weights, dtype=torch.float32),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'reaction_weighted_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] Train edges: {len(train_replies)}")
        print(f"   [OK] Saved to: {save_path}")
        return result

    def build_cross_category_graph(self, min_shared_cats=2) -> Dict:
        """
        Build Cross-Category Explorer Graph (G7).
        Extends G3 with user-user edges between users who share ≥min_shared_cats categories.
        Uses ONLY training interactions.
        """
        print(f"\nBuilding Cross-Category Explorer Graph (G7, min_shared={min_shared_cats})...")
        
        # Merge with categories
        self.articles['article_url'] = self.articles['url']
        
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        merged = train_replies.merge(
            self.articles[['article_url', 'source_category']], 
            on='article_url', how='inner'
        )
        
        # Per-user category sets
        user_cats = merged.groupby('user_idx')['source_category'].apply(set).to_dict()
        
        # Category nodes and user-category edges (same as G3)
        categories = sorted(merged['source_category'].unique())
        cat_map = {c: i for i, c in enumerate(categories)}
        n_categories = len(cat_map)
        
        user_cat_counts = merged.groupby(['user_idx', 'source_category']).size().reset_index(name='count')
        uc_src = torch.tensor(user_cat_counts['user_idx'].values, dtype=torch.long)
        uc_dst = torch.tensor([cat_map[c] for c in user_cat_counts['source_category']], dtype=torch.long)
        uc_weights = torch.tensor(user_cat_counts['count'].values, dtype=torch.float32)
        
        # User-user edges: shared category overlap
        multi_cat_users = {u: cats for u, cats in user_cats.items() if len(cats) >= min_shared_cats}
        print(f"   Multi-category users (≥{min_shared_cats} cats): {len(multi_cat_users)}")
        
        uu_src, uu_dst = [], []
        user_list = list(multi_cat_users.keys())
        for i in range(len(user_list)):
            u1 = user_list[i]
            cats1 = multi_cat_users[u1]
            for j in range(i + 1, len(user_list)):
                u2 = user_list[j]
                cats2 = multi_cat_users[u2]
                shared = len(cats1 & cats2)
                if shared >= min_shared_cats:
                    uu_src.extend([u1, u2])
                    uu_dst.extend([u2, u1])
        
        print(f"   User-user shared-category edges: {len(uu_src)}")
        
        # Splits
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'user_category_edge_index': torch.stack([uc_src, uc_dst]),
            'user_category_edge_weight': uc_weights,
            'cat_map': cat_map,
            'n_users': len(self.user_map),
            'n_categories': n_categories,
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        if uu_src:
            result['user_user_edges'] = torch.tensor([uu_src, uu_dst], dtype=torch.long)
        
        save_path = self.output_dir / 'cross_category_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] Categories: {n_categories}")
        print(f"   [OK] Saved to: {save_path}")
        return result

    def build_user_tenure_graph(self) -> Dict:
        """
        Build User-Tenure-Aware Graph (G8).
        Same topology as G1, but replaces random user features with 
        engineered features from user_profiles.csv (account age, tenure bucket).
        Uses ONLY training interactions.
        """
        print("\nBuilding User-Tenure-Aware Graph (G8)...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric")
        
        data = HeteroData()
        
        # Load user profiles
        profiles = None
        if hasattr(self, 'users_path') and os.path.exists(self.users_path):
            profiles = pd.read_csv(self.users_path)
        elif os.path.exists('data/raw/user_profiles.csv'):
            profiles = pd.read_csv('data/raw/user_profiles.csv')
        
        n_users = len(self.user_map)
        
        if profiles is not None:
            print(f"   Loaded {len(profiles)} user profiles")
            
            # Parse join dates
            profiles['join_month'] = pd.to_datetime(profiles['join_date'], format='%m/%Y', errors='coerce')
            now = pd.Timestamp.now()
            profiles['age_months'] = (now - profiles['join_month']).dt.days / 30.0
            
            # Build feature vectors for each user
            # Features: [age_months_norm, is_veteran, is_newcomer, log_age, category_entropy]
            feature_dim = 5
            user_features = torch.zeros(n_users, feature_dim)
            
            # Create a lookup by user_id
            profile_lookup = {}
            for _, row in profiles.iterrows():
                uid = str(int(float(row['user_id']))) if pd.notna(row['user_id']) else None
                if uid and pd.notna(row.get('age_months')):
                    profile_lookup[uid] = row['age_months']
            
            matched = 0
            for uid_str, user_idx in self.user_map.items():
                if uid_str in profile_lookup:
                    age = profile_lookup[uid_str]
                    user_features[user_idx, 0] = age / 120.0  # Normalized age (cap ~10 years)
                    user_features[user_idx, 1] = 1.0 if age > 60 else 0.0  # Veteran flag
                    user_features[user_idx, 2] = 1.0 if age < 6 else 0.0   # Newcomer flag
                    user_features[user_idx, 3] = np.log1p(age) / 5.0       # Log-age normalized
                    matched += 1
            
            # Add per-user interaction count as 5th feature (from train only)
            train_mask = self._get_train_mask()
            train_replies = self.replies[train_mask]
            user_counts = train_replies.groupby('user_idx').size()
            for uidx, count in user_counts.items():
                if uidx < n_users:
                    user_features[uidx, 4] = np.log1p(count) / 5.0  # Log-count normalized
            
            print(f"   Matched profiles: {matched}/{n_users} ({matched/n_users*100:.1f}%)")
            
            # Project to hidden_dim via a linear layer
            torch.manual_seed(42)
            proj = torch.nn.Linear(feature_dim, self.hidden_dim)
            data['user'].x = proj(user_features).detach()
        else:
            print("   [WARNING] No user profiles found. Using random features.")
            data['user'].x = self._create_user_features()
        
        data['article'].x = self._create_article_features()
        
        # User-Article edges (train only)
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]
        
        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        data = T.ToUndirected()(data)
        
        # Splits
        indices = self._get_split_indices()
        num_pos = len(self.replies)
        train_end = int(num_pos * 0.8)
        
        result = {
            'graph': data,
            'n_users': len(self.user_map),
            'n_items': len(self.article_map),
            'splits': {
                'train': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[:train_end]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[:train_end]]['article_idx'].values, dtype=torch.long),
                },
                'test': {
                    'pos_users': torch.tensor(self.replies.iloc[indices[train_end:]]['user_idx'].values, dtype=torch.long),
                    'pos_articles': torch.tensor(self.replies.iloc[indices[train_end:]]['article_idx'].values, dtype=torch.long),
                }
            }
        }
        
        save_path = self.output_dir / 'user_tenure_graph.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"   [OK] User features: {data['user'].x.shape}")
        print(f"   [OK] Saved to: {save_path}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert crawled VnExpress data to GNN-Ready Benchmark Formats (G1-G8)'
    )
    parser.add_argument(
        '--graph-type', '-g',
        choices=['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'all'],
        default='g2',
        help='Benchmark: g1 (Bipartite), g2 (Full Hetero), g3 (User-Category), '
             'g4 (Author-Mediated), g5 (Temporal Co-Occurrence), g6 (Reaction-Weighted), '
             'g7 (Cross-Category Explorer), g8 (User-Tenure-Aware)'
    )
    parser.add_argument('--articles', '-a', default='data/raw/articles.csv')
    parser.add_argument('--replies', '-r', default='data/raw/replies.csv')
    parser.add_argument('--output', '-o', default='data/processed')
    parser.add_argument('--hidden-dim', '-d', type=int, default=64)
    parser.add_argument('--use-phobert', action='store_true')
    parser.add_argument('--min-user-interactions', type=int, default=0)
    parser.add_argument('--min-article-interactions', type=int, default=0)
    
    args = parser.parse_args()
    
    # Base directory
    base_output = Path(args.output)
    
    def get_converter(folder_name):
        out_dir = base_output / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return GNNDataConverter(
            articles_path=args.articles,
            replies_path=args.replies,
            output_dir=out_dir,
            hidden_dim=args.hidden_dim,
            use_phobert=args.use_phobert,
            min_user_interactions=args.min_user_interactions,
            min_article_interactions=args.min_article_interactions
        )
    
    if args.graph_type in ['g1', 'all']:
        converter = get_converter('strict_g1')
        converter.build_user_article_graph()
    if args.graph_type in ['g2', 'all']:
        converter = get_converter('strict_g2')
        converter.build_full_hetero_graph()
    if args.graph_type in ['g3', 'all']:
        converter = get_converter('strict_g3')
        converter.build_user_category_graph()
    if args.graph_type in ['g4', 'all']:
        converter = get_converter('strict_g4')
        converter.build_author_mediated_graph()
    if args.graph_type in ['g5', 'all']:
        converter = get_converter('strict_g5')
        converter.build_temporal_cooccurrence_graph()
    if args.graph_type in ['g6', 'all']:
        converter = get_converter('strict_g6')
        converter.build_reaction_weighted_graph()
    if args.graph_type in ['g7', 'all']:
        converter = get_converter('strict_g7')
        converter.build_cross_category_graph()
    if args.graph_type in ['g8', 'all']:
        converter = get_converter('strict_g8')
        converter.build_user_tenure_graph()

    print("\n" + "=" * 60)
    print("GNN Benchmark Data Conversion Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

