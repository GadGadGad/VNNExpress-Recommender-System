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
            print(f"   → Deduplicated articles: {initial_count} -> {len(self.articles)} (Removed {initial_count - len(self.articles)} duplicates)")
        
        print(f"   → {len(self.articles):,} articles loaded")
        
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
            
            # 1. Parent Interaction
            if p_id:
                p_date = self._parse_vn_date(row['parent_date'])
                interactions.append({
                    'user_id': p_id, 'article_url': url, 
                    'date': p_date, 'reactions': row.get('parent_reactions', 0)
                })
            
            # 2. Reply Interaction
            if r_id:
                r_date = self._parse_vn_date(row['reply_date'])
                interactions.append({
                    'user_id': r_id, 'article_url': url, 
                    'date': r_date, 'reactions': row.get('reply_reactions', 0)
                })
                
                # 3. Social Signal (User-User Reply)
                if p_id:
                    social_edges.append({'from': r_id, 'to': p_id, 'type': 'reply', 'article_url': url})

        self.replies = pd.DataFrame(interactions).dropna(subset=['user_id', 'article_url'])
        
        # LEAKAGE FIX: Deduplicate interactions!
        initial_len = len(self.replies)
        
        # --- DATE HANDLING ---
        # Parse dates but keep NaT for interactions that don't have them
        self.replies['date'] = pd.to_datetime(self.replies['date'], errors='coerce')
        
        # Sắp xếp và deduplicate. Với NaT, Pandas sẽ đẩy xuống cuối.
        self.replies = self.replies.sort_values('date', na_position='last').drop_duplicates(
            subset=['user_id', 'article_url'], keep='first'
        )
        print(f"   → Deduplicated interactions: {initial_len:,} -> {len(self.replies):,} (Removed {initial_len - len(self.replies):,})")

        self.social_df = pd.DataFrame(social_edges)
        
        print(f"   → {len(self.replies):,} initial interactions captured")
        
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
            print("   -> Using PhoBERT embeddings (Projected to hidden_dim)...")
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
            print("   → Adding text features (TF-IDF)...")
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
        """Get indices split by Time or Randomly."""
        import numpy as np
        
        # Consistent cache filename
        filename = f'split_indices_{self.split_strategy}.pt' 
        split_file = self.output_dir / filename
        num_positives = len(self.replies)
        
        if split_file.exists():
            indices = torch.load(split_file, weights_only=False)
            if len(indices) == num_positives:
                return indices.numpy() if torch.is_tensor(indices) else indices
        
        # Default: Random split
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
        
        # --- FIX LEAKAGE START ---
        print("   → [FIX] Splitting Train/Test to prevent leakage...")
        
        # 1. Lấy mask cho tập Train (80%)
        # Hàm _get_train_mask đã có sẵn trong class của bạn
        train_mask = self._get_train_mask(train_ratio=0.8) 
        
        # 2. Chỉ dùng dữ liệu Train để xây dựng cạnh cho đồ thị (Message Passing)
        train_replies = self.replies[train_mask]
        
        src_train = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst_train = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        
        # Edge Index: Chỉ chứa cạnh Train -> Model không nhìn thấy tương lai
        data['user', 'comments', 'article'].edge_index = torch.stack([src_train, dst_train])
        
        # Label Index: Dùng để tính Loss trong Trainer
        data['user', 'comments', 'article'].edge_label_index = torch.stack([src_train, dst_train])
        
        # 3. Tính trọng số (Reactions + Time Decay) chỉ trên tập Train
        reactions = pd.to_numeric(train_replies['reactions'], errors='coerce').fillna(0).values
        reactions = np.clip(reactions, 0, None)
        
        edge_weights = (1.0 + np.log1p(reactions)) * train_replies['time_decay'].values
        data['user', 'comments', 'article'].edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        # 4. (Quan trọng) Lưu cạnh Test riêng để dùng khi Evaluate sau này
        test_replies = self.replies[~train_mask]
        src_test = torch.tensor(test_replies['user_idx'].values, dtype=torch.long)
        dst_test = torch.tensor(test_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].test_edge_index = torch.stack([src_test, dst_test])
        
        print(f"     Train edges: {len(train_replies)} | Test edges: {len(test_replies)}")
        # --- FIX LEAKAGE END ---
        
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
        print("   → Filtering interactions for User-Article edges...")
        train_mask = self._get_train_mask()
        train_replies = self.replies[train_mask]

        src = torch.tensor(train_replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(train_replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # User -> User (Social Reply Network)
        if hasattr(self, 'social_df') and not self.social_df.empty:
            self.social_df['from_idx'] = self.social_df['from'].map(self.user_map)
            self.social_df['to_idx'] = self.social_df['to'].map(self.user_map)
            valid_social = self.social_df.dropna(subset=['from_idx', 'to_idx'])
            
            print("   → Filtering social reply edges...")
            train_articles = set(train_replies['article_url'].unique())
            valid_social = valid_social[valid_social['article_url'].isin(train_articles)]
            
            if not valid_social.empty:
                data['user', 'replied_to', 'user'].edge_index = torch.tensor(
                    valid_social[['from_idx', 'to_idx']].values.T, dtype=torch.long
                )
        
        # User -> User (Shared Interest)
        if not (hasattr(self, 'no_aux_edges') and self.no_aux_edges):
            print("   → Filtering interactions for Shared Interest edges...")
            user_articles = train_replies.groupby('user_idx')['article_idx'].apply(set).to_dict()
            shared_edges = []
            user_ids = list(user_articles.keys())
            
            print(f"   → Computing latent user-user edges (shared >= 2)...")
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
        
        # LEAKAGE FIX: Only use training interactions for category weights
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
        
        # Build edge weights (user -> category interaction count)
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

def main():
    parser = argparse.ArgumentParser(
        description='Convert crawled VnExpress data to GNN-Ready Benchmark Formats (G1, G2, G3)'
    )
    parser.add_argument(
        '--graph-type', '-g',
        choices=['g1', 'g2', 'g3', 'all'],
        default='g2',
        help='Benchmark type: g1 (Bipartite), g2 (Full Hetero), g3 (User-Category)'
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

    print("\n" + "=" * 60)
    print("GNN Benchmark Data Conversion Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
