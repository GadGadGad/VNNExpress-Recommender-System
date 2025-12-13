"""
Convert Crawled VnExpress Data to GNN-Ready Format (With Social Edges)
======================================================================
"""

import argparse
import json
import os
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
        output_dir: str = 'data/processed_phobert',
        hidden_dim: int = 64,
        add_text_features: bool = False, use_phobert: bool = False,
        text_max_features: int = 1000,
        no_social: bool = False
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
        self.no_social = no_social
        
        # Data containers
        self.articles = None
        self.replies = None
        self.users = None
        
        # Mappings
        self.user_map = {}
        self.article_map = {}
        self.category_encoder = None
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess all data sources."""
        print("=" * 60)
        print("GNN Data Converter - Loading Data")
        print("=" * 60)
        
        self.articles = pd.read_csv(self.articles_path)
        print(f"   → {len(self.articles):,} articles loaded")
        
        self.replies = pd.read_csv(self.replies_path)
        
        # Remove NO_COMMENT markers
        self.replies = self.replies[self.replies['parent_user_id'] != 'NO_COMMENT'].copy()
        
        # Clean user IDs
        def clean_id(val):
            try:
                if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                    return None
                return str(int(float(val)))
            except:
                return str(val)
        
        # Extract distinct IDs
        self.replies['parent_id_clean'] = self.replies['parent_user_id'].apply(clean_id)
        self.replies['reply_id_clean'] = self.replies['reply_user_id'].apply(clean_id)
        
        # Filter to valid articles only
        valid_urls = set(self.articles['url'].unique())
        self.replies = self.replies[self.replies['article_url'].isin(valid_urls)]
        print(f"   → {len(self.replies):,} valid interactions")
        
        self._create_mappings()
        
    def _create_mappings(self):
        """Create ID to index mappings."""
        parents = self.replies['parent_id_clean'].dropna().unique()
        repliers = self.replies['reply_id_clean'].dropna().unique()
        unique_users = np.unique(np.concatenate([parents, repliers]))
        
        unique_articles = self.articles['url'].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.article_map = {a: i for i, a in enumerate(unique_articles)}
        
        # Map indices
        self.replies['parent_idx'] = self.replies['parent_id_clean'].map(self.user_map)
        self.replies['reply_idx'] = self.replies['reply_id_clean'].map(self.user_map)
        
        # Determine Primary Actor (User_Idx) for User-Article edges
        self.replies['user_idx'] = np.where(
            self.replies['reply_idx'].notna(),
            self.replies['reply_idx'],
            self.replies['parent_idx']
        )
        
        # Drop rows where mapping failed
        self.replies = self.replies.dropna(subset=['user_idx'])
        self.replies['user_idx'] = self.replies['user_idx'].astype(int)
        
        self.replies['article_idx'] = self.replies['article_url'].map(self.article_map)
        
        # Encode categories
        self.category_encoder = LabelEncoder()
        self.articles['category_idx'] = self.category_encoder.fit_transform(self.articles['category'])
        
        print(f"\nGraph Statistics:")
        print(f"   Users: {len(self.user_map):,}")
        print(f"   Articles: {len(self.article_map):,}")
        print(f"   Categories: {len(self.category_encoder.classes_)}")
        print(f"   Edges (interactions): {len(self.replies):,}")
        
    def _save_mappings(self):
        """Save ID mappings for later use."""
        with open(self.output_dir / 'user_map.json', 'w') as f:
            json.dump(self.user_map, f)
        with open(self.output_dir / 'article_map.json', 'w') as f:
            json.dump(self.article_map, f)
        with open(self.output_dir / 'category_classes.json', 'w') as f:
            json.dump(list(self.category_encoder.classes_), f)
        print(f"   [OK] Saved mappings to {self.output_dir}")
            
    def _create_user_features(self) -> torch.Tensor:
        """Create user node features."""
        num_users = len(self.user_map)
        features = torch.randn(num_users, self.hidden_dim)
        return features

    def _create_article_features(self) -> torch.Tensor:
        """Create article node features (Text/PhoBERT/Random)."""
        num_articles = len(self.article_map)
        
        if self.use_phobert:
            print("   → Using Pre-computed PhoBERT embeddings...")
            emb_path = self.output_dir / 'phobert_embeddings.pt'
            if emb_path.exists():
                emb_dict = torch.load(emb_path, map_location='cpu')
                
                # Align embeddings with article_map (index order)
                features_list = []
                missing_count = 0
                
                # article_map values are 0..N-1, keys are URLs.
                # Invert map for easy lookup by index, or just iterate sorted items.
                # Sorting article_map items by value guarantees index 0, 1, 2...
                sorted_articles = sorted(self.article_map.items(), key=lambda x: x[1])
                
                for url, idx in sorted_articles:
                    if url in emb_dict:
                        features_list.append(emb_dict[url])
                    else:
                        # Fallback: Random or Zero
                        features_list.append(torch.randn(self.hidden_dim))
                        missing_count += 1
                        
                if missing_count > 0:
                    print(f"      [WARN] {missing_count} articles missing embeddings (used random).")
                
                features = torch.stack(features_list)
                
                # Project if dimensions don't match
                if features.shape[1] != self.hidden_dim:
                    print(f"      [INFO] Projecting features {features.shape[1]} -> {self.hidden_dim}")
                    projector = torch.nn.Linear(features.shape[1], self.hidden_dim)
                    # Initialize nicely (PCA-like would be better but random projection is standard for initialization)
                    features = projector(features).detach()
                    
                return features
            else:
                print(f"      [WARN] {emb_path} not found. Falling back to TF-IDF/Random.")
        
        if self.add_text_features:
            print("   → Generating TF-IDF features...")
            tfidf = TfidfVectorizer(max_features=self.hidden_dim, stop_words='english')
            text_data = self.articles['title'].fillna('') + " " + self.articles['short_description'].fillna('')
            sorted_text = []
            for i in range(num_articles):
                url = list(self.article_map.keys())[list(self.article_map.values()).index(i)]
                row = self.articles[self.articles['url'] == url].iloc[0]
                sorted_text.append(f"{row['title']} {row.get('short_description','')}")
            features = tfidf.fit_transform(sorted_text).toarray()
            return torch.tensor(features, dtype=torch.float32)
            
        print("   → Using Random article embeddings...")
        cat_one_hot = F.one_hot(torch.tensor(self.articles['category_idx'].values), num_classes=len(self.category_encoder.classes_)).float()
        projector = torch.nn.Linear(cat_one_hot.shape[1], self.hidden_dim)
        features = projector(cat_one_hot).detach()
        return features

    def build_full_hetero_graph(self) -> 'HeteroData':
        """
        Build full heterogeneous graph including DIRECT REPLY edges.
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
        num_cats = len(self.category_encoder.classes_)
        data['category'].x = torch.eye(num_cats)
        
        # ===== EDGES =====
        # 1. User -> Article (comments)
        src = torch.tensor(self.replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(self.replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # 2. Article -> Category (belongs_to)
        article_indices = []
        category_indices = []
        for url, cat_idx in zip(self.articles['url'], self.articles['category_idx']):
            if url in self.article_map:
                article_indices.append(self.article_map[url])
                category_indices.append(cat_idx)
        data['article', 'belongs_to', 'category'].edge_index = torch.tensor(
            [article_indices, category_indices], dtype=torch.long
        )
        
        # 3. User -> User (Shared Interest - Implicit)
        user_articles = self.replies.groupby('user_idx')['article_idx'].apply(set).to_dict()
        implicit_edges = []
        user_ids = list(user_articles.keys())
        
        print("   → Computing implicit user-user edges (shared history)...")
        if len(user_ids) < 10000:
            for i in tqdm(range(len(user_ids))):
                u1 = user_ids[i]
                for j in range(i + 1, len(user_ids)):
                    u2 = user_ids[j]
                    if len(user_articles[u1] & user_articles[u2]) >= 2:
                        implicit_edges.append([u1, u2])
                        implicit_edges.append([u2, u1])
            if implicit_edges:
                data['user', 'interacts_with', 'user'].edge_index = torch.tensor(implicit_edges, dtype=torch.long).T
                
        # 4. User -> User (Direct Replies - Explicit)
        if self.no_social:
            print("   -> Skipping explicit social edges (--no-social flag active)")
        else:
            print("   → Computing explicit replies (social)...")
            social_df = self.replies.dropna(subset=['reply_idx', 'parent_idx'])
        
            if not social_df.empty:
                social_src = torch.tensor(social_df['reply_idx'].values.astype(int), dtype=torch.long)
                social_dst = torch.tensor(social_df['parent_idx'].values.astype(int), dtype=torch.long)
                
                # (Replier) -> replies_to -> (Parent)
                data['user', 'replies_to', 'user'].edge_index = torch.stack([social_src, social_dst])
                print(f"      Added {len(social_df)} social edges.")
            else:
                print("      [WARN] No social edges found.")
        
        data = T.ToUndirected()(data)
        
        save_path = self.output_dir / 'full_hetero_graph.pt'
        torch.save(data, save_path)
        self._save_mappings()
        
        print(f"   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Saved to: {save_path}")
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', default='hetero')
    parser.add_argument('--articles', default='data/raw/articles.csv')
    parser.add_argument('--replies', default='data/raw/replies.csv')
    parser.add_argument('--output', default='data/processed_phobert')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--use-phobert', action='store_true')
    parser.add_argument('--add-text-features', action='store_true')
    
    parser.add_argument('--no-social', action='store_true', help='Disable creation of social edges')
    
    args = parser.parse_args()
    
    converter = GNNDataConverter(
        articles_path=args.articles,
        replies_path=args.replies,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        use_phobert=args.use_phobert,
        add_text_features=args.add_text_features,
        no_social=args.no_social
    )
    
    if args.graph_type == 'hetero':
        converter.build_full_hetero_graph()

if __name__ == "__main__":
    main()
