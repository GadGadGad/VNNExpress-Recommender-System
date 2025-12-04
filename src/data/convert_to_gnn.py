"""
Convert Crawled VnExpress Data to GNN-Ready Format
===================================================
This script provides comprehensive graph construction options for GNN training:

1. User-Article Bipartite Graph (for recommendations)
2. User-User Graph (based on shared article interactions)
3. Article-Article Graph (based on shared commenters / same category)
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
        add_text_features: bool = False,
        text_max_features: int = 1000
    ):
        self.articles_path = articles_path
        self.replies_path = replies_path
        self.users_path = users_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.hidden_dim = hidden_dim
        self.add_text_features = add_text_features
        self.text_max_features = text_max_features
        
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
        
        # Load articles
        print(f"\nLoading articles from {self.articles_path}...")
        self.articles = pd.read_csv(self.articles_path)
        print(f"   → {len(self.articles):,} articles loaded")
        
        # Load replies
        print(f"Loading replies from {self.replies_path}...")
        self.replies = pd.read_csv(self.replies_path)
        
        # Clean replies - remove NO_COMMENT markers
        self.replies = self.replies[self.replies['parent_user_id'] != 'NO_COMMENT'].copy()
        
        # Clean user IDs
        def clean_id(val):
            try:
                return str(int(float(val)))
            except:
                return str(val)
        
        self.replies['user_id'] = self.replies['parent_user_id'].apply(clean_id)
        
        # Filter to valid articles only
        valid_urls = set(self.articles['url'].unique())
        self.replies = self.replies[self.replies['article_url'].isin(valid_urls)]
        print(f"   → {len(self.replies):,} valid interactions")
        
        # Load user profiles (optional)
        if os.path.exists(self.users_path):
            print(f"Loading users from {self.users_path}...")
            self.users = pd.read_csv(self.users_path)
            print(f"   → {len(self.users):,} users loaded")
        
        # Create ID mappings
        self._create_mappings()
    
    def apply_kcore_filter(self, min_user_interactions: int = 5, min_article_interactions: int = 5):
        """
        Apply k-core filtering to keep only active users and popular articles.
        
        This iteratively removes:
        - Users with fewer than min_user_interactions
        - Articles with fewer than min_article_interactions
        
        Until convergence (no more removals).
        
        Args:
            min_user_interactions: Minimum interactions for a user to be kept
            min_article_interactions: Minimum interactions for an article to be kept
        """
        print(f"\nApplying k-core filtering (min_user={min_user_interactions}, min_article={min_article_interactions})...")
        print(f"   Before: {len(self.replies):,} interactions")
        
        prev_len = 0
        iteration = 0
        
        while len(self.replies) != prev_len:
            prev_len = len(self.replies)
            iteration += 1
            
            # Count interactions per user
            user_counts = self.replies['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            
            # Count interactions per article
            article_counts = self.replies['article_url'].value_counts()
            valid_articles = article_counts[article_counts >= min_article_interactions].index
            
            # Filter
            self.replies = self.replies[
                (self.replies['user_id'].isin(valid_users)) &
                (self.replies['article_url'].isin(valid_articles))
            ].copy()
            
        print(f"   After {iteration} iterations: {len(self.replies):,} interactions")
        
        # Also filter articles dataframe
        valid_article_urls = self.replies['article_url'].unique()
        original_articles = len(self.articles)
        self.articles = self.articles[self.articles['url'].isin(valid_article_urls)].copy()
        
        # Print stats
        unique_users = self.replies['user_id'].nunique()
        unique_articles = self.replies['article_url'].nunique()
        
        print(f"   Users: {unique_users:,}")
        print(f"   Articles: {unique_articles:,} (removed {original_articles - len(self.articles):,})")
        
        # Calculate new density
        density = len(self.replies) / (unique_users * unique_articles) * 100
        print(f"   New density: {density:.4f}%")
        
        # Recreate mappings with filtered data
        self._create_mappings()
        
    def _create_mappings(self):
        """Create ID to index mappings."""
        unique_users = self.replies['user_id'].unique()
        unique_articles = self.articles['url'].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.article_map = {a: i for i, a in enumerate(unique_articles)}
        
        # Add index columns
        self.replies['user_idx'] = self.replies['user_id'].map(self.user_map)
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
        
        # Option 1: Random learnable embeddings
        features = torch.randn(num_users, self.hidden_dim)
        
        # Option 2: Activity-based features (if we want richer features)
        # Could add: comment count, avg reactions, etc.
        
        return features
    
    def _create_article_features(self) -> torch.Tensor:
        """Create article node features."""
        num_articles = len(self.article_map)
        
        # Category one-hot encoding
        cat_tensor = torch.tensor(self.articles['category_idx'].values)
        cat_one_hot = F.one_hot(cat_tensor, num_classes=len(self.category_encoder.classes_)).float()
        
        if self.add_text_features:
            print("   → Adding text features (TF-IDF)...")
            # TF-IDF on titles
            titles = self.articles['title'].fillna('').tolist()
            tfidf = TfidfVectorizer(max_features=self.text_max_features)
            text_features = tfidf.fit_transform(titles).toarray()
            text_tensor = torch.tensor(text_features, dtype=torch.float32)
            
            # Combine category + text features
            combined = torch.cat([cat_one_hot, text_tensor], dim=1)
            
            # Project to hidden_dim
            projector = torch.nn.Linear(combined.shape[1], self.hidden_dim)
            features = projector(combined).detach()
        else:
            # Project category one-hot to hidden_dim
            projector = torch.nn.Linear(cat_one_hot.shape[1], self.hidden_dim)
            features = projector(cat_one_hot).detach()
        
        return features
    
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
        
        # Node features
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # Edge indices
        src = torch.tensor(self.replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(self.replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # Edge weights (based on reaction count)
        reactions = pd.to_numeric(self.replies['parent_reactions'], errors='coerce').fillna(1).values
        reactions = np.clip(reactions, 1, None)  # Min weight of 1
        data['user', 'comments', 'article'].edge_weight = torch.tensor(reactions, dtype=torch.float32)
        
        # Make undirected for bidirectional message passing
        data = T.ToUndirected()(data)
        
        # Save
        save_path = self.output_dir / 'user_article_graph.pt'
        torch.save(data, save_path)
        self._save_mappings()
        
        print(f"   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Edges: {data['user', 'comments', 'article'].edge_index.shape}")
        print(f"   [OK] Saved to: {save_path}")
        
        return data
    
    def generate_negative_samples(
        self,
        num_negatives: int = 1,
        strategy: str = 'random',
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate negative samples (user-article pairs that don't exist).
        
        Args:
            num_negatives: Number of negatives per positive sample
            strategy: 'random', 'popular', or 'hard'
                - random: Uniform random negative articles
                - popular: Sample from popular articles (harder negatives)
                - hard: Sample from same category (hardest negatives)
            seed: Random seed for reproducibility
        
        Returns:
            neg_users, neg_articles: Arrays of negative user-article pairs
        """
        print(f"\nGenerating negative samples (strategy={strategy}, ratio={num_negatives})...")
        
        np.random.seed(seed)
        
        num_users = len(self.user_map)
        num_articles = len(self.article_map)
        num_positives = len(self.replies)
        num_neg_samples = num_positives * num_negatives
        
        # Build positive set for fast lookup
        positive_set = set(
            zip(self.replies['user_idx'].values, self.replies['article_idx'].values)
        )
        
        neg_users = []
        neg_articles = []
        
        if strategy == 'random':
            # Uniform random sampling
            print("   -> Random negative sampling...")
            attempts = 0
            max_attempts = num_neg_samples * 10
            
            while len(neg_users) < num_neg_samples and attempts < max_attempts:
                batch_size = min(num_neg_samples - len(neg_users), 10000)
                u_batch = np.random.randint(0, num_users, batch_size)
                a_batch = np.random.randint(0, num_articles, batch_size)
                
                for u, a in zip(u_batch, a_batch):
                    if (u, a) not in positive_set:
                        neg_users.append(u)
                        neg_articles.append(a)
                        if len(neg_users) >= num_neg_samples:
                            break
                
                attempts += batch_size
            
        elif strategy == 'popular':
            # Sample from popular articles (harder negatives)
            print("   -> Popularity-based negative sampling...")
            article_counts = self.replies['article_idx'].value_counts()
            article_probs = article_counts / article_counts.sum()
            popular_articles = article_probs.index.values
            probs = article_probs.values
            
            user_indices = self.replies['user_idx'].unique()
            
            for _ in tqdm(range(num_negatives), desc="   Generating negatives"):
                for user_idx in user_indices:
                    # Sample from popular articles
                    for _ in range(10):  # Max attempts per user
                        article_idx = np.random.choice(popular_articles, p=probs)
                        if (user_idx, article_idx) not in positive_set:
                            neg_users.append(user_idx)
                            neg_articles.append(article_idx)
                            break
                    
                    if len(neg_users) >= num_neg_samples:
                        break
                        
        elif strategy == 'hard':
            # Sample from same category (hardest negatives)
            print("   -> Hard negative sampling (same category)...")
            
            # Build article -> category mapping
            article_to_cat = {}
            cat_to_articles = defaultdict(list)
            
            for url, cat_idx in zip(self.articles['url'], self.articles['category_idx']):
                if url in self.article_map:
                    art_idx = self.article_map[url]
                    article_to_cat[art_idx] = cat_idx
                    cat_to_articles[cat_idx].append(art_idx)
            
            # For each positive, sample negative from same category
            for _, row in tqdm(self.replies.iterrows(), total=len(self.replies), 
                             desc="   Generating hard negatives"):
                user_idx = int(row['user_idx'])
                art_idx = int(row['article_idx'])
                
                if art_idx in article_to_cat:
                    cat = article_to_cat[art_idx]
                    same_cat_articles = cat_to_articles[cat]
                    
                    for _ in range(num_negatives):
                        for _ in range(10):  # Max attempts
                            neg_art = np.random.choice(same_cat_articles)
                            if (user_idx, neg_art) not in positive_set:
                                neg_users.append(user_idx)
                                neg_articles.append(neg_art)
                                break
                        
                        if len(neg_users) >= num_neg_samples:
                            break
                
                if len(neg_users) >= num_neg_samples:
                    break
        
        neg_users = np.array(neg_users[:num_neg_samples])
        neg_articles = np.array(neg_articles[:num_neg_samples])
        
        print(f"   [OK] Generated {len(neg_users):,} negative samples")
        
        return neg_users, neg_articles
    
    def build_graph_with_negatives(
        self,
        neg_ratio: int = 1,
        neg_strategy: str = 'random',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict:
        """
        Build graph with explicit positive and negative samples, pre-split into train/val/test.
        
        This is the format commonly used in recommendation papers.
        
        Args:
            neg_ratio: Number of negative samples per positive
            neg_strategy: Negative sampling strategy
            train_ratio: Proportion for training
            val_ratio: Proportion for validation (test = 1 - train - val)
            seed: Random seed
        
        Returns:
            Dictionary containing train/val/test data with pos/neg samples
        """
        print("\nBuilding Graph with Negative Samples...")
        
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        np.random.seed(seed)
        
        # Get positive edges
        pos_users = self.replies['user_idx'].values
        pos_articles = self.replies['article_idx'].values
        num_positives = len(pos_users)
        
        # Shuffle and split positives
        indices = np.random.permutation(num_positives)
        train_end = int(num_positives * train_ratio)
        val_end = int(num_positives * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"   Positive splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        # Generate negative samples
        neg_users, neg_articles = self.generate_negative_samples(
            num_negatives=neg_ratio,
            strategy=neg_strategy,
            seed=seed
        )
        
        # Split negatives proportionally
        neg_indices = np.random.permutation(len(neg_users))
        neg_train_end = int(len(neg_users) * train_ratio)
        neg_val_end = int(len(neg_users) * (train_ratio + val_ratio))
        
        neg_train_idx = neg_indices[:neg_train_end]
        neg_val_idx = neg_indices[neg_train_end:neg_val_end]
        neg_test_idx = neg_indices[neg_val_end:]
        
        print(f"   Negative splits: train={len(neg_train_idx)}, val={len(neg_val_idx)}, test={len(neg_test_idx)}")
        
        # Create base HeteroData with node features
        data = HeteroData()
        data['user'].x = self._create_user_features()
        data['article'].x = self._create_article_features()
        
        # Store all edges for message passing (positive only)
        data['user', 'comments', 'article'].edge_index = torch.stack([
            torch.tensor(pos_users[train_idx], dtype=torch.long),
            torch.tensor(pos_articles[train_idx], dtype=torch.long)
        ])
        
        # Create splits dictionary
        splits = {
            'train': {
                'pos_users': torch.tensor(pos_users[train_idx], dtype=torch.long),
                'pos_articles': torch.tensor(pos_articles[train_idx], dtype=torch.long),
                'neg_users': torch.tensor(neg_users[neg_train_idx], dtype=torch.long),
                'neg_articles': torch.tensor(neg_articles[neg_train_idx], dtype=torch.long),
            },
            'val': {
                'pos_users': torch.tensor(pos_users[val_idx], dtype=torch.long),
                'pos_articles': torch.tensor(pos_articles[val_idx], dtype=torch.long),
                'neg_users': torch.tensor(neg_users[neg_val_idx], dtype=torch.long),
                'neg_articles': torch.tensor(neg_articles[neg_val_idx], dtype=torch.long),
            },
            'test': {
                'pos_users': torch.tensor(pos_users[test_idx], dtype=torch.long),
                'pos_articles': torch.tensor(pos_articles[test_idx], dtype=torch.long),
                'neg_users': torch.tensor(neg_users[neg_test_idx], dtype=torch.long),
                'neg_articles': torch.tensor(neg_articles[neg_test_idx], dtype=torch.long),
            }
        }
        
        # Package everything
        result = {
            'graph': data,
            'splits': splits,
            'num_users': len(self.user_map),
            'num_articles': len(self.article_map),
            'neg_ratio': neg_ratio,
            'neg_strategy': neg_strategy
        }
        
        # Save
        save_path = self.output_dir / 'graph_with_negatives.pt'
        torch.save(result, save_path)
        self._save_mappings()
        
        print(f"\n   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Train edges: {len(train_idx)} pos + {len(neg_train_idx)} neg")
        print(f"   [OK] Val edges: {len(val_idx)} pos + {len(neg_val_idx)} neg")
        print(f"   [OK] Test edges: {len(test_idx)} pos + {len(neg_test_idx)} neg")
        print(f"   [OK] Saved to: {save_path}")
        
        return result
    
    def build_user_user_graph(self, min_shared: int = 2) -> 'Data':
        """
        Build User-User graph based on shared article interactions.
        
        Edge: Two users are connected if they commented on >= min_shared same articles.
        Weight: Number of shared articles.
        """
        print(f"\nBuilding User-User Graph (min_shared={min_shared})...")
        
        try:
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        # Find articles each user commented on
        user_articles = self.replies.groupby('user_idx')['article_idx'].apply(set).to_dict()
        
        # Build edges
        edges = []
        weights = []
        user_ids = list(user_articles.keys())
        
        print("   → Computing user-user edges...")
        for i in tqdm(range(len(user_ids)), desc="   Processing users"):
            u1 = user_ids[i]
            articles_u1 = user_articles[u1]
            
            for j in range(i + 1, len(user_ids)):
                u2 = user_ids[j]
                articles_u2 = user_articles[u2]
                
                shared = len(articles_u1 & articles_u2)
                if shared >= min_shared:
                    edges.append([u1, u2])
                    edges.append([u2, u1])  # Undirected
                    weights.extend([shared, shared])
        
        if not edges:
            print("   No edges found! Try lowering min_shared.")
            return None
        
        # Create Data object
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        data = Data(
            x=self._create_user_features(),
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        
        # Save
        save_path = self.output_dir / 'user_user_graph.pt'
        torch.save(data, save_path)
        self._save_mappings()
        
        print(f"   [OK] Nodes: {data.x.shape[0]}")
        print(f"   [OK] Edges: {edge_index.shape[1]}")
        print(f"   [OK] Saved to: {save_path}")
        
        return data
    
    def build_article_article_graph(self, method: str = 'category') -> 'Data':
        """
        Build Article-Article graph.
        
        Methods:
            - 'category': Connect articles in the same category
            - 'users': Connect articles with shared commenters
        """
        print(f"\nBuilding Article-Article Graph (method={method})...")
        
        try:
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        edges = []
        weights = []
        
        if method == 'category':
            # Group articles by category
            category_articles = self.articles.groupby('category_idx').apply(
                lambda x: [self.article_map.get(url) for url in x['url'] if url in self.article_map]
            ).to_dict()
            
            print("   → Connecting articles by category...")
            for cat, article_list in tqdm(category_articles.items(), desc="   Processing categories"):
                article_list = [a for a in article_list if a is not None]
                for i, a1 in enumerate(article_list):
                    for a2 in article_list[i+1:]:
                        edges.append([a1, a2])
                        edges.append([a2, a1])
                        weights.extend([1.0, 1.0])
                        
        elif method == 'users':
            # Find users who commented on each article
            article_users = self.replies.groupby('article_idx')['user_idx'].apply(set).to_dict()
            article_ids = list(article_users.keys())
            
            print("   → Connecting articles by shared users...")
            for i in tqdm(range(len(article_ids)), desc="   Processing articles"):
                a1 = article_ids[i]
                users_a1 = article_users[a1]
                
                for j in range(i + 1, len(article_ids)):
                    a2 = article_ids[j]
                    users_a2 = article_users[a2]
                    
                    shared = len(users_a1 & users_a2)
                    if shared >= 1:
                        edges.append([a1, a2])
                        edges.append([a2, a1])
                        weights.extend([shared, shared])
        
        if not edges:
            print("   No edges found!")
            return None
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        data = Data(
            x=self._create_article_features(),
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        
        # Save
        save_path = self.output_dir / f'article_article_graph_{method}.pt'
        torch.save(data, save_path)
        self._save_mappings()
        
        print(f"   [OK] Nodes: {data.x.shape[0]}")
        print(f"   [OK] Edges: {edge_index.shape[1]}")
        print(f"   [OK] Saved to: {save_path}")
        
        return data
    
    def build_full_hetero_graph(self) -> 'HeteroData':
        """
        Build full heterogeneous graph with all node types and relations.
        
        Nodes:
            - user: Commenting users
            - article: News articles
            - category: Article categories
        
        Edges:
            - (user, comments, article)
            - (article, belongs_to, category)
            - (user, interacts_with, user) - shared article interactions
        """
        print("\nBuilding Full Heterogeneous Graph...")
        
        try:
            from torch_geometric.data import HeteroData
            import torch_geometric.transforms as T
        except ImportError:
            raise ImportError("Please install torch_geometric: pip install torch-geometric")
        
        data = HeteroData()
        
        # ===== NODE FEATURES =====
        # Users
        data['user'].x = self._create_user_features()
        
        # Articles  
        data['article'].x = self._create_article_features()
        
        # Categories
        num_cats = len(self.category_encoder.classes_)
        data['category'].x = torch.eye(num_cats)  # One-hot identity
        
        # ===== EDGES =====
        # User -> Article (comments)
        src = torch.tensor(self.replies['user_idx'].values, dtype=torch.long)
        dst = torch.tensor(self.replies['article_idx'].values, dtype=torch.long)
        data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])
        
        # Article -> Category (belongs_to)
        article_indices = []
        category_indices = []
        for url, cat_idx in zip(self.articles['url'], self.articles['category_idx']):
            if url in self.article_map:
                article_indices.append(self.article_map[url])
                category_indices.append(cat_idx)
        
        data['article', 'belongs_to', 'category'].edge_index = torch.tensor(
            [article_indices, category_indices], dtype=torch.long
        )
        
        # User -> User (shared articles, min 2)
        user_articles = self.replies.groupby('user_idx')['article_idx'].apply(set).to_dict()
        user_user_edges = []
        user_ids = list(user_articles.keys())
        
        print("   → Computing user-user edges...")
        for i in range(len(user_ids)):
            u1 = user_ids[i]
            for j in range(i + 1, len(user_ids)):
                u2 = user_ids[j]
                if len(user_articles[u1] & user_articles[u2]) >= 2:
                    user_user_edges.append([u1, u2])
                    user_user_edges.append([u2, u1])
        
        if user_user_edges:
            data['user', 'interacts_with', 'user'].edge_index = torch.tensor(
                user_user_edges, dtype=torch.long
            ).T
        
        # Make undirected
        data = T.ToUndirected()(data)
        
        # Save
        save_path = self.output_dir / 'full_hetero_graph.pt'
        torch.save(data, save_path)
        self._save_mappings()
        
        print(f"   [OK] User nodes: {data['user'].x.shape}")
        print(f"   [OK] Article nodes: {data['article'].x.shape}")
        print(f"   [OK] Category nodes: {data['category'].x.shape}")
        print(f"   [OK] Saved to: {save_path}")
        
        return data
    
    def export_to_networkx(self, graph_type: str = 'user-article'):
        """Export graph to NetworkX format for visualization/analysis."""
        print(f"\nExporting to NetworkX ({graph_type})...")
        
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Please install networkx: pip install networkx")
        
        G = nx.Graph()
        
        if graph_type == 'user-article':
            # Add nodes
            for user_id, idx in self.user_map.items():
                G.add_node(f"u_{idx}", node_type='user', original_id=user_id)
            
            for article_url, idx in self.article_map.items():
                cat = self.articles[self.articles['url'] == article_url]['category'].values
                cat = cat[0] if len(cat) > 0 else 'unknown'
                G.add_node(f"a_{idx}", node_type='article', category=cat)
            
            # Add edges
            for _, row in self.replies.iterrows():
                G.add_edge(
                    f"u_{int(row['user_idx'])}",
                    f"a_{int(row['article_idx'])}",
                    weight=float(row.get('parent_reactions', 1) or 1)
                )
        
        # Save
        save_path = self.output_dir / f'{graph_type}_networkx.gpickle'
        nx.write_gpickle(G, save_path)
        
        print(f"   [OK] Nodes: {G.number_of_nodes()}")
        print(f"   [OK] Edges: {G.number_of_edges()}")
        print(f"   [OK] Saved to: {save_path}")
        
        return G
    
    def export_to_dgl(self, graph_type: str = 'user-article'):
        """Export to DGL format."""
        print(f"\nExporting to DGL ({graph_type})...")
        
        try:
            import dgl
        except ImportError:
            raise ImportError("Please install dgl: pip install dgl")
        
        if graph_type == 'user-article':
            src = torch.tensor(self.replies['user_idx'].values, dtype=torch.long)
            dst = torch.tensor(self.replies['article_idx'].values, dtype=torch.long)
            
            # Create heterogeneous graph
            graph_data = {
                ('user', 'comments', 'article'): (src, dst),
                ('article', 'commented_by', 'user'): (dst, src),  # Reverse
            }
            
            g = dgl.heterograph(graph_data)
            
            # Add features
            g.nodes['user'].data['x'] = self._create_user_features()
            g.nodes['article'].data['x'] = self._create_article_features()
            
            # Save
            save_path = self.output_dir / 'user_article_dgl.bin'
            dgl.save_graphs(str(save_path), [g])
            self._save_mappings()
            
            print(f"   [OK] User nodes: {g.num_nodes('user')}")
            print(f"   [OK] Article nodes: {g.num_nodes('article')}")
            print(f"   [OK] Saved to: {save_path}")
            
            return g
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert crawled VnExpress data to GNN-ready format'
    )
    parser.add_argument(
        '--graph-type', '-g',
        choices=['user-article', 'user-user', 'article-article', 'hetero', 'with-negatives', 'all'],
        default='user-article',
        help='Type of graph to build (default: user-article)'
    )
    parser.add_argument(
        '--articles', '-a',
        default='data/raw/articles.csv',
        help='Path to articles CSV'
    )
    parser.add_argument(
        '--replies', '-r',
        default='data/raw/replies.csv',
        help='Path to replies CSV'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--hidden-dim', '-d',
        type=int,
        default=64,
        help='Hidden dimension for node features'
    )
    parser.add_argument(
        '--add-text-features',
        action='store_true',
        help='Add TF-IDF text features to article nodes'
    )
    parser.add_argument(
        '--export-networkx',
        action='store_true',
        help='Also export to NetworkX format'
    )
    parser.add_argument(
        '--export-dgl',
        action='store_true',
        help='Also export to DGL format'
    )
    
    # Negative sampling options
    parser.add_argument(
        '--neg-ratio',
        type=int,
        default=1,
        help='Number of negative samples per positive (default: 1)'
    )
    parser.add_argument(
        '--neg-strategy',
        choices=['random', 'popular', 'hard'],
        default='random',
        help='Negative sampling strategy: random, popular, hard (default: random)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # K-core filtering options
    parser.add_argument(
        '--min-user-interactions',
        type=int,
        default=0,
        help='Minimum interactions per user (0 = no filtering, 5 recommended)'
    )
    parser.add_argument(
        '--min-article-interactions',
        type=int,
        default=0,
        help='Minimum interactions per article (0 = no filtering, 5 recommended)'
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = GNNDataConverter(
        articles_path=args.articles,
        replies_path=args.replies,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        add_text_features=args.add_text_features
    )
    
    # Apply k-core filtering if specified
    if args.min_user_interactions > 0 or args.min_article_interactions > 0:
        converter.apply_kcore_filter(
            min_user_interactions=max(1, args.min_user_interactions),
            min_article_interactions=max(1, args.min_article_interactions)
        )
    
    # Build requested graph type(s)
    if args.graph_type in ['user-article', 'all']:
        converter.build_user_article_graph()
        
    if args.graph_type in ['user-user', 'all']:
        converter.build_user_user_graph()
        
    if args.graph_type in ['article-article', 'all']:
        converter.build_article_article_graph(method='category')
        converter.build_article_article_graph(method='users')
        
    if args.graph_type in ['hetero', 'all']:
        converter.build_full_hetero_graph()
    
    if args.graph_type in ['with-negatives', 'all']:
        converter.build_graph_with_negatives(
            neg_ratio=args.neg_ratio,
            neg_strategy=args.neg_strategy,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    
    # Optional exports
    if args.export_networkx:
        converter.export_to_networkx()
        
    if args.export_dgl:
        converter.export_to_dgl()
    
    print("\n" + "=" * 60)
    print("GNN Data Conversion Complete!")
    print(f"   Output directory: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
