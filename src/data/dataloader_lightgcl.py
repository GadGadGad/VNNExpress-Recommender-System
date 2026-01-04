"""
LightGCL Data Loader
====================
Load và xử lý data VnExpress cho LightGCL model.

Interaction: User (parent_user_id) comment trên Article (article_url)
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import pickle
from typing import Tuple, Dict, List, Optional


class LightGCLDataLoader:
    """
    Data Loader cho LightGCL - VnExpress Dataset
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_path = os.path.join(data_path, 'raw')
        self.processed_path = os.path.join(data_path, 'processed')
        
        try:
            os.makedirs(self.processed_path, exist_ok=True)
        except OSError:
            # Handle read-only file system
            if os.path.exists(self.processed_path):
                print(f"  [Info] Processed path {self.processed_path} exists and is read-only. Using it for loading.")
            else:
                # Fallback to local writable dir if input is read-only and processed folder missing
                print(f"  [Warning] Path {self.processed_path} is read-only and missing. Fallback to ./processed_cache")
                self.processed_path = './processed_cache'
                os.makedirs(self.processed_path, exist_ok=True)
        
        # DataFrames
        self.users_df = None
        self.articles_df = None
        self.replies_df = None
        
        # Mappings
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        
        self.n_users = 0
        self.n_items = 0
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files"""
        print("=" * 60)
        print("Loading Raw Data")
        print("=" * 60)
        
        # Load files
        self.users_df = pd.read_csv(os.path.join(self.raw_path, 'user_profiles.csv'))
        self.articles_df = pd.read_csv(os.path.join(self.raw_path, 'articles.csv'))
        self.replies_df = pd.read_csv(os.path.join(self.raw_path, 'replies.csv'))
        
        print(f"  Users: {len(self.users_df):,}")
        print(f"  Articles: {len(self.articles_df):,}")
        print(f"  Replies: {len(self.replies_df):,}")
        
        return self.users_df, self.articles_df, self.replies_df
    
    def preprocess(self, min_user_interactions: int = 1, 
                   min_article_interactions: int = 1) -> pd.DataFrame:
        """
        Preprocess data:
        1. Clean user IDs
        2. Remove NO_COMMENT entries
        3. Filter by minimum interactions (k-core filtering)
        
        Args:
            min_user_interactions: Minimum comments per user
            min_article_interactions: Minimum comments per article
        """
        print("\n" + "=" * 60)
        print("Preprocessing Data")
        print("=" * 60)
        
        # Copy để không modify original
        df = self.replies_df.copy()
        
        # Remove NO_COMMENT entries
        df = df[df['parent_user_id'] != 'NO_COMMENT'].copy()
        print(f"  After removing NO_COMMENT: {len(df):,} rows")
        
        # Clean user IDs - convert to string, remove NaN
        df['parent_user_id'] = df['parent_user_id'].astype(str)
        df = df[df['parent_user_id'].notna() & (df['parent_user_id'] != 'nan')]
        
        # Clean article URLs
        df = df[df['article_url'].notna()]
        
        print(f"  After cleaning IDs: {len(df):,} rows")
        
        # K-core filtering
        if min_user_interactions > 1 or min_article_interactions > 1:
            print(f"\n  Applying k-core filter (min_user={min_user_interactions}, min_article={min_article_interactions})...")
            
            prev_len = 0
            iteration = 0
            
            while len(df) != prev_len:
                prev_len = len(df)
                iteration += 1
                
                # Filter users
                user_counts = df['parent_user_id'].value_counts()
                valid_users = user_counts[user_counts >= min_user_interactions].index
                
                # Filter articles
                article_counts = df['article_url'].value_counts()
                valid_articles = article_counts[article_counts >= min_article_interactions].index
                
                df = df[
                    (df['parent_user_id'].isin(valid_users)) &
                    (df['article_url'].isin(valid_articles))
                ].copy()
                
            print(f"  After {iteration} iterations: {len(df):,} interactions")
        
        # Create unique interactions (user, article)
        df = df[['parent_user_id', 'article_url']].drop_duplicates()
        
        print(f"\n  Unique interactions: {len(df):,}")
        print(f"  Unique users: {df['parent_user_id'].nunique():,}")
        print(f"  Unique articles: {df['article_url'].nunique():,}")
        
        # Calculate density
        density = len(df) / (df['parent_user_id'].nunique() * df['article_url'].nunique()) * 100
        print(f"  Density: {density:.4f}%")
        
        self.interactions_df = df
        return df
    
    def create_mappings(self):
        """Create ID to index mappings"""
        print("\n  Creating ID mappings...")
        
        unique_users = self.interactions_df['parent_user_id'].unique()
        unique_articles = self.interactions_df['article_url'].unique()
        
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.item2idx = {a: i for i, a in enumerate(unique_articles)}
        self.idx2item = {i: a for a, i in self.item2idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_articles)
        
        print(f"  Mapped users: {self.n_users}")
        print(f"  Mapped items (articles): {self.n_items}")
        
    def create_interactions(self) -> List[Tuple[int, int]]:
        """
        Create list of (user_idx, item_idx) interactions
        """
        interactions = []
        
        for _, row in self.interactions_df.iterrows():
            user_id = row['parent_user_id']
            article_url = row['article_url']
            
            if user_id in self.user2idx and article_url in self.item2idx:
                u_idx = self.user2idx[user_id]
                i_idx = self.item2idx[article_url]
                interactions.append((u_idx, i_idx))
                
        print(f"  Created {len(interactions):,} interaction pairs")
        return interactions
    
    def train_test_split(self, interactions: List[Tuple[int, int]], 
                         test_ratio: float = 0.2,
                         seed: int = 42) -> Tuple[List, Dict, Dict]:
        """
        Split data into train and test sets.
        Ensures each user in test has at least 1 item in train.
        
        Returns:
            train_data: List of (user, item) tuples
            train_dict: Dict {user: set(items)}
            test_data: Dict {user: [items]}
        """
        print("\n" + "=" * 60)
        print(f"Train/Test Split (test_ratio={test_ratio})")
        print("=" * 60)
        
        np.random.seed(seed)
        
        # Group by user
        user_items = defaultdict(list)
        for u, i in interactions:
            user_items[u].append(i)
            
        train_data = []
        test_data = {}
        train_dict = defaultdict(set)
        
        n_train_only = 0
        n_split = 0
        
        for user, items in user_items.items():
            if len(items) < 2:
                # User has only 1 interaction: train only
                train_data.extend([(user, item) for item in items])
                train_dict[user].update(items)
                n_train_only += 1
            else:
                # Split train/test
                n_test = max(1, int(len(items) * test_ratio))
                items_shuffled = items.copy()
                np.random.shuffle(items_shuffled)
                
                test_items = items_shuffled[:n_test]
                train_items = items_shuffled[n_test:]
                
                train_data.extend([(user, item) for item in train_items])
                train_dict[user].update(train_items)
                test_data[user] = test_items
                n_split += 1
                
        print(f"  Users with train only: {n_train_only}")
        print(f"  Users with train+test: {n_split}")
        print(f"  Train interactions: {len(train_data):,}")
        print(f"  Test users: {len(test_data)}")
        print(f"  Test interactions: {sum(len(v) for v in test_data.values()):,}")
        
        return train_data, dict(train_dict), test_data
    
    def save_processed(self, train_data, train_dict, test_data):
        """Save processed data to pickle"""
        save_data = {
            'train_data': train_data,
            'train_dict': train_dict,
            'test_data': test_data,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user2idx': self.user2idx,
            'idx2user': self.idx2user,
            'item2idx': self.item2idx,
            'idx2item': self.idx2item
        }
        
        save_path = os.path.join(self.processed_path, 'lightgcl_data.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"\n  Saved to: {save_path}")
        
    def load_processed(self) -> Optional[Tuple]:
        """Load processed data if exists"""
        load_path = os.path.join(self.processed_path, 'lightgcl_data.pkl')
        
        if not os.path.exists(load_path):
            return None
            
        print(f"Loading processed data from {load_path}...")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.user2idx = data['user2idx']
        self.idx2user = data['idx2user']
        self.item2idx = data['item2idx']
        self.idx2item = data['idx2item']
        
        print(f"  Users: {self.n_users}, Items: {self.n_items}")
        print(f"  Train: {len(data['train_data']):,}, Test users: {len(data['test_data'])}")
        
        return data['train_data'], data['train_dict'], data['test_data']
    
    def get_article_info(self, item_idx: int) -> Optional[Dict]:
        """Get article info from index"""
        if item_idx not in self.idx2item:
            return None
            
        url = self.idx2item[item_idx]
        article = self.articles_df[self.articles_df['url'] == url]
        
        if len(article) == 0:
            return None
            
        return {
            'url': url,
            'title': article['title'].values[0],
            'category': article['category'].values[0]
        }


def load_data(data_path: str, 
              min_user_interactions: int = 2,
              min_article_interactions: int = 2,
              test_ratio: float = 0.2,
              force_reload: bool = False,
              seed: int = 42) -> Tuple:
    """
    Helper function to load data for LightGCL
    
    Args:
        data_path: Path to data directory
        min_user_interactions: Minimum interactions per user (k-core)
        min_article_interactions: Minimum interactions per article (k-core)
        test_ratio: Test set ratio
        force_reload: Force reload from raw data
        seed: Random seed
        
    Returns:
        n_users, n_items, train_data, train_dict, test_data, loader
    """
    loader = LightGCLDataLoader(data_path)
    
    # Load processed data
    if not force_reload:
        result = loader.load_processed()
        if result is not None:
            train_data, train_dict, test_data = result
            return (loader.n_users, loader.n_items,
                    train_data, train_dict, test_data, loader)
    
    # Process from raw
    loader.load_raw_data()
    loader.preprocess(
        min_user_interactions=min_user_interactions,
        min_article_interactions=min_article_interactions
    )
    loader.create_mappings()
    interactions = loader.create_interactions()
    train_data, train_dict, test_data = loader.train_test_split(
        interactions, test_ratio=test_ratio, seed=seed
    )
    loader.save_processed(train_data, train_dict, test_data)
    
    return (loader.n_users, loader.n_items,
            train_data, train_dict, test_data, loader)


if __name__ == '__main__':
    # Test loading
    data_path = 'data'
    
    loader = LightGCLDataLoader(data_path)
    loader.load_raw_data()
    loader.preprocess(min_user_interactions=2, min_article_interactions=2)
    loader.create_mappings()
    interactions = loader.create_interactions()
    train_data, train_dict, test_data = loader.train_test_split(interactions)
    loader.save_processed(train_data, train_dict, test_data)