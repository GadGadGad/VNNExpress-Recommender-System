"""
Content-Based Data Loader
=========================
Extends LightGCL DataLoader with article text loading for PhoBERT encoding.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional

from src.data.dataloader_lightgcl import LightGCLDataLoader, load_data as load_lightgcl_data


class ContentDataLoader(LightGCLDataLoader):
    """
    Data Loader for Content-Based models.
    Extends LightGCLDataLoader with article text features.
    """
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.article_texts = None
        
    def load_article_texts(
        self,
        text_columns: List[str] = ['title', 'short_description'],
        sep: str = ' '
    ) -> List[str]:
        """
        Load and prepare article texts for encoding.
        
        Args:
            text_columns: Which columns to concatenate for text
            sep: Separator between columns
            
        Returns:
            List of texts in same order as item indices (0, 1, 2, ...)
        """
        if self.articles_df is None:
            articles_path = os.path.join(self.raw_path, 'articles.csv')
            self.articles_df = pd.read_csv(articles_path)
            
        print(f"\n[ContentDataLoader] Preparing article texts...")
        print(f"  Text columns: {text_columns}")
        print(f"  Total articles in CSV: {len(self.articles_df)}")
        print(f"  Items in dataset: {self.n_items}")
        
        # Build URL to text mapping
        url_to_text = {}
        for _, row in self.articles_df.iterrows():
            parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    parts.append(str(row[col]).strip())
            url_to_text[row['url']] = sep.join(parts) if parts else ""
        
        # Create ordered list matching item indices
        article_texts = []
        missing_count = 0
        
        for idx in range(self.n_items):
            url = self.idx2item.get(idx, "")
            text = url_to_text.get(url, "")
            
            if not text:
                # Fallback: use URL as text
                text = url.split('/')[-1].replace('-', ' ').replace('.html', '')
                missing_count += 1
                
            article_texts.append(text)
            
        print(f"  Articles with text: {self.n_items - missing_count}")
        print(f"  Missing (using URL fallback): {missing_count}")
        
        # Sample
        if article_texts:
            sample_idx = min(3, len(article_texts))
            print(f"\n  Sample texts:")
            for i in range(sample_idx):
                print(f"    [{i}] {article_texts[i][:100]}...")
                
        self.article_texts = article_texts
        return article_texts
    
    def get_article_text(self, item_idx: int) -> str:
        """Get text for a single article by index"""
        if self.article_texts is None:
            self.load_article_texts()
        return self.article_texts[item_idx] if item_idx < len(self.article_texts) else ""
    
    def get_user_history_texts(self, user_idx: int, train_dict: Dict) -> List[str]:
        """Get texts of articles a user has interacted with"""
        if self.article_texts is None:
            self.load_article_texts()
            
        item_indices = train_dict.get(user_idx, set())
        return [self.article_texts[i] for i in item_indices if i < len(self.article_texts)]


def load_content_data(
    data_path: str,
    text_columns: List[str] = ['title', 'short_description'],
    force_reload: bool = False
) -> Tuple[Dict, ContentDataLoader]:
    """
    Load data for content-based models.
    
    Returns:
        data_dict: {
            'n_users', 'n_items', 
            'train_data', 'train_dict', 'test_dict',
            'article_texts', 'idx2item', 'item2idx', ...
        }
        loader: ContentDataLoader instance
    """
    # Use LightGCL loader first
    n_users, n_items, train_data, train_dict, test_data, base_loader = load_lightgcl_data(
        data_path, force_reload=force_reload
    )
    
    # Create content loader with same data
    loader = ContentDataLoader(data_path)
    loader.n_users = base_loader.n_users
    loader.n_items = base_loader.n_items
    loader.user2idx = base_loader.user2idx
    loader.idx2user = base_loader.idx2user
    loader.item2idx = base_loader.item2idx
    loader.idx2item = base_loader.idx2item
    loader.articles_df = base_loader.articles_df
    
    # Load article texts
    article_texts = loader.load_article_texts(text_columns=text_columns)
    
    return {
        'n_users': n_users,
        'n_items': n_items,
        'train_data': train_data,
        'train_dict': train_dict,
        'test_dict': test_data,  # Already a dict {user: [items]}
        'article_texts': article_texts,
        'idx2item': loader.idx2item,
        'item2idx': loader.item2idx,
        'idx2user': loader.idx2user,
        'user2idx': loader.user2idx
    }, loader


if __name__ == '__main__':
    # Test
    data_dict, loader = load_content_data('data', text_columns=['title', 'short_description'])
    
    print(f"\n{'='*60}")
    print("Data Summary:")
    print(f"  Users: {data_dict['n_users']}")
    print(f"  Items: {data_dict['n_items']}")
    print(f"  Train interactions: {len(data_dict['train_data'])}")
    print(f"  Test users: {len(data_dict['test_dict'])}")
    print(f"  Article texts: {len(data_dict['article_texts'])}")
