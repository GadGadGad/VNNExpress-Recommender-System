
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_sparsity(data_dir='data/raw'):
    print(f"Analyzing data in {data_dir}...")
    
    replies_path = Path(data_dir) / 'replies.csv'
    if not replies_path.exists():
        print("replies.csv not found.")
        return

    replies = pd.read_csv(replies_path)
    
    # Clean User IDs
    def clean_id(val):
        try:
            return str(int(float(val))) if pd.notna(val) else None
        except:
             return None
    
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()]
    
    n_users = replies['user_id'].nunique()
    n_items = replies['article_url'].nunique()
    n_interactions = len(replies)
    
    print(f"\n--- Statistics ---")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Interactions: {n_interactions}")
    
    density = n_interactions / (n_users * n_items)
    sparsity = 1 - density
    print(f"Density: {density:.6f} ({density*100:.4f}%)")
    print(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
    
    # Degree Distribution
    user_counts = replies['user_id'].value_counts()
    item_counts = replies['article_url'].value_counts()
    
    print(f"\n--- User Degrees ---")
    print(f"Mean: {user_counts.mean():.2f}")
    print(f"Median: {user_counts.median():.2f}")
    print(f"Max: {user_counts.max()}")
    print(f"Users with < 5 interactions: {len(user_counts[user_counts < 5])} ({len(user_counts[user_counts < 5])/n_users*100:.1f}%)")

    print(f"\n--- Item Degrees ---")
    print(f"Mean: {item_counts.mean():.2f}")
    print(f"Median: {item_counts.median():.2f}")
    print(f"Max: {item_counts.max()}")
    
analyze_sparsity()
