import pandas as pd
import argparse

def check_sparsity(replies_path):
    print(f"Loading {replies_path}...")
    df = pd.read_csv(replies_path)
    
    # Extract interactions
    # From inspect_recs: 
    # df1 = replies_df[['parent_user_id', 'article_url']]
    # df2 = replies_df[['reply_user_id', 'article_url']]
    
    df1 = df[['parent_user_id', 'article_url']].rename(columns={'parent_user_id': 'user_id'})
    df2 = df[['reply_user_id', 'article_url']].rename(columns={'reply_user_id': 'user_id'})
    interactions = pd.concat([df1, df2]).dropna().drop_duplicates()
    
    n_interactions = len(interactions)
    n_users = interactions['user_id'].nunique()
    n_items = interactions['article_url'].nunique()
    
    density = n_interactions / (n_users * n_items) if n_users * n_items > 0 else 0
    
    print("\n" + "="*40)
    print("DATASET SPARSITY METRICS")
    print("="*40)
    print(f"Num Users:              {n_users}")
    print(f"Num Items:              {n_items}")
    print(f"Num Interactions:       {n_interactions}")
    print(f"Sparsity (Density):     {density:.4%} ({density:.6f})")
    print(f"Avg Interactions/User:  {n_interactions/n_users:.2f}")
    print(f"Avg Interactions/Item:  {n_interactions/n_items:.2f}")
    print("="*40)
    
    # Distribution check
    user_counts = interactions['user_id'].value_counts()
    print(f"\nUser Interaction Stats:")
    print(f"   - Min:    {user_counts.min()}")
    print(f"   - Max:    {user_counts.max()}")
    print(f"   - Median: {user_counts.median()}")
    print(f"   - < 3 interactions: {(user_counts < 3).sum()} users ({(user_counts < 3).mean():.1%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('replies_path', nargs='?', default='data/raw/replies.csv')
    args = parser.parse_args()
    check_sparsity(args.replies_path)
