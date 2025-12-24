#!/usr/bin/env python3
"""
Extract Dense Core Subset
=========================
Analyzes the dataset and extracts a dense, high-quality subset of users and articles
that are likely to produce good recommendation performance.

Approach:
1. Start with users who have >= K interactions
2. Keep only articles that these users interacted with
3. Optionally iterate to ensure all remaining users still meet threshold
4. Export the filtered subset as new CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import Counter

def analyze_density(replies_df, name="Dataset"):
    """Analyze and print density statistics."""
    user_counts = replies_df['user_id'].value_counts()
    article_counts = replies_df['article_url'].value_counts()
    
    n_users = len(user_counts)
    n_articles = len(article_counts)
    n_interactions = len(replies_df)
    density = n_interactions / (n_users * n_articles) * 100 if (n_users * n_articles) > 0 else 0
    
    print(f"\n{name}:")
    print(f"  Users: {n_users:,}")
    print(f"  Articles: {n_articles:,}")
    print(f"  Interactions: {n_interactions:,}")
    print(f"  Density: {density:.4f}%")
    print(f"  Avg interactions/user: {user_counts.mean():.2f}")
    print(f"  Median interactions/user: {user_counts.median():.0f}")
    print(f"  Avg interactions/article: {article_counts.mean():.2f}")
    
    return {
        'users': n_users,
        'articles': n_articles,
        'interactions': n_interactions,
        'density': density,
        'avg_user_int': user_counts.mean(),
        'median_user_int': user_counts.median()
    }

def extract_dense_core(articles_df, replies_df, users_df, 
                       min_user_int=5, min_article_int=3, 
                       target_users=None, max_iterations=10):
    """
    Extract a dense subset using iterative k-core-like filtering.
    
    Args:
        min_user_int: Minimum interactions required per user
        min_article_int: Minimum interactions required per article
        target_users: Optional target number of users (will adjust thresholds)
        max_iterations: Maximum filtering iterations
    """
    
    # Clean user IDs
    def clean_id(val):
        try:
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                return None
            return str(int(float(val)))
        except:
            return str(val)
    
    replies = replies_df.copy()
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    print("=" * 60)
    print("DENSE CORE EXTRACTION")
    print("=" * 60)
    
    analyze_density(replies, "Original Dataset")
    
    # Iterative filtering
    for iteration in range(max_iterations):
        prev_len = len(replies)
        
        # Filter users
        user_counts = replies['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_int].index
        replies = replies[replies['user_id'].isin(valid_users)]
        
        # Filter articles
        article_counts = replies['article_url'].value_counts()
        valid_articles = article_counts[article_counts >= min_article_int].index
        replies = replies[replies['article_url'].isin(valid_articles)]
        
        if len(replies) == prev_len:
            print(f"\n[Converged after {iteration + 1} iterations]")
            break
        
        if len(replies) == 0:
            print(f"\n[WARNING] All data filtered out at iteration {iteration + 1}!")
            print("Try lowering min_user_int or min_article_int")
            return None, None, None
    
    stats = analyze_density(replies, "Dense Core")
    
    # Filter related dataframes
    valid_user_set = set(replies['user_id'].unique())
    valid_article_set = set(replies['article_url'].unique())
    
    # Filter articles
    dense_articles = articles_df[articles_df['url'].isin(valid_article_set)].copy()
    
    # Filter users (convert IDs for matching)
    if users_df is not None and 'user_id' in users_df.columns:
        users_df_copy = users_df.copy()
        users_df_copy['user_id'] = users_df_copy['user_id'].apply(clean_id)
        dense_users = users_df_copy[users_df_copy['user_id'].isin(valid_user_set)].copy()
    else:
        dense_users = None
    
    # Create dense replies (with original columns)
    dense_replies = replies_df[
        replies_df['article_url'].isin(valid_article_set)
    ].copy()
    
    # Re-clean and filter user IDs in the output
    dense_replies['_temp_user'] = dense_replies['reply_user_id'].apply(clean_id)
    dense_replies = dense_replies[dense_replies['_temp_user'].isin(valid_user_set)]
    dense_replies = dense_replies.drop('_temp_user', axis=1)
    
    return dense_articles, dense_replies, dense_users, stats

def find_optimal_threshold(replies_df, target_density=2.0, target_min_users=100):
    """
    Find optimal filtering thresholds to achieve target density while keeping enough users.
    """
    def clean_id(val):
        try:
            if pd.isna(val) or val == '' or str(val).lower() == 'nan':
                return None
            return str(int(float(val)))
        except:
            return str(val)
    
    replies = replies_df.copy()
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 60)
    print(f"Target: Density >= {target_density}%, Users >= {target_min_users}")
    
    best_config = None
    best_score = 0
    
    results = []
    
    for min_user in range(2, 15):
        for min_article in range(2, 10):
            # Simulate filtering
            temp = replies.copy()
            
            for _ in range(10):
                prev_len = len(temp)
                user_counts = temp['user_id'].value_counts()
                valid_users = user_counts[user_counts >= min_user].index
                temp = temp[temp['user_id'].isin(valid_users)]
                
                article_counts = temp['article_url'].value_counts()
                valid_articles = article_counts[article_counts >= min_article].index
                temp = temp[temp['article_url'].isin(valid_articles)]
                
                if len(temp) == prev_len or len(temp) == 0:
                    break
            
            if len(temp) == 0:
                continue
            
            n_users = temp['user_id'].nunique()
            n_articles = temp['article_url'].nunique()
            n_int = len(temp)
            density = n_int / (n_users * n_articles) * 100 if (n_users * n_articles) > 0 else 0
            avg_user_int = n_int / n_users if n_users > 0 else 0
            
            results.append({
                'min_user': min_user,
                'min_article': min_article,
                'users': n_users,
                'articles': n_articles,
                'interactions': n_int,
                'density': density,
                'avg_user_int': avg_user_int
            })
            
            # Score: prioritize density while ensuring enough users
            if n_users >= target_min_users:
                score = density * (n_users / target_min_users) * avg_user_int
                if score > best_score:
                    best_score = score
                    best_config = (min_user, min_article, n_users, density, avg_user_int)
    
    # Print top configurations
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('density', ascending=False)
    
    print("\nTop 10 Configurations by Density:")
    print("-" * 80)
    print(f"{'MinUser':>8} {'MinArt':>8} {'Users':>8} {'Articles':>10} {'Interact':>10} {'Density%':>10} {'Avg/User':>10}")
    print("-" * 80)
    
    for _, row in results_df.head(10).iterrows():
        print(f"{row['min_user']:>8} {row['min_article']:>8} {row['users']:>8} {row['articles']:>10} {row['interactions']:>10} {row['density']:>10.4f} {row['avg_user_int']:>10.2f}")
    
    if best_config:
        print(f"\n[RECOMMENDED] min_user={best_config[0]}, min_article={best_config[1]}")
        print(f"   -> {best_config[2]} users, {best_config[3]:.4f}% density, {best_config[4]:.2f} avg interactions/user")
    
    return results_df, best_config

def main():
    parser = argparse.ArgumentParser(description='Extract dense core subset from dataset')
    parser.add_argument('--input', '-i', default='data/raw', help='Input data directory')
    parser.add_argument('--output', '-o', default='data/dense_core', help='Output directory for dense subset')
    parser.add_argument('--min-user-int', type=int, default=5, help='Minimum interactions per user')
    parser.add_argument('--min-article-int', type=int, default=3, help='Minimum interactions per article')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not save')
    parser.add_argument('--find-optimal', action='store_true', help='Find optimal thresholds')
    parser.add_argument('--target-users', type=int, default=100, help='Target minimum users for optimization')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    articles = pd.read_csv(input_path / 'articles.csv')
    replies = pd.read_csv(input_path / 'replies.csv')
    
    users = None
    if (input_path / 'user_profiles.csv').exists():
        users = pd.read_csv(input_path / 'user_profiles.csv')
    
    if args.find_optimal:
        results, best = find_optimal_threshold(replies, target_min_users=args.target_users)
        if best:
            args.min_user_int = best[0]
            args.min_article_int = best[1]
            print(f"\nUsing recommended thresholds: min_user={args.min_user_int}, min_article={args.min_article_int}")
    
    # Extract dense core
    dense_articles, dense_replies, dense_users, stats = extract_dense_core(
        articles, replies, users,
        min_user_int=args.min_user_int,
        min_article_int=args.min_article_int
    )
    
    if dense_articles is None:
        print("\n[FAILED] Could not extract dense core. Try lower thresholds.")
        return
    
    if args.analyze_only:
        print("\n[Analyze only mode - not saving]")
        return
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dense_articles.to_csv(output_path / 'articles.csv', index=False)
    dense_replies.to_csv(output_path / 'replies.csv', index=False)
    if dense_users is not None:
        dense_users.to_csv(output_path / 'user_profiles.csv', index=False)
    
    print(f"\n[OK] Dense core saved to: {output_path}")
    print(f"     You can now use --data-source {output_path} in the interactive pipeline")

if __name__ == "__main__":
    main()
