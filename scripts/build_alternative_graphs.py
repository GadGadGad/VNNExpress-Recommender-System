#!/usr/bin/env python3
"""
Build Alternative Graph Types for Comparison
============================================
Creates three graph variants for CF model comparison:
1. User-Category bipartite
2. User-Article with reaction weights
3. Article-Article category-aware
"""

import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from pathlib import Path
from collections import defaultdict
import argparse


def clean_id(val):
    """Clean user ID values."""
    try:
        if pd.isna(val) or val == '' or str(val).lower() == 'nan':
            return None
        return str(int(float(val)))
    except:
        return str(val)


def build_user_category_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Priority 1: User-Category bipartite graph
    Users connected to categories based on their article interactions.
    """
    print("\n" + "=" * 60)
    print("Building User-Category Graph")
    print("=" * 60)
    
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)
    
    # Clean data
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Merge to get category for each interaction
    merged = replies.merge(
        articles[['url', 'source_category']], 
        left_on='article_url', 
        right_on='url', 
        how='inner'
    )
    
    # Filter users with min interactions
    user_counts = merged['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    merged = merged[merged['user_id'].isin(valid_users)]
    
    # Create mappings
    categories = merged['source_category'].unique()
    users = merged['user_id'].unique()
    
    cat_map = {c: i for i, c in enumerate(categories)}
    user_map = {u: i for i, u in enumerate(users)}
    
    n_users = len(user_map)
    n_cats = len(cat_map)
    
    print(f"  Users: {n_users}")
    print(f"  Categories: {n_cats}")
    
    # Build edge weights (user -> category count)
    user_cat_counts = merged.groupby(['user_id', 'source_category']).size().reset_index(name='count')
    
    src = [user_map[u] for u in user_cat_counts['user_id']]
    dst = [cat_map[c] for c in user_cat_counts['source_category']]
    weights = user_cat_counts['count'].values
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    # Create features
    user_features = torch.randn(n_users, 64)
    cat_features = torch.randn(n_cats, 64)
    
    # Also build category->article mapping for inference
    article_cat = articles[['url', 'source_category']].drop_duplicates()
    article_cat = article_cat[article_cat['url'].isin(merged['article_url'])]
    article_map = {url: i for i, url in enumerate(article_cat['url'].unique())}
    n_articles = len(article_map)
    
    cat_article_edges = []
    for _, row in article_cat.iterrows():
        if row['url'] in article_map and row['source_category'] in cat_map:
            cat_article_edges.append([cat_map[row['source_category']], article_map[row['url']]])
    
    # Save data
    data = {
        'user_features': user_features,
        'category_features': cat_features,
        'user_category_edge_index': edge_index,
        'user_category_edge_weight': edge_weight,
        'n_users': n_users,
        'n_categories': n_cats,
        'n_articles': n_articles,
        'user_map': user_map,
        'cat_map': cat_map,
        'article_map': article_map,
        'cat_article_edges': torch.tensor(cat_article_edges, dtype=torch.long).T if cat_article_edges else None,
        'graph_type': 'user_category'
    }
    
    # For CF training, create user-article pairs through category
    interactions = []
    for _, row in merged.iterrows():
        u_idx = user_map[row['user_id']]
        if row['article_url'] in article_map:
            a_idx = article_map[row['article_url']]
            interactions.append((u_idx, a_idx))
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(interactions)
    split = int(len(interactions) * 0.8)
    
    train_pairs = interactions[:split]
    test_pairs = interactions[split:]
    
    train_dict = defaultdict(set)
    for u, i in train_pairs: train_dict[u].add(i)
    test_dict = defaultdict(set)
    for u, i in test_pairs: test_dict[u].add(i)

    data.update({
        'train_pairs': train_pairs,
        'train_dict': dict(train_dict),
        'test_dict': dict(test_dict)
    })
    
    output_path = Path(output_dir) / 'user_category_graph.pt'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Saved to: {output_path}")
    
    return data


def build_reaction_weighted_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Priority 2: User-Article with reaction weights
    Standard bipartite but edges weighted by reaction count.
    """
    print("\n" + "=" * 60)
    print("Building Reaction-Weighted User-Article Graph")
    print("=" * 60)
    
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)
    
    # Clean data
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Get reaction weights (combine parent and reply reactions)
    replies['reaction_weight'] = (
        pd.to_numeric(replies['parent_reactions'], errors='coerce').fillna(0) +
        pd.to_numeric(replies['reply_reactions'], errors='coerce').fillna(0) + 1  # +1 base weight
    )
    
    # Filter
    user_counts = replies['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    article_counts = replies['article_url'].value_counts()
    valid_articles = article_counts[article_counts >= min_interactions].index
    
    replies = replies[
        (replies['user_id'].isin(valid_users)) &
        (replies['article_url'].isin(valid_articles))
    ].copy()
    
    # Aggregate by user-article pair (sum reactions)
    aggregated = replies.groupby(['user_id', 'article_url']).agg({
        'reaction_weight': 'sum'
    }).reset_index()
    
    # Create mappings
    users = aggregated['user_id'].unique()
    article_urls = aggregated['article_url'].unique()
    
    user_map = {u: i for i, u in enumerate(users)}
    article_map = {a: i for i, a in enumerate(article_urls)}
    
    n_users = len(user_map)
    n_articles = len(article_map)
    
    print(f"  Users: {n_users}")
    print(f"  Articles: {n_articles}")
    
    # Build edges
    src = [user_map[u] for u in aggregated['user_id']]
    dst = [article_map[a] for a in aggregated['article_url']]
    weights = aggregated['reaction_weight'].values
    
    # Normalize weights (log transform)
    weights = np.log1p(weights)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    # Features
    user_features = torch.randn(n_users, 64)
    article_features = torch.randn(n_articles, 64)
    
    # Shuffle and split
    interactions = list(zip(src, dst))
    np.random.seed(42)
    np.random.shuffle(interactions)
    split = int(len(interactions) * 0.8)
    
    train_pairs = interactions[:split]
    test_pairs = interactions[split:]
    
    train_dict = defaultdict(set)
    for u, i in train_pairs: train_dict[u].add(i)
    
    test_dict = defaultdict(set)
    for u, i in test_pairs: test_dict[u].add(i)
    
    data = {
        'user_features': user_features,
        'article_features': article_features,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'n_users': n_users,
        'n_items': n_articles,
        'user_map': user_map,
        'article_map': article_map,
        'train_pairs': train_pairs,
        'train_dict': dict(train_dict),
        'test_dict': dict(test_dict),
        'graph_type': 'reaction_weighted'
    }
    
    output_path = Path(output_dir) / 'reaction_weighted_graph.pt'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Avg weight: {weights.mean():.2f}")
    print(f"  Max weight: {weights.max():.2f}")
    print(f"  Saved to: {output_path}")
    
    return data


def build_article_article_category_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Priority 3: Article-Article category-aware graph
    Articles in same category connected, plus shared-user connections.
    """
    print("\n" + "=" * 60)
    print("Building Article-Article Category-Aware Graph")
    print("=" * 60)
    
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)
    
    # Clean data
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Filter
    article_counts = replies['article_url'].value_counts()
    valid_articles = article_counts[article_counts >= min_interactions].index
    
    user_counts = replies['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    replies = replies[
        (replies['article_url'].isin(valid_articles)) &
        (replies['user_id'].isin(valid_users))
    ].copy()
    
    # Get articles with category
    valid_article_data = articles[articles['url'].isin(valid_articles)][['url', 'source_category']].drop_duplicates()
    
    article_map = {url: i for i, url in enumerate(valid_article_data['url'])}
    n_articles = len(article_map)
    
    print(f"  Articles: {n_articles}")
    
    # Build category-based edges (same category = connected)
    category_edges = []
    category_weights = []
    
    articles_by_cat = valid_article_data.groupby('source_category')['url'].apply(list).to_dict()
    
    for cat, art_list in articles_by_cat.items():
        art_indices = [article_map[a] for a in art_list if a in article_map]
        # Connect within category (sample to avoid O(n^2))
        if len(art_indices) > 1:
            for i in range(len(art_indices)):
                # Connect to next 5 articles in same category
                for j in range(i+1, min(i+6, len(art_indices))):
                    category_edges.append([art_indices[i], art_indices[j]])
                    category_edges.append([art_indices[j], art_indices[i]])
                    category_weights.extend([1.0, 1.0])
    
    # Build shared-user edges
    user_articles = replies.groupby('user_id')['article_url'].apply(set).to_dict()
    
    shared_user_edges = []
    shared_user_weights = []
    
    # For each user, connect their articles
    for user_id, art_set in user_articles.items():
        art_list = [article_map[a] for a in art_set if a in article_map]
        if len(art_list) > 1:
            for i in range(len(art_list)):
                for j in range(i+1, len(art_list)):
                    shared_user_edges.append([art_list[i], art_list[j]])
                    shared_user_edges.append([art_list[j], art_list[i]])
                    shared_user_weights.extend([2.0, 2.0])  # Higher weight for user co-occurrence
    
    # Combine edges
    all_edges = category_edges + shared_user_edges
    all_weights = category_weights + shared_user_weights
    
    if not all_edges:
        print("  No edges found!")
        return None
    
    edge_index = torch.tensor(all_edges, dtype=torch.long).T
    edge_weight = torch.tensor(all_weights, dtype=torch.float32)
    
    # Article features
    article_features = torch.randn(n_articles, 64)
    
    # For training, we need user-article pairs
    user_map = {u: i for i, u in enumerate(valid_users)}
    train_pairs = []
    train_dict = defaultdict(set)
    
    for _, row in replies.iterrows():
        if row['user_id'] in user_map and row['article_url'] in article_map:
            u_idx = user_map[row['user_id']]
            a_idx = article_map[row['article_url']]
            train_pairs.append((u_idx, a_idx))
            train_dict[u_idx].add(a_idx)
    
    data = {
        'article_features': article_features,
        'article_edge_index': edge_index,
        'article_edge_weight': edge_weight,
        'n_users': len(user_map),
        'n_items': n_articles,
        'user_map': user_map,
        'article_map': article_map,
        'train_pairs': train_pairs,
        'train_dict': dict(train_dict),
        'graph_type': 'article_article_category'
    }
    
    output_path = Path(output_dir) / 'article_article_category_graph.pt'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    
    print(f"  Category edges: {len(category_edges)}")
    print(f"  Shared-user edges: {len(shared_user_edges)}")
    print(f"  Total edges: {edge_index.shape[1]}")
    print(f"  Saved to: {output_path}")
    
    return data


def build_category_category_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Priority 4: Category-Category Graph (Co-occurrence)
    Categories are connected if users frequently read both.
    """
    print("\n" + "=" * 60)
    print("Building Category-Category Co-occurrence Graph")
    print("=" * 60)
    
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Merge to get categories (use suffixes to avoid column conflict)
    merged = replies.merge(articles[['url', 'source_category']], left_on='article_url', right_on='url', suffixes=('', '_article'))
    
    # Use the category from articles (source_category or source_category_article depending on conflict)
    cat_col = 'source_category_article' if 'source_category_article' in merged.columns else 'source_category'
    
    # Get user -> categories
    user_cats = merged.groupby('user_id')[cat_col].apply(set).to_dict()
    
    categories = articles['source_category'].unique()
    cat_map = {c: i for i, c in enumerate(categories)}
    n_cats = len(cat_map)
    
    edges = defaultdict(int)
    
    # Count co-occurrences
    print(f"  Analysing {len(user_cats)} users for category co-occurrence...")
    for uid, cats in user_cats.items():
        cat_list = list(cats)
        if len(cat_list) > 1:
            for i in range(len(cat_list)):
                for j in range(i+1, len(cat_list)):
                    c1, c2 = cat_list[i], cat_list[j]
                    k = tuple(sorted((c1, c2)))
                    edges[k] += 1
    
    # Thresholding?
    src, dst, w = [], [], []
    for (c1, c2), count in edges.items():
        if count >= min_interactions:
            idx1, idx2 = cat_map[c1], cat_map[c2]
            src.extend([idx1, idx2])
            dst.extend([idx2, idx1])
            w.extend([count, count])
            
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    
    # Features (Identity)
    x = torch.eye(n_cats)
    
    data = {
        'x': x,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'cat_map': cat_map,
        'n_nodes': n_cats,
        'graph_type': 'category_category'
    }
    
    output_path = Path(output_dir) / 'category_category_graph.pt'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"  Nodes: {n_cats}, Edges: {edge_index.shape[1]}")
    print(f"  Saved to: {output_path}")
    return data


def build_user_author_graph(articles_path, replies_path, output_dir, min_interactions=2):
    """
    Priority 5: User-Author Bipartite Graph
    """
    print("\n" + "=" * 60)
    print("Building User-Author Graph")
    print("=" * 60)
    
    articles = pd.read_csv(articles_path)
    # Clean authors
    articles['author'] = articles['author'].fillna('Unknown').apply(lambda x: x.strip())
    
    replies = pd.read_csv(replies_path)
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    replies = replies[replies['user_id'].notna()].copy()
    
    # Merge
    merged = replies.merge(articles[['url', 'author']], left_on='article_url', right_on='url')
    
    # Filter
    author_counts = merged['author'].value_counts()
    valid_authors = author_counts[author_counts >= min_interactions].index
    merged = merged[merged['author'].isin(valid_authors)]
    
    users = merged['user_id'].unique()
    authors = merged['author'].unique()
    
    user_map = {u: i for i, u in enumerate(users)}
    author_map = {a: i for i, a in enumerate(authors)}
    
    n_users = len(user_map)
    n_authors = len(author_map)
    
    # Edges
    pair_counts = merged.groupby(['user_id', 'author']).size()
    
    src, dst, w = [], [], []
    for (uid, auth), count in pair_counts.items():
        src.append(user_map[uid])
        dst.append(author_map[auth])
        w.append(count)
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    
    # Create standard train/test splits for this graph too!
    train_pairs, test_pairs = [], []
    interactions = list(zip(src, dst))
    np.random.seed(42)
    np.random.shuffle(interactions)
    split = int(len(interactions) * 0.8)
    
    train_pairs = interactions[:split]
    test_pairs = interactions[split:]
    
    train_dict = defaultdict(set)
    for u, i in train_pairs: train_dict[u].add(i)
    test_dict = defaultdict(set)
    for u, i in test_pairs: test_dict[u].add(i)
    
    data = {
        'n_users': n_users,
        'n_items': n_authors, # Authors as items
        'user_features': torch.randn(n_users, 64),
        'author_features': torch.randn(n_authors, 64),
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'user_map': user_map,
        'author_map': author_map,
        'train_pairs': train_pairs, 
        'train_dict': dict(train_dict),
        'test_dict': dict(test_dict),
        'graph_type': 'user_author'
    }
    
    output_path = Path(output_dir) / 'user_author_graph.pt'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    
    print(f"  Users: {n_users}, Authors: {n_authors}")
    print(f"  Edges: {len(src)}")
    print(f"  Saved to: {output_path}")
    return data

def main():
    parser = argparse.ArgumentParser(description='Build alternative graph types')
    parser.add_argument('--articles', default='data/raw/articles.csv')
    parser.add_argument('--replies', default='data/raw/replies.csv')
    parser.add_argument('--output', default='data/processed_graphs')
    parser.add_argument('--min-interactions', type=int, default=2)
    parser.add_argument('--graph-type', choices=['all', 'user-category', 'reaction-weighted', 'article-article', 'category-category', 'user-author'],
                        default='all')
    args = parser.parse_args()
    
    if args.graph_type in ['all', 'user-category']:
        build_user_category_graph(args.articles, args.replies, args.output, args.min_interactions)
    
    if args.graph_type in ['all', 'reaction-weighted']:
        build_reaction_weighted_graph(args.articles, args.replies, args.output, args.min_interactions)
    
    if args.graph_type in ['all', 'article-article']:
        build_article_article_category_graph(args.articles, args.replies, args.output, args.min_interactions)

    if args.graph_type in ['all', 'category-category']:
        build_category_category_graph(args.articles, args.replies, args.output, args.min_interactions)

    if args.graph_type in ['all', 'user-author']:
        build_user_author_graph(args.articles, args.replies, args.output, args.min_interactions)

    
    print("\n" + "=" * 60)
    print("All graphs built successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
