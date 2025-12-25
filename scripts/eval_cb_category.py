#!/usr/bin/env python3
"""
Evaluate Content-Based Model with Category Match
=================================================
Instead of exact article match, check if recommended articles 
match the CATEGORY of test articles.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data():
    """Load articles, metadata, and create user-category mappings."""
    print("Loading data...")
    
    # Load raw data
    articles = pd.read_csv('data/raw/articles.csv')
    metadata = pd.read_csv('data/raw/metadata.csv')
    replies = pd.read_csv('data/raw/replies.csv')
    
    # Clean replies
    replies = replies[replies['parent_user_id'] != 'NO_COMMENT'].copy()
    replies['user_id'] = replies['reply_user_id'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) else None
    )
    replies = replies[replies['user_id'].notna()].copy()
    
    # Merge with metadata
    merged = replies.merge(metadata, on='article_url', how='left')
    merged = merged[merged['category'].notna()]
    
    print(f"  Valid interactions with category: {len(merged)}")
    
    # Create article -> category mapping
    article_to_cat = dict(zip(metadata['article_url'], metadata['category']))
    
    # Create URL to index mapping
    unique_urls = merged['article_url'].unique()
    url_to_idx = {url: i for i, url in enumerate(unique_urls)}
    idx_to_url = {i: url for url, i in url_to_idx.items()}
    idx_to_cat = {i: article_to_cat.get(url, 'Unknown') for i, url in idx_to_url.items()}
    
    # Filter users with min interactions
    user_counts = merged['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 3].index
    merged = merged[merged['user_id'].isin(valid_users)]
    
    # Create user -> articles mapping
    user_articles = merged.groupby('user_id')['article_url'].apply(list).to_dict()
    
    # Train/test split per user
    train_dict = {}
    test_dict = {}
    
    np.random.seed(42)
    for user, urls in user_articles.items():
        unique_urls = list(set(urls))
        if len(unique_urls) >= 2:
            np.random.shuffle(unique_urls)
            split = max(1, int(len(unique_urls) * 0.8))
            train_urls = unique_urls[:split]
            test_urls = unique_urls[split:]
            
            if test_urls:
                train_dict[user] = [url_to_idx[u] for u in train_urls if u in url_to_idx]
                test_dict[user] = [url_to_idx[u] for u in test_urls if u in url_to_idx]
    
    print(f"  Users with train+test: {len(test_dict)}")
    print(f"  Total articles: {len(url_to_idx)}")
    
    return {
        'train_dict': train_dict,
        'test_dict': test_dict,
        'idx_to_url': idx_to_url,
        'idx_to_cat': idx_to_cat,
        'n_items': len(url_to_idx),
        'articles': articles,
        'metadata': metadata
    }


def get_article_texts(data, articles):
    """Get texts for all articles."""
    url_to_text = {}
    for _, row in articles.iterrows():
        text = f"{row.get('title', '')} {row.get('short_description', '')}"
        url_to_text[row['url']] = text.strip()
    
    texts = []
    for idx in range(data['n_items']):
        url = data['idx_to_url'].get(idx, '')
        texts.append(url_to_text.get(url, ''))
    
    return texts


def compute_embeddings(texts, device='cpu'):
    """Compute PhoBERT embeddings for articles."""
    print("\nComputing PhoBERT embeddings...")
    
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModel.from_pretrained("vinai/phobert-base")
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        # Truncate long texts
        batch = [t[:256] if t else "empty" for t in batch]
        
        inputs = tokenizer(batch, padding=True, truncation=True, 
                          max_length=128, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def evaluate_category_match(embeddings, data, k_list=[5, 10, 20]):
    """
    Evaluate CB model with category match metric.
    
    For each user:
    1. Build user preference from train articles
    2. Recommend top-k articles
    3. Check if recommended categories match test article categories
    """
    print("\nEvaluating with Category Match...")
    
    train_dict = data['train_dict']
    test_dict = data['test_dict']
    idx_to_cat = data['idx_to_cat']
    
    # Normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    max_k = max(k_list)
    results = {f'cat_precision@{k}': [] for k in k_list}
    results.update({f'exact_recall@{k}': [] for k in k_list})
    results['cat_map'] = []  # Category-based mAP
    results['exact_map'] = []  # Exact match mAP
    
    for user, test_items in tqdm(test_dict.items(), desc="Evaluating"):
        train_items = train_dict.get(user, [])
        if not train_items or not test_items:
            continue
        
        # Build user preference (mean of train embeddings)
        train_emb = embeddings[train_items].mean(dim=0, keepdim=True)
        
        # Compute similarity to all items
        scores = torch.mm(train_emb, embeddings.t()).squeeze()
        
        # Mask train items
        for item in train_items:
            scores[item] = -float('inf')
        
        # Get test categories
        test_cats = set(idx_to_cat.get(i, 'Unknown') for i in test_items)
        test_items_set = set(test_items)
        
        # Get top-k recommendations
        _, topk = torch.topk(scores, min(max_k, len(scores)))
        topk = topk.numpy().tolist()
        
        # Calculate Average Precision for category match
        # For category AP: count each unique category match only once
        cat_hits = 0
        cat_precision_sum = 0
        found_cats = set()
        exact_hits = 0
        exact_precision_sum = 0
        
        for rank, item in enumerate(topk, 1):
            item_cat = idx_to_cat.get(item, 'Unknown')
            
            # Category AP (count each category only once)
            if item_cat in test_cats and item_cat not in found_cats:
                found_cats.add(item_cat)
                cat_hits += 1
                cat_precision_sum += cat_hits / rank
            
            # Exact AP
            if item in test_items_set:
                exact_hits += 1
                exact_precision_sum += exact_hits / rank
        
        # Compute AP (Average Precision)
        results['cat_map'].append(cat_precision_sum / len(test_cats) if test_cats else 0)
        results['exact_map'].append(exact_precision_sum / len(test_items) if test_items else 0)
        
        # Evaluate at different k for precision
        for k in k_list:
            topk_k = topk[:k]
            rec_cats = [idx_to_cat.get(i, 'Unknown') for i in topk_k]
            cat_hits_k = sum(1 for c in rec_cats if c in test_cats)
            
            results[f'cat_precision@{k}'].append(cat_hits_k / k)
            
            exact_hits_k = len(set(topk_k) & test_items_set)
            results[f'exact_recall@{k}'].append(exact_hits_k / len(test_items))
    
    # Average
    return {k: np.mean(v) for k, v in results.items()}


def main():
    print("=" * 60)
    print("CATEGORY-BASED CONTENT RECOMMENDATION EVALUATION")
    print("=" * 60)
    
    # Load data
    data = load_data()
    
    # Get article texts
    texts = get_article_texts(data, data['articles'])
    
    # Compute embeddings (or load cached)
    cache_path = Path('checkpoints/cb_eval_embeddings.pt')
    if cache_path.exists():
        print(f"\nLoading cached embeddings from {cache_path}...")
        embeddings = torch.load(cache_path)
    else:
        embeddings = compute_embeddings(texts)
        cache_path.parent.mkdir(exist_ok=True)
        torch.save(embeddings, cache_path)
        print(f"Saved embeddings to {cache_path}")
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Evaluate
    metrics = evaluate_category_match(embeddings, data)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS: Category-Based Evaluation")
    print("=" * 60)
    
    print("\n📊 Mean Average Precision (mAP):")
    print(f"  Category mAP:  {metrics['cat_map']:.2%}")
    print(f"  Exact mAP:     {metrics['exact_map']:.2%}")
    
    print("\n📊 Category Precision (% of recommendations in correct category):")
    for k in [5, 10, 20]:
        print(f"  Cat_Precision@{k}: {metrics[f'cat_precision@{k}']:.2%}")
    
    print("\n📊 Exact Recall (for comparison):")
    for k in [5, 10, 20]:
        print(f"  Exact_Recall@{k}:  {metrics[f'exact_recall@{k}']:.2%}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("Category mAP: Quality of category-based ranking (higher = better ranking)")
    print("Cat_Precision: % of recommendations in correct category")
    print("Exact_Recall: % of exact test articles found (stricter metric)")


if __name__ == "__main__":
    main()
