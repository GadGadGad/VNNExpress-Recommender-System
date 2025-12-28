#!/usr/bin/env python3
"""
Train & Evaluate Content-Based Recommender Models
==================================================

Usage:
    # TF-IDF (fastest, CPU)
    python scripts/train_cb_models.py --encoder tfidf
    
    # Precomputed embeddings (fast)
    python scripts/train_cb_models.py --encoder precomputed --embedding-path checkpoints/bge-m3_article_embeddings.pt
    
    # PhoBERT (slow, requires GPU)
    python scripts/train_cb_models.py --encoder phobert
    
    # SentenceTransformers (slow, requires GPU)
    python scripts/train_cb_models.py --encoder bge-m3
    python scripts/train_cb_models.py --encoder vndoc
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.content_based import ContentBasedRecommender, TFIDFRecommender


def load_data(data_path: str, articles_path: str = None):
    """
    Load interaction data and article texts.
    
    Returns:
        train_dict: {user_id: [item_indices]}
        test_dict: {user_id: [item_indices]}
        article_texts: List of article text strings
        n_users, n_items: counts
    """
    data_path = Path(data_path)
    
    # Try loading from processed graph file (has train/test splits)
    graph_file = data_path / "user_article_graph.pt"
    if not graph_file.exists():
        # Fallback: try any .pt file
        graph_files = list(data_path.glob("*.pt"))
        graph_file = graph_files[0] if graph_files else None
    
    train_dict, test_dict = defaultdict(list), defaultdict(list)
    n_users, n_items = 0, 0
    idx2item = {}
    
    if graph_file and graph_file.exists():
        # Load from .pt file
        data = torch.load(graph_file, weights_only=False)
        
        if 'n_users' in data:
            n_users = data['n_users']
        if 'n_items' in data:
            n_items = data['n_items']
        if 'idx2item' in data:
            idx2item = data['idx2item']
        
        # Handle different split formats
        if 'train_dict' in data and 'test_dict' in data:
            train_dict = data['train_dict']
            test_dict = data['test_dict']
        elif 'splits' in data:
            splits = data['splits']
            
            # Format: splits['train'] = {'pos_users': tensor, 'pos_articles': tensor}
            if 'train' in splits and isinstance(splits['train'], dict):
                train_split = splits['train']
                if 'pos_users' in train_split and 'pos_articles' in train_split:
                    users = train_split['pos_users'].tolist()
                    items = train_split['pos_articles'].tolist()
                    for u, i in zip(users, items):
                        train_dict[u].append(i)
                        
            if 'test' in splits and isinstance(splits['test'], dict):
                test_split = splits['test']
                if 'pos_users' in test_split and 'pos_articles' in test_split:
                    users = test_split['pos_users'].tolist()
                    items = test_split['pos_articles'].tolist()
                    for u, i in zip(users, items):
                        test_dict[u].append(i)
    
    # Convert defaultdict to regular dict
    train_dict = dict(train_dict)
    test_dict = dict(test_dict)
            
    print(f"Loaded: {n_users} users, {n_items} items")
    print(f"Train: {len(train_dict)} users, Test: {len(test_dict)} users")
    
    # Load article texts
    article_texts = []
    if articles_path and Path(articles_path).exists():
        articles_df = pd.read_csv(articles_path)
        article_texts = (
            articles_df['title'].fillna('') + ' ' + 
            articles_df['short_description'].fillna('')
        ).tolist()
        print(f"Loaded {len(article_texts)} article texts")
    
    return train_dict, test_dict, article_texts, n_users, n_items, idx2item


def evaluate(model, test_dict, train_dict, k_list=[1, 5, 10, 50]):
    """Evaluate Content-Based model"""
    
    results = {f'{metric}@{k}': [] for metric in ['recall', 'ndcg', 'precision', 'f1', 'hitrate', 'map'] for k in k_list}
    results['mrr'] = []
    
    max_k = max(k_list)
    evaluated_users = 0
    skipped_users = 0
    
    for user, test_items in tqdm(test_dict.items(), desc="Evaluating", ncols=80):
        test_items = set(test_items) if isinstance(test_items, list) else test_items
        
        if not test_items:
            skipped_users += 1
            continue
            
        history = list(train_dict.get(user, []))
        if not history:
            skipped_users += 1
            continue
        
        # Get recommendations
        try:
            recs, scores = model.recommend(history, k=max_k, exclude_read=True)
        except Exception as e:
            skipped_users += 1
            continue
        
        evaluated_users += 1
        
        # MRR
        mrr = 0.0
        for i, item in enumerate(recs):
            if item in test_items:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'].append(mrr)
        
        # Metrics at each k
        for k in k_list:
            topk = set(recs[:k])
            hits = len(topk & test_items)
            
            prec = hits / k
            rec = hits / len(test_items)
            
            results[f'recall@{k}'].append(rec)
            results[f'precision@{k}'].append(prec)
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
            
            # F1
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            results[f'f1@{k}'].append(f1)
            
            # NDCG
            dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs[:k]) if item in test_items)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
            results[f'ndcg@{k}'].append(dcg / idcg if idcg > 0 else 0)
            
            # MAP
            ap = 0.0
            n_hits = 0
            for i, item in enumerate(recs[:k]):
                if item in test_items:
                    n_hits += 1
                    ap += n_hits / (i + 1)
            ap = ap / min(k, len(test_items)) if len(test_items) > 0 else 0
            results[f'map@{k}'].append(ap)
    
    print(f"\nEvaluated: {evaluated_users} users, Skipped: {skipped_users} users")
    
    return {k: np.mean(v) for k, v in results.items() if len(v) > 0}


def print_results(metrics):
    """Print results in table format"""
    print(f"\nBest Metrics (Final Evaluation):")
    metrics_to_print = ['recall', 'ndcg', 'precision', 'f1', 'hitrate', 'map']
    k_list = [1, 5, 10, 50]
    
    header = f"{'Metric':<12} | " + " | ".join([f"K={k:<8}" for k in k_list])
    print(header)
    print("-" * len(header))
    
    for m in metrics_to_print:
        row = f"{m.upper():<12} | "
        values = []
        for k in k_list:
            val = metrics.get(f'{m}@{k}', 0)
            values.append(f"{val:.6f}")
        print(row + " | ".join(values))
    
    print("-" * len(header))
    print(f"MRR: {metrics.get('mrr', 0):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train Content-Based Recommender')
    parser.add_argument('--encoder', '-e', 
                        choices=['tfidf', 'phobert', 'bge-m3', 'vndoc', 'e5-large', 'e5-base', 'vn-sbert', 'gte', 'precomputed'],
                        default='tfidf', help='Encoder type')
    parser.add_argument('--embedding-path', type=str, default=None,
                        help='Path to precomputed embeddings .pt file (required if encoder=precomputed)')
    parser.add_argument('--data-path', default='data/processed/strict_g1',
                        help='Path to processed data directory')
    parser.add_argument('--articles-path', default='data/raw/articles.csv',
                        help='Path to articles CSV file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--save-results', type=str, default=None, help='Path to save results JSON')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Content-Based Recommender Evaluation")
    print(f"Encoder: {args.encoder.upper()}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load data
    train_dict, test_dict, article_texts, n_users, n_items, idx2item = load_data(
        args.data_path, args.articles_path
    )
    
    if n_items == 0:
        print("Error: No items found!")
        return
        
    # Handle precomputed embeddings
    if args.encoder == 'precomputed':
        if not args.embedding_path or not Path(args.embedding_path).exists():
            print(f"Error: --embedding-path required for precomputed encoder")
            print(f"Available: checkpoints/*.pt or provide full path")
            return
    
    # Create model
    if args.encoder == 'tfidf':
        # TF-IDF is special case (doesn't use ContentBasedRecommender)
        print("\nUsing TFIDFRecommender (lightweight)")
        model = TFIDFRecommender(n_users, n_items, max_features=10000)
        
        if not article_texts:
            print("Error: article_texts required for TF-IDF. Provide --articles-path")
            return
            
        model.encode_articles(article_texts)
    else:
        # ContentBasedRecommender with various encoders
        print(f"\nUsing ContentBasedRecommender with {args.encoder}")
        
        model = ContentBasedRecommender(
            n_users=n_users,
            n_items=n_items,
            encoder_type=args.encoder if args.encoder != 'precomputed' else 'precomputed',
            precomputed_path=args.embedding_path,
            device=args.device
        )
        
        # Encode articles
        if args.encoder == 'precomputed':
            model.encode_articles()  # Load from file
        else:
            if not article_texts:
                print("Error: article_texts required. Provide --articles-path")
                return
            model.encode_articles(article_texts, batch_size=args.batch_size)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, test_dict, train_dict)
    
    # Print results
    print_results(metrics)
    
    # Save results
    if args.save_results:
        results_path = Path(args.save_results)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'encoder': args.encoder,
                'data_path': args.data_path,
                **metrics
            }, f, indent=2)
        print(f"\nSaved results to {results_path}")


if __name__ == '__main__':
    main()
