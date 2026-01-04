#!/usr/bin/env python3
"""
Train & Evaluate Content-Based Recommender Models
==================================================

Usage:
    # Zero-shot (No training, just evaluation)
    python scripts/train_cb_models.py --encoder tfidf
    python scripts/train_cb_models.py --encoder bge-m3 --data-path data/processed/strict_g1

    # Training (Train MLP for 20 epochs)
    python scripts/train_cb_models.py \
        --encoder bge-m3 \
        --data-path data/processed/strict_g1 \
        --epochs 20 \
        --lr 1e-3 \
        --save-results results/cb_trained.json
"""

import sys
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.content_based import ContentBasedRecommender, TFIDFRecommender


def load_data(data_path: str, articles_path: str = None):
    """
    Load interaction data and article texts.
    """
    data_path = Path(data_path)
    
    # Priority: user_article_graph (bipartite) > hetero_graph > category_graph > any .pt
    priority_files = ["user_article_graph.pt", "hetero_graph.pt", "category_graph.pt"]
    graph_file = None
    
    for fname in priority_files:
        if (data_path / fname).exists():
            graph_file = data_path / fname
            break
            
    if not graph_file:
        # Fallback
        graph_files = list(data_path.glob("*.pt"))
        valid_files = [f for f in graph_files if "split_indices" not in f.name]
        graph_file = valid_files[0] if valid_files else (graph_files[0] if graph_files else None)
    
    if graph_file:
        print(f"Loading data from: {graph_file.name}")
    
    train_dict, test_dict = defaultdict(list), defaultdict(list)
    n_users, n_items = 0, 0
    idx2item = {}
    
    if graph_file and graph_file.exists():
        data = torch.load(graph_file, weights_only=False)
        
        if 'n_users' in data: n_users = data['n_users']
        if 'n_items' in data: n_items = data['n_items']
        if 'idx2item' in data: idx2item = data['idx2item']
        
        if 'train_dict' in data and 'test_dict' in data:
            train_dict = data['train_dict']
            test_dict = data['test_dict']
        elif 'splits' in data:
            splits = data['splits']
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
    
    train_dict = dict(train_dict)
    test_dict = dict(test_dict)
            
    print(f"Loaded: {n_users} users, {n_items} items")
    print(f"Train: {len(train_dict)} users, Test: {len(test_dict)} users")
    
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
    
    print("Evaluating...")
    # model.eval() # CB recommender works in eval mode logically
    
    for user, test_items in tqdm(test_dict.items(), desc="Eval", ncols=80):
        test_items = set(test_items) if isinstance(test_items, list) else test_items
        if not test_items: continue
            
        history = list(train_dict.get(user, []))
        if not history: continue
        
        try:
            # exclude_read=True is default
            recs, scores = model.recommend(history, k=max_k)
        except Exception as e:
            if skipped_users < 3:
                print(f"Error for user {user}: {e}")
            skipped_users += 1
            continue
        
        evaluated_users += 1
        
        # Calculate metrics
        # MRR
        mrr = 0.0
        for i, item in enumerate(recs):
            if item in test_items:
                mrr = 1.0 / (i + 1)
                break
        results['mrr'].append(mrr)
        
        for k in k_list:
            topk = set(recs[:k])
            hits = len(topk & test_items)
            prec = hits / k
            rec = hits / len(test_items)
            results[f'recall@{k}'].append(rec)
            results[f'precision@{k}'].append(prec)
            results[f'hitrate@{k}'].append(1.0 if hits > 0 else 0.0)
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
    
    print(f"Evaluated: {evaluated_users}, Skipped: {skipped_users}")
    return {k: np.mean(v) for k, v in results.items() if len(v) > 0}


def precompute_user_means(model, train_dict):
    """Precompute simple mean of history embeddings for all users"""
    n_users = model.n_users
    dim = model.embedding_dim
    user_means = torch.zeros(n_users, dim, device=model.device)
    
    print("Precomputing user means for training...")
    with torch.no_grad():
        # Iterate over all users in train_dict
        for u, items in train_dict.items():
            if not items: continue
            # Clip items just in case
            items = [i for i in items if i < model.n_items]
            if not items: continue
            
            items_tensor = torch.LongTensor(items).to(model.device)
            embeds = model.article_embeddings[items_tensor]
            user_means[u] = embeds.mean(dim=0)
            
    return user_means


def train_model(model, train_dict, n_items, epochs=10, batch_size=1024, lr=1e-3, test_dict=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 1. Precompute static inputs for MLP (mean of history)
    # This is much faster than re-averaging every batch
    user_means = precompute_user_means(model, train_dict)
    
    # 2. Prepare all positive interactions
    all_users = []
    all_items = []
    for u, items in train_dict.items():
        all_users.extend([u] * len(items))
        all_items.extend(items)
        
    all_users = torch.LongTensor(all_users)
    all_items = torch.LongTensor(all_items)
    n_samples = len(all_users)
    
    print(f"\nStart Training: {epochs} epochs, {n_samples} interactions")
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", ncols=100)
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        # Shuffle
        perm = torch.randperm(n_samples)
        users = all_users[perm]
        items = all_items[perm]
        
        # Inner loop: iterate without tqdm to minimize spam
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(0, n_samples, batch_size):
            batch_idx = slice(i, i+batch_size)
            batch_u = users[batch_idx].to(model.device)
            batch_pos = items[batch_idx].to(model.device)
            batch_neg = torch.randint(0, n_items, (len(batch_u),), device=model.device)
            
            # Forward MLP
            u_input = user_means[batch_u] 
            u_vec = model.user_preference_encoder(u_input) # [B, Dim]
            
            # Use item_encoder 
            pos_raw = model.article_embeddings[batch_pos]
            neg_raw = model.article_embeddings[batch_neg]
            
            pos_vec = model.item_encoder(pos_raw)
            neg_vec = model.item_encoder(neg_raw)
            
            # Dot product scores
            pos_score = (u_vec * pos_vec).sum(dim=1)
            neg_score = (u_vec * neg_vec).sum(dim=1)
            
            # BPR Loss
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / n_batches
        epoch_pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

    print("Training finished.")


def print_results(metrics):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', '-e', default='tfidf', help='Encoder type')
    parser.add_argument('--embedding-path', type=str, default=None)
    parser.add_argument('--data-path', default='data/processed/strict_g1')
    parser.add_argument('--articles-path', default='data/raw/articles.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=0, help='Number of training epochs. 0 = Zero-shot.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save-results', type=str, default=None)
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"Content-Based Recommender (Epochs={args.epochs})")
    print(f"Encoder: {args.encoder.upper()}")
    print("="*60)
    
    # Load Data
    train_dict, test_dict, article_texts, n_users, n_items, idx2item = load_data(
        args.data_path, args.articles_path
    )
    
    if n_users == 0 or n_items == 0:
        print("Error: No users or items found!")
        return

    # TF-IDF Pipeline
    if args.encoder == 'tfidf':
        print("\nUsing TFIDFRecommender")
        model = TFIDFRecommender(n_users, n_items, max_features=10000)
        if not article_texts:
            print("Error: article_texts required")
            return
        model.encode_articles(article_texts)
        
        # TF-IDF has no training, just eval
        metrics = evaluate(model, test_dict, train_dict)
        print_results(metrics)
        
        if args.save_results:
            Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_results, 'w') as f:
                json.dump({'encoder': args.encoder, 'epochs': 0, 'data_path': args.data_path, **metrics}, f, indent=2)
        return

    # Two-Tower Pipeline
    print(f"\nUsing ContentBasedRecommender with {args.encoder}")
    model = ContentBasedRecommender(
        n_users=n_users,
        n_items=n_items,
        encoder_type=args.encoder if args.encoder != 'precomputed' else 'precomputed',
        precomputed_path=args.embedding_path,
        device=args.device
    )
    
    # Move model to device explicitly
    model.to(args.device)
    
    # Encode articles
    if args.encoder == 'precomputed':
        model.encode_articles()
    else:
        if not article_texts:
            print("Error required article texts")
            return
        model.encode_articles(article_texts, batch_size=args.batch_size)
    
    # Training or Zero-shot
    if args.epochs > 0:
        # Full training
        train_model(model, train_dict, n_items, epochs=args.epochs, lr=args.lr, test_dict=test_dict)
    else:
        # Zero-shot: Replace MLPs with Identity
        print("[Info] Zero-shot evaluation: Replacing MLPs with Identity")
        model.user_preference_encoder = nn.Identity()
        model.item_encoder = nn.Identity()
        model.to(args.device) # Re-move to device just in case
        
        
    # Final Evaluation
    metrics = evaluate(model, test_dict, train_dict)
    print_results(metrics)
    
    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump({
                'encoder': args.encoder,
                'epochs': args.epochs,
                'data_path': args.data_path,
                **metrics
            }, f, indent=2)
            
if __name__ == '__main__':
    main()
