"""
Run Content-Based Models (PhoBERT)
==================================

Vietnamese Text-based Recommendation using PhoBERT

Usage:
    python scripts/run_content_based.py
    python scripts/run_content_based.py --model phobert --device cuda
    python scripts/run_content_based.py --model hybrid --epochs 100
    python scripts/run_content_based.py --model simcse

Available models:
    - phobert: Pure content-based using PhoBERT
    - hybrid: CF + Content-based combined
    - simcse: Using SimCSE Vietnamese for better sentence similarity
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import argparse
import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Content-Based Model Training')
    
    # Model selection
    parser.add_argument('--model', type=str, default='phobert',
                        choices=['phobert', 'hybrid', 'simcse'],
                        help='Model type: phobert, hybrid, simcse')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload data from raw files')
    
    # Model hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--cf_dim', type=int, default=64,
                        help='CF embedding dim for hybrid model')
    parser.add_argument('--bert_model', type=str, default='vinai/phobert-base',
                        help='PhoBERT model name')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max sequence length for BERT')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='Freeze BERT parameters')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Hybrid alpha (weight for CF)')
    
    # Training (for hybrid)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--eval_every', type=int, default=10)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    # Text columns
    parser.add_argument('--text_cols', nargs='+', 
                        default=['title', 'short_description'],
                        help='Article columns to use for text encoding')
    
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save metrics as JSON')
    
    return parser.parse_args()


def load_data(data_path: str, text_columns: List[str], force_reload: bool = False):
    """Load data using ContentDataLoader"""
    from src.data.dataloader_content import load_content_data
    
    data_dict, loader = load_content_data(
        data_path,
        text_columns=text_columns,
        force_reload=force_reload
    )
    
    return data_dict, loader


def run_phobert_model(args, data_dict):
    """Run pure content-based PhoBERT model"""
    from src.models.content_based import ContentBasedRecommender
    from src.training.trainer_content_based import compute_metrics, print_metrics
    from tqdm import tqdm
    import torch.nn.functional as F
    
    print("\n" + "=" * 60)
    print("Running PhoBERT Content-Based Model")
    print("=" * 60)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize model
    model = ContentBasedRecommender(
        n_users=data_dict['n_users'],
        n_items=data_dict['n_items'],
        embedding_dim=args.embedding_dim,
        bert_model=args.bert_model,
        max_length=args.max_length,
        freeze_bert=True,
        device=device
    ).to(device)
    
    # Get article texts
    article_texts = data_dict['article_texts']
    
    # Check for cached embeddings
    embeddings_path = 'checkpoints/phobert_article_embeddings.pt'
    if not args.force_reload and os.path.exists(embeddings_path):
        print(f"\nLoading cached embeddings from {embeddings_path}")
        model.load_embeddings(embeddings_path)
    else:
        # Encode articles
        print(f"\nEncoding {len(article_texts)} articles with PhoBERT...")
        model.encode_articles(article_texts, batch_size=32)
        
        # Save embeddings
        os.makedirs('checkpoints', exist_ok=True)
        model.save_embeddings(embeddings_path)
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluating PhoBERT Content-Based Model")
    print("-" * 40)
    
    model.eval()
    predictions = {}
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    
    with torch.no_grad():
        for user_id in tqdm(test_dict.keys(), desc="Predicting"):
            history = list(train_dict.get(user_id, set()))
            if len(history) == 0:
                continue
                
            user_embed = model.get_user_preference(history)
            user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)
            article_embeds = F.normalize(model.article_embeddings, dim=-1)
            
            scores = torch.mm(user_embed, article_embeds.T).squeeze(0)
            predictions[user_id] = scores.cpu().numpy()
    
    # Compute metrics
    metrics = compute_metrics(predictions, test_dict, train_dict, k_list=[10, 20, 50])
    print_metrics(metrics)
    
    # Demo recommendations
    print("\n" + "-" * 40)
    print("Sample Recommendations")
    print("-" * 40)
    
    sample_users = list(test_dict.keys())[:3]
    for sample_user in sample_users:
        user_history = list(train_dict.get(sample_user, set()))
        if len(user_history) > 0:
            top_items, scores = model.recommend(user_history, k=5)
            
            print(f"\nUser {sample_user} (read {len(user_history)} articles)")
            print("Top 5 recommendations:")
            
            for i, (item_idx, score) in enumerate(zip(top_items, scores)):
                text = article_texts[item_idx][:80] if item_idx < len(article_texts) else "N/A"
                print(f"  {i+1}. [{score:.4f}] {text}...")
            break
    
    if args.save_results:
        try:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception as e:
            print(f"Error saving results: {e}")
            
    return metrics


def run_hybrid_model(args, data_dict):
    """Run hybrid CF + Content model"""
    from src.models.content_based import HybridRecommender
    from src.training.trainer_content_based import HybridTrainer, print_metrics
    
    print("\n" + "=" * 60)
    print("Running Hybrid (CF + PhoBERT) Model")
    print("=" * 60)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load articles for trainer
    articles_path = os.path.join(args.data_path, 'raw', 'articles.csv')
    articles_df = pd.read_csv(articles_path)
    
    # Initialize model
    model = HybridRecommender(
        n_users=data_dict['n_users'],
        n_items=data_dict['n_items'],
        cf_embedding_dim=args.cf_dim,
        content_embedding_dim=args.embedding_dim,
        bert_model=args.bert_model,
        alpha=args.alpha,
        freeze_bert=True,
        device=device
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.user_cf_embedding.parameters()},
        {'params': model.item_cf_embedding.parameters()},
        {'params': model.user_content_projection.parameters()},

    ], lr=args.lr)
    
    # Initialize trainer
    trainer = HybridTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        n_users=data_dict['n_users'],
        n_items=data_dict['n_items'],
        article_texts=data_dict['article_texts'],
        # articles_df=articles_df, # Use article_texts instead
        text_columns=args.text_cols
    )
    
    # Train
    best_recall = trainer.train(
        train_data=data_dict['train_data'],
        train_dict=data_dict['train_dict'],
        test_dict=data_dict['test_dict'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        save_path='checkpoints/hybrid_best.pth'
    )
    
    results = {'best_recall@20': best_recall}
    
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=4)
            
    return results


def run_simcse_model(args, data_dict):
    """Run SimCSE Vietnamese model (better for similarity)"""
    from src.models.content_based import ContentBasedRecommender
    from src.training.trainer_content_based import compute_metrics, print_metrics
    from tqdm import tqdm
    import torch.nn.functional as F
    
    print("\n" + "=" * 60)
    print("Running SimCSE Vietnamese Model")
    print("=" * 60)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Use SimCSE model
    simcse_model = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    
    model = ContentBasedRecommender(
        n_users=data_dict['n_users'],
        n_items=data_dict['n_items'],
        embedding_dim=args.embedding_dim,
        bert_model=simcse_model,
        max_length=args.max_length,
        freeze_bert=True,
        device=device
    ).to(device)
    
    article_texts = data_dict['article_texts']
    
    # Encode articles
    embeddings_path = 'checkpoints/simcse_article_embeddings.pt'
    if not args.force_reload and os.path.exists(embeddings_path):
        print(f"\nLoading cached embeddings from {embeddings_path}")
        model.load_embeddings(embeddings_path)
    else:
        print(f"\nEncoding {len(article_texts)} articles with SimCSE...")
        model.encode_articles(article_texts, batch_size=32)
        os.makedirs('checkpoints', exist_ok=True)
        model.save_embeddings(embeddings_path)
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluating SimCSE Content-Based Model")
    print("-" * 40)
    
    model.eval()
    predictions = {}
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    
    with torch.no_grad():
        for user_id in tqdm(test_dict.keys(), desc="Predicting"):
            history = list(train_dict.get(user_id, set()))
            if len(history) == 0:
                continue
                
            user_embed = model.get_user_preference(history)
            user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)
            article_embeds = F.normalize(model.article_embeddings, dim=-1)
            
            scores = torch.mm(user_embed, article_embeds.T).squeeze(0)
            predictions[user_id] = scores.cpu().numpy()
            
    # Compute metrics
    metrics = compute_metrics(predictions, test_dict, train_dict, k_list=[10, 20, 50])
    print_metrics(metrics)
    
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    return metrics
    
    metrics = compute_metrics(predictions, test_dict, train_dict, k_list=[10, 20, 50])
    print_metrics(metrics)
    
    return metrics


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Content-Based Recommendation for Vietnamese News")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"BERT: {args.bert_model}")
    print(f"Device: {args.device}")
    print(f"Text columns: {args.text_cols}")
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading Data")
    print("-" * 40)
    
    data_dict, loader = load_data(
        args.data_path, 
        text_columns=args.text_cols,
        force_reload=args.force_reload
    )
    
    print(f"\nData loaded:")
    print(f"  Users: {data_dict['n_users']}")
    print(f"  Items: {data_dict['n_items']}")
    print(f"  Train: {len(data_dict['train_data'])}")
    print(f"  Test users: {len(data_dict['test_dict'])}")
    print(f"  Article texts: {len(data_dict['article_texts'])}")
    
    # Run selected model
    if args.model == 'phobert':
        metrics = run_phobert_model(args, data_dict)
    elif args.model == 'hybrid':
        metrics = run_hybrid_model(args, data_dict)
    elif args.model == 'simcse':
        metrics = run_simcse_model(args, data_dict)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
