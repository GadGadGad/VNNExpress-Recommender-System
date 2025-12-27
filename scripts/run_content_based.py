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
    # Model selection
    parser.add_argument('--model', type=str, default='phobert',
                        choices=['phobert', 'hybrid', 'simcse', 'tfidf', 'e5', 'combined'],
                        help='Model type: phobert, hybrid, simcse, tfidf, e5, combined')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload data from raw files')
    
    # Eval
    parser.add_argument('--eval-protocol', type=str, default='full',
                        choices=['full', 'loo100', 'cold'],
                        help='Evaluation protocol: full, loo100, cold')
    
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
    parser.add_argument('--raw', action='store_true',
                        help='Use raw BERT embeddings (no projection)')
    
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


def run_phobert_model(args, data_dict, return_scores=False):
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
        device=device,
        use_raw=args.raw
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
    metrics = compute_metrics(predictions, test_dict, train_dict, 
                              k_list=[10, 20, 50], protocol=args.eval_protocol)
    print_metrics(metrics)
    
    if return_scores:
        return metrics, predictions
    
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
        device=device,
        use_raw=args.raw
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
    metrics = compute_metrics(predictions, test_dict, train_dict, 
                              k_list=[10, 20, 50], protocol=args.eval_protocol)
    print_metrics(metrics)
    
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    return metrics
    
    metrics = compute_metrics(predictions, test_dict, train_dict, k_list=[10, 20, 50])
    print_metrics(metrics)
    
    return metrics


def run_tfidf_model(args, data_dict, return_scores=False):
    """Run TF-IDF Content-Based Model"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from src.training.trainer_content_based import compute_metrics, print_metrics
    from tqdm import tqdm
    
    print("\n" + "=" * 60)
    print("Running TF-IDF Model")
    print("=" * 60)
    
    article_texts = data_dict['article_texts']
    
    # 1. Compute TF-IDF Matrix
    print("Computing TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=None) 
    # Note: Vietnamese stop words could be added here if available
    tfidf_matrix = vectorizer.fit_transform(article_texts)
    
    # 2. Predict for valid test users
    print("Generating predictions...")
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    
    predictions = {}
    
    # Get user profiles (average of history items)
    # This can be slow for many users, so we batch or iterate
    # For simplicity, let's iterate test users
    
    for user_id in tqdm(test_dict.keys(), desc="Predicting"):
        history = list(train_dict.get(user_id, set()))
        if not history:
            continue
            
        # Get indices of history items
        # data_dict['article_texts'] is a list, aligned with item_id 0..N-1
        # BUT we must ensure item_ids in train_dict match list indices.
        # usually load_content_data ensures item_id 0..N-1 maps to the list index.
        
        user_profile = np.asarray(tfidf_matrix[history].mean(axis=0)) # (1, vocab)
        
        # Calculate cosine similarity with ALL items
        # (1, vocab) dot (n_items, vocab).T -> (1, n_items)
        scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
        predictions[user_id] = scores
        
    # 3. Compute Metrics
    metrics = compute_metrics(predictions, test_dict, train_dict, 
                              k_list=[10, 20, 50], protocol=args.eval_protocol)
    print_metrics(metrics)
    
    if return_scores:
        return metrics, predictions
    
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    return metrics


def run_e5_model(args, data_dict):
    """Run Multilingual-E5 Content-Based Model"""
    from src.models.content_based import ContentBasedRecommender
    from src.training.trainer_content_based import compute_metrics, print_metrics
    from tqdm import tqdm
    import torch.nn.functional as F
    
    print("\n" + "=" * 60)
    print("Running Multilingual-E5 Model")
    print("=" * 60)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_name = "intfloat/multilingual-e5-small"
    
    # Reuse PhOBERT wrapper but with E5
    # E5 expects "query: " and "passage: " prefixes usually, but for similarity 
    # straightforward encoding often works. We'll stick to simple encoding for now.
    
    model = ContentBasedRecommender(
        n_users=data_dict['n_users'],
        n_items=data_dict['n_items'],
        embedding_dim=384, # E5-small dim
        bert_model=model_name,
        max_length=args.max_length,
        freeze_bert=True,
        device=device,
        use_raw=args.raw
    ).to(device)
    
    article_texts = data_dict['article_texts']
    
    # E5 specific: Add "passage: " prefix for document encoding is recommended
    # But ContentBasedRecommender.encode_articles doesn't do prefixes.
    # We will modify the input texts slightly before passing
    print("Adding 'passage: ' prefix for E5 encoding...")
    e5_texts = [f"passage: {t}" for t in article_texts]
    
    embeddings_path = 'checkpoints/e5_article_embeddings.pt'
    if not args.force_reload and os.path.exists(embeddings_path):
        print(f"Loading cached embeddings from {embeddings_path}")
        model.load_embeddings(embeddings_path)
    else:
        print(f"Encoding {len(e5_texts)} articles with E5...")
        # We need to manually call the encoder because wrapper expects 'texts'
        # The wrapper's encode_articles calls self.phobert_encoder(texts)
        model.encode_articles(e5_texts, batch_size=32)
        os.makedirs('checkpoints', exist_ok=True)
        model.save_embeddings(embeddings_path)
        
    # Evaluate
    print("\n" + "-" * 40)
    print(f"Evaluating E5 (Protocol: {args.eval_protocol})")
    print("-" * 40)
    
    model.eval()
    predictions = {}
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    
    with torch.no_grad():
        for user_id in tqdm(test_dict.keys(), desc="Predicting"):
            history = list(train_dict.get(user_id, set()))
            if not history:
                continue
                
            # user profile = mean of item embeddings
            # For query side, E5 usually expects "query: ". 
            # But here we are doing Item-Item centroid. 
            # Ideally: user_embed = Mean(Item_Embeddings).
            # This is symmetric similarity.
            
            user_embed = model.get_user_preference(history)
            user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)
            article_embeds = F.normalize(model.article_embeddings, dim=-1)
            
            scores = torch.mm(user_embed, article_embeds.T).squeeze(0)
            predictions[user_id] = scores.cpu().numpy()
            
    metrics = compute_metrics(predictions, test_dict, train_dict, 
                              k_list=[10, 20, 50], protocol=args.eval_protocol)
    print_metrics(metrics)
    
    if args.save_results:
        try:
            import json
            with open(args.save_results, 'w') as f:
                json.dump(metrics, f, indent=4)
        except Exception:
            pass
            
    return metrics


    return metrics


def run_combined_model(args, data_dict):
    """Run Combined TF-IDF + PhoBERT Model"""
    from src.training.trainer_content_based import compute_metrics, print_metrics
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    print("\n" + "=" * 60)
    print("Running Combined Model (TF-IDF + PhoBERT)")
    print("=" * 60)
    
    # Run TF-IDF
    print("\n>>> Phase 1: Running TF-IDF...")
    _, tfidf_scores = run_tfidf_model(args, data_dict, return_scores=True)
    
    # Run PhoBERT
    print("\n>>> Phase 2: Running PhoBERT...")
    _, phobert_scores = run_phobert_model(args, data_dict, return_scores=True)
    
    print("\n>>> Phase 3: Combining Scores...")
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    
    combined_predictions = {}
    alpha = args.alpha
    print(f"Mixing weight alpha (TF-IDF): {alpha}")
    
    # Intersection of users
    users = set(tfidf_scores.keys()) & set(phobert_scores.keys())
    
    for user_id in users:
        s1 = tfidf_scores[user_id]
        s2 = phobert_scores[user_id]
        
        # Normalize scores to [0, 1] for fair combination
        if s1.max() > s1.min():
            s1 = (s1 - s1.min()) / (s1.max() - s1.min())
            
        if s2.max() > s2.min():
            s2 = (s2 - s2.min()) / (s2.max() - s2.min())
            
        combined = alpha * s1 + (1 - alpha) * s2
        combined_predictions[user_id] = combined
        
    # Evaluate
    metrics = compute_metrics(combined_predictions, test_dict, train_dict, 
                              k_list=[10, 20, 50], protocol=args.eval_protocol)
    print_metrics(metrics)
    
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, indent=4)
            
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
        # Hybrid logic is more complex and might need separate handling for full/loo
        # Assuming run_hybrid_model handles its own eval loop; we might need to update it too
        # but for now let's focus on pure CB models
        metrics = run_hybrid_model(args, data_dict)
    elif args.model == 'simcse':
        metrics = run_simcse_model(args, data_dict)
    elif args.model == 'tfidf':
        metrics = run_tfidf_model(args, data_dict)
    elif args.model == 'e5':
        metrics = run_e5_model(args, data_dict)
    elif args.model == 'combined':
        metrics = run_combined_model(args, data_dict)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

