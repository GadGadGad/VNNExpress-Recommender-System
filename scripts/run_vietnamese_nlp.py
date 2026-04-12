"""
Run Traditional Vietnamese NLP Recommenders
============================================

Script to run and compare traditional NLP methods for Vietnamese news recommendation:
- TF-IDF + Cosine Similarity
- BM25 (Okapi, BM25+)
- Word2Vec/FastText

Usage:
    python scripts/run_vietnamese_nlp.py --method tfidf
    python scripts/run_vietnamese_nlp.py --method bm25
    python scripts/run_vietnamese_nlp.py --method word2vec --embedding_path path/to/cc.vi.300.bin
    python scripts/run_vietnamese_nlp.py --method all --compare
"""

import sys
import os
import argparse
import time
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description='Vietnamese NLP Recommenders')
    
    # Method selection
    parser.add_argument('--method', type=str, default='tfidf',
                        choices=['tfidf', 'bm25', 'word2vec', 'fasttext', 'all', 'ensemble'],
                        help='Method to run')
    
    # Data
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--text_cols', nargs='+',
                        default=['title', 'short_description'],
                        help='Text columns to use')
    
    # Preprocessing
    parser.add_argument('--use_segmentation', action='store_true', default=True,
                        help='Use Vietnamese word segmentation')
    parser.add_argument('--no_segmentation', action='store_false', dest='use_segmentation',
                        help='Disable word segmentation')
    parser.add_argument('--remove_stopwords', action='store_true', default=True,
                        help='Remove Vietnamese stopwords')
    parser.add_argument('--no_stopwords', action='store_false', dest='remove_stopwords',
                        help='Keep stopwords')
    
    # TF-IDF parameters
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Max vocabulary size for TF-IDF')
    parser.add_argument('--ngram_range', type=int, nargs=2, default=[1, 2],
                        help='N-gram range (e.g., 1 2 for unigrams and bigrams)')
    
    # BM25 parameters
    parser.add_argument('--bm25_k1', type=float, default=1.5,
                        help='BM25 k1 parameter')
    parser.add_argument('--bm25_b', type=float, default=0.75,
                        help='BM25 b parameter')
    parser.add_argument('--bm25_delta', type=float, default=0.5,
                        help='BM25+ delta parameter (0.0 for Okapi)')
    
    # Word2Vec/FastText parameters
    parser.add_argument('--embedding_path', type=str, default=None,
                        help='Path to pretrained word vectors')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Word embedding dimension')
    parser.add_argument('--aggregation', type=str, default='sif',
                        choices=['mean', 'tfidf_weighted', 'sif'],
                        help='How to aggregate word vectors')
    
    # Evaluation
    parser.add_argument('--k_list', type=int, nargs='+', default=[10, 20, 50],
                        help='K values for metrics')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all methods')
    
    # Output
    parser.add_argument('--save_model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory')
    
    return parser.parse_args()


def load_data(data_path: str, text_columns: List[str]):
    """Load data for content-based models"""
    from src.data.dataloader_content import load_content_data
    
    data_dict, loader = load_content_data(
        data_path,
        text_columns=text_columns
    )
    
    return data_dict, loader


def run_tfidf(args, data_dict) -> Dict[str, float]:
    """Run TF-IDF based recommender"""
    from src.vietnamese_nlp import TFIDFRecommender
    
    print("\n" + "=" * 60)
    print("TF-IDF Recommender")
    print("=" * 60)
    
    # Initialize
    recommender = TFIDFRecommender(
        use_preprocessing=True,
        use_word_segmentation=args.use_segmentation,
        remove_stopwords=args.remove_stopwords,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range),
        sublinear_tf=True
    )
    
    # Fit
    start_time = time.time()
    recommender.fit(data_dict['article_texts'])
    fit_time = time.time() - start_time
    print(f"\nFit time: {fit_time:.2f}s")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation")
    print("-" * 40)
    
    start_time = time.time()
    metrics = recommender.evaluate(
        data_dict['train_dict'],
        data_dict['test_dict'],
        k_list=args.k_list
    )
    eval_time = time.time() - start_time
    
    print_metrics(metrics)
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, 'tfidf_recommender.pkl')
        recommender.save(save_path)
    
    # Demo
    demo_recommendations(recommender, data_dict)
    
    return metrics


def run_bm25(args, data_dict) -> Dict[str, float]:
    """Run BM25 based recommender"""
    from src.vietnamese_nlp import BM25Recommender
    
    print("\n" + "=" * 60)
    print("BM25 Recommender")
    print("=" * 60)
    print(f"Parameters: k1={args.bm25_k1}, b={args.bm25_b}")
    
    # Initialize
    recommender = BM25Recommender(
        use_preprocessing=True,
        use_word_segmentation=args.use_segmentation,
        remove_stopwords=args.remove_stopwords,
        k1=args.bm25_k1,
        b=args.bm25_b,
        delta=args.bm25_delta
    )
    
    # Fit
    start_time = time.time()
    recommender.fit(data_dict['article_texts'])
    fit_time = time.time() - start_time
    print(f"\nFit time: {fit_time:.2f}s")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation")
    print("-" * 40)
    
    start_time = time.time()
    metrics = recommender.evaluate(
        data_dict['train_dict'],
        data_dict['test_dict'],
        k_list=args.k_list
    )
    eval_time = time.time() - start_time
    
    print_metrics(metrics)
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, 'bm25_recommender.pkl')
        recommender.save(save_path)
    
    # Demo
    demo_recommendations(recommender, data_dict)
    
    return metrics


def run_word2vec(args, data_dict) -> Dict[str, float]:
    """Run Word2Vec/FastText based recommender"""
    from src.vietnamese_nlp import Word2VecRecommender
    
    print("\n" + "=" * 60)
    print("Word2Vec/FastText Recommender")
    print("=" * 60)
    
    recommender = Word2VecRecommender(
        use_preprocessing=True,
        use_word_segmentation=args.use_segmentation,
        remove_stopwords=args.remove_stopwords,
        embedding_dim=args.embedding_dim,
        aggregation=args.aggregation
    )
    
    # Load embeddings
    if args.embedding_path:
        print(f"\nLoading pretrained embeddings from: {args.embedding_path}")
        recommender.load_pretrained(args.embedding_path)
    else:
        print("\n[Warning] No pretrained embeddings provided.")
        print("Training on corpus (may take a while)...")
        recommender.train_embeddings(
            data_dict['article_texts'],
            vector_size=args.embedding_dim,
            window=5,
            min_count=2,
            epochs=5
        )
    
    # Fit
    start_time = time.time()
    recommender.fit(data_dict['article_texts'])
    fit_time = time.time() - start_time
    print(f"\nFit time: {fit_time:.2f}s")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation")
    print("-" * 40)
    
    start_time = time.time()
    metrics = recommender.evaluate(
        data_dict['train_dict'],
        data_dict['test_dict'],
        k_list=args.k_list
    )
    eval_time = time.time() - start_time
    
    print_metrics(metrics)
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    # Save model
    if args.save_model:
        save_path = os.path.join(args.output_dir, 'word2vec_recommender.pkl')
        recommender.save(save_path)
    
    # Demo
    demo_recommendations(recommender, data_dict)
    
    return metrics



# -------------------------------------------------------------------------
# Ensemble Recommender (Dense + Sparse)
# -------------------------------------------------------------------------
def run_ensemble(args, data_dict) -> Dict[str, float]:
    """Run Ensemble (Dense + Sparse) Recommender"""
    from src.vietnamese_nlp import TFIDFRecommender
    import torch
    import torch.nn.functional as F
    
    print("\n" + "=" * 60)
    print("Ensemble Recommender (TF-IDF + PhoBERT)")
    print("=" * 60)
    
    class EnsembleRecommender(TFIDFRecommender):
        def __init__(self, dense_path, alpha=0.5, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            print(f"Loading dense embeddings from {dense_path}...")
            if os.path.exists(dense_path):
                self.dense_embeddings = torch.load(dense_path, map_location='cpu')
                # Normalize
                self.dense_embeddings = F.normalize(self.dense_embeddings, p=2, dim=1)
                print(f"Loaded embeddings: {self.dense_embeddings.shape}")
            else:
                print(f"Error: {dense_path} not found!")
                self.dense_embeddings = None
                
        def compute_scores(self, user_history, aggregation='mean'):
            # 1. Sparse Scores (TF-IDF)
            sparse_scores = super().compute_scores(user_history, aggregation)
            
            # 2. Dense Scores (PhoBERT)
            if self.dense_embeddings is not None and len(user_history) > 0:
                # User profile = Mean of history embeddings
                # Check indices
                valid_indices = [idx for idx in user_history if idx < self.dense_embeddings.shape[0]]
                if not valid_indices:
                    dense_scores = np.zeros_like(sparse_scores)
                else:
                    history_embeds = self.dense_embeddings[valid_indices]
                    user_embed = history_embeds.mean(dim=0).unsqueeze(0)
                    # Normalize user profile?
                    # Cosine sim requires normalized vectors. 
                    # History embeds already normalized. Mean might not be.
                    user_embed = F.normalize(user_embed, p=2, dim=1)
                    
                    # Compute similarity
                    # Dense shape: [N_items, Dim]
                    # Score shape: [1, N_items]
                    dense_scores = torch.mm(user_embed, self.dense_embeddings.T).squeeze(0).numpy()
                    
                    # Validate shapes
                    # They should match if data_dict is consistent.
                    if len(dense_scores) != len(sparse_scores):
                        # Pad or truncate?
                        min_len = min(len(dense_scores), len(sparse_scores))
                        dense_scores = dense_scores[:min_len]
                        sparse_scores = sparse_scores[:min_len]
            else:
                dense_scores = np.zeros_like(sparse_scores)
                
            # Combine: Sparse [-1, 1] + Dense [-1, 1]
            return self.alpha * dense_scores + (1 - self.alpha) * sparse_scores

    # Initialize
    alpha = args.alpha if hasattr(args, 'alpha') else 0.2
    print(f"Ensemble Alpha (Dense weight): {alpha}")
    
    recommender = EnsembleRecommender(
        dense_path='checkpoints/phobert_article_embeddings.pt',
        alpha=alpha,
        use_preprocessing=True,
        use_word_segmentation=args.use_segmentation,
        remove_stopwords=args.remove_stopwords,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range)
    )
    
    # Fit TF-IDF
    start_time = time.time()
    recommender.fit(data_dict['article_texts'])
    fit_time = time.time() - start_time
    print(f"\nFit time: {fit_time:.2f}s")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation")
    print("-" * 40)
    
    start_time = time.time()
    metrics = recommender.evaluate(
        data_dict['train_dict'],
        data_dict['test_dict'],
        k_list=args.k_list
    )
    eval_time = time.time() - start_time
    
    print_metrics(metrics)
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    return metrics



# -------------------------------------------------------------------------
# Ensemble Recommender (Dense + Sparse)
# -------------------------------------------------------------------------
def run_ensemble(args, data_dict) -> Dict[str, float]:
    """Run Ensemble (Dense + Sparse) Recommender"""
    from src.vietnamese_nlp import TFIDFRecommender
    import torch
    import torch.nn.functional as F
    
    print("\n" + "=" * 60)
    print("Ensemble Recommender (TF-IDF + PhoBERT)")
    print("=" * 60)
    
    class EnsembleRecommender(TFIDFRecommender):
        def __init__(self, dense_path, alpha=0.5, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            print(f"Loading dense embeddings from {dense_path}...")
            if os.path.exists(dense_path):
                self.dense_embeddings = torch.load(dense_path, map_location='cpu')
                # Normalize (important for cosine)
                self.dense_embeddings = F.normalize(self.dense_embeddings, p=2, dim=1)
                print(f"Loaded embeddings: {self.dense_embeddings.shape}")
            else:
                print(f"Error: {dense_path} not found!")
                self.dense_embeddings = None
                
        def compute_scores(self, user_history, aggregation='mean'):
            # 1. Sparse Scores (TF-IDF)
            sparse_scores = super().compute_scores(user_history, aggregation)
            
            # 2. Dense Scores (PhoBERT)
            if self.dense_embeddings is not None and len(user_history) > 0:
                # User profile = Mean of history embeddings
                # Check valid indices
                valid_indices = [idx for idx in user_history if idx < self.dense_embeddings.shape[0]]
                if not valid_indices:
                    dense_scores = np.zeros_like(sparse_scores)
                else:
                    history_embeds = self.dense_embeddings[valid_indices]
                    user_embed = history_embeds.mean(dim=0).unsqueeze(0)
                    user_embed = F.normalize(user_embed, p=2, dim=1)
                    
                    # Compute similarity [1, N_items]
                    dense_scores = torch.mm(user_embed, self.dense_embeddings.T).squeeze(0).numpy()
                    
                    # Ensure shapes match
                    if len(dense_scores) != len(sparse_scores):
                        min_len = min(len(dense_scores), len(sparse_scores))
                        dense_scores = dense_scores[:min_len]
                        sparse_scores = sparse_scores[:min_len]
            else:
                dense_scores = np.zeros_like(sparse_scores)
                
            # 3. Combine
            return self.alpha * dense_scores + (1 - self.alpha) * sparse_scores

    # Initialize
    alpha = args.alpha if hasattr(args, 'alpha') else 0.2
    print(f"Ensemble Alpha (Dense weight): {alpha}")
    
    recommender = EnsembleRecommender(
        dense_path='checkpoints/phobert_article_embeddings.pt',
        alpha=alpha,
        use_preprocessing=True,
        use_word_segmentation=args.use_segmentation,
        remove_stopwords=args.remove_stopwords,
        max_features=args.max_features,
        ngram_range=tuple(args.ngram_range)
    )
    
    # Fit TF-IDF
    start_time = time.time()
    recommender.fit(data_dict['article_texts'])
    fit_time = time.time() - start_time
    print(f"\nFit time: {fit_time:.2f}s")
    
    # Evaluate
    print("\n" + "-" * 40)
    print("Evaluation")
    print("-" * 40)
    
    start_time = time.time()
    metrics = recommender.evaluate(
        data_dict['train_dict'],
        data_dict['test_dict'],
        k_list=args.k_list
    )
    eval_time = time.time() - start_time
    
    print_metrics(metrics)
    print(f"\nEvaluation time: {eval_time:.2f}s")
    
    return metrics


def run_all_methods(args, data_dict) -> Dict[str, Dict[str, float]]:
    """Run and compare all methods"""
    results = {}
    
    # TF-IDF
    try:
        results['TF-IDF'] = run_tfidf(args, data_dict)
    except Exception as e:
        print(f"[Error] TF-IDF failed: {e}")
    
    # Ensemble (0.2 Dense)
    try:
        args.alpha = 0.2
        results['Ensemble (0.2)'] = run_ensemble(args, data_dict)
    except Exception as e:
        print(f"[Error] Ensemble failed: {e}")
        
    # BM25
    try:
        results['BM25'] = run_bm25(args, data_dict)
    except Exception as e:
        print(f"[Error] BM25 failed: {e}")
        
    # Word2Vec
    try:
        results['Word2Vec'] = run_word2vec(args, data_dict)
    except Exception as e:
        print(f"[Error] Word2Vec failed: {e}")

    # Ensemble (0.2 Dense)
    try:
        args.alpha = 0.2
        results['Ensemble (0.2)'] = run_ensemble(args, data_dict)
    except Exception as e:
        print(f"[Error] Ensemble failed: {e}")
    
    return results





def print_metrics(metrics: Dict[str, float]):
    """Pretty print metrics"""
    recalls = sorted([(k, v) for k, v in metrics.items() if 'Recall' in k])
    ndcgs = sorted([(k, v) for k, v in metrics.items() if 'NDCG' in k])
    hrs = sorted([(k, v) for k, v in metrics.items() if 'HR' in k])
    
    if recalls:
        print("Recall:  " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in recalls]))
    if ndcgs:
        print("NDCG:    " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in ndcgs]))
    if hrs:
        print("HR:      " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in hrs]))


def print_comparison(all_results: Dict[str, Dict[str, float]]):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("COMPARISON OF METHODS")
    print("=" * 80)
    
    # Build comparison table
    methods = list(all_results.keys())
    if not methods:
        print("No results to compare.")
        return
        
    metrics = list(all_results[methods[0]].keys())
    
    # Print header
    header = f"{'Method':<15}"
    for metric in sorted(metrics):
        header += f"{metric:>12}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for method in methods:
        row = f"{method:<15}"
        for metric in sorted(metrics):
            value = all_results[method].get(metric, 0)
            row += f"{value:>12.4f}"
        print(row)
        
    # Find best method for each metric
    print("\n" + "-" * 40)
    print("Best method for each metric:")
    for metric in sorted(metrics):
        best_method = max(methods, key=lambda m: all_results[m].get(metric, 0))
        best_value = all_results[best_method].get(metric, 0)
        print(f"  {metric}: {best_method} ({best_value:.4f})")


def demo_recommendations(recommender, data_dict):
    """Show sample recommendations"""
    print("\n" + "-" * 40)
    print("Sample Recommendations")
    print("-" * 40)
    
    # Find a user with some history
    train_dict = data_dict['train_dict']
    test_dict = data_dict['test_dict']
    article_texts = data_dict['article_texts']
    
    for user_id in list(test_dict.keys())[:5]:
        history = list(train_dict.get(user_id, set()))
        if len(history) >= 3:
            print(f"\nUser {user_id} (read {len(history)} articles)")
            
            # Show history
            print("  Recent reads:")
            for idx in history[:3]:
                text = article_texts[idx][:60] if idx < len(article_texts) else "N/A"
                print(f"    [{idx}] {text}...")
            
            # Get recommendations
            top_items, top_scores = recommender.recommend(history, k=5)
            
            print("  Top 5 recommendations:")
            for i, (idx, score) in enumerate(zip(top_items, top_scores)):
                text = article_texts[idx][:60] if idx < len(article_texts) else "N/A"
                print(f"    {i+1}. [{score:.4f}] {text}...")
            
            break


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Vietnamese NLP Content-Based Recommendation")
    print("=" * 60)
    print(f"Method: {args.method}")
    print(f"Word segmentation: {args.use_segmentation}")
    print(f"Remove stopwords: {args.remove_stopwords}")
    print(f"Text columns: {args.text_cols}")
    
    # Load data
    print("\n" + "-" * 40)
    print("Loading Data")
    print("-" * 40)
    
    data_dict, loader = load_data(args.data_path, args.text_cols)
    
    print(f"\nData loaded:")
    print(f"  Users: {data_dict['n_users']}")
    print(f"  Items: {data_dict['n_items']}")
    print(f"  Train interactions: {len(data_dict['train_data'])}")
    print(f"  Test users: {len(data_dict['test_dict'])}")
    print(f"  Article texts: {len(data_dict['article_texts'])}")
    
    # Run methods
    if args.method == 'all' or args.compare:
        all_results = run_all_methods(args, data_dict)
        print_comparison(all_results)
    elif args.method == 'tfidf':
        run_tfidf(args, data_dict)
    elif args.method == 'ensemble':
        run_ensemble(args, data_dict)
    elif args.method == 'bm25':
        run_bm25(args, data_dict)
    elif args.method in ['word2vec', 'fasttext']:
        run_word2vec(args, data_dict)
    else:
        print(f"Unknown method: {args.method}")
        
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
