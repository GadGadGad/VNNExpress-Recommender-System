#!/usr/bin/env python3
"""
Train Collaborative Filtering / Contrastive Learning Models
Supports: SimpleX, DirectAU, SGL, SimGCL, NCL, LightGCL
"""
import os
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cf_data_loader import load_data, compute_normalized_adjacency
from src.data.embedding_loader import load_pretrained_embeddings, SemanticEmbeddingLayer, UserPriorLayer
from src.training.cf_evaluator import load_item_categories
from src.training.cf_trainer import train_model
from src.models.lightgcl_wrapper import LightGCLWrapper

from src.models import LightGCL, SimGCL, XSimGCL, MAHGN, LightGCN
from src.models.ma_hcl import MAHCL
from src.models.bigcf import BIGCF
from src.models.semantic_id import generate_semantic_ids
from src.inference.re_ranker import CalibratedReRanker


def main():
    parser = argparse.ArgumentParser(description='Train CF/CL Models')
    parser.add_argument('--model', '-m', choices=['simgcl', 'xsimgcl', 'lightgcl', 'ma-hcl', 'ma_hgn', 'bigcf', 'lightgcn'], 
                        default='simgcl', help='Model to train')
    parser.add_argument('--data-path', default='data/processed/strict_g1', help='Path to graph data')
    parser.add_argument('--articles-path', default=None, help='Explicit path to articles.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')
    parser.add_argument('--re-rank', action='store_true', help='Use Semantic Re-ranking')
    parser.add_argument('--gnn-type', choices=['gat', 'sage', 'gcn', 'transformer'], default='gat', help='GNN type for MA-HGN (gat, sage, gcn, transformer)')
    parser.add_argument('--cold-p', type=float, default=0.2, help='Percentage of cold users for evaluation')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # LightGCL-specific parameters
    parser.add_argument('--svd-q', type=int, default=20, help='LightGCL: SVD components (default: 20)')
    parser.add_argument('--ssl-weight', type=float, default=0.1, help='LightGCL: SSL loss weight (default: 0.1)')
    parser.add_argument('--temp', type=float, default=0.2, help='LightGCL: Temperature for contrastive loss (default: 0.2)')
    
    # Embedding arguments
    parser.add_argument('--embedding', choices=['random', 'phobert', 'tfidf', 'vndoc', 'bge-m3', 'gte', 'e5-large', 'e5-base', 'vn-sbert'], default='random',
                        help='Embedding initialization')
    parser.add_argument('--augment', choices=['none', 'llmrec'], default='none',
                        help='Graph augmentation strategy')
    parser.add_argument('--semantic-id-bits', type=int, default=0,
                        help='Number of semantic ID codebooks (0 to disable)')
    parser.add_argument('--denoise-ratio', type=float, default=0.0,
                        help='Ratio of noisy interactions to prune per batch (0.0 to disable)')
    parser.add_argument('--rerank', choices=['none', 'mmr', 'calib'], default='none',
                        help='Post-processing re-ranking strategy')
    parser.add_argument('--eval-protocol', choices=['full', 'loo100', 'cold'], default='full',
                        help='Evaluation protocol: full (all items), loo100 (leave-one-out + 100 neg), cold (cold-start users)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save metrics in JSON format')
    parser.add_argument('--neg-ratio', type=int, default=1,
                        help='Number of negative samples per positive sample (default: 1)')
    parser.add_argument('--cold-start', type=int, default=10,
                        help='Number of cold-start users to evaluate (default: 10)')
    parser.add_argument('--social-weight', type=float, default=1.0,
                        help='Weight for social reply edges in adjacency matrix (default: 1.0)')
    parser.add_argument('--graph-type', choices=['bipartite', 'hetero', 'article', 'category'], default='bipartite',
                        help='Graph type: bipartite, hetero, article (Article-Augmented), or category (Category Hubs)')
    parser.add_argument('--split-strategy', choices=['random', 'time'], default='random',
                        help='Data splitting strategy: "random" (shuffle) or "time" (chronological)')

    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon for MA-HCL (default: 0.1)')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 60)
    print(f"Training {args.model.upper()}")
    print(f"Embedding Initialization: {args.embedding.upper()}")
    print(f"Graph Type: {args.graph_type.upper()}")
    print(f"Data Path: {args.data_path}")
    print("=" * 60)
    
    # --- Load Data ---
    data = _load_graph_data(args)
    n_users, n_items = data['n_users'], data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Train: {len(data['train_pairs'])}, Test users: {len(data['test_dict'])}")
    
    # Identify cold-start users (users with <= 3 training interactions)
    train_dict = data['train_dict']
    cold_threshold = 3
    cold_users = {u for u, items in train_dict.items() if len(items) <= cold_threshold}
    print(f"Cold-start users (≤{cold_threshold} interactions): {len(cold_users)}")
    
    # --- Precompute Adjacency ---
    _precompute_adjacency(args, data, device)

    # --- Load Embeddings ---
    train_item_indices = list(set([i for u, i in data['train_pairs']]))
    pretrained_emb = load_pretrained_embeddings(
        args.embedding, n_items, args.hidden_dim, device, 
        train_item_indices=train_item_indices,
        data_path=args.data_path, articles_path=args.articles_path
    )
    
    # --- Semantic IDs & User Priors ---
    semantic_ids = _load_semantic_ids(args, pretrained_emb, device)
    user_priors = _load_user_priors(args, n_users, device)
    
    # --- Re-ranker ---
    re_ranker = _load_reranker(args)

    # --- Build Model ---
    model = _build_model(args, n_users, n_items, data, device, pretrained_emb, semantic_ids, user_priors)
    
    # Ensure model is on the correct device if not handled internally (e.g., LightGCL)
    if args.model != 'lightgcl':
        model = model.to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- Train ---
    if hasattr(model, 'forward') and pretrained_emb is not None:
        model.item_content = pretrained_emb # Attach for evaluate
        
    best_metrics = train_model(model, data, args, device, item_content=pretrained_emb, 
                               semantic_ids=semantic_ids, user_priors=user_priors, 
                               re_ranker=re_ranker, cold_users=cold_users)

    # --- Save & Print Results ---
    _save_and_print_results(args, model, n_users, n_items, best_metrics)


def _load_graph_data(args):
    """Load graph data based on graph type argument."""
    if args.graph_type == 'hetero':
        hetero_path = Path(args.data_path) / 'full_hetero_graph.pt'
        if not hetero_path.exists():
            hetero_path = Path(args.data_path) / 'all_graphs' / 'full_hetero_graph.pt'
            
        if hetero_path.exists():
            print(f"  Loading Heterogeneous Graph from: {hetero_path}")
            return load_data(str(hetero_path), split_strategy=args.split_strategy)
        else:
            print(f"  Warning: {hetero_path} not found. Falling back to default bipartite graph.")
            return load_data(args.data_path, split_strategy=args.split_strategy)
    
    elif args.graph_type == 'category':
        cat_path = Path(args.data_path) / 'all_graphs' / 'category_graph.pt'
        if not cat_path.exists():
             cat_path = Path(args.data_path) / 'category_graph.pt'
             
        if cat_path.exists():
            print(f"  Loading Category-Augmented Graph from: {cat_path}")
            return load_data(str(cat_path), split_strategy=args.split_strategy)
        else:
            print(f"  Warning: {cat_path} not found. Falling back.")
            return load_data(args.data_path, split_strategy=args.split_strategy)

    elif args.graph_type == 'article':
        hetero_path = Path(args.data_path) / 'all_graphs' / 'full_hetero_graph.pt'
        if hetero_path.exists():
            print(f"  Loading Full Base Graph from: {hetero_path}")
            data = load_data(args.data_path, split_strategy=args.split_strategy)
            
            # REMOVE Social Edges to ensure purely Article-Augmented experiment
            if isinstance(data, dict):
                 if 'edge_index_dict' in data and ('user', 'replied_to', 'user') in data['edge_index_dict']:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['edge_index_dict'][('user', 'replied_to', 'user')]
            elif hasattr(data, 'edge_index_dict'):
                 if ('user', 'replied_to', 'user') in data.edge_types:
                     print("  Removing Social Edges for Article-Augmented experiment...")
                     del data['user', 'replied_to', 'user']
        else:
             print("Full graph not found, falling back (might fail dimensions)")
             data = load_data(args.data_path, split_strategy=args.split_strategy)
        
        # Load auxiliary Article-Article graph
        article_path = Path(args.data_path) / 'all_graphs' / 'article_article_graph_users.pt'
        if article_path.exists():
            print(f"  Loading Article-Article Edges from: {article_path}")
            article_data = torch.load(article_path, weights_only=False)
            data['article_edge_index'] = article_data.edge_index
        else:
            print(f"  Warning: {article_path} not found. Using Bipartite only.")
        return data
    
    else:
        return load_data(args.data_path)


def _precompute_adjacency(args, data, device):
    """Precompute normalized adjacency for graph-based models."""
    graph_models = ['simgcl', 'bigcf', 'igcl', 'xsimgcl', 'lightgcl', 'ma-hcl', 'ma_hgn', 'lightgcn']
    if args.model not in graph_models:
        return
    
    n_users, n_items = data['n_users'], data['n_items']
    print(f"\nPrecomputing normalized adjacency for {args.model.upper()}...")
    
    augmented_pairs = None
    # Fix ambiguous boolean evaluation for Tensors
    item_item_edges = data.get('article_edge_index')
    if item_item_edges is None:
        item_item_edges = data.get('user_category_edge_index')
    if item_item_edges is None:
        item_item_edges = data.get('cat_article_edges')
    
    if args.augment == 'llmrec':
        augment_path = Path('data/processed/augmented_edges.pt')
        if augment_path.exists():
            print(f"  Loading augmented interactions from {augment_path}...")
            aug_data = torch.load(augment_path, weights_only=False)
            augmented_pairs = aug_data.get('augmented_pairs', None)
            item_item_edges = aug_data.get('item_item_edges', item_item_edges) 
        else:
            print(f"  Warning: {augment_path} not found. Running without augmentation.")
    
    # Check for weights in the data
    edge_weights = data.get('edge_weight')
    if edge_weights is not None:
        if isinstance(edge_weights, torch.Tensor):
            edge_weights = edge_weights.numpy()
        print(f"  Found {len(edge_weights)} edge weights. Using weighted adjacency.")
    
    user_user_edges = data.get('user_user_edges') 
    
    if user_user_edges is not None:
         print(f"  Using {user_user_edges.shape[1]} pre-filtered social edges from data dictionary.")
    else:
         # Legacy Fallback
         user_user_edges = None
         social_graph_path = Path(args.data_path) / 'full_hetero_graph.pt'
         if social_graph_path.exists():
            print(f"  [WARNING] Loading social signals from {social_graph_path} (Potential Leakage if not filtered!)")
            social_data = torch.load(social_graph_path, weights_only=False)
            if isinstance(social_data, dict) and 'edge_index_dict' in social_data:
                edges_dict = social_data['edge_index_dict']
                user_user_edges = edges_dict.get(('user', 'replied_to', 'user'))
            elif hasattr(social_data, 'edge_index_dict'):
                user_user_edges = social_data['user', 'replied_to', 'user'].edge_index
    
    data['adj_norm'] = compute_normalized_adjacency(n_users, n_items, data['train_pairs'], device, 
                                                    item_item_edges, user_user_edges, 
                                                    edge_weights=edge_weights, social_weight=args.social_weight)
    data['augmented_pairs'] = augmented_pairs


def _load_semantic_ids(args, pretrained_emb, device):
    """Generate Semantic IDs if requested."""
    if args.semantic_id_bits <= 0:
        return None
    if pretrained_emb is not None:
        print(f"\nGenerating Semantic IDs ({args.semantic_id_bits} stages)...")
        semantic_ids, _ = generate_semantic_ids(pretrained_emb, n_codebooks=args.semantic_id_bits, codebook_size=32)
        semantic_ids = semantic_ids.to(device)
        print(f"  Generated IDs shape: {semantic_ids.shape}")
        return semantic_ids
    else:
        print("\n  Warning: Cannot generate Semantic IDs without pretrained embeddings. Skipping.")
        return None


def _load_user_priors(args, n_users, device):
    """Load User Priors if requested."""
    if args.model != 'xsimgcl':
        return None
    prior_path = Path('data/processed/user_priors.pt')
    if prior_path.exists():
        print(f"\nLoading User Priors from {prior_path}...")
        priors = torch.load(prior_path, weights_only=False).to(device)
        if priors.shape[0] < n_users:
            print(f"  Padding User Priors: {priors.shape[0]} -> {n_users}")
            user_priors = torch.zeros((n_users, priors.shape[1]), device=device)
            user_priors[:priors.shape[0]] = priors
        else:
            user_priors = priors
        print(f"  Final Priors shape: {user_priors.shape}")
        return user_priors
    else:
        print(f"\n  Warning: User Priors not found at {prior_path}. Running without priors.")
        return None


def _load_reranker(args):
    """Load Re-ranker if requested."""
    if args.rerank == 'none':
        return None
    print("\nInitializing Re-ranker...")
    with open('data/processed/lightgcl_data.pkl', 'rb') as f:
        idx_data = pickle.load(f)
        idx2item = idx_data['idx2item']
    
    categories, n_cats = load_item_categories(idx2item)
    re_ranker = CalibratedReRanker(categories, alpha=0.5, lambda_mmr=0.5)
    print(f"  Loaded {n_cats} categories for {len(categories)} items.")
    return re_ranker


def _build_model(args, n_users, n_items, data, device, pretrained_emb, semantic_ids, user_priors):
    """Instantiate the chosen model and inject pretrained embeddings."""
    if args.model == 'lightgcl':
        model = LightGCLWrapper(n_users, n_items, embed_dim=args.hidden_dim, n_layers=args.n_layers, 
                                device=args.device, svd_q=args.svd_q, ssl_weight=args.ssl_weight, temp=args.temp)
        model.setup(data['train_pairs'], data.get('augmented_pairs', None))
        if pretrained_emb is not None:
            model.model.E_i_0.data.copy_(pretrained_emb)
            print("  Transferred embeddings to LightGCL (E_i_0)")
            
    elif args.model == 'lightgcn':
        model = LightGCN(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers, 
                         dropout=args.dropout).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to LightGCN")
             
    elif args.model == 'bigcf':
        model = BIGCF(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to BIGCF")

    elif args.model == 'ma-hcl':
        model = MAHCL(
            n_users=n_users, n_items=n_items,
            embedding_dim=args.hidden_dim, n_layers=args.n_layers,
            ssl_weight=args.ssl_weight, eps=args.eps, temp=args.temp,
            n_categories=data.get('n_categories', 0)
        ).to(device)
        if pretrained_emb is not None:
             model.item_emb.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to MA-HCL")
             
    elif args.model == 'ma_hgn':
        num_cats = 0
        if isinstance(data, dict):
            num_cats = data.get('n_categories', 0)
        elif hasattr(data, 'n_categories'):
            num_cats = data.n_categories
        
        model = MAHGN(n_users, n_items, args.hidden_dim, n_layers=args.n_layers, 
                      n_categories=num_cats, gnn_type=args.gnn_type,
                      cl_weight=args.ssl_weight, temp=args.temp).to(device)

    elif args.model == 'xsimgcl':
        model = XSimGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers,
                         ssl_weight=args.ssl_weight, temp=args.temp).to(device)
        if semantic_ids is not None:
            model.semantic_layer = SemanticEmbeddingLayer(args.semantic_id_bits, 32, args.hidden_dim).to(device)
            model.semantic_ids = semantic_ids
            print("  Initialized XSimGCL with Semantic ID Fusion (Pillar 1)")
            
        if user_priors is not None:
            model.user_prior_layer = UserPriorLayer(user_priors.shape[1], args.hidden_dim).to(device)
            model.user_priors = user_priors
            print("  Initialized XSimGCL with User Prior Fusion (Pillar 2)")
            
        if pretrained_emb is not None:
             model.item_embedding.weight.data.copy_(pretrained_emb)
             print("  Transferred embeddings to XSimGCL")

    else:
        # Default: SimGCL
        if args.model == 'simgcl':
            model = SimGCL(n_users, n_items, embedding_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
        
        if pretrained_emb is not None:
            model.item_embedding.weight.data.copy_(pretrained_emb)
            print(f"  Transferred embeddings to {args.model.upper()}")
    
    return model


def _save_and_print_results(args, model, n_users, n_items, best_metrics):
    """Save model checkpoint and print final metrics."""
    p = Path(args.data_path)
    if p.parent.name in ["strict_g1", "strict_g2", "strict_g3", "regular_g2", "enhanced_v1", "enhanced_v2"]:
        graph_name = p.parent.name
    else:
        graph_name = p.stem
        
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_path = f"models/{args.model}_{graph_name}_{timestamp}.pt"
    
    Path("models").mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': n_users,
        'n_items': n_items,
        'config': vars(args),
        'metrics': best_metrics
    }, save_path)

    print(f"\nModel saved: {save_path}")
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
            val = best_metrics.get(f'{m}@{k}', 0)
            values.append(f"{val:.6f}")
        print(row + " | ".join(values))
    
    print("-" * len(header))
    print(f"MRR: {best_metrics.get('mrr', 0):.6f}")
    if 'entropy' in best_metrics:
        print(f"Entropy: {best_metrics['entropy']:.6f}")
    
    if args.save_results:
        import json
        
        results_path = Path(args.save_results)
        if not results_path.parent.exists() or results_path.parent == Path('.'):
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            results_path = results_dir / results_path.name
        
        def convert(o):
            if isinstance(o, np.float32): return float(o)
            return o
            
        with open(results_path, 'w') as f:
            json.dump(best_metrics, f, default=convert)
        print(f"Saved metrics to {results_path}")


if __name__ == '__main__':
    main()
