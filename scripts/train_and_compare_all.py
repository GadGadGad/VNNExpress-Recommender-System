#!/usr/bin/env python3
"""
Train and Compare All Models
Trains all available models and outputs a comparison table.
"""
import subprocess
import argparse
import subprocess
import time
import json
import os
from tabulate import tabulate
from datetime import datetime
from tabulate import tabulate
import subprocess
import json

def run_cb_ablation(args):
    from tabulate import tabulate
    print("\n" + "="*70)
    print("RUNNING CONTENT-BASED ABLATION STUDY")
    print("="*70 + "\n")
    
    study_results = []
    
    # 1. PhoBERT (Semantic Embeddings)
    print("\n>>> Strategy 1: PhoBERT (Deep Semantic)...")
    try:
        # We need to run phobert specifically
        # run_cb_model handles metrics parsing
        metrics = run_cb_model("phobert")
        study_results.append({
            'Model': 'PhoBERT',
            'Type': 'Deep Semantic',
            'R@10': f"{metrics.get('recall@10', 0):.4f}",
            'N@10': f"{metrics.get('ndcg@10', 0):.4f}",
            'Details': 'Pre-trained Vietnamese BERT'
        })
    except Exception as e:
        print(f"PhoBERT failed: {e}")

    # 2. TF-IDF (Keyword Matching)
    print("\n>>> Strategy 2: TF-IDF (Keyword)...")
    try:
        metrics = run_vnlp_model("tfidf")
        study_results.append({
             'Model': 'TF-IDF',
             'Type': 'Keyword',
             'R@10': f"{metrics.get('recall@10', 0):.4f}",
             'N@10': f"{metrics.get('ndcg@10', 0):.4f}",
             'Details': 'Traditional Vector Space'
        })
    except Exception as e:
        print(f"TF-IDF failed: {e}")
        
    # 3. SimCSE (Contrastive Sentence Embeddings)
    print("\n>>> Strategy 3: SimCSE (Contrastive Semantic)...")
    try:
        metrics = run_cb_model("simcse")
        study_results.append({
             'Model': 'SimCSE',
             'Type': 'Contrastive',
             'R@10': f"{metrics.get('recall@10', 0):.4f}",
             'N@10': f"{metrics.get('ndcg@10', 0):.4f}",
             'Details': 'Optimized for similarity'
        })
    except Exception as e:
        print(f"SimCSE failed: {e}")

    print("\n" + "="*60)
    print("CONTENT-BASED ABLATION RESULTS")
    print("="*60)
    print(tabulate(study_results, headers="keys", tablefmt="grid"))


def run_graph_ablation(args):
    from tabulate import tabulate
    print("\n" + "="*70)
    print("RUNNING GRAPH TYPE ABLATION STUDY (Hetero vs Bipartite)")
    print("="*70 + "\n")
    
    study_results = []
    
    # Define scenarios
    scenarios = [
        {'name': 'Full Hetero', 'no_aux': False, 'label': 'Hetero'},
        {'name': 'Bipartite Only', 'no_aux': True, 'label': 'Bipartite'}
    ]
    
    for scenario in scenarios:
        print(f"\n>>> Scenerio: {scenario['name']}...")
        
        # 1. Generate Graph
        cmd = [
             "python", "src/data/convert_to_gnn.py",
             "--output", "data/processed_phobert",
             "--graph-type", "hetero",
             "--min-user-interactions", str(args.min_interactions),
             "--min-article-interactions", str(args.min_interactions)
        ]
        if args.embedding == 'phobert': cmd.append("--use-phobert")
        elif args.embedding == 'tfidf': cmd.extend(["--add-text-features", "--text-max-features", "500"])
        
        if scenario['no_aux']:
            cmd.append("--no-aux-edges")
            
        print("   Generating graph data...")
        subprocess.run(cmd, check=True)
        
        # Determine filename based on flag (see convert_to_gnn.py changes)
        filename = "full_hetero_graph_no_aux.pt" if scenario['no_aux'] else "full_hetero_graph.pt"
        data_path = f"data/processed_phobert/{filename}"
        
        # 2. Run GNN Models
        gnn_models = ['sage', 'gat', 'gcn'] 
        
        for model in gnn_models:
             metrics = run_gnn_model_custom(model, args.epochs, data_path)
             study_results.append({
                 'Model': model.upper(),
                 'Graph': scenario['label'],
                 'R@10': f"{metrics.get('recall@10', 0):.4f}",
                 'N@10': f"{metrics.get('ndcg@10', 0):.4f}",
                 'HR@10': f"{metrics.get('hitrate@10', 0):.4f}",
                 'MRR': f"{metrics.get('mrr', 0):.4f}"
             })
             
    print("\n" + "="*60)
    print("GRAPH ABLATION RESULTS")
    print("="*60)
    print(tabulate(study_results, headers="keys", tablefmt="grid"))

def run_gnn_model_custom(model_name, epochs, data_path):
    # Specialized runner allowing custom data path
     fd, temp_path = tempfile.mkstemp(suffix='.json')
     os.close(fd)
     
     cmd = [
        "python", "src/training/train_gnn_baseline.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--data-path", data_path,
        "--save-results", temp_path
     ]
     subprocess.run(cmd, check=False)
     
     metrics = {}
     if os.path.exists(temp_path):
         try:
             with open(temp_path, 'r') as f:
                 raw = json.load(f)
                 for k, v in raw.items(): metrics[k.lower()] = v
         except: pass
         finally: os.remove(temp_path)
     return metrics

def run_cf_model(model_name, epochs=50, batch_size=2048):
    """Train a CF model and return metrics."""
    print(f"\n>>> Training {model_name.upper()}...")
    
    # Create temp file for results
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    cmd = [
        "python", "scripts/train_cf_models.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", "cpu",
        "--save-results", temp_path
    ]
    
    # Run with output visible (tqdm will show)
    # capture_output=False allows stdout/stderr to flow
    subprocess.run(cmd, check=False)
    
    # Read results from temp file
    metrics = {}
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"Error reading metrics for {model_name}: {e}")
        finally:
            os.remove(temp_path)
    else:
        print(f"Warning: No metrics file generated for {model_name}")
    
    return metrics


def run_gnn_model(model_name, epochs=50):
    """Train a GNN model and return metrics."""
    print(f"\n>>> Training {model_name.upper()}...")
    
    # Create temp file for results
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    cmd = [
        "python", "src/training/train_gnn_baseline.py",
        "--model", model_name,
        "--epochs", str(epochs),
        "--data-path", "data/processed/full_hetero_graph.pt",
        "--save-results", temp_path
    ]
    
    # Run with output visible (tqdm will show)
    subprocess.run(cmd, check=False)
    
    metrics = {}
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r') as f:
                raw_metrics = json.load(f)
                # Map keys to standard lowercase format if needed
                # GNN script usually outputs capitalized keys like 'Recall@10'
                for k, v in raw_metrics.items():
                    metrics[k.lower()] = v
        except Exception as e:
            print(f"Error reading metrics for {model_name}: {e}")
        finally:
            os.remove(temp_path)
    else:
        print(f"Warning: No metrics file generated for {model_name}")
    
    return metrics


def run_lightgcl(epochs=50):
    """Train LightGCL and return metrics."""
    print(f"\n>>> Training LIGHTGCL...")
    
    # Note: LightGCL is now covered by run_cf_model in the main loop logic,
    # but kept here for backward compatibility or if referenced elsewhere.
    # We will just wrapper call run_cf_model
    return run_cf_model('lightgcl', epochs)


def run_cb_model(model_name, data_path='data'):
    """Train a content-based model and return metrics."""
    print(f"\n>>> Training {model_name.upper()}...")
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    cmd = [
        "python", "scripts/run_content_based.py",
        "--model", model_name,
        "--data_path", data_path,
        "--device", "cpu",
        "--save-results", temp_path
    ]
    
    # Run with output visible (tqdm will show)
    subprocess.run(cmd, check=False)
    
    metrics = {}
    if os.path.exists(temp_path):
        try:
            with open(temp_path, 'r') as f:
                raw_metrics = json.load(f)
                for k, v in raw_metrics.items():
                    metrics[k.lower()] = v
        except Exception as e:
            print(f"Error reading metrics for {model_name}: {e}")
        finally:
            os.remove(temp_path)
    else:
        print(f"Warning: No metrics file generated for {model_name}")
        
    return metrics


def run_vnlp_model(model_name, data_path='data'):
    """Train a Vietnamese NLP model (TF-IDF, BM25, Word2Vec) and return metrics."""
    print(f"\n>>> Training {model_name.upper()}...")
    
    cmd = [
        "python", "scripts/run_vietnamese_nlp.py",
        "--method", model_name,
        "--data_path", data_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    import re
    metrics = {}
    
    # Parse metrics from VNLP output (format: "Recall:  @10=0.0504")
    for k in [1, 5, 10]:
        match = re.search(rf"Recall[:\s]*@{k}[=\s]+(\d+\.?\d*)", output)
        metrics[f'recall@{k}'] = float(match.group(1)) if match else 0.0
        
        match = re.search(rf"NDCG[:\s]*@{k}[=\s]+(\d+\.?\d*)", output)
        metrics[f'ndcg@{k}'] = float(match.group(1)) if match else 0.0
        
        match = re.search(rf"HR[:\s]*@{k}[=\s]+(\d+\.?\d*)", output)
        metrics[f'hitrate@{k}'] = float(match.group(1)) if match else 0.0
    
    # MRR not in VNLP output
    metrics['mrr'] = 0.0
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train and Compare All Models')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per model')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size for CF models')
    parser.add_argument('--embedding', choices=['random', 'tfidf', 'phobert'], default='random',
                        help='Embedding type for articles: random, tfidf, or phobert')
    parser.add_argument('--min-interactions', type=int, default=2, 
                        help='Minimum interactions filter (1=all, 3=active users)')
    parser.add_argument('--prepare', action='store_true', 
                        help='Run data preparation before training')
    parser.add_argument('--skip-gnn', action='store_true', help='Skip GNN models (SAGE, GCN, GAT, LightGCN)')
    parser.add_argument('--skip-cf', action='store_true', help='Skip CF models (NGCF, SimpleX, DirectAU)')
    parser.add_argument('--skip-cl', action='store_true', help='Skip CL models (SGL, SimGCL, NCL, LightGCL)')
    parser.add_argument('--skip-cb', action='store_true', help='Skip CB models (TF-IDF, PhoBERT)')
    parser.add_argument('--skip-vnlp', action='store_true', help='Skip Vietnamese NLP models (TF-IDF, BM25, Word2Vec)')
    parser.add_argument('--force', action='store_true', help='Delete old models first')
    parser.add_argument('--compare-graphs', action='store_true', help='Run ablation study: Hetero vs Bipartite GNN')
    parser.add_argument('--compare-cb', action='store_true', help='Run ablation study: Content-Based Strategies')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAINING AND COMPARING ALL RECOMMENDATION MODELS")
    print("=" * 70)
    print(f"Epochs: {args.epochs} | Embedding: {args.embedding} | Min interactions: {args.min_interactions}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Cleanup if requested
    # Cleanup if requested
    if args.force:
        print("\n>>> CLEANUP: Removing old model checkpoints...")
        import glob
        import os # Re-import here just to be safe, though top-level import is better.
        # Remove pt files in models/
        # Remove pt files in models/
        files = glob.glob("models/*.pt")
        for f in files:
            try:
                os.remove(f)
                print(f"  Deleted {f}")
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        
        # Remove json results if any (though usually temp)
        # But maybe we want to clear logs? 
        print("  Cleanup complete.")

    # Data preparation (optional)
    if args.prepare:
        # Clear old caches
        import os
        for cache_file in ['cf_cache.pt', 'lightgcl_cache.pt']:
            cache_path = f"data/processed_phobert/{cache_file}"
            if os.path.exists(cache_path):
                os.remove(cache_path)
                print(f"  Cleared {cache_file}")
        
        # Generate PhoBERT embeddings if needed
        if args.embedding == 'phobert':
            print("\n>>> STEP 1: Generating PhoBERT embeddings...")
            result = subprocess.run(
                ["python", "src/data/generate_embeddings.py"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  Warning: {result.stderr}")
            else:
                print("  Done!")
        
        # Convert data with appropriate embedding
        print(f"\n>>> STEP 2: Converting data ({args.embedding} embeddings)...")
        cmd = [
            "python", "src/data/convert_to_gnn.py",
            "--output", "data/processed_phobert",
            "--graph-type", "hetero",
            "--min-user-interactions", str(args.min_interactions),
            "--min-article-interactions", str(args.min_interactions)
        ]
        
        if args.embedding == 'phobert':
            cmd.append("--use-phobert")
        elif args.embedding == 'tfidf':
            cmd.extend(["--add-text-features", "--text-max-features", "500"])
        # random = default (no extra flags)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: {result.stderr[:500]}")
        else:
            print("  Done!")
            
    if args.compare_graphs:
        run_graph_ablation(args)
        return
        
    if args.compare_cb:
        run_cb_ablation(args)
        return

    results = []
    
    def add_result(model_name, model_type, metrics):
        results.append({
            'Model': model_name.upper(),
            'Type': model_type,
            'R@1': f"{metrics.get('recall@1', 0):.3f}",
            'R@5': f"{metrics.get('recall@5', 0):.3f}",
            'R@10': f"{metrics.get('recall@10', 0):.3f}",
            'N@1': f"{metrics.get('ndcg@1', 0):.3f}",
            'N@5': f"{metrics.get('ndcg@5', 0):.3f}",
            'N@10': f"{metrics.get('ndcg@10', 0):.3f}",
            'HR@1': f"{metrics.get('hitrate@1', 0):.3f}",
            'HR@5': f"{metrics.get('hitrate@5', 0):.3f}",
            'HR@10': f"{metrics.get('hitrate@10', 0):.3f}",
            'MRR': f"{metrics.get('mrr', 0):.3f}"
        })
    
    # GNN Models (SAGE, GCN, GAT, LightGCN)
    if not args.skip_gnn:
        for model in ['sage', 'gcn', 'gat', 'lightgcn']:
            try:
                metrics = run_gnn_model(model, args.epochs)
                add_result(model, 'GNN', metrics)
            except Exception as e:
                print(f"  Error training {model}: {e}")
    
    # CF Models (NGCF, SimpleX, DirectAU)
    if not args.skip_cf:
        for model in ['ngcf', 'simplex', 'directau']:
            try:
                metrics = run_cf_model(model, args.epochs, args.batch_size)
                add_result(model, 'CF', metrics)
            except Exception as e:
                print(f"  Error training {model}: {e}")
    
    # Contrastive Learning Models (SGL, SimGCL, NCL, XSimGCL, LightGCL)
    if not args.skip_cl:
        for model in ['sgl', 'simgcl', 'ncl', 'xsimgcl', 'lightgcl']:
            try:
                metrics = run_cf_model(model, args.epochs, args.batch_size)
                add_result(model, 'CL', metrics)
            except Exception as e:
                print(f"  Error training {model}: {e}")
    
    # Content-Based Models (PhoBERT, Hybrid, SimCSE)
    if not args.skip_cb:
        for model in ['phobert', 'hybrid', 'simcse']:
            try:
                metrics = run_cb_model(model)
                add_result(model, 'CB', metrics)
            except Exception as e:
                print(f"  Error training {model}: {e}")
    
    # Vietnamese NLP Models (TF-IDF, BM25, Word2Vec)
    if not args.skip_vnlp:
        for model in ['tfidf', 'bm25', 'word2vec']:
            try:
                metrics = run_vnlp_model(model)
                add_result(model, 'VNLP', metrics)
            except Exception as e:
                print(f"  Error training {model}: {e}")
    
    # Print results table
    print("\n" + "=" * 70)
    print("FINAL RESULTS: MODEL COMPARISON")
    print("=" * 70)
    
    # Sort by R@10 descending
    results_sorted = sorted(results, key=lambda x: float(x['R@10']), reverse=True)
    
    print(tabulate(results_sorted, headers='keys', tablefmt='grid'))
    
    # Save results
    os.makedirs("models", exist_ok=True)
    results_path = f"models/comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_path, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
