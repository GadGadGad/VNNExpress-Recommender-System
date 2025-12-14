#!/usr/bin/env python3
"""
GNN Recommendation System - Full Pipeline
==========================================
Runs the complete pipeline from data processing to model evaluation.

Usage:
    python scripts/run_pipeline.py                    # Use config.yaml
    python scripts/run_pipeline.py --config my.yaml  # Custom config
    python scripts/run_pipeline.py --epochs 50       # Override epochs
"""
import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import json
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_step(name: str, cmd: list, verbose: bool = True) -> tuple:
    """Run a pipeline step and return (success, output)."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {name}")
        print(f"Error: {result.stderr[:500]}")
        return False, result.stderr
    
    print(f"✓ Completed: {name}")
    return True, result.stdout


def parse_metrics(output: str) -> dict:
    """Parse metrics from training script output."""
    metrics = {}
    
    # Parse Recall@K, NDCG@K format
    for k in [1, 5, 10]:
        match = re.search(rf"Recall@{k}:\s+([\d.]+)", output)
        metrics[f'recall@{k}'] = float(match.group(1)) if match else 0.0
        
        match = re.search(rf"NDCG@{k}:\s+([\d.]+)", output)
        metrics[f'ndcg@{k}'] = float(match.group(1)) if match else 0.0
    
    match = re.search(r"HitRate@10:\s+([\d.]+)", output)
    metrics['hitrate@10'] = float(match.group(1)) if match else 0.0
    
    match = re.search(r"MRR:\s+([\d.]+)", output)
    metrics['mrr'] = float(match.group(1)) if match else 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run GNN Recommendation Pipeline')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--embedding', choices=['random', 'tfidf', 'bert'], help='Override embedding type')
    parser.add_argument('--skip-data', action='store_true', help='Skip data processing steps')
    parser.add_argument('--skip-gnn', action='store_true', help='Skip GNN models')
    parser.add_argument('--skip-cf', action='store_true', help='Skip CF models')
    parser.add_argument('--skip-cl', action='store_true', help='Skip CL models')
    parser.add_argument('--models', nargs='+', help='Train specific models only')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.embedding:
        config['data']['embedding'] = args.embedding
    
    print("=" * 60)
    print("GNN RECOMMENDATION SYSTEM - FULL PIPELINE")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Embedding: {config['data']['embedding']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Min interactions: {config['data']['min_user_interactions']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # =========================================================================
    # STEP 1: Data Processing
    # =========================================================================
    if not args.skip_data and config['pipeline']['convert_data']:
        
        # Clear cache if requested
        if config['pipeline']['clear_cache']:
            cache_dir = Path(config['data']['processed_dir'])
            for cache_file in ['cf_cache.pt', 'lightgcl_cache.pt']:
                cache_path = cache_dir / cache_file
                if cache_path.exists():
                    os.remove(cache_path)
                    print(f"  Cleared {cache_file}")
        
        # Generate BERT embeddings if needed
        if config['data']['embedding'] == 'bert' and config['pipeline']['generate_embeddings']:
            bert_model = config['data'].get('bert_model', 'vinai/phobert-base')
            success, _ = run_step(
                f"Generate BERT Embeddings ({bert_model})",
                ["python", "src/data/generate_embeddings.py", "--model", bert_model]
            )
            if not success:
                print("⚠️ Warning: BERT embedding generation failed")
        
        # Convert data to GNN format
        cmd = [
            "python", "src/data/convert_to_gnn.py",
            "--output", config['data']['processed_dir'],
            "--graph-type", "hetero",
            "--min-user-interactions", str(config['data']['min_user_interactions']),
            "--min-article-interactions", str(config['data']['min_article_interactions'])
        ]
        
        if config['data']['embedding'] == 'bert':
            cmd.append("--use-phobert")  # uses bert embeddings file
        elif config['data']['embedding'] == 'tfidf':
            cmd.extend(["--add-text-features", "--text-max-features", str(config['data']['tfidf_max_features'])])
        
        success, _ = run_step("Convert Data to GNN Format", cmd)
        if not success:
            print("❌ Data conversion failed, exiting")
            return
    
    # =========================================================================
    # STEP 2: Train Models
    # =========================================================================
    if config['pipeline']['train_models']:
        
        # Determine which models to train
        models_to_train = []
        
        if args.models:
            # Use explicitly specified models
            models_to_train = [(m, 'USER') for m in args.models]
        else:
            # Use config
            if config['models']['gnn']['enabled'] and not args.skip_gnn:
                for m in config['models']['gnn']['models']:
                    models_to_train.append((m, 'GNN'))
            
            if config['models']['cf']['enabled'] and not args.skip_cf:
                for m in config['models']['cf']['models']:
                    models_to_train.append((m, 'CF'))
            
            if config['models']['cl']['enabled'] and not args.skip_cl:
                for m in config['models']['cl']['models']:
                    models_to_train.append((m, 'CL'))
        
        # Train each model
        for model_name, model_type in models_to_train:
            print(f"\n>>> Training {model_name.upper()} ({model_type})...")
            
            # GNN models use different training script
            if model_type == 'GNN':
                cmd = [
                    "python", "src/training/train_gnn_baseline.py",
                    "--model", model_name,
                    "--epochs", str(config['training']['epochs']),
                    "--data-path", f"{config['data']['processed_dir']}/full_hetero_graph.pt"
                ]
            else:
                # CF and CL models
                cmd = [
                    "python", "scripts/train_cf_models.py",
                    "--model", model_name,
                    "--epochs", str(config['training']['epochs']),
                    "--batch-size", str(config['training']['batch_size']),
                    "--lr", str(config['training']['learning_rate']),
                    "--hidden-dim", str(config['training']['hidden_dim']),
                    "--device", config['training']['device']
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            output = result.stdout + result.stderr
            
            if result.returncode != 0:
                print(f"  ❌ Failed: {model_name}")
                continue
            
            # Parse metrics
            metrics = parse_metrics(output)
            
            results.append({
                'Model': model_name.upper(),
                'Type': model_type,
                'R@1': f"{metrics.get('recall@1', 0):.3f}",
                'R@5': f"{metrics.get('recall@5', 0):.3f}",
                'R@10': f"{metrics.get('recall@10', 0):.3f}",
                'N@10': f"{metrics.get('ndcg@10', 0):.3f}",
                'HR@10': f"{metrics.get('hitrate@10', 0):.3f}",
                'MRR': f"{metrics.get('mrr', 0):.3f}"
            })
            
            print(f"  ✓ {model_name.upper()}: R@10={metrics.get('recall@10', 0):.3f}")
    
    # =========================================================================
    # STEP 3: Generate Report
    # =========================================================================
    if config['pipeline']['generate_report'] and results:
        print("\n" + "=" * 70)
        print("FINAL RESULTS: MODEL COMPARISON")
        print("=" * 70)
        
        # Sort by R@10
        results_sorted = sorted(results, key=lambda x: float(x['R@10']), reverse=True)
        
        print(tabulate(results_sorted, headers='keys', tablefmt='grid'))
        
        # Save results
        if config['evaluation']['save_results']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = Path(config['evaluation']['results_dir']) / f"results_{timestamp}.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump({
                    'config': config,
                    'results': results_sorted,
                    'timestamp': timestamp
                }, f, indent=2)
            
            print(f"\nResults saved to: {results_path}")
    
    print(f"\n{'='*60}")
    print(f"Pipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
