#!/usr/bin/env python3
"""
End-to-End Pipeline for Article Recommendation System
=====================================================
Combines: Data Preparation → EDA → Training → Experiments → Comparison

Usage:
    # Full pipeline
    python scripts/run_all.py --full
    
    # Quick run (skip EDA, 10 epochs)
    python scripts/run_all.py --quick
    
    # Custom run
    python scripts/run_all.py --epochs 50 --min-int 2,5 --models sgl,ncl --embedding phobert
"""
import subprocess
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, desc, cwd=None):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"[WARNING] Command failed with code {result.returncode}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Pipeline for Article Recommendation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_all.py --full
  python scripts/run_all.py --quick
  python scripts/run_all.py --models sgl,ncl
  python scripts/run_all.py --experiments
        """
    )
    
    # Presets
    parser.add_argument('--full', action='store_true', 
                        help='Full pipeline: merge, EDA, train all models, 30 epochs')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: skip EDA, 10 epochs, top 3 models only')
    
    # Data options
    parser.add_argument('--merge', action='store_true',
                        help='Merge data from crawlers/data_small_* folders')
    parser.add_argument('--eda', action='store_true',
                        help='Run exploratory data analysis')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--embedding', choices=['random', 'tfidf', 'phobert'], 
                        default='phobert', help='Embedding type (default: phobert)')
    parser.add_argument('--min-int', type=str, default='5',
                        help='Min interactions, comma-separated for experiments (default: 5)')
    parser.add_argument('--models', type=str, default='all',
                        help='Models to train: all, or comma-separated (sgl,ncl,simgcl,...)')
    
    # Experiment options
    parser.add_argument('--experiments', action='store_true',
                        help='Run experiments with multiple min_interaction values')
    parser.add_argument('--compare', action='store_true',
                        help='Run train_and_compare_all.py')
    
    # Skip options
    parser.add_argument('--skip-gnn', action='store_true', help='Skip GNN models')
    parser.add_argument('--skip-cf', action='store_true', help='Skip CF models')
    parser.add_argument('--skip-cl', action='store_true', help='Skip CL models')
    
    args = parser.parse_args()
    
    # Apply presets
    if args.full:
        args.merge = True
        args.eda = True
        args.compare = True
        args.epochs = 30
        args.models = 'all'
    
    if args.quick:
        args.merge = False
        args.eda = False
        args.compare = True
        args.epochs = 10
        args.models = 'sgl,ncl,simgcl'
        args.skip_gnn = True
        args.skip_cf = True
    
    print("="*70)
    print("ARTICLE RECOMMENDATION SYSTEM - END-TO-END PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: epochs={args.epochs}, embedding={args.embedding}, min_int={args.min_int}")
    print(f"Models: {args.models}")
    
    # Ensure project root context
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Step 1: Merge Data
    if args.merge:
        run_cmd(
            ["python", "scripts/merge_data.py"],
            "STEP 1: Merging crawler data"
        )
    
    # Step 2: EDA
    if args.eda:
        run_cmd(
            ["python", "src/eda/eda_crawled_data.py"],
            "STEP 2a: Running Crawled Data EDA"
        )
        run_cmd(
            ["python", "src/eda/eda_graph_data.py", "-d", "data/processed/full_hetero_graph.pt", "-o", "plots/graph"],
            "STEP 2b: Running Graph EDA"
        )
    
    # Step 3: Run Experiments (multiple min_interactions)
    if args.experiments:
        min_ints = args.min_int
        run_cmd(
            ["python", "scripts/run_experiments.py",
             "--epochs", str(args.epochs),
             "--min-interactions", min_ints,
             "--models", args.models],
            f"STEP 3: Running Experiments (min_interactions={min_ints})"
        )
    
    # Step 4: Train and Compare All
    if args.compare:
        cmd = [
            "python", "scripts/train_and_compare_all.py",
            "--epochs", str(args.epochs),
            "--embedding", args.embedding,
            "--min-interactions", args.min_int.split(',')[0],
            "--prepare"
        ]
        if args.skip_gnn:
            cmd.append("--skip-gnn")
        if args.skip_cf:
            cmd.append("--skip-cf")
        if args.skip_cl:
            cmd.append("--skip-cl")
        
        run_cmd(cmd, "STEP 4: Training and Comparing All Models")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print("  - EDA plots: plots/")
    print("  - Graph EDA: plots/graph/")
    print("  - Model results: models/")
    print("  - Comparison: models/comparison_results_*.json")


if __name__ == '__main__':
    main()
