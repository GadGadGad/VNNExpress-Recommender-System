#!/usr/bin/env python3
"""
Interactive Pipeline for Article Recommendation
===============================================
Guides the user through configuration and executes the full pipeline:
1. Embeddings Generation (if needed)
2. Data Conversion (with optional KNN)
3. Training & Evaluation (looping through min_interaction settings)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_input(prompt, default=None):
    """Get user input with optional default."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def get_yes_no(prompt, default='y'):
    """Get yes/no input."""
    while True:
        user_input = get_input(prompt, default).lower()
        if user_input in ['y', 'yes']:
            return True
        elif user_input in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'.")

def run_cmd(cmd, desc):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f">>> {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("\n" + "="*60)
    print("INTERACTIVE ARTICLE RECOMMENDATION PIPELINE")
    print("="*60)
    
    # 1. Configuration
    print("\n--- Configuration ---")
    
    # Embedding
    embedding_map = {'1': 'random', '2': 'tfidf', '3': 'phobert'}
    print("\nSelect Embedding Type:")
    print("1. Random (Fastest)")
    print("2. TF-IDF (Text features)")
    print("3. PhoBERT (Pre-computed embeddings)")
    emb_choice = get_input("Choice", "3")
    embedding = embedding_map.get(emb_choice, 'phobert')
    
    # Data Source Selection
    print("\nSelect Data Source:")
    print("1. Merged (All categories - data/raw)")
    
    # Detect available individual sources
    from glob import glob
    category_dirs = sorted(glob("crawlers/data_small_*"))
    categories = [Path(d).name.replace("data_small_", "") for d in category_dirs]
    
    for i, cat in enumerate(categories, 2):
        print(f"{i}. {cat.title()} only")
    
    data_choice = get_input("Choice", "1")
    
    if data_choice == "1":
        data_source = "data/raw"
        data_source_name = "merged"
    else:
        try:
            cat_idx = int(data_choice) - 2
            if 0 <= cat_idx < len(categories):
                data_source = category_dirs[cat_idx]
                data_source_name = categories[cat_idx]
            else:
                print(f"Invalid choice, using merged data.")
                data_source = "data/raw"
                data_source_name = "merged"
        except ValueError:
            data_source = "data/raw"
            data_source_name = "merged"
    
    print(f"Selected Data Source: {data_source_name.upper()} ({data_source})")

    
    # KNN Enrichment
    knn_enriched = get_yes_no("\nEnable KNN Enrichment for Articles?", "n")
    knn_k = 10
    if knn_enriched:
        knn_k = int(get_input("Number of neighbors (K)", "10"))
    
    # Experiments
    epoch_default = "30"
    epochs = get_input("\nNumber of Epochs", epoch_default)
    
    min_ints_input = get_input("\nMin Interactions to test (comma separated)", "2,5")
    min_ints = [x.strip() for x in min_ints_input.split(',')]
    
    models_input = get_input("\nModels to train (comma separated or 'all')", "all")
    
    # Filter Mode
    filter_mode_input = get_input("\nFilter Mode (active/strict): \n  - 'active' (Recommended): Keeps users with >= N interactions initially (Static)\n  - 'strict': Recursive removal (K-Core) - May empty data", "active")
    filter_mode = 'static' if 'active' in filter_mode_input.lower() else 'iterative'
    print(f"Selected Filter Mode: {filter_mode.upper()}")
    
    # Negative Sampling Ratio
    neg_ratio = int(get_input("\nNegative Sampling Ratio (per positive)", "4"))
    print(f"Negative Ratio: {neg_ratio}")

    # Advanced Options (Content-Enriched & Evaluation)
    print("\n--- Advanced Evaluation Strategies ---")
    use_content_enriched = get_yes_no("Enable Content-Enriched GNN Initialization? (Uses PhoBERT to init GNN)", "n")
    if use_content_enriched:
        print("  -> Will enforce 'phobert' embedding generation and initialize LightGCL with it.")
        embedding = 'phobert' # Enforce
    
    run_advanced_eval = get_yes_no("Run Advanced Evaluation? (Switching Models / CB Ensembling analysis)", "n")

    
    # 2. Execution
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # # Generate Embeddings if PhoBERT (check if exists is handled by script, but we force run if selected?)
    # # Usually better to check or just run it. generate_embeddings.py caches result? 
    # # Let's assume user wants to run it if they selected it, or we can rely on convert_to_gnn checking.
    # # But convert_to_gnn fails if missing. So let's run generation if PhoBERT is chosen.
    # if embedding == 'phobert':
    #     if get_yes_no("\nRun PhoBERT embedding generation first?", "n"):
    #          run_cmd(["python", "src/data/generate_embeddings.py"], "Generating PhoBERT Embeddings")

    # Loop through experiments
    for min_int in min_ints:
        print(f"\n\n{'#'*60}")
        print(f"RUNNING EXPERIMENT: Min Interactions = {min_int}")
        print(f"{'#'*60}")
        
        # Construct conversion command
        # We use train_and_compare_all.py --prepare to handle conversion + training
        # BUT we need to pass filter mode. 
        # I need to update train_and_compare_all.py to accept --filter-mode OR pass it via config.
        # Passing via config is cleaner for now since I am editing it dynamically anyway.
        
        # Let's update config.yaml using a helper script or just python code here.
        # Since I am in python, I can rewrite it.
        import yaml
        config_path = "config.yaml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update values
            if 'data' not in config: config['data'] = {}
            config['data']['knn_enriched'] = knn_enriched
            config['data']['knn_k'] = knn_k
            config['data']['filter_mode'] = filter_mode
            config['data']['source_dir'] = data_source  # NEW: Data source path
            
            # Write back
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"\n[Config] Updated config.yaml: knn={knn_enriched}, k={knn_k}, filter={filter_mode}, source={data_source}")
            
        except Exception as e:
            print(f"[WARN] Failed to update user config.yaml manually: {e}")
            print(" proceeding, but KNN/Filter settings might be wrong if not passed via CLI...")

        # Run Train & Compare
        cmd = [
            "python", "scripts/train_and_compare_all.py",
            "--epochs", str(epochs),
            "--embedding", embedding,
            "--min-interactions", str(min_int),
            "--data-source", data_source,  # NEW: Pass data source
            "--neg-ratio", str(neg_ratio),  # Negative sampling ratio
            "--prepare" # Always prepare data for each fresh min_int run
        ]
        
        if use_content_enriched:
             # Assume path based on train_and_compare_all logic
             pretrained_path = "data/processed/phobert_embeddings.pt"
             cmd.extend(["--pretrained-path", pretrained_path])


        if models_input.lower() != 'all':
            # Parse model selection and determine which types to skip
            # Model types: GNN (sage,gcn,gat,lightgcn), CF (ngcf,simplex,directau), CL (sgl,simgcl,ncl,lightgcl), CB (tfidf)
            selected = [m.strip().lower() for m in models_input.split(',')]
            
            # Define model groups
            gnn_models = {'sage', 'gcn', 'gat', 'lightgcn', 'gnn'}
            cf_models = {'ngcf', 'simplex', 'directau', 'cf'}
            cl_models = {'sgl', 'simgcl', 'ncl', 'lightgcl', 'cl'}
            cb_models = {'tfidf', 'cb', 'content'}
            
            # Check if user wants each type
            wants_gnn = any(m in gnn_models for m in selected)
            wants_cf = any(m in cf_models for m in selected)
            wants_cl = any(m in cl_models for m in selected)
            wants_cb = any(m in cb_models for m in selected)
            
            # Add skip flags for types not wanted
            if not wants_gnn:
                cmd.append("--skip-gnn")
            if not wants_cf:
                cmd.append("--skip-cf")
            if not wants_cl:
                cmd.append("--skip-cl")
            if not wants_cb:
                cmd.append("--skip-cb")
            
            print(f"[Models] Running: GNN={wants_gnn}, CF={wants_cf}, CL={wants_cl}, CB={wants_cb}")
        
        run_cmd(cmd, f"Training & Evaluating (Min Int: {min_int})")
        
        if run_advanced_eval:
             print("\n>>> Running Advanced Strategy Evaluation (Switching/Ensemble)...")
             subprocess.run(["python", "scripts/eval_strategies.py"], check=False)


    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
