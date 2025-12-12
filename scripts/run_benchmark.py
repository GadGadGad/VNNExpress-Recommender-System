import subprocess
import re
import sys
from tabulate import tabulate  # Assuming user might not have it, I'll fallback to simple formatting if needed, but standard python doesn't include it. I'll write a simple formatter.

def run_evaluation(min_interactions, model_path, graph_path, data_dir):
    print(f"Running evaluation for Min Interactions = {min_interactions} (Model: {model_path})...")
    
    cmd = [
        "python", "src/evaluation/evaluate_filtered.py",
        "--graph", graph_path,
        "--data-dir", data_dir,
        "--min", str(min_interactions),
        "--model", model_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation for min={min_interactions}")
        print(result.stderr)
        return None, None
        
    # extract metrics
    output = result.stdout
    metrics = {}
    for m in ["Recall", "NDCG", "Precision", "HitRate", "MRR"]:
        match = re.search(rf"{m}@10:\s+([\d.]+)", output)
        metrics[m.lower()] = float(match.group(1)) * 100 if match else 0.0
    
    return metrics

def main():
    thresholds = [1, 2, 3, 5, 10]
    
    # Define Configurations
    configs = {
        "baseline": {
            "model": "models/baseline/lightgcn.pt", # We need to find exact name or use glob in python
            "graph": "data/processed_baseline/full_hetero_graph.pt",
            "data": "data/processed_baseline"
        },
        "social": {
            "model": "models/social/lightgcn.pt",
            "graph": "data/processed_social/full_hetero_graph.pt",
            "data": "data/processed_social"
        }
    }
    
    # Helper to find latest model in dir if specific file not known
    import glob
    import os
    for key in configs:
        # Assuming training script saves as lightgcn_TIMESTAMP.pt in the save-dir
        # But wait, save-dir arg in train script just sets WHERE to save.
        # It saves as lightgcn_YYYYMMDD_HHMMSS.pt
        # We need to find the latest file in that directory.
        files = glob.glob(f"{configs[key]['model'].replace('/lightgcn.pt', '')}/lightgcn_*.pt")
        if files:
            configs[key]['model'] = max(files, key=os.path.getctime)
        else:
            print(f"Warning: No model found for {key} in {configs[key]['model']}")

    results = []
    
    print("Starting Comprehensive Benchmark (Live Training)...")
    print("=" * 60)
    
    for k in thresholds:
        # Run Social
        m_social = run_evaluation(k, configs["social"]["model"], configs["social"]["graph"], configs["social"]["data"])
        
        # Run Baseline
        m_base = run_evaluation(k, configs["baseline"]["model"], configs["baseline"]["graph"], configs["baseline"]["data"])
        
        row = {"min": k}
        for key in ["recall", "ndcg", "precision", "hitrate", "mrr"]:
            val_social = m_social.get(key, 0.0)
            val_base = m_base.get(key, 0.0)
            
            row[f"social_{key}"] = f"{val_social:.2f}%" if key != "mrr" else f"{val_social:.4f}"
            row[f"base_{key}"] = f"{val_base:.2f}%" if key != "mrr" else f"{val_base:.4f}"
            
            diff = val_social - val_base
            row[f"diff_{key}"] = f"{diff:+.2f}%" if key != "mrr" else f"{diff:+.4f}"
            
        results.append(row)
        
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK REPORT: Social Graph vs Baseline")
    print("=" * 80)
    
    # Print Table
    headers = ["Segment", "Metric", "Original (No Social)", "Social Graph", "Impact"]
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<20} | {headers[3]:<15} | {headers[4]:<10}")
    print("-" * 95)
    
    labels = {
        1: "All Users (Min 1)",
        2: "Min 2",
        3: "Active (Min 3)",
        5: "Very Active (Min 5)",
        10: "Power Users (Min 10)"
    }
    
    for row in results:
        label = labels.get(row['min'], f"Min {row['min']}")
        
        for metric_name, key in [
            ("Recall@10", "recall"),
            ("HitRate@10", "hitrate"),
            ("Precision@10", "precision"),
            ("NDCG@10", "ndcg"),
            ("MRR@10", "mrr")
        ]:
            if key == "recall":
                prefix = label
            else:
                prefix = ""
                
            print(f"{prefix:<25} | {metric_name:<12} | {row[f'base_{key}']:<20} | {row[f'social_{key}']:<15} | {row[f'diff_{key}']:<10}")
        print("-" * 95)

if __name__ == "__main__":
    main()
