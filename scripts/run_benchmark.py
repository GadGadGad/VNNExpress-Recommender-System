import subprocess
import re
import sys
from tabulate import tabulate

def run_evaluation(min_interactions):
    print(f"Running evaluation for Min Interactions = {min_interactions}...")
    
    cmd = [
        "python", "src/evaluation/evaluate_filtered.py",
        "--graph", "data/processed_phobert/full_hetero_graph.pt",
        "--data-dir", "data/processed_phobert",
        "--min", str(min_interactions)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation for min={min_interactions}")
        print(result.stderr)
        return None
        
    # extract metrics
    output = result.stdout
    metrics = {}
    for m in ["Recall", "NDCG", "Precision", "HitRate", "MRR"]:
        match = re.search(rf"{m}@10:\s+([\d.]+)", output)
        metrics[m.lower()] = float(match.group(1)) * 100 if match else 0.0
    
    return metrics

def main():
    thresholds = [1, 2, 3, 5, 10]
    
    # Baseline Results (Hardcoded from previous experiments)
    baseline = {
        1: {"recall": 6.2, "ndcg": 3.1, "hitrate": 7.8, "precision": None, "mrr": None},
        2: {"recall": None, "ndcg": None, "hitrate": None, "precision": None, "mrr": None},
        3: {"recall": 10.04, "ndcg": 5.43, "hitrate": None, "precision": None, "mrr": None},
        5: {"recall": 8.11, "ndcg": 5.27, "hitrate": None, "precision": None, "mrr": None},
        10: {"recall": None, "ndcg": None, "hitrate": None, "precision": None, "mrr": None}
    }
    
    results = []
    
    print("Starting Comprehensive Benchmark...")
    print("=" * 60)
    
    for k in thresholds:
        m = run_evaluation(k)
        
        if m is None:
            print(f"Skipping min={k} due to error")
            continue
        
        row = {"min": k}
        for key in ["recall", "ndcg", "precision", "hitrate", "mrr"]:
            val = m.get(key, 0.0)
            base = baseline[k].get(key)
            
            row[f"social_{key}"] = f"{val:.2f}%"
            row[f"base_{key}"] = f"{base:.2f}%" if base is not None else "--"
            row[f"diff_{key}"] = f"{val - base:+.2f}%" if base is not None else "--"
            
        results.append(row)
        
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK REPORT: Social Graph vs Baseline")
    print("=" * 80)
    
    # Print Table
    headers = ["Segment", "Metric", "Original (No Social)", "Social Graph", "Impact"]
    print(f"{headers[0]:<25} | {headers[1]:<12} | {headers[2]:<20} | {headers[3]:<15} | {headers[4]:<10}")
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
