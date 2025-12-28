import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def aggregate_and_plot_results(results_dir="/kaggle/working/project/results"):
    results_dir = Path(results_dir)
    results = []

    if results_dir.exists():
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse filename to extract metadata (fallback if not in json)
                filename = json_file.stem
                parts = filename.split('_')
                
                row = data.copy()
                
                # Logic to determine Model, Graph, and Protocol
                if 'encoder' in row: # Content-Based
                    row['Model'] = f"CB_{row['encoder'].upper()}"
                    row['Protocol'] = 'Full' 
                    if 'data_path' in row:
                        row['Graph'] = Path(row['data_path']).name
                    else:
                        row['Graph'] = '_'.join(parts[2:])
                else: # CF Models
                    if 'model' not in row: row['Model'] = parts[0].upper()
                    else: row['Model'] = row['model'].upper()
                    
                    if 'data_path' in row: row['Graph'] = Path(row['data_path']).name
                    elif len(parts) >= 2: row['Graph'] = '_'.join(parts[1:-1])
                    
                    if 'eval_protocol' in row: row['Protocol'] = row['eval_protocol'].title()
                    elif len(parts) >= 1: row['Protocol'] = parts[-1].title()

                results.append(row)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    if results:
        df = pd.DataFrame(results)
        print(f"✅ Loaded {len(df)} results. Columns: {df.columns.tolist()}")
        
        # Save aggregated results
        df.to_csv(results_dir / "benchmark_summary.csv", index=False)
        print(f"Saved summary to {results_dir / 'benchmark_summary.csv'}")
        
        # Plotting
        metrics_to_plot = ['recall@10', 'f1@10', 'ndcg@10']
        
        for metric in metrics_to_plot:
            if metric in df.columns:
                try:
                    plt.figure(figsize=(10, 6)) # Create new figure for each plot
                    g = sns.catplot(
                        data=df, 
                        x='Model', 
                        y=metric, 
                        hue='Graph', 
                        col='Protocol', 
                        kind='bar', 
                        height=5, 
                        aspect=1.2,
                        sharey=False
                    )
                    g.set_xticklabels(rotation=45)
                    g.fig.suptitle(f'Performance Comparison: {metric.upper()}', y=1.05)
                    plt.show()
                except Exception as e:
                    print(f"Could not plot {metric}: {e}")
        
        # Display Table
        cols = ['Model', 'Graph', 'Protocol', 'recall@10', 'ndcg@10', 'f1@10', 'mrr']
        available_cols = [c for c in cols if c in df.columns]
        
        print("\nBenchmark Results Table:")
        # In notebook environments, just returning the DF or display() works. 
        # For script, we print.
        print(df[available_cols].sort_values(by=['Protocol', 'recall@10'], ascending=[True, False]).to_markdown(index=False))
        
    else:
        print("❌ No results found. Run benchmark first.")

if __name__ == "__main__":
    aggregate_and_plot_results()
