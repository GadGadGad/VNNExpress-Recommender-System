import json
import os

NOTEBOOK_PATH = "notebooks/ds300-benchmark.ipynb"

NEW_CONTENT = """# Step 4: Run complete ablation study
import os

EPOCHS = 100
BATCH_SIZE = 2048
PATIENCE = 15

# 1. CF Models
cf_models = ["simgcl", "xsimgcl", "lightgcl", "ma-hcl", "ma_hgn", "ngcf"]
graphs = ["strict_g1", "strict_g2", "strict_g3"]
protocols = ["cold", "full", "loo100"]

print(">>> PART 1: CF Models")
for protocol in protocols:
    for model in cf_models:
        for graph in graphs:
            if model == "hetgnn": continue  # Removed
            print(f"\\nTraining CF: {model} on {graph} ({protocol})")
            outfile = f"results/ablation/cf_{model}_{graph}_{protocol}.json"
            if os.path.exists(outfile):
                print(f"Skipping existing result: {outfile}")
                continue
            !python scripts/train_cf_models.py --model {model} --data-path data/processed/{graph} --epochs {EPOCHS} --patience {PATIENCE} --batch-size {BATCH_SIZE} --eval-protocol {protocol} --save-results {outfile}

# 2. CB Models
cb_models = ["tfidf", "bge-m3", "vn-sbert"]
print("\\n>>> PART 2: CB Models")
for protocol in protocols:
    for model in cb_models:
        for graph in graphs:
            print(f"\\nBenchmarking CB: {model} on {graph} ({protocol})")
            outfile = f"results/ablation/cb_{model}_{graph}_{protocol}.json"
            if os.path.exists(outfile): continue
            !python scripts/benchmark_cbf.py --model {model} --data-path data/processed/{graph} --eval-protocol {protocol} --output {outfile}

# 3. Hybrid Models
hybrid_models = ["simgcl", "xsimgcl"]
embeddings = ["vn-sbert", "bge-m3"]
hybrid_graph = "strict_g2"
print("\\n>>> PART 3: Hybrid Models")
for protocol in protocols:
    for model in hybrid_models:
        for emb in embeddings:
            print(f"\\nTraining Hybrid: {model} + {emb} on {hybrid_graph} ({protocol})")
            outfile = f"results/ablation/hybrid_{model}_{emb}_{hybrid_graph}_{protocol}.json"
            if os.path.exists(outfile): continue
            !python scripts/train_cf_models.py --model {model} --data-path data/processed/{hybrid_graph} --embedding {emb} --epochs {EPOCHS} --patience {PATIENCE} --batch-size {BATCH_SIZE} --eval-protocol {protocol} --save-results {outfile}
"""

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

found = False
for cell in data['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if "Step 4: Run ablation study" in source and "run_ablation.sh" in source:
            print("Found target cell.")
            lines = NEW_CONTENT.split('\n')
            # Standard nbformat uses a list of strings ending with \n
            new_source = [line + '\n' for line in lines[:-1]] + [lines[-1]]
            cell['source'] = new_source
            found = True
            break

if found:
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("Notebook updated.")
else:
    print("Target cell not found. Dumping cells to debug:")
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
             print(f"Cell {i}: {cell['source'][:50]}")
