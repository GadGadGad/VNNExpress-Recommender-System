#!/usr/bin/env python3
"""
Comprehensive Kaggle Notebook Reconstruction for Publishable Benchmarks.
Generates kaggle_train.ipynb with all necessary steps.
"""
import json

# Notebook configuration
KAGGLE_INPUT = "/kaggle/input/vnexpress-data"
DATA_DIR = "data/processed/strict_g2"

# Models to benchmark
CB_MODELS = ["tfidf", "vn-sbert", "bge-m3"]  # Only models with proper embeddings
CF_MODELS = ["simgcl", "xsimgcl", "lightgcl", "ma-hcl", "sim-mahgn"]
PROTOCOLS = ["full", "loo100", "cold"]

def create_notebook():
    """Create comprehensive Kaggle notebook."""
    cells = []
    
    # ==================== SETUP ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 VNExpress Recommendation Benchmark\n",
            "\n",
            "**Comprehensive benchmark** comparing Content-Based (CB) and Collaborative Filtering (CF) approaches.\n",
            "\n",
            "## 📋 Experiment Design\n",
            "- **CB Models**: TF-IDF (Vietnamese), VN-SBERT, BGE-M3\n",
            "- **CF Models**: SimGCL, XSimGCL, LightGCL, MA-HCL, Sim-MAHGN\n",
            "- **Protocols**: Full Ranking, LOO100, Cold-Start\n",
            "- **Metrics**: Recall@10, NDCG@10, AUC\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Clone repository and setup\n",
            "!git clone https://github.com/GadGadGad/DS300-Final-Project.git project\n",
            "%cd project\n",
            "!pip install -q torch torch_geometric pandas numpy scikit-learn tqdm sentence-transformers underthesea\n",
            "print('✅ Setup complete!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== DATA COPY ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📁 Step 1: Data Setup\n", "Copy data from Kaggle input and prepare directories."]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            f"# Copy data from Kaggle input\n",
            f"!mkdir -p data/raw data/processed checkpoints models results\n",
            f"!cp -r {KAGGLE_INPUT}/* data/raw/ 2>/dev/null || echo 'Using existing data'\n",
            f"!ls data/raw/"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== GRAPH GENERATION ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Step 2: Graph Generation\\n",
            "Build three graph variants for ablation study:\\n",
            "- **G1 (Bipartite)**: User-Article edges only\\n",
            "- **G2 (Heterogeneous)**: + Social edges (reply/interaction)\\n",
            "- **G3 (Category)**: + Category hub nodes\\n"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ===== G1: Strict Bipartite =====\\n",
            "!python src/data/convert_to_gnn.py --graph-type bipartite --output data/processed/strict_g1 --min-user-interactions 2 --min-article-interactions 2\\n",
            "!python src/data/convert_to_gnn.py --graph-type with-negatives --output data/processed/strict_g1 --min-user-interactions 2 --min-article-interactions 2\\n",
            "print('✅ G1 (Bipartite) complete!')\\n",
            "\\n",
            "# ===== G2: Strict Heterogeneous =====\\n",
            "!python src/data/convert_to_gnn.py --graph-type hetero --output data/processed/strict_g2 --min-user-interactions 2 --min-article-interactions 2\\n",
            "!python src/data/convert_to_gnn.py --graph-type with-negatives --output data/processed/strict_g2 --min-user-interactions 2 --min-article-interactions 2\\n",
            "print('✅ G2 (Heterogeneous) complete!')\\n",
            "\\n",
            "# ===== G3: Strict Category-Augmented =====\\n",
            "!python src/data/convert_to_gnn.py --graph-type hetero --output data/processed/strict_g3 --min-user-interactions 2 --min-article-interactions 2\\n",
            "!python src/data/augment_graph_with_categories.py --input data/processed/strict_g3 --output data/processed/strict_g3\\n",
            "!python src/data/convert_to_gnn.py --graph-type with-negatives --output data/processed/strict_g3 --min-user-interactions 2 --min-article-interactions 2\\n",
            "print('✅ G3 (Category) complete!')\\n",
            "\\n",
            "# List all graphs\\n",
            "!ls data/processed/"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== EMBEDDING GENERATION ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧠 Step 3: Embedding Generation\n",
            "Generate TF-IDF and neural embeddings for Content-Based models."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Generate TF-IDF embeddings with Vietnamese preprocessing\n",
            "!python scripts/generate_multi_embeddings.py --model tfidf --data-path data/processed/strict_g2\n",
            "print('✅ TF-IDF embeddings generated!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Generate neural embeddings (VN-SBERT, BGE-M3)\n",
            "!python scripts/generate_multi_embeddings.py --model vietnamese-sbert --data-path data/processed/strict_g2\n",
            "!python scripts/generate_multi_embeddings.py --model bge-m3 --data-path data/processed/strict_g2\n",
            "print('✅ Neural embeddings generated!')\n",
            "!ls checkpoints/"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== HYPERPARAMETERS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## ⚙️ Step 4: Hyperparameters\n", "Define standard hyperparameters for fair comparison."]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Standard Hyperparameters (Fair Comparison)\\n",
            "EPOCHS = 100\\n",
            "LR = 0.001\\n",
            "HIDDEN_DIM = 64\\n",
            "N_LAYERS = 3\\n",
            "BATCH_SIZE = 1024\\n",
            "\\n",
            "# Graph variants for ablation study\\n",
            "GRAPH_PATHS = {\\n",
            "    'G1': 'data/processed/strict_g1',  # Bipartite only\\n",
            "    'G2': 'data/processed/strict_g2',  # + Social edges\\n",
            "    'G3': 'data/processed/strict_g3',  # + Category hubs\\n",
            "}\\n",
            "\\n",
            "# Models and Protocols\\n",
            f"CB_MODELS = {CB_MODELS}\\n",
            f"CF_MODELS = {CF_MODELS}\\n",
            f"PROTOCOLS = {PROTOCOLS}\\n",
            "\\n",
            "print('✅ Hyperparameters set!')\\n",
            "print(f'Graph variants: {{list(GRAPH_PATHS.keys())}}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== CB EXPERIMENTS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📝 Step 5: Content-Based (CB) Experiments\n",
            "Benchmark CB models across all protocols."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Run CB experiments on G2 (main graph with embeddings)\\n",
            "DATA_PATH = GRAPH_PATHS['G2']  # CB uses G2 for embeddings\\n",
            "for model in CB_MODELS:\\n",
            "    for protocol in PROTOCOLS:\\n",
            "        print(f'\\\\n{\\\"=\\\"*50}')\\n",
            "        print(f'Evaluating CB: {model} | Protocol: {protocol}')\\n",
            "        print(f'{\\\"=\\\"*50}')\\n",
            "        !python scripts/benchmark_cbf.py \\\\\\n",
            "            --model {model} \\\\\\n",
            "            --data-path {DATA_PATH} \\\\\\n",
            "            --eval-protocol {protocol} \\\\\\n",
            "            --output results/cb_{model}_{protocol}.json\\n",
            "\\n",
            "print('\\\\n✅ CB experiments complete!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== CF EXPERIMENTS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔗 Step 6: Collaborative Filtering (CF) Experiments\\n",
            "Train and evaluate GNN-based CF models on all graph variants (G1, G2, G3)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Run CF experiments on ALL graph variants\\n",
            "for graph_name, data_path in GRAPH_PATHS.items():\\n",
            "    for model in CF_MODELS:\\n",
            "        for protocol in PROTOCOLS:\\n",
            "            print(f'\\\\n{\\\"=\\\"*60}')\\n",
            "            print(f'Training CF: {model} | Graph: {graph_name} | Protocol: {protocol}')\\n",
            "            print(f'{\\\"=\\\"*60}')\\n",
            "            !python scripts/train_cf_models.py \\\\\\n",
            "                --model {model} \\\\\\n",
            "                --data-path {data_path} \\\\\\n",
            "                --epochs {EPOCHS} \\\\\\n",
            "                --lr {LR} \\\\\\n",
            "                --hidden-dim {HIDDEN_DIM} \\\\\\n",
            "                --n-layers {N_LAYERS} \\\\\\n",
            "                --batch-size {BATCH_SIZE} \\\\\\n",
            "                --eval-protocol {protocol} \\\\\\n",
            "                --save-results results/cf_{model}_{graph_name}_{protocol}.json\\n",
            "\\n",
            "print('\\\\n✅ CF experiments complete!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== RESULTS AGGREGATION ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Step 7: Results Aggregation\n",
            "Compile all results into a publishable format."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import json\n",
            "import glob\n",
            "import pandas as pd\n",
            "\n",
            "# Collect all results\n",
            "results = []\n",
            "for path in glob.glob('results/*.json'):\n",
            "    try:\n",
            "        with open(path, 'r') as f:\n",
            "            data = json.load(f)\n",
            "        \n",
            "        # Parse filename: cb_model_protocol.json or cf_model_protocol.json\n",
            "        basename = path.split('/')[-1].replace('.json', '')\n",
            "        parts = basename.split('_')\n",
            "        \n",
            "        if len(parts) >= 3:\n",
            "            model_type = parts[0].upper()\n",
            "            protocol = parts[-1]\n",
            "            model_name = '_'.join(parts[1:-1])\n",
            "        else:\n",
            "            model_type, model_name, protocol = 'Unknown', basename, 'unknown'\n",
            "        \n",
            "        results.append({\n",
            "            'Type': model_type,\n",
            "            'Model': model_name,\n",
            "            'Protocol': protocol,\n",
            "            'Recall@10': data.get('Recall@10') or data.get('recall@10') or 0,\n",
            "            'NDCG@10': data.get('NDCG@10') or data.get('ndcg@10') or 0,\n",
            "            'AUC': data.get('AUC') or data.get('auc') or 0\n",
            "        })\n",
            "    except Exception as e:\n",
            "        print(f'Error loading {path}: {e}')\n",
            "\n",
            "# Create DataFrame\n",
            "df = pd.DataFrame(results)\n",
            "df = df.sort_values(['Type', 'Protocol', 'Recall@10'], ascending=[True, True, False])\n",
            "\n",
            "print('\\n📊 BENCHMARK RESULTS')\n",
            "print('=' * 80)\n",
            "print(df.to_string(index=False))\n",
            "\n",
            "# Save to CSV\n",
            "df.to_csv('results/benchmark_summary.csv', index=False)\n",
            "print('\\n✅ Results saved to results/benchmark_summary.csv')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== BEST MODELS ====================
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Highlight best models per protocol\n",
            "print('\\n🏆 BEST MODELS BY PROTOCOL')\n",
            "print('=' * 50)\n",
            "\n",
            "for protocol in PROTOCOLS:\n",
            "    protocol_df = df[df['Protocol'] == protocol]\n",
            "    if len(protocol_df) > 0:\n",
            "        best = protocol_df.loc[protocol_df['Recall@10'].idxmax()]\n",
            "        print(f\"\\n{protocol.upper()}:\")\n",
            "        print(f\"  🥇 {best['Type']}/{best['Model']}: R@10={best['Recall@10']:.4f}, NDCG@10={best['NDCG@10']:.4f}\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==================== DOWNLOAD ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📥 Step 8: Download Results\n", "Download trained models and benchmark results."]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Create archive for download\n",
            "!zip -r benchmark_results.zip results/ models/ checkpoints/\n",
            "print('\\n✅ Results archived! Download benchmark_results.zip')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Create notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }
    
    return notebook

if __name__ == "__main__":
    notebook = create_notebook()
    
    output_path = "notebooks/kaggle_train.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Notebook saved to {output_path}")
    print(f"   - CB Models: {CB_MODELS}")
    print(f"   - CF Models: {CF_MODELS}")
    print(f"   - Protocols: {PROTOCOLS}")
