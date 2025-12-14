# GNN-Based Recommendation System for VnExpress

A Graph Neural Network (GNN) based article recommendation system using user-comment interaction data from VnExpress.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (uses config.yaml)
python scripts/run_pipeline.py

# 3. Or run with custom settings
python scripts/run_pipeline.py --epochs 50 --embedding bert
```

## Project Structure

```
main/
├── config.yaml              # Pipeline configuration
├── scripts/
│   ├── run_pipeline.py      # Main pipeline script
│   ├── train_cf_models.py   # CF/CL model training
│   └── train_and_compare_all.py  # Quick comparison script
├── src/
│   ├── data/
│   │   ├── generate_embeddings.py  # PhoBERT embeddings
│   │   └── convert_to_gnn.py       # Data conversion
│   ├── models/               # Model implementations
│   │   ├── ngcf.py          # NGCF
│   │   ├── sgl.py           # SGL (Contrastive)
│   │   ├── simgcl.py        # SimGCL (Contrastive)
│   │   ├── ncl.py           # NCL (Contrastive)
│   │   └── lightgcl.py      # LightGCL (SVD-based)
│   └── training/
│       └── train_gnn_baseline.py   # GNN baseline training
├── data/
│   ├── raw/                 # Raw CSV files
│   └── processed_phobert/   # Processed graph data
└── models/                  # Saved models & results
```

## Configuration (config.yaml)

Edit `config.yaml` to customize the pipeline:

```yaml
data:
  embedding: "random"          # random | tfidf | phobert
  min_user_interactions: 2     # k-core filter

training:
  epochs: 30
  batch_size: 2048
  learning_rate: 0.001
  hidden_dim: 64

models:
  gnn:
    enabled: true
    models: ["sage", "gcn", "gat", "lightgcn"]
  cf:
    enabled: true
    models: ["ngcf", "simplex", "directau"]
  cl:
    enabled: true
    models: ["sgl", "simgcl", "ncl", "lightgcl"]
```

## Pipeline Usage

### Full Pipeline
```bash
# Run complete pipeline with config.yaml
python scripts/run_pipeline.py

# Override specific settings
python scripts/run_pipeline.py --epochs 50 --embedding phobert

# Skip certain model types
python scripts/run_pipeline.py --skip-gnn --skip-cf

# Train specific models only
python scripts/run_pipeline.py --models sgl ncl lightgcl
```

### Step-by-Step

```bash
# Step 1: Generate PhoBERT embeddings (optional)
python src/data/generate_embeddings.py

# Step 2: Convert data to GNN format
python src/data/convert_to_gnn.py \
    --output data/processed_phobert \
    --graph-type hetero \
    --use-phobert \
    --min-user-interactions 2

# Step 3: Train individual model
python scripts/train_cf_models.py --model sgl --epochs 30

# Step 4: Train all and compare
python scripts/train_and_compare_all.py --epochs 30
```

## Implemented Models

| Model | Type | Description |
|-------|------|-------------|
| **SAGE** | GNN | GraphSAGE with sampling |
| **GCN** | GNN | Graph Convolutional Network |
| **GAT** | GNN | Graph Attention Network |
| **LightGCN** | GNN | Lightweight GCN (no feature transform) |
| **NGCF** | CF | Neural Graph Collaborative Filtering |
| **SimpleX** | CF | Simplified negative sampling |
| **DirectAU** | CF | Alignment & Uniformity loss |
| **SGL** | CL | Self-supervised Graph Learning |
| **SimGCL** | CL | Simple Graph Contrastive Learning |
| **NCL** | CL | Neighborhood-enriched Contrastive |
| **LightGCL** | CL | SVD-based Contrastive Learning |

## Evaluation Metrics

- **Recall@K**: Percentage of relevant items in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **HitRate@K**: Percentage of users with at least one hit
- **MRR**: Mean Reciprocal Rank

## Example Results

```
+----------+--------+-------+-------+--------+--------+---------+-------+
| Model    | Type   | R@1   | R@5   | R@10   | N@10   | HR@10   | MRR   |
+==========+========+=======+=======+========+========+=========+=======+
| SGL      | CL     | 0.790 | 0.818 | 0.859  | 0.809  | 0.916   | 0.831 |
| NCL      | CL     | 0.732 | 0.784 | 0.822  | 0.767  | 0.887   | 0.785 |
| SIMGCL   | CL     | 0.687 | 0.738 | 0.782  | 0.720  | 0.855   | 0.736 |
| LIGHTGCL | CL     | 0.650 | 0.701 | 0.756  | 0.698  | 0.830   | 0.701 |
| NGCF     | CF     | 0.031 | 0.072 | 0.131  | 0.070  | 0.168   | 0.060 |
+----------+--------+-------+-------+--------+--------+---------+-------+
```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- transformers (for PhoBERT)
- pandas, numpy, scipy
- tabulate, pyyaml

## License

MIT License
