# GNN-Based Recommendation System for VnExpress

A Graph Neural Network (GNN) based article recommendation system using user-comment interaction data from VnExpress.

## Quick Start

```bash
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_pipeline.py

# Or run with custom settings
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
  min_user_interactions: 2

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
# Generate PhoBERT embeddings (optional)
python src/data/generate_embeddings.py

# Convert data to GNN format
python src/data/convert_to_gnn.py \
    --output data/processed_phobert \
    --graph-type hetero \
    --use-phobert \
    --min-user-interactions 2

# Train individual model
python scripts/train_cf_models.py --model sgl --epochs 30

# Train all and compare
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
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
| Model    | Type   |   R@1 |   R@5 |   R@10 |   N@1 |   N@5 |   N@10 |   HR@1 |   HR@5 |   HR@10 |   MRR |
+==========+========+=======+=======+========+=======+=======+========+========+========+=========+=======+
| SGL      | CL     | 0.732 | 0.820 |  0.871 | 0.732 | 0.774 |  0.794 |  0.732 |  0.890 |   0.929 | 0.801 |
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
| NCL      | CL     | 0.706 | 0.762 |  0.816 | 0.706 | 0.724 |  0.744 |  0.706 |  0.836 |   0.886 | 0.760 |
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
| SIMGCL   | CL     | 0.653 | 0.739 |  0.784 | 0.653 | 0.692 |  0.709 |  0.653 |  0.816 |   0.854 | 0.722 |
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
| NGCF     | CF     | 0.030 | 0.116 |  0.192 | 0.030 | 0.073 |  0.100 |  0.030 |  0.143 |   0.247 | 0.081 |
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
| LIGHTGCL | CL     | 0.010 | 0.045 |  0.083 | 0.010 | 0.028 |  0.040 |  0.010 |  0.061 |   0.110 | 0.031 |
+----------+--------+-------+-------+--------+-------+-------+--------+--------+--------+---------+-------+
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
