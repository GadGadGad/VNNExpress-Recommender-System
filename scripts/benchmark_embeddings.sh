#!/bin/bash
# Benchmark all embedding models with SimGCL on G2 (hetero graph)

MODELS=("bge-m3" "gte-multilingual" "e5-large" "e5-base" "vietnamese-sbert")

for model in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Testing embedding: $model"
    echo "============================================================"
    
    # First, update the embedding loading in Python
    python3 -c "
import torch
import os

# Load embedding
emb_path = f'checkpoints/${model}_article_embeddings.pt'
if os.path.exists(emb_path):
    emb = torch.load(emb_path, map_location='cpu')
    print(f'Loaded {emb_path}: {emb.shape}')
    
    # Save as current embedding for training
    torch.save(emb, 'checkpoints/current_article_embeddings.pt')
    print('Saved as current_article_embeddings.pt')
else:
    print(f'Not found: {emb_path}')
    exit(1)
"
    
    # Run SimGCL with this embedding
    python scripts/train_cf_models.py \
        --model simgcl \
        --graph-type hetero \
        --epochs 30 \
        --device cpu \
        --data-path data/processed/enhanced_v1 \
        --save-results "results_simgcl_${model}.json"
        
    echo ""
done

echo "All benchmarks complete!"
