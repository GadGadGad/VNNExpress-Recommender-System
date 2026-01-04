#!/bin/bash

# Script to regenerate missing models and embeddings for the Demo
# Usage: ./train_demo.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================================"
echo "STARTING DEMO TRAINING PROCESS"
echo "========================================================"

# Create directories
echo ">> Creating directories..."
mkdir -p models/checkpoints
mkdir -p checkpoints

# Generate Embeddings (Content-Based)
echo ""
echo ">> 1. Generating ALL Content Embeddings..."
# Using the 'all' option I saw in generate_multi_embeddings.py
python scripts/generate_multi_embeddings.py \
    --model all \
    --data-path data/processed/strict_g2 \
    --output-dir checkpoints

echo ""
echo ">> 2. Retraining ALL CF Models (Fast Mode)..."
# List of CF models from the report
MODELS=("SimGCL" "LightGCL" "XSimGCL" "MA-HCL" "LightGCN")

for model in "${MODELS[@]}"; do
    echo "   -> Training $model..."
    # Convert model name to lowercase for filename
    model_lower=$(echo "$model" | tr '[:upper:]' '[:lower:]')
    python scripts/train_cf_models.py \
        --model "$model_lower" \
        --data-path data/processed/strict_g2 \
        --epochs 5 \
        --batch-size 2048 \
        --lr 0.001 \
        --patience 3 \
        --save-results "models/${model_lower}_restored.pt"
done

echo "========================================================"
echo "TRAINING COMPLETE!"
echo "Please restart the backend/full-stack script now."
echo "========================================================"
