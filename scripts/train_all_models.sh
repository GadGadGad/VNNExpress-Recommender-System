#!/bin/bash
# ==============================================================================
# Train All Models - Compare different GNN models
# ==============================================================================
# Usage:
#   chmod +x scripts/train_all_models.sh
#   ./scripts/train_all_models.sh
# ==============================================================================

set -e

EPOCHS="${EPOCHS:-100}"
DATA_PATH="data/processed/user_article_graph.pt"
MODELS_DIR="models"
RESULTS_FILE="$MODELS_DIR/comparison_results.txt"

echo "============================================================"
echo "Training All GNN Models"
echo "============================================================"
echo "Epochs: $EPOCHS"
echo ""

# Create results file header
echo "Model Comparison Results - $(date)" > "$RESULTS_FILE"
echo "============================================================" >> "$RESULTS_FILE"

for model in sage gcn gat lightgcn; do
    echo ""
    echo ">>> Training $model..."
    echo "------------------------------------------------------------"
    
    python src/training/train_gnn_baseline.py \
        --model "$model" \
        --data-path "$DATA_PATH" \
        --epochs "$EPOCHS" \
        --save-dir "$MODELS_DIR" \
        --k-values "5,10,20" 2>&1 | tee -a "$RESULTS_FILE"
    
    echo "" >> "$RESULTS_FILE"
done

echo ""
echo "============================================================"
echo "All models trained! Results saved to: $RESULTS_FILE"
echo "============================================================"
