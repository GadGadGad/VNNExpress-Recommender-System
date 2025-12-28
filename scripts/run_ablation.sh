#!/bin/bash
# ============================================================
# Ablation Study Script for Kaggle
# Run all model x graph combinations for comprehensive comparison
# ============================================================

# Configuration
EPOCHS=100
PATIENCE=15
EVAL_PROTOCOL="cold"  # Options: full, loo100, cold
BATCH_SIZE=2048

# Define models and graphs
MODELS=("simgcl" "xsimgcl" "lightgcl" "ma-hcl" "hetgnn" "ma_hgn")
GRAPHS=("strict_g1" "strict_g2" "strict_g3")

# Results directory
mkdir -p results/ablation

echo "============================================================"
echo "Starting Ablation Study"
echo "Epochs: $EPOCHS, Protocol: $EVAL_PROTOCOL"
echo "============================================================"

for MODEL in "${MODELS[@]}"; do
    for GRAPH in "${GRAPHS[@]}"; do
        RESULT_FILE="results/ablation/${MODEL}_${GRAPH}_${EVAL_PROTOCOL}.json"
        
        echo ""
        echo ">>> Training: $MODEL on $GRAPH ($EVAL_PROTOCOL)"
        echo ">>> Output: $RESULT_FILE"
        echo ""
        
        python scripts/train_cf_models.py \
            --model "$MODEL" \
            --data-path "data/processed/$GRAPH" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --batch-size "$BATCH_SIZE" \
            --eval-protocol "$EVAL_PROTOCOL" \
            --save-results "$RESULT_FILE"
        
        echo ">>> Completed: $MODEL on $GRAPH"
        echo "============================================================"
    done
done

echo ""
echo "============================================================"
echo "Ablation Study Complete!"
echo "Results saved in: results/ablation/"
echo "============================================================"

# Optional: Print summary
echo ""
echo "Summary of Results:"
for f in results/ablation/*.json; do
    if [ -f "$f" ]; then
        echo "--- $(basename $f) ---"
        cat "$f"
        echo ""
    fi
done
