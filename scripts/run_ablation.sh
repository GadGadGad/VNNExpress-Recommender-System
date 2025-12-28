#!/bin/bash
# ============================================================
# Ablation Study Script for Kaggle
# Run all model x graph combinations for comprehensive comparison
# ============================================================

# Configuration
EPOCHS=100
PATIENCE=15
BATCH_SIZE=2048

# Define models, graphs, and protocols
MODELS=("simgcl" "xsimgcl" "lightgcl" "ma-hcl" "hetgnn" "ma_hgn" "sim-mahgn" "ngcf")
GRAPHS=("strict_g1" "strict_g2" "strict_g3")
PROTOCOLS=("cold" "full")

# Hybrid models (CF + CB embeddings)
HYBRID_MODELS=("simgcl" "xsimgcl" "cgrc")
EMBEDDINGS=("vn-sbert" "bge-m3")

# Results directory
mkdir -p results/ablation

echo "============================================================"
echo "Starting Ablation Study"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "============================================================"

# ============================================================
# Part 1: CF Models on All Graphs (Cold + Full protocols)
# ============================================================
echo ""
echo ">>> PART 1: CF Models on All Graphs"
echo "============================================================"

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for GRAPH in "${GRAPHS[@]}"; do
            RESULT_FILE="results/ablation/${MODEL}_${GRAPH}_${PROTOCOL}.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo ">>> Skipping (exists): $RESULT_FILE"
                continue
            fi
            
            echo ""
            echo ">>> Training: $MODEL on $GRAPH ($PROTOCOL)"
            
            python scripts/train_cf_models.py \
                --model "$MODEL" \
                --data-path "data/processed/$GRAPH" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --batch-size "$BATCH_SIZE" \
                --eval-protocol "$PROTOCOL" \
                --save-results "$RESULT_FILE"
            
            echo ">>> Completed: $MODEL on $GRAPH ($PROTOCOL)"
        done
    done
done

# ============================================================
# Part 2: Hybrid Models (CF + NLP Embeddings)
# ============================================================
echo ""
echo ">>> PART 2: Hybrid Models (CF + NLP Embeddings)"
echo "============================================================"

HYBRID_GRAPH="strict_g2"  # Best performing graph from Part 1

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for MODEL in "${HYBRID_MODELS[@]}"; do
        for EMB in "${EMBEDDINGS[@]}"; do
            RESULT_FILE="results/ablation/${MODEL}_${EMB}_${HYBRID_GRAPH}_${PROTOCOL}.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo ">>> Skipping (exists): $RESULT_FILE"
                continue
            fi
            
            echo ""
            echo ">>> Training: $MODEL + $EMB on $HYBRID_GRAPH ($PROTOCOL)"
            
            python scripts/train_cf_models.py \
                --model "$MODEL" \
                --data-path "data/processed/$HYBRID_GRAPH" \
                --embedding "$EMB" \
                --epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                --batch-size "$BATCH_SIZE" \
                --eval-protocol "$PROTOCOL" \
                --save-results "$RESULT_FILE"
            
            echo ">>> Completed: $MODEL + $EMB"
        done
    done
done

echo ""
echo "============================================================"
echo "Ablation Study Complete!"
echo "Results saved in: results/ablation/"
echo "============================================================"

# Print summary
echo ""
echo "Summary of Results:"
echo "==================="
for f in results/ablation/*.json; do
    if [ -f "$f" ]; then
        NAME=$(basename "$f" .json)
        RECALL=$(python -c "import json; d=json.load(open('$f')); print(f\"R@10={d.get('recall@10',0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "$NAME: $RECALL"
    fi
done
