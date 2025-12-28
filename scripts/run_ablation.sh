#!/bin/bash
# ============================================================
# Ablation Study Script for Kaggle
# Run all model x graph x protocol combinations
# ============================================================

# Configuration
EPOCHS=100
PATIENCE=15
BATCH_SIZE=2048

# Define models, graphs, and protocols
CF_MODELS=("simgcl" "xsimgcl" "lightgcl" "ma-hcl" "ma_hgn" "sim-mahgn" "ngcf")
CB_MODELS=("tfidf" "bge-m3" "vn-sbert")
GRAPHS=("strict_g1" "strict_g2" "strict_g3")
PROTOCOLS=("cold" "full" "loo100")

# Hybrid models (CF + CB embeddings)
HYBRID_MODELS=("simgcl" "xsimgcl" "cgrc")
EMBEDDINGS=("vn-sbert" "bge-m3")

# Results directory
mkdir -p results/ablation

echo "============================================================"
echo "Starting Ablation Study"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "Protocols: ${PROTOCOLS[*]}"
echo "============================================================"

# ============================================================
# Part 1: CF Models on All Graphs
# ============================================================
echo ""
echo ">>> PART 1: CF Models on All Graphs"
echo "============================================================"

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for MODEL in "${CF_MODELS[@]}"; do
        for GRAPH in "${GRAPHS[@]}"; do
            RESULT_FILE="results/ablation/cf_${MODEL}_${GRAPH}_${PROTOCOL}.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo ">>> Skipping (exists): $RESULT_FILE"
                continue
            fi
            
            echo ""
            echo ">>> Training CF: $MODEL on $GRAPH ($PROTOCOL)"
            
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
# Part 2: CB Models (Content-Based)
# ============================================================
echo ""
echo ">>> PART 2: CB Models (Content-Based)"
echo "============================================================"

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for MODEL in "${CB_MODELS[@]}"; do
        for GRAPH in "${GRAPHS[@]}"; do
            RESULT_FILE="results/ablation/cb_${MODEL}_${GRAPH}_${PROTOCOL}.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo ">>> Skipping (exists): $RESULT_FILE"
                continue
            fi
            
            echo ""
            echo ">>> Benchmarking CB: $MODEL on $GRAPH ($PROTOCOL)"
            
            python scripts/benchmark_cbf.py \
                --model "$MODEL" \
                --data-path "data/processed/$GRAPH" \
                --protocol "$PROTOCOL" \
                --output "$RESULT_FILE"
            
            echo ">>> Completed: $MODEL on $GRAPH ($PROTOCOL)"
        done
    done
done

# ============================================================
# Part 3: Hybrid Models (CF + NLP Embeddings)
# ============================================================
echo ""
echo ">>> PART 3: Hybrid Models (CF + NLP Embeddings)"
echo "============================================================"

HYBRID_GRAPH="strict_g2"  # Best performing graph

for PROTOCOL in "${PROTOCOLS[@]}"; do
    for MODEL in "${HYBRID_MODELS[@]}"; do
        for EMB in "${EMBEDDINGS[@]}"; do
            RESULT_FILE="results/ablation/hybrid_${MODEL}_${EMB}_${HYBRID_GRAPH}_${PROTOCOL}.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo ">>> Skipping (exists): $RESULT_FILE"
                continue
            fi
            
            echo ""
            echo ">>> Training Hybrid: $MODEL + $EMB on $HYBRID_GRAPH ($PROTOCOL)"
            
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
echo ""
echo "--- CF Models ---"
for f in results/ablation/cf_*.json; do
    if [ -f "$f" ]; then
        NAME=$(basename "$f" .json | sed 's/cf_//')
        RECALL=$(python -c "import json; d=json.load(open('$f')); print(f\"R@10={d.get('recall@10',0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "$NAME: $RECALL"
    fi
done
echo ""
echo "--- CB Models ---"
for f in results/ablation/cb_*.json; do
    if [ -f "$f" ]; then
        NAME=$(basename "$f" .json | sed 's/cb_//')
        RECALL=$(python -c "import json; d=json.load(open('$f')); print(f\"R@10={d.get('recall@10',0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "$NAME: $RECALL"
    fi
done
echo ""
echo "--- Hybrid Models ---"
for f in results/ablation/hybrid_*.json; do
    if [ -f "$f" ]; then
        NAME=$(basename "$f" .json | sed 's/hybrid_//')
        RECALL=$(python -c "import json; d=json.load(open('$f')); print(f\"R@10={d.get('recall@10',0):.4f}\")" 2>/dev/null || echo "N/A")
        echo "$NAME: $RECALL"
    fi
done
