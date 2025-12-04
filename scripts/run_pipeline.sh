#!/bin/bash
# ==============================================================================
# VnExpress News Recommendation Pipeline
# ==============================================================================
# This script runs the complete pipeline from data analysis to model training.
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Or run specific stages:
#   ./run_pipeline.sh --eda-only
#   ./run_pipeline.sh --train-only
#   ./run_pipeline.sh --model gat
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# Configuration
# ==============================================================================
DATA_RAW="data/raw"
DATA_PROCESSED="data/processed"
PLOTS_DIR="plots"
MODELS_DIR="models"

# Model settings
MODEL="${MODEL:-sage}"           # sage, gcn, gat, lightgcn
HIDDEN_DIM="${HIDDEN_DIM:-64}"
OUT_DIM="${OUT_DIM:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.01}"

# Negative sampling settings
NEG_RATIO="${NEG_RATIO:-1}"
NEG_STRATEGY="${NEG_STRATEGY:-random}"  # random, popular, hard
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"

# ==============================================================================
# Parse Arguments
# ==============================================================================
EDA_ONLY=false
CONVERT_ONLY=false
TRAIN_ONLY=false
SKIP_EDA=false
WITH_NEGATIVES=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --eda-only)
            EDA_ONLY=true
            shift
            ;;
        --convert-only)
            CONVERT_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            SKIP_EDA=true
            shift
            ;;
        --skip-eda)
            SKIP_EDA=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --neg-ratio)
            NEG_RATIO="$2"
            shift 2
            ;;
        --neg-strategy)
            NEG_STRATEGY="$2"
            shift 2
            ;;
        --no-negatives)
            WITH_NEGATIVES=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eda-only        Run EDA only"
            echo "  --convert-only    Convert data to GNN format only"
            echo "  --train-only      Run training only (skip EDA and conversion)"
            echo "  --skip-eda        Skip EDA step"
            echo "  --model MODEL     Model type: sage, gcn, gat, lightgcn (default: sage)"
            echo "  --epochs N        Number of training epochs (default: 100)"
            echo "  --neg-ratio N     Negative samples per positive (default: 1)"
            echo "  --neg-strategy S  Strategy: random, popular, hard (default: random)"
            echo "  --no-negatives    Don't generate negative samples"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_step() {
    echo ""
    echo ">>> $1"
    echo "------------------------------------------------------------"
}

check_data() {
    if [ ! -f "$DATA_RAW/articles.csv" ] || [ ! -f "$DATA_RAW/replies.csv" ]; then
        echo "ERROR: Data files not found in $DATA_RAW/"
        echo "Expected: articles.csv, replies.csv"
        exit 1
    fi
}

# ==============================================================================
# Main Pipeline
# ==============================================================================
print_header "VnExpress News Recommendation Pipeline"
echo "Model: $MODEL | Epochs: $EPOCHS | Neg Ratio: $NEG_RATIO | Strategy: $NEG_STRATEGY"

# Check data exists
check_data

# Create directories
mkdir -p "$PLOTS_DIR" "$MODELS_DIR" "$DATA_PROCESSED"

# ------------------------------------------------------------------------------
# Step 1: Exploratory Data Analysis
# ------------------------------------------------------------------------------
if [ "$TRAIN_ONLY" = false ] && [ "$SKIP_EDA" = false ]; then
    print_step "Step 1: Exploratory Data Analysis (EDA)"
    
    echo "Running EDA on raw crawled data..."
    python src/eda/eda_crawled_data.py
    
    if [ "$EDA_ONLY" = true ]; then
        print_header "EDA Complete"
        echo "Plots saved to: $PLOTS_DIR/"
        exit 0
    fi
fi

# ------------------------------------------------------------------------------
# Step 2: Convert to GNN-Ready Format
# ------------------------------------------------------------------------------
if [ "$TRAIN_ONLY" = false ]; then
    print_step "Step 2: Converting Data to GNN Format"
    
    # Build basic user-article graph
    echo "Building user-article bipartite graph..."
    python src/data/convert_to_gnn.py --graph-type user-article \
        --hidden-dim "$HIDDEN_DIM" \
        --output "$DATA_PROCESSED"
    
    # Build graph with negative samples
    if [ "$WITH_NEGATIVES" = true ]; then
        echo ""
        echo "Building graph with negative samples..."
        python src/data/convert_to_gnn.py --graph-type with-negatives \
            --neg-ratio "$NEG_RATIO" \
            --neg-strategy "$NEG_STRATEGY" \
            --train-ratio "$TRAIN_RATIO" \
            --val-ratio "$VAL_RATIO" \
            --hidden-dim "$HIDDEN_DIM" \
            --output "$DATA_PROCESSED"
    fi
    
    # Run EDA on graph data
    echo ""
    echo "Running EDA on graph data..."
    python src/eda/eda_graph_data.py --data-path "$DATA_PROCESSED/user_article_graph.pt"
    
    if [ "$CONVERT_ONLY" = true ]; then
        print_header "Data Conversion Complete"
        echo "Graph data saved to: $DATA_PROCESSED/"
        exit 0
    fi
fi

# ------------------------------------------------------------------------------
# Step 3: Train Baseline Model
# ------------------------------------------------------------------------------
print_step "Step 3: Training $MODEL Model"

echo "Training configuration:"
echo "  Model: $MODEL"
echo "  Hidden dim: $HIDDEN_DIM"
echo "  Output dim: $OUT_DIM"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo ""

python src/training/train_gnn_baseline.py \
    --model "$MODEL" \
    --data-path "$DATA_PROCESSED/user_article_graph.pt" \
    --hidden-dim "$HIDDEN_DIM" \
    --out-dim "$OUT_DIM" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --save-dir "$MODELS_DIR" \
    --k-values "5,10,20"

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print_header "Pipeline Complete!"

echo ""
echo "Results:"
echo "  - EDA plots: $PLOTS_DIR/"
echo "  - Graph data: $DATA_PROCESSED/"
echo "  - Trained model: $MODELS_DIR/"
echo ""
echo "To train a different model:"
echo "  ./run_pipeline.sh --train-only --model gat --epochs 200"
echo ""
echo "To compare all models:"
echo "  for model in sage gcn gat lightgcn; do"
echo "    ./run_pipeline.sh --train-only --model \$model"
echo "  done"
