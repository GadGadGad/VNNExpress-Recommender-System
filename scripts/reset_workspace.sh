#!/bin/bash
# Clean workspace script

echo "Cleaning processed data and models..."

#
find data/processed -name "*.pt" ! -name "phobert_embeddings.pt" -delete
find data/processed -name "*.json" -delete
find data/processed -name "*.pkl" -delete

# 2. Clear models
rm -rf models/*.pt
rm -rf models/*.json

# 3. Clear logs and caches
find . -name "__pycache__" -type d -exec rm -rf {} +
rm -f benchmark.log

echo "Done! Workspace cleaned."
