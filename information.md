# News Recommendation Information & Technical Summary

## 1. Project Objective

Objective: Develop a SOTA news recommendation system designed to handle **extreme data sparsity** and **noisy feedback** in the Vietnamese news landscape.

## 2. Technical Pillars (Solving Sparsity)

The project is built on three central pillars that bridge the gap between collaborative filtering and content understanding:

| Pillar | Technique | Purpose |
| :--- | :--- | :--- |
| **Pillar 1** | **Discrete Semantic IDs** | Uses RQ-VAE to convert high-dim PhoBERT embeddings into discrete tokens (Semantic IDs), providing a robust "content-based" anchor for graph models. |
| **Pillar 2** | **LLM User Priors** | Pre-computes user interest anchors from historical interaction text, providing a dense starting point for cold-start users. |
| **Pillar 3** | **Adaptive Denoising (ADT)** | Implements a Truncated Loss mechanism during early training to prune or downweight low-confidence interactions (noisy comments). |

## 3. Model Architecture (Technical Champion)
We utilize **XSimGCL** (eXtremely Simple Graph Contrastive Learning) as our champion architecture:
- **Noise-based Augment**: Creates contrastive views by adding small Gaussian noise to latent embeddings instead of manual graph dropout.
- **Gated Fusion**: Adaptively weighs the influence of Collaborative Filtering (interactions), Semantic IDs (item content), and User Priors (user context).

## 4. Graph Construction
The system supports multiple graph views to capture different signal types:
- **User-Article (Bipartite)**: The core interaction graph.
- **User-User**: Shared interest connections.
- **Article-Article**: Category-based and commentator-based connections.
- **Heterogeneous**: A unified graph combining all node and edge types.

## 5. Post-Processing & Personalization
Beyond the model, we use a **Calibrated & Diverse Re-ranker** to improve user experience:
- **MMR (Maximal Marginal Relevance)**: Promotes topical diversity to prevent filter bubbles.
- **KL-Divergence Calibration**: Aligns the categorical distribution of recommendations with the user's historical preferences.

## 6. Final Leaderboard (Technical Results)

| Model Configuration | Recall@10 | NDCG@10 | Topical Entropy |
| :--- | :--- | :--- | :--- |
| **XSimGCL + Pillars + ADT** | **6.44%** | **0.037** | **0.93** |
| XSimGCL + Pillars | 6.50% | 0.038 | 0.82 |
| LightGCL (Baseline) | 5.30% | 0.029 | 0.78 |
| TF-IDF (Content-Only) | 4.10% | 0.025 | 0.95 |

## 7. Key File Directory
- `app.py`: Streamlit Dashboard (Demo & Controls).
- `scripts/train_cf_models.py`: Main training and benchmarking engine.
- `src/models/xsimgcl.py`: Implementation of the champion model with technical pillars.
- `src/inference/re_ranker.py`: Calibrated diversity logic.
- `src/data/convert_to_gnn.py`: Graph construction pipeline.
- `data/processed/`: Processed graphs and caches.
- `models/`: Best model checkpoints (.pt).

## 8. Usage Quickstart

- **Run Dashboard**: `streamlit run app.py`
- **Generate Graph**: `python src/data/convert_to_gnn.py --graph-type all --add-text-features`
- **Train Champion**: `python scripts/train_cf_models.py --model xsimgcl --semantic-id-bits 3 --denoise-ratio 0.1 --rerank calib`

## 9. Master Execution Guide (Run ALL Cases)

### A. Data Preparation & Crawling

```bash
# Run full crawler pipeline (Crawling -> Metadata -> User Profiles)
bash crawlers/run_pipeline.sh

# Merge all crawled categories into unified datasets
python scripts/merge_data.py
```

### B. End-to-End RecSys Pipeline

```bash
# Full Pipeline: Merge -> EDA -> Train All Models -> Comparison
python scripts/run_all.py --full --epochs 50

# Quick Test: Skip EDA, run top 3 models, 10 epochs
python scripts/run_all.py --quick
```

### C. Comprehensive Benchmarking

```bash
# Compare ALL Collaborative Filtering models (NGCF, LightGCL, SimGCL, XSimGCL, etc.)
python scripts/train_and_compare_all.py --epochs 50 --batch-size 2048 --prepare

# Run all Content-Based & Hybrid variants
python scripts/run_content_based.py --model all --epochs 10

# Evaluate multi-strategy re-ranking (MMR vs Calibrated)
python scripts/eval_strategies.py --model xsimgcl --k 10
```

### D. Technical Champion Ablations (Phase 6)

```bash
# Case 1: Base Graph + Semantic IDs (Pillar 1)
python scripts/train_cf_models.py --model xsimgcl --semantic-id-bits 3

# Case 2: Base Graph + User Priors (Pillar 2)
# (Automatically loads data/processed/user_priors.pt if present)
python scripts/train_cf_models.py --model xsimgcl

# Case 3: Champion Flow (Pillars 1+2 + Denoising + Calibration)
python scripts/train_cf_models.py --model xsimgcl --semantic-id-bits 3 --denoise-ratio 0.1 --rerank calib
```

---

*Last Updated: 2025-12-25. For detailed architecture details, see [walkthrough.md](file:///home/gad/.gemini/antigravity/brain/d22cdfd2-6c46-43aa-81bf-4c88aee541b9/walkthrough.md).*
