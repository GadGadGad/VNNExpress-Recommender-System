# News Recommendation System with GNN & Hybrid Features

A state-of-the-art news recommendation system leveraging Graph Neural Networks (GNN), Contrastive Learning (CL), and Semantic embeddings to provide personalized reading experiences.

## Features

- **Advanced Models**: Support for MA-HCL, SimGCL, XSimGCL, LightGCL, and various Graph Neural Networks.
- **Hybrid Intelligence**: Combines Collaborative Filtering, Content-Based retrieval (PhoBERT, TF-IDF), and Social Graph signals.
- **Premium UI**: Modern Next.js frontend with Glassmorphism design and interactive visualizations.
- **Real-time API**: High-performance FastAPI backend.
- **Semantic Atlas**: Interactive 2D visualization of user interest space.

## Installation

### Option 1: Using Mamba/Conda (Recommended)

Easily set up the environment with all dependencies including CUDA support.

```bash
mamba env create -f environment.yaml
mamba activate news-recsys
```

### Option 2: Using Pip

```bash
pip install -r requirements.txt
```

### Frontend Setup

The frontend requires Node.js 18+ and npm/yarn/pnpm.

```bash
cd frontend
npm install
# or
yarn install
```

## Quick Start

You can run the full stack (Backend + Frontend) with a single command:

```bash
chmod +x run_full_stack.sh
./run_full_stack.sh
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/docs

## Model Training

To train the collaborative filtering models:

```bash
# Train a specific model
python scripts/train_cf_models.py --model ma-hcl --epochs 50

# Or use the demo script
./train_demo.sh
```

## Project Structure

```
main/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── api/             # API Endpoints
│   │   ├── services/        # Business Logic (RecSys, Visuals)
│   │   └── core/            # Config & Security
├── frontend/                # Next.js Frontend
│   ├── app/                 # App Router Pages & Components
│   └── public/              # Static Assets
├── data/                    # Dataset Directory
├── models/                  # Trained Model Checkpoints
├── scripts/                 # Training & Utility Scripts
├── environment.yaml         # Conda Environment
├── requirements.txt         # Pip Dependencies
└── run_full_stack.sh        # Startup Script
```

## License

idk
