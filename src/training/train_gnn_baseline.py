"""
GNN Baseline Training & Evaluation Pipeline
============================================
Comprehensive training script for recommendation models on VnExpress data.

Models:
    - GraphSAGE (default baseline)
    - GCN (Graph Convolutional Network)
    - GAT (Graph Attention Network)
    - LightGCN (Collaborative Filtering)

Metrics:
    - Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR
    - AUC-ROC for link prediction

Usage:
    python src/train_gnn_baseline.py --model sage --epochs 100 --k 10
    python src/train_gnn_baseline.py --model gat --hidden-dim 128 --lr 0.001
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# PyTorch Geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, LGConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class GraphSAGEEncoder(nn.Module):
    """GraphSAGE-based encoder for heterogeneous graphs."""
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        # First layer
        self.convs.append(SAGEConv((-1, -1), hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv((-1, -1), out_dim))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GCNEncoder(nn.Module):
    """GCN-based encoder."""
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GCNConv(-1, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, out_dim))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    """GAT-based encoder with multi-head attention."""
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2, 
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GATConv((-1, -1), hidden_dim // heads, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class LightGCNEncoder(nn.Module):
    """
    LightGCN: Simplified GCN without feature transformation and nonlinearity.
    FIXED: Uses LGConv to be compatible with to_hetero() tracing.
    """
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        # Use PyG's built-in LGConv which handles normalization and message passing correctly
        # and is compatible with to_hetero tracing
        self.conv = LGConv() 
        self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        
    def forward(self, x, edge_index):
        # Simple propagation without transformation
        all_embeddings = [x]
        
        for _ in range(self.num_layers):
            # LGConv handles the normalized adjacency propagation
            x = self.conv(x, edge_index)
            all_embeddings.append(x)
        
        # Mean pooling over all layers (key insight of LightGCN)
        x = torch.stack(all_embeddings, dim=0).mean(dim=0)
        x = self.out_proj(x)
        return x


def get_model(model_name: str, hidden_dim: int, out_dim: int, 
              num_layers: int = 2, dropout: float = 0.3) -> nn.Module:
    """Factory function to create models."""
    models = {
        'sage': GraphSAGEEncoder,
        'gcn': GCNEncoder,
        'gat': GATEncoder,
        'lightgcn': LightGCNEncoder,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](hidden_dim, out_dim, num_layers, dropout)


# ============================================================================
# METRICS
# ============================================================================

class RecommenderMetrics:
    """Comprehensive evaluation metrics for recommender systems."""
    
    @staticmethod
    def compute_all(
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        ground_truth: Dict[int, set],
        k_values: List[int] = [5, 10, 20],
        exclude_train: Optional[Dict[int, set]] = None
    ) -> Dict[str, float]:
        """
        Compute all ranking metrics.
        
        Args:
            user_embeddings: (num_users, dim)
            item_embeddings: (num_items, dim)
            ground_truth: {user_idx: set(item_indices)}
            k_values: List of K values for top-K metrics
            exclude_train: Items to exclude (training interactions)
        """
        results = {}
        
        for k in k_values:
            precision_list = []
            recall_list = []
            ndcg_list = []
            hit_list = []
            mrr_list = []
            
            for user_idx, true_items in ground_truth.items():
                if len(true_items) == 0:
                    continue
                
                # Compute scores
                scores = (user_embeddings[user_idx].unsqueeze(0) @ item_embeddings.T).squeeze()
                
                # Exclude training items if provided
                if exclude_train and user_idx in exclude_train:
                    for item in exclude_train[user_idx]:
                        scores[item] = float('-inf')
                
                # Get top-K recommendations
                _, top_k_indices = torch.topk(scores, min(k, len(scores)))
                recommendations = set(top_k_indices.cpu().numpy().tolist())
                
                # Calculate metrics
                hits = recommendations & true_items
                num_hits = len(hits)
                
                # Precision@K
                precision_list.append(num_hits / k)
                
                # Recall@K
                recall_list.append(num_hits / len(true_items))
                
                # Hit Rate@K
                hit_list.append(1.0 if num_hits > 0 else 0.0)
                
                # NDCG@K
                dcg = 0.0
                for rank, item_idx in enumerate(top_k_indices.cpu().numpy()):
                    if item_idx in true_items:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
                
                # MRR (Mean Reciprocal Rank)
                mrr = 0.0
                for rank, item_idx in enumerate(top_k_indices.cpu().numpy()):
                    if item_idx in true_items:
                        mrr = 1.0 / (rank + 1)
                        break
                mrr_list.append(mrr)
            
            results[f'Precision@{k}'] = np.mean(precision_list)
            results[f'Recall@{k}'] = np.mean(recall_list)
            results[f'NDCG@{k}'] = np.mean(ndcg_list)
            results[f'HitRate@{k}'] = np.mean(hit_list)
            results[f'MRR@{k}'] = np.mean(mrr_list)
        
        return results
    
    @staticmethod
    def compute_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
        """Compute AUC-ROC for link prediction."""
        pos_scores = pos_scores.cpu().numpy()
        neg_scores = neg_scores.cpu().numpy()
        
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        scores = np.concatenate([pos_scores, neg_scores])
        
        return roc_auc_score(labels, scores)


# ============================================================================
# TRAINER
# ============================================================================

class GNNTrainer:
    """Comprehensive trainer for GNN recommendation models."""
    
    def __init__(
        self,
        model: nn.Module,
        data: HeteroData,
        edge_type: Tuple[str, str, str],
        device: torch.device,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        neg_sampling_ratio: float = 1.0,
        scheduler_type: str = 'plateau'
    ):
        self.model = model.to(device)
        self.device = device
        self.edge_type = edge_type
        self.neg_ratio = neg_sampling_ratio
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Split data
        self.train_data, self.val_data, self.test_data = self._split_data(data)
        
        # Move to device
        self.train_data = self.train_data.to(device)
        self.val_data = self.val_data.to(device)
        self.test_data = self.test_data.to(device)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
    def _split_data(self, data: HeteroData):
        """Split data into train/val/test."""
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
            neg_sampling_ratio=self.neg_ratio,
            edge_types=[self.edge_type],
            rev_edge_types=[(self.edge_type[2], f'rev_{self.edge_type[1]}', self.edge_type[0])]
        )
        return transform(data)
    
    def _compute_loss(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute BPR loss with negative sampling."""
        # Forward pass
        z_dict = self.model(data.x_dict, data.edge_index_dict)
        
        # Positive edges
        pos_edge_index = data[self.edge_type].edge_label_index
        user_emb = z_dict['user'][pos_edge_index[0]]
        item_emb = z_dict['article'][pos_edge_index[1]]
        pos_scores = (user_emb * item_emb).sum(dim=-1)
        
        # Negative sampling
        num_neg = int(pos_edge_index.size(1) * self.neg_ratio)
        neg_users = torch.randint(0, data['user'].num_nodes, (num_neg,), device=self.device)
        neg_items = torch.randint(0, data['article'].num_nodes, (num_neg,), device=self.device)
        
        neg_user_emb = z_dict['user'][neg_users]
        neg_item_emb = z_dict['article'][neg_items]
        neg_scores = (neg_user_emb * neg_item_emb).sum(dim=-1)
        
        # BPR Loss: -log(sigmoid(pos - neg))
        # BCE Loss variant for simplicity
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        
        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        
        return loss, pos_scores, neg_scores
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, _, _ = self._compute_loss(self.train_data)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        
        loss, pos_scores, neg_scores = self._compute_loss(self.val_data)
        auc = RecommenderMetrics.compute_auc(pos_scores, neg_scores)
        
        return loss.item(), auc
    
    def train(self, epochs: int, patience: int = 20, verbose: bool = True) -> Dict:
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        pbar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in pbar:
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'val_auc': f'{val_auc:.4f}'
                })
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    @torch.no_grad()
    def evaluate(self, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Full evaluation on test set."""
        self.model.eval()
        
        # Get embeddings
        z_dict = self.model(self.test_data.x_dict, self.test_data.edge_index_dict)
        user_emb = z_dict['user']
        item_emb = z_dict['article']
        
        # Build ground truth from test set
        edge_index = self.test_data[self.edge_type].edge_label_index
        edge_label = self.test_data[self.edge_type].edge_label
        
        pos_edges = edge_index[:, edge_label == 1]
        ground_truth = {}
        
        for u, i in zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()):
            if u not in ground_truth:
                ground_truth[u] = set()
            ground_truth[u].add(i)
        
        # Build training exclusion set
        train_edges = self.train_data[self.edge_type].edge_index
        exclude_train = {}
        for u, i in zip(train_edges[0].cpu().numpy(), train_edges[1].cpu().numpy()):
            if u not in exclude_train:
                exclude_train[u] = set()
            exclude_train[u].add(i)
        
        # Compute metrics
        metrics = RecommenderMetrics.compute_all(
            user_emb, item_emb, ground_truth, k_values, exclude_train
        )
        
        # Add AUC
        _, pos_scores, neg_scores = self._compute_loss(self.test_data)
        metrics['AUC'] = RecommenderMetrics.compute_auc(pos_scores, neg_scores)
        
        return metrics


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(args):
    """Run a complete training experiment."""
    print("=" * 70)
    print(f"GNN Baseline Training - {args.model.upper()}")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}. Run convert_to_gnn.py first!")
    
    data = torch.load(args.data_path, weights_only=False)
    print(f"   Users: {data['user'].x.shape[0]}")
    print(f"   Articles: {data['article'].x.shape[0]}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    base_model = get_model(
        args.model,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Convert to heterogeneous
    model = to_hetero(base_model, data.metadata(), aggr='sum')
    
    # Create trainer
    edge_type = ('user', 'comments', 'article')
    trainer = GNNTrainer(
        model=model,
        data=data,
        edge_type=edge_type,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        neg_sampling_ratio=args.neg_ratio,
        scheduler_type='plateau'
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    history = trainer.train(epochs=args.epochs, patience=args.patience)
    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.1f}s")
    
    # Evaluate
    print("\nEvaluating on test set...")
    k_values = [int(k) for k in args.k_values.split(',')]
    metrics = trainer.evaluate(k_values=k_values)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<20} {value:.4f}")
    
    # Save results
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = save_dir / f"{args.model}_{timestamp}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'model': args.model,
                'hidden_dim': args.hidden_dim,
                'out_dim': args.out_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            },
            'metrics': metrics,
            'history': history
        }, model_path)
        print(f"\nModel saved: {model_path}")
        
        # Save metrics
        results = {
            'model': args.model,
            'config': vars(args),
            'metrics': metrics,
            'train_time': train_time,
            'timestamp': timestamp
        }
        
        results_path = save_dir / f"results_{args.model}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GNN baseline models')
    
    # Data
    parser.add_argument('--data-path', '-d', 
                        default='data/processed/user_article_graph.pt',
                        help='Path to processed graph data')
    
    # Model
    parser.add_argument('--model', '-m', 
                        choices=['sage', 'gcn', 'gat', 'lightgcn'],
                        default='sage',
                        help='Model architecture')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--out-dim', type=int, default=32,
                        help='Output embedding dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--neg-ratio', type=float, default=1.0,
                        help='Negative sampling ratio')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Evaluation
    parser.add_argument('--k-values', type=str, default='5,10,20',
                        help='Comma-separated K values for top-K metrics')
    
    # Output
    parser.add_argument('--save-dir', '-o', default='models',
                        help='Directory to save model and results')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == "__main__":
    main()