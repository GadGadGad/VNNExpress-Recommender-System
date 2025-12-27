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

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, GraphConv, GATConv, to_hetero
from torch_geometric.transforms import RandomLinkSplit


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE-based encoder for heterogeneous graphs."""
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(SAGEConv((-1, -1), hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
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
    """GCN-based encoder using GraphConv (supports bipartite graphs)."""
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GraphConv((-1, -1), hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv((-1, -1), hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GraphConv((-1, -1), out_dim))
        
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
                 dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        heads = int(heads)
        head_dim = hidden_dim // heads
        
        self.convs.append(GATConv((-1, -1), head_dim, heads=heads, concat=True, add_self_loops=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, head_dim, heads=heads, concat=True, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False))
        
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
    Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network"
    """
    
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()
        
    def forward(self, x, edge_index):
        all_embeddings = [x]
        
        for _ in range(self.num_layers):
            x = self._propagate(x, edge_index)
            all_embeddings.append(x)
        
        x = torch.stack(all_embeddings, dim=0).mean(dim=0)
        x = self.out_proj(x)
        return x
    
    def _propagate(self, x, edge_index):
        """Simple message passing (normalized sum)."""
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * norm.unsqueeze(-1))
        return out


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
                
                scores = (user_embeddings[user_idx].unsqueeze(0) @ item_embeddings.T).squeeze()
                
                if exclude_train and user_idx in exclude_train:
                    for item in exclude_train[user_idx]:
                        scores[item] = float('-inf')
                
                _, top_k_indices = torch.topk(scores, min(k, len(scores)))
                recommendations = set(top_k_indices.cpu().numpy().tolist())
                
                hits = recommendations & true_items
                num_hits = len(hits)
                
                precision_list.append(num_hits / k)
                recall_list.append(num_hits / len(true_items))
                hit_list.append(1.0 if num_hits > 0 else 0.0)
                
                dcg = 0.0
                for rank, item_idx in enumerate(top_k_indices.cpu().numpy()):
                    if item_idx in true_items:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
                
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
        scheduler_type: str = 'plateau',
        # --- NEW PARAMS ---
        splits: Optional[Dict] = None,   # Nhận split từ bên ngoài
        neg_strategy: str = 'random'     # Chiến lược negative
    ):
        self.model = model.to(device)
        self.device = device
        self.edge_type = edge_type
        self.neg_ratio = neg_sampling_ratio
        self.neg_strategy = neg_strategy
        self.splits = splits
        
        # Optimizer & Scheduler (Giữ nguyên)
        params = list(model.parameters())
        for node_type in data.node_types:
            if isinstance(data[node_type].x, nn.Parameter):
                params.append(data[node_type].x)
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        # --- FIX ERROR 2: Dùng Split có sẵn (Time-based) thay vì RandomLinkSplit ---
        if self.splits is not None:
            print("   -> Using pre-computed splits (Time-based / Hard Negatives)")
            # Gán trực tiếp dữ liệu đã split
            self.train_data = data # Bản gốc chứa tất cả node features
            self.val_data = data
            self.test_data = data
        else:
            print("   -> [WARNING] Using RandomLinkSplit (Might cause Data Leakage!)")
            self.train_data, self.val_data, self.test_data = self._split_data(data)

        self.train_data = self.train_data.to(device)
        self.val_data = self.val_data.to(device)
        self.test_data = self.test_data.to(device)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        
    def _split_data(self, data: HeteroData):
        """Split data into train/val/test (Old Random Method - Fallback)."""
        transform = RandomLinkSplit(
            num_val=0.1, num_test=0.1, is_undirected=True,
            add_negative_train_samples=False, neg_sampling_ratio=self.neg_ratio,
            edge_types=[self.edge_type],
            rev_edge_types=[(self.edge_type[2], f'rev_{self.edge_type[1]}', self.edge_type[0])]
        )
        return transform(data)
    
    def _compute_loss(self, data: HeteroData, split_name='train') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute BPR loss with support for Precomputed Negatives."""
        z_dict = self.model(data.x_dict, data.edge_index_dict)
        
        # 1. Lấy Positive Edges (Tương tác thật)
        if self.splits is not None:
            # Lấy từ dictionary splits truyền vào
            pos_u = self.splits[split_name]['pos_users'].to(self.device)
            pos_i = self.splits[split_name]['pos_articles'].to(self.device)
        else:
            # Lấy từ RandomLinkSplit
            pos_edge_index = data[self.edge_type].edge_label_index
            pos_u, pos_i = pos_edge_index[0], pos_edge_index[1]

        user_emb = z_dict['user'][pos_u]
        item_emb = z_dict['article'][pos_i]
        pos_scores = (user_emb * item_emb).sum(dim=-1)
        
        # 2. Lấy Negative Edges (Tương tác giả)
        # --- FIX ERROR 3: Hỗ trợ Precomputed Negatives ---
        if self.neg_strategy == 'precomputed' and self.splits is not None:
            neg_u = self.splits[split_name]['neg_users'].to(self.device)
            neg_i = self.splits[split_name]['neg_articles'].to(self.device)
            
            # Cắt ngắn nếu kích thước không khớp (do batching hoặc ratio)
            min_len = min(len(pos_u), len(neg_u))
            pos_scores = pos_scores[:min_len]
            
            neg_user_emb = z_dict['user'][neg_u[:min_len]]
            neg_item_emb = z_dict['article'][neg_i[:min_len]]
            
        else:
            # Cách cũ: Random ngẫu nhiên
            num_neg = int(len(pos_u) * self.neg_ratio)
            neg_users = torch.randint(0, data['user'].num_nodes, (num_neg,), device=self.device)
            neg_items = torch.randint(0, data['article'].num_nodes, (num_neg,), device=self.device)
            
            neg_user_emb = z_dict['user'][neg_users]
            neg_item_emb = z_dict['article'][neg_items]
            
        neg_scores = (neg_user_emb * neg_item_emb).sum(dim=-1)
        
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        return loss, pos_scores, neg_scores
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, _, _ = self._compute_loss(self.train_data, 'train')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        
        loss, pos_scores, neg_scores = self._compute_loss(self.val_data, 'val')
        auc = RecommenderMetrics.compute_auc(pos_scores, neg_scores)
        
        return loss.item(), auc
    
    def train(self, epochs: int, patience: int = 20, verbose: bool = True) -> Dict:
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        pbar = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in pbar:
            train_loss = self.train_epoch()
            val_loss, val_auc = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
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
        
        z_dict = self.model(self.test_data.x_dict, self.test_data.edge_index_dict)
        user_emb = z_dict['user']
        item_emb = z_dict['article']
        
        edge_index = self.test_data[self.edge_type].edge_label_index
        edge_label = self.test_data[self.edge_type].edge_label
        
        pos_edges = edge_index[:, edge_label == 1]
        ground_truth = {}
        for u, i in zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()):
            if u not in ground_truth:
                ground_truth[u] = set()
            ground_truth[u].add(i)
        
        train_edges = self.train_data[self.edge_type].edge_index
        exclude_train = {}
        for u, i in zip(train_edges[0].cpu().numpy(), train_edges[1].cpu().numpy()):
            if u not in exclude_train:
                exclude_train[u] = set()
            exclude_train[u].add(i)
        
        metrics = RecommenderMetrics.compute_all(
            user_emb, item_emb, ground_truth, k_values, exclude_train
        )
        
        _, pos_scores, neg_scores = self._compute_loss(self.test_data)
        metrics['AUC'] = RecommenderMetrics.compute_auc(pos_scores, neg_scores)
        
        return metrics


def run_experiment(args):
    """Run a complete training experiment."""
    print("=" * 70)
    print(f"GNN Baseline Training - {args.model.upper()}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")
    
    print(f"\nLoading data from {args.data_path}...")
    loaded_content = torch.load(args.data_path, weights_only=False)
    
    splits = None
    if isinstance(loaded_content, dict) and 'graph' in loaded_content and 'splits' in loaded_content:
        print("   -> Detected dictionary format with pre-computed SPLITS.")
        data = loaded_content['graph']
        splits = loaded_content['splits']
    else:
        print("   -> Detected raw HeteroData (will use Random Split).")
        data = loaded_content
    print(f"   Users: {data['user'].x.shape[0]}")
    print(f"   Articles: {data['article'].x.shape[0]}")
    
    print(f"\nCreating {args.model.upper()} model...")
    base_model = get_model(
        args.model,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print("   -> Initializing node features...")
    for node_type in data.node_types:
        x = data[node_type].x
        num_nodes, feat_dim = x.shape
        
        if node_type == 'user' or feat_dim != args.hidden_dim:
            print(f"      [{node_type}] Replacing features ({feat_dim}) with learnable embeddings ({args.hidden_dim})")
            data[node_type].x = nn.Parameter(torch.randn(num_nodes, args.hidden_dim) * 0.1)
        else:
            print(f"      [{node_type}] Converting features ({feat_dim}) to learnable parameters")
            data[node_type].x = nn.Parameter(x.clone())

    model = to_hetero(base_model, data.metadata(), aggr='sum')
    
    edge_type = ('user', 'comments', 'article')
    trainer = GNNTrainer(
        model=model,
        data=data,
        edge_type=edge_type,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        neg_sampling_ratio=args.neg_ratio,
        splits=splits,
        neg_strategy=args.neg_strategy
    )
    
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    history = trainer.train(epochs=args.epochs, patience=args.patience)
    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.1f}s")
    
    print("\nEvaluating on test set...")
    k_values = [int(k) for k in args.k_values.split(',')]
    metrics = trainer.evaluate(k_values=k_values)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<20} {value:.4f}")
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = save_dir / f"{args.model}_{timestamp}.pt"
        torch.save({
            'node_embeddings': {nt: data[nt].x for nt in data.node_types},
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

    # Explicit save-results path for external tools
    if args.save_results:
        # Convert numpy types to native python types for JSON serialization
        def convert(o):
            if isinstance(o, (np.int64, np.int32)): return int(o)
            if isinstance(o, (np.float64, np.float32)): return float(o)
            return o
            
        with open(args.save_results, 'w') as f:
            json.dump(metrics, f, default=convert)
        print(f"Metrics saved to explicit path: {args.save_results}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train GNN baseline models')
    
    parser.add_argument('--data-path', '-d', default='data/processed/user_article_graph.pt',
                        help='Path to processed graph data')
    parser.add_argument('--model', '-m', choices=['sage', 'gcn', 'gat', 'lightgcn'],
                        default='sage', help='Model architecture')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--out-dim', type=int, default=32, help='Output embedding dimension')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--neg-ratio', type=float, default=1.0, help='Negative sampling ratio')
    parser.add_argument('--neg-strategy', choices=['random', 'precomputed'], default='random', 
                        help='Strategy: random (default) or precomputed (load from file)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--k-values', type=str, default='5,10,20', help='K values for top-K metrics')
    parser.add_argument('--save-dir', '-o', default='models', help='Directory to save model')
    parser.add_argument('--save-results', type=str, default=None, help='Path to save metrics in JSON format')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    
    run_experiment(args)


if __name__ == "__main__":
    main()