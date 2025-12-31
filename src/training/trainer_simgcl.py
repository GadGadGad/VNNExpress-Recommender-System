"""
SimGCL Trainer
==============
Training loop for SimGCL - simpler than LightGCL (no SVD computation)
"""

import torch
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict
import time
import os
from typing import Dict, List, Tuple, Optional


def compute_metrics(predictions: Dict[int, np.ndarray],
                    test_data: Dict[int, List[int]],
                    train_dict: Dict[int, set],
                    k_list: List[int] = [10, 20, 50]) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K, HR@K"""
    results = defaultdict(list)
    max_k = max(k_list)
    
    for user in test_data:
        if user not in train_dict:
            continue
            
        gt_items = set(test_data[user])
        if len(gt_items) == 0:
            continue
            
        scores = predictions[user].copy()
        for item in train_dict.get(user, set()):
            if item < len(scores):
                scores[item] = -np.inf
                
        top_items = np.argsort(scores)[::-1][:max_k]
        
        for k in k_list:
            top_k = top_items[:k]
            hits = len(set(top_k) & gt_items)
            
            results[f'Recall@{k}'].append(hits / len(gt_items))
            
            dcg = sum([1.0 / np.log2(i + 2) 
                      for i, item in enumerate(top_k) if item in gt_items])
            idcg = sum([1.0 / np.log2(i + 2) 
                       for i in range(min(len(gt_items), k))])
            results[f'NDCG@{k}'].append(dcg / idcg if idcg > 0 else 0)
            
            results[f'HR@{k}'].append(1.0 if hits > 0 else 0.0)
    
    return {key: np.mean(values) for key, values in results.items()}


def print_metrics(metrics: Dict[str, float], epoch: int = None):
    """Pretty print metrics"""
    if epoch is not None:
        print(f"\n  === Epoch {epoch} Evaluation ===")
    
    recalls = sorted([(k, v) for k, v in metrics.items() if 'Recall' in k])
    ndcgs = sorted([(k, v) for k, v in metrics.items() if 'NDCG' in k])
    hrs = sorted([(k, v) for k, v in metrics.items() if 'HR' in k])
    
    if recalls:
        print("  Recall:  " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in recalls]))
    if ndcgs:
        print("  NDCG:    " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in ndcgs]))
    if hrs:
        print("  HR:      " + "  ".join([f"@{k.split('@')[1]}={v:.4f}" for k, v in hrs]))


class SimGCLTrainer:
    """Trainer for SimGCL model"""
    
    def __init__(self, model, optimizer, device, n_users: int, n_items: int):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        
        self.adj_norm = None
        self.best_recall = 0
        self.best_epoch = 0
        
    def create_adj_matrix(self, train_data: List[Tuple[int, int]]) -> torch.sparse.Tensor:
        """Create normalized adjacency matrix"""
        print("\nCreating adjacency matrix...")
        
        row = np.array([u for u, i in train_data])
        col = np.array([i for u, i in train_data])
        data = np.ones(len(train_data))
        
        R = sp.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
        
        # Bipartite adjacency
        adj_size = self.n_users + self.n_items
        adj_mat = sp.dok_matrix((adj_size, adj_size), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.tocsr()
        
        print(f"  Adjacency shape: {adj_mat.shape}")
        print(f"  Non-zeros: {adj_mat.nnz:,}")
        
        # Symmetric normalization
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        
        # Convert to torch sparse
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor(np.array([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        self.adj_norm = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        
        print(f"  Normalized adj created on {self.device}")
        
        return self.adj_norm
    
    def sample_negative(self, users: List[int], 
                        train_dict: Dict[int, set]) -> List[int]:
        """Sample negative items"""
        neg_items = []
        for u in users:
            pos_items = train_dict.get(u, set())
            while True:
                neg = np.random.randint(0, self.n_items)
                if neg not in pos_items:
                    neg_items.append(neg)
                    break
        return neg_items
    
    def train_epoch(self, train_data: List[Tuple[int, int]], 
                    train_dict: Dict[int, set],
                    batch_size: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        indices = np.random.permutation(len(train_data))
        
        total_loss = 0
        total_bpr = 0
        total_reg = 0
        total_ssl = 0
        n_batches = 0
        
        pbar = tqdm(range(0, len(train_data), batch_size), 
                    desc='Training', leave=False)
        
        for start in pbar:
            end = min(start + batch_size, len(train_data))
            batch_idx = indices[start:end]
            
            users = [train_data[i][0] for i in batch_idx]
            pos_items = [train_data[i][1] for i in batch_idx]
            neg_items = self.sample_negative(users, train_dict)
            
            users_t = torch.LongTensor(users).to(self.device)
            pos_items_t = torch.LongTensor(pos_items).to(self.device)
            neg_items_t = torch.LongTensor(neg_items).to(self.device)
            
            self.optimizer.zero_grad()
            loss, bpr, reg, ssl = self.model.calculate_loss(
                self.adj_norm, users_t, pos_items_t, neg_items_t
            )
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bpr += bpr.item()
            total_reg += reg.item()
            total_ssl += ssl.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'loss': total_loss / n_batches,
            'bpr': total_bpr / n_batches,
            'reg': total_reg / n_batches,
            'ssl': total_ssl / n_batches
        }
    
    def evaluate(self, test_data: Dict[int, List[int]],
                 train_dict: Dict[int, set],
                 k_list: List[int] = [10, 20, 50],
                 batch_size: int = 1024) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            user_emb, item_emb = self.model.forward(self.adj_norm, perturb=False)
            
        user_emb_np = user_emb.cpu().numpy()
        item_emb_np = item_emb.cpu().numpy()
        
        test_users = list(test_data.keys())
        predictions = {}
        
        for i in range(0, len(test_users), batch_size):
            batch_users = test_users[i:i+batch_size]
            batch_emb = user_emb_np[batch_users]
            batch_scores = batch_emb @ item_emb_np.T
            
            for j, user in enumerate(batch_users):
                predictions[user] = batch_scores[j]
                
        return compute_metrics(predictions, test_data, train_dict, k_list)
    
    def train(self, train_data: List[Tuple[int, int]],
              train_dict: Dict[int, set],
              test_data: Dict[int, List[int]],
              n_epochs: int,
              batch_size: int,
              eval_every: int = 5,
              patience: int = 20,
              save_path: Optional[str] = None) -> Dict[str, float]:
        """Full training loop"""
        print("\n" + "=" * 60)
        print("Starting SimGCL Training")
        print("=" * 60)
        print(f"  Epochs: {n_epochs}, Batch size: {batch_size}")
        print(f"  Eval every: {eval_every}, Patience: {patience}")
        
        no_improve = 0
        best_metrics = None
        
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(train_data, train_dict, batch_size)
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch}/{n_epochs} ({epoch_time:.1f}s)")
            print(f"  Loss: {train_metrics['loss']:.4f} | "
                  f"BPR: {train_metrics['bpr']:.4f} | "
                  f"Reg: {train_metrics['reg']:.6f} | "
                  f"SSL: {train_metrics['ssl']:.4f}")
            
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate(test_data, train_dict)
                print_metrics(eval_metrics, epoch)
                
                current_recall = eval_metrics.get('Recall@20', 0)
                if current_recall > self.best_recall:
                    self.best_recall = current_recall
                    self.best_epoch = epoch
                    best_metrics = eval_metrics
                    no_improve = 0
                    
                    if save_path:
                        self.save_model(save_path, epoch, eval_metrics)
                        print("  ★ New best model saved!")
                else:
                    no_improve += eval_every
                    
                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
                    
        print("\n" + "=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Best Recall@20: {self.best_recall:.4f} at epoch {self.best_epoch}")
        
        return best_metrics
    
    def save_model(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'best_recall': float(self.best_recall)
        }, path)
        
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_recall = checkpoint.get('best_recall', 0)
        print(f"Loaded model from {path}")
        print(f"  Epoch: {checkpoint['epoch']}, Recall@20: {self.best_recall:.4f}")