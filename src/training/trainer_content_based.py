"""
Content-Based Model Trainer
============================

Training loop for PhoBERT-based content recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd


def compute_metrics(
    predictions: Dict[int, np.ndarray],
    test_data: Dict[int, List[int]],
    train_dict: Dict[int, set],
    k_list: List[int] = [10, 20, 50],
    protocol: str = 'full',
    n_items: int = None
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, HR@K using specified protocol.
    
    Args:
        protocol: 'full', 'loo100', 'cold'
    """
    results = defaultdict(list)
    max_k = max(k_list)
    
    for user in test_data:
        # Check if user has predictions
        if user not in predictions or user not in train_dict:
            continue
            
        gt_items = set(test_data[user])
        if len(gt_items) == 0:
            continue
            
        scores = predictions[user].copy()
        
        # Mask training items
        for item in train_dict.get(user, set()):
            if item < len(scores):
                scores[item] = -np.inf
                
        # Protocol-specific filtering
        if protocol == 'loo100':
            # Leave-One-Out + 100 Negatives
            # We assume gt_items has 1 item for LOO, but our split might have more.
            # We take the FIRST item in gt as the target for LOO simulation if needed,
            # but generally we just rank all GT items against 100 random negatives.
            
            # Simple simulation: Keep GT items + 100 random negatives, mask others
            all_indices = np.arange(len(scores))
            neg_indices = [i for i in all_indices 
                           if i not in gt_items and i not in train_dict.get(user, set())]
            
            if len(neg_indices) > 100:
                sampled_negs = np.random.choice(neg_indices, 100, replace=False)
                
                # Mask everything EXCEPT gt and sampled negs
                mask = np.ones_like(scores, dtype=bool) # True = mask (set to -inf)
                mask[list(gt_items)] = False
                mask[sampled_negs] = False
                
                scores[mask] = -np.inf

        # Get top K
        top_items = np.argsort(scores)[::-1][:max_k]
        
        for k in k_list:
            top_k = top_items[:k]
            hits = len(set(top_k) & gt_items)
            
            recall = hits / len(gt_items)
            results[f'Recall@{k}'].append(recall)
            
            # NDCG
            dcg = sum([1.0 / np.log2(i + 2) 
                      for i, item in enumerate(top_k) if item in gt_items])
            idcg = sum([1.0 / np.log2(i + 2) 
                       for i in range(min(len(gt_items), k))])
            ndcg = dcg / idcg if idcg > 0 else 0
            results[f'NDCG@{k}'].append(ndcg)
            
            hr = 1.0 if hits > 0 else 0.0
            results[f'HR@{k}'].append(hr)
    
    avg_results = {}
    for key, values in results.items():
        avg_results[key] = np.mean(values) if len(values) > 0 else 0.0
        
    return avg_results


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


class ContentBasedTrainer:
    """
    Trainer for Content-Based models (PhoBERT)
    """
    
    def __init__(
        self,
        model,
        device: str,
        n_users: int,
        n_items: int,
        articles_df: pd.DataFrame,
        text_columns: List[str] = ['title', 'short_description']
    ):
        self.model = model
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.articles_df = articles_df
        self.text_columns = text_columns
        
        self.article_texts = self._prepare_texts()
        
    def _prepare_texts(self) -> List[str]:
        """Prepare article texts for encoding"""
        texts = []
        for _, row in self.articles_df.iterrows():
            parts = []
            for col in self.text_columns:
                if col in row and pd.notna(row[col]):
                    parts.append(str(row[col]))
            texts.append(" ".join(parts))
        return texts
    
    def encode_articles(self, batch_size: int = 32):
        """Pre-encode all articles using PhoBERT"""
        print(f"\n[Trainer] Encoding {len(self.article_texts)} articles...")
        self.model.encode_articles(self.article_texts, batch_size=batch_size)
        
    def evaluate(
        self,
        train_dict: Dict[int, set],
        test_dict: Dict[int, List[int]],
        k_list: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """
        Evaluate content-based model
        """
        self.model.eval()
        predictions = {}
        
        print("\n[Evaluation] Computing predictions...")
        
        with torch.no_grad():
            for user_id in tqdm(test_dict.keys(), desc="Predicting"):
                history = list(train_dict.get(user_id, set()))
                user_embed = self.model.get_user_preference(history)
                
                # Normalize
                user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)
                article_embeds = F.normalize(self.model.article_embeddings, dim=-1)
                
                scores = torch.mm(user_embed, article_embeds.T).squeeze(0)
                predictions[user_id] = scores.cpu().numpy()
        
        metrics = compute_metrics(predictions, test_dict, train_dict, k_list)
        return metrics
    
    def save_model(self, path: str):
        """Save model and embeddings"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'article_embeddings': self.model.article_embeddings.cpu() 
                if self.model.article_embeddings is not None else None
        }, path)
        
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        """Load model and embeddings"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if checkpoint.get('article_embeddings') is not None:
                self.model.article_embeddings = checkpoint['article_embeddings'].to(self.device)
                
            print(f"Model loaded from {path}")
            return True
        return False


class HybridTrainer:
    """
    Trainer for Hybrid Recommender (CF + Content)
    """
    
    def __init__(
        self,
        model,
        optimizer,
        device: str,
        n_users: int,
        n_items: int,
        articles_df: pd.DataFrame = None,
        article_texts: List[str] = None,
        text_columns: List[str] = ['title', 'short_description']
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_users = n_users
        self.n_items = n_items
        self.articles_df = articles_df
        self.text_columns = text_columns
        
        if article_texts is not None:
            self.article_texts = article_texts
        elif articles_df is not None:
            self.article_texts = self._prepare_texts()
        else:
            raise ValueError("Must provide either articles_df or article_texts")
            
    def _prepare_texts(self) -> List[str]:
        texts = []
        if self.articles_df is not None:
            for _, row in self.articles_df.iterrows():
                parts = []
                for col in self.text_columns:
                    if col in row and pd.notna(row[col]):
                        parts.append(str(row[col]))
                texts.append(" ".join(parts))
        return texts
    
    def encode_articles(self, batch_size: int = 32):
        """Pre-encode all articles"""
        print(f"\n[Trainer] Encoding {len(self.article_texts)} articles...")
        self.model.encode_articles(self.article_texts, batch_size=batch_size)
        
    def train_epoch(
        self,
        train_data: List[Tuple[int, int]],
        train_dict: Dict[int, set],
        batch_size: int = 1024,
        neg_samples: int = 1
    ) -> float:
        """Train one epoch with BPR loss"""
        self.model.train()
        
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = [train_data[j] for j in batch_indices]
            
            users = torch.tensor([d[0] for d in batch_data], device=self.device)
            pos_items = torch.tensor([d[1] for d in batch_data], device=self.device)
            
            # Sample negative items
            neg_items = []
            for user_id, _ in batch_data:
                user_items = train_dict.get(user_id, set())
                neg = np.random.randint(0, self.n_items)
                while neg in user_items:
                    neg = np.random.randint(0, self.n_items)
                neg_items.append(neg)
            neg_items = torch.tensor(neg_items, device=self.device)
            
            # User histories for content
            user_histories = {
                u: list(train_dict.get(u, set())) 
                for u in users.tolist()
            }
            
            # Compute loss
            self.optimizer.zero_grad()
            loss = self.model.bpr_loss(users, pos_items, neg_items, user_histories)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def evaluate(
        self,
        train_dict: Dict[int, set],
        test_dict: Dict[int, List[int]],
        k_list: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """Evaluate hybrid model"""
        self.model.eval()
        predictions = {}
        
        print("\n[Evaluation] Computing predictions...")
        
        with torch.no_grad():
            users = list(test_dict.keys())
            
            for user_id in tqdm(users, desc="Predicting"):
                user_tensor = torch.tensor([user_id], device=self.device)
                user_histories = {user_id: list(train_dict.get(user_id, set()))}
                
                scores = self.model(user_tensor, user_histories)
                predictions[user_id] = scores.squeeze(0).cpu().numpy()
        
        metrics = compute_metrics(predictions, test_dict, train_dict, k_list)
        return metrics
    

    
    def train(
        self,
        train_data: List[Tuple[int, int]],
        train_dict: Dict[int, set],
        test_dict: Dict[int, List[int]],
        epochs: int = 100,
        batch_size: int = 1024,
        eval_every: int = 10,
        patience: int = 20,
        save_path: str = 'checkpoints/hybrid_best.pth'
    ):
        """Full training loop"""
        best_recall = 0.0
        best_epoch = 0
        best_metrics = {}
        no_improve = 0
        
        # Encode articles first
        self.encode_articles(batch_size=32)
        
        # Precompute user profiles (speed optimization)
        # Convert set to list for compatibility
        print("\n[Trainer] Precomputing user profiles for training...")
        full_histories = {k: list(v) for k, v in train_dict.items()}
        self.model.precompute_user_profiles(full_histories)
        
        print(f"\n{'='*60}")
        print(f"Starting Hybrid Training")
        print(f"  Epochs: {epochs}, Batch size: {batch_size}")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Eval every: {eval_every} epochs")
        print(f"{'='*60}")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            loss = self.train_epoch(train_data, train_dict, batch_size)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")
            
            # Evaluate
            if epoch % eval_every == 0:
                metrics = self.evaluate(train_dict, test_dict)
                print_metrics(metrics, epoch)
                
                recall_20 = metrics.get('Recall@20', 0)
                if recall_20 > best_recall:
                    best_recall = recall_20
                    best_metrics = metrics.copy()
                    best_epoch = epoch
                    no_improve = 0
                    self.save_model(save_path)
                    print(f"  * New best Recall@20: {best_recall:.4f}")
                else:
                    no_improve += 1
                    
                if no_improve >= patience // eval_every:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
                    
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Recall@20: {best_recall:.4f} at epoch {best_epoch}")
        
        if best_metrics:
            print("\nFINAL_METRICS")
            print_metrics(best_metrics)
            
        print(f"{'='*60}")
        
        return best_recall
    
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'article_embeddings': self.model.article_content_embeddings.cpu()
                if self.model.article_content_embeddings is not None else None
        }, path)
        print(f"Model saved to {path}")
        
    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint.get('article_embeddings') is not None:
                self.model.article_content_embeddings = checkpoint['article_embeddings'].to(self.device)
            print(f"Model loaded from {path}")
            return True
        return False
