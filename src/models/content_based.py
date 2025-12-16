"""
Content-Based Models for Vietnamese News Recommendation
========================================================

Models:
1. PhoBERTEncoder - Encode articles using PhoBERT
2. ContentBasedRecommender - Pure content-based recommendation
3. HybridRecommender - Combine collaborative filtering + content-based

Pretrained models for Vietnamese:
- vinai/phobert-base (recommended)
- vinai/phobert-large
- VoVanPhuc/sup-SimCSE-VietNamese-phobert-base (for sentence similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class PhoBERTEncoder(nn.Module):
    """
    Encode Vietnamese text using PhoBERT
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        embedding_dim: int = 256,
        max_length: int = 256,
        pooling: str = "mean",  # "mean", "cls", "max"
        freeze_bert: bool = False,
        device: str = "cuda"
    ):
        super(PhoBERTEncoder, self).__init__()
        
        from transformers import AutoModel, AutoTokenizer
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.pooling = pooling
        self.device = device
        
        print(f"\n[PhoBERTEncoder] Loading {model_name}...")
        
        # Load pretrained PhoBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # PhoBERT hidden size (768 for base, 1024 for large)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # Projection layer: 768 -> embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.bert_hidden_size, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        if freeze_bert:
            print("  Freezing BERT parameters...")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        print(f"  BERT hidden size: {self.bert_hidden_size}")
        print(f"  Output embedding dim: {embedding_dim}")
        print(f"  Pooling strategy: {pooling}")
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode list of texts to embeddings
        
        Args:
            texts: List of Vietnamese text strings
            
        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Forward through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Pooling
        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling == "max":
            # Max pooling
            hidden_states[attention_mask == 0] = -1e9
            pooled, _ = torch.max(hidden_states, dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Project to embedding_dim
        embeddings = self.projection(pooled)
        
        return embeddings
    
    @torch.no_grad()
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode large list of texts in batches
        """
        self.eval()
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
        else:
            iterator = range(0, len(texts), batch_size)
            
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            embeddings = self.forward(batch_texts)
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)


class ContentBasedRecommender(nn.Module):
    """
    Pure Content-Based Recommendation using PhoBERT
    
    Recommends articles based on:
    1. User's history (mean of read articles embeddings)
    2. Article content similarity
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 256,
        bert_model: str = "vinai/phobert-base",
        max_length: int = 256,
        freeze_bert: bool = True,
        device: str = "cuda"
    ):
        super(ContentBasedRecommender, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Article encoder (PhoBERT)
        self.article_encoder = PhoBERTEncoder(
            model_name=bert_model,
            embedding_dim=embedding_dim,
            max_length=max_length,
            pooling="mean",
            freeze_bert=freeze_bert,
            device=device
        )
        
        # User preference encoder
        # Aggregates user's reading history into preference embedding
        self.user_preference_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Precomputed article embeddings cache
        self.article_embeddings: Optional[torch.Tensor] = None
        
        print(f"\n[ContentBasedRecommender] Initialized")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  Embedding dim: {embedding_dim}")
        
    def encode_articles(
        self,
        article_texts: List[str],
        batch_size: int = 32
    ):
        """
        Pre-encode all articles and cache embeddings
        """
        print(f"\n[ContentBasedRecommender] Encoding {len(article_texts)} articles...")
        
        embeddings = self.article_encoder.encode_batch(
            article_texts,
            batch_size=batch_size,
            show_progress=True
        )
        
        self.article_embeddings = torch.tensor(embeddings, device=self.device)
        print(f"  Cached embeddings shape: {self.article_embeddings.shape}")
        
    def get_user_preference(
        self,
        user_history: List[int],
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        Compute user preference embedding from reading history
        
        Args:
            user_history: List of article indices user has read
            aggregation: "mean", "attention", "last"
        """
        if self.article_embeddings is None:
            raise ValueError("Articles not encoded! Call encode_articles() first.")
            
        if len(user_history) == 0:
            # Cold start: return zeros
            return torch.zeros(self.embedding_dim, device=self.device)
        
        # Get embeddings of read articles
        history_embeddings = self.article_embeddings[user_history]  # [n_history, dim]
        
        if aggregation == "mean":
            user_embed = history_embeddings.mean(dim=0)
        elif aggregation == "last":
            user_embed = history_embeddings[-1]
        else:
            user_embed = history_embeddings.mean(dim=0)
            
        # Transform through user encoder
        user_embed = self.user_preference_encoder(user_embed)
        
        return user_embed
    
    def forward(
        self,
        user_histories: Dict[int, List[int]],
        users: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scores for all items for given users
        
        Returns:
            scores: [batch_size, n_items]
        """
        batch_size = len(users)
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        for i, user_id in enumerate(users.tolist()):
            history = user_histories.get(user_id, [])
            user_embed = self.get_user_preference(history)  # [dim]
            
            # Cosine similarity with all articles
            user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)  # [1, dim]
            article_embeds = F.normalize(self.article_embeddings, dim=-1)  # [n_items, dim]
            
            scores[i] = torch.mm(user_embed, article_embeds.T).squeeze(0)
            
        return scores
    
    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_read: bool = True
    ) -> Tuple[List[int], List[float]]:
        """
        Recommend top-k articles for a user
        """
        user_embed = self.get_user_preference(user_history)
        user_embed = F.normalize(user_embed.unsqueeze(0), dim=-1)
        article_embeds = F.normalize(self.article_embeddings, dim=-1)
        
        scores = torch.mm(user_embed, article_embeds.T).squeeze(0)  # [n_items]
        
        if exclude_read:
            scores[user_history] = -float('inf')
            
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        return top_k_indices.tolist(), top_k_scores.tolist()
    
    def save_embeddings(self, path: str):
        """Save cached article embeddings"""
        if self.article_embeddings is not None:
            torch.save(self.article_embeddings.cpu(), path)
            print(f"Saved article embeddings to {path}")
            
    def load_embeddings(self, path: str):
        """Load cached article embeddings"""
        if os.path.exists(path):
            self.article_embeddings = torch.load(path).to(self.device)
            print(f"Loaded article embeddings from {path}")
            return True
        return False


class HybridRecommender(nn.Module):
    """
    Hybrid Recommendation: Collaborative Filtering + Content-Based
    
    Combines:
    1. User-Item embeddings (collaborative signal)
    2. Article content embeddings (PhoBERT)
    
    Final score = alpha * CF_score + (1-alpha) * Content_score
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cf_embedding_dim: int = 64,
        content_embedding_dim: int = 256,
        bert_model: str = "vinai/phobert-base",
        alpha: float = 0.5,  # Weight for CF vs Content
        freeze_bert: bool = True,
        device: str = "cuda"
    ):
        super(HybridRecommender, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.cf_embedding_dim = cf_embedding_dim
        self.content_embedding_dim = content_embedding_dim
        self.alpha = alpha
        self.device = device
        
        # ========== Collaborative Filtering Component ==========
        self.user_cf_embedding = nn.Embedding(n_users, cf_embedding_dim)
        self.item_cf_embedding = nn.Embedding(n_items, cf_embedding_dim)
        
        nn.init.xavier_uniform_(self.user_cf_embedding.weight)
        nn.init.xavier_uniform_(self.item_cf_embedding.weight)
        
        # ========== Content-Based Component ==========
        self.article_encoder = PhoBERTEncoder(
            model_name=bert_model,
            embedding_dim=content_embedding_dim,
            pooling="mean",
            freeze_bert=freeze_bert,
            device=device
        )
        
        # User content preference (aggregated from read articles)
        self.user_content_projection = nn.Sequential(
            nn.Linear(content_embedding_dim, content_embedding_dim),
            nn.LayerNorm(content_embedding_dim),
            nn.GELU()
        )
        
        # ========== Fusion Layer ==========
        # Learns to combine CF and Content scores
        self.fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Cached embeddings
        self.article_content_embeddings: Optional[torch.Tensor] = None
        
        print(f"\n[HybridRecommender] Initialized")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  CF dim: {cf_embedding_dim}, Content dim: {content_embedding_dim}")
        print(f"  Alpha (CF weight): {alpha}")
        
    def encode_articles(self, article_texts: List[str], batch_size: int = 32):
        """Pre-encode all articles"""
        print(f"\n[HybridRecommender] Encoding {len(article_texts)} articles...")
        
        embeddings = self.article_encoder.encode_batch(
            article_texts,
            batch_size=batch_size,
            show_progress=True
        )
        
        self.article_content_embeddings = torch.tensor(embeddings, device=self.device)
        
    def get_cf_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Get collaborative filtering scores"""
        user_embeds = self.user_cf_embedding(users)  # [batch, cf_dim]
        item_embeds = self.item_cf_embedding.weight  # [n_items, cf_dim]
        
        # Dot product
        scores = torch.mm(user_embeds, item_embeds.T)  # [batch, n_items]
        return scores
    
    def get_content_scores(
        self,
        user_histories: Dict[int, List[int]],
        users: torch.Tensor
    ) -> torch.Tensor:
        """Get content-based scores"""
        batch_size = len(users)
        scores = torch.zeros(batch_size, self.n_items, device=self.device)
        
        article_embeds = F.normalize(self.article_content_embeddings, dim=-1)
        
        for i, user_id in enumerate(users.tolist()):
            history = user_histories.get(user_id, [])
            
            if len(history) == 0:
                continue
                
            # Mean of read articles
            history_embeds = self.article_content_embeddings[history]
            user_pref = history_embeds.mean(dim=0)
            user_pref = self.user_content_projection(user_pref)
            user_pref = F.normalize(user_pref.unsqueeze(0), dim=-1)
            
            scores[i] = torch.mm(user_pref, article_embeds.T).squeeze(0)
            
        return scores
    
    def forward(
        self,
        users: torch.Tensor,
        user_histories: Dict[int, List[int]]
    ) -> torch.Tensor:
        """
        Compute hybrid scores
        
        Returns:
            scores: [batch_size, n_items]
        """
        cf_scores = self.get_cf_scores(users)
        content_scores = self.get_content_scores(user_histories, users)
        
        # Simple weighted average
        scores = self.alpha * cf_scores + (1 - self.alpha) * content_scores
        
        return scores
    
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_histories: Dict[int, List[int]]
    ) -> torch.Tensor:
        """BPR loss for training"""
        # CF scores
        user_embeds = self.user_cf_embedding(users)
        pos_embeds = self.item_cf_embedding(pos_items)
        neg_embeds = self.item_cf_embedding(neg_items)
        
        pos_cf = (user_embeds * pos_embeds).sum(dim=1)
        neg_cf = (user_embeds * neg_embeds).sum(dim=1)
        
        # Content scores (optional, can be slow)
        # For now, just use CF loss
        
        cf_loss = -F.logsigmoid(pos_cf - neg_cf).mean()
        
        # Regularization
        reg_loss = 0.01 * (
            user_embeds.norm(2).pow(2) +
            pos_embeds.norm(2).pow(2) +
            neg_embeds.norm(2).pow(2)
        ) / len(users)
        
        return cf_loss + reg_loss


class SimCSEVietnameseEncoder(nn.Module):
    """
    Vietnamese Sentence Encoder using SimCSE-trained PhoBERT
    
    Better for sentence similarity than vanilla PhoBERT
    Model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
    """
    
    def __init__(
        self,
        model_name: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        embedding_dim: int = 256,
        max_length: int = 256,
        device: str = "cuda"
    ):
        super(SimCSEVietnameseEncoder, self).__init__()
        
        from transformers import AutoModel, AutoTokenizer
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = device
        
        print(f"\n[SimCSEVietnameseEncoder] Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.hidden_size = self.model.config.hidden_size
        
        # Projection
        self.projection = nn.Linear(self.hidden_size, embedding_dim)
        
        print(f"  Hidden size: {self.hidden_size} -> {embedding_dim}")
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS pooling for SimCSE
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(embeddings)
        
        return embeddings


# ============= Utility Functions =============

def prepare_article_texts(articles_df, text_columns=['title', 'short_description']):
    """
    Prepare article texts for encoding
    
    Args:
        articles_df: DataFrame with article data
        text_columns: Which columns to use for text
        
    Returns:
        List of text strings
    """
    texts = []
    for _, row in articles_df.iterrows():
        parts = []
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        texts.append(" ".join(parts))
    return texts


def compute_article_similarity(
    encoder: PhoBERTEncoder,
    article_texts: List[str],
    query_text: str,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find most similar articles to a query
    """
    # Encode all articles
    article_embeds = encoder.encode_batch(article_texts, batch_size=32)
    article_embeds = torch.tensor(article_embeds)
    article_embeds = F.normalize(article_embeds, dim=-1)
    
    # Encode query
    with torch.no_grad():
        query_embed = encoder([query_text])
        query_embed = F.normalize(query_embed, dim=-1).cpu()
    
    # Compute similarity
    similarities = torch.mm(query_embed, article_embeds.T).squeeze(0)
    
    # Get top-k
    top_k_scores, top_k_indices = torch.topk(similarities, top_k)
    
    return list(zip(top_k_indices.tolist(), top_k_scores.tolist()))


# Import pandas for prepare_article_texts
try:
    import pandas as pd
except ImportError:
    pd = None
