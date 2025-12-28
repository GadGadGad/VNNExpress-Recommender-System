"""
Content-Based Models for Vietnamese News Recommendation
========================================================

Models:
1. UniversalEncoder - Flexible encoder supporting multiple backends
2. ContentBasedRecommender - Pure content-based recommendation  
3. HybridRecommender - Combine collaborative filtering + content-based
4. TFIDFRecommender - Lightweight TF-IDF baseline

Supported embeddings:
- phobert: vinai/phobert-base
- bge-m3, vndoc, e5-large, e5-base, vn-sbert, gte: SentenceTransformers
- tfidf: TF-IDF + SVD
- precomputed: Load from .pt file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class UniversalEncoder:
    """
    Universal encoder supporting multiple backends:
    - phobert: PhoBERT (transformers)
    - sentence-transformers: bge-m3, vndoc, e5, vn-sbert, etc.
    - tfidf: TF-IDF + SVD
    - precomputed: Load from .pt file
    """
    
    SENTENCE_TRANSFORMER_MODELS = {
        'bge-m3': 'BAAI/bge-m3',
        'vndoc': 'bkai-foundation-models/vietnamese-bi-encoder',
        'e5-large': 'intfloat/multilingual-e5-large',
        'e5-base': 'intfloat/multilingual-e5-base',
        'vn-sbert': 'keepitreal/vietnamese-sbert',
        'gte': 'Alibaba-NLP/gte-multilingual-base',
    }
    
    def __init__(
        self,
        encoder_type: str = 'phobert',
        model_name: str = None,
        device: str = 'cuda',
        precomputed_path: str = None
    ):
        self.encoder_type = encoder_type
        self.device = device
        self.model = None
        self.embedding_dim = None
        self.precomputed_embeddings = None
        
        if encoder_type == 'precomputed':
            if precomputed_path and os.path.exists(precomputed_path):
                self.precomputed_embeddings = torch.load(precomputed_path, map_location='cpu')
                self.embedding_dim = self.precomputed_embeddings.shape[1]
                print(f"[UniversalEncoder] Loaded precomputed embeddings: {self.precomputed_embeddings.shape}")
            else:
                raise ValueError(f"Precomputed path not found: {precomputed_path}")
                
        elif encoder_type == 'phobert':
            model_name = model_name or 'vinai/phobert-base'
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.embedding_dim = self.model.config.hidden_size
            print(f"[UniversalEncoder] Loaded PhoBERT: {model_name} (dim={self.embedding_dim})")
            
        elif encoder_type in self.SENTENCE_TRANSFORMER_MODELS or encoder_type == 'sentence-transformer':
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Please install: pip install sentence-transformers")
            
            if encoder_type in self.SENTENCE_TRANSFORMER_MODELS:
                model_name = self.SENTENCE_TRANSFORMER_MODELS[encoder_type]
            elif model_name is None:
                raise ValueError("Must provide model_name for sentence-transformer")
                
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"[UniversalEncoder] Loaded SentenceTransformer: {model_name} (dim={self.embedding_dim})")
            
        elif encoder_type == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self.vectorizer = TfidfVectorizer(max_features=10000)
            self.svd = TruncatedSVD(n_components=256)
            self.embedding_dim = 256
            self._fitted = False
            print(f"[UniversalEncoder] TF-IDF + SVD (dim={self.embedding_dim})")
            
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        
        if self.encoder_type == 'precomputed':
            raise ValueError("Cannot encode new texts with precomputed embeddings. Use get_embeddings() instead.")
        
        if self.encoder_type == 'phobert':
            return self._encode_phobert(texts, batch_size)
        
        if self.encoder_type in self.SENTENCE_TRANSFORMER_MODELS or self.encoder_type == 'sentence-transformer':
            return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        
        if self.encoder_type == 'tfidf':
            return self._encode_tfidf(texts)
        
    def _encode_phobert(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode using PhoBERT with mean pooling"""
        self.model.eval()
        all_embeddings = []
        
        from tqdm import tqdm
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding (PhoBERT)"):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded['input_ids'].to(self.device),
                    attention_mask=encoded['attention_mask'].to(self.device)
                )
                # Mean pooling
                mask = encoded['attention_mask'].unsqueeze(-1).float().to(self.device)
                embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_embeddings.append(embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)
    
    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """Encode using TF-IDF + SVD"""
        if not self._fitted:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            embeddings = self.svd.fit_transform(tfidf_matrix)
            self._fitted = True
        else:
            tfidf_matrix = self.vectorizer.transform(texts)
            embeddings = self.svd.transform(tfidf_matrix)
        return embeddings
    
    def get_embeddings(self, indices: List[int] = None) -> torch.Tensor:
        """Get precomputed embeddings by indices"""
        if self.precomputed_embeddings is None:
            raise ValueError("No precomputed embeddings loaded")
        if indices is None:
            return self.precomputed_embeddings
        return self.precomputed_embeddings[indices]


class ContentBasedRecommender(nn.Module):
    """
    Pure Content-Based Recommendation with flexible encoder support.
    
    Supported encoder_types:
    - 'phobert': PhoBERT (vinai/phobert-base)
    - 'bge-m3', 'vndoc', 'e5-large', 'e5-base', 'vn-sbert', 'gte': SentenceTransformers
    - 'tfidf': TF-IDF + SVD
    - 'precomputed': Load from .pt file
    
    Example:
        recommender = ContentBasedRecommender(n_users, n_items, encoder_type='bge-m3')
        recommender.encode_articles(article_texts)
        recs = recommender.recommend(user_history, k=10)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        encoder_type: str = 'phobert',
        model_name: str = None,
        precomputed_path: str = None,
        device: str = "cuda",
        **kwargs
    ):
        super(ContentBasedRecommender, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.encoder_type = encoder_type
        
        # Initialize universal encoder
        self.encoder = UniversalEncoder(
            encoder_type=encoder_type,
            model_name=model_name,
            device=device,
            precomputed_path=precomputed_path
        )
        
        self.embedding_dim = self.encoder.embedding_dim
        
        # User preference encoder (trainable)
        self.user_preference_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
            
        # Cached article embeddings
        self.article_embeddings: Optional[torch.Tensor] = None
        
        print(f"\n[ContentBasedRecommender] Initialized")
        print(f"  Encoder: {encoder_type}")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  Embedding dim: {self.embedding_dim}")
        
    def encode_articles(
        self,
        article_texts: List[str] = None,
        batch_size: int = 32
    ):
        """
        Pre-encode all articles and cache embeddings.
        If encoder_type='precomputed', article_texts is ignored.
        """
        if self.encoder_type == 'precomputed':
            self.article_embeddings = self.encoder.get_embeddings().to(self.device)
            print(f"[ContentBasedRecommender] Using precomputed embeddings: {self.article_embeddings.shape}")
        else:
            if article_texts is None:
                raise ValueError("article_texts required for non-precomputed encoders")
            print(f"\n[ContentBasedRecommender] Encoding {len(article_texts)} articles...")
            embeddings = self.encoder.encode(article_texts, batch_size=batch_size)
            self.article_embeddings = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
            
        print(f"  Cached embeddings shape: {self.article_embeddings.shape}")
        
    def load_precomputed_embeddings(self, path: str):
        """Load pre-computed article embeddings from file"""
        if os.path.exists(path):
            self.article_embeddings = torch.load(path, map_location=self.device)
            self.embedding_dim = self.article_embeddings.shape[1]
            print(f"[ContentBasedRecommender] Loaded embeddings from {path}: {self.article_embeddings.shape}")
        else:
            raise FileNotFoundError(f"Embedding file not found: {path}")

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
        encoder_type: str = 'phobert',
        model_name: str = None,
        precomputed_path: str = None,
        alpha: float = 0.5,  # Weight for CF vs Content
        device: str = "cuda"
    ):
        super(HybridRecommender, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.cf_embedding_dim = cf_embedding_dim
        self.alpha = alpha
        self.device = device
        
        # ========== Collaborative Filtering Component ==========
        self.user_cf_embedding = nn.Embedding(n_users, cf_embedding_dim)
        self.item_cf_embedding = nn.Embedding(n_items, cf_embedding_dim)
        
        nn.init.xavier_uniform_(self.user_cf_embedding.weight)
        nn.init.xavier_uniform_(self.item_cf_embedding.weight)
        
        # ========== Content-Based Component (Universal Encoder) ==========
        self.encoder = UniversalEncoder(
            encoder_type=encoder_type,
            model_name=model_name,
            device=device,
            precomputed_path=precomputed_path
        )
        self.content_embedding_dim = self.encoder.embedding_dim
        
        # User content preference projection
        self.user_content_projection = nn.Sequential(
            nn.Linear(self.content_embedding_dim, self.content_embedding_dim),
            nn.LayerNorm(self.content_embedding_dim),
            nn.GELU()
        )
        
        # Cached embeddings
        self.article_content_embeddings: Optional[torch.Tensor] = None
        self.raw_user_profiles: Optional[torch.Tensor] = None
        
        print(f"\n[HybridRecommender] Initialized")
        print(f"  Encoder: {encoder_type}")
        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  CF dim: {cf_embedding_dim}, Content dim: {self.content_embedding_dim}")
        print(f"  Alpha (CF weight): {alpha}")
        
    def encode_articles(self, article_texts: List[str] = None, batch_size: int = 32):
        """Pre-encode all articles using UniversalEncoder"""
        if self.encoder.encoder_type == 'precomputed':
            self.article_content_embeddings = self.encoder.get_embeddings().to(self.device)
            print(f"[HybridRecommender] Using precomputed embeddings: {self.article_content_embeddings.shape}")
        else:
            if article_texts is None:
                raise ValueError("article_texts required for non-precomputed encoders")
            print(f"\n[HybridRecommender] Encoding {len(article_texts)} articles...")
            embeddings = self.encoder.encode(article_texts, batch_size=batch_size)
            self.article_content_embeddings = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
        
    def precompute_user_profiles(self, user_histories: Dict[int, List[int]]):
        """
        Precompute raw user profiles (mean of history embeddings)
        This speeds up training significantly by avoiding loop aggregation in the forward pass.
        """
        if self.article_content_embeddings is None:
            raise ValueError("Must encode articles before computing user profiles")
            
        print("[HybridRecommender] Precomputing user profiles...")
        self.raw_user_profiles = torch.zeros(
            self.n_users, self.content_embedding_dim, device=self.device
        )
        
        hit_count = 0
        for user_id, history in user_histories.items():
            if user_id < self.n_users and len(history) > 0:
                # Filter valid items
                valid_items = [i for i in history if i < self.article_content_embeddings.size(0)]
                if valid_items:
                    embeds = self.article_content_embeddings[valid_items]
                    self.raw_user_profiles[user_id] = embeds.mean(dim=0)
                    hit_count += 1
                    
        print(f"  Computed profiles for {hit_count}/{self.n_users} users")

    def get_cf_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Get collaborative filtering scores"""
        user_embeds = self.user_cf_embedding(users)  # [batch, cf_dim]
        item_embeds = self.item_cf_embedding.weight  # [n_items, cf_dim]
        scores = torch.mm(user_embeds, item_embeds.T)
        return scores
    
    def get_content_scores(self, users: torch.Tensor) -> torch.Tensor:
        """
        Get content-based scores using precomputed raw profiles
        """
        if self.raw_user_profiles is None:
            raise ValueError("Run precompute_user_profiles first!")
            
        # 1. Get raw mean embeddings [batch, dim]
        raw_profiles = self.raw_user_profiles[users]
        
        # 2. Project to preference space (TRAINABLE)
        user_pref = self.user_content_projection(raw_profiles)
        
        # 3. Normalize
        user_pref = F.normalize(user_pref, dim=-1)
        article_embeds = F.normalize(self.article_content_embeddings, dim=-1)
        
        # 4. Dot product with all articles
        scores = torch.mm(user_pref, article_embeds.T)
        
        return scores

    def forward(
        self,
        users: torch.Tensor,
        user_histories: Dict[int, List[int]] = None # Kept for API compatibility, unused if precomputed
    ) -> torch.Tensor:
        """
        Compute hybrid scores for all items
        """
        # CF Component
        cf_scores = self.get_cf_scores(users)
        
        # Content Component
        if self.raw_user_profiles is not None:
            content_scores = self.get_content_scores(users)
        else:
            # Fallback (slow, mostly for inference on new users if needed, but we rely on precompute)
            content_scores = torch.zeros_like(cf_scores)
            
        # Hybrid
        return self.alpha * cf_scores + (1 - self.alpha) * content_scores
    
    def get_hybrid_score_pointwise(self, users, items):
        """
        Compute scores for specific (user, item) pairs.
        Used for BPR loss.
        """
        # --- CF ---
        u_cf = self.user_cf_embedding(users)
        i_cf = self.item_cf_embedding(items)
        cf_score = (u_cf * i_cf).sum(dim=1)
        
        # --- Content ---
        if self.raw_user_profiles is not None:
            # Get PROJECTED user pref
            raw_u = self.raw_user_profiles[users]
            u_pref = self.user_content_projection(raw_u)
            u_pref = F.normalize(u_pref, dim=-1)
            
            # Get item content
            i_content = self.article_content_embeddings[items]
            i_content = F.normalize(i_content, dim=-1)
            
            content_score = (u_pref * i_content).sum(dim=1)
        else:
            content_score = torch.zeros_like(cf_score)
            
        # --- Hybrid ---
        return self.alpha * cf_score + (1 - self.alpha) * content_score

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_histories: Dict[int, List[int]] = None
    ) -> torch.Tensor:
        """BPR loss optimizing the Hybrid score"""
        
        # If raw profiles not computed, try to compute (if histories passed)
        if self.raw_user_profiles is None and user_histories is not None:
             self.precompute_user_profiles(user_histories)
        
        # Compute Hybrid Scores
        pos_scores = self.get_hybrid_score_pointwise(users, pos_items)
        neg_scores = self.get_hybrid_score_pointwise(users, neg_items)
        
        # Loss
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Regularization (L2 on CF embeddings)
        # Optional: Add reg on content projection? Default L2 usually covers weights.
        reg_loss = 1e-4 * (
            self.user_cf_embedding(users).norm(2).pow(2) +
            self.item_cf_embedding(pos_items).norm(2).pow(2) +
            self.item_cf_embedding(neg_items).norm(2).pow(2)
        ) / len(users)
        
        return loss + reg_loss


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
    encoder: UniversalEncoder,
    article_texts: List[str],
    query_text: str,
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find most similar articles to a query
    """
    # Encode all articles
    article_embeds = encoder.encode(article_texts, batch_size=32)
    article_embeds = torch.tensor(article_embeds)
    article_embeds = F.normalize(article_embeds, dim=-1)
    
    # Encode query
    query_embed = encoder.encode([query_text])
    query_embed = torch.tensor(query_embed)
    query_embed = F.normalize(query_embed, dim=-1)
    
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

# ===============================================
# Standard TF-IDF Recommender (for Demo Fallback)
# ===============================================
class TFIDFRecommender:
    """
    Standard TF-IDF + Cosine Similarity Recommender
    """
    def __init__(self, n_users, n_items, max_features=5000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.n_users = n_users
        self.n_items = n_items
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.article_vectors = None
        # Not inheriting from nn.Module to keep it lightweight for CPU
        
    def encode_articles(self, texts: List[str]):
        """Fit TF-IDF on texts"""
        # Ensure imports
        import operator
            
        print(f"[TF-IDF] Fitting on {len(texts)} articles...")
        self.article_vectors = self.vectorizer.fit_transform(texts)
        print(f"  Shape: {self.article_vectors.shape}")
        
    def recommend(self, history_indices: List[int], k: int = 10):
        """Recommend items similar to user history (mean profile)"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if self.article_vectors is None:
            raise ValueError("Fit model first!")
            
        # Filter valid indices
        # Ensure we don't crash if index out of bounds
        max_idx = self.article_vectors.shape[0]
        valid_indices = [i for i in history_indices if i < max_idx]
        
        if not valid_indices:
             return [], []
             
        # User profile = Mean of history vectors
        user_profile = np.asarray(self.article_vectors[valid_indices].mean(axis=0))
        
        # Cosine sim (sparse matrix friendly)
        scores = cosine_similarity(user_profile, self.article_vectors).flatten()
        
        # Mask history (set seen scores to -1 so they are not recommended)
        scores[valid_indices] = -1.0
        
        # Top K
        top_k_indices = scores.argsort()[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        
        # Note: scores are numpy arrays
        return top_k_indices.tolist(), top_k_scores.tolist()
