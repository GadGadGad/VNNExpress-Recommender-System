import torch
import torch.nn.functional as F
import pandas as pd


class PhoBERTWrapper:
    """Wrapper for pre-computed PhoBERT/SimCSE embeddings for recommendation."""
    
    def __init__(self, embeddings, articles_df):
        self.embeddings = embeddings  # Shape: (n_articles, embed_dim)
        self.articles_df = articles_df
        # Normalize embeddings for cosine similarity
        self.embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend based on cosine similarity to history."""
        if not history_indices:
            return [], []
        
        # Average embedding of history articles
        history_embs = self.embeddings_norm[history_indices]
        query = history_embs.mean(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        # Compute similarity to all articles
        similarities = torch.mm(query, self.embeddings_norm.t()).squeeze()
        
        # Exclude history from recommendations
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        # Top-k
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()


class LSAWrapper:
    """Wrapper for LSA (Latent Semantic Analysis) based recommendation."""
    
    def __init__(self, lsa_matrix, articles_df):
        self.lsa_matrix = torch.tensor(lsa_matrix, dtype=torch.float32)
        self.articles_df = articles_df
        self.lsa_norm = F.normalize(self.lsa_matrix, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend based on cosine similarity in latent topic space."""
        if not history_indices:
            return [], []
        
        # Average LSA vector of history
        history_vecs = self.lsa_norm[history_indices]
        query = history_vecs.mean(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        similarities = torch.mm(query, self.lsa_norm.t()).squeeze()
        
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()


class NaiveBayesWrapper:
    """Wrapper for Naive Bayes probabilistic recommendation."""
    
    def __init__(self, vectorizer, nb_model, X_matrix, articles_df):
        self.vectorizer = vectorizer
        self.nb = nb_model
        self.X = X_matrix  # Sparse matrix
        self.articles_df = articles_df
    
    def recommend(self, history_indices, k=10):
        """Recommend articles with similar category probability distribution."""
        if not history_indices:
            return [], []
        
        # Get category distribution from history
        history_cats = self.articles_df.iloc[history_indices]['source_category'].value_counts()
        target_cat = history_cats.idxmax()  # Most frequent category
        
        # Get probability of each article belonging to target category
        proba = self.nb.predict_proba(self.X)
        cat_idx = list(self.nb.classes_).index(target_cat) if target_cat in self.nb.classes_ else 0
        scores = proba[:, cat_idx]
        
        # Exclude history
        scores = list(scores)
        for idx in history_indices:
            scores[idx] = -1
        
        # Top-k
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:k]
        top_scores = [scores[i] for i in top_indices]
        
        return top_indices, top_scores


class SessionWrapper:
    """Wrapper for Session-based recommendation with attention over recent history."""
    
    def __init__(self, embeddings, articles_df):
        self.embeddings = embeddings
        self.articles_df = articles_df
        self.embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend using exponential decay attention over recent history."""
        if not history_indices:
            return [], []
        
        # Session-based: use only last N items with decay weights
        max_session = min(10, len(history_indices))
        recent_indices = history_indices[-max_session:]
        
        # Exponential decay: more recent = higher weight
        weights = torch.tensor([0.9 ** (max_session - i - 1) for i in range(len(recent_indices))])
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of recent item embeddings
        recent_embs = self.embeddings_norm[recent_indices]
        query = (recent_embs * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        # Similarity
        similarities = torch.mm(query, self.embeddings_norm.t()).squeeze()
        
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()
