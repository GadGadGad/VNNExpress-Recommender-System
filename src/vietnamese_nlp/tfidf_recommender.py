"""
TF-IDF Based Recommender for Vietnamese News
=============================================

Content-based recommendation using TF-IDF (Term Frequency - Inverse Document Frequency).

This is a classic and effective method for text-based recommendation:
1. Convert articles to TF-IDF vectors
2. Compute user preference from reading history
3. Find similar articles using cosine similarity

Advantages:
    - Fast and scalable
    - No training required
    - Interpretable results
    - Works well for sparse data
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import pickle
import os

from .preprocessing import VietnameseTextPreprocessor


class TFIDFRecommender:
    """
    TF-IDF based content recommender for Vietnamese news.
    
    Example:
        recommender = TFIDFRecommender()
        recommender.fit(article_texts)
        
        # Get recommendations
        user_history = [0, 5, 10]  # article indices user read
        recommendations = recommender.recommend(user_history, k=10)
    """
    
    def __init__(
        self,
        # Preprocessing options
        use_preprocessing: bool = True,
        use_word_segmentation: bool = True,
        remove_stopwords: bool = True,
        
        # TF-IDF options
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        use_idf: bool = True,
        
        # Custom tokenizer
        custom_tokenizer: Optional[callable] = None,
    ):
        """
        Initialize TF-IDF recommender.
        
        Args:
            use_preprocessing: Use Vietnamese preprocessing
            use_word_segmentation: Use word segmentation
            remove_stopwords: Remove Vietnamese stopwords
            max_features: Maximum number of features (vocabulary size)
            min_df: Minimum document frequency
            max_df: Maximum document frequency ratio
            ngram_range: N-gram range (e.g., (1,2) for unigrams and bigrams)
            sublinear_tf: Use sublinear TF scaling (1 + log(tf))
            use_idf: Use IDF weighting
            custom_tokenizer: Custom tokenizer function
        """
        self.use_preprocessing = use_preprocessing
        self.use_word_segmentation = use_word_segmentation
        self.remove_stopwords = remove_stopwords
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.use_idf = use_idf
        self.custom_tokenizer = custom_tokenizer
        
        # Preprocessor
        self.preprocessor = None
        if use_preprocessing:
            self.preprocessor = VietnameseTextPreprocessor(
                use_word_segmentation=use_word_segmentation,
                remove_stopwords=remove_stopwords
            )
            
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            use_idf=use_idf,
            tokenizer=custom_tokenizer if custom_tokenizer else self._tokenize,
            preprocessor=lambda x: x,  # We do our own preprocessing
            token_pattern=None  # Required when using tokenizer
        )
        
        # Article vectors
        self.article_vectors: Optional[sp.csr_matrix] = None
        self.article_texts: Optional[List[str]] = None
        self.n_articles: int = 0
        
        # Stats
        self.is_fitted = False
        self.vocabulary_size = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using preprocessor"""
        if self.preprocessor:
            return self.preprocessor.tokenize(text)
        return text.lower().split()
        
    def fit(
        self,
        article_texts: List[str],
        show_progress: bool = True
    ):
        """
        Fit TF-IDF on article texts.
        
        Args:
            article_texts: List of article texts
            show_progress: Show progress
        """
        self.article_texts = article_texts
        self.n_articles = len(article_texts)
        
        print(f"\n[TFIDFRecommender] Fitting on {self.n_articles} articles...")
        
        # Preprocess texts
        if self.use_preprocessing and self.preprocessor:
            print("  Preprocessing texts...")
            processed_texts = self.preprocessor.preprocess_batch(
                article_texts, 
                show_progress=show_progress
            )
        else:
            processed_texts = article_texts
            
        # Fit TF-IDF
        print("  Building TF-IDF matrix...")
        self.article_vectors = self.vectorizer.fit_transform(processed_texts)
        
        self.vocabulary_size = len(self.vectorizer.vocabulary_)
        self.is_fitted = True
        
        print(f"  Matrix shape: {self.article_vectors.shape}")
        print(f"  Vocabulary size: {self.vocabulary_size}")
        print(f"  Sparsity: {1 - self.article_vectors.nnz / (self.n_articles * self.vocabulary_size):.4f}")
        
    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Transform new texts to TF-IDF vectors.
        
        Args:
            texts: List of texts
            
        Returns:
            TF-IDF sparse matrix
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        # Preprocess
        if self.use_preprocessing and self.preprocessor:
            processed_texts = self.preprocessor.preprocess_batch(texts, show_progress=False)
        else:
            processed_texts = texts
            
        return self.vectorizer.transform(processed_texts)
        
    def get_user_profile(
        self,
        user_history: List[int],
        aggregation: str = 'mean',
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute user profile vector from reading history.
        
        Args:
            user_history: List of article indices user has read
            aggregation: How to aggregate ('mean', 'sum', 'max', 'weighted')
            weights: Weights for each article (for 'weighted' aggregation)
            
        Returns:
            User profile vector
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        if len(user_history) == 0:
            return np.zeros(self.vocabulary_size)
            
        # Get vectors for read articles
        history_vectors = self.article_vectors[user_history].toarray()
        
        if aggregation == 'mean':
            user_profile = np.mean(history_vectors, axis=0)
        elif aggregation == 'sum':
            user_profile = np.sum(history_vectors, axis=0)
        elif aggregation == 'max':
            user_profile = np.max(history_vectors, axis=0)
        elif aggregation == 'weighted':
            if weights is None:
                # Recency weighting: more recent = higher weight
                weights = np.linspace(0.5, 1.0, len(user_history))
            weights = np.array(weights).reshape(-1, 1)
            user_profile = np.sum(history_vectors * weights, axis=0) / weights.sum()
        else:
            user_profile = np.mean(history_vectors, axis=0)
            
        return user_profile.flatten()
        
    def compute_scores(
        self,
        user_history: List[int],
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Compute similarity scores for all articles.
        
        Args:
            user_history: User's reading history
            aggregation: How to aggregate user profile
            
        Returns:
            Scores for all articles
        """
        user_profile = self.get_user_profile(user_history, aggregation)
        user_profile = user_profile.reshape(1, -1)
        
        # Compute cosine similarity with all articles
        scores = cosine_similarity(user_profile, self.article_vectors).flatten()
        
        return scores
        
    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_read: bool = True,
        aggregation: str = 'mean'
    ) -> Tuple[List[int], List[float]]:
        """
        Recommend top-k articles for a user.
        
        Args:
            user_history: List of article indices user has read
            k: Number of recommendations
            exclude_read: Exclude already read articles
            aggregation: How to aggregate user profile
            
        Returns:
            (article_indices, scores)
        """
        scores = self.compute_scores(user_history, aggregation)
        
        # Exclude read articles
        if exclude_read:
            scores[user_history] = -np.inf
            
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
        
    def find_similar_articles(
        self,
        article_idx: int,
        k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Find similar articles to a given article.
        
        Args:
            article_idx: Index of the target article
            k: Number of similar articles
            
        Returns:
            (article_indices, similarity_scores)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        article_vector = self.article_vectors[article_idx]
        similarities = cosine_similarity(article_vector, self.article_vectors).flatten()
        
        # Exclude self
        similarities[article_idx] = -np.inf
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
        
    def get_top_terms(self, article_idx: int, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top terms for an article.
        
        Args:
            article_idx: Article index
            k: Number of top terms
            
        Returns:
            List of (term, tfidf_score)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        feature_names = self.vectorizer.get_feature_names_out()
        article_vector = self.article_vectors[article_idx].toarray().flatten()
        
        top_indices = np.argsort(article_vector)[::-1][:k]
        
        return [(feature_names[i], article_vector[i]) for i in top_indices if article_vector[i] > 0]
        
    def get_vocabulary(self) -> Dict[str, int]:
        """Get the vocabulary mapping"""
        return self.vectorizer.vocabulary_
        
    def save(self, path: str):
        """
        Save the recommender to disk.
        
        Args:
            path: Save path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'vectorizer': self.vectorizer,
            'article_vectors': self.article_vectors,
            'article_texts': self.article_texts,
            'n_articles': self.n_articles,
            'vocabulary_size': self.vocabulary_size,
            'is_fitted': self.is_fitted,
            'config': {
                'use_preprocessing': self.use_preprocessing,
                'use_word_segmentation': self.use_word_segmentation,
                'remove_stopwords': self.remove_stopwords,
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"[TFIDFRecommender] Saved to {path}")
        
    def load(self, path: str):
        """
        Load the recommender from disk.
        
        Args:
            path: Load path
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.vectorizer = data['vectorizer']
        self.article_vectors = data['article_vectors']
        self.article_texts = data['article_texts']
        self.n_articles = data['n_articles']
        self.vocabulary_size = data['vocabulary_size']
        self.is_fitted = data['is_fitted']
        
        print(f"[TFIDFRecommender] Loaded from {path}")
        print(f"  Articles: {self.n_articles}, Vocabulary: {self.vocabulary_size}")
        
    def evaluate(
        self,
        train_dict: Dict[int, Set[int]],
        test_dict: Dict[int, List[int]],
        k_list: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """
        Evaluate recommendation performance.
        
        Args:
            train_dict: {user_id: set of train item indices}
            test_dict: {user_id: list of test item indices}
            k_list: List of K values for metrics
            
        Returns:
            Dictionary of metrics
        """
        from tqdm import tqdm
        
        results = defaultdict(list)
        max_k = max(k_list)
        
        for user_id in tqdm(test_dict.keys(), desc="Evaluating"):
            train_items = list(train_dict.get(user_id, set()))
            test_items = set(test_dict[user_id])
            
            if len(train_items) == 0 or len(test_items) == 0:
                continue
                
            # Get recommendations
            scores = self.compute_scores(train_items)
            
            # Exclude train items
            for item in train_items:
                if item < len(scores):
                    scores[item] = -np.inf
                    
            top_items = np.argsort(scores)[::-1][:max_k]
            
            for k in k_list:
                top_k = top_items[:k]
                hits = len(set(top_k) & test_items)
                
                # Recall@K
                recall = hits / len(test_items)
                results[f'Recall@{k}'].append(recall)
                
                # NDCG@K
                dcg = sum([1.0 / np.log2(i + 2) 
                          for i, item in enumerate(top_k) if item in test_items])
                idcg = sum([1.0 / np.log2(i + 2) 
                           for i in range(min(len(test_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                results[f'NDCG@{k}'].append(ndcg)
                
                # HR@K
                hr = 1.0 if hits > 0 else 0.0
                results[f'HR@{k}'].append(hr)
                
        # Average
        avg_results = {}
        for key, values in results.items():
            avg_results[key] = np.mean(values) if len(values) > 0 else 0.0
            
        return avg_results


class TFIDFWithCategoryBoost(TFIDFRecommender):
    """
    TF-IDF recommender with category boosting.
    
    Boosts recommendations from categories the user has shown interest in.
    """
    
    def __init__(
        self,
        category_boost: float = 0.2,
        **kwargs
    ):
        """
        Args:
            category_boost: Extra weight for matching categories
            **kwargs: Arguments for TFIDFRecommender
        """
        super().__init__(**kwargs)
        self.category_boost = category_boost
        self.article_categories: Optional[List[str]] = None
        
    def fit_with_categories(
        self,
        article_texts: List[str],
        article_categories: List[str],
        show_progress: bool = True
    ):
        """
        Fit with category information.
        
        Args:
            article_texts: List of article texts
            article_categories: Category for each article
        """
        self.article_categories = article_categories
        self.fit(article_texts, show_progress)
        
    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_read: bool = True,
        aggregation: str = 'mean'
    ) -> Tuple[List[int], List[float]]:
        """
        Recommend with category boosting.
        """
        scores = self.compute_scores(user_history, aggregation)
        
        # Category boost
        if self.article_categories is not None:
            user_categories = set(
                self.article_categories[i] for i in user_history
                if i < len(self.article_categories)
            )
            
            for i, cat in enumerate(self.article_categories):
                if cat in user_categories:
                    scores[i] += self.category_boost
                    
        # Exclude read articles
        if exclude_read:
            scores[user_history] = -np.inf
            
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()


if __name__ == '__main__':
    # Demo
    print("=" * 60)
    print("TF-IDF Recommender Demo")
    print("=" * 60)
    
    # Sample Vietnamese news articles
    articles = [
        "Đội tuyển bóng đá Việt Nam giành chiến thắng trước Thái Lan trong trận chung kết AFF Cup",
        "Cầu thủ Nguyễn Quang Hải ghi bàn thắng quyết định trong hiệp 2",
        "Kinh tế Việt Nam tăng trưởng 6.5% trong quý 3 năm 2024",
        "Thị trường chứng khoán Việt Nam có dấu hiệu phục hồi",
        "Apple ra mắt iPhone 16 với nhiều tính năng AI mới",
        "Samsung cạnh tranh với Apple trong phân khúc điện thoại cao cấp",
        "Thời tiết Hà Nội ngày mai có mưa nhỏ, nhiệt độ 20-25 độ C",
        "Dự báo thời tiết miền Bắc có không khí lạnh tràn về",
        "Chính phủ ban hành chính sách mới hỗ trợ doanh nghiệp",
        "Quốc hội thông qua luật đầu tư mới thu hút vốn nước ngoài",
    ]
    
    # Initialize and fit
    recommender = TFIDFRecommender(
        use_preprocessing=True,
        use_word_segmentation=True,
        remove_stopwords=True,
        ngram_range=(1, 2),
        max_features=5000
    )
    
    recommender.fit(articles)
    
    # Test recommendations
    print("\n" + "-" * 40)
    print("User read articles about sports (indices 0, 1):")
    print(f"  [0] {articles[0][:60]}...")
    print(f"  [1] {articles[1][:60]}...")
    
    top_items, top_scores = recommender.recommend([0, 1], k=3)
    print("\nRecommendations:")
    for idx, score in zip(top_items, top_scores):
        print(f"  [{idx}] (score={score:.4f}) {articles[idx][:60]}...")
        
    # Similar articles
    print("\n" + "-" * 40)
    print("Similar articles to article 2 (economy):")
    print(f"  [2] {articles[2]}")
    
    similar_items, similar_scores = recommender.find_similar_articles(2, k=3)
    for idx, score in zip(similar_items, similar_scores):
        print(f"  [{idx}] (sim={score:.4f}) {articles[idx][:60]}...")
        
    # Top terms
    print("\n" + "-" * 40)
    print("Top terms for article 0:")
    top_terms = recommender.get_top_terms(0, k=5)
    for term, score in top_terms:
        print(f"  {term}: {score:.4f}")
