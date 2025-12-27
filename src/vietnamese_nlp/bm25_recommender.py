"""
BM25 Based Recommender for Vietnamese News
==========================================

BM25 (Best Matching 25) is a ranking function widely used in 
information retrieval and is particularly effective for news search.

BM25 improves on TF-IDF by:
1. Saturating term frequency (diminishing returns for repeated terms)
2. Normalizing for document length
3. Tunable parameters k1, b, and delta

Use cases:
    - News article search and ranking
    - Content-based recommendation
    - Query-based retrieval
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import math
import pickle
import os

from .preprocessing import VietnameseTextPreprocessor


class BM25:
    """
    BM25 ranking algorithm implementation.
    
    BM25 formula:
        score(D, Q) = sum over terms t in Q of:
            IDF(t) * (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * |D|/avgdl))
    
    Where:
        - IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        - N = total documents
        - df(t) = document frequency of term t
        - tf(t, D) = term frequency in document D
        - |D| = document length
        - avgdl = average document length
        - k1, b = tunable parameters
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5  # BM25+ extension
    ):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Document length normalization (0.75 typical, 0=no normalization)
            delta: BM25+ parameter for lower-bounding (default 0.5)
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        # Corpus statistics
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_lengths = []
        self.doc_freqs = {}  # term -> document frequency
        self.idf = {}  # term -> IDF score
        self.doc_term_freqs = []  # list of Counter for each doc
        
        self.is_fitted = False
        
    def fit(self, tokenized_corpus: List[List[str]]):
        """
        Fit BM25 on tokenized corpus.
        
        Args:
            tokenized_corpus: List of tokenized documents
        """
        self.corpus_size = len(tokenized_corpus)
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)
        self.doc_term_freqs = []
        
        total_length = 0
        
        for doc in tokenized_corpus:
            doc_len = len(doc)
            self.doc_lengths.append(doc_len)
            total_length += doc_len
            
            # Count term frequencies
            term_freq = Counter(doc)
            self.doc_term_freqs.append(term_freq)
            
            # Count document frequencies
            for term in set(doc):
                self.doc_freqs[term] += 1
                
        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0
        
        # Compute IDF for all terms
        self._compute_idf()
        
        self.is_fitted = True
        
    def _compute_idf(self):
        """Compute IDF for all terms in vocabulary"""
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF (Robertson-Sparck Jones)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = idf
            
    def get_scores(self, query: List[str]) -> np.ndarray:
        """
        Compute BM25 scores for a query against all documents.
        
        Args:
            query: Tokenized query
            
        Returns:
            Scores for all documents
        """
        if not self.is_fitted:
            raise ValueError("BM25 not fitted! Call fit() first.")
            
        scores = np.zeros(self.corpus_size)
        
        for doc_idx in range(self.corpus_size):
            score = self._score_document(query, doc_idx)
            scores[doc_idx] = score
            
        return scores
        
    def _score_document(self, query: List[str], doc_idx: int) -> float:
        """Score a single document against query"""
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        score = 0.0
        
        for term in query:
            if term not in self.idf:
                continue
                
            tf = term_freqs.get(term, 0)
            idf = self.idf[term]
            
            # BM25 scoring
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            # BM25+ adds delta to prevent zero scores
            score += idf * (numerator / denominator + self.delta)
            
        return score
        
    def get_top_k(
        self,
        query: List[str],
        k: int = 10,
        exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-k documents for a query.
        
        Args:
            query: Tokenized query
            k: Number of results
            exclude_indices: Document indices to exclude
            
        Returns:
            (doc_indices, scores)
        """
        scores = self.get_scores(query)
        
        if exclude_indices:
            for idx in exclude_indices:
                if idx < len(scores):
                    scores[idx] = -np.inf
                    
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()


class BM25Recommender:
    """
    BM25-based content recommender for Vietnamese news.
    
    Uses BM25 for content-based recommendation by:
    1. Building BM25 index from articles
    2. Creating user profile from reading history
    3. Finding relevant articles using BM25 scoring
    
    Example:
        recommender = BM25Recommender()
        recommender.fit(article_texts)
        
        user_history = [0, 5, 10]
        recommendations = recommender.recommend(user_history, k=10)
    """
    
    def __init__(
        self,
        # Preprocessing options
        use_preprocessing: bool = True,
        use_word_segmentation: bool = True,
        remove_stopwords: bool = True,
        
        # BM25 parameters
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        
        # Recommendation options
        query_expansion: bool = True,
        max_query_terms: int = 100,
    ):
        """
        Initialize BM25 recommender.
        
        Args:
            use_preprocessing: Use Vietnamese preprocessing
            use_word_segmentation: Use word segmentation
            remove_stopwords: Remove stopwords
            k1: BM25 term saturation parameter
            b: BM25 length normalization parameter
            delta: BM25+ delta parameter
            query_expansion: Expand user profile query with important terms
            max_query_terms: Maximum terms in user profile query
        """
        self.use_preprocessing = use_preprocessing
        self.use_word_segmentation = use_word_segmentation
        self.remove_stopwords = remove_stopwords
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.query_expansion = query_expansion
        self.max_query_terms = max_query_terms
        
        # Preprocessor
        self.preprocessor = None
        if use_preprocessing:
            self.preprocessor = VietnameseTextPreprocessor(
                use_word_segmentation=use_word_segmentation,
                remove_stopwords=remove_stopwords
            )
            
        # BM25 index
        self.bm25 = BM25(k1=k1, b=b, delta=delta)
        
        # Article data
        self.article_texts: Optional[List[str]] = None
        self.tokenized_articles: Optional[List[List[str]]] = None
        self.n_articles: int = 0
        
        self.is_fitted = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        if self.preprocessor:
            return self.preprocessor.tokenize(text)
        return text.lower().split()
        
    def fit(
        self,
        article_texts: List[str],
        show_progress: bool = True
    ):
        """
        Fit BM25 on article texts.
        
        Args:
            article_texts: List of article texts
            show_progress: Show progress
        """
        self.article_texts = article_texts
        self.n_articles = len(article_texts)
        
        print(f"\n[BM25Recommender] Fitting on {self.n_articles} articles...")
        
        # Tokenize articles
        print("  Tokenizing articles...")
        if self.preprocessor:
            self.tokenized_articles = self.preprocessor.tokenize_batch(
                article_texts,
                show_progress=show_progress
            )
        else:
            self.tokenized_articles = [text.lower().split() for text in article_texts]
            
        # Fit BM25
        print("  Building BM25 index...")
        self.bm25.fit(self.tokenized_articles)
        
        self.is_fitted = True
        
        print(f"  Vocabulary size: {len(self.bm25.doc_freqs)}")
        print(f"  Average document length: {self.bm25.avgdl:.1f} tokens")
        
    def get_user_profile_query(
        self,
        user_history: List[int],
        method: str = 'tf',
        top_k_terms: Optional[int] = None
    ) -> List[str]:
        """
        Build user profile query from reading history.
        
        Args:
            user_history: List of article indices user has read
            method: 'tf' (term frequency), 'tfidf', 'concat'
            top_k_terms: Keep only top-k terms (None for all)
            
        Returns:
            Query terms for user profile
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        if len(user_history) == 0:
            return []
            
        # Collect all terms from reading history
        all_terms = []
        for idx in user_history:
            if idx < len(self.tokenized_articles):
                all_terms.extend(self.tokenized_articles[idx])
                
        if method == 'concat':
            # Just concatenate (with dedup if needed)
            query = all_terms
            
        elif method == 'tf':
            # Weight by term frequency
            term_counts = Counter(all_terms)
            query = []
            for term, count in term_counts.most_common():
                # Repeat term proportionally (capped)
                repeat = min(count, 3)
                query.extend([term] * repeat)
                
        elif method == 'tfidf':
            # Weight by TF-IDF
            term_counts = Counter(all_terms)
            term_scores = {}
            
            for term, tf in term_counts.items():
                if term in self.bm25.idf:
                    term_scores[term] = tf * self.bm25.idf[term]
                    
            # Sort by score
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            query = [term for term, _ in sorted_terms]
            
        else:
            query = list(set(all_terms))
            
        # Limit query length
        max_terms = top_k_terms or self.max_query_terms
        if len(query) > max_terms:
            query = query[:max_terms]
            
        return query
        
    def compute_scores(
        self,
        user_history: List[int],
        query_method: str = 'tfidf'
    ) -> np.ndarray:
        """
        Compute BM25 scores for all articles.
        
        Args:
            user_history: User's reading history
            query_method: How to build user query
            
        Returns:
            Scores for all articles
        """
        query = self.get_user_profile_query(user_history, method=query_method)
        
        if len(query) == 0:
            return np.zeros(self.n_articles)
            
        scores = self.bm25.get_scores(query)
        
        return scores
        
    def recommend(
        self,
        user_history: List[int],
        k: int = 10,
        exclude_read: bool = True,
        query_method: str = 'tfidf'
    ) -> Tuple[List[int], List[float]]:
        """
        Recommend top-k articles for a user.
        
        Args:
            user_history: List of article indices user has read
            k: Number of recommendations
            exclude_read: Exclude already read articles
            query_method: How to build user profile query
            
        Returns:
            (article_indices, scores)
        """
        scores = self.compute_scores(user_history, query_method)
        
        # Exclude read articles
        if exclude_read:
            for idx in user_history:
                if idx < len(scores):
                    scores[idx] = -np.inf
                    
        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
        
    def search(
        self,
        query_text: str,
        k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Search articles by query text.
        
        Args:
            query_text: Search query in Vietnamese
            k: Number of results
            
        Returns:
            (article_indices, scores)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        query_tokens = self._tokenize(query_text)
        
        if len(query_tokens) == 0:
            return [], []
            
        return self.bm25.get_top_k(query_tokens, k)
        
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
            (article_indices, scores)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted! Call fit() first.")
            
        if article_idx >= len(self.tokenized_articles):
            return [], []
            
        # Use article as query
        query = self.tokenized_articles[article_idx]
        
        return self.bm25.get_top_k(query, k + 1, exclude_indices={article_idx})
        
    def save(self, path: str):
        """Save recommender to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'bm25': {
                'k1': self.bm25.k1,
                'b': self.bm25.b,
                'delta': self.bm25.delta,
                'corpus_size': self.bm25.corpus_size,
                'avgdl': self.bm25.avgdl,
                'doc_lengths': self.bm25.doc_lengths,
                'doc_freqs': dict(self.bm25.doc_freqs),
                'idf': dict(self.bm25.idf),
                'doc_term_freqs': [dict(c) for c in self.bm25.doc_term_freqs],
            },
            'article_texts': self.article_texts,
            'tokenized_articles': self.tokenized_articles,
            'n_articles': self.n_articles,
            'config': {
                'use_preprocessing': self.use_preprocessing,
                'use_word_segmentation': self.use_word_segmentation,
                'remove_stopwords': self.remove_stopwords,
                'k1': self.k1,
                'b': self.b,
                'delta': self.delta,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"[BM25Recommender] Saved to {path}")
        
    def load(self, path: str):
        """Load recommender from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # Restore BM25
        bm25_data = data['bm25']
        self.bm25.k1 = bm25_data['k1']
        self.bm25.b = bm25_data['b']
        self.bm25.delta = bm25_data['delta']
        self.bm25.corpus_size = bm25_data['corpus_size']
        self.bm25.avgdl = bm25_data['avgdl']
        self.bm25.doc_lengths = bm25_data['doc_lengths']
        self.bm25.doc_freqs = defaultdict(int, bm25_data['doc_freqs'])
        self.bm25.idf = bm25_data['idf']
        self.bm25.doc_term_freqs = [Counter(d) for d in bm25_data['doc_term_freqs']]
        self.bm25.is_fitted = True
        
        self.article_texts = data['article_texts']
        self.tokenized_articles = data['tokenized_articles']
        self.n_articles = data['n_articles']
        self.is_fitted = True
        
        print(f"[BM25Recommender] Loaded from {path}")
        print(f"  Articles: {self.n_articles}")
        
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


class BM25Okapi(BM25):
    """
    BM25 Okapi variant - the most standard implementation.
    Same as BM25 but without the BM25+ delta.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(k1=k1, b=b, delta=0.0)


class BM25L(BM25):
    """
    BM25L variant - addresses term frequency normalization issues.
    
    Adds delta to document length normalization instead of final score.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        super().__init__(k1=k1, b=b, delta=delta)
        
    def _score_document(self, query: List[str], doc_idx: int) -> float:
        """BM25L scoring"""
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        score = 0.0
        
        for term in query:
            if term not in self.idf:
                continue
                
            tf = term_freqs.get(term, 0)
            idf = self.idf[term]
            
            # BM25L - delta in length normalization
            ctf = tf / (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (self.k1 + 1) * (ctf + self.delta) / (self.k1 + ctf + self.delta)
            
        return score


if __name__ == '__main__':
    # Demo
    print("=" * 60)
    print("BM25 Recommender Demo")
    print("=" * 60)
    
    # Sample Vietnamese news articles
    articles = [
        "Đội tuyển bóng đá Việt Nam giành chiến thắng trước Thái Lan trong trận chung kết AFF Cup",
        "Cầu thủ Nguyễn Quang Hải ghi bàn thắng quyết định trong hiệp 2 giúp đội tuyển vô địch",
        "Kinh tế Việt Nam tăng trưởng 6.5% trong quý 3 năm 2024, vượt kỳ vọng",
        "Thị trường chứng khoán Việt Nam có dấu hiệu phục hồi mạnh mẽ sau giai đoạn suy thoái",
        "Apple ra mắt iPhone 16 với nhiều tính năng AI mới, hứa hẹn thay đổi thị trường",
        "Samsung cạnh tranh với Apple trong phân khúc điện thoại cao cấp tại thị trường Việt Nam",
        "Thời tiết Hà Nội ngày mai có mưa nhỏ, nhiệt độ dao động từ 20 đến 25 độ C",
        "Dự báo thời tiết miền Bắc có không khí lạnh tràn về vào cuối tuần này",
        "Chính phủ ban hành chính sách mới hỗ trợ doanh nghiệp vừa và nhỏ phát triển",
        "Quốc hội thông qua luật đầu tư mới nhằm thu hút vốn đầu tư nước ngoài",
    ]
    
    # Initialize and fit
    recommender = BM25Recommender(
        use_preprocessing=True,
        use_word_segmentation=True,
        remove_stopwords=True,
        k1=1.5,
        b=0.75
    )
    
    recommender.fit(articles)
    
    # Test search
    print("\n" + "-" * 40)
    print("Search for 'bóng đá Việt Nam':")
    
    results, scores = recommender.search("bóng đá Việt Nam", k=3)
    for idx, score in zip(results, scores):
        print(f"  [{idx}] (score={score:.4f}) {articles[idx][:60]}...")
        
    # Test recommendations
    print("\n" + "-" * 40)
    print("User read sports articles (indices 0, 1):")
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
