"""
Word2Vec/FastText Based Recommender for Vietnamese News
========================================================

Uses word embeddings to represent articles and compute semantic similarity.

Pretrained Vietnamese Word Embeddings:
1. fastText Vietnamese (cc.vi.300.bin) - Facebook's fastText
2. PhoW2V - Vietnamese Word2Vec from VinAI
3. Custom trained on news corpus

Advantages over TF-IDF:
    - Captures semantic similarity (synonyms, related words)
    - Dense representations (lower dimension)
    - Handles OOV words (FastText subword)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict
import pickle
import os

from .preprocessing import VietnameseTextPreprocessor


class Word2VecRecommender:
    """
    Word2Vec/FastText based content recommender for Vietnamese news.
    
    Uses word embeddings to:
    1. Compute article embeddings (weighted average of word vectors)
    2. Build user profile from reading history
    3. Find similar articles via cosine similarity
    
    Example:
        # With pretrained fastText
        recommender = Word2VecRecommender()
        recommender.load_pretrained('path/to/cc.vi.300.bin')
        recommender.fit(article_texts)
        
        recommendations = recommender.recommend([0, 1, 2], k=10)
    """
    
    def __init__(
        self,
        # Preprocessing
        use_preprocessing: bool = True,
        use_word_segmentation: bool = True,
        remove_stopwords: bool = True,
        
        # Embedding options
        embedding_dim: int = 300,
        aggregation: str = 'mean',  # 'mean', 'tfidf_weighted', 'sif'
        
        # SIF parameters (Smooth Inverse Frequency)
        sif_alpha: float = 0.001,
        use_pca_removal: bool = True,
    ):
        """
        Initialize Word2Vec recommender.
        
        Args:
            use_preprocessing: Use Vietnamese preprocessing
            use_word_segmentation: Use word segmentation
            remove_stopwords: Remove stopwords
            embedding_dim: Word embedding dimension
            aggregation: How to aggregate word vectors
                - 'mean': Simple average
                - 'tfidf_weighted': TF-IDF weighted average
                - 'sif': Smooth Inverse Frequency (recommended)
            sif_alpha: SIF parameter for word frequency weighting
            use_pca_removal: Remove first principal component (for SIF)
        """
        self.use_preprocessing = use_preprocessing
        self.use_word_segmentation = use_word_segmentation
        self.remove_stopwords = remove_stopwords
        self.embedding_dim = embedding_dim
        self.aggregation = aggregation
        self.sif_alpha = sif_alpha
        self.use_pca_removal = use_pca_removal
        
        # Preprocessor
        self.preprocessor = None
        if use_preprocessing:
            self.preprocessor = VietnameseTextPreprocessor(
                use_word_segmentation=use_word_segmentation,
                remove_stopwords=remove_stopwords
            )
            
        # Word vectors
        self.word_vectors: Optional[Dict[str, np.ndarray]] = None
        self.word_dim: int = 0
        self.vocabulary: Set[str] = set()
        
        # Word frequencies (for SIF)
        self.word_freq: Dict[str, float] = {}
        self.total_words: int = 0
        
        # Article data
        self.article_texts: Optional[List[str]] = None
        self.tokenized_articles: Optional[List[List[str]]] = None
        self.article_embeddings: Optional[np.ndarray] = None
        self.n_articles: int = 0
        
        # TF-IDF weights (optional)
        self.idf: Dict[str, float] = {}
        
        self.is_fitted = False
        self.embeddings_loaded = False
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        if self.preprocessor:
            return self.preprocessor.tokenize(text)
        return text.lower().split()
        
    def load_pretrained(
        self,
        model_path: str,
        model_type: str = 'auto',
        limit: Optional[int] = None
    ):
        """
        Load pretrained word vectors.
        
        Args:
            model_path: Path to word vectors file
            model_type: 'fasttext', 'word2vec', 'glove', or 'auto'
            limit: Limit vocabulary size
        """
        print(f"\n[Word2VecRecommender] Loading pretrained embeddings...")
        print(f"  Path: {model_path}")
        
        # Detect model type
        if model_type == 'auto':
            if model_path.endswith('.bin'):
                model_type = 'fasttext_binary'
            elif model_path.endswith('.vec'):
                model_type = 'fasttext_text'
            elif 'glove' in model_path.lower():
                model_type = 'glove'
            else:
                model_type = 'word2vec'
                
        if model_type in ['fasttext_binary', 'fasttext']:
            self._load_fasttext_binary(model_path, limit)
        elif model_type == 'fasttext_text':
            self._load_fasttext_text(model_path, limit)
        elif model_type == 'word2vec':
            self._load_word2vec(model_path, limit)
        elif model_type == 'glove':
            self._load_glove(model_path, limit)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.embeddings_loaded = True
        print(f"  Vocabulary size: {len(self.word_vectors)}")
        print(f"  Embedding dim: {self.word_dim}")
        
    def _load_fasttext_binary(self, path: str, limit: Optional[int] = None):
        """Load FastText binary model"""
        try:
            import fasttext
            model = fasttext.load_model(path)
            
            words = model.get_words()
            if limit:
                words = words[:limit]
                
            self.word_vectors = {}
            for word in words:
                self.word_vectors[word] = model.get_word_vector(word)
                
            self.word_dim = model.get_dimension()
            self.vocabulary = set(self.word_vectors.keys())
            
        except ImportError:
            print("[Warning] fasttext not installed. Install with: pip install fasttext")
            raise
            
    def _load_fasttext_text(self, path: str, limit: Optional[int] = None):
        """Load FastText text format (.vec)"""
        self.word_vectors = {}
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # First line: vocab_size dim
            header = f.readline().split()
            vocab_size = int(header[0])
            self.word_dim = int(header[1])
            
            count = 0
            for line in f:
                if limit and count >= limit:
                    break
                    
                parts = line.rstrip().split(' ')
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                
                if len(vector) == self.word_dim:
                    self.word_vectors[word] = vector
                    count += 1
                    
        self.vocabulary = set(self.word_vectors.keys())
        
    def _load_word2vec(self, path: str, limit: Optional[int] = None):
        """Load Word2Vec format (gensim)"""
        try:
            from gensim.models import KeyedVectors
            
            wv = KeyedVectors.load_word2vec_format(path, binary=path.endswith('.bin'), limit=limit)
            
            self.word_vectors = {}
            for word in wv.key_to_index:
                self.word_vectors[word] = wv[word]
                
            self.word_dim = wv.vector_size
            self.vocabulary = set(self.word_vectors.keys())
            
        except ImportError:
            print("[Warning] gensim not installed. Install with: pip install gensim")
            raise
            
    def _load_glove(self, path: str, limit: Optional[int] = None):
        """Load GloVe format"""
        self.word_vectors = {}
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            count = 0
            for line in f:
                if limit and count >= limit:
                    break
                    
                parts = line.rstrip().split(' ')
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]])
                
                if self.word_dim == 0:
                    self.word_dim = len(vector)
                    
                if len(vector) == self.word_dim:
                    self.word_vectors[word] = vector
                    count += 1
                    
        self.vocabulary = set(self.word_vectors.keys())
        
    def train_embeddings(
        self,
        corpus: List[str],
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        algorithm: str = 'skipgram'
    ):
        """
        Train word embeddings on corpus.
        
        Args:
            corpus: List of texts to train on
            vector_size: Embedding dimension
            window: Context window size
            min_count: Minimum word frequency
            epochs: Training epochs
            algorithm: 'skipgram' or 'cbow'
        """
        try:
            from gensim.models import Word2Vec
        except ImportError:
            print("[Warning] gensim not installed. Install with: pip install gensim")
            raise
            
        print(f"\n[Word2VecRecommender] Training word embeddings...")
        print(f"  Corpus size: {len(corpus)}")
        print(f"  Algorithm: {algorithm}")
        
        # Tokenize corpus
        if self.preprocessor:
            tokenized_corpus = self.preprocessor.tokenize_batch(corpus, show_progress=True)
        else:
            tokenized_corpus = [text.lower().split() for text in corpus]
            
        # Train Word2Vec
        sg = 1 if algorithm == 'skipgram' else 0
        
        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            sg=sg,
            workers=4
        )
        
        # Extract vectors
        self.word_vectors = {}
        for word in model.wv.key_to_index:
            self.word_vectors[word] = model.wv[word]
            
        self.word_dim = vector_size
        self.vocabulary = set(self.word_vectors.keys())
        self.embeddings_loaded = True
        
        print(f"  Vocabulary size: {len(self.word_vectors)}")
        print(f"  Embedding dim: {self.word_dim}")
        
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word"""
        if self.word_vectors is None:
            return None
        return self.word_vectors.get(word)
        
    def _compute_word_frequencies(self, tokenized_docs: List[List[str]]):
        """Compute word frequencies for SIF"""
        word_counts = defaultdict(int)
        total = 0
        
        for doc in tokenized_docs:
            for word in doc:
                word_counts[word] += 1
                total += 1
                
        self.word_freq = {w: c / total for w, c in word_counts.items()}
        self.total_words = total
        
    def _compute_idf(self, tokenized_docs: List[List[str]]):
        """Compute IDF for TF-IDF weighted aggregation"""
        doc_freq = defaultdict(int)
        n_docs = len(tokenized_docs)
        
        for doc in tokenized_docs:
            for word in set(doc):
                doc_freq[word] += 1
                
        self.idf = {w: np.log((n_docs + 1) / (df + 1)) for w, df in doc_freq.items()}
        
    def get_document_embedding(
        self,
        tokens: List[str],
        aggregation: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute document embedding from tokens.
        
        Args:
            tokens: List of tokens
            aggregation: Override default aggregation
            
        Returns:
            Document embedding vector
        """
        if self.word_vectors is None:
            raise ValueError("No word vectors loaded!")
            
        agg = aggregation or self.aggregation
        
        # Filter words with vectors
        valid_words = [w for w in tokens if w in self.word_vectors]
        
        if len(valid_words) == 0:
            return np.zeros(self.word_dim)
            
        vectors = np.array([self.word_vectors[w] for w in valid_words])
        
        if agg == 'mean':
            return np.mean(vectors, axis=0)
            
        elif agg == 'sum':
            return np.sum(vectors, axis=0)
            
        elif agg == 'tfidf_weighted':
            weights = np.array([self.idf.get(w, 1.0) for w in valid_words])
            weights = weights / weights.sum()
            return np.sum(vectors * weights.reshape(-1, 1), axis=0)
            
        elif agg == 'sif':
            # Smooth Inverse Frequency weighting
            weights = np.array([
                self.sif_alpha / (self.sif_alpha + self.word_freq.get(w, 0.0001))
                for w in valid_words
            ])
            return np.sum(vectors * weights.reshape(-1, 1), axis=0) / len(valid_words)
            
        else:
            return np.mean(vectors, axis=0)
            
    def _remove_principal_component(self, embeddings: np.ndarray) -> np.ndarray:
        """Remove first principal component (for SIF)"""
        # Compute principal component
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        
        # SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Remove first PC
        pc = Vt[0]
        return embeddings - np.outer(embeddings @ pc, pc)
        
    def fit(
        self,
        article_texts: List[str],
        show_progress: bool = True
    ):
        """
        Fit recommender on article texts.
        
        Args:
            article_texts: List of article texts
            show_progress: Show progress bar
        """
        if not self.embeddings_loaded:
            raise ValueError("No word embeddings loaded! Call load_pretrained() or train_embeddings() first.")
            
        self.article_texts = article_texts
        self.n_articles = len(article_texts)
        
        print(f"\n[Word2VecRecommender] Fitting on {self.n_articles} articles...")
        
        # Tokenize articles
        print("  Tokenizing articles...")
        if self.preprocessor:
            self.tokenized_articles = self.preprocessor.tokenize_batch(
                article_texts,
                show_progress=show_progress
            )
        else:
            self.tokenized_articles = [text.lower().split() for text in article_texts]
            
        # Compute word frequencies (for SIF)
        if self.aggregation == 'sif':
            print("  Computing word frequencies...")
            self._compute_word_frequencies(self.tokenized_articles)
            
        # Compute IDF (for TF-IDF weighted)
        if self.aggregation == 'tfidf_weighted':
            print("  Computing IDF weights...")
            self._compute_idf(self.tokenized_articles)
            
        # Compute article embeddings
        print("  Computing article embeddings...")
        embeddings = []
        
        iterator = self.tokenized_articles
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Embedding")
            except ImportError:
                pass
                
        for tokens in iterator:
            embed = self.get_document_embedding(tokens)
            embeddings.append(embed)
            
        self.article_embeddings = np.array(embeddings)
        
        # Remove first PC for SIF
        if self.aggregation == 'sif' and self.use_pca_removal:
            print("  Removing first principal component...")
            self.article_embeddings = self._remove_principal_component(self.article_embeddings)
            
        # Normalize embeddings
        norms = np.linalg.norm(self.article_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.article_embeddings = self.article_embeddings / norms
        
        self.is_fitted = True
        
        # Stats
        oov_count = sum(1 for tokens in self.tokenized_articles 
                       for w in tokens if w not in self.word_vectors)
        total_tokens = sum(len(tokens) for tokens in self.tokenized_articles)
        oov_rate = oov_count / total_tokens if total_tokens > 0 else 0
        
        print(f"  OOV rate: {oov_rate:.2%}")
        print(f"  Embeddings shape: {self.article_embeddings.shape}")
        
    def get_user_profile(
        self,
        user_history: List[int],
        aggregation: str = 'mean',
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute user profile embedding from reading history.
        
        Args:
            user_history: List of article indices
            aggregation: How to aggregate article embeddings
            weights: Optional weights for each article
            
        Returns:
            User profile embedding
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted!")
            
        if len(user_history) == 0:
            return np.zeros(self.word_dim)
            
        history_embeddings = self.article_embeddings[user_history]
        
        if aggregation == 'mean':
            profile = np.mean(history_embeddings, axis=0)
        elif aggregation == 'sum':
            profile = np.sum(history_embeddings, axis=0)
        elif aggregation == 'weighted' and weights is not None:
            weights = np.array(weights).reshape(-1, 1)
            profile = np.sum(history_embeddings * weights, axis=0) / weights.sum()
        elif aggregation == 'recency':
            # Recent articles get higher weight
            weights = np.linspace(0.5, 1.0, len(user_history)).reshape(-1, 1)
            profile = np.sum(history_embeddings * weights, axis=0) / weights.sum()
        else:
            profile = np.mean(history_embeddings, axis=0)
            
        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
            
        return profile
        
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
            Similarity scores
        """
        user_profile = self.get_user_profile(user_history, aggregation)
        
        # Cosine similarity (embeddings are normalized)
        scores = self.article_embeddings @ user_profile
        
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
        
        if exclude_read:
            for idx in user_history:
                if idx < len(scores):
                    scores[idx] = -np.inf
                    
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
            article_idx: Target article index
            k: Number of similar articles
            
        Returns:
            (article_indices, similarity_scores)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted!")
            
        article_embed = self.article_embeddings[article_idx]
        similarities = self.article_embeddings @ article_embed
        
        # Exclude self
        similarities[article_idx] = -np.inf
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
        
    def find_similar_by_text(
        self,
        query_text: str,
        k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Find articles similar to query text.
        
        Args:
            query_text: Query text
            k: Number of results
            
        Returns:
            (article_indices, similarity_scores)
        """
        if not self.is_fitted:
            raise ValueError("Recommender not fitted!")
            
        tokens = self._tokenize(query_text)
        query_embed = self.get_document_embedding(tokens)
        
        # Normalize
        norm = np.linalg.norm(query_embed)
        if norm > 0:
            query_embed = query_embed / norm
            
        similarities = self.article_embeddings @ query_embed
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.tolist(), top_k_scores.tolist()
        
    def word_similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words"""
        v1 = self.get_word_vector(word1)
        v2 = self.get_word_vector(word2)
        
        if v1 is None or v2 is None:
            return 0.0
            
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    def most_similar_words(self, word: str, k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words"""
        vector = self.get_word_vector(word)
        
        if vector is None:
            return []
            
        similarities = []
        for w, v in self.word_vectors.items():
            if w != word:
                sim = np.dot(vector, v) / (np.linalg.norm(vector) * np.linalg.norm(v))
                similarities.append((w, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
        
    def save(self, path: str):
        """Save recommender to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'article_embeddings': self.article_embeddings,
            'article_texts': self.article_texts,
            'tokenized_articles': self.tokenized_articles,
            'n_articles': self.n_articles,
            'word_dim': self.word_dim,
            'word_freq': self.word_freq,
            'idf': self.idf,
            'config': {
                'use_preprocessing': self.use_preprocessing,
                'aggregation': self.aggregation,
                'sif_alpha': self.sif_alpha,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"[Word2VecRecommender] Saved to {path}")
        
    def load(self, path: str):
        """Load recommender from disk (word vectors must be loaded separately)"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        self.article_embeddings = data['article_embeddings']
        self.article_texts = data['article_texts']
        self.tokenized_articles = data['tokenized_articles']
        self.n_articles = data['n_articles']
        self.word_dim = data['word_dim']
        self.word_freq = data['word_freq']
        self.idf = data['idf']
        self.is_fitted = True
        
        print(f"[Word2VecRecommender] Loaded from {path}")
        print(f"  Articles: {self.n_articles}")
        
    def evaluate(
        self,
        train_dict: Dict[int, Set[int]],
        test_dict: Dict[int, List[int]],
        k_list: List[int] = [10, 20, 50]
    ) -> Dict[str, float]:
        """Evaluate recommendation performance"""
        from tqdm import tqdm
        
        results = defaultdict(list)
        max_k = max(k_list)
        
        for user_id in tqdm(test_dict.keys(), desc="Evaluating"):
            train_items = list(train_dict.get(user_id, set()))
            test_items = set(test_dict[user_id])
            
            if len(train_items) == 0 or len(test_items) == 0:
                continue
                
            scores = self.compute_scores(train_items)
            
            for item in train_items:
                if item < len(scores):
                    scores[item] = -np.inf
                    
            top_items = np.argsort(scores)[::-1][:max_k]
            
            for k in k_list:
                top_k = top_items[:k]
                hits = len(set(top_k) & test_items)
                
                recall = hits / len(test_items)
                results[f'Recall@{k}'].append(recall)
                
                dcg = sum([1.0 / np.log2(i + 2) 
                          for i, item in enumerate(top_k) if item in test_items])
                idcg = sum([1.0 / np.log2(i + 2) 
                           for i in range(min(len(test_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                results[f'NDCG@{k}'].append(ndcg)
                
                hr = 1.0 if hits > 0 else 0.0
                results[f'HR@{k}'].append(hr)
                
        avg_results = {}
        for key, values in results.items():
            avg_results[key] = np.mean(values) if len(values) > 0 else 0.0
            
        return avg_results


class FastTextRecommender(Word2VecRecommender):
    """
    Specialized recommender using FastText.
    
    FastText advantages:
    - Handles OOV words via subword embeddings
    - Better for morphologically rich languages
    """
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        if model_path:
            self.load_fasttext(model_path)
            
        # FastText model for OOV handling
        self._fasttext_model = None
        
    def load_fasttext(self, path: str):
        """Load FastText model with OOV support"""
        try:
            import fasttext
            self._fasttext_model = fasttext.load_model(path)
            
            # Also populate word_vectors dict
            words = self._fasttext_model.get_words()
            self.word_vectors = {}
            for word in words:
                self.word_vectors[word] = self._fasttext_model.get_word_vector(word)
                
            self.word_dim = self._fasttext_model.get_dimension()
            self.vocabulary = set(self.word_vectors.keys())
            self.embeddings_loaded = True
            
            print(f"[FastTextRecommender] Loaded model: {path}")
            print(f"  Vocabulary: {len(self.word_vectors)}")
            print(f"  Dimension: {self.word_dim}")
            
        except ImportError:
            print("[Warning] fasttext not installed")
            raise
            
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get vector for word (handles OOV with subwords)"""
        if self._fasttext_model is not None:
            # FastText can generate vectors for OOV words
            return self._fasttext_model.get_word_vector(word)
        return super().get_word_vector(word)


if __name__ == '__main__':
    # Demo
    print("=" * 60)
    print("Word2Vec/FastText Recommender Demo")
    print("=" * 60)
    
    # Sample Vietnamese news articles
    articles = [
        "Đội tuyển bóng đá Việt Nam giành chiến thắng trước Thái Lan",
        "Cầu thủ Quang Hải ghi bàn thắng quyết định giúp đội vô địch",
        "Kinh tế Việt Nam tăng trưởng mạnh trong năm 2024",
        "Thị trường chứng khoán có dấu hiệu phục hồi",
        "Apple ra mắt iPhone 16 với nhiều tính năng mới",
        "Samsung cạnh tranh trong phân khúc điện thoại cao cấp",
        "Thời tiết Hà Nội ngày mai có mưa nhỏ",
        "Dự báo miền Bắc có không khí lạnh tràn về",
    ]
    
    print("\n[Note] This demo requires pretrained word vectors.")
    print("       Download Vietnamese fastText: cc.vi.300.bin")
    print("       Or train on your own corpus.\n")
    
    # Demo with mock word vectors
    print("Creating mock word vectors for demo...")
    
    recommender = Word2VecRecommender(
        use_preprocessing=True,
        use_word_segmentation=True,
        aggregation='mean'
    )
    
    # Create mock word vectors
    np.random.seed(42)
    mock_vocab = ['bóng_đá', 'việt_nam', 'thái_lan', 'kinh_tế', 'chứng_khoán', 
                  'iphone', 'samsung', 'thời_tiết', 'mưa', 'hà_nội', 'đội_tuyển',
                  'cầu_thủ', 'quang_hải', 'bàn_thắng', 'vô_địch', 'tăng_trưởng',
                  'thị_trường', 'phục_hồi', 'điện_thoại', 'cao_cấp', 'dự_báo',
                  'miền_bắc', 'không_khí', 'lạnh', 'chiến_thắng']
                  
    recommender.word_vectors = {
        word: np.random.randn(100) for word in mock_vocab
    }
    recommender.word_dim = 100
    recommender.vocabulary = set(mock_vocab)
    recommender.embeddings_loaded = True
    
    # Fit on articles
    recommender.fit(articles)
    
    # Test recommendations
    print("\n" + "-" * 40)
    print("User read sports articles (indices 0, 1):")
    print(f"  [0] {articles[0]}")
    print(f"  [1] {articles[1]}")
    
    top_items, top_scores = recommender.recommend([0, 1], k=3)
    print("\nRecommendations:")
    for idx, score in zip(top_items, top_scores):
        print(f"  [{idx}] (score={score:.4f}) {articles[idx]}")
        
    # Similar articles
    print("\n" + "-" * 40)
    print("Similar articles to article 2 (economy):")
    print(f"  [2] {articles[2]}")
    
    similar_items, similar_scores = recommender.find_similar_articles(2, k=3)
    for idx, score in zip(similar_items, similar_scores):
        print(f"  [{idx}] (sim={score:.4f}) {articles[idx]}")
