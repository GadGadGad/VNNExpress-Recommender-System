"""
Vietnamese NLP for News Recommendation
=======================================

This module provides traditional NLP methods for Vietnamese text,
specifically designed for news recommendation.

Modules:
    - preprocessing: Text preprocessing with stopwords, word segmentation
    - tfidf_recommender: TF-IDF based content recommendation
    - bm25_recommender: BM25 ranking for news articles
    - word2vec_recommender: Word2Vec/FastText based recommendation
"""

from .preprocessing import VietnameseTextPreprocessor
from .tfidf_recommender import TFIDFRecommender
from .bm25_recommender import BM25Recommender
from .word2vec_recommender import Word2VecRecommender

__all__ = [
    'VietnameseTextPreprocessor',
    'TFIDFRecommender',
    'BM25Recommender',
    'Word2VecRecommender'
]
