from rank_bm25 import BM25Plus
import pandas as pd

def build_search_engine(df):
    """
    Xây dựng chỉ mục BM25+ từ dữ liệu bài báo.
    """
    corpus = (df['title'].fillna("") + " " + df['short_description'].fillna("")).astype(str).tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Plus(tokenized_corpus)
    return bm25
