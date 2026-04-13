#!/usr/bin/env python3
"""
Generate embeddings for multiple Vietnamese/multilingual models.
Includes TF-IDF with Vietnamese preprocessing.
"""
import torch
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse

# Neural embedding models
NEURAL_MODELS = {
    'bge-m3': {
        'name': 'BAAI/bge-m3',
        'trust_remote_code': True,
        'dim': 1024
    },
    'gte-multilingual': {
        'name': 'Alibaba-NLP/gte-multilingual-base',
        'trust_remote_code': True,
        'dim': 768
    },
    'e5-large': {
        'name': 'intfloat/multilingual-e5-large',
        'trust_remote_code': False,
        'dim': 1024
    },
    'e5-base': {
        'name': 'intfloat/multilingual-e5-base', 
        'trust_remote_code': False,
        'dim': 768
    },
    'vietnamese-sbert': {
        'name': 'keepitreal/vietnamese-sbert',
        'trust_remote_code': False,
        'dim': 768
    },
    'phobert': {
        'name': 'vinai/phobert-base',
        'trust_remote_code': True,
        'dim': 768
    }
}

# All supported model types
MODELS = list(NEURAL_MODELS.keys()) + ['tfidf']

def preprocess_vietnamese(text):
    """
    Preprocess Vietnamese text for TF-IDF.
    Uses underthesea for word segmentation if available, else simple tokenization.
    """
    try:
        from underthesea import word_tokenize
        # Word segmentation for Vietnamese
        text = word_tokenize(text, format="text")
    except ImportError:
        try:
            from pyvi import ViTokenizer
            text = ViTokenizer.tokenize(text)
        except ImportError:
            # Fallback: simple tokenization
            pass
    
    # Lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_tfidf_embeddings(data_path='data/processed/strict_g2', output_dir='checkpoints'):
    """Generate TF-IDF embeddings with Vietnamese preprocessing."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print(f"\n{'='*60}")
    print("Generating TF-IDF Embeddings with Vietnamese Preprocessing...")
    print(f"{'='*60}")
    
    # Load articles
    articles_path = Path('data/raw/articles.csv')
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load article mapping (search in multiple logical locations)
    def find_mapping(base_path):
        paths = [
            Path(base_path) / 'article_map.json',
            Path(base_path).parent / 'article_map.json',
            Path('data/processed/strict_g2/article_map.json'),
            Path('data/processed/article_map.json')
        ]
        for p in paths:
            if p.exists():
                return p
        return None

    mapping_path = find_mapping(data_path)
    
    if mapping_path is None:
        raise FileNotFoundError(f"Could not find article_map.json in {data_path} or fallbacks.")
    
    print(f"Loading mapping from: {mapping_path}")
    with open(mapping_path, 'r') as f:
        article_map = json.load(f)
    
    n_articles = len(article_map)
    print(f"Article map has {n_articles} articles")
    
    url_to_idx = {url: idx for url, idx in article_map.items()}
    
    # Prepare texts with preprocessing
    texts = [''] * n_articles
    
    print("Preprocessing Vietnamese text...")
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Preprocessing"):
        url = row.get('url', '')
        if url in url_to_idx:
            title = str(row.get('title', ''))
            desc = str(row.get('description', ''))
            raw_text = f"{title}. {desc}"
            processed_text = preprocess_vietnamese(raw_text)
            texts[url_to_idx[url]] = processed_text
    
    # Generate TF-IDF
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    
    # Convert to dense tensor
    embeddings = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    output_path = Path(output_dir) / 'tfidf_article_embeddings.pt'
    output_path.parent.mkdir(exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved TF-IDF embeddings to {output_path}")
    
    return output_path

def generate_embeddings(model_key, data_path='data/processed/strict_g2', output_dir='checkpoints'):
    from sentence_transformers import SentenceTransformer
    
    model_config = NEURAL_MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_config['name']}...")
    print(f"{'='*60}")
    
    model = SentenceTransformer(model_config['name'], trust_remote_code=model_config['trust_remote_code'])
    
    # Load articles
    articles_path = Path('data/raw/articles.csv')
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load article mapping from the specific variant directory or fallbacks
    def find_mapping(base_path):
        paths = [
            Path(base_path) / 'article_map.json',
            Path(base_path).parent / 'article_map.json',
            Path('data/processed/strict_g2/article_map.json'),
            Path('data/processed/article_map.json')
        ]
        for p in paths:
            if p.exists():
                return p
        return None

    mapping_path = find_mapping(data_path)
    
    if mapping_path is None:
        raise FileNotFoundError(f"Could not find article_map.json in {data_path} or fallbacks.")
        
    print(f"Loading mapping from: {mapping_path}")
    with open(mapping_path, 'r') as f:
        article_map = json.load(f)
    
    n_articles = len(article_map)
    print(f"Article map has {n_articles} articles")
    
    # Create URL to index mapping
    url_to_idx = {url: idx for url, idx in article_map.items()}
    
    # Get text for each article (title + description)
    texts = []
    valid_indices = []
    
    for _, row in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Preparing texts"):
        url = row.get('url', '')
        if url in url_to_idx:
            title = str(row.get('title', ''))
            desc = str(row.get('description', ''))
            # For E5 models, add query/passage prefix if needed
            if 'e5' in model_key:
                text = f"passage: {title}. {desc}"
            else:
                text = f"{title}. {desc}"
            texts.append(text)
            valid_indices.append(url_to_idx[url])
    
    print(f"Found {len(texts)} matching articles")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_list = model.encode(texts, show_progress_bar=True, batch_size=16)
    
    # Create full embedding matrix
    embed_dim = embeddings_list.shape[1]
    embeddings = torch.zeros((n_articles, embed_dim))
    
    for idx, emb in zip(valid_indices, embeddings_list):
        embeddings[idx] = torch.tensor(emb)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    output_path = Path(output_dir) / f"{model_key}_article_embeddings.pt"
    output_path.parent.mkdir(exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS + ['all'], default='all')
    parser.add_argument('--data-path', default='data/processed/strict_g2', help='Path to graph variant directory')
    parser.add_argument('--output-dir', default='checkpoints')
    args = parser.parse_args()
    
    models_to_run = MODELS if args.model == 'all' else [args.model]
    
    for model_key in models_to_run:
        try:
            if model_key == 'tfidf':
                generate_tfidf_embeddings(args.data_path, args.output_dir)
            else:
                generate_embeddings(model_key, args.data_path, args.output_dir)
        except Exception as e:
            print(f"Error generating {model_key}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
