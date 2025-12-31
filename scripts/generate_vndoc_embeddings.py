#!/usr/bin/env python3
"""
Generate Vietnamese Document Embeddings for Articles
Uses: dangvantuan/vietnamese-document-embedding
"""
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def main():
    print("Loading Vietnamese Document Embedding model...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('dangvantuan/vietnamese-document-embedding', trust_remote_code=True)
    
    # Load articles
    articles_path = Path('data/raw/articles.csv')
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load article mapping
    with open('data/processed/enhanced_v1/article_map.json', 'r') as f:
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
            text = f"{title}. {desc}"
            texts.append(text)
            valid_indices.append(url_to_idx[url])
    
    print(f"Found {len(texts)} matching articles")
    
    # Generate embeddings
    print("Generating embeddings (this may take a while)...")
    embeddings_list = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Create full embedding matrix
    embed_dim = embeddings_list.shape[1]
    embeddings = torch.zeros((n_articles, embed_dim))
    
    for idx, emb in zip(valid_indices, embeddings_list):
        embeddings[idx] = torch.tensor(emb)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save
    output_path = Path('checkpoints/vndoc_article_embeddings.pt')
    output_path.parent.mkdir(exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Saved embeddings to {output_path}")

if __name__ == '__main__':
    main()
