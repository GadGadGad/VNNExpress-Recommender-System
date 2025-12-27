#!/usr/bin/env python3
"""
Generate embeddings for multiple Vietnamese/multilingual models.
"""
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import argparse

MODELS = {
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
    }
}

def generate_embeddings(model_key, data_path='data/processed/strict_g2', output_dir='checkpoints'):
    from sentence_transformers import SentenceTransformer
    
    model_config = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_config['name']}...")
    print(f"{'='*60}")
    
    model = SentenceTransformer(model_config['name'], trust_remote_code=model_config['trust_remote_code'])
    
    # Load articles
    articles_path = Path('data/raw/articles.csv')
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    # Load article mapping from the specific variant directory
    mapping_path = Path(data_path) / 'article_map.json'
    if not mapping_path.exists():
        # Fallback to base dir if variant subfolder doesn't exist
        mapping_path = Path('data/processed/article_map.json')
        
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
    parser.add_argument('--model', choices=list(MODELS.keys()) + ['all'], default='all')
    parser.add_argument('--data-path', default='data/processed/strict_g2', help='Path to graph variant directory')
    parser.add_argument('--output-dir', default='checkpoints')
    args = parser.parse_args()
    
    if args.model == 'all':
        for model_key in MODELS.keys():
            try:
                generate_embeddings(model_key, args.data_path, args.output_dir)
            except Exception as e:
                print(f"Error generating {model_key}: {e}")
    else:
        generate_embeddings(args.model, args.data_path, args.output_dir)

if __name__ == '__main__':
    main()
