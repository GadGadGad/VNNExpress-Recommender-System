#!/usr/bin/env python3
"""
Embedding loading, resolution, and projection utilities for CF models.
Handles pretrained embeddings (PhoBERT, BGE-M3, TF-IDF, etc.), 
on-the-fly encoding fallbacks, and semantic/user-prior layers.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class SemanticEmbeddingLayer(nn.Module):
    """
    Combines hierarchical semantic IDs into a single embedding.
    """
    def __init__(self, n_codebooks, codebook_size, embedding_dim):
        super(SemanticEmbeddingLayer, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim)
            for _ in range(n_codebooks)
        ])
        
    def forward(self, semantic_ids):
        # semantic_ids: (n_items, n_codebooks)
        out = 0
        for i, emb in enumerate(self.embeddings):
            out = out + emb(semantic_ids[:, i])
        return out

class UserPriorLayer(nn.Module):
    """
    Projects dense user priors to matching embedding dimension.
    """
    def __init__(self, input_dim, output_dim):
        super(UserPriorLayer, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, user_priors):
        return self.proj(user_priors)


def load_pretrained_embeddings(embedding_type, n_items, target_dim, device='cpu', train_item_indices=None, data_path=None, articles_path=None):
    """Load pretrained embeddings and project to target dimension."""
    if embedding_type == 'random':
        return None
        
    print(f"\nLoading {embedding_type} embeddings...")
    if articles_path:
        articles_path = Path(articles_path)
    embeddings = None

    # Helper to find file in multiple locations
    def resolve_path(filename, search_paths):
        for p in search_paths:
            if p is None: continue
            candidate = Path(p) / filename
            if candidate.exists():
                return candidate
        return None

    # Common locations to search
    search_dirs = [
        'checkpoints', 
        'data/raw', 
        'data',
        '/kaggle/working/checkpoints',
        '/kaggle/input/vnexpress-news-dataset'
    ]
    if data_path:
        search_dirs.insert(0, Path(data_path).parent) # e.g., /kaggle/input/vnexpress-graph-processed
        search_dirs.insert(0, Path(data_path))
    
    # 1. Resolve Embedding Path
    emb_filename = None
    if embedding_type == 'phobert':
        emb_filename = 'phobert_article_embeddings.pt'
    elif embedding_type == 'vndoc':
        emb_filename = 'vndoc_article_embeddings.pt'
    elif embedding_type in ['bge-m3', 'gte', 'e5-large', 'e5-base', 'vn-sbert', 'vndoc']:
        name_map = {
            'bge-m3': 'bge-m3_article_embeddings.pt',
            'gte': 'gte-multilingual_article_embeddings.pt',
            'e5-large': 'e5-large_article_embeddings.pt',
            'e5-base': 'e5-base_article_embeddings.pt',
            'vn-sbert': 'vietnamese-sbert_article_embeddings.pt',
            'vndoc': 'vietnamese-document-embedding.pt'
        }
        emb_filename = name_map[embedding_type]
    
    if emb_filename:
        # Special check for checkpoints folder
        path = resolve_path(emb_filename, ['checkpoints'] + search_dirs)
        
        if path and path.exists():
             embeddings = torch.load(path, map_location='cpu')
             print(f"  Loaded {embedding_type} embeddings from {path}: {embeddings.shape}")
        else:
             print(f"  Warning: {emb_filename} not found locally.")
             
             # Fallback: Auto-Download & Encode
             print("  Attempting to download and encode on-the-fly (this may take time)...")
             try:
                 from sentence_transformers import SentenceTransformer
                 import pandas as pd
                 
                 # HF Model Names
                 hf_map = {
                     'bge-m3': 'BAAI/bge-m3',
                     'gte': 'thenlper/gte-large', 
                     'e5-large': 'intfloat/multilingual-e5-large',
                     'e5-base': 'intfloat/multilingual-e5-base',
                     'vn-sbert': 'keepitreal/vietnamese-sbert',
                     'vndoc': 'dangvantuan/vietnamese-document-embedding'
                 }
                 
                 model_name = hf_map.get(embedding_type)
                 if not model_name:
                     print(f"  No HF model mapping for {embedding_type}. Fallback to Random.")
                     return None
                     
                 # Load Articles
                 # Priority: Explicit Path > Resolved Path > Default Fallback
                 if articles_path:
                     if Path(articles_path).exists():
                         articles_path = Path(articles_path)
                         print(f"  Using explicit articles file: {articles_path}")
                     else:
                         print(f"  Warning: Explicit articles path not found: {articles_path}")
                     print(f"  ... Falling back to auto-search in: {search_dirs}")
                     if not articles_path:
                         articles_path = resolve_path('articles.csv', search_dirs)
                     if (not articles_path) and data_path: articles_path = Path(data_path).parent / 'articles.csv'
                     if not articles_path: articles_path = Path('data/raw/articles.csv')
                 else:
                     if not articles_path:
                         articles_path = resolve_path('articles.csv', search_dirs)
                     if (not articles_path) and data_path: articles_path = Path(data_path).parent / 'articles.csv'
                     if not articles_path: articles_path = Path('data/raw/articles.csv')

                 if not articles_path or not articles_path.exists():
                     print("  Standard paths failed. Attempting deep search for 'articles.csv'...")
                     potential_roots = ['/kaggle/input', 'data']
                     found = False
                     for root in potential_roots:
                         if not os.path.exists(root): continue
                         for r, d, f in os.walk(root):
                             if 'articles.csv' in f:
                                 articles_path = Path(r) / 'articles.csv'
                                 print(f"  Found articles.csv via deep search: {articles_path}")
                                 found = True
                                 break
                         if found: break

                 
                 if not articles_path or not articles_path.exists():
                     print(f"  Articles file not found. Cannot encode. Fallback to Random.")
                     return None
                     
                 df = pd.read_csv(articles_path)
                 print(f"  Loaded {len(df)} articles. Encoding with {model_name}...")
                 
                 # Prepare Text
                 text_col = 'abstract' if 'abstract' in df.columns else 'short_description'
                 if text_col not in df.columns:
                     print(f"  neither 'abstract' nor 'short_description' found. Using title only.")
                     df['text'] = df['title'].fillna('')
                 else:
                     df['text'] = df['title'].fillna('') + ' ' + df[text_col].fillna('')
                 
                 texts = df['text'].tolist()
                 
                 # Encode
                 model = SentenceTransformer(model_name, device=device)
                 emb_matrix = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
                 embeddings = emb_matrix.cpu()
                 
                 # Save for future
                 save_dir = Path('checkpoints')
                 save_dir.mkdir(exist_ok=True)
                 save_path = save_dir / emb_filename
                 torch.save(embeddings, save_path)
                 print(f"  Saved cached embeddings to {save_path}")
                 
             except ImportError:
                 print("  `sentence_transformers` not installed. Cannot auto-encode. Fallback to Random.")
                 return None
             except Exception as e:
                 print(f"  Auto-encoding failed: {e}. Fallback to Random.")
                 return None

    elif embedding_type == 'tfidf':
        print("   Computing TF-IDF embeddings (LSA)...")
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        try:
            texts = []
            
            if not articles_path:
                articles_path = resolve_path('articles.csv', search_dirs)
            
            if not articles_path:
                 articles_path = Path('data/raw/articles.csv')
            
            if not articles_path.exists():
                print(f"      Error: Articles file not found at {articles_path}")
                return None
                
            articles = pd.read_csv(articles_path)


            # Load mapping idx2item
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.data.dataloader_lightgcl import LightGCLDataLoader
            loader = LightGCLDataLoader(data_path if data_path else 'data')
            if loader.load_processed() is None:
                print("      Error: Could not load processed LightGCL data for mapping.")
                return None
            
            idx2item = loader.idx2item
            print(f"      Loaded mapping for {len(idx2item)} items")
            
            # Flatten loop for speedr
            article_map = dict(zip(articles['url'], zip(articles['title'], articles['short_description'])))
            
            for idx in range(n_items):
                url = idx2item.get(idx, None)
                if url and url in article_map:
                    title, desc = article_map[url]
                    title = str(title) if pd.notna(title) else ""
                    desc = str(desc) if pd.notna(desc) else ""
                    texts.append(f"{title} {desc}")
                else:
                    texts.append("")
            
            print(f"      Collected {len(texts)} texts. Computing TF-IDF...")
            
            vectorizer = TfidfVectorizer(max_features=10000, stop_words=None) 
            
            if train_item_indices is not None:
                print("      -> Fitting TF-IDF only on TRAIN items to prevent leakage...")
                train_texts = [texts[i] for i in train_item_indices if i < len(texts)]
                vectorizer.fit(train_texts)
                tfidf_matrix = vectorizer.transform(texts)
            else:
                print("      -> [WARNING] Fitting on ALL items (Data Snooping).")
                tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Reduce dimension with SVD (LSA)
            print(f"      Reducing dimension {tfidf_matrix.shape[1]} -> {target_dim} with SVD...")
            svd = TruncatedSVD(n_components=target_dim, random_state=42)
            embeddings = torch.tensor(svd.fit_transform(tfidf_matrix), dtype=torch.float32)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            print(f"      Computed LSA embeddings: {embeddings.shape}")
            return embeddings.to(device)

        except Exception as e:
            print(f"      Error computing TF-IDF: {e}. Using random.")
            import traceback
            traceback.print_exc()
            return None
        
    if embeddings is not None:
        # Match item count (truncate or pad if needed)
        # Check item count
        curr_items, curr_dim = embeddings.shape
        
        if curr_items != n_items:
            print(f"  Warning: Embedding items ({curr_items}) != Dataset items ({n_items})")
            # If fewer embeddings, pad with random
            if curr_items < n_items:
                padding = torch.randn(n_items - curr_items, curr_dim)
                embeddings = torch.cat([embeddings, padding], dim=0)
            else:
                embeddings = embeddings[:n_items]
        
        # Project dimension if needed
        if curr_dim != target_dim:
            print(f"  Projecting dimension: {curr_dim} -> {target_dim}")
            # Use random projection matrix
            projection = torch.randn(curr_dim, target_dim) / np.sqrt(curr_dim)
            embeddings = torch.matmul(embeddings.float(), projection.float())
            
        # Normalize embeddings to prevent numerical instability (NaN loss)
        # LightGCL uses Xavier init which produces small values. Large pretrained norms cause exp() explosion.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Check for NaNs
        if torch.isnan(embeddings).any():
             print("  Warning: Embeddings contain NaN! Replacing with random.")
             return None
             
        # Clamp values
        embeddings = torch.clamp(embeddings, -1.0, 1.0)
        
        # Scale down slightly to ensure stability (Xavier often < 1)
        embeddings = embeddings * 0.1
        
        print(f"  Final embeddings (mean={embeddings.mean():.4f}, std={embeddings.std():.4f}, max={embeddings.abs().max():.4f})")
            
        return embeddings.to(device)
            
    return None
