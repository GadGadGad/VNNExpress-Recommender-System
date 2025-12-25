import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import os

def augment_graph(data_path='data/processed', top_k=5, similarity_threshold=0.7):
    """
    Augment the interaction graph with semantic content edges.
    1. Loads processed articles.
    2. Computes article-article similarity.
    3. Adds edges between items.
    """
    print("Augmenting graph with semantic edges...")
    
    # 1. Load Data
    lightgcl_data_path = os.path.join(data_path, 'lightgcl_data.pkl')
    if not os.path.exists(lightgcl_data_path):
        print(f"Error: {lightgcl_data_path} not found.")
        return
        
    import pickle
    with open(lightgcl_data_path, 'rb') as f:
        lgcl_data = pickle.load(f)
        
    item2idx = lgcl_data['item2idx'] # item_url -> index
    idx2item = lgcl_data['idx2item']
    n_items = len(item2idx)
    
    # Load articles to get text
    articles_df = pd.read_csv('data/raw/articles.csv')
    
    # Create text list in index order
    article_map = dict(zip(articles_df['url'], zip(articles_df['title'], articles_df['short_description'].fillna(""))))
    
    texts = []
    for idx in range(n_items):
        url = idx2item.get(idx, None)
        if url and url in article_map:
            title, desc = article_map[url]
            texts.append(f"{title} {desc}")
        else:
            texts.append("")
    
    # 2. Compute Similarities
    print(f"Computing TF-IDF for {len(texts)} articles...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    print(f"Injecting semantic user-item edges...")
    # Load training pairs to get user interactions
    with open(lightgcl_data_path, 'rb') as f:
        lgcl_data = pickle.load(f)
    train_data = lgcl_data['train_data'] # List of (u, i)
    
    # Create item->item mapping for fast lookup
    item_sim_map = {}
    for i in range(n_items):
        sim_scores = sim_matrix[i]
        sim_scores[i] = 0
        top_idx = np.argmax(sim_scores)
        if sim_scores[top_idx] >= similarity_threshold:
            item_sim_map[i] = top_idx
            
    augmented_pairs = []
    for u, i in train_data:
        if i in item_sim_map:
            augmented_pairs.append((u, item_sim_map[i]))
            
    print(f"Generated {len(augmented_pairs)} synthetic interactions.")
    
    # 4. Save Augmented Edges
    output_path = os.path.join(data_path, 'augmented_edges.pt')
    torch.save({
        'augmented_pairs': augmented_pairs,
    }, output_path)
    print(f"Augmented data saved to {output_path}")

if __name__ == "__main__":
    augment_graph()
