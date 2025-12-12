import torch
import torch.nn.functional as F
import pandas as pd
import json
import argparse
import random
from pathlib import Path
import glob
import os
from collections import Counter
import numpy as np

def load_latest_model(model_dir='models'):
    files = glob.glob(f"{model_dir}/lightgcn_*.pt")
    if not files:
        raise FileNotFoundError("No models found!")
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading latest model: {latest_file}")
    try:
        return torch.load(latest_file, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(latest_file, map_location='cpu')

def check_bias(data_dir, articles_path, replies_path, n_users=500, k=10):
    print("Loading data...")
    # Load Model
    checkpoint = load_latest_model()
    node_embeddings = checkpoint.get('node_embeddings')
    if not node_embeddings:
        print("[ERROR] No node_embeddings in checkpoint.")
        return

    user_emb = node_embeddings['user']
    article_emb = node_embeddings['article']
    
    # Load Mappings
    with open(f"{data_dir}/user_map.json") as f:
        user_map = json.load(f)
    with open(f"{data_dir}/article_map.json") as f:
        article_map = json.load(f)
        
    idx_to_article = {v: k for k, v in article_map.items()}
    
    # Load Popularity Data
    replies_df = pd.read_csv(replies_path)
    # Calculate interaction count for each article
    article_pop_counts = replies_df['article_url'].value_counts()
    
    # Global statistics
    total_interactions = article_pop_counts.sum()
    unique_articles = len(article_map)
    
    # Map raw popularity to index
    pop_by_idx = torch.zeros(len(article_emb))
    for url, count in article_pop_counts.items():
        if url in article_map:
            pop_by_idx[article_map[url]] = count
            
    avg_article_pop = pop_by_idx.mean().item()
    print(f"Global Average Article Popularity: {avg_article_pop:.2f}")

    # --- SIMULATE RECOMMENDATIONS ---
    print(f"\nSimulating recommendations for {n_users} random users...")
    
    all_users = list(user_map.values())
    sampled_users = random.sample(all_users, min(n_users, len(all_users)))
    
    recommended_items = []
    total_rec_pop = 0
    count_recs = 0
    
    # Batch processing could be faster, but loop is fine for n=500
    for i, u_idx in enumerate(sampled_users):
        u_vec = user_emb[u_idx].unsqueeze(0)
        scores = (u_vec @ article_emb.T).squeeze(0)
        
        # We assume training data exclusion is handled or we just care about raw model bias here
        # For pure bias check, typically we might not filter seen items to check model tendency,
        # but to match production, let's just take top K raw scores for simplicity 
        # (filtering seen requires loading history which is heavy, and bias exists regardless)
        
        topk = torch.topk(scores, k)
        indices = topk.indices.tolist()
        
        recommended_items.extend(indices)
        
        # Sum popularity of recommended items
        batch_pops = pop_by_idx[indices].sum().item()
        total_rec_pop += batch_pops
        count_recs += k
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} users...")

    # --- CALCULATE METRICS ---
    arp = total_rec_pop / count_recs
    unique_recs = len(set(recommended_items))
    coverage = unique_recs / unique_articles
    
    print("\n" + "="*40)
    print("POPULARITY BIAS METRICS")
    print("="*40)
    print(f"Average Recommendation Popularity (ARP): {arp:.2f}")
    print(f"Global Average Popularity:             {avg_article_pop:.2f}")
    print(f"ARP / Global Ratio:                      {arp/avg_article_pop:.2f}x")
    print(f"Item Coverage:                           {coverage:.2%} ({unique_recs}/{unique_articles})")
    print("="*40)
    
    if arp > avg_article_pop * 5:
        print("\n[CONCLUSION] Strong popularity bias detected.")
    elif arp > avg_article_pop * 2:
        print("\n[CONCLUSION] Moderate popularity bias detected.")
    else:
        print("\n[CONCLUSION] Low or no popularity bias detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed_phobert')
    parser.add_argument('--articles', default='data/raw/articles.csv')
    parser.add_argument('--replies', default='data/raw/replies.csv')
    parser.add_argument('--n-users', type=int, default=500, help='Number of users to sample')
    parser.add_argument('--k', type=int, default=10, help='Top-K items to recommend')
    
    args = parser.parse_args()
    
    check_bias(args.data_dir, args.articles, args.replies, args.n_users, args.k)
