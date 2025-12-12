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

def inspect(data_dir, articles_path, replies_path, rerank=False):
    # Load Model
    checkpoint = load_latest_model()
    
    # Load Embeddings
    node_embeddings = checkpoint.get('node_embeddings')
    if not node_embeddings:
        print("[ERROR] No node_embeddings in checkpoint.")
        return
        
    user_emb = node_embeddings['user']
    article_emb = node_embeddings['article']
    
    # Load Mappings
    print(f"Loading mappings from {data_dir}...")
    with open(f"{data_dir}/user_map.json") as f:
        user_map = json.load(f)
    with open(f"{data_dir}/article_map.json") as f:
        article_map = json.load(f)
        
    idx_to_user = {v: k for k, v in user_map.items()}
    idx_to_article = {v: k for k, v in article_map.items()}
    
    # Load Articles Metadata
    print(f"Loading articles from {articles_path}...")
    articles_df = pd.read_csv(articles_path)
    url_to_meta = {}
    for _, row in articles_df.iterrows():
        url_to_meta[row['url']] = {
            'title': row.get('title', 'No Title'),
            'desc': row.get('short_description', ''),
            'cat': row.get('category', 'Unknown')
        }
        
    # Load Replies for Context & Popularity
    replies_df = pd.read_csv(replies_path)
    article_pop = replies_df['article_url'].value_counts().to_dict()
    
    df1 = replies_df[['parent_user_id', 'article_url']].rename(columns={'parent_user_id': 'user_id'})
    df2 = replies_df[['reply_user_id', 'article_url']].rename(columns={'reply_user_id': 'user_id'})
    combined = pd.concat([df1, df2]).dropna()
    combined['user_id'] = combined['user_id'].astype(str)
    combined['user_id'] = combined['user_id'].str.replace(r'\.0$', '', regex=True)
    
    user_history = combined.groupby('user_id')['article_url'].apply(list).to_dict()
    print(f"Users with history loaded: {len(user_history):,}")

    # Interactive Loop
    print("\n" + "="*60)
    print("RECOMMENDATION INSPECTOR (With Metrics)")
    print("="*60)
    
    while True:
        try:
            inp = input("\nEnter User ID (or 'r', 'q'): ").strip()
        except EOFError:
            break
            
        if inp.lower() == 'q':
            break
            
        if inp.lower() == 'r' or inp == "":
            user_idx = random.randint(0, len(user_emb)-1)
            original_uid = str(idx_to_user[user_idx])
            print(f"\n[Random User] Index: {user_idx} | ID: {original_uid}")
        else:
            if inp in user_map:
                original_uid = inp
                user_idx = user_map[inp]
            elif int(inp) in user_map:
                 original_uid = int(inp)
                 user_idx = user_map[original_uid]
            else:
                print("User ID not found.")
                continue

        # --- CONTEXT ---
        history_urls = []
        if original_uid in user_history:
            history_urls = user_history[original_uid]
        elif int(original_uid) in user_history:
            history_urls = user_history[int(original_uid)]
        elif str(original_uid) in user_history:
            history_urls = user_history[str(original_uid)]
            
        print(f"\n--- User Profile ({len(history_urls)} interactions) ---")
        
        cats = [url_to_meta.get(u, {}).get('cat', 'Unknown') for u in history_urls]
        user_top_cats = set()
        if cats:
            print("Top Categories:")
            total = len(cats)
            for c, count in Counter(cats).most_common(3):
                print(f"   - {c}: {count} ({count/total:.1%})")
                user_top_cats.add(c)

        # --- RECOMMENDATIONS ---
        u_vec = user_emb[user_idx].unsqueeze(0)
        scores = (u_vec @ article_emb.T).squeeze(0)
        
        # Re-Ranking Boost
        if rerank and user_top_cats:
            print(f"\n[INFO] Boosting Top Categories: {user_top_cats} by 1.5x")
            boost_mask = torch.ones_like(scores)
            for i in range(len(scores)):
                url = idx_to_article[i]
                cat = url_to_meta.get(url, {}).get('cat', 'Unknown')
                if cat in user_top_cats:
                    boost_mask[i] = 1.5
            scores = scores * boost_mask
        
        # Filter seen
        seen_indices = []
        for url in set(history_urls):
            if url in article_map:
                seen_indices.append(article_map[url])
        scores[seen_indices] = -float('inf')
        
        topk = torch.topk(scores, 10)
        indices = topk.indices.tolist()
        values = topk.values.tolist()
        
        print(f"\n--- Top 10 Recommendations ---")
        rec_cats = [url_to_meta.get(idx_to_article[i], {}).get('cat', 'Unknown') for i in indices]
        rec_pops = [article_pop.get(idx_to_article[i], 0) for i in indices]
        
        # METRICS
        avg_pop = sum(rec_pops) / 10
        matches = sum(1 for c in rec_cats if c in user_top_cats)
        match_rate = matches / 10
        
        print(f"METRICS:\n   - Avg Popularity: {avg_pop:.1f} (Lower is more niche)\n   - Category Match: {match_rate:.0%} (Higher is better)")
        print("-" * 40)
            
        for rank, (idx, score) in enumerate(zip(indices, values), 1):
            url = idx_to_article[idx]
            meta = url_to_meta.get(url, {'title': 'Unknown', 'desc': '', 'cat': '?'})
            pop = article_pop.get(url, 0)
            print(f"{rank}. [{score:.4f}] [Pop:{pop}] [{meta['cat']}] {meta['title']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/processed_phobert')
    parser.add_argument('--articles', default='data/raw/articles.csv')
    parser.add_argument('--replies', default='data/raw/replies.csv')
    parser.add_argument('--rerank', action='store_true', help='Apply category-based re-ranking to mitigate popularity bias')
    args = parser.parse_args()
    
    inspect(args.data_dir, args.articles, args.replies, args.rerank)
