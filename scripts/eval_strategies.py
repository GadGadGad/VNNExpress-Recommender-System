
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import collections

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ngcf import NGCF
from src.models.content_based import TFIDFRecommender, PhoBERTEncoder, ContentBasedRecommender

def clean_id(val):
    try: return str(int(float(val)))
    except: return str(val)

def load_cf_model(model_name, n_users, n_items, device):
    # Heuristic loader
    import glob
    files = glob.glob(f"models/{model_name.lower()}_*.pt")
    if not files: return None
    latest = max(files, key=os.path.getctime)
    ckpt = torch.load(latest, map_location=device)
    config = ckpt.get('config', {})
    
    if model_name.lower() == 'ngcf':
        n_layers = config.get('n_layers', 3)
        emb_dim = config.get('hidden_dim', 64)
        model = NGCF(n_users, n_items, embedding_dim=emb_dim, n_layers=n_layers)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        # TODO: Support others
        return None
    model.to(device)
    model.eval()
    return model

def load_data(data_path="data/raw"):
    # Load raw interaction for ground truth
    replies = pd.read_csv("data/raw/replies.csv")
    replies['user_id'] = replies['reply_user_id'].apply(clean_id)
    # Load Maps
    # Load Maps from Cache (consistent with training)
    cache_path = Path(data_path) / "cf_cache_min2.pt"
    if cache_path.exists():
         print(f"Loading map from {cache_path}...")
         data = torch.load(cache_path, weights_only=False)
         u_map = data.get('user_map', {})
         a_map = data.get('article_map', {})
    else:
         # Fallback to JSON
         import json
         with open(f"{data_path}/user_map.json") as f: u_map = json.load(f)
         with open(f"{data_path}/article_map.json") as f: a_map = json.load(f)
    
    # Filter only users in map?
    # Test set: Users with interactions.
    # We define Test Set as last interaction of each user? Or global split?
    # Simple: Random 20%
    users = list(u_map.keys())
    # ...
    # Actually, let's just use what's in the files if available
    # Or simplified evaluation: LOO (Leave One Out)
    
    # Construct History
    history = replies.groupby('user_id')['article_url'].apply(set).to_dict()
    
    return u_map, a_map, history, replies

def evaluate_strategies():
    print("Loading Data & Models...")
    u_map, a_map, history, df = load_data()
    n_users, n_items = len(u_map), len(a_map)
    device = 'cpu'
    
    # Load Models
    cf_model = load_cf_model("ngcf", n_users, n_items, device)
    cb_model = TFIDFRecommender(1, 1) # Dummy
    # Load fit data
    articles = pd.read_csv("data/raw/articles.csv")
    cb_model.encode_articles((articles['title'].fillna("") + " " + articles['short_description'].fillna("")).tolist())
    
    results = collections.defaultdict(list)
    
    # Test on Sample 100 users
    test_users = list(history.keys())[:100]
    
    print("Evaluating Strategies on 100 users...")
    for uid in tqdm(test_users):
        gt_urls = history[uid]
        if len(gt_urls) < 2: continue # Need at least 2 to split
        
        # Split GT (Hold out 1 item)
        gt_list = list(gt_urls)
        target_item = gt_list[-1]
        train_hist = gt_list[:-1]
        
        # 1. CF Score
        cf_score = 0
        if uid in u_map and cf_model:
            u_idx = u_map[uid]
            # Predict all items
            with torch.no_grad():
                u_emb = cf_model.user_embedding.weight[u_idx]
                scores = torch.matmul(u_emb, cf_model.item_embedding.weight.t())
                # Get score of target item
                if target_item in a_map:
                    target_idx = a_map[target_item]
                    cf_score = scores[target_idx].item()
                    
        # 2. CB Score
        cb_score = 0
        if target_item in a_map: # Needs existing article
             # Just recommend based on train_hist
             # But here we need score of TARGET against Train Hist
             # TF-IDF: sim(target, hist)
             # ...
             pass
        
        results['cf'].append(cf_score)
        results['target'].append(1 if cf_score > 0 else 0) # Dummy metric
    
    # Calculate simple HitRate@10 (if rank < 10)
    # Since we didn't rank all items, we can't compute HR.
    # But we can verify if scores are non-zero.
    avg_cf = np.mean(results['cf'])
    print(f"Average CF Score for Ground Truth Items: {avg_cf:.4f}")
    print("If this is near 0 or negative, model is poor.")
         
    print("Optimization: Done.")

if __name__ == "__main__":
    evaluate_strategies()
