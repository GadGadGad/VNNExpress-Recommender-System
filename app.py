
import streamlit as st
import pandas as pd
import numpy as np
import json
import subprocess
import sys
import os
import glob
from pathlib import Path
import re
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.inference.re_ranker import CalibratedReRanker
import random
import shutil
from src.models.semantic_id import generate_semantic_ids


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="News RecSys Comprehensive",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI/UX Customization ---
st.markdown("""
<style>
    /* Navigation Bar (Radio) */
    div.row-widget.stRadio > div {
        flex-direction: row;
        justify-content: center;
        background-color: #f0f2f6;
        padding: 5px;
        border-radius: 12px;
        gap: 5px;
    }
    div.row-widget.stRadio > div > label {
        flex: 1;
        text-align: center;
        padding: 8px 16px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
        margin: 0;
    }
    div.row-widget.stRadio > div > label:hover {
        background-color: #e0e2e6;
    }
    div.row-widget.stRadio > div > label[data-baseweb="radio"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Global Typography */
    h1, h2, h3 {
        color: #1a1a1a; 
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
MODELS_DIR = "models"

MODEL_OPTIONS = {
    "CF": ["NGCF", "LightGCL", "SimGCL", "XSimGCL", "CGRC", "IGCL", "BIGCF"], 
    "CB": ["TF-IDF", "PhoBERT", "SimCSE", "Hybrid"]
}

CATEGORY_MAP = {
    "giaoduc": "📚 Giáo dục",
    "khcn": "🔬 Khoa học & Công nghệ",
    "kinhdoanh": "💼 Kinh doanh",
    "thegioi": "🌍 Thế giới",
    "thethao": "⚽ Thể thao",
    "thoisu": "📰 Thời sự",
    "giai-tri": "🎬 Giải trí",
    "suc-khoe": "🏥 Sức khỏe",
    "xe": "🚗 Xe",
    "du-lich": "✈️ Du lịch",
    "N/A": "❓ Khác"
}


def score_to_color(score, base_hue=120, min_lightness=30, max_lightness=70):
    """Convert score (0-1) to HSL color. Higher score = darker (more saturated)."""
    score = max(0, min(1, score))  # Clamp to 0-1
    lightness = max_lightness - score * (max_lightness - min_lightness)
    saturation = 50 + score * 30  # 50-80%
    return f"hsl({base_hue}, {saturation:.0f}%, {lightness:.0f}%)"


class PhoBERTWrapper:
    """Wrapper for pre-computed PhoBERT/SimCSE embeddings for recommendation."""
    
    def __init__(self, embeddings, articles_df):
        self.embeddings = embeddings  # Shape: (n_articles, embed_dim)
        self.articles_df = articles_df
        # Normalize embeddings for cosine similarity
        self.embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend based on cosine similarity to history."""
        if not history_indices:
            return [], []
        
        # Average embedding of history articles
        history_embs = self.embeddings_norm[history_indices]
        query = history_embs.mean(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        # Compute similarity to all articles
        similarities = torch.mm(query, self.embeddings_norm.t()).squeeze()
        
        # Exclude history from recommendations
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        # Top-k
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()



@st.cache_data
def load_resources(data_dir=DATA_DIR, raw_dir=RAW_DIR):
    status = {"errors": [], "warnings": []}
    
    # 1. Articles
    articles_path = Path(raw_dir) / "articles.csv"
    articles_df = pd.DataFrame() # Default empty
    if articles_path.exists():
        articles_df = pd.read_csv(articles_path)
        articles_df['title'] = articles_df['title'].fillna("Untitled")
        articles_df['short_description'] = articles_df['short_description'].fillna("")
        articles_df = articles_df.drop_duplicates(subset=['url'])
        url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
        meta_map = articles_df.set_index('url')[['title', 'short_description']].to_dict('index')
    else:
        status["errors"].append(f"Missing articles.csv in {raw_dir}")
        return None, None, None, None, None, None, status


    # 2. Mappings (from converted data) - Used for CF
    u_map_path = Path(data_dir) / "user_map.json"
    a_map_path = Path(data_dir) / "article_map.json"
    
    user_map_cf = {}
    article_map_cf = {}
    
    if u_map_path.exists():
        with open(u_map_path) as f: user_map_cf = json.load(f)
    if a_map_path.exists():
        with open(a_map_path) as f: article_map_cf = json.load(f)

    # 3. Adjacency Matrix & Mappings (from converted data cache) - Priority
    adj_norm = None
    cf_cache_path = Path(data_dir) / "cf_cache.pt"
    if cf_cache_path.exists():
        try:
            cache = torch.load(cf_cache_path, map_location='cpu', weights_only=False)
            adj_norm = cache.get('adj_norm')
            # Cache mappings take priority over JSON files if they exist there
            if 'user_map' in cache: user_map_cf = cache['user_map']
            if 'article_map' in cache: article_map_cf = cache['article_map']
        except: pass

    # 4. Interactions (Raw) -> User History & Fallback Mappings
    replies_path = Path(raw_dir) / "replies.csv"
    user_history = {}
    if replies_path.exists():
        try:
            df_rep = pd.read_csv(replies_path)
            # Clean User IDs
            def clean(x):
                try: return str(int(float(x)))
                except: return str(x)
            col = 'user_id' if 'user_id' in df_rep.columns else 'reply_user_id'
            if col in df_rep.columns:
                df_rep[col] = df_rep[col].apply(clean)
                user_history = df_rep.groupby(col)['article_url'].apply(list).to_dict()
                
                # Fallback for user_map_cf only if STILL missing
                if not user_map_cf:
                    counts = df_rep[col].value_counts()
                    valid_u = counts[counts >= 2].index
                    user_ids = sorted([uid for uid in df_rep[col].unique() if uid in valid_u])
                    user_map_cf = {uid: i for i, uid in enumerate(user_ids)}
                    status["warnings"].append("user_map_missing")
        except Exception as e:
            status["warnings"].append(f"history_load_error: {e}")
    
    # Fallback for article_map_cf only if STILL missing
    if not article_map_cf and not user_history == {}:
        all_interacted = []
        for urls in user_history.values(): all_interacted.extend(urls)
        from collections import Counter
        counts = Counter(all_interacted)
        valid_a = [url for url, c in counts.items() if c >= 2]
        article_map_cf = {url: i for i, url in enumerate(sorted(valid_a))}
        status["warnings"].append("article_map_missing")

    # 5. Technical Pillars (Priors)
    user_priors = None
    a_priors_path = Path(data_dir) / "user_priors.pt"
    if a_priors_path.exists():
        try:
            user_priors = torch.load(a_priors_path, map_location='cpu', weights_only=False)
        except Exception as e:
            status["warnings"].append(f"Failed to load user priors: {e}")
        
    # Semantic IDs are usually generated on the fly or cached. 
    # For now, we'll try to load them if they exist in a model checkpoint or generate dummy if needed.
    
    return articles_df, user_map_cf, article_map_cf, user_history, adj_norm, user_priors, status

@st.cache_resource
def load_cf_model(model_name, n_users, n_items):
    """Generic CF Model Loader"""
    try:
        import torch
        # Lazy imports for model classes
        if model_name.lower() == 'ngcf':
            from src.models.ngcf import NGCF as ModelClass
        elif model_name.lower() == 'lightgcl':
            from src.models.lightgcl import LightGCL as ModelClass
        elif model_name.lower() == 'simgcl':
            from src.models.simgcl import SimGCL as ModelClass
        elif model_name.lower() == 'xsimgcl':
            from src.models.xsimgcl import XSimGCL as ModelClass
        elif model_name.lower() == 'cgrc':
            from src.models.cgrc import CGRC as ModelClass
        elif model_name.lower() == 'igcl':
            from src.models.igcl import IGCL as ModelClass
        elif model_name.lower() == 'bigcf':
            from src.models.bigcf import BIGCF as ModelClass
        else:
            return None

        # Find latest checkpoint
        files = glob.glob(f"{MODELS_DIR}/{model_name.lower()}_*.pt")
        if not files: return None
        latest = max(files, key=os.path.getctime)
        
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        state_dict = checkpoint['model_state_dict']
        
        # Check for size mismatch in checkpoint
        checkpoint_n_users = config.get('n_users', None)
        checkpoint_n_items = config.get('n_items', None)
        
        # Also check from state_dict shapes
        for key in ['E_u_0', 'user_embedding.weight']:
            if key in state_dict:
                checkpoint_n_users = state_dict[key].shape[0]
                break
        for key in ['E_i_0', 'item_embedding.weight']:
            if key in state_dict:
                checkpoint_n_items = state_dict[key].shape[0]
                break
        
        if checkpoint_n_users and checkpoint_n_users != n_users:
            st.warning(f"⚠️ {model_name} checkpoint was trained on {checkpoint_n_users} users, but current data has {n_users}. **Retrain for best results!**")
        if checkpoint_n_items and checkpoint_n_items != n_items:
            st.warning(f"⚠️ {model_name} checkpoint was trained on {checkpoint_n_items} items, but current data has {n_items}. **Retrain for best results!**")
        

        # Init model args
        # Note: Different models have different __init__ args. 
        # Most accept n_users, n_items, embed_dim/emb_dim
        # We need to be careful.
        
        common_args = {
            'n_users': n_users,
            'n_items': n_items,
        }
        
        # Handle dimension name mismatch
        dim = config.get('emb_dim', config.get('embed_dim', 64))
        layers = config.get('layers', config.get('n_layers', [64, 64]))
        if isinstance(layers, int): layers = [dim] * layers # LightGCL uses int n_layers
        
        if model_name.lower() in ['ngcf', 'lightgcl', 'simgcl', 'igcl', 'bigcf']:
             if model_name.lower() == 'ngcf':
                  # NGCF takes n_layers (int), not layers list
                  n_l = len(layers) if isinstance(layers, list) else layers
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=n_l)
             elif model_name.lower() == 'lightgcl':
                  n_l = len(layers) if isinstance(layers, list) else layers
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=n_l)

             elif model_name.lower() == 'simgcl':
                  model = ModelClass(n_users, n_items, embedding_dim=dim)
             elif model_name.lower() == 'igcl':
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=layers[0] if isinstance(layers, list) else layers)
             elif model_name.lower() == 'bigcf':
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=layers[0] if isinstance(layers, list) else layers)
        
        elif model_name.lower() == 'xsimgcl':
            n_l = layers[0] if isinstance(layers, list) else layers
            model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=n_l)
            
            # Re-init layers for Semantic IDs and User Priors if they exist in state_dict
            if 'semantic_layer.weight' in state_dict:
                from scripts.train_cf_models import SemanticEmbeddingLayer
                s_dim = state_dict['semantic_layer.weight'].shape[1]
                model.semantic_layer = SemanticEmbeddingLayer(s_dim, dim)
            if 'user_prior_layer.weight' in state_dict:
                from scripts.train_cf_models import UserPriorLayer
                p_dim = state_dict['user_prior_layer.weight'].shape[1]
                model.user_prior_layer = UserPriorLayer(p_dim, dim)
        
        elif model_name.lower() == 'cgrc':
            model = ModelClass(n_users, n_items, embedding_dim=dim)
             
        try:
             model.load_state_dict(state_dict, strict=False)
        except Exception as e:
             st.warning(f"Loose loading for {model_name}: {e}")
             
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None

@st.cache_resource
def load_cb_model(model_type, articles_df):
    """Load Content-Based Models (TF-IDF, PhoBERT)"""
    # Note: PhoBERT is heavy. Hybrid uses Precomputed Embeddings usually.
    # Here we support TF-IDF mainly for live demo.
    # User requested ALL models.
    try:
        if model_type == 'TF-IDF':
            from src.models.content_based import TFIDFRecommender
            texts = (articles_df['title'] + " " + articles_df['short_description']).tolist()
            model = TFIDFRecommender(1, len(articles_df))
            with st.spinner("Fitting TF-IDF..."):
                model.encode_articles(texts)
            return model
            
        elif model_type == 'PhoBERT':
             # Load pre-computed PhoBERT embeddings
             emb_path = Path(st.session_state.get('data_dir', DATA_DIR)) / "phobert_embeddings.pt"
             if emb_path.exists():
                 try:
                     emb_dict = torch.load(emb_path, map_location='cpu', weights_only=False)
                     
                     # Convert dict (URL -> embedding) to aligned matrix
                     urls = articles_df['url'].tolist()
                     embed_dim = list(emb_dict.values())[0].shape[0]
                     embeddings = torch.zeros(len(urls), embed_dim)
                     
                     matched = 0
                     for i, url in enumerate(urls):
                         if url in emb_dict:
                             embeddings[i] = emb_dict[url]
                             matched += 1
                     
                     st.success(f"Loaded PhoBERT embeddings: {embeddings.shape}, matched {matched}/{len(urls)} articles")
                     return PhoBERTWrapper(embeddings, articles_df)
                 except Exception as e:
                     st.warning(f"Failed to load PhoBERT embeddings: {e}")
                     return None
             else:
                 st.warning(f"PhoBERT embeddings not found at {emb_path}. Run training first.")
                 return None
             
        elif model_type == 'SimCSE':
             # Load pre-computed SimCSE embeddings if they exist
             emb_path = Path(st.session_state.get('data_dir', DATA_DIR)) / "simcse_embeddings.pt"
             if emb_path.exists():
                 try:
                     emb_dict = torch.load(emb_path, map_location='cpu', weights_only=False)
                     
                     # Convert dict to aligned matrix
                     urls = articles_df['url'].tolist()
                     embed_dim = list(emb_dict.values())[0].shape[0] if emb_dict else 768
                     embeddings = torch.zeros(len(urls), embed_dim)
                     
                     matched = 0
                     for i, url in enumerate(urls):
                         if url in emb_dict:
                             embeddings[i] = emb_dict[url]
                             matched += 1
                     
                     st.success(f"Loaded SimCSE embeddings: {embeddings.shape}, matched {matched}/{len(urls)} articles")
                     return PhoBERTWrapper(embeddings, articles_df)
                 except Exception as e:
                     st.warning(f"Failed to load SimCSE embeddings: {e}")
                     return None
             else:
                 st.warning(f"SimCSE embeddings not found. Run training first.")
                 return None
             

             

             
        return None
    except Exception as e:
        st.error(f"CB Load Error: {e}")
        return None

def get_recs(model, model_type, user_idx, history_urls, article_map, articles_df, k=10, 
             adj_norm=None, user_priors=None, semantic_ids=None):
    try:
        import torch
        if model_type in MODEL_OPTIONS["CF"]:
            idx_to_url = {v: k for k, v in article_map.items()}
            model.eval()
            with torch.no_grad():
                # Handling XSimGCL and other champions
                if model_type.lower() == 'xsimgcl':
                    # XSimGCL prediction with fusion
                    # We need to pad user_priors if mismatch (handled in load_resources/main)
                    u_idx_torch = torch.tensor([user_idx])
                    scores = model.predict(adj_norm, users=u_idx_torch, 
                                          semantic_ids=semantic_ids, 
                                          user_priors=user_priors).squeeze()
                elif hasattr(model, 'forward'):
                    # Generic GNN-based
                    # Check for size mismatch before calling forward
                    model_n_users = None
                    if hasattr(model, 'n_users'):
                        model_n_users = model.n_users
                    elif hasattr(model, 'E_u_0'):
                        model_n_users = model.E_u_0.shape[0]
                    elif hasattr(model, 'user_embedding'):
                        model_n_users = model.user_embedding.weight.shape[0]
                    
                    if model_n_users and user_idx >= model_n_users:
                        st.error(f"⚠️ User index {user_idx} >= model capacity ({model_n_users}). Model needs retraining!")
                        return []
                    
                    if adj_norm is not None:
                         # Most champions (LightGCL, SimGCL, IGCL, BIGCF)
                         try:
                             user_all, item_all = model(adj_norm)
                             scores = torch.mm(user_all[user_idx].unsqueeze(0), item_all.t()).squeeze()
                         except Exception as e:
                             st.error(f"Model forward failed: {e}. Try retraining the model.")
                             return []
                    else:
                         # Fallback to embeddings if no graph
                         # Support different naming: user_embedding, E_u_0, etc.
                         if hasattr(model, 'user_embedding'):
                             u_emb = model.user_embedding.weight[user_idx]
                             i_embs = model.item_embedding.weight
                         elif hasattr(model, 'E_u_0'):
                             u_emb = model.E_u_0[user_idx]
                             i_embs = model.E_i_0
                         else:
                             st.warning(f"Model {model_type} has no recognized embedding layer")
                             return []
                         scores = torch.matmul(u_emb, i_embs.t())


                
                vals, indices = torch.topk(scores, k=min(k, len(scores)))
                
            recs = []
            for idx, score in zip(indices.cpu().numpy(), vals.cpu().numpy()):
                url = idx_to_url.get(idx, None)
                if url: recs.append((url, float(score)))
            return recs
            
        elif model_type == 'TF-IDF':
            # CB Logic
            url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
            hist_indices = [url_to_idx[u] for u in history_urls if u in url_to_idx]
            if not hist_indices: return []
            top, scores = model.recommend(hist_indices, k=k)
            recs = []
            for idx, sc in zip(top, scores):
                if idx < len(articles_df):
                     recs.append((articles_df.iloc[idx]['url'], float(sc)))
            return recs
        
        elif model_type in ['PhoBERT', 'SimCSE']:
            # CB using PhoBERTWrapper
            url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
            hist_indices = [url_to_idx[u] for u in history_urls if u in url_to_idx]
            if not hist_indices: 
                st.warning("No history found for this user.")
                return []
            top, scores = model.recommend(hist_indices, k=k)
            recs = []
            for idx, sc in zip(top, scores):
                if idx < len(articles_df):
                     recs.append((articles_df.iloc[idx]['url'], float(sc)))
            return recs
            

            
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return []


def main():

    st.title("📰 Comprehensive RecSys Dashboard")
    
    # Session State for Paths
    if 'data_dir' not in st.session_state: st.session_state['data_dir'] = DATA_DIR
    if 'raw_dir' not in st.session_state: st.session_state['raw_dir'] = RAW_DIR
    
    # -------------------------------------------------------------------------
    # DATASET MANAGER (SIDEBAR)
    # -------------------------------------------------------------------------
    st.sidebar.header("🗂️ Dataset Context")
    
    # 1. Load Articles Metadata (Global) to find Categories
    # We always use the global RAW articles to find categories
    global_articles_path = Path(RAW_DIR) / "articles.csv"
    categories = ["Merged (All)"]
    if global_articles_path.exists():
        try:
            df_g = pd.read_csv(global_articles_path)
            # Check for source_category or category column
            cat_col = 'source_category' if 'source_category' in df_g.columns else 'category'
            if cat_col in df_g.columns:
                cats = sorted(df_g[cat_col].dropna().unique().tolist())
                categories.extend(cats)

        except: pass
        
    selected_ds = st.sidebar.selectbox("Active Dataset", categories)
    
    # Path Logic
    target_raw_dir = RAW_DIR # Default
    target_data_dir = DATA_DIR # Default
    
    if selected_ds != "Merged (All)":
        # Create dedicated folders
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', selected_ds).lower()
        target_raw_dir = f"data/subsets/{safe_name}/raw"
        target_data_dir = f"data/subsets/{safe_name}/processed"
        
    st.sidebar.caption(f"Context: {selected_ds}")
    
    # Check if Ready
    is_ready = (Path(target_data_dir) / "user_map.json").exists()
    
    if not is_ready and selected_ds != "Merged (All)":
        st.sidebar.warning("Not Prepared!")
        if st.sidebar.button("⚙️ Generate & Convert"):
            with st.spinner("Filtering & Converting..."):
                try:
                    # 1. Prepare Directory
                    os.makedirs(target_raw_dir, exist_ok=True)
                    os.makedirs(target_data_dir, exist_ok=True)
                    
                    # 2. Filter CSVs
                    df_a = pd.read_csv(Path(RAW_DIR)/"articles.csv")
                    df_r = pd.read_csv(Path(RAW_DIR)/"replies.csv")
                    
                    # Filter Articles
                    cat_col = 'source_category' if 'source_category' in df_a.columns else 'category'
                    df_a_sub = df_a[df_a[cat_col] == selected_ds]
                    valid_urls = set(df_a_sub['url'])

                    
                    # Filter Replies (Users who commented on these articles)
                    # Note: Original 'replies.csv' has 'article_url'.
                    df_r_sub = df_r[df_r['article_url'].isin(valid_urls)]
                    
                    # Also need User Profiles? 
                    # convert_to_gnn uses 'user_profiles.csv' mainly for features?
                    # Let's copy it or filter. (If users are subset).
                    # For simplicity, copy full user_profiles or filter by appearing in replies.
                    df_u = pd.read_csv(Path(RAW_DIR)/"user_profiles.csv")
                    valid_users = set(df_r_sub['user_id'].astype(str)) if 'user_id' in df_r_sub.columns else set()
                    if not valid_users: valid_users = set(df_r_sub['reply_user_id'].astype(str))
                    
                    # Filter user profiles 
                    # df_u['user_id'] is typically float/intStr. 
                    # Let's just save full user profiles to avoid mismatch, or simple filter.
                    # Copy user_profiles.csv is safer if small, or filter.
                    df_u.to_csv(Path(target_raw_dir)/"user_profiles.csv", index=False)
                    
                    # Save filtered
                    df_a_sub.to_csv(Path(target_raw_dir)/"articles.csv", index=False)
                    df_r_sub.to_csv(Path(target_raw_dir)/"replies.csv", index=False)
                    
                    # 3. Call Conversion Script
                    # Must use FULL PATHS for subprocess
                    script = "src/data/convert_to_gnn.py"
                    cmd = [
                        sys.executable, script,
                        "--articles-path", f"{target_raw_dir}/articles.csv",
                        "--replies-path", f"{target_raw_dir}/replies.csv",
                        "--users-path", f"{target_raw_dir}/user_profiles.csv",
                        "--output", target_data_dir,
                        "--graph-type", "user-article", # Default minimal
                        "--add-text-features" # Optional
                    ]
                    # Also run "KNN Enrichment" if user wants?
                    # For now, standard conversion.
                    
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0:
                        st.sidebar.success("Done!")
                        st.cache_data.clear() # Clear resource cache
                        # Reload logic happens below
                        is_ready = True
                    else:
                        st.sidebar.error("Conversion Failed!")
                        st.sidebar.code(res.stderr)
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

    # Update Session if changed (and ready/merged)
    # If merged, always update. If subset, update if ready.
    if selected_ds == "Merged (All)" or is_ready:
        st.session_state['data_dir'] = target_data_dir
        st.session_state['raw_dir'] = target_raw_dir
    else:
        # Keep old or show error
        pass

    # Load Data (Using Session Ptrs)
    res = load_resources(
        st.session_state['data_dir'], 
        st.session_state['raw_dir']
    )
    if res[0] is None:
        if res[-1]["errors"]:
            st.sidebar.error(res[-1]["errors"][0])
        else:
            st.sidebar.error("Data Load Failed. Check paths.")
        return
        
    articles_df, user_map_cf, article_map_cf, user_history, adj_norm, user_priors, data_status = res
    
    # Handle Warnings from load_resources
    if "user_map_missing" in data_status["warnings"]:
        st.warning("⚠️ User mappings missing. Built temporary map from raw interactions. Indices might not match trained models!")
        if st.button("🔧 Re-sync User Mappings", key="sync_u"):
            with st.spinner("Syncing..."):
                cmd = [sys.executable, "scripts/train_cf_models.py", "--epochs", "0", "--data-path", st.session_state['data_dir']]
                subprocess.run(cmd)
                st.rerun()
    if "article_map_missing" in data_status["warnings"]:
        st.warning("⚠️ Article mappings missing. Built temporary map from articles.csv.")
        if st.button("🔧 Re-sync Article Mappings", key="sync_a"):
             with st.spinner("Syncing..."):
                cmd = [sys.executable, "src/data/convert_to_gnn.py", "--graph-type", "user-article", "--output", st.session_state['data_dir']]
                subprocess.run(cmd)
                st.rerun()
    
    # Optional: Generate Semantic IDs (Pillar 1)
    # For demo, we use a fixed bits/stages if not cached
    semantic_ids = None
    if 'xsimgcl' in [m.lower() for m in MODEL_OPTIONS["CF"]]:
        # Try to load pretrained embeddings for Semantic ID generation

        # (This might be slow if not cached, but small articles.csv makes it ok)
        try:
             # Look for article_embeddings.pt
             emb_path = Path(st.session_state['data_dir']) / "article_embeddings.pt"
             if emb_path.exists():
                 pretrained = torch.load(emb_path, map_location='cpu', weights_only=False)
                 semantic_ids = generate_semantic_ids(pretrained, bits=3)
        except: pass
    
    # Sidebar Controls
    st.sidebar.divider()
    st.sidebar.header("🕹️ Controls")
    
    # Min interactions filter
    min_interactions = st.sidebar.slider("Min Interactions", 1, 50, 3, help="Filter users by minimum reading history")
    
    # User - filter by min interactions
    user_ids = sorted([uid for uid, hist in user_history.items() if len(hist) >= min_interactions])
    st.sidebar.caption(f"📌 {len(user_ids)} users with ≥{min_interactions} interactions")
    
    user_options = ["📊 All Users (Overview)"] + user_ids
    default_u = 0
    selected_user_option = st.sidebar.selectbox("Select User", user_options, index=default_u)
    
    # Determine if we're in "All Users" mode
    all_users_mode = selected_user_option == "📊 All Users (Overview)"
    selected_user = user_ids[0] if (all_users_mode and user_ids) else selected_user_option
    


    # Persistent Tabs (Radio styled as tabs)
    nav = st.radio("Navigation", ["🤝 CF Inference", "📝 CB Inference", "🛠️ Training", "📊 Stats", "🕸️ Graph"], 
                   key="main_nav", horizontal=True, label_visibility="collapsed")
    st.divider()

    
    # --- PAGE 1: CF INFERENCE ---
    if nav == "🤝 CF Inference":
        st.header("🤝 Collaborative Filtering")
        st.caption("Graph-based: Learn user preferences from interaction patterns")
        
        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            st.subheader("CF Config")
            cf_model_name = st.selectbox("CF Model", MODEL_OPTIONS["CF"], key="cf_model")
            k_cf = st.slider("Top K", 5, 20, 10, key="k_cf")
            
            # User History
            st.divider()
            if all_users_mode:
                st.write("**All Users Mode**")
                n_sample_cf = st.slider("Sample Size", 5, 50, 10, key="n_cf")
            else:
                st.write(f"**History ({len(user_history.get(selected_user, []))})**")
                url_to_meta = articles_df.set_index('url')[['title', 'short_description']].to_dict('index')
                for u in user_history.get(selected_user, [])[-5:]:
                    st.caption(f"- {str(url_to_meta.get(u, {}).get('title', 'Unknown'))[:40]}")
        
        with col_R:
            st.subheader(f"CF Results: {cf_model_name}")
            
            if all_users_mode:
                # Batch mode
                model = load_cf_model(cf_model_name, len(user_map_cf), len(article_map_cf))
                if model:
                    import random
                    sample_users = random.sample(list(user_map_cf.keys()), min(n_sample_cf, len(user_map_cf)))
                    
                    results = []
                    url_to_meta = articles_df.set_index('url')[['title', 'short_description']].to_dict('index')
                    
                    with st.spinner(f"Generating CF recommendations for {len(sample_users)} users..."):
                        for user_id in sample_users:
                            user_idx = user_map_cf[user_id]
                            user_recs = get_recs(model, cf_model_name, user_idx, [], article_map_cf, articles_df, k_cf,
                                            adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids)
                            
                            for rank, (url, score) in enumerate(user_recs[:k_cf], 1):
                                title = str(url_to_meta.get(url, {}).get('title', 'Unknown'))[:35]
                                results.append({
                                    "User": str(user_id)[:12],
                                    "Rank": rank,
                                    "Article": title,
                                    "Score": f"{score:.3f}"
                                })
                    
                    st.dataframe(pd.DataFrame(results), use_container_width=True, height=400)
                    st.success(f"Generated {len(results)} recommendations.")
                else:
                    st.warning(f"Model {cf_model_name} not found. Train it first.")
            else:
                # Single user mode
                if selected_user in user_map_cf:
                    model = load_cf_model(cf_model_name, len(user_map_cf), len(article_map_cf))
                    if model:
                        recs = get_recs(model, cf_model_name, user_map_cf[selected_user], [], article_map_cf, articles_df, k_cf,
                                      adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids)
                        
                        url_to_meta = articles_df.set_index('url')[['title', 'short_description', 'source_category']].to_dict('index')
                        
                        for i, (url, score) in enumerate(recs):
                            meta = url_to_meta.get(url, {'title': 'Unknown', 'short_description': '', 'source_category': 'N/A'})
                            title = str(meta.get('title', 'Unknown'))[:70]
                            desc = str(meta.get('short_description', ''))[:150]
                            cat_raw = meta.get('source_category', 'N/A')
                            cat = CATEGORY_MAP.get(cat_raw, f"📁 {cat_raw}")
                            
                            # Score color: green for high, yellow for medium
                            score_color = score_to_color(score, base_hue=120)  # Green gradient
                            

                            st.markdown(f"""
                            <div style="padding:15px; margin:10px 0; border-radius:12px; background:linear-gradient(135deg, #ffffff, #f9f9f9); border: 1px solid #eee; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left:5px solid {score_color};">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                    <span style="color:#888; font-weight:bold; font-size:0.9em;">RANK #{i+1}</span>
                                    <span style="background:{score_color}; color:white; padding:4px 10px; border-radius:20px; font-size:0.85em; font-weight:bold;">{score:.3f}</span>
                                </div>
                                <div style="font-weight:bold; font-size:1.15em; color:#222; margin-bottom:6px; line-height:1.4;">
                                    <a href="{url}" target="_blank" style="text-decoration:none; color:#2c3e50;">{title}</a>
                                </div>
                                <div style="font-size:0.95em; color:#555; margin-bottom:10px; line-height:1.5;">{desc}...</div>
                                <div style="display:flex; align-items:center; justify-content:space-between;">
                                    <span style="background:#eef2ff; color:#4f46e5; padding:3px 10px; border-radius:6px; font-size:0.8em; font-weight:500;">{cat}</span>
                                    <a href="{url}" target="_blank" style="font-size:0.85em; color:#4f46e5; text-decoration:none;">Read More →</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.warning(f"Model {cf_model_name} not found. Please train it.")
                else:
                    st.warning(f"❄️ Cold Start: User not in CF training set. Try CB tab instead.")
    
    # --- PAGE 2: CB INFERENCE ---
    if nav == "📝 CB Inference":
        st.header("📝 Content-Based Filtering")
        st.caption("Text-based: Recommend articles similar to user's reading history")
        
        col_L2, col_R2 = st.columns([1, 2])
        
        with col_L2:
            st.subheader("CB Config")
            cb_model_name = st.selectbox("CB Model", MODEL_OPTIONS["CB"], key="cb_model")
            k_cb = st.slider("Top K", 5, 20, 10, key="k_cb")
            
            # User History for CB
            st.divider()
            history_urls = user_history.get(selected_user, [])
            st.write(f"**📚 Reading History ({len(history_urls)})** - Used as query")
            
            if history_urls:
                url_to_meta = articles_df.set_index('url')[['title', 'short_description', 'source_category']].to_dict('index')
                
                # Show history as styled cards
                for i, u in enumerate(history_urls[-5:], 1):
                    meta = url_to_meta.get(u, {})
                    title = str(meta.get('title', 'Unknown'))[:60]
                    cat_raw = meta.get('source_category', 'N/A')
                    cat = CATEGORY_MAP.get(cat_raw, f"📁 {cat_raw}")

                    with st.container():
                        st.markdown(f"""
                        <div style="padding:8px; margin:4px 0; border-radius:8px; background:rgba(100,100,100,0.1);">
                            <span style="color:#888;">#{i}</span> <b>{title}</b>
                            <br><small style="color:#666;">{cat}</small>

                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No reading history for this user")

        
        with col_R2:
            st.subheader(f"CB Results: {cb_model_name}")
            
            if cb_model_name == 'TF-IDF':
                model = load_cb_model('TF-IDF', articles_df)
                recs = get_recs(model, 'TF-IDF', 0, user_history.get(selected_user, []), {}, articles_df, k_cb)
            elif cb_model_name == 'Hybrid':
                st.info("Hybrid = Interleaving CF (NGCF) + CB (TF-IDF)")
                cf_model = load_cf_model('NGCF', len(user_map_cf), len(article_map_cf))
                cb_model = load_cb_model('TF-IDF', articles_df)
                
                cf_recs = []
                cb_recs = []
                
                if cf_model and selected_user in user_map_cf:
                    cf_recs = get_recs(cf_model, 'NGCF', user_map_cf[selected_user], [], article_map_cf, articles_df, k_cb)
                if cb_model:
                    cb_recs = get_recs(cb_model, 'TF-IDF', 0, user_history.get(selected_user, []), {}, articles_df, k_cb)
                
                # Interleave
                recs = []
                seen = set()
                for i in range(max(len(cf_recs), len(cb_recs))):
                    if i < len(cf_recs):
                        u, s = cf_recs[i]
                        if u not in seen: recs.append((u, s)); seen.add(u)
                    if i < len(cb_recs):
                        u, s = cb_recs[i]
                        if u not in seen: recs.append((u, s)); seen.add(u)
                    if len(recs) >= k_cb: break
            else:
                # PhoBERT, SimCSE
                model = load_cb_model(cb_model_name, articles_df)
                if model:
                    recs = get_recs(model, cb_model_name, 0, user_history.get(selected_user, []), {}, articles_df, k_cb)
                else:
                    recs = []
                    st.warning(f"Model {cb_model_name} not found.")
            
            # Render CB results
            url_to_meta = articles_df.set_index('url')[['title', 'short_description', 'source_category']].to_dict('index')
            for i, (url, score) in enumerate(recs):
                meta = url_to_meta.get(url, {'title': 'Unknown', 'short_description': '', 'source_category': 'N/A'})
                title = str(meta.get('title', 'Unknown'))[:70]
                desc = str(meta.get('short_description', ''))[:150]
                cat_raw = meta.get('source_category', 'N/A')
                cat = CATEGORY_MAP.get(cat_raw, f"📁 {cat_raw}")
                
                score_color = score_to_color(score, base_hue=210)  # Blue gradient

                
                st.markdown(f"""
                <div style="padding:12px; margin:8px 0; border-radius:10px; background:linear-gradient(135deg, rgba(33,150,243,0.1), rgba(33,150,243,0.02)); border-left:4px solid {score_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#888; font-weight:bold;">#{i+1}</span>
                        <span style="background:{score_color}; color:white; padding:2px 8px; border-radius:12px; font-size:0.8em;">{score:.3f}</span>
                    </div>
                    <div style="font-weight:bold; margin:4px 0;">{title}</div>
                    <div style="color:#666; font-size:0.9em;">{desc}...</div>
                    <div style="margin-top:6px;">
                        <span style="font-size:0.85em;">{cat}</span>
                        <a href="{url}" target="_blank" style="float:right; text-decoration:none;">🔗 Read</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)



    # --- PAGE 3: TRAINING ---
    if nav == "🛠️ Training":
        st.header("Train New Model")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            train_cat = st.radio("Train Type", ["CF Models", "Content Models"])
        with c2:
            if train_cat == "CF Models":
                train_model_name = st.selectbox("Target Model", MODEL_OPTIONS["CF"])
            else:
                train_model_name = st.selectbox("Target Model", MODEL_OPTIONS["CB"])
        with c3:
            epochs = st.number_input("Epochs", 1, 100, 10)
            use_all_data = st.checkbox("Train on Full Data (Demo)", value=False)
            denoise_ratio = st.slider("Adaptive Denoising Ratio", 0.0, 0.5, 0.0, 0.05, 
                                      help="Prune noisy interactions during early training (Phase 6)")
            eval_protocol = st.selectbox("Evaluation Protocol", ["full", "loo100", "cold"],
                                         help="full=all items (hard), loo100=100 neg (paper-style), cold=cold-start users")
            
            # Graph Selection (only for CF relevant models)
            selected_graph_path = None
            if train_cat == "CF Models":
                st.markdown("---")
                st.subheader("Graph Architecture")
                
                GRAPH_TYPES = {
                    "Standard Bipartite": "data/processed/cf_cache.pt",
                    "Reaction-Weighted": "data/processed_graphs/reaction_weighted_graph.pt",
                    "User-Category Hetero": "data/processed_graphs/user_category_graph.pt",
                    "Full Hetero (Multi-relational)": "data/processed/full_hetero_graph.pt"
                }
                
                graph_mode = st.radio("Selection Mode", ["Conceptual Types", "Custom File"], horizontal=True)
                
                if graph_mode == "Conceptual Types":
                    selected_type = st.selectbox("Select Graph Type", list(GRAPH_TYPES.keys()))
                    selected_graph_path = GRAPH_TYPES[selected_type]
                    
                    if not os.path.exists(selected_graph_path):
                        st.warning(f"⚠️ {selected_type} graph file not found at {selected_graph_path}")
                        
                        col_build1, col_build2 = st.columns([2, 1])
                        with col_build1:
                            st.info("You can build this graph using the generation scripts.")
                        with col_build2:
                            if st.button("🔨 Build Graph"):
                                with st.spinner(f"Building {selected_type}..."):
                                    if selected_type == "Full Hetero (Multi-relational)":
                                        script = "src/data/convert_to_gnn.py"
                                        args = ["--graph-type", "hetero"]
                                    elif selected_type == "Standard Bipartite":
                                        script = "src/data/convert_to_gnn.py"
                                        args = ["--graph-type", "user-article"]
                                    else:
                                        script = "scripts/build_alternative_graphs.py"
                                        type_map = {
                                            "Reaction-Weighted": "reaction-weighted",
                                            "User-Category Hetero": "user-category"
                                        }
                                        args = ["--graph-type", type_map.get(selected_type)]
                                    
                                    if script:
                                        cmd = [sys.executable, script] + args
                                        result = subprocess.run(cmd, capture_output=True, text=True)
                                        if result.returncode == 0:
                                            st.success("Graph built successfully!")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to build graph: {result.stderr}")
                else:
                    graph_files = glob.glob("data/**/*.pt", recursive=True)
                    graph_options = {os.path.basename(f): f for f in graph_files}
                    if graph_options:
                        default_graph = "cf_cache.pt" if "cf_cache.pt" in graph_options else list(graph_options.keys())[0]
                        selected_graph_name = st.selectbox("Select .pt File", list(graph_options.keys()), 
                                                        index=list(graph_options.keys()).index(default_graph))
                        selected_graph_path = graph_options.get(selected_graph_name)
                    else:
                        st.warning("No .pt files found.")
                        selected_graph_path = "data/processed/cf_cache.pt"




            
        start = st.button("🚀 Start Training Process", type="primary")
        
        if start:
            st.info(f"Launching training for {train_model_name}...")
            log_area = st.empty()
            
            # Build Command
            if train_model_name in MODEL_OPTIONS["CF"]:
                script = "scripts/train_cf_models.py"
                cmd = [sys.executable, script, "--model", train_model_name.lower(), "--epochs", str(epochs),
                       "--eval-protocol", eval_protocol, "--data-path", selected_graph_path]

                if denoise_ratio > 0:
                    cmd.extend(["--denoise-ratio", str(denoise_ratio)])

            else:
                # Content-Based models
                if train_model_name == 'TF-IDF':
                    st.success("TF-IDF does not require pre-training! It fits instantly when used for inference.")
                    st.info("Simply select TF-IDF in the Inference tab to use it.")
                    st.stop()
                elif train_model_name == 'Hybrid':
                    st.info("Hybrid model training...")
                    script = "scripts/run_content_based.py"
                    cmd = [sys.executable, script, "--model", "hybrid", "--epochs", str(epochs)]
                else:
                    # PhoBERT, SimCSE
                    script = "scripts/run_content_based.py"
                    model_arg = train_model_name.lower().replace("-","")
                    cmd = [sys.executable, script, "--model", model_arg, "--epochs", str(epochs)]
            
            # Exec
            try:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                    text=True, bufsize=1, universal_newlines=True
                )
                logs = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None: break
                    if line:
                        logs.append(line.strip())
                        log_area.code("\n".join(logs[-15:]))
                        
                if process.returncode == 0:
                    st.success("Internal Training Finished.")
                    st.cache_resource.clear()
                else:
                    st.error("Failed.")
            except Exception as e:
                st.error(str(e))

    # --- PAGE 4: STATS ---
    if nav == "📊 Stats":
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Users", len(user_history))
        col_m2.metric("Articles", len(articles_df))
        if col_m3.button("🗑️ Delete All Models", help="Clear models/ directory"):
             if os.path.exists(MODELS_DIR):
                 shutil.rmtree(MODELS_DIR)
                 os.makedirs(MODELS_DIR)
                 st.success("All models deleted!")
                 st.rerun()


        # Add Comparison results
        st.divider()
        files = glob.glob(f"{MODELS_DIR}/comparison_results_*.json")
        if files:
            latest = max(files, key=os.path.getctime)
            with open(latest) as f: st.dataframe(json.load(f))
        
        # Batch Recommendations
        st.divider()
        st.subheader("📋 Batch Recommendations (Sample Users)")
        st.write("Generate recommendations for multiple users at once.")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_model = st.selectbox("Model for Batch", MODEL_OPTIONS["CF"], key="batch_model")
            n_sample_users = st.slider("Sample Users", 5, 50, 10)
        with col2:
            batch_k = st.slider("Top-K per User", 3, 20, 5)
        
        if st.button("🚀 Generate Batch Recommendations"):
            if batch_model and len(user_map_cf) > 0:
                model = load_cf_model(batch_model, len(user_map_cf), len(article_map_cf))
                if model:
                    with st.spinner("Generating recommendations..."):
                        sample_users = random.sample(list(user_map_cf.keys()), min(n_sample_users, len(user_map_cf)))

                        
                        results = []
                        idx_to_url = {v: k for k, v in article_map_cf.items()}
                        url_to_title = dict(zip(articles_df['url'], articles_df['title']))
                        
                        for user_id in sample_users:
                            user_idx = user_map_cf[user_id]
                            recs = get_recs(model, batch_model, user_idx, [], article_map_cf, articles_df, batch_k,
                                           adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids)
                            
                            for rank, (url, score) in enumerate(recs[:batch_k], 1):
                                title = url_to_title.get(url, "Unknown")[:40]
                                results.append({
                                    "User": user_id[:10] + "..." if len(str(user_id)) > 10 else user_id,
                                    "Rank": rank,
                                    "Article": title,
                                    "Score": f"{score:.3f}"
                                })
                        
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True)
                        st.success(f"Generated {len(results)} recommendations for {len(sample_users)} users.")
                else:
                    st.warning(f"Model {batch_model} not found. Train it first.")
            else:
                st.warning("No users or model available.")
            

    # --- PAGE 5: GRAPH ---
    if nav == "🕸️ Graph":
        st.header("🕸️ Graph Inspector")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📂 Available Graphs")
            # list files in data_dir
            curr_dir = Path(st.session_state.get('data_dir', DATA_DIR))
            graph_files = list(curr_dir.glob("*.pt"))
            graph_names = [gf.name for gf in graph_files]
            
            if graph_files:
                selected_graph = st.selectbox("Select Graph for EDA", graph_names)
                selected_path = curr_dir / selected_graph
                st.caption(f"Size: {selected_path.stat().st_size / 1024:.1f} KB")
                
                if st.button("📊 Load & Analyze Graph"):
                    with st.spinner("Loading graph..."):
                        try:
                            data = torch.load(selected_path, weights_only=False)
                            st.session_state['eda_graph'] = data
                            st.session_state['eda_graph_name'] = selected_graph
                        except Exception as e:
                            st.error(f"Failed to load: {e}")
                
                # Display EDA if loaded
                if 'eda_graph' in st.session_state and st.session_state.get('eda_graph_name') == selected_graph:
                    data = st.session_state['eda_graph']
                    st.success(f"Loaded: {selected_graph}")
                    
                    # Detect graph type
                    is_hetero = hasattr(data, 'node_types') or hasattr(data, 'edge_types')
                    
                    if is_hetero:
                        st.markdown("**Type:** Heterogeneous Graph")
                        
                        # Node types
                        st.markdown("**Node Types:**")
                        for nt in data.node_types:
                            n_nodes = data[nt].x.shape[0] if hasattr(data[nt], 'x') else data[nt].num_nodes
                            feat_dim = data[nt].x.shape[1] if hasattr(data[nt], 'x') and data[nt].x is not None else 0
                            st.write(f"  - `{nt}`: {n_nodes} nodes, {feat_dim}D features")
                        
                        # Edge types
                        st.markdown("**Edge Types:**")
                        for et in data.edge_types:
                            edge_index = data[et].edge_index
                            n_edges = edge_index.shape[1]
                            st.write(f"  - `{et}`: {n_edges} edges")
                            
                    else:
                        st.markdown("**Type:** Homogeneous Graph")
                        n_nodes = data.x.shape[0] if hasattr(data, 'x') and data.x is not None else data.num_nodes
                        n_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 0
                        feat_dim = data.x.shape[1] if hasattr(data, 'x') and data.x is not None else 0
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Nodes", n_nodes)
                        col_b.metric("Edges", n_edges)
                        col_c.metric("Features", feat_dim)
                        
                        # Density
                        if n_nodes > 0:
                            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
                            st.write(f"**Density:** {density:.6f}")
                        
                        # Sample edges
                        if hasattr(data, 'edge_index') and data.edge_index.shape[1] > 0:
                            st.markdown("**Sample Edges (first 10):**")
                            ei = data.edge_index[:, :10].T.tolist()
                            st.dataframe(ei, column_config={"0": "Source", "1": "Target"})
                            
                            # Draw sample graph
                            if st.button("🎨 Draw Sample Subgraph (50 nodes)"):
                                # Take first 50 nodes and their edges

                                sample_nodes = set(range(min(50, n_nodes)))
                                edges = data.edge_index.T.tolist()
                                sample_edges = [(s, t) for s, t in edges if s in sample_nodes and t in sample_nodes]
                                
                                G = nx.Graph()
                                G.add_nodes_from(sample_nodes)
                                G.add_edges_from(sample_edges[:200])  # Limit edges
                                
                                fig, ax = plt.subplots(figsize=(10, 8))
                                pos = nx.spring_layout(G, seed=42)
                                nx.draw(G, pos, ax=ax, node_size=100, node_color='steelblue', 
                                       edge_color='gray', alpha=0.7, with_labels=False)
                                ax.set_title(f"Sample Subgraph ({len(G.nodes)} nodes, {len(G.edges)} edges)")
                                st.pyplot(fig)
                                plt.close()
            else:
                st.warning("No .pt graph files found.")

                
        with c2:
            st.subheader("🛠️ Generate Graph")
            st.write("Select a graph type and generate from raw data:")
            
            graph_type_options = {
                "User-Article Bipartite": "user-article",
                "User-User (Shared Interest)": "user-user",
                "Article-Article (Co-occurrence)": "article-article",
                "Full Heterogeneous": "hetero",
                "All Graph Types": "all"
            }
            selected_graph_type = st.selectbox("Graph Type", list(graph_type_options.keys()))
            add_text_features = st.checkbox("Add Text Features (TF-IDF)", value=True)
            
            if st.button("🔧 Generate Graph", type="primary"):
                graph_arg = graph_type_options[selected_graph_type]
                raw_dir = st.session_state.get('raw_dir', RAW_DIR)
                data_dir = st.session_state.get('data_dir', DATA_DIR)
                
                cmd = [
                    sys.executable, "src/data/convert_to_gnn.py",
                    "--graph-type", graph_arg,
                    "--articles", f"{raw_dir}/articles.csv",
                    "--replies", f"{raw_dir}/replies.csv",
                    "--output", data_dir
                ]
                if add_text_features:
                    cmd.append("--add-text-features")
                
                st.info(f"Running: `{' '.join(cmd[-6:])}`...")
                log_area = st.empty()
                
                try:
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, universal_newlines=True
                    )
                    logs = []
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            logs.append(line.strip())
                            log_area.code("\n".join(logs[-20:]))
                    
                    if process.returncode == 0:
                        st.success(f"Graph '{selected_graph_type}' generated successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Graph generation failed.")
                except Exception as e:
                    st.error(f"Error: {e}")


        st.divider()
        st.subheader("Visualize User Ego Graph")
        
        # Load Graph
        graph_path = curr_dir / "user_article_graph.pt"
        if not graph_path.exists():
             # Try fallback
             graph_path = curr_dir / "full_hetero_graph.pt"
             
        if not graph_path.exists():
            st.warning(f"Graph data not found in {curr_dir}. Run pipeline first.")
        else:
            if 'graph_data' not in st.session_state:
                if st.button(f"Load {graph_path.name}"):
                    try:
                         with st.spinner("Loading PyG Graph..."):
                            data = torch.load(graph_path, weights_only=False)
                            st.session_state['graph_data'] = data
                            st.success(f"Loaded! Nodes: {getattr(data, 'num_nodes', 'N/A')}")
                    except Exception as e:
                         st.error(str(e))
                         
            # Rest of Visualization Logic (Existing)
            if 'graph_data' in st.session_state:
                data = st.session_state['graph_data']
                
                # Check Maps (Using Session Path)
                u_map_path = curr_dir / "user_map.json"
                a_map_path = curr_dir / "article_map.json"
                
                if not (u_map_path.exists() and a_map_path.exists()):
                    st.error("Missing maps for visualization.")
                else:
                    target_user_id = st.text_input("User ID (from Stats)", value="162")
                    draw = st.button("Draw Graph")
                    
                    if draw:
                          with open(u_map_path) as f: u_map = json.load(f)
                          with open(a_map_path) as f: a_map = json.load(f)
                          inv_a_map = {int(v): k for k,v in a_map.items()} # idx -> url (int cast important)
                          
                          # Load Title Map from CSV
                          articles_df = pd.read_csv("data/raw/articles.csv")
                          url_to_title = dict(zip(articles_df['url'], articles_df['title']))

                          if target_user_id not in u_map:
                               # Try checking if it's an index
                               if target_user_id.isdigit():
                                   val = int(target_user_id)
                                   idx_to_user = {v: k for k,v in u_map.items()}
                                   if val in idx_to_user:
                                       u_idx = val
                                       target_user_id = idx_to_user[val]
                                       st.success(f"Interpreted '{val}' as Index. User ID: {target_user_id}")
                                   else:
                                        st.error(f"User ID '{target_user_id}' not found in mapping (and not a valid index).")
                                        st.stop()
                               else:
                                   st.error(f"User ID '{target_user_id}' not found in mapping.")
                                   st.stop()
                          else:
                               u_idx = u_map[target_user_id]
                               st.write(f"Visualizing User Index: {u_idx}")


                          # Build NX Graph (always executed after u_idx is set)
                          G = nx.Graph()


                          G.add_node("User", color='red', label=f'User {target_user_id}', size=20)
                          
                          # 1. User -> Article Edges
                          try:
                              if ('user', 'comments', 'article') in data.edge_index_dict:
                                  edge_index = data['user', 'comments', 'article'].edge_index
                              else:
                                  st.warning("Direct User-Article edges not found. Checking keys...")
                                  st.write(data.edge_types)
                                  edge_index = None

                              if edge_index is not None:
                                  mask = edge_index[0] == u_idx
                                  connected_articles = [int(x) for x in edge_index[1][mask].tolist()]
                                  
                                  st.write(f"Read {len(connected_articles)} articles.")
                                  
                                  added_articles = set()
                                  for aid in connected_articles:
                                       url = inv_a_map.get(aid, "Unknown")
                                       title = str(url_to_title.get(url, f"Article {aid}"))[:30]
                                       G.add_node(f"A_{aid}", color='blue', label=title, size=10)
                                       G.add_edge("User", f"A_{aid}")
                                       added_articles.add(aid)

                                       
                                  # 2. Article -> Similar Article (KNN)
                                  if ('article', 'similar_to', 'article') in data.edge_index_dict:
                                       sim_index = data['article', 'similar_to', 'article'].edge_index
                                       src_sim = sim_index[0]
                                       mask_sim = torch.isin(src_sim, torch.tensor(list(added_articles)))
                                       
                                       s_nodes = [int(x) for x in src_sim[mask_sim].tolist()]
                                       d_nodes = [int(x) for x in sim_index[1][mask_sim].tolist()]
                                       
                                       count_sim = 0
                                       for s, d in zip(s_nodes, d_nodes):
                                           if count_sim > 50: break
                                           d_node = f"A_{d}"
                                           s_node = f"A_{s}"
                                           if d_node not in G.nodes:
                                                url = inv_a_map.get(d, "Unknown")
                                                title = str(url_to_title.get(url, f"Sim {d}"))[:30]
                                                G.add_node(d_node, color='green', label=title, size=8)
                                           if s_node in G.nodes:
                                               G.add_edge(s_node, d_node)
                                           count_sim += 1
                                       st.write(f"Added {count_sim} similarity edges.")

                              
                              # Plot
                              fig, ax = plt.subplots(figsize=(10, 8))
                              pos = nx.spring_layout(G, k=0.3)
                              
                              # Safe access for node attributes
                              colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
                              sizes = [G.nodes[n].get('size', 10)*30 for n in G.nodes()]
                              labels = {n: G.nodes[n].get('label', str(n)) for n in G.nodes()}

                              
                              nx.draw(G, pos, ax=ax, node_color=colors, node_size=sizes, with_labels=False)
                              nx.draw_networkx_labels(G, pos, labels, font_size=8)
                              
                              st.pyplot(fig)
                              st.info("Red: User | Blue: Read History | Green: Similar Articles (KNN)")
                              
                          except Exception as e:
                              import traceback
                              st.error(f"Error drawing graph: {e}")
                              st.code(traceback.format_exc())



if __name__ == "__main__":
    main()
