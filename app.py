
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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
from src.inference.re_ranker import CalibratedReRanker
import random
import shutil
from src.models.semantic_id import generate_semantic_ids


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="News RecSys Comprehensive",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def discover_trained_models():
    """Scan models/ and checkpoints/ to find which models have available weights."""
    discovered = {
        "CF": [],
        "CB": ["tf-idf", "lsa", "naivebayes"], # Base procedural models always available
        "graph_variants": ["strict_g1", "strict_g2", "strict_g3"] # Default variants
    }
    
    # 1. Discover CF Models (GNNs)
    # Search root models/ and any subdirs (like models/models/ from unzipped results)
    search_dirs = [MODELS_DIR, os.path.join(MODELS_DIR, "models")]
    cf_catalog = ["MA-HCL", "SimGCL", "XSimGCL", "LightGCL", "Sim-MAHGN"]
    
    found_cf = set()
    found_variants = set()
    
    for d in search_dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if not f.endswith(".pt"): continue
            fname = f.lower()
            for model in cf_catalog:
                if model.lower() in fname.replace('_', '-'): # handle naming variations
                    found_cf.add(model)
                    # Extract variant if possible
                    for v in ["strict_g1", "strict_g2", "strict_g3"]:
                        if v in fname:
                            found_variants.add(v)
    
    discovered["CF"] = sorted(list(found_cf)) if found_cf else cf_catalog
    if found_variants:
        discovered["graph_variants"] = sorted(list(found_variants))
    
    # 2. Discover CB Models (Embedders)
    emb_map = {
        "vn-sbert": "vietnamese-sbert",
        "bge-m3": "bge-m3",
    }
    
    for display_name, file_prefix in emb_map.items():
        # Check standard checkpoint folder or data dir
        paths = [
            Path(f"checkpoints/{file_prefix}_article_embeddings.pt"),
            Path(DATA_DIR) / f"{file_prefix}_embeddings.pt",
            Path("models/checkpoints") / f"{file_prefix}_article_embeddings.pt" # Check unzipped
        ]
        if any(p.exists() for p in paths):
            discovered["CB"].append(display_name)
            
    # Always include 'session' if we have any dense embeddings
    if len(discovered["CB"]) > 3:
        discovered["CB"].append("session")
        
    return discovered

MODEL_OPTIONS = discover_trained_models()


EMBEDDING_OPTIONS = {
    "vn-sbert": "VN-SBERT (Standard Semantic, 540MB)",
    "bge-m3": "BGE-M3 (High Accuracy, 2.27GB)",
    "tfidf": "TF-IDF (Statistical Baseline)",
    "random": "Random (Initial State)"
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


class LSAWrapper:
    """Wrapper for LSA (Latent Semantic Analysis) based recommendation."""
    
    def __init__(self, lsa_matrix, articles_df):
        self.lsa_matrix = torch.tensor(lsa_matrix, dtype=torch.float32)
        self.articles_df = articles_df
        self.lsa_norm = F.normalize(self.lsa_matrix, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend based on cosine similarity in latent topic space."""
        if not history_indices:
            return [], []
        
        # Average LSA vector of history
        history_vecs = self.lsa_norm[history_indices]
        query = history_vecs.mean(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        similarities = torch.mm(query, self.lsa_norm.t()).squeeze()
        
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()


class NaiveBayesWrapper:
    """Wrapper for Naive Bayes probabilistic recommendation."""
    
    def __init__(self, vectorizer, nb_model, X_matrix, articles_df):
        self.vectorizer = vectorizer
        self.nb = nb_model
        self.X = X_matrix  # Sparse matrix
        self.articles_df = articles_df
    
    def recommend(self, history_indices, k=10):
        """Recommend articles with similar category probability distribution."""
        if not history_indices:
            return [], []
        
        # Get category distribution from history
        history_cats = self.articles_df.iloc[history_indices]['source_category'].value_counts()
        target_cat = history_cats.idxmax()  # Most frequent category
        
        # Get probability of each article belonging to target category
        proba = self.nb.predict_proba(self.X)
        cat_idx = list(self.nb.classes_).index(target_cat) if target_cat in self.nb.classes_ else 0
        scores = proba[:, cat_idx]
        
        # Exclude history
        scores = list(scores)
        for idx in history_indices:
            scores[idx] = -1
        
        # Top-k
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[:k]
        top_scores = [scores[i] for i in top_indices]
        
        return top_indices, top_scores


class SessionWrapper:
    """Wrapper for Session-based recommendation with attention over recent history."""
    
    def __init__(self, embeddings, articles_df):
        self.embeddings = embeddings
        self.articles_df = articles_df
        self.embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    def recommend(self, history_indices, k=10):
        """Recommend using exponential decay attention over recent history."""
        if not history_indices:
            return [], []
        
        # Session-based: use only last N items with decay weights
        max_session = min(10, len(history_indices))
        recent_indices = history_indices[-max_session:]
        
        # Exponential decay: more recent = higher weight
        weights = torch.tensor([0.9 ** (max_session - i - 1) for i in range(len(recent_indices))])
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of recent item embeddings
        recent_embs = self.embeddings_norm[recent_indices]
        query = (recent_embs * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        query = F.normalize(query, p=2, dim=1)
        
        # Similarity
        similarities = torch.mm(query, self.embeddings_norm.t()).squeeze()
        
        for idx in history_indices:
            similarities[idx] = -float('inf')
        
        scores, indices = torch.topk(similarities, k=min(k, len(similarities)))
        return indices.tolist(), scores.tolist()


@st.cache_data
def load_resources(data_dir=DATA_DIR, raw_dir=RAW_DIR, specific_graph_path=None):
    status = {"errors": [], "warnings": []}
    
    # Articles loading
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
    if specific_graph_path and Path(specific_graph_path).exists():
        cf_cache_path = Path(specific_graph_path)
    else:
        cf_cache_path = Path(data_dir) / "cf_cache.pt"
        
    if cf_cache_path.exists():
        try:
            cache = torch.load(cf_cache_path, map_location='cpu', weights_only=False)
            adj_norm = cache.get('adj_norm')
            
            # Compute adj_norm from edge_index if missing
            if adj_norm is None and 'edge_index' in cache:
                edge_index = cache['edge_index']
                n_users = cache.get('n_users', edge_index[0].max().item() + 1)
                n_items = cache.get('n_items', edge_index[1].max().item() + 1)
                
                # Build sparse adjacency matrix for bipartite graph
                n_nodes = n_users + n_items
                row = torch.cat([edge_index[0], edge_index[1] + n_users])
                col = torch.cat([edge_index[1] + n_users, edge_index[0]])
                
                # Symmetric normalization: D^(-0.5) * A * D^(-0.5)
                edge_weight = cache.get('edge_weight')
                if edge_weight is not None:
                    # Duplicate weights for symmetric matrix (bipartite)
                    values = torch.cat([edge_weight, edge_weight])
                else:
                    values = torch.ones(row.size(0))
                
                adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (n_nodes, n_nodes))
                
                # Compute degree
                degree = torch.sparse.sum(adj, dim=1).to_dense()
                d_inv_sqrt = torch.pow(degree, -0.5)
                d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
                
                # Normalize
                row_norm = d_inv_sqrt[row]
                col_norm = d_inv_sqrt[col]
                norm_values = row_norm * col_norm
                adj_norm = torch.sparse_coo_tensor(torch.stack([row, col]), norm_values, (n_nodes, n_nodes)).coalesce()
            
            # Cache mappings take priority over JSON files if they exist there
            if 'user_map' in cache: user_map_cf = cache['user_map']
            if 'article_map' in cache: article_map_cf = cache['article_map']
        except: pass

    # 3b. Try lightgcl_data.pkl for mappings (has filtered IDs from k-core)
    lgcl_path = Path(data_dir) / "lightgcl_data.pkl"
    if lgcl_path.exists() and (not user_map_cf or not article_map_cf):
        try:
            import pickle
            with open(lgcl_path, 'rb') as f:
                lgcl_data = pickle.load(f)
            if 'user2idx' in lgcl_data and not user_map_cf:
                user_map_cf = lgcl_data['user2idx']
            if 'item2idx' in lgcl_data and not article_map_cf:
                article_map_cf = lgcl_data['item2idx']
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
def load_cf_model(model_name, n_users, n_items, graph_name=None):
    """Generic CF Model Loader with graph-aware loading"""
    try:
        import torch
        if model_name.lower() == 'lightgcl':
            from src.models.lightgcl import LightGCL as ModelClass
        elif model_name.lower() == 'simgcl':
            from src.models.simgcl import SimGCL as ModelClass
        elif model_name.lower() == 'xsimgcl':
            from src.models.xsimgcl import XSimGCL as ModelClass
        elif model_name.lower() == 'ma-hcl':
            from src.models.ma_hcl import MAHCL as ModelClass
        elif model_name.lower() == 'sim-mahgn':
            from src.models.sim_mahgn import SimMAHGN as ModelClass
        else:
            return None

        # Find best matching checkpoint
        files = []
        search_dirs = [MODELS_DIR, os.path.join(MODELS_DIR, "models")]
        
        for d in search_dirs:
            if not os.path.exists(d): continue
            if graph_name:
                # Try graph-specific first (e.g., simgcl_strict_g2_*.pt)
                f_list = glob.glob(f"{d}/{model_name.lower()}_{graph_name}_*.pt")
                if f_list: files.extend(f_list)
            
            # Fallback to general if no graph-specific found in THIS dir
            if not files:
                f_list = glob.glob(f"{d}/{model_name.lower()}_*.pt")
                if f_list: files.extend(f_list)
            
        if not files: 
            # Final fallback: legacy pattern
            files = glob.glob(f"{MODELS_DIR}/{model_name.lower()}_*.pt")
            
        if not files: return None
        
        # SMART SELECTION: Find latest file that matches the current dimensions
        # This prevents "Loose loading" errors by skipping incompatible weights
        candidate_files = sorted(files, key=os.path.getctime, reverse=True)
        best_checkpoint = None
        best_state_dict = None
        best_config = None
        
        for f in candidate_files:
            try:
                ckpt = torch.load(f, map_location='cpu', weights_only=False)
                sd = ckpt['model_state_dict']
                cfg = ckpt.get('config', {})
                
                # Check dimensions from state_dict
                ckpt_u = None
                ckpt_i = None
                for key in ['E_u_0', 'user_embedding.weight']:
                    if key in sd: ckpt_u = sd[key].shape[0]; break
                for key in ['E_i_0', 'item_embedding.weight']:
                    if key in sd: ckpt_i = sd[key].shape[0]; break
                
                # If dimensions match, we found our winner
                if ckpt_u == n_users and ckpt_i == n_items:
                    best_checkpoint = ckpt
                    best_state_dict = sd
                    best_config = cfg
                    break
            except: continue
        
        # Fallback to the latest one even if mismatched (legacy behavior, but silent)
        if best_checkpoint is None:
            latest = candidate_files[0]
            best_checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            best_state_dict = best_checkpoint['model_state_dict']
            best_config = best_checkpoint.get('config', {})
        
        config = best_config
        state_dict = best_state_dict
        
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

        elif model_name.lower() == 'ma-hcl':
            model = ModelClass(
                n_users=n_users,
                n_items=n_items,
                embedding_dim=dim,
                n_layers=config.get('n_layers', 3),
                ssl_weight=config.get('cl_rate', config.get('ssl_weight', 0.1)),
                eps=config.get('eps', 0.1),
                temp=config.get('temp', 0.2),
                n_categories=config.get('n_categories', 0)
            )

        elif model_name.lower() == 'sim-mahgn':
            model = ModelClass(
                n_users=n_users, 
                n_items=n_items, 
                embedding_dim=dim, 
                n_layers=config.get('n_layers', 3),
                n_categories=config.get('n_categories', 0),
                heads=config.get('heads', 2)
            )
             
        try:
             model.load_state_dict(state_dict, strict=False)
        except:
             pass
             
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None

@st.cache_resource
def load_cb_model(model_type, articles_df, embedding_name=None):
    """Load Content-Based Models (TF-IDF, PhoBERT, VN-SBERT, etc.)"""
    try:
        if model_type.lower() == 'tf-idf':
            from src.models.content_based import TFIDFRecommender
            texts = (articles_df['title'].fillna('') + " " + articles_df['short_description'].fillna('')).tolist()
            model = TFIDFRecommender(1, len(articles_df))
            with st.spinner("Fitting TF-IDF..."):
                model.encode_articles(texts)
            return model
            
        elif model_type.lower() in ['phobert', 'simcse', 'e5', 'hybrid', 'session', 'vn-sbert', 'bge-m3', 'vndoc', 'e5-large', 'e5-base', 'gte']:
             # Determine path based on embedding_name or model_type fallback
             emb_key = embedding_name if embedding_name else model_type.lower()
             
             # Map short names to file patterns
             file_map = {
                 'vn-sbert': 'vietnamese-sbert',
                 'bge-m3': 'bge-m3',
                 'vndoc': 'vietnamese-document-embedding', 
                 'e5-large': 'e5-large',
                 'e5-base': 'e5-base',
                 'gte': 'gte-multilingual',
                 'phobert': 'phobert',
                 'simcse': 'simcse',
                 'e5': 'e5-large' # Default for e5 model option
             }
             
             file_prefix = file_map.get(emb_key, emb_key)
             emb_path = Path(f"checkpoints/{file_prefix}_article_embeddings.pt")
             
             if not emb_path.exists():
                 # Try alternative path
                 emb_path = Path(st.session_state.get('data_dir', DATA_DIR)) / f"{file_prefix}_embeddings.pt"
             
             if emb_path.exists():
                 try:
                     emb_dict = torch.load(emb_path, map_location='cpu', weights_only=False)
                     
                     # Handle different formats (Tensor vs Dict)
                     if isinstance(emb_dict, torch.Tensor):
                         embeddings = emb_dict
                         # Alignment logic: Pad or truncate to match articles_df
                         n_emb = embeddings.shape[0]
                         n_art = len(articles_df)
                         if n_emb != n_art:
                             embed_dim = embeddings.shape[1]
                             new_embeddings = torch.zeros(n_art, embed_dim)
                             # Copy what we have
                             copy_size = min(n_emb, n_art)
                             new_embeddings[:copy_size] = embeddings[:copy_size]
                             embeddings = new_embeddings
                     elif isinstance(emb_dict, dict):
                         # Convert dict (URL -> embedding) to aligned matrix
                         urls = articles_df['url'].tolist()
                         # Get dim from first value
                         first_emb = next(iter(emb_dict.values()))
                         embed_dim = first_emb.shape[0]
                         embeddings = torch.zeros(len(urls), embed_dim)
                         
                         matched = 0
                         for i, url in enumerate(urls):
                             if url in emb_dict:
                                 embeddings[i] = emb_dict[url]
                                 matched += 1
                         # st.success(f"Loaded {emb_key} embeddings: Matched {matched}/{len(urls)}")
                     
                     if model_type.lower() == 'session':
                         return SessionWrapper(embeddings, articles_df)
                     else:
                         return PhoBERTWrapper(embeddings, articles_df) # Generic wrapper works for all dense embeddings
                         
                 except Exception as e:
                     st.warning(f"Failed to load {emb_key} embeddings: {e}")
                     return None
             else:
                 st.warning(f"Embeddings {emb_key} not found. Generate them first.")
                 return None
                 
        elif model_type.lower() == 'lsa':
             # Placeholder for LSA
             return None
             
        elif model_type.lower() == 'naivebayes':
             # Placeholder
             return None
             
        
        # 3. LSA (Latent Semantic Analysis) - Latent Semantic CBRS
        elif model_type.upper() == 'LSA':

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            texts = (articles_df['title'].fillna('') + " " + articles_df['short_description'].fillna('')).tolist()
            
            with st.spinner("Building LSA model (TF-IDF + SVD)..."):
                # TF-IDF first
                tfidf = TfidfVectorizer(max_features=5000, stop_words=None)
                tfidf_matrix = tfidf.fit_transform(texts)
                
                # SVD to reduce to latent topics
                n_components = min(100, tfidf_matrix.shape[1] - 1)
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                lsa_matrix = svd.fit_transform(tfidf_matrix)
                
                st.success(f"LSA: {tfidf_matrix.shape[1]} features → {n_components} latent topics (explained var: {svd.explained_variance_ratio_.sum():.2%})")
            
            return LSAWrapper(lsa_matrix, articles_df)
        
        # 4. Naive Bayes - Probabilistic CBRS
        elif model_type.upper() == 'NAIVEBAYES':
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.naive_bayes import MultinomialNB
            
            with st.spinner("Building Naive Bayes model..."):
                texts = (articles_df['title'].fillna('') + " " + articles_df['short_description'].fillna('')).tolist()
                categories = articles_df['source_category'].fillna('unknown').tolist()
                
                # Vectorize
                vectorizer = CountVectorizer(max_features=3000)
                X = vectorizer.fit_transform(texts)
                
                # Train NB classifier (for category prediction)
                nb = MultinomialNB()
                nb.fit(X, categories)
                
                st.success(f"NaiveBayes: {len(set(categories))} categories, {X.shape[1]} features")
            
            return NaiveBayesWrapper(vectorizer, nb, X, articles_df)
        
              
        return None
    except Exception as e:
        st.error(f"CB Load Error: {e}")
        return None

def get_recs(model, model_type, user_idx, history_urls, article_map, articles_df, k=10, 
             adj_norm=None, user_priors=None, semantic_ids=None, score_type="Normalized (0-1)"):
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
                            result = model(adj_norm)
                            # Handle models returning more than 2 values
                            if isinstance(result, tuple) and len(result) >= 2:
                                user_all, item_all = result[0], result[1]
                            else:
                                raise ValueError("Unexpected return type")
                            scores = torch.mm(user_all[user_idx].unsqueeze(0), item_all.t()).squeeze()
                        except Exception:
                            # Fallback to raw embeddings if model(adj_norm) fails
                            if hasattr(model, 'user_embedding'):
                                u_emb = model.user_embedding.weight[user_idx]
                                i_embs = model.item_embedding.weight
                            elif hasattr(model, 'E_u_0'):
                                u_emb = model.E_u_0[user_idx]
                                i_embs = model.E_i_0
                            else:
                                return []
                            scores = torch.matmul(u_emb, i_embs.t())
                    else: # adj_norm is None, use raw embeddings directly
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
                
                # Apply score normalization based on score_type
                vals_np = vals.cpu().numpy()
                if len(vals_np) > 0:
                    if score_type == "Normalized (0-1)":
                        # Min-max normalization within top-K
                        score_min, score_max = vals_np.min(), vals_np.max()
                        if score_max > score_min:
                            vals_np = (vals_np - score_min) / (score_max - score_min)
                        else:
                            vals_np = np.ones_like(vals_np)
                    elif score_type == "Sigmoid":
                        # Sigmoid to [0, 1]
                        vals_np = 1 / (1 + np.exp(-vals_np))
                    # else: Raw Scores - keep as is
                
            recs = []
            for idx, score in zip(indices.cpu().numpy(), vals_np):
                url = idx_to_url.get(idx, None)
                if url: recs.append((url, float(score)))
            return recs
            
        elif model_type == 'TF-IDF':
            url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
            hist_indices = [url_to_idx[u] for u in history_urls if u in url_to_idx]
            if not hist_indices: return []
            top, scores = model.recommend(hist_indices, k=k)
            
            # Apply score normalization
            if len(scores) > 0:
                scores = np.array(scores)
                if score_type == "Normalized (0-1)":
                    s_min, s_max = scores.min(), scores.max()
                    if s_max > s_min:
                        scores = (scores - s_min) / (s_max - s_min)
                    else:
                        scores = np.ones_like(scores)
                elif score_type == "Sigmoid":
                    scores = 1 / (1 + np.exp(-scores))

            recs = []
            for idx, sc in zip(top, scores):
                if idx < len(articles_df):
                     recs.append((articles_df.iloc[idx]['url'], float(sc)))
            return recs
        
        elif model_type in ['PhoBERT', 'SimCSE']:
            url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
            hist_indices = [url_to_idx[u] for u in history_urls if u in url_to_idx]
            if not hist_indices: 
                st.warning("No history found for this user.")
                return []
            top, scores = model.recommend(hist_indices, k=k)
            
            # Apply score normalization
            if len(scores) > 0:
                scores = np.array(scores)
                if score_type == "Normalized (0-1)":
                    s_min, s_max = scores.min(), scores.max()
                    if s_max > s_min:
                        scores = (scores - s_min) / (s_max - s_min)
                    else:
                        scores = np.ones_like(scores)
                elif score_type == "Sigmoid":
                    scores = 1 / (1 + np.exp(-scores))

            recs = []
            for idx, sc in zip(top, scores):
                if idx < len(articles_df):
                     recs.append((articles_df.iloc[idx]['url'], float(sc)))
            return recs
            

            
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return []


CF_GRAPHS = {
    "🚀 Deep Social (Strict G2)": "data/processed/strict_g2/full_hetero_graph.pt",
    "🔥 Large Scale (Regular G2)": "data/processed/regular_g2/full_hetero_graph.pt",
    "🧪 Enhanced V1": "data/processed/enhanced_v1/full_hetero_graph.pt",
    "Standard Bipartite": "data/processed/cf_cache.pt"
}

def main():

    st.title("📰 Comprehensive RecSys Dashboard")
    
    # Session State for Paths
    if 'data_dir' not in st.session_state: st.session_state['data_dir'] = DATA_DIR
    if 'raw_dir' not in st.session_state: st.session_state['raw_dir'] = RAW_DIR
    
    # --- SIDEBAR: GLOBAL CONFIG ---
    st.sidebar.title("📊 Experiment Dashboard")
    
    # 1. Graph Topology Selector
    topology_options = {
        "strict_g2": "🔥 Deep Social (Strict G2)",
        "strict_g3": "📂 Category Hubs (Strict G3)",
        "strict_g1": "🔗 Bipartite (Strict G1)",
        "g2": "🌊 Dense Social (Regular G2)"
    }
    available_variants = [v for v in topology_options.keys() if os.path.exists(os.path.join(DATA_DIR, v)) or v in MODEL_OPTIONS["graph_variants"]]
    if not available_variants: available_variants = ["strict_g2"]
    
    selected_variant = st.sidebar.selectbox("Graph Topology", available_variants, 
                                            format_func=lambda x: topology_options.get(x, x),
                                            index=0, help="Select the graph variant for and models.")
    
    # 2. CF Model Selection (Global)
    cf_model_choice = st.sidebar.selectbox("Recommendation Model", MODEL_OPTIONS["CF"], 
                                          help="Choose the GNN architecture to test")

    # Auto-align DATA_DIR based on variant
    active_data_dir = os.path.join(DATA_DIR, selected_variant) if os.path.exists(os.path.join(DATA_DIR, selected_variant)) else DATA_DIR
    
    res = load_resources(
        data_dir=active_data_dir,
        raw_dir=RAW_DIR,
        specific_graph_path=os.path.join(active_data_dir, "cf_cache.pt")
    )
    articles_df, user_map_cf, article_map_cf, user_history, adj_norm, user_priors, data_status = res
    
    # --- GLOBAL MODEL LOADING ---
    # Load model once for use in all tabs
    cf_model = load_cf_model(cf_model_choice, len(user_map_cf), len(article_map_cf), graph_name=selected_variant)
    
    semantic_ids = None
    if 'xsimgcl' in [m.lower() for m in MODEL_OPTIONS["CF"]]:
        try:
             emb_path = Path(st.session_state['data_dir']) / "article_embeddings.pt"
             if emb_path.exists():
                 pretrained = torch.load(emb_path, map_location='cpu', weights_only=False)
                 semantic_ids = generate_semantic_ids(pretrained, bits=3)
        except: pass
    
    # Sidebar Controls
    st.sidebar.divider()
    
    with st.sidebar.expander("🔍 User Search & Filtering", expanded=True):
        min_interactions = st.slider("Min Interactions", 1, 20, 2, help="Filter users by reading history length")
        user_type_filter = st.selectbox("Market Segment", ["All Users", "Warm Start (Trained)", "Cold Start (New)"],
                                       help="Warm: In training set, Cold: New users")
        
        # user_ids MUST be defined here
        user_ids = sorted([uid for uid, hist in user_history.items() if len(hist) >= min_interactions])
        
        # Apply Market Segment filter
        if user_type_filter == "Warm Start (Trained)":
            user_ids = [uid for uid in user_ids if uid in user_map_cf]
        elif user_type_filter == "Cold Start (New)":
            user_ids = [uid for uid in user_ids if uid not in user_map_cf]
            
        search_query = st.text_input("🔍 Search ID", placeholder="User ID...", label_visibility="collapsed")
        if search_query:
            user_ids = [uid for uid in user_ids if search_query.lower() in str(uid).lower()]

    col_s1, col_s2 = st.sidebar.columns([3, 1])
    with col_s1:
        st.markdown(f"**Found:** {len(user_ids)} users")
    with col_s2:
        if st.button("🎲", help="Randomize from filtered list"):
            import random
            if user_ids:
                rand_u = random.choice(user_ids)
                st.session_state['selected_user_id_manual'] = rand_u
                st.rerun()
            else:
                st.toast("No users match filters!")
    
    # Format user options with cold-start indicator
    def format_user_option(uid):
        hist_len = len(user_history.get(uid, []))
        is_cold = uid not in user_map_cf  # Not in training set
        indicator = "❄️" if is_cold else "✅"
        return f"{indicator} {uid} ({hist_len} articles)"
    
    user_options = ["📊 All Users (Overview)"] + [format_user_option(uid) for uid in user_ids]
    
    # Handle manual selection from random button
    default_idx = 0
    if st.session_state.get('selected_user_id_manual'):
        target_uid = st.session_state['selected_user_id_manual']
        for i, uid in enumerate(user_ids):
            if uid == target_uid:
                default_idx = i + 1
                break
        # Clear it so it doesn't stick forever
        st.session_state['selected_user_id_manual'] = None

    selected_idx = st.sidebar.selectbox("Select User", list(range(len(user_options))), 
                                        format_func=lambda i: user_options[i], index=default_idx)
    
    # Determine if we're in "All Users" mode
    all_users_mode = (selected_idx == 0)
    
    # Get actual user ID from index (index 0 is "All Users", so subtract 1)
    if all_users_mode:
        selected_user = user_ids[0] if user_ids else None
    else:
        selected_user = user_ids[int(selected_idx) - 1]
    


    # Persistent Tabs (Radio styled as tabs)
    # Inference Selection
    tabs = ["🚀 Recommendations", "⚔️ Comparison"]
    nav = st.radio("Navigation", tabs, key="main_nav", horizontal=True, label_visibility="collapsed")
    
    # Internal flag for advanced features (hidden but can be toggled by query param if needed)
    show_advanced = st.query_params.get("dev", "false").lower() == "true"
    st.divider()


    
    # --- PAGE 1: RECOMMENDATIONS ---
    if nav == "🚀 Recommendations":
        st.header("🚀 Personalized Recommendations")
        st.caption("Hybrid System: Combining Social Patterns (CF) + Content Similarity (CB)")
        
        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            st.subheader("🛠️ Engine Config")
            
            # Recommendation Strategy Toggle
            strategy = st.radio("Primary Strategy", ["Hybrid (Recommended)", "Pure Social (CF)", "Pure Content (CB)"], 
                                help="Hybrid blends social structure with text similarity.")
            
            # Auto-set Alpha based on strategy
            if strategy == "Pure Social (CF)": alpha_default = 1.0
            elif strategy == "Pure Content (CB)": alpha_default = 0.0
            else: alpha_default = 0.5
            
            alpha = st.slider("Social vs Content Weight (α)", 0.0, 1.0, alpha_default, 0.1, 
                             help="1.0 = Social only, 0.0 = Content only")
            
            with st.expander("⚙️ Content & Display Settings", expanded=False):
                st.markdown("**📝 Content Models**")
                selected_cb = st.multiselect("Active Embedders", MODEL_OPTIONS["CB"], 
                                             default=[m for m in ["tf-idf", "vn-sbert", "bge-m3"] if m in MODEL_OPTIONS["CB"]],
                                             key="rec_cb_multi")
                
                k_rec = st.slider("Top K", 5, 20, 10, key="k_rec")
                
                # Score normalization options
                score_type = st.selectbox("Score Display", ["Normalized (0-1)", "Raw Scores", "Sigmoid"], 
                                         help="How to display recommendation scores")
            
            # Freshness Boost (Important for News)
            use_freshness = st.checkbox("🕐 Boost Recent News", value=True)
            freshness_weight = st.slider("Freshness Intensity", 0.0, 0.5, 0.2, 0.05) if use_freshness else 0.0
            
            if selected_user:
                history = list(dict.fromkeys(user_history.get(selected_user, [])))
                is_cold = selected_user not in user_map_cf
                
                status_color = "blue" if not is_cold else "cyan"
                status_text = "Warm Start (Social Enabled)" if not is_cold else "Cold Start (Content Only)"
                st.markdown(f"**User Status:** :{status_color}[{status_text}]")
                
                with st.expander(f"📚 Recent History ({len(history)})", expanded=True):
                    if history:
                        url_to_meta = articles_df.set_index('url')[['title', 'source_category']].to_dict('index')
                        for i, u in enumerate(history[:5], 1):
                            meta = url_to_meta.get(u, {})
                            title = str(meta.get('title', 'Unknown'))
                            st.markdown(f"<small>**#{i}** {title[:50]}...</small>", unsafe_allow_html=True)
                        if len(history) > 5: st.caption(f"... +{len(history)-5} more")

        with col_R:
            st.subheader("📊 Recommended for You")
            if selected_user:
                with st.spinner("Analyzing patterns..."):
                    # 1. CF Scores
                    cf_scores = {}
                    actual_alpha = 0.0 if is_cold else alpha
                    
                    if actual_alpha > 0:
                        if cf_model:
                            recs = get_recs(cf_model, cf_model_choice, user_map_cf[selected_user], [], article_map_cf, articles_df, 100,
                                         adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids, score_type=score_type)
                            cf_scores = {u: s for u, s in recs}

                    # 2. CB Scores
                    cb_scores = {}
                    if (1 - actual_alpha) > 0 and selected_cb:
                        cb_results = []
                        weight_per_cb = 1.0 / len(selected_cb)
                        
                        for cb_name in selected_cb:
                            m = load_cb_model(cb_name, articles_df)
                            if m:
                                recs = get_recs(m, cb_name.upper() if cb_name in ["tf-idf", "lsa", "naivebayes"] else "PhoBERT", 
                                               0, history, {}, articles_df, 100, score_type=score_type)
                                cb_results.append((recs, weight_per_cb))
                        
                        for rec_list, w in cb_results:
                            for u, s in rec_list:
                                cb_scores[u] = cb_scores.get(u, 0) + s * w
                    
                    # 3. Merge and Rerank
                    all_urls = set(cf_scores.keys()) | set(cb_scores.keys())
                    hybrid_scores = []
                    for url in all_urls:
                        s_cf = cf_scores.get(url, 0)
                        s_cb = cb_scores.get(url, 0)
                        combined = actual_alpha * s_cf + (1 - actual_alpha) * s_cb
                        source = "Hybrid" if url in cf_scores and url in cb_scores else ("Social" if url in cf_scores else "Content")
                        hybrid_scores.append((url, combined, s_cf, s_cb, source))
                    
                    # Freshness Boost
                    if use_freshness and freshness_weight > 0:
                        url_to_date = dict(zip(articles_df['url'], articles_df.get('crawled_at', pd.Series([None]*len(articles_df)))))
                        if any(url_to_date.values()):
                            scores_arr = np.array([h[1] for h in hybrid_scores])
                            dates_arr = [url_to_date.get(h[0]) for h in hybrid_scores]
                            reranker = CalibratedReRanker(np.zeros(len(hybrid_scores)), freshness_lambda=0.1)
                            boosted = reranker.freshness_boost(scores_arr, dates_arr, boost_weight=freshness_weight)
                            hybrid_scores = [(h[0], float(boosted[i]), h[2], h[3], h[4]) for i, h in enumerate(hybrid_scores)]

                    hybrid_scores.sort(key=lambda x: -x[1])
                    top_recs = hybrid_scores[:k_rec]
                    
                    # Display Results
                    if not top_recs:
                        st.warning("No recommendations found. Try a different strategy.")
                    else:
                        url_to_meta = articles_df.set_index('url')[['title', 'short_description', 'source_category']].to_dict('index')
                        for i, (url, score, cf_s, cb_s, source) in enumerate(top_recs):
                            meta = url_to_meta.get(url, {})
                            title = str(meta.get('title', 'Unknown'))[:80]
                            desc = str(meta.get('short_description', ''))[:120]
                            cat = CATEGORY_MAP.get(meta.get('source_category', ''), meta.get('source_category', 'N/A'))
                            
                            source_colors = {"Hybrid": "#9c27b0", "Social": "#2196f3", "Content": "#4caf50"}
                            s_color = source_colors.get(source, "#888")
                            
                            st.markdown(f"""
                            <div style="padding:15px; margin:10px 0; border-radius:12px; background:#fff; border-left:5px solid {s_color}; box-shadow:0 2px 4px rgba(0,0,0,0.05);">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="color:#888; font-weight:bold;">#{i+1}</span>
                                    <span style="background:{s_color}; color:white; padding:2px 8px; border-radius:10px; font-size:0.75em;">{source} • {score:.3f}</span>
                                </div>
                                <div style="font-weight:bold; margin:6px 0; font-size:1.1em; color:#2c3e50;">
                                    <a href="{url}" target="_blank" style="text-decoration:none; color:inherit;">{title}</a>
                                </div>
                                <div style="font-size:0.9em; color:#666; margin-bottom:10px;">{desc}...</div>
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="background:#f0f2f6; padding:2px 8px; border-radius:5px; font-size:0.8em; color:#555;">{cat}</span>
                                    <a href="{url}" target="_blank" style="font-size:0.85em; color:{s_color}; text-decoration:none; font-weight:bold;">Đọc ngay →</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("👈 Select a user to generate recommendations")

    # --- PAGE 4: COMPARISON ---
    if nav == "⚔️ Comparison":
        st.header("⚔️ A/B Model Comparison")
        st.caption("Compare two models side-by-side on the same user")
        
        col_config, col_a, col_b = st.columns([1, 1.5, 1.5])
        
        with col_config:
            st.subheader("⚙️ Config")
            model_a = st.selectbox("Model A", MODEL_OPTIONS["CF"], key="ab_model_a")
            model_b = st.selectbox("Model B", MODEL_OPTIONS["CF"], index=1 if len(MODEL_OPTIONS["CF"]) > 1 else 0, key="ab_model_b")
            k_compare = st.slider("Top K", 5, 15, 10, key="ab_k")
            
            if st.button("🔄 Compare Models", type="primary"):
                st.session_state['ab_compare_run'] = True
        
        if st.session_state.get('ab_compare_run') and selected_user in user_map_cf:
            user_idx = user_map_cf[selected_user]
            url_to_meta = articles_df.set_index('url')[['title', 'source_category']].to_dict('index')
            
            # Model A
            with col_a:
                st.subheader(f"🅰️ {model_a}")
                model_obj_a = load_cf_model(model_a, len(user_map_cf), len(article_map_cf), graph_name=selected_variant)
                if model_obj_a:
                    recs_a = get_recs(model_obj_a, model_a, user_idx, [], article_map_cf, articles_df, k_compare,
                                     adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids)
                    for i, (url, score) in enumerate(recs_a):
                        meta = url_to_meta.get(url, {})
                        title = str(meta.get('title', 'Unknown'))[:50]
                        cat = CATEGORY_MAP.get(meta.get('source_category', ''), 'N/A')
                        st.markdown(f"**{i+1}.** {title}... `{score:.3f}` {cat}")
                else:
                    st.warning(f"Model {model_a} not trained yet")
            
            # Model B
            with col_b:
                st.subheader(f"🅱️ {model_b}")
                model_obj_b = load_cf_model(model_b, len(user_map_cf), len(article_map_cf), graph_name=selected_variant)
                if model_obj_b:
                    recs_b = get_recs(model_obj_b, model_b, user_idx, [], article_map_cf, articles_df, k_compare,
                                     adj_norm=adj_norm, user_priors=user_priors, semantic_ids=semantic_ids)
                    for i, (url, score) in enumerate(recs_b):
                        meta = url_to_meta.get(url, {})
                        title = str(meta.get('title', 'Unknown'))[:50]
                        cat = CATEGORY_MAP.get(meta.get('source_category', ''), 'N/A')
                        st.markdown(f"**{i+1}.** {title}... `{score:.3f}` {cat}")
                else:
                    st.warning(f"Model {model_b} not trained yet")
            
            # Overlap Analysis
            if 'recs_a' in dir() and 'recs_b' in dir():
                urls_a = set([u for u, s in recs_a])
                urls_b = set([u for u, s in recs_b])
                overlap = len(urls_a & urls_b)
                st.info(f"📊 Overlap: {overlap}/{k_compare} items ({overlap/k_compare*100:.0f}%) appear in both lists")
        
        elif st.session_state.get('ab_compare_run'):
            st.warning("User not found in CF data. Please select a valid user.")





if __name__ == "__main__":
    main()
