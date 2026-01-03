import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Tuple, List, Optional
import sys

# Add root project path to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.app.core.config import settings
from backend.app.services.models import PhoBERTWrapper, LSAWrapper, NaiveBayesWrapper, SessionWrapper

# Constants
MODEL_OPTIONS = {
    "CF": ["MA-HCL", "SimGCL", "XSimGCL", "LightGCL", "LightGCN"],
    "CB": ["vn-sbert", "bge-m3", "gte-multilingual", "e5-large", "e5-base", "tf-idf", "random"],
    "graph_variants": ["strict_g1", "strict_g2", "strict_g3"]
}

def discover_trained_models():
    """Scan models/ and checkpoints/ to find which models have available weights."""
    discovered = {
        "CF": [],
        "CB": ["tf-idf", "lsa", "naivebayes"], # Base procedural models always available
        "graph_variants": ["strict_g1", "strict_g2", "strict_g3"] # Default variants
    }
    
    # 1. Discover CF Models (GNNs)
    search_dirs = [settings.MODELS_DIR, os.path.join(settings.MODELS_DIR, "models")]
    cf_catalog = ["MA-HCL", "SimGCL", "XSimGCL", "LightGCL", "LightGCN"]
    
    found_cf = set()
    found_variants = set()
    
    for d in search_dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if not f.endswith(".pt"): continue
            fname = f.lower()
            for model in cf_catalog:
                if model.lower() in fname.replace('_', '-'):
                    found_cf.add(model)
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
        paths = [
            Path(f"checkpoints/{file_prefix}_article_embeddings.pt"),
            Path(settings.DATA_DIR) / f"{file_prefix}_embeddings.pt",
            Path(settings.MODELS_DIR) / "checkpoints" / f"{file_prefix}_article_embeddings.pt"
        ]
        if any(p.exists() for p in paths):
            discovered["CB"].append(display_name)
            
    if len(discovered["CB"]) > 3:
        discovered["CB"].append("session")
        
    return discovered

MODEL_OPTIONS = discover_trained_models()

def load_resources(data_dir=settings.DATA_DIR, raw_dir=settings.RAW_DIR, specific_graph_path=None):
    status = {"errors": [], "warnings": []}
    
    # Articles loading
    articles_path = Path(raw_dir) / "articles.csv"
    articles_df = pd.DataFrame()
    if articles_path.exists():
        articles_df = pd.read_csv(articles_path)
        articles_df['title'] = articles_df['title'].fillna("Untitled")
        articles_df['short_description'] = articles_df['short_description'].fillna("")
        articles_df = articles_df.drop_duplicates(subset=['url'])
    else:
        status["errors"].append(f"Missing articles.csv in {raw_dir}")
        return None, None, None, None, None, None, status

    # Mappings from JSON
    user_map_cf = {}
    article_map_cf = {}
    u_map_path = Path(data_dir) / "user_map.json"
    a_map_path = Path(data_dir) / "article_map.json"

    if u_map_path.exists():
        with open(u_map_path) as f: user_map_cf = json.load(f)
    if a_map_path.exists():
        with open(a_map_path) as f: article_map_cf = json.load(f)

    # 3. Graph Data (Adj matrix + Mappings cache)
    adj_norm = None
    if specific_graph_path and Path(specific_graph_path).exists():
        cf_cache_path = Path(specific_graph_path)
    else:
        cf_cache_path = Path(data_dir) / "cf_cache.pt"
        
    if cf_cache_path.exists():
        try:
            cache = torch.load(cf_cache_path, map_location='cpu', weights_only=False)
            adj_norm = cache.get('adj_norm')
            edge_index = cache.get('edge_index')
            if 'user_map' in cache: user_map_cf = cache['user_map']
            if 'article_map' in cache: article_map_cf = cache['article_map']
        except: pass
    
    # Fallback: Load from full_hetero_graph.pt if adj_norm or edge_index is missing
    if adj_norm is None or edge_index is None:
        hetero_path = Path(data_dir) / "full_hetero_graph.pt"
        if hetero_path.exists():
            try:
                # Needed for HeteroData unpickling
                from torch_geometric.data import HeteroData 
                data = torch.load(hetero_path, map_location='cpu', weights_only=False)
                
                # Extract edge index (User -> Article)
                edge_index = None
                
                # Helper to find edge_index
                target_data = data
                if isinstance(data, dict):
                    if 'graph' in data: target_data = data['graph']
                    elif 'edge_index_dict' in data: target_data = data # Treat dict as having keys
                
                # Check if it's HeteroData (has edge_types attr) or Dict with specific keys
                edge_types = []
                if hasattr(target_data, 'edge_types'):
                    edge_types = target_data.edge_types
                elif isinstance(target_data, dict) and 'edge_index_dict' in target_data:
                    edge_types = target_data['edge_index_dict'].keys()
                
                for edge_type in edge_types:
                    src, rel, dst = edge_type
                    if src == 'user' and dst == 'article':
                        if hasattr(target_data, 'edge_types'):
                            edge_index = target_data[edge_type].edge_index
                        else:
                            edge_index = target_data['edge_index_dict'][edge_type]
                        break
                
                if edge_index is not None:
                    if hasattr(target_data, 'node_types'): # HeteroData
                         n_users = target_data['user'].num_nodes
                         n_items = target_data['article'].num_nodes
                    elif isinstance(target_data, dict): # Dict
                         n_users = target_data.get('n_users') or target_data.get('num_users') or 0
                         n_items = target_data.get('n_items') or target_data.get('num_articles') or 0
                         if n_users == 0 and 'user' in target_data.get('num_nodes_dict', {}):
                             n_users = target_data['num_nodes_dict']['user']
                         if n_items == 0 and 'article' in target_data.get('num_nodes_dict', {}):
                             n_items = target_data['num_nodes_dict']['article']

                    n_nodes = n_users + n_items
                    
                    # Build symmetric bipartite adjacency
                    row = torch.cat([edge_index[0], edge_index[1] + n_users])
                    col = torch.cat([edge_index[1] + n_users, edge_index[0]])
                    values = torch.ones(row.size(0))
                    
                    adj = torch.sparse_coo_tensor(torch.stack([row, col]), values, (n_nodes, n_nodes))
                    degree = torch.sparse.sum(adj, dim=1).to_dense()
                    d_inv_sqrt = torch.pow(degree, -0.5)
                    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
                    
                    row_norm = d_inv_sqrt[row]
                    col_norm = d_inv_sqrt[col]
                    norm_values = row_norm * col_norm
                    
                    adj_norm = torch.sparse_coo_tensor(torch.stack([row, col]), norm_values, (n_nodes, n_nodes)).coalesce()
                    print(f"Constructed adj_norm from {hetero_path}")
            except Exception as e:
                print(f"Failed to load hetero graph: {e}")

    # LightGCL fallback
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

    # User History
    replies_path = Path(raw_dir) / "replies.csv"
    user_history = {}
    if replies_path.exists():
        try:
            df_rep = pd.read_csv(replies_path)
            def clean(x):
                try: return str(int(float(x)))
                except: return str(x)
            col = 'user_id' if 'user_id' in df_rep.columns else 'reply_user_id'
            if col in df_rep.columns:
                df_rep[col] = df_rep[col].apply(clean)
                user_history = df_rep.groupby(col)['article_url'].apply(list).to_dict()
                
                if not user_map_cf:
                    counts = df_rep[col].value_counts()
                    valid_u = counts[counts >= 2].index
                    user_ids = sorted([uid for uid in df_rep[col].unique() if uid in valid_u])
                    user_map_cf = {uid: i for i, uid in enumerate(user_ids)}
                    status["warnings"].append("user_map_missing")
        except Exception as e:
            status["warnings"].append(f"history_load_error: {e}")
            
    if not article_map_cf and not user_history == {}:
        all_interacted = []
        for urls in user_history.values(): all_interacted.extend(urls)
        from collections import Counter
        counts = Counter(all_interacted)
        valid_a = [url for url, c in counts.items() if c >= 2]
        article_map_cf = {url: i for i, url in enumerate(sorted(valid_a))}
        status["warnings"].append("article_map_missing")

    # User Priors
    user_priors = None
    a_priors_path = Path(data_dir) / "user_priors.pt"
    if a_priors_path.exists():
        try:
            user_priors = torch.load(a_priors_path, map_location='cpu', weights_only=False)
        except Exception as e:
            status["warnings"].append(f"Failed to load user priors: {e}")

    return articles_df, user_map_cf, article_map_cf, user_history, adj_norm, user_priors, edge_index, status

@lru_cache(maxsize=10)
def load_cf_model(model_name, n_users, n_items, graph_name=None):
    """Generic CF Model Loader with graph-aware loading"""
    try:
        if model_name.lower() == 'lightgcl':
            from src.models.lightgcl import LightGCL as ModelClass
        elif model_name.lower() == 'simgcl':
            from src.models.simgcl import SimGCL as ModelClass
        elif model_name.lower() == 'xsimgcl':
            from src.models.xsimgcl import XSimGCL as ModelClass
        elif model_name.lower() == 'ma-hcl':
            from src.models.ma_hcl import MAHCL as ModelClass
        elif model_name.lower() == 'lightgcn':
            from src.models.lightgcn import LightGCN as ModelClass
        else:
            return None

        files = []
        search_dirs = [settings.MODELS_DIR, os.path.join(settings.MODELS_DIR, "models")]
        
        for d in search_dirs:
            if not os.path.exists(d): continue
            if graph_name:
                f_list = glob.glob(f"{d}/{model_name.lower()}_{graph_name}_*.pt")
                if f_list: files.extend(f_list)
            
            if not files:
                f_list = glob.glob(f"{d}/{model_name.lower()}_*.pt")
                if f_list: files.extend(f_list)
            
        if not files: 
            files = glob.glob(f"{settings.MODELS_DIR}/{model_name.lower()}_*.pt")
            
        if not files: return None
        
        candidate_files = sorted(files, key=os.path.getctime, reverse=True)
        best_checkpoint = None
        best_state_dict = None
        best_config = None
        
        for f in candidate_files:
            try:
                ckpt = torch.load(f, map_location='cpu', weights_only=False)
                sd = ckpt['model_state_dict']
                cfg = ckpt.get('config', {})
                
                ckpt_u = None
                ckpt_i = None
                for key in ['E_u_0', 'user_embedding.weight']:
                    if key in sd: ckpt_u = sd[key].shape[0]; break
                for key in ['E_i_0', 'item_embedding.weight']:
                    if key in sd: ckpt_i = sd[key].shape[0]; break
                
                if ckpt_u == n_users and ckpt_i == n_items:
                    best_checkpoint = ckpt
                    best_state_dict = sd
                    best_config = cfg
                    break
            except: continue
        
        if best_checkpoint is None:
            latest = candidate_files[0]
            best_checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
            best_state_dict = best_checkpoint['model_state_dict']
            best_config = best_checkpoint.get('config', {})
        
        config = best_config
        state_dict = best_state_dict

        dim = config.get('emb_dim', config.get('embed_dim', 64))
        layers = config.get('layers', config.get('n_layers', [64, 64]))
        if isinstance(layers, int): layers = [dim] * layers
        
        if model_name.lower() in ['ngcf', 'lightgcl', 'simgcl', 'igcl', 'bigcf', 'lightgcn']:
             if model_name.lower() == 'lightgcl':
                  n_l = len(layers) if isinstance(layers, list) else layers
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=n_l)

             elif model_name.lower() == 'simgcl':
                  model = ModelClass(n_users, n_items, embedding_dim=dim)
             
             elif model_name.lower() == 'lightgcn':
                  model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=len(layers) if isinstance(layers, list) else layers)
        
        elif model_name.lower() == 'xsimgcl':
            n_l = layers[0] if isinstance(layers, list) else layers
            model = ModelClass(n_users, n_items, embedding_dim=dim, n_layers=n_l)
            if 'semantic_layer.weight' in state_dict:
                from scripts.train_cf_models import SemanticEmbeddingLayer
                s_dim = state_dict['semantic_layer.weight'].shape[1]
                model.semantic_layer = SemanticEmbeddingLayer(s_dim, dim)
            if 'user_prior_layer.weight' in state_dict:
                from scripts.train_cf_models import UserPriorLayer
                p_dim = state_dict['user_prior_layer.weight'].shape[1]
                model.user_prior_layer = UserPriorLayer(p_dim, dim)

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
             
        try:
             model.load_state_dict(state_dict, strict=False)
        except:
             pass
             
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

# Caching this might be too heavy if articles_df is large, but for now ok.
# Actually we can't cache `articles_df` in arguments easily. We pass it in.
def load_cb_model(model_type, articles_df, embedding_name=None):
    """Load Content-Based Models (TF-IDF, PhoBERT, VN-SBERT, etc.)"""
    try:
        if model_type.lower() == 'tf-idf':
            from src.models.content_based import TFIDFRecommender
            texts = (articles_df['title'].fillna('') + " " + articles_df['short_description'].fillna('')).tolist()
            model = TFIDFRecommender(1, len(articles_df))
            model.encode_articles(texts)
            return model
            
        elif model_type.lower() in ['phobert', 'simcse', 'e5', 'hybrid', 'session', 'vn-sbert', 'vietnamese-sbert', 'bge-m3', 'vndoc', 'e5-large', 'e5-base', 'gte', 'gte-multilingual']:
             emb_key = embedding_name if embedding_name else model_type.lower()
             file_map = {
                 'vn-sbert': 'vietnamese-sbert',
                 'vietnamese-sbert': 'vietnamese-sbert',
                 'bge-m3': 'bge-m3',
                 'vndoc': 'vietnamese-document-embedding', 
                 'e5-large': 'e5-large',
                 'e5-base': 'e5-base',
                 'gte': 'gte-multilingual',
                 'gte-multilingual': 'gte-multilingual',
                 'phobert': 'phobert',
                 'simcse': 'simcse',
                 'e5': 'e5-large'
             }
             
             file_prefix = file_map.get(emb_key, emb_key)
             emb_path = Path("checkpoints") / f"{file_prefix}_article_embeddings.pt"
             
             if not emb_path.exists():
                 emb_path = Path(settings.DATA_DIR) / f"{file_prefix}_embeddings.pt"
             
             if emb_path.exists():
                 try:
                     emb_dict = torch.load(emb_path, map_location='cpu', weights_only=False)
                     
                     if isinstance(emb_dict, torch.Tensor):
                         embeddings = emb_dict
                         n_emb = embeddings.shape[0]
                         n_art = len(articles_df)
                         if n_emb != n_art:
                             embed_dim = embeddings.shape[1]
                             new_embeddings = torch.zeros(n_art, embed_dim)
                             copy_size = min(n_emb, n_art)
                             new_embeddings[:copy_size] = embeddings[:copy_size]
                             embeddings = new_embeddings
                     elif isinstance(emb_dict, dict):
                         urls = articles_df['url'].tolist()
                         first_emb = next(iter(emb_dict.values()))
                         embed_dim = first_emb.shape[0]
                         embeddings = torch.zeros(len(urls), embed_dim)
                         for i, url in enumerate(urls):
                             if url in emb_dict:
                                 embeddings[i] = emb_dict[url]
                     
                     if model_type.lower() == 'session':
                         return SessionWrapper(embeddings, articles_df)
                     else:
                         return PhoBERTWrapper(embeddings, articles_df)
                         
                 except Exception as e:
                     print(f"Failed to load {emb_key} embeddings: {e}")
                     return None
             else:
                 print(f"Embeddings {emb_key} not found. Falling back to TF-IDF.")
                 return load_cb_model('tf-idf', articles_df)
                 
        elif model_type.upper() == 'LSA':
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            texts = (articles_df['title'].fillna('') + " " + articles_df['short_description'].fillna('')).tolist()
            tfidf = TfidfVectorizer(max_features=5000, stop_words=None)
            tfidf_matrix = tfidf.fit_transform(texts)
            n_components = min(100, tfidf_matrix.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa_matrix = svd.fit_transform(tfidf_matrix)
            return LSAWrapper(lsa_matrix, articles_df)
        
        return None
    except Exception as e:
        print(f"CB Load Error: {e}")
        return None

def get_recs(model, model_type, user_idx, history_urls, article_map, articles_df, k=10, 
             adj_norm=None, user_priors=None, semantic_ids=None, edge_index=None, score_type="Normalized (0-1)",
             use_adt=False):
    try:
        import torch
        if model_type in MODEL_OPTIONS["CF"]:
            idx_to_url = {v: k for k, v in article_map.items()}
            model.eval()
            with torch.no_grad():
                if model_type.lower() == 'xsimgcl':
                    if adj_norm is not None:
                         u_idx_torch = torch.tensor([user_idx])
                         scores = model.predict(adj_norm, users=u_idx_torch, 
                                               semantic_ids=semantic_ids, 
                                               user_priors=user_priors).squeeze()
                    else:
                         # Fallback for XSimGCL if adj_norm is missing: Dot product of embeddings
                         u_emb = model.user_embedding.weight[user_idx]
                         i_embs = model.item_embedding.weight
                         scores = torch.matmul(u_emb, i_embs.t())

                elif model_type.lower() == 'ma-hcl':
                     if edge_index is not None:
                          scores = model.predict(
                              users=torch.tensor([user_idx]), 
                              items=torch.arange(len(article_map)), 
                              edge_index_dict=edge_index
                          ).squeeze()
                     else:
                          # Fallback: Dot product
                          u_emb = model.user_emb.weight[user_idx]
                          i_embs = model.item_emb.weight
                          scores = torch.matmul(u_emb, i_embs.t())

                elif hasattr(model, 'forward'):
                    model_n_users = None
                    if hasattr(model, 'n_users'):
                        model_n_users = model.n_users
                    elif hasattr(model, 'E_u_0'):
                        model_n_users = model.E_u_0.shape[0]
                    elif hasattr(model, 'user_embedding'):
                        model_n_users = model.user_embedding.weight.shape[0]
                    
                    if model_n_users and user_idx >= model_n_users:
                        # User index out of bounds
                        return []
                    
                    if adj_norm is not None:
                        try:
                            result = model(adj_norm)
                            if isinstance(result, tuple) and len(result) >= 2:
                                user_all, item_all = result[0], result[1]
                            else:
                                raise ValueError("Unexpected return type")
                            scores = torch.mm(user_all[user_idx].unsqueeze(0), item_all.t()).squeeze()
                        except Exception:
                            if hasattr(model, 'user_embedding'):
                                u_emb = model.user_embedding.weight[user_idx]
                                i_embs = model.item_embedding.weight
                            elif hasattr(model, 'E_u_0'):
                                u_emb = model.E_u_0[user_idx]
                                i_embs = model.E_i_0
                            else:
                                return []
                            scores = torch.matmul(u_emb, i_embs.t())
                    else:
                        if hasattr(model, 'user_embedding'):
                            u_emb = model.user_embedding.weight[user_idx]
                            i_embs = model.item_embedding.weight
                        elif hasattr(model, 'E_u_0'):
                            u_emb = model.E_u_0[user_idx]
                            i_embs = model.E_i_0
                        else:
                            return []
                        scores = torch.matmul(u_emb, i_embs.t())

                vals, indices = torch.topk(scores, k=min(k, len(scores)))
                vals_np = vals.cpu().numpy()
                if len(vals_np) > 0:
                    if score_type == "Normalized (0-1)":
                        score_min, score_max = vals_np.min(), vals_np.max()
                        if score_max > score_min:
                            vals_np = (vals_np - score_min) / (score_max - score_min)
                        else:
                            vals_np = np.ones_like(vals_np)
                    elif score_type == "Sigmoid":
                        vals_np = 1 / (1 + np.exp(-vals_np))
                
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
        
        elif model_type in ['PhoBERT', 'SimCSE', 'vn-sbert', 'bge-m3']:
            url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
            hist_indices = [url_to_idx[u] for u in history_urls if u in url_to_idx]
            if not hist_indices: return []
            top, scores = model.recommend(hist_indices, k=k)
            
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
        print(f"Inference Error: {e}")
        return []
