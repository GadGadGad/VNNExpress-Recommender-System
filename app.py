
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

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# CONFIG & UTILS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="News RecSys Comprehensive",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
MODELS_DIR = "models"

MODEL_OPTIONS = {
    "CF": ["NGCF", "LightGCL", "SimGCL"], 
    "CB": ["TF-IDF", "PhoBERT", "Hybrid"]
}

@st.cache_data
def load_resources(data_dir=DATA_DIR, raw_dir=RAW_DIR):
    """Load static data: Articles, Mappings"""
    # 1. Articles
    articles_path = Path(raw_dir) / "articles.csv"
    if articles_path.exists():
        articles_df = pd.read_csv(articles_path)
        articles_df['title'] = articles_df['title'].fillna("Untitled")
        articles_df['short_description'] = articles_df['short_description'].fillna("")
        # Create Map: URL -> Meta
        url_to_idx = {url: i for i, url in enumerate(articles_df['url'])}
        meta_map = articles_df.set_index('url')[['title', 'short_description']].to_dict('index')
    else:
        # Fallback if raw_dir doesn't have articles (e.g. processed context)
        # Try checking parent or default
        st.error(f"Missing articles.csv in {raw_dir}")
        return None, None, None, None

    # 2. Mappings (from converted data) - Used for CF
    u_map_path = Path(data_dir) / "user_map.json"
    a_map_path = Path(data_dir) / "article_map.json"
    
    user_map_cf = {}
    article_map_cf = {}
    
    if u_map_path.exists():
        with open(u_map_path) as f: user_map_cf = json.load(f)
    if a_map_path.exists():
        with open(a_map_path) as f: article_map_cf = json.load(f)

    # 3. Interactions (Raw) -> User History
    replies_path = Path(RAW_DIR) / "replies.csv"
    user_history = {}
    if replies_path.exists():
        try:
            df_rep = pd.read_csv(replies_path)
            # Clean User IDs
            def clean(x):
                try: return str(int(float(x)))
                except: return str(x)
            col = 'user_id' if 'user_id' in df_rep.columns else 'reply_user_id'
            df_rep[col] = df_rep[col].apply(clean)
            user_history = df_rep.groupby(col)['article_url'].apply(list).to_dict()
        except Exception as e:
            st.warning(f"Error loading User History: {e}")
    
    return articles_df, user_map_cf, article_map_cf, user_history

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
        else:
            return None

        # Find latest checkpoint
        files = glob.glob(f"{MODELS_DIR}/{model_name.lower()}_*.pt")
        if not files: return None
        latest = max(files, key=os.path.getctime)
        
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        config = checkpoint.get('config', {})
        state_dict = checkpoint['model_state_dict']
        
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
        
        if model_name.lower() == 'ngcf':
             model = ModelClass(n_users, n_items, emb_dim=dim, layers=layers)
        elif model_name.lower() == 'lightgcl':
             # LightGCL: n_users, n_items, embed_dim, n_layers
             n_l = len(layers) if isinstance(layers, list) else layers
             model = ModelClass(n_users, n_items, embed_dim=dim, n_layers=n_l)
        elif model_name.lower() == 'simgcl':
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
             # Check if saved embeddings exist?
             # For demo, mapping raw text to PhoBERT in real-time is slow (CPU).
             # We might check if `article_embeddings.pt` exists.
             # Fallback to TF-IDF if too heavy?
             # Let's try to load if `run_content_based.py` saved something.
             # Usually `run_content_based.py` saves `content_model.pt`?
             # If not, let's skip or show warning.
             st.info("PhoBERT inference requires pre-computed embeddings (heavy). Using dummy standard.")
             return None
             
        return None
    except Exception as e:
        st.error(f"CB Load Error: {e}")
        return None

def get_recs(model, model_type, user_idx, history_urls, article_map, articles_df, k=10):
    try:
        import torch
        if model_type in MODEL_OPTIONS["CF"]:
            # CF Logical
            idx_to_url = {v: k for k, v in article_map.items()}
            with torch.no_grad():
                # Extract embeddings depends on model structure
                # NGCF: user_embedding, item_embedding
                # LightGCL: embedding_user, embedding_item
                # SimGCL: user_emb, item_emb (via forward)
                
                # Unified forward if possible, or manual extraction
                if hasattr(model, 'user_embedding'): # NGCF
                    u_emb = model.user_embedding.weight[user_idx]
                    i_embs = model.item_embedding.weight
                elif hasattr(model, 'embedding_user'): # LightGCL
                    u_emb = model.embedding_user.weight[user_idx]
                    i_embs = model.embedding_item.weight
                else: # Generic/SimGCL (might need edge_index)
                    # SimGCL often needs graph for propagation
                    # If we don't have graph here, we use base embeddings (Matrix Factorization style)
                    # or fail.
                    return []
                
                scores = torch.matmul(u_emb, i_embs.t())
                vals, indices = torch.topk(scores, k=min(k, len(i_embs)))
                
            recs = []
            for idx, score in zip(indices.numpy(), vals.numpy()):
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
            
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return []

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
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
            if 'category' in df_g.columns:
                cats = sorted(df_g['category'].dropna().unique().tolist())
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
                    df_a_sub = df_a[df_a['category'] == selected_ds]
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
    articles_df, user_map_cf, article_map_cf, user_history = load_resources(
        st.session_state['data_dir'], 
        st.session_state['raw_dir']
    )
    
    if articles_df is None:
        st.sidebar.error("Data Load Failed. Check paths.")
        return
    
    # Sidebar Controls
    st.sidebar.divider()
    st.sidebar.header("🕹️ Controls")
    
    # User
    user_ids = sorted(list(user_history.keys()))
    default_u = 0
    selected_user = st.sidebar.selectbox("Select User", user_ids, index=default_u)
    
    # Tabs
    tab_infer, tab_train, tab_stats, tab_graph = st.tabs(["🔮 Inference", "🛠️ Training", "📊 Stats", "🕸️ Graph"])
    
    # --- TAB 1: INFERENCE ---
    with tab_infer:
        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            st.subheader("Config")
            # Model Selection
            model_cat = st.radio("Category", ["CF (Collaborative)", "CB (Content-Based)"])
            if "CF" in model_cat:
                model_name = st.selectbox("Model", MODEL_OPTIONS["CF"])
            else:
                model_name = st.selectbox("Model", MODEL_OPTIONS["CB"])
                
            k = st.slider("Top K", 5, 20, 10)
            
            # Show History
            st.divider()
            st.write(f"**History ({len(user_history.get(selected_user, []))})**")
            url_to_meta = articles_df.set_index('url')[['title', 'short_description']].to_dict('index')
            for u in user_history.get(selected_user, [])[-5:]:
                st.caption(f"- {url_to_meta.get(u, {}).get('title', 'Unknown')}")
                
        with col_R:
            st.subheader(f"Results: {model_name}")
            
            # RERANKING CONTROLS
            # Only show if not Hybrid (Hybrid already mixes)
            use_rerank = False
            if model_name != "Hybrid":
                with st.expander("🚀 Reranking Options (MMR)", expanded=False):
                    use_rerank = st.checkbox("Enable Diversity Reranking", value=False)
                    lambda_mmr = st.slider("Diversity Factor (Lambda)", 0.0, 1.0, 0.7, help="Higher = More Relevant, Lower = More Diverse")
                    
            recs = []
            
            # Routing
            if model_name in MODEL_OPTIONS["CF"]:
                if selected_user in user_map_cf:
                     model = load_cf_model(model_name, len(user_map_cf), len(article_map_cf))
                     if model:
                         recs = get_recs(model, model_name, user_map_cf[selected_user], [], article_map_cf, articles_df, k)
                     else: st.warning(f"Model {model_name} not found. Please train it.")
                else:
                    st.info(f"❄️ Cold Start Detected (User not in CF training set).")
                    st.warning(f"Falling back to Content-Based (TF-IDF) for model: {model_name}")
                    model = load_cb_model('TF-IDF', articles_df)
                    recs = get_recs(model, 'TF-IDF', 0, user_history.get(selected_user, []), {}, articles_df, k)
                    
            elif model_name == 'TF-IDF':
                model = load_cb_model('TF-IDF', articles_df)
                recs = get_recs(model, 'TF-IDF', 0, user_history.get(selected_user, []), {}, articles_df, k)
                
            elif model_name == 'Hybrid':
                # Quick Hybrid reuse
                st.info("Hybrid = Interleaving CF (NGCF) + CB (TF-IDF)")
                # Fetch both
                cf_model = load_cf_model('NGCF', len(user_map_cf), len(article_map_cf))
                cb_model = load_cb_model('TF-IDF', articles_df)
                
                cf_recs = []
                cb_recs = []
                
                if cf_model and selected_user in user_map_cf:
                    cf_recs = get_recs(cf_model, 'NGCF', user_map_cf[selected_user], [], article_map_cf, articles_df, k)
                if cb_model:
                     cb_recs = get_recs(cb_model, 'TF-IDF', 0, user_history.get(selected_user, []), {}, articles_df, k)
                
                # Interleave
                seen = set()
                for i in range(max(len(cf_recs), len(cb_recs))):
                    if i < len(cf_recs):
                        u, s = cf_recs[i]
                        if u not in seen: recs.append((u, s)); seen.add(u)
                    if i < len(cb_recs):
                         u, s = cb_recs[i]
                         if u not in seen: recs.append((u, s)); seen.add(u)
                    if len(recs) >= k: break
                     
            # APPLY RERANKING
            if use_rerank and len(recs) > 1:
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    st.info(f"Reranking {len(recs)} items with MMR (Lambda={lambda_mmr})...")
                    
                    # 1. Get Text for candidates
                    cand_urls = [r[0] for r in recs]
                    cand_scores = [r[1] for r in recs]
                    
                    # Lookup text
                    meta_list = [url_to_meta.get(u, {'title': '', 'short_description': ''}) for u in cand_urls]
                    corpus = [str(m['title']) + " " + str(m['short_description']) for m in meta_list]
                    
                    # 2. Compute Similarity Matrix (TF-IDF on the fly for small K is fast)
                    vectorizer = TfidfVectorizer().fit_transform(corpus)
                    sim_matrix = cosine_similarity(vectorizer)
                    
                    # 3. MMR Greedy
                    S = [] # Selected indices
                    R = list(range(len(recs))) # Candidate indices
                    
                    # Select 1st item (highest score)
                    S.append(R.pop(0))
                    
                    while len(S) < min(k, len(recs)) and len(R) > 0:
                        best_mmr = -float('inf')
                        best_cand_idx_in_R = -1
                        
                        for i, r_idx in enumerate(R):
                            # Relevance
                            relevance = cand_scores[r_idx]
                             # Normalize Score? 
                            # If scores are small (0.01), they might be outweighed by Sim (0.5).
                            # Normalize scores to 0-1 range locally?
                            # Simplify: relevance 
                            
                            # Diversity Penalty: Max sim with already selected
                            # Sim is 0..1
                            max_sim = max([sim_matrix[r_idx, s_idx] for s_idx in S])
                            
                            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * max_sim
                            
                            if mmr_score > best_mmr:
                                best_mmr = mmr_score
                                best_cand_idx_in_R = i
                        
                        if best_cand_idx_in_R != -1:
                            S.append(R.pop(best_cand_idx_in_R))
                            
                    # Reorder recs
                    recs = [recs[i] for i in S]
                    
                except Exception as e:
                    st.warning(f"Reranking failed: {e}")
                    
            # Render
            for i, (url, score) in enumerate(recs):
                meta = url_to_meta.get(url, {'title': 'Unknown', 'short_description': ''})
                with st.expander(f"#{i+1} {meta['title']} ({score:.3f})"):
                    st.write(meta['short_description'])
                    st.markdown(f"[Read]({url})")

    # --- TAB 2: TRAINING ---
    with tab_train:
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
            use_all_data = st.checkbox("Train on Full Data (Demo)", value=False, help="Use ALL data for training (No Test Split). Solves Cold Start for existing users.")
            
        start = st.button("🚀 Start Training Process", type="primary")
        
        if start:
            st.info(f"Launching training for {train_model_name}...")
            log_area = st.empty()
            
            # Build Command
            if train_model_name in MODEL_OPTIONS["CF"]:
                script = "scripts/train_cf_models.py"
                cmd = [sys.executable, script, "--model", train_model_name.lower(), "--epochs", str(epochs), "--no-cache"]
                if use_all_data:
                    cmd.append("--use-all-data")
            else:
                script = "scripts/run_content_based.py"
                cmd = [sys.executable, script, "--model", train_model_name.lower().replace("-",""), "--epochs", str(epochs)]
            
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

    # --- TAB 3: STATS ---
    with tab_stats:
        st.metric("Users", len(user_history))
        st.metric("Articles", len(articles_df))
        # Add Comparison results
        st.divider()
        files = glob.glob(f"{MODELS_DIR}/comparison_results_*.json")
        if files:
            latest = max(files, key=os.path.getctime)
            with open(latest) as f: st.dataframe(json.load(f))
            
    # --- TAB 4: GRAPH ---
    with tab_graph:
        st.header("🕸️ Graph Inspector")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📂 Available Graphs (Current Context)")
            # list files in data_dir
            curr_dir = Path(st.session_state.get('data_dir', DATA_DIR))
            graph_files = list(curr_dir.glob("*.pt"))
            if graph_files:
                for gf in graph_files:
                    st.code(f"{gf.name} ({gf.stat().st_size / 1024:.1f} KB)")
            else:
                st.warning("No .pt graph files found.")
                
        with c2:
            st.subheader("🛠️ Possible Graph Types")
            st.write("We can construct the following from Raw Data:")
            st.markdown("""
            - **User-Article Bipartite**: Basic Interaction Graph (Comments).
            - **User-User**: Shared Interest Graph (Users commenting on same articles).
            - **Article-Article**: Content Similarity (KNN) or Co-occurrences.
            - **Heterogeneous**: Combination of all above nodes and edges.
            """)
            st.info("To generate these, use 'convert_to_gnn.py' or Pipeline.")

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
                               st.error("User ID not found in mapping.")
                          else:
                               u_idx = u_map[target_user_id]
                               st.write(f"Visualizing User Index: {u_idx}")

                               # Build NX
                               G = nx.Graph()
                               G.add_node("User", color='red', label=f'User {target_user_id}', size=20)
                               
                               # 1. User -> Article Edges
                               # Edge index for ('user', 'comments', 'article') or ('user', 'writes', 'comment') -> ('comment', 'on', 'article')?
                               # convert_to_gnn saves ('user', 'comments', 'article') directly if direct_edge=True?
                               # Standard: ('user', 'comments', 'article')
                               try:
                                   if ('user', 'comments', 'article') in data.edge_index_dict:
                                       edge_index = data['user', 'comments', 'article'].edge_index
                                   else:
                                       # Maybe comments?
                                       edge_index = data['user', 'writes', 'comment'].edge_index # Indirect?
                                       # We need User-Article.
                                       # Let's assume User-Article exist.
                                       st.warning("Direct User-Article edges not found. Checking keys...")
                                       st.write(data.edge_types)
                                       edge_index = None

                                   if edge_index is not None:
                                       mask = edge_index[0] == u_idx
                                       connected_articles = edge_index[1][mask].tolist()
                                       
                                       st.write(f"Read {len(connected_articles)} articles.")
                                       
                                       added_articles = set()
                                       for aid in connected_articles:
                                            url = inv_a_map.get(aid, "Unknown")
                                            title = url_to_title.get(url, f"Article {aid}")[:30]
                                            G.add_node(aid, color='blue', label=title, size=10)
                                            G.add_edge("User", aid, color='black')
                                            added_articles.add(aid)
                                            
                                       # 2. Article -> Similar Article (KNN)
                                       if ('article', 'similar_to', 'article') in data.edge_index_dict:
                                            sim_index = data['article', 'similar_to', 'article'].edge_index
                                            src_sim = sim_index[0]
                                            # Find neighbors of added_articles
                                            # Uses tensor mask
                                            mask_sim = torch.isin(src_sim, torch.tensor(list(added_articles)))
                                            
                                            s_nodes = src_sim[mask_sim].tolist()
                                            d_nodes = sim_index[1][mask_sim].tolist()
                                            
                                            count_sim = 0
                                            for s, d in zip(s_nodes, d_nodes):
                                                if count_sim > 50: break # Limit
                                                if d not in G.nodes:
                                                     url = inv_a_map.get(d, "Unknown")
                                                     title = url_to_title.get(url, f"Sim {d}")[:30]
                                                     G.add_node(d, color='green', label=title, size=8)
                                                G.add_edge(s, d, color='gray', style='dashed')
                                                count_sim += 1
                                            st.write(f"Added {count_sim} similarity edges.")
                                   
                                   # Plot
                                   fig, ax = plt.subplots(figsize=(10, 8))
                                   pos = nx.spring_layout(G, k=0.3)
                                   
                                   # Colors
                                   colors = [G.nodes[n]['color'] for n in G.nodes()]
                                   sizes = [G.nodes[n]['size']*30 for n in G.nodes()]
                                   labels = nx.get_node_attributes(G, 'label')
                                   
                                   nx.draw(G, pos, ax=ax, node_color=colors, node_size=sizes, with_labels=False)
                                   nx.draw_networkx_labels(G, pos, labels, font_size=8)
                                   
                                   st.pyplot(fig)
                                   st.info("Red: User | Blue: Read History | Green: Similar Articles (KNN)")
                                   
                               except Exception as e:
                                   st.error(f"Error drawing graph: {e}")
if __name__ == "__main__":
    main()
