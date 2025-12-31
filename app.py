
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
from rank_bm25 import BM25Plus
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

@st.cache_resource
def build_search_engine(df):
    """
    Xây dựng chỉ mục BM25+ từ dữ liệu bài báo.
    """
    corpus = (df['title'].fillna("") + " " + df['short_description'].fillna("")).astype(str).tolist()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Plus(tokenized_corpus)
    return bm25

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

def plot_embedding_space(model, user_idx, history_urls, recommended_urls, article_map, articles_df, custom_user_vector=None):
    """
    Vẽ biểu đồ phân tán 2D (Phiên bản Final - Visual Pro - Fix Legend Color):
    - Chống đè: Tăng khoảng cách, thêm jitter.
    - Rõ ràng nguồn gốc User.
    - Hover Label tương phản cao.
    - Legend luôn hiển thị rõ (nền trắng, chữ đen).
    """
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    import numpy as np
    import streamlit as st
    
    try:
        # --- 1. XÁC ĐỊNH VECTOR USER (NGUỒN GỐC) ---
        user_emb = None
        user_source_note = ""
        
        if custom_user_vector is not None:
            # Trường hợp 1: User Vector tự tạo (Cold Start / Custom)
            user_emb = custom_user_vector
            user_source_note = "User Vector: Tổng hợp từ sở thích bạn vừa nhập"
            user_display_name = "👤 BẠN (Sở thích mới)"
        else:
            # Trường hợp 2: User Vector từ Model (Đã học)
            if hasattr(model, 'user_embedding'):
                user_emb = model.user_embedding.weight[user_idx].detach().cpu().numpy()
            elif hasattr(model, 'E_u_0'): 
                user_emb = model.E_u_0[user_idx].detach().cpu().numpy()
            elif hasattr(model, 'gu'):
                user_emb = model.gu.weight[user_idx].detach().cpu().numpy()
            
            user_source_note = f"User Vector: Đã học từ lịch sử đọc (ID: {user_idx})"
            user_display_name = "👤 BẠN (User đã học)"

        if user_emb is None: return None

        # --- 2. LẤY VECTOR ITEM ---
        item_matrix = None
        if hasattr(model, 'item_embedding'):
            item_matrix = model.item_embedding.weight.detach().cpu().numpy()
        elif hasattr(model, 'E_i_0'):
             item_matrix = model.E_i_0.detach().cpu().numpy()
        elif hasattr(model, 'gi'):
             item_matrix = model.gi.weight.detach().cpu().numpy()
        
        if item_matrix is None: return None

        # --- 3. CHUẨN BỊ DỮ LIỆU ---
        url_to_idx = {u: i for u, i in article_map.items()}
        meta_map = articles_df.set_index('url')[['title', 'source_category', 'short_description']].fillna("").to_dict('index')
        
        vectors = [user_emb]
        names = [f"<b>{user_display_name}</b>"]
        types = ["User"]
        colors = ["#FF4757"] # Đỏ rực rỡ
        sizes = [40]         # Rất to
        texts = [user_source_note]
        
        # Thêm History
        valid_hist = [u for u in history_urls if u in url_to_idx][-20:] # Lấy 20 bài
        for u in valid_hist:
            idx = url_to_idx[u]
            info = meta_map.get(u, {})
            title = str(info.get('title', 'Unknown'))
            cat = CATEGORY_MAP.get(info.get('source_category', ''), 'Khác')
            
            vectors.append(item_matrix[idx])
            names.append(f"<b>Đã đọc:</b> {title[:40]}...")
            types.append("History")
            colors.append("#2ED573") # Xanh lá neon
            sizes.append(15)         
            texts.append(f"Chuyên mục: {cat}")
            
        # Thêm Recommendations
        valid_rec = [u for u in recommended_urls if u in url_to_idx]
        for u in valid_rec:
            idx = url_to_idx[u]
            info = meta_map.get(u, {})
            title = str(info.get('title', 'Unknown'))
            cat = CATEGORY_MAP.get(info.get('source_category', ''), 'Khác')
            
            vectors.append(item_matrix[idx])
            names.append(f"<b>✨ Gợi ý:</b> {title[:40]}...")
            types.append("Recommendation")
            colors.append("#FFA502") # Vàng cam đậm
            sizes.append(20)         
            texts.append(f"Chuyên mục: {cat}")
            
        if len(vectors) < 2: return None

        # --- 4. PCA & JITTER (CHỐNG ĐÈ) ---
        n_comp = 2 if len(vectors) > 2 else 1
        pca = PCA(n_components=n_comp)
        vectors_2d = pca.fit_transform(np.array(vectors))
        
        if vectors_2d.shape[1] == 1:
            vectors_2d = np.hstack((vectors_2d, np.zeros((vectors_2d.shape[0], 1))))
            
        # Thêm Jitter (Nhiễu ngẫu nhiên nhỏ) để tách các điểm trùng nhau
        # Chỉ thêm vào các điểm không phải User (để User đứng yên chuẩn xác)
        jitter_strength = 0.02 # Điều chỉnh độ mạnh của nhiễu
        noise = np.random.normal(0, jitter_strength, vectors_2d.shape)
        noise[0] = 0 # Giữ nguyên vị trí User
        vectors_2d = vectors_2d + noise

        # --- 5. VẼ BIỂU ĐỒ ---
        fig = go.Figure()
        
        groups = [
            ("User", "#FF4757", user_display_name), 
            ("History", "#2ED573", "📚 Đã đọc"), 
            ("Recommendation", "#FFA502", "✨ Gợi ý")
        ]
        
        for g_type, g_color, g_label in groups:
            mask = [t == g_type for t in types]
            if not any(mask): continue
            
            fig.add_trace(go.Scatter(
                x=vectors_2d[mask, 0],
                y=vectors_2d[mask, 1],
                mode='markers', # Bỏ text mặc định để đỡ rối, chỉ hiện khi hover
                name=g_label,
                marker=dict(
                    size=[s for s, m in zip(sizes, mask) if m], 
                    color=g_color,
                    line=dict(width=2, color='white'), # Viền trắng cho nổi trên nền màu
                    opacity=0.9 if g_type == 'User' else 0.7 # Item trong suốt hơn để thấy điểm chồng
                ),
                text=[n for n, m in zip(names, mask) if m], # Text dùng cho hover
                # Tùy chỉnh Hover Label (Tooltip)
                hoverlabel=dict(
                    bgcolor="white",          # Nền trắng hoàn toàn
                    font_size=14,             # Chữ to vừa phải
                    font_family="Arial",
                    font_color="#333333",     # Chữ đen đậm
                    bordercolor=g_color       # Viền tooltip theo màu nhóm
                ),
                hovertemplate="<b>%{text}</b><br><br>%{customdata}<extra></extra>",
                customdata=[t for t, m in zip(texts, mask) if m]
            ))

        # Cấu hình Layout Thoáng đãng
        fig.update_layout(
            title=dict(
                text=f"🗺️ BẢN ĐỒ SỞ THÍCH & GỢI Ý<br><sup style='color:#555; font-size:12px'>Vị trí User được tính toán từ: {user_source_note}</sup>",
                y=0.95, x=0.0, xanchor='left', yanchor='top', # Đưa title sang trái
                font=dict(size=22, color='#1E272E')
            ),
            xaxis=dict(showgrid=True, gridcolor='#F1F2F6', zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='#F1F2F6', zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=600, # Tăng chiều cao để tách điểm
            margin=dict(l=20, r=20, t=80, b=20),
            
            # --- SỬA CHỖ NÀY: CẤU HÌNH LEGEND CỐ ĐỊNH MÀU ---
            legend=dict(
                yanchor="top", y=1, xanchor="right", x=1.1, # Đưa Legend ra ngoài bên phải
                bgcolor="#FFFFFF",                  # Nền Trắng tuyệt đối (Không trong suốt)
                bordercolor="#E0E0E0",              # Viền xám nhạt
                borderwidth=1,
                font=dict(size=14, color="#000000") # Chữ Đen tuyệt đối (Bất chấp Dark Mode)
            )
        )
        
        return fig

    except Exception as e:
        return None
def generate_user_wordcloud(history_urls, articles_df):
    """Tạo WordCloud từ tiêu đề lịch sử đọc"""
    if not history_urls: return None
    try:
        # Lấy text từ tiêu đề các bài đã đọc
        hist_df = articles_df[articles_df['url'].isin(history_urls)]
        text = " ".join(hist_df['title'].astype(str).tolist())
        
        # Config WordCloud
        wc = WordCloud(
            width=400, height=200, 
            background_color='white',
            colormap='viridis',
            max_words=50,
            font_path=None # Có thể thêm font tiếng Việt nếu cần
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        return None
def explain_recommendation(rec_url, history_urls, articles_df, source_type):
    """
    Tạo câu giải thích lý do gợi ý dựa trên Content hoặc Social.
    Sử dụng so sánh chuỗi đơn giản (Jaccard) để tìm bài tương đồng trong lịch sử.
    """
    try:
        # 1. Nếu là nguồn Social (CF), giải thích theo đám đông
        if source_type == "Social":
            return "👥 <b>Cộng đồng:</b> Phổ biến với những người dùng có gu giống bạn."

        # 2. Nếu là Content/Hybrid, tìm bài tương tự nhất trong lịch sử
        rec_row = articles_df[articles_df['url'] == rec_url]
        if rec_row.empty: return ""
        rec_row = rec_row.iloc[0]
        rec_tokens = set(str(rec_row['title']).lower().split())
        rec_cat = rec_row.get('source_category', '')

        best_match_title = None
        max_score = 0
        
        # Chỉ so sánh với 20 bài gần nhất để nhanh
        recent_history = history_urls[-20:]
        hist_rows = articles_df[articles_df['url'].isin(recent_history)]
        
        for _, h_row in hist_rows.iterrows():
            if h_row['url'] == rec_url: continue
            
            # Tính độ trùng lặp từ khóa (Jaccard Similarity đơn giản)
            h_tokens = set(str(h_row['title']).lower().split())
            intersection = len(rec_tokens & h_tokens)
            union = len(rec_tokens | h_tokens)
            score = intersection / union if union > 0 else 0
            
            if score > max_score:
                max_score = score
                best_match_title = h_row['title']
        
        # Ngưỡng chọn giải thích
        if max_score > 0.1: # Nếu trùng lặp từ khóa kha khá
            return f"💡 <b>Nội dung:</b> Tương tự bài <i>'{str(best_match_title)[:30]}...'</i> bạn đã đọc."
        elif rec_cat: # Nếu không trùng từ khóa, giải thích theo Category
            cat_name = CATEGORY_MAP.get(rec_cat, rec_cat)
            return f"📂 <b>Chủ đề:</b> Thuộc danh mục <i>{cat_name}</i> mà bạn quan tâm."
        else:
            return "✨ <b>Gợi ý:</b> Có thể bạn sẽ thích bài này."
            
    except Exception:
        return ""

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


    if u_map_path.exists():
        with open(u_map_path) as f: user_map_cf = json.load(f)
    if a_map_path.exists():
        with open(a_map_path) as f: article_map_cf = json.load(f)

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
    st.set_page_config(page_title="News RecSys Portal", page_icon="📰", layout="wide")
    
    if 'open_url' in st.session_state and st.session_state['open_url']:
        url_to_open = st.session_state['open_url']
        st.session_state['open_url'] = None
        st.components.v1.html(f"""<script>window.open("{url_to_open}", "_blank");</script>""", height=0)

    st.markdown("""
    <style>
        .block-container {padding-top: 1rem;}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 1.1rem; font-weight: bold;}
        
        div[data-testid="stButton"] > button[kind="secondary"] {
            border: none !important;
            background: none !important;
            padding: 0 !important;
            text-align: left !important;
            box-shadow: none !important;
            height: auto !important;
            white-space: normal !important;
            line-height: 1.3 !important;
            font-family: 'Source Sans Pro', sans-serif !important;
        }
        
        /* Chỉnh font size cho text bên trong */
        div[data-testid="stButton"] > button[kind="secondary"] p {
            font-size: 1.4em !important; /* To và rõ */
            font-weight: 700 !important;  /* Đậm */
            padding-top: 5px !important;
        }

        /* Hover vẫn giữ gạch chân */
        div[data-testid="stButton"] > button[kind="secondary"]:hover p {
            text-decoration: underline !important;
        }

        /* Nút Bình luận (Primary) */
        div.stButton > button[kind="primary"] {
            width: 100%; border-radius: 8px; background-color: #f0f2f6; 
            color: #31333F; border: 1px solid #d2d2d2;
        }
        div.stButton > button[kind="primary"]:hover {border-color: #28a745; color: #28a745;}
        
        .source-badge {font-size: 0.75em; padding: 2px 6px; border-radius: 4px; color: white;}
        a.article-link {color: #2c3e50 !important; text-decoration: none; font-weight: bold; font-size: 0.95em;}
        a.article-link:visited {color: #663399 !important;}
    </style>
    """, unsafe_allow_html=True)

    # 2. INIT STATE
    if 'data_dir' not in st.session_state: st.session_state['data_dir'] = DATA_DIR
    if 'session_interactions' not in st.session_state: st.session_state['session_interactions'] = []
    if 'viewed_posts' not in st.session_state: st.session_state['viewed_posts'] = set()
    if 'user_mode' not in st.session_state: st.session_state['user_mode'] = 'existing'
    if 'guest_profile' not in st.session_state: st.session_state['guest_profile'] = None
    if 'last_selected_user' not in st.session_state: st.session_state['last_selected_user'] = None
    if 'open_url' not in st.session_state: st.session_state['open_url'] = None

    st.sidebar.title("🎛️ Control Panel")
    mode_select = st.sidebar.radio("Chế độ:", ["🔑 Thành viên", "🆕 Khách (Cold Start)"], index=0 if st.session_state['user_mode'] == 'existing' else 1)
    
    selected_variant = "strict_g2"
    active_data_dir = os.path.join(DATA_DIR, selected_variant) if os.path.exists(os.path.join(DATA_DIR, selected_variant)) else DATA_DIR
    res = load_resources(data_dir=active_data_dir, raw_dir=RAW_DIR, specific_graph_path=os.path.join(active_data_dir, "cf_cache.pt"))
    articles_df, user_map_cf, article_map_cf, user_history, adj_norm, user_priors, status = res
    
    search_engine = build_search_engine(articles_df)
    
    selected_user = None
    is_cold_start_user = False
    
    if mode_select == "🔑 Thành viên":
        st.session_state['user_mode'] = 'existing'
        existing_users = sorted(list(user_map_cf.keys()))
        selected_uid = st.sidebar.selectbox("Chọn User ID:", existing_users)
        
        user_idx = user_map_cf[selected_uid]
        
        if selected_user != st.session_state['last_selected_user']:
            st.session_state['session_interactions'] = []
            st.session_state['viewed_posts'] = set()
            st.session_state['last_selected_user'] = selected_user
    else:
        st.session_state['user_mode'] = 'guest'
        is_cold_start_user = True
        selected_user = "GUEST"
        with st.sidebar.expander("📝 Hồ sơ Khách hàng", expanded=True):
            all_cats = sorted(articles_df['source_category'].dropna().unique().tolist())
            selected_cats = st.multiselect("Chủ đề:", [c for c in all_cats if c in CATEGORY_MAP], format_func=lambda x: CATEGORY_MAP.get(x, x))
            if st.button("🚀 Tạo Profile"):
                mask = pd.Series([True] * len(articles_df))
                if selected_cats: mask &= articles_df['source_category'].isin(selected_cats)
                matched = np.flatnonzero(mask)
                if len(matched) > 0:
                    samples = articles_df.iloc[np.random.choice(matched, min(5, len(matched)), replace=False)]['url'].tolist()
                    st.session_state['guest_profile'] = samples
                    st.session_state['session_interactions'] = []
                    st.session_state['viewed_posts'] = set()
                    st.rerun()

    with st.sidebar.expander("⚙️ Cấu hình Thuật toán", expanded=False):
        cf_model_choice = st.selectbox("Model CF:", MODEL_OPTIONS["CF"])
        alpha = st.slider("Trọng số Social (Alpha)", 0.0, 1.0, 0.5, 0.1)
        k_rec = st.slider("Số lượng hiển thị", 5, 50, 10)
    
    cf_model = load_cf_model(cf_model_choice, len(user_map_cf), len(article_map_cf), graph_name=selected_variant)
    cb_model = load_cb_model("vn-sbert", articles_df) 

    st.title("🗞️ Personalized News Engine")
    
    tab_feed, tab_analytics = st.tabs(["📰 News Feed & Tìm kiếm", "🛠️ Dev & Analytics"])

    with tab_feed:
        base_history = st.session_state.get('guest_profile', []) if is_cold_start_user else user_history.get(selected_user, [])
        if not base_history and is_cold_start_user:
            st.info("👈 Vui lòng tạo hồ sơ khách hàng ở cột bên trái!")
            st.stop()
            
        full_history = list(base_history) + list(st.session_state['session_interactions'])
        
        # Search UI
        with st.container():
            col_search, col_filter = st.columns([3, 1])
            with col_search: search_query = st.text_input("🔍 Tìm kiếm bài viết:", placeholder="Nhập từ khóa...")
            with col_filter:
                unique_cats = sorted(articles_df['source_category'].dropna().unique().tolist())
                filter_cats = st.multiselect("🏷️ Lọc danh mục:", unique_cats, format_func=lambda x: CATEGORY_MAP.get(x, x))
        st.divider()

        col_content, col_info = st.columns([0.75, 0.25])
        
        # LOG UI
        with col_info:
            st.markdown("### 📊 Log Hoạt động")
            with st.container(border=True):
                if st.session_state['session_interactions']:
                    st.write("💬 **Vừa bình luận:**")
                    for u in reversed(st.session_state['session_interactions']):
                        row = articles_df[articles_df['url'] == u]
                        if not row.empty:
                            title = row.iloc[0]['title']
                            raw_t = str(row.iloc[0].get('published_at', ''))
                            if raw_t == 'nan': raw_t = ""
                            if raw_t and raw_t[0].isdigit(): pub_time = raw_t.split('T')[0] 
                            else: pub_time = raw_t
                            st.markdown(f"<div style='line-height:1.2;'><a href='{u}' target='_blank' class='article-link' style='font-size:0.95em'>{title}</a><div style='color:gray;font-size:0.75em'>📅 {pub_time}</div></div>", unsafe_allow_html=True)
                            st.write("")
                    if st.button("🔄 Reset phiên"):
                        st.session_state['session_interactions'] = []
                        st.session_state['viewed_posts'] = set()
                        st.rerun()
                else: st.caption("Chưa có tương tác mới.")
            st.divider()
            wc_fig = generate_user_wordcloud(full_history, articles_df)
            if wc_fig: st.pyplot(wc_fig, use_container_width=True)

        # CONTENT
        with col_content:
            final_display_list = []
            mode_display = "" 

            # SEARCH
            if search_query.strip():
                mode_display = "search"
                st.markdown(f"### 🔎 Kết quả cho: '{search_query}'")
                tokenized_query = search_query.lower().split()
                doc_scores = search_engine.get_scores(tokenized_query)
                top_indices = np.argsort(doc_scores)[::-1][:100]
                
                count = 0
                for idx in top_indices:
                    if count >= k_rec: break
                    if doc_scores[idx] <= 0: continue
                    row = articles_df.iloc[idx]
                    if filter_cats and row['source_category'] not in filter_cats: continue
                    final_display_list.append((row['url'], doc_scores[idx], "Kết quả tìm kiếm"))
                    count += 1
                if not final_display_list: st.warning(f"Không tìm thấy kết quả.")

            # RECSYS
            else:
                mode_display = "rec"
                st.markdown(f"### 🔥 Tin dành cho {'Khách' if is_cold_start_user else selected_user}")
                
                with st.spinner("Đang tổng hợp tin tức..."):
                    candidate_scores = {}
                    active_alpha = 0.0 if is_cold_start_user else alpha
                    
                    if active_alpha > 0 and cf_model:
                        raw_recs = get_recs(cf_model, cf_model_choice, user_map_cf[selected_user], [], article_map_cf, articles_df, 200, adj_norm=adj_norm)
                        for u, s in raw_recs: candidate_scores[u] = {'score': s * active_alpha, 'source': 'Gợi ý cho bạn'}
                    
                    if (1 - active_alpha) > 0 and cb_model:
                        hist_sample = full_history[-10:]
                        h_indices = [article_map_cf.get(u, -1) if u in article_map_cf else articles_df[articles_df['url']==u].index[0] for u in hist_sample if u in articles_df['url'].values]
                        h_indices = [i for i in h_indices if i >= 0]
                        if h_indices:
                            cb_idx, cb_val = cb_model.recommend(h_indices, k=150)
                            max_v = max(cb_val) if cb_val else 1
                            for idx, val in zip(cb_idx, cb_val):
                                u = articles_df.iloc[idx]['url']
                                norm = (val/max_v) * (1 - active_alpha)
                                curr = candidate_scores.get(u, {'score': 0, 'source': 'Nội dung tương đồng'})
                                candidate_scores[u] = {'score': curr['score'] + norm, 'source': curr['source']}
                    
                    if st.session_state['session_interactions'] and cb_model:
                        last_url = st.session_state['session_interactions'][-1]
                        last_row = articles_df[articles_df['url'] == last_url]
                        if not last_row.empty:
                            sim_idx, sim_val = cb_model.recommend([last_row.index[0]], k=50)
                            max_s = max(sim_val) if sim_val else 1
                            for idx, val in zip(sim_idx, sim_val):
                                u = articles_df.iloc[idx]['url']
                                boost = (val/max_s) * 0.6
                                if u in candidate_scores:
                                    candidate_scores[u]['score'] += boost
                                    if boost > 0.3: candidate_scores[u]['source'] = '⚡ Liên quan bài vừa comment'
                                else:
                                    candidate_scores[u] = {'score': boost, 'source': '⚡ Liên quan bài vừa comment'}

                    temp_recs = []
                    for u, data in candidate_scores.items():
                        if u not in full_history:
                            if filter_cats:
                                row_chk = articles_df[articles_df['url'] == u]
                                if not row_chk.empty and row_chk.iloc[0]['source_category'] not in filter_cats: continue
                            temp_recs.append((u, data['score'], data['source']))
                    temp_recs.sort(key=lambda x: x[1], reverse=True)
                    final_display_list = temp_recs[:k_rec]

            # RENDER LIST
            if not final_display_list: st.info("Chưa có kết quả.")
            else:
                for i, (url, score, src) in enumerate(final_display_list):
                    row = articles_df[articles_df['url'] == url].iloc[0]
                    
                    raw_time = str(row.get('published_at', ''))
                    if raw_time == 'nan': raw_time = ""
                    if raw_time and raw_time[0].isdigit(): pub_time = raw_time.replace('T', ' ').split('.')[0]
                    else: pub_time = raw_time
                    if pub_time == '': pub_time = "Vừa xong"

                    badge_color = "#28a745" if "⚡" in src else ("#17a2b8" if "Tìm kiếm" in src else ("#007bff" if "Gợi ý" in src else "#6c757d"))
                    
                    with st.container():
                        c1, c2 = st.columns([0.85, 0.15])
                        with c1:
                            st.markdown(f"""
                            <div style="margin-bottom: 4px;">
                                <span class="source-badge" style="background-color: {badge_color}">{src}</span>
                                <span style="color: #888; font-size: 0.8em; margin-left: 5px;">🕒 {pub_time}</span> 
                                {f'<span style="color: #bbb; font-size: 0.8em;">• Score: {score:.2f}</span>' if mode_display == 'rec' else ''}
                            </div>""", unsafe_allow_html=True)
                            
                            is_visited = url in st.session_state.get('viewed_posts', set())
                            
                            if is_visited:
                                label = f":violet[{row['title']}]"
                            else:
                                label = f":blue[{row['title']}]"

                            # Nút bấm hiển thị title
                            if st.button(label, key=f"title_{i}_{url}", type="secondary"):
                                st.session_state['viewed_posts'].add(url)
                                st.session_state['open_url'] = url
                                st.rerun()
                            
                            desc = row['short_description'] if not pd.isna(row['short_description']) else "Không có mô tả."
                            st.markdown(f"<div style='color: #555; font-size: 0.95em; margin-top: 2px;'>{desc}</div>", unsafe_allow_html=True)
                            st.caption(f"📂 {CATEGORY_MAP.get(row['source_category'], 'Tin tức')}")

                        with c2:
                            st.write("")
                            st.write("")
                            if st.button("💬 Bình luận", key=f"btn_{i}_{url}", type="primary", help="Tương tác mạnh"):
                                if url in st.session_state.get('viewed_posts', set()):
                                    st.session_state['session_interactions'].append(url)
                                    st.toast("Đã ghi nhận bình luận! Đề xuất đang thay đổi...", icon="✅")
                                    st.rerun()
                                else:
                                    st.toast("⚠️ Mời bạn click vào tiêu đề để đọc bài trước khi bình luận!", icon="🚫")
                        st.divider()

            if not search_query.strip() and not is_cold_start_user and cf_model:
                with st.expander("🗺️ Phân tích Không gian Vector (Visualization)", expanded=False):
                     rec_urls_viz = [x[0] for x in final_display_list[:15]]
                     fig = plot_embedding_space(cf_model, user_map_cf[selected_user], full_history, rec_urls_viz, article_map_cf, articles_df)
                     if fig: st.plotly_chart(fig, use_container_width=True)

    with tab_analytics:
        st.header("⚔️ So sánh & Đánh giá Mô hình")
        col_conf, col_a, col_b = st.columns([1, 1.5, 1.5])
        with col_conf:
            m_a = st.selectbox("Model A", MODEL_OPTIONS["CF"], key="ma")
            m_b = st.selectbox("Model B", MODEL_OPTIONS["CF"], index=1, key="mb")
            if st.button("Chạy so sánh"): st.session_state['run_compare'] = True
        
        if st.session_state.get('run_compare') and not is_cold_start_user:
            with st.spinner("Đang chạy đối sánh..."):
                ma = load_cf_model(m_a, len(user_map_cf), len(article_map_cf))
                mb = load_cf_model(m_b, len(user_map_cf), len(article_map_cf))
                ra = get_recs(ma, m_a, user_map_cf[selected_user], [], article_map_cf, articles_df, 10) if ma else []
                rb = get_recs(mb, m_b, user_map_cf[selected_user], [], article_map_cf, articles_df, 10) if mb else []
                
                with col_a:
                    st.subheader(f"{m_a}")
                    for u, s in ra: st.write(f"- {articles_df[articles_df['url']==u]['title'].values[0][:40]}... ({s:.2f})")
                with col_b:
                    st.subheader(f"{m_b}")
                    for u, s in rb: st.write(f"- {articles_df[articles_df['url']==u]['title'].values[0][:40]}... ({s:.2f})")
                
                overlap = len(set([x[0] for x in ra]) & set([x[0] for x in rb]))
                st.success(f"Độ trùng lặp (Overlap): {overlap}/10 bài")
        elif is_cold_start_user: st.warning("Chức năng so sánh chỉ dành cho User cũ.")

if __name__ == "__main__":
    main()