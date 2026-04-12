import numpy as np
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import io
import base64

CATEGORY_MAP = {
    "giaoduc": "Giáo dục",
    "khcn": "Khoa học & Công nghệ",
    "kinhdoanh": "Kinh doanh",
    "thegioi": "Thế giới",
    "thethao": "Thể thao",
    "thoisu": "Thời sự",
    "giai-tri": "Giải trí",
    "suc-khoe": "Sức khỏe",
    "xe": "Xe",
    "du-lich": "Du lịch",
    "N/A": "Khác"
}

def plot_embedding_space(model, user_idx, history_urls, recommended_urls, article_map, articles_df, custom_user_vector=None):
    """
    Vẽ biểu đồ phân tán 2D - Returns JSON for Plotly
    """
    try:

        user_emb = None
        user_source_note = ""
        user_display_name = ""
        
        if custom_user_vector is not None:
            user_emb = custom_user_vector
            user_source_note = "User Vector: Tổng hợp từ sở thích bạn vừa nhập"
            user_display_name = "BẠN (Sở thích mới)"
        else:
            if hasattr(model, 'user_embedding'):
                user_emb = model.user_embedding.weight[user_idx].detach().cpu().numpy()
            elif hasattr(model, 'user_emb'): # For MA-HCL
                user_emb = model.user_emb.weight[user_idx].detach().cpu().numpy()
            elif hasattr(model, 'E_u_0'): 
                user_emb = model.E_u_0[user_idx].detach().cpu().numpy()
            elif hasattr(model, 'gu'):
                user_emb = model.gu.weight[user_idx].detach().cpu().numpy()
                
            user_source_note = f"User Vector: Đã học từ lịch sử đọc (ID: {user_idx})"
            user_display_name = "BẠN (User đã học)"

        if user_emb is None: return None


        item_matrix = None
        if hasattr(model, 'item_embedding'):
            item_matrix = model.item_embedding.weight.detach().cpu().numpy()
        elif hasattr(model, 'item_emb'): # For MA-HCL
            item_matrix = model.item_emb.weight.detach().cpu().numpy()
        elif hasattr(model, 'E_i_0'):
             item_matrix = model.E_i_0.detach().cpu().numpy()
        elif hasattr(model, 'gi'):
             item_matrix = model.gi.weight.detach().cpu().numpy()
        
        if item_matrix is None: return None


        url_to_idx = {u: i for u, i in article_map.items()}
        meta_map = articles_df.set_index('url')[['title', 'source_category', 'short_description']].fillna("").to_dict('index')
        
        vectors = [user_emb]
        names = [f"<b>{user_display_name}</b>"]
        types = ["User"]
        colors = ["#FF4757"] 
        sizes = [40]         
        texts = [user_source_note]
        
        valid_hist = [u for u in history_urls if u in url_to_idx][-20:]
        for u in valid_hist:
            idx = url_to_idx[u]
            info = meta_map.get(u, {})
            title = str(info.get('title', 'Unknown'))
            cat = CATEGORY_MAP.get(info.get('source_category', ''), 'Khác')
            
            vectors.append(item_matrix[idx])
            names.append(f"<b>Đã đọc:</b> {title[:40]}...")
            types.append("History")
            colors.append("#2ED573") 
            sizes.append(15)         
            texts.append(f"Chuyên mục: {cat}")
            
        valid_rec = [u for u in recommended_urls if u in url_to_idx]
        for u in valid_rec:
            idx = url_to_idx[u]
            info = meta_map.get(u, {})
            title = str(info.get('title', 'Unknown'))
            cat = CATEGORY_MAP.get(info.get('source_category', ''), 'Khác')
            
            vectors.append(item_matrix[idx])
            names.append(f"<b>Gợi ý:</b> {title[:40]}...")
            types.append("Recommendation")
            colors.append("#FFA502") 
            sizes.append(20)         
            texts.append(f"Chuyên mục: {cat}")
            
        if len(vectors) < 2: return None


        n_comp = 2 if len(vectors) > 2 else 1
        pca = PCA(n_components=n_comp)
        vectors_2d = pca.fit_transform(np.array(vectors))
        
        if vectors_2d.shape[1] == 1:
            vectors_2d = np.hstack((vectors_2d, np.zeros((vectors_2d.shape[0], 1))))
            
        jitter_strength = 0.02
        noise = np.random.normal(0, jitter_strength, vectors_2d.shape)
        noise[0] = 0
        vectors_2d = vectors_2d + noise


        plot_data = []
        groups = [
            ("User", "#FF4757", user_display_name), 
            ("History", "#2ED573", "Đã đọc"), 
            ("Recommendation", "#FFA502", "Gợi ý")
        ]
        
        for g_type, g_color, g_label in groups:
            mask = [t == g_type for t in types]
            if not any(mask): continue
            
            trace = {
                "x": vectors_2d[np.array(mask), 0].tolist(),
                "y": vectors_2d[np.array(mask), 1].tolist(),
                "mode": 'markers',
                "name": g_label,
                "marker": {
                    "size": [s for s, m in zip(sizes, mask) if m], 
                    "color": g_color,
                    "line": {"width": 2, "color": 'white'},
                    "opacity": 0.9 if g_type == 'User' else 0.7
                },
                "text": [n for n, m in zip(names, mask) if m],
                "hoverinfo": "text",
                "customdata": [t for t, m in zip(texts, mask) if m]
            }
            plot_data.append(trace)
            
        return plot_data

    except Exception as e:
        print(f"Plot Error: {e}")
        return None

def generate_user_wordcloud(history_urls, articles_df):
    """Returns base64 image"""
    if not history_urls: return None
    try:
        hist_df = articles_df[articles_df['url'].isin(history_urls)]
        text = " ".join(hist_df['title'].astype(str).tolist())
        
        wc = WordCloud(
            width=800, height=400, 
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate(text)
        
        img = wc.to_image()
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        return None
