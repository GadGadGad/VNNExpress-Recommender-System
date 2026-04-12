import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

def load_data(articles_path, replies_path, hidden_dim=64):
    print(f"--- [Dataset] Loading data from {articles_path} & {replies_path} ---")

    # Load CSVs
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)

    # Clean User IDs
    # Loại bỏ dòng lỗi và chuyển đổi user_id sang string chuẩn
    replies = replies[replies['parent_user_id'] != 'NO_COMMENT']

    def clean_id(val):
        try:
            return str(int(float(val)))
        except:
            return str(val)

    replies['user_id'] = replies['parent_user_id'].apply(clean_id)

    # Filter Valid Interactions
    # Chỉ giữ lại comment của những bài báo có trong file articles.csv
    valid_urls = set(articles['url'].unique())
    replies = replies[replies['article_url'].isin(valid_urls)]

    # Map ID to Index (0, 1, 2...)
    unique_users = replies['user_id'].unique()
    unique_articles = articles['url'].unique()

    user_map = {u: i for i, u in enumerate(unique_users)}
    article_map = {a: i for i, a in enumerate(unique_articles)}

    replies['user_idx'] = replies['user_id'].map(user_map)
    replies['article_idx'] = replies['article_url'].map(article_map)

    print(f"Nodes: {len(unique_users)} Users, {len(unique_articles)} Articles")
    print(f"Edges: {len(replies)} Interactions")

    # Construct HeteroData
    data = HeteroData()
    num_users = len(unique_users)
    num_articles = len(unique_articles)


    # User Feature: Random Noise (Learnable Latent Factors)
    # Shape: [Num_Users, 64]. Lightweight, low VRAM usage.
    data['user'].x = torch.randn(num_users, hidden_dim)

    # Article Feature: Random Initialization (no category)
    data['article'].x = torch.randn(num_articles, hidden_dim)


    src = torch.tensor(replies['user_idx'].values, dtype=torch.long)
    dst = torch.tensor(replies['article_idx'].values, dtype=torch.long)

    data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])

    # Biến đổi thành Graph vô hướng (Undirected) để message passing 2 chiều
    data = T.ToUndirected()(data)

    return data, user_map, article_map


def load_processed_data(processed_path='data/processed/graph_data.pt'):
    """
    Load graph đã xử lý từ đĩa thay vì build từ CSV.
    """
    print(f"--- [Dataset] Loading processed graph from {processed_path} ---")

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Không tìm thấy {processed_path}. Hãy chạy src/preprocess.py trước!")

    data = torch.load(processed_path, weights_only=False)
    return data
