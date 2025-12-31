import pandas as pd
import torch
import torch_geometric.transforms as T
import os
import json
from torch_geometric.data import HeteroData

def process_and_save(articles_path, replies_path, output_dir='data/processed', hidden_dim=64):
    print("--- [Preprocess] Starting Data Pipeline ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load & Clean
    articles = pd.read_csv(articles_path)
    replies = pd.read_csv(replies_path)

    replies = replies[replies['parent_user_id'] != 'NO_COMMENT']
    replies['user_id'] = replies['parent_user_id'].apply(lambda x: str(int(float(x))) if str(x).replace('.','').isdigit() else str(x))

    # Filter valid interactions
    valid_urls = set(articles['url'].unique())
    replies = replies[replies['article_url'].isin(valid_urls)]

    # 2. Map IDs
    unique_users = replies['user_id'].unique()
    unique_articles = articles['url'].unique()

    user_map = {u: i for i, u in enumerate(unique_users)}
    article_map = {a: i for i, a in enumerate(unique_articles)}

    # Lưu map lại để sau này còn biết User ID 0 là ông nào
    with open(f'{output_dir}/user_map.json', 'w') as f: json.dump(user_map, f)
    with open(f'{output_dir}/article_map.json', 'w') as f: json.dump(article_map, f)

    replies['user_idx'] = replies['user_id'].map(user_map)
    replies['article_idx'] = replies['article_url'].map(article_map)

    print(f"Graph Info: {len(unique_users)} Users, {len(unique_articles)} Articles, {len(replies)} Edges")

    # 3. Build Graph
    data = HeteroData()

    # Features
    data['user'].x = torch.randn(len(unique_users), hidden_dim) # Random Init

    # Article Features - Random Initialization (no category)
    data['article'].x = torch.randn(len(unique_articles), hidden_dim)

    # Edges
    src = torch.tensor(replies['user_idx'].values, dtype=torch.long)
    dst = torch.tensor(replies['article_idx'].values, dtype=torch.long)
    data['user', 'comments', 'article'].edge_index = torch.stack([src, dst])

    data = T.ToUndirected()(data)

    # 4. Save
    save_path = f'{output_dir}/graph_data.pt'
    torch.save(data, save_path)
    print(f"Processed graph saved to: {save_path}")

if __name__ == "__main__":
    process_and_save('data/raw/articles.csv', 'data/raw/replies.csv')
