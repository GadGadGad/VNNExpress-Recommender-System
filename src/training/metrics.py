import torch
import numpy as np

def evaluate_top_k(model, data, edge_type, k=5):
    """
    Đánh giá mô hình trên tập Test dùng metrics ranking:
    - Precision@K
    - Recall@K
    - NDCG@K
    """
    model.eval()
    with torch.no_grad():
        # 1. Lấy Embeddings cuối cùng từ model
        z_dict = model(data.x_dict, data.edge_index_dict)
        user_emb = z_dict['user']
        article_emb = z_dict['article']

        # 2. Xây dựng Ground Truth từ tập Test (Positive Edges only)
        edge_index = data[edge_type].edge_label_index
        labels = data[edge_type].edge_label

        # Lọc ra các edge có nhãn là 1 (tương tác thật)
        pos_edges = edge_index[:, labels == 1]

        # Dictionary map: user_idx -> set(article_indices)
        ground_truth = {}
        src_cpu = pos_edges[0].cpu().numpy()
        dst_cpu = pos_edges[1].cpu().numpy()

        for u, a in zip(src_cpu, dst_cpu):
            if u not in ground_truth:
                ground_truth[u] = set()
            ground_truth[u].add(a)

        # 3. Tính toán Metric cho từng User trong Test Set
        precisions, recalls, ndcgs = [], [], []

        for u_idx in ground_truth.keys():
            true_items = ground_truth[u_idx]

            # Tính điểm tương đồng (Dot Product) của User này với TẤT CẢ Articles
            # Shape: (1, Out_Dim) @ (Num_Articles, Out_Dim).T -> (1, Num_Articles)
            scores = (user_emb[u_idx].unsqueeze(0) @ article_emb.T).squeeze()

            # Lấy Top K bài có điểm cao nhất
            actual_k = min(k, len(scores))
            _, top_k_indices = torch.topk(scores, k=actual_k)
            recommendations = top_k_indices.cpu().numpy()

            # Tính Metrics
            hits = 0
            dcg = 0
            idcg = 0

            # DCG (Discounted Cumulative Gain)
            for rank, item in enumerate(recommendations):
                if item in true_items:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2)

            # IDCG (Ideal DCG - Giả sử tất cả item đúng đều nằm trên đầu)
            for i in range(min(len(true_items), actual_k)):
                idcg += 1.0 / np.log2(i + 2)

            precisions.append(hits / actual_k)
            recalls.append(hits / len(true_items))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)
