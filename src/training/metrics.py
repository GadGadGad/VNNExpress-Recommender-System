import torch
import numpy as np

# --- FIX ERROR 4: Thêm tham số train_edge_index để lọc lịch sử ---
def evaluate_top_k(model, data, edge_type, k=5, train_edge_index=None):
    """
    Đánh giá mô hình: Precision, Recall, NDCG.
    ĐÃ FIX: Loại bỏ các bài đã xuất hiện trong tập Train (Masking).
    """
    model.eval()
    device = data['user'].x.device # Lấy device hiện tại của data
    
    with torch.no_grad():
        # 1. Lấy Embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)
        user_emb = z_dict['user']
        article_emb = z_dict['article']

        # 2. Chuẩn bị Ground Truth (Tập Test)
        edge_label_index = data[edge_type].edge_label_index
        labels = data[edge_type].edge_label
        pos_edges = edge_label_index[:, labels == 1] # Chỉ lấy cạnh Positive trong Test

        ground_truth = {}
        src_cpu = pos_edges[0].cpu().numpy()
        dst_cpu = pos_edges[1].cpu().numpy()

        for u, a in zip(src_cpu, dst_cpu):
            if u not in ground_truth: ground_truth[u] = set()
            ground_truth[u].add(a)
            
        # --- CHUẨN BỊ MASK TỪ TẬP TRAIN ---
        train_history = {}
        if train_edge_index is not None:
            t_src = train_edge_index[0].cpu().numpy()
            t_dst = train_edge_index[1].cpu().numpy()
            for u, a in zip(t_src, t_dst):
                if u not in train_history: train_history[u] = set()
                train_history[u].add(a)

        # 3. Tính Metrics
        precisions, recalls, ndcgs = [], [], []

        for u_idx in ground_truth.keys():
            true_items = ground_truth[u_idx]

            # Tính điểm với TOÀN BỘ articles
            # Shape: (Num_Articles,)
            scores = (user_emb[u_idx].unsqueeze(0) @ article_emb.T).squeeze()

            # --- FIX LEAKAGE: MASKING TRAIN ITEMS ---
            if u_idx in train_history:
                seen_items = list(train_history[u_idx])
                # Gán điểm cực thấp cho các bài đã xem để không bao giờ lọt vào Top K
                scores[seen_items] = -float('inf')
            # ----------------------------------------

            # Lấy Top K
            actual_k = min(k, len(scores))
            _, top_k_indices = torch.topk(scores, k=actual_k)
            recommendations = top_k_indices.cpu().numpy()

            # Tính điểm số
            hits = 0
            dcg = 0
            idcg = 0

            for rank, item in enumerate(recommendations):
                if item in true_items:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2)

            for i in range(min(len(true_items), actual_k)):
                idcg += 1.0 / np.log2(i + 2)

            precisions.append(hits / actual_k)
            recalls.append(hits / len(true_items))
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)