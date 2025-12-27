import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, data, optimizer, epochs, device, edge_type, 
                neg_strategy='random', precomputed_neg_edge_index=None):
    
    print(f"--- [Trainer] Starting training ({epochs} epochs) | Negative Strategy: {neg_strategy.upper()} ---")

    # Progress Bar
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # 1. Forward Pass
        z_dict = model(data.x_dict, data.edge_index_dict)

        # 2. Positive Edges (Tương tác thật trong tập Train)
        pos_idx = data[edge_type].edge_label_index
        user_emb = z_dict['user'][pos_idx[0]]
        article_emb = z_dict['article'][pos_idx[1]]
        
        # Tính điểm Positive
        pos_score = (user_emb * article_emb).sum(dim=-1)

        # 3. Negative Sampling (Xử lý theo Strategy)
        if neg_strategy == 'precomputed' and precomputed_neg_edge_index is not None:
            # --- CÁCH 1: Dùng Hard Negatives đã tính sẵn ---
            # Lưu ý: Số lượng precomputed có thể khác pos_idx, cần slice hoặc sample lại nếu cần khớp size
            # Ở đây ta giả sử convert_to_gnn đã sinh đủ số lượng tương ứng
            neg_src = precomputed_neg_edge_index[0].to(device)
            neg_dst = precomputed_neg_edge_index[1].to(device)
            
            # Nếu số lượng negative nhiều hơn positive (do cấu hình ratio), ta cần cắt bớt hoặc lặp lại positive
            # Để đơn giản, ta lấy min length
            min_len = min(pos_idx.size(1), neg_src.size(0))
            pos_score = pos_score[:min_len] # Cắt bớt pos nếu cần
            neg_src = neg_src[:min_len]
            neg_dst = neg_dst[:min_len]
            
        else:
            # --- CÁCH 2: Random (Cách cũ - Dynamic) ---
            # Random ngẫu nhiên mỗi epoch
            neg_src = torch.randint(0, data['user'].num_nodes, (pos_idx.size(1),), device=device)
            neg_dst = torch.randint(0, data['article'].num_nodes, (pos_idx.size(1),), device=device)

        neg_user_emb = z_dict['user'][neg_src]
        neg_article_emb = z_dict['article'][neg_dst]
        neg_score = (neg_user_emb * neg_article_emb).sum(dim=-1)

        # 4. Loss Function
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # 5. Backward Pass
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    print("--- [Trainer] Training Finished ---")
