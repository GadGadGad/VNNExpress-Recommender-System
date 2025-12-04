import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, data, optimizer, epochs, device, edge_type):
    print(f"--- [Trainer] Starting training for {epochs} epochs ---")

    # Progress Bar
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # 1. Forward Pass
        z_dict = model(data.x_dict, data.edge_index_dict)

        # 2. Positive Edges (Tương tác thật trong tập Train)
        pos_idx = data[edge_type].edge_label_index

        # Lấy vector của user và article tương ứng
        user_emb = z_dict['user'][pos_idx[0]]
        article_emb = z_dict['article'][pos_idx[1]]

        # Tính điểm (Dot Product)
        pos_score = (user_emb * article_emb).sum(dim=-1)

        # 3. Negative Sampling (Tương tác giả)
        # Random ngẫu nhiên User ghép với Article bất kỳ
        # Số lượng negative = số lượng positive
        neg_src = torch.randint(0, data['user'].num_nodes, (pos_idx.size(1),), device=device)
        neg_dst = torch.randint(0, data['article'].num_nodes, (pos_idx.size(1),), device=device)

        neg_user_emb = z_dict['user'][neg_src]
        neg_article_emb = z_dict['article'][neg_dst]

        neg_score = (neg_user_emb * neg_article_emb).sum(dim=-1)

        # 4. Loss Function (Binary Cross Entropy)
        # Gom Pos và Neg lại
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # 5. Backward Pass
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    print("--- [Trainer] Training Finished ---")
