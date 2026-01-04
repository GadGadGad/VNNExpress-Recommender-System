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


        z_dict = model(data.x_dict, data.edge_index_dict)

        # Positive Edges (Train set interactions)
        pos_idx = data[edge_type].edge_label_index
        user_emb = z_dict['user'][pos_idx[0]]
        article_emb = z_dict['article'][pos_idx[1]]
        
        # Calculate Positive scores
        pos_score = (user_emb * article_emb).sum(dim=-1)

        # Negative Sampling (Strategy-based)
        if neg_strategy == 'precomputed' and precomputed_neg_edge_index is not None:
            # Use Precomputed Hard Negatives
            # Precomputed count may be different from pos_idx, need slicing or sampling
            # Assume convert_to_gnn has generated enough
            neg_src = precomputed_neg_edge_index[0].to(device)
            neg_dst = precomputed_neg_edge_index[1].to(device)
            
            # If Negatives > Positives, truncate
            # Simplify by taking min length
            min_len = min(pos_idx.size(1), neg_src.size(0))
            pos_score = pos_score[:min_len] # Truncate pos if needed
            neg_src = neg_src[:min_len]
            neg_dst = neg_dst[:min_len]
            
        else:
            # Random (Old - Dynamic)
            # Randomize every epoch
            neg_src = torch.randint(0, data['user'].num_nodes, (pos_idx.size(1),), device=device)
            neg_dst = torch.randint(0, data['article'].num_nodes, (pos_idx.size(1),), device=device)

        neg_user_emb = z_dict['user'][neg_src]
        neg_article_emb = z_dict['article'][neg_dst]
        neg_score = (neg_user_emb * neg_article_emb).sum(dim=-1)

        # Loss Function
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # Backward Pass
        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    print("--- [Trainer] Training Finished ---")
