import torch
import numpy as np


def evaluate_top_k(model, data, edge_type, k=5, train_edge_index=None):
    """
    Evaluate model: Precision, Recall, NDCG.
    Mask items appearing in Train set.
    """
    model.eval()
    device = data['user'].x.device # Get current device from data
    
    with torch.no_grad():
        # Get Embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)
        user_emb = z_dict['user']
        article_emb = z_dict['article']

        # Prepare Ground Truth (Test set)
        edge_label_index = data[edge_type].edge_label_index
        labels = data[edge_type].edge_label
        pos_edges = edge_label_index[:, labels == 1] # Only take Positive edges in Test

        ground_truth = {}
        src_cpu = pos_edges[0].cpu().numpy()
        dst_cpu = pos_edges[1].cpu().numpy()

        for u, a in zip(src_cpu, dst_cpu):
            if u not in ground_truth: ground_truth[u] = set()
            ground_truth[u].add(a)
            

        train_history = {}
        if train_edge_index is not None:
            t_src = train_edge_index[0].cpu().numpy()
            t_dst = train_edge_index[1].cpu().numpy()
            for u, a in zip(t_src, t_dst):
                if u not in train_history: train_history[u] = set()
                train_history[u].add(a)

        # Compute Metrics
        precisions, recalls, ndcgs = [], [], []

        for u_idx in ground_truth.keys():
            true_items = ground_truth[u_idx]

            # Calculate scores for ALL articles
            # Shape: (Num_Articles,)
            scores = (user_emb[u_idx].unsqueeze(0) @ article_emb.T).squeeze()


            if u_idx in train_history:
                seen_items = list(train_history[u_idx])
                # Assign very low score to seen items to reduce their rank
                scores[seen_items] = -float('inf')


            # Get Top K
            actual_k = min(k, len(scores))
            _, top_k_indices = torch.topk(scores, k=actual_k)
            recommendations = top_k_indices.cpu().numpy()

            # Calculate scores
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