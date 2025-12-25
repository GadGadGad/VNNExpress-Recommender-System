import numpy as np
import torch
import torch.nn.functional as F

class CalibratedReRanker:
    """
    Post-processing for News Recommendation to balance Accuracy, Diversity, and Calibration.
    """
    def __init__(self, item_categories, alpha=0.5, lambda_mmr=0.5):
        """
        item_categories: array/list where index is item_idx, value is category_id
        alpha: weight for calibration (0=pure relevance, 1=pure calibration)
        lambda_mmr: weight for MMR (0=pure diversity, 1=pure relevance)
        """
        self.item_categories = np.array(item_categories)
        self.n_categories = len(np.unique(item_categories))
        self.alpha = alpha
        self.lambda_mmr = lambda_mmr

    def get_distribution(self, item_indices):
        """Compute categorical distribution for a set of items."""
        cats = self.item_categories[item_indices]
        counts = np.bincount(cats, minlength=self.n_categories)
        dist = counts / (len(item_indices) + 1e-9)
        return dist

    def calibrate(self, scores, user_history_items, top_k=20):
        """
        Re-rank based on KL-divergence between user history and top-K distribution.
        Steiglitz-McBride or simple greedy calibration.
        """
        # Get target distribution from history
        p_target = self.get_distribution(user_history_items)
        
        # Candidate set (e.g., top 100 from model)
        candidate_indices = np.argsort(-scores)[:100]
        
        selected = []
        p_current = np.zeros_like(p_target)
        
        for _ in range(top_k):
            best_idx = -1
            max_val = -float('inf')
            
            for i in candidate_indices:
                if i in selected: continue
                
                # Check how distribution changes if we add this item
                temp_selected = selected + [i]
                p_new = self.get_distribution(temp_selected)
                
                # KL Divergence improvement
                # Score = Relevance - alpha * KL(p_new || p_target)
                kl = np.sum(p_new * np.log((p_new + 1e-9) / (p_target + 1e-9)))
                
                val = (1 - self.alpha) * scores[i] - self.alpha * kl
                
                if val > max_val:
                    max_val = val
                    best_idx = i
            
            if best_idx == -1: break
            selected.append(best_idx)
            
        return selected

    def mmr_rerank(self, item_embeddings, scores, top_k=20):
        """
        Maximal Marginal Relevance for Diversity.
        Score = lambda * relevance - (1-lambda) * max_similarity(item, selected)
        """
        candidate_indices = np.argsort(-scores)[:100]
        selected = [candidate_indices[0]]
        
        embs = item_embeddings # (N, dim)
        
        for _ in range(1, top_k):
            best_idx = -1
            max_mmr = -float('inf')
            
            for i in candidate_indices:
                if i in selected: continue
                
                # Max similarity to selected set
                sims = F.cosine_similarity(embs[i].unsqueeze(0), embs[selected])
                max_sim = torch.max(sims).item()
                
                mmr_val = self.lambda_mmr * scores[i] - (1 - self.lambda_mmr) * max_sim
                
                if mmr_val > max_mmr:
                    max_mmr = mmr_val
                    best_idx = i
                    
            if best_idx == -1: break
            selected.append(best_idx)
            
        return selected
