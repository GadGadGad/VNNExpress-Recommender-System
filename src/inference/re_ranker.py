import numpy as np
import torch
import torch.nn.functional as F
import datetime

class CalibratedReRanker:
    """
    Post-processing for News Recommendation to balance Accuracy, Diversity, and Calibration.
    """
    def __init__(self, item_categories, alpha=0.5, lambda_mmr=0.5, freshness_lambda=0.1):
        """
        item_categories: array/list where index is item_idx, value is category_id
        alpha: weight for calibration (0=pure relevance, 1=pure calibration)
        lambda_mmr: weight for MMR (0=pure diversity, 1=pure relevance)
        freshness_lambda: decay rate for freshness boost (higher = faster decay)
        """
        self.item_categories = np.array(item_categories)
        self.n_categories = len(np.unique(item_categories))
        self.alpha = alpha
        self.lambda_mmr = lambda_mmr
        self.freshness_lambda = freshness_lambda

    def get_distribution(self, item_indices):
        """Compute categorical distribution for a set of items."""
        cats = self.item_categories[item_indices]
        counts = np.bincount(cats, minlength=self.n_categories)
        dist = counts / (len(item_indices) + 1e-9)
        return dist

    def freshness_boost(self, scores, item_dates, reference_date=None, boost_weight=0.3):
        """
        Boost scores for newer articles using exponential decay.
        
        scores: array of relevance scores (n_items,)
        item_dates: array of datetime objects or None for each item
        reference_date: the "now" date to compare against (defaults to today)
        boost_weight: how much to blend freshness (0=ignore, 1=freshness only)
        
        Returns: boosted_scores
        """
        if reference_date is None:
            reference_date = datetime.datetime.now()
        
        n_items = len(scores)
        freshness_scores = np.zeros(n_items)
        
        for i in range(n_items):
            if item_dates is not None and i < len(item_dates) and item_dates[i] is not None:
                try:
                    if isinstance(item_dates[i], str):
                        # Parse common date formats
                        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                            try:
                                dt = datetime.datetime.strptime(item_dates[i], fmt)
                                break
                            except:
                                dt = None
                    else:
                        dt = item_dates[i]
                    
                    if dt:
                        days_old = (reference_date - dt).days
                        # Exponential decay: e^(-lambda * days)
                        freshness_scores[i] = np.exp(-self.freshness_lambda * max(0, days_old))
                    else:
                        freshness_scores[i] = 0.5
                except:
                    freshness_scores[i] = 0.5
            else:
                freshness_scores[i] = 0.5
        
        # Normalize freshness to [0, 1]
        if freshness_scores.max() > 0:
            freshness_scores = freshness_scores / freshness_scores.max()
        
        # Blend: (1 - boost_weight) * relevance + boost_weight * freshness
        boosted = (1 - boost_weight) * scores + boost_weight * freshness_scores * scores.max()
        
        return boosted

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
                
                # Check how distribution changes if item is added
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

    def full_rerank(self, scores, item_embeddings=None, item_dates=None, 
                    user_history_items=None, top_k=20, 
                    use_freshness=True, use_calibration=True, use_mmr=True,
                    freshness_weight=0.2):
        """
        Full post-processing pipeline combining all strategies.
        
        Order: Freshness Boost -> Calibration -> MMR Diversity
        """
        boosted_scores = scores.copy() if isinstance(scores, np.ndarray) else np.array(scores)
        
        # Step 1: Freshness Boost
        if use_freshness and item_dates is not None:
            boosted_scores = self.freshness_boost(boosted_scores, item_dates, 
                                                   boost_weight=freshness_weight)
        
        # Step 2: Calibration (if user history available)
        if use_calibration and user_history_items is not None and len(user_history_items) > 0:
            selected = self.calibrate(boosted_scores, user_history_items, top_k=min(top_k * 2, 50))
            # Re-order boosted_scores to prioritize calibrated selection
            calibrated_indices = set(selected)
        else:
            calibrated_indices = None
        
        # Step 3: MMR Diversity (if embeddings available)
        if use_mmr and item_embeddings is not None:
            final_list = self.mmr_rerank(item_embeddings, boosted_scores, top_k=top_k)
        else:
            final_list = np.argsort(-boosted_scores)[:top_k].tolist()
        
        return final_list
