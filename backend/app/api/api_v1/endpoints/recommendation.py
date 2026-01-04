from fastapi import APIRouter, Request, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional, Any
import numpy as np

from backend.app.services.recommendation import get_recs, load_cf_model, load_cb_model
from backend.app.services.search import build_search_engine

router = APIRouter()

class RecRequest(BaseModel):
    user_id: Optional[str] = None
    history: List[str] = [] # List of URLs
    model_choice: str = "MA-HCL"
    cb_model_choice: str = "vn-sbert"
    alpha: float = 0.5
    k: int = 10
    use_adt: bool = True
    filters: List[str] = [] # Category filters

class SearchRequest(BaseModel):
    query: str
    k: int = 20
    filters: List[str] = []

@router.post("/")
def get_recommendations(request: Request, payload: RecRequest):
    res = request.app.state.resources
    articles_df = res["articles_df"]
    user_map_cf = res["user_map_cf"]
    article_map_cf = res["article_map_cf"]
    adj_norm = res["adj_norm"]
    edge_index = res.get("edge_index")
    user_priors = res["user_priors"]
    
    if articles_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Load Models (Cached)
    # These load functions are LRU Cached
    cf_model = None
    if payload.alpha > 0 and payload.model_choice:
         cf_model = load_cf_model(
             payload.model_choice, 
             len(user_map_cf), 
             len(article_map_cf), 
             graph_name="strict_g2" # Defaulting for now
         )
         
    cb_model = None
    if payload.alpha < 1.0:
        cb_model = load_cb_model(payload.cb_model_choice, articles_df)
        
    # LOGIC MIRRORING app.py
    candidate_scores = {}
    
    # CF Branch
    if cf_model and payload.user_id and payload.user_id in user_map_cf:
        uidx = user_map_cf[payload.user_id]
        raw_recs = get_recs(
            cf_model, payload.model_choice, uidx, [], 
            article_map_cf, articles_df, k=200, 
            adj_norm=adj_norm, user_priors=user_priors, 
            edge_index=edge_index,
            use_adt=payload.use_adt
        )
        for u, s in raw_recs:
             candidate_scores[u] = {'score': s * payload.alpha, 'source': 'Social'}

    # CB Branch (History based)
    if cb_model and payload.history:
        # Get indices for history
        h_indices = []
        for u in payload.history[-10:]: # Last 10
             matches = articles_df.index[articles_df['url'] == u].tolist()
             if matches: h_indices.append(matches[0])
             
        if h_indices:
             cb_idx, cb_val = cb_model.recommend(h_indices, k=150)
             max_v = max(cb_val) if cb_val else 1
             for idx, val in zip(cb_idx, cb_val):
                 u = articles_df.iloc[idx]['url']
                 norm = (val/max_v) * (1 - payload.alpha)
                 curr = candidate_scores.get(u, {'score': 0, 'source': 'Content'})
                 candidate_scores[u] = {'score': curr['score'] + norm, 'source': curr['source']}

    # Sort & Filter
    final_recs = []
    
    # Convert dict to list
    candidates = [(u, d['score'], d['source']) for u, d in candidate_scores.items()]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    count = 0
    history_set = set(payload.history)
    
    for u, score, source in candidates:
        if count >= payload.k: break
        if u in history_set: continue
        
        row = articles_df[articles_df['url'] == u].iloc[0]
        cat = row['source_category']
        
        if payload.filters and cat not in payload.filters: continue
        
        final_recs.append({
            "url": u,
            "title": row['title'],
            "description": row['short_description'],
            "score": score,
            "source": source,
            "category": cat,
            "published_at": str(row.get('published_at', '')),
            "image": row.get('image_url', '') # If exists
        })
        count += 1
        
    return {"recommendations": final_recs}

@router.post("/search")
def search_articles(request: Request, payload: SearchRequest):
    res = request.app.state.resources
    articles_df = res["articles_df"]
    
    # Simple on-the-fly search engine build (or cache it in resources if slow)
    # Using the cached one in app.py was better, let's try to cache it in state too
    if not hasattr(request.app.state, "search_engine"):
         request.app.state.search_engine = build_search_engine(articles_df)
    
    engine = request.app.state.search_engine
    
    tokenized_query = payload.query.lower().split()
    doc_scores = engine.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:100]
    
    results = []
    count = 0
    for idx in top_indices:
        if count >= payload.k: break
        if doc_scores[idx] <= 0: continue
        
        row = articles_df.iloc[idx]
        if payload.filters and row['source_category'] not in payload.filters: continue
        
        results.append({
            "url": row['url'],
            "title": row['title'],
            "description": row['short_description'],
            "score": float(doc_scores[idx]),
            "category": row['source_category'],
            "published_at": str(row.get('published_at', ''))
        })
        count += 1
        
    return {"results": results}
