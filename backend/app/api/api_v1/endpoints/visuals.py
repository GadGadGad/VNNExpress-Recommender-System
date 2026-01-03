from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from backend.app.services.visuals import plot_embedding_space, generate_user_wordcloud
from backend.app.services.recommendation import load_cf_model

router = APIRouter()

class VisualRequest(BaseModel):
    user_id: str
    history: List[str]
    rec_urls: List[str]
    model_name: str = "MA-HCL"

@router.post("/embedding-space")
def get_embedding_plot(request: Request, payload: VisualRequest):
    res = request.app.state.resources
    articles_df = res["articles_df"]
    article_map = res["article_map_cf"]
    user_map = res["user_map_cf"]
    
    if payload.user_id not in user_map:
        return {"error": "User not found"}
        
    cf_model = load_cf_model(payload.model_name, len(user_map), len(article_map), graph_name="strict_g2")
    
    if not cf_model:
        return {"error": "Model not loaded"}
        
    plot_data = plot_embedding_space(
        cf_model, 
        user_map[payload.user_id], 
        payload.history, 
        payload.rec_urls, 
        article_map, 
        articles_df
    )
    
    return {"plot_data": plot_data}

@router.post("/wordcloud")
def get_wordcloud(request: Request, payload: VisualRequest):
    res = request.app.state.resources
    articles_df = res["articles_df"]
    
    b64_img = generate_user_wordcloud(payload.history, articles_df)
    return {"image": b64_img}
