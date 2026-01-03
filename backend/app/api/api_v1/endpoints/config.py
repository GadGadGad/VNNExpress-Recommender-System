from fastapi import APIRouter, Request, HTTPException
from typing import Dict, List, Any
from backend.app.core.config import settings

router = APIRouter()

@router.get("/init")
def get_initial_config(request: Request):
    """
    Return all static configuration needed for Frontend initialization.
    """
    if not hasattr(request.app.state, "resources"):
         raise HTTPException(status_code=503, detail="Server initializing...")
         
    user_map = request.app.state.resources["user_map_cf"]
    
    return {
        "model_options": request.app.state.model_options,
        "categories": settings.CATEGORY_MAP,
        "users": sorted(list(user_map.keys())) if user_map else [],
        "user_map_available": bool(user_map)
    }

@router.get("/categories")
def get_categories():
    return settings.CATEGORY_MAP
