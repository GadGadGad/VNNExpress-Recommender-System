from fastapi import APIRouter
from backend.app.api.api_v1.endpoints import config, recommendation, visuals

api_router = APIRouter()
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(recommendation.router, prefix="/recommend", tags=["recommendation"])
api_router.include_router(visuals.router, prefix="/visuals", tags=["visuals"])
