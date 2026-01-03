from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from backend.app.core.config import settings
from backend.app.services.recommendation import load_resources, MODEL_OPTIONS
from backend.app.api.api_v1.api import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Resources
    print("Loading resources...")
    start_time = time.time()
    res = load_resources()
    app.state.resources = {
        "articles_df": res[0],
        "user_map_cf": res[1],
        "article_map_cf": res[2],
        "user_history": res[3],
        "adj_norm": res[4],
        "user_priors": res[5],
        "edge_index": res[6],
        "status": res[7]
    }
    app.state.model_options = MODEL_OPTIONS
    print(f"Resources loaded in {time.time() - start_time:.2f}s")
    
    yield
    # Shutdown logic (if any)
    print("Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME, 
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to News RecSys API", "status": "running"}
