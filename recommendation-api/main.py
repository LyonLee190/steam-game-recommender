# Steam Game Recommendation API
# ==============================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import uvicorn
import logging
import time
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
embeddings = None
df_games = None
knn_model = None

def load_models():
    """Load BERT model, embeddings, KNN model, and game data"""
    global model, embeddings, df_games, knn_model
    
    # Get paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_files = {
        'embeddings': os.path.join(script_dir, 'game_embeddings.npy'),
        'knn': os.path.join(script_dir, 'knn_model.pkl'),
        'games': os.path.join(script_dir, 'games_with_embeddings.csv')
    }
    
    # Verify all required files exist
    for file_type, file_path in model_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load models and data
    logger.info("Loading BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logger.info("Loading embeddings and data...")
    embeddings = np.load(model_files['embeddings'])
    
    with open(model_files['knn'], 'rb') as f:
        knn_model = pickle.load(f)
    
    df_games = pd.read_csv(model_files['games'])
    
    logger.info(f"Successfully loaded {len(df_games)} games with {embeddings.shape[1]}D embeddings")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    load_models()
    yield
    # Shutdown - cleanup if needed

# Initialize FastAPI app
app = FastAPI(
    title="Steam Game Recommendation API",
    description="Get game recommendations based on description",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class RecommendationRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=500)
    num_recommendations: int = Field(default=5, ge=1, le=10)

class GameRecommendation(BaseModel):
    name: str
    genres: str
    price: float
    similarity_score: float

class RecommendationResponse(BaseModel):
    recommendations: List[GameRecommendation]
    processing_time_ms: float

@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Steam Game Recommendation API",
        "endpoint": "/recommend",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_ready = all([model, embeddings is not None, df_games is not None, knn_model])
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "total_games": len(df_games) if df_games is not None else 0
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_games(request: RecommendationRequest):
    """Get game recommendations based on text description"""
    if not all([model, embeddings is not None, df_games is not None, knn_model]):
        raise HTTPException(status_code=503, detail="Service unavailable: models not loaded")
    
    try:
        start_time = time.time()
        
        # Generate embedding for input description
        desc_embedding = model.encode([request.description])
        
        # Find similar games using KNN
        distances, indices = knn_model.kneighbors(
            desc_embedding, 
            n_neighbors=request.num_recommendations
        )
        
        # Convert distances to similarity scores (1 - cosine_distance)
        similarity_scores = 1 - distances[0]
        
        # Build response
        recommendations = []
        for idx, similarity in zip(indices[0], similarity_scores):
            game = df_games.iloc[idx]
            recommendations.append(GameRecommendation(
                name=game['name'],
                genres=game.get('genres', 'Unknown'),
                price=float(game.get('price', 0)),
                similarity_score=float(similarity)
            ))
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            recommendations=recommendations,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)