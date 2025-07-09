# ===================================================================
# FastAPI Game Info Service
# Simple API that returns game information by game ID
# ===================================================================

from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import os
import json
import gzip
import base64

# ===================================================================
# Data Models
# ===================================================================

class GameInfo(BaseModel):
    """Full game information response model"""
    app_id: str
    name: str
    detailed_description: str
    short_description: str
    genres: str
    categories: str
    developers: str
    publishers: str
    price: float
    release_date: str

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str

# ===================================================================
# Database Connection
# ===================================================================

class GameDatabase:
    """Simple database interface for game data"""
    
    def __init__(self, db_path: str = "games.db"):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file {db_path} not found")
    
    def get_game_by_id(self, app_id: str) -> Optional[dict]:
        """Get full game information with proper encoding handling"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT app_id, name, detailed_description, short_description,
                       genres, categories, developers, publishers, price, release_date
                FROM games 
                WHERE app_id = ?
            """, (app_id,))
            
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # Clean all text fields properly
                for key, value in result.items():
                    if value is None:
                        result[key] = ""
                    elif isinstance(value, str):
                        # Clean text but keep full content
                        cleaned = value.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
                        # Remove or replace problematic Unicode characters
                        cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
                        result[key] = cleaned
                    else:
                        result[key] = value
                return result
            return None
            
        finally:
            conn.close()
    
    def game_exists(self, app_id: str) -> bool:
        """Check if game exists in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT 1 FROM games WHERE app_id = ? LIMIT 1", (app_id,))
            return cursor.fetchone() is not None
        finally:
            conn.close()
    
    def get_total_games(self) -> int:
        """Get total number of games in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM games")
            return cursor.fetchone()[0]
        finally:
            conn.close()

# ===================================================================
# FastAPI Application
# ===================================================================

# Initialize FastAPI app with size limits
app = FastAPI(
    title="Steam Game Info API",
    description="Simple API to get game information by game ID",
    version="1.0.0"
)

# Add middleware to handle large responses
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
try:
    db = GameDatabase("games.db")
    print(f"Connected to database with {db.get_total_games()} games")
except FileNotFoundError:
    print("Warning: games.db not found. Please create database first.")
    db = None

# ===================================================================
# API Endpoints
# ===================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    total_games = db.get_total_games() if db else 0
    
    return {
        "message": "Steam Game Info API",
        "version": "1.0.0",
        "total_games": total_games,
        "endpoints": {
            "get_game": "/game/{app_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        total_games = db.get_total_games()
        return {
            "status": "healthy",
            "database": "connected",
            "total_games": total_games
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

@app.get("/game/{app_id}")
async def get_game(app_id: str):
    """Get full game information with proper content handling"""
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Basic validation
    if not app_id.strip() or len(app_id) > 20:
        raise HTTPException(status_code=400, detail="Invalid app_id")
    
    try:
        # Get game from database
        game_data = db.get_game_by_id(app_id.strip())
        
        if not game_data:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Create response with proper content handling
        response_data = {
            "app_id": game_data.get("app_id", ""),
            "name": game_data.get("name", ""),
            "detailed_description": game_data.get("detailed_description", ""),
            "short_description": game_data.get("short_description", ""),
            "genres": game_data.get("genres", ""),
            "categories": game_data.get("categories", ""),
            "developers": game_data.get("developers", ""),
            "publishers": game_data.get("publishers", ""),
            "price": float(game_data.get("price", 0)),
            "release_date": game_data.get("release_date", "")
        }
        
        # Create JSON response with explicit encoding
        json_str = json.dumps(response_data, ensure_ascii=False, separators=(',', ':'))
        
        return JSONResponse(
            content=response_data,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(json_str.encode('utf-8')))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error for app_id {app_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Alternative endpoint with compression for very large descriptions
@app.get("/game/{app_id}/compressed")
async def get_game_compressed(app_id: str):
    """Get game info with gzip compression for large descriptions"""
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        game_data = db.get_game_by_id(app_id.strip())
        
        if not game_data:
            raise HTTPException(status_code=404, detail="Game not found")
        
        # Compress large descriptions
        if len(game_data.get("detailed_description", "")) > 1000:
            description = game_data["detailed_description"]
            compressed = gzip.compress(description.encode('utf-8'))
            encoded = base64.b64encode(compressed).decode('ascii')
            game_data["detailed_description"] = f"COMPRESSED:{encoded}"
            game_data["_compressed"] = True
        
        return JSONResponse(content=game_data)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Server error")

# ===================================================================
# Optional: Batch endpoint for multiple games
# ===================================================================

@app.get("/games", response_model=List[GameInfo])
async def get_games(
    app_ids: str = Path(..., description="Comma-separated App IDs", example="730,440,570")
):
    """Get multiple games by comma-separated App IDs"""
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Parse app_ids
    try:
        app_id_list = [id.strip() for id in app_ids.split(",") if id.strip()]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid app_ids format")
    
    if not app_id_list:
        raise HTTPException(status_code=400, detail="No valid App IDs provided")
    
    if len(app_id_list) > 50:  # Limit to prevent abuse
        raise HTTPException(status_code=400, detail="Too many App IDs (max 50)")
    
    # Get games from database
    games = []
    for app_id in app_id_list:
        game_data = db.get_game_by_id(app_id)
        if game_data:
            games.append(GameInfo(**game_data))
    
    return games

# ===================================================================
# Error Handlers
# ===================================================================

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "status_code": 404
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error", 
            "message": "An internal server error occurred",
            "status_code": 500
        }
    )

# Add favicon handler to prevent 500 errors
@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return JSONResponse(
        status_code=204,
        content=None
    )

# ===================================================================
# Main Application
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Steam Game Info API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Example: http://localhost:8000/game/730")
    print("Compressed: http://localhost:8000/game/730/compressed")
    
    # Run with explicit settings to handle large responses
    uvicorn.run(
        "main:app",  # Change to your filename:app
        host="0.0.0.0",
        port=8000,
        reload=True,
        # Add these settings for large content
        limit_max_requests=1000,
        limit_concurrency=100,
        timeout_keep_alive=30
    )

# ===================================================================
# Installation and Usage Instructions
# ===================================================================

"""
Installation:
pip install fastapi uvicorn

Run the server:
python main.py

OR

uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Usage Examples:

1. Get single game:
   GET http://localhost:8000/game/730

2. Health check:
   GET http://localhost:8000/health

3. Get multiple games:
   GET http://localhost:8000/games/730,440,570

4. API documentation:
   GET http://localhost:8000/docs

5. Alternative docs:
   GET http://localhost:8000/redoc

Response format:
{
  "app_id": "730",
  "name": "Counter-Strike: Global Offensive",
  "detailed_description": "...",
  "short_description": "...",
  "genres": "Action, FPS",
  "categories": "Multi-player, Online",
  "developers": "Valve",
  "publishers": "Valve",
  "price": 0.0,
  "release_date": "2012-08-21"
}

Error responses:
{
  "detail": "Game with App ID '999999' not found"
}
"""

# ===================================================================
# Docker Support (Optional)
# ===================================================================

"""
Dockerfile:

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt:
fastapi==0.104.1
uvicorn[standard]==0.24.0

Docker commands:
docker build -t game-api .
docker run -p 8000:8000 -v $(pwd)/games.db:/app/games.db game-api
"""

# ===================================================================
# Testing Script
# ===================================================================

def test_api():
    """Simple test script"""
    import requests
    
    base_url = "http://localhost:8000"
    
    print("Testing Game Info API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test game endpoint
    try:
        response = requests.get(f"{base_url}/game/730")
        print(f"Game lookup: {response.status_code}")
        if response.status_code == 200:
            game = response.json()
            print(f"Found game: {game['name']}")
        else:
            print(response.json())
    except Exception as e:
        print(f"Game lookup failed: {e}")
    
    # Test non-existent game
    try:
        response = requests.get(f"{base_url}/game/999999")
        print(f"Non-existent game: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error test failed: {e}")

# Uncomment to run tests:
# test_api()