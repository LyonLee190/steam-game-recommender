# Steam Game Recommender

A BERT-based game recommendation system that suggests Steam games based on natural language descriptions. The system uses semantic similarity to find games that match user preferences without relying on explicit keywords.

## Features

- **Semantic Understanding**: Uses BERT embeddings to understand game descriptions beyond simple keyword matching
- **Natural Language Queries**: Input descriptions like "tactical shooter with economy system" to get relevant recommendations
- **Fast API**: RESTful API built with FastAPI for easy integration
- **Docker Support**: Containerized application for easy deployment

## Project Structure

```
steam-game-recommender/
├── data-collection/          # Web scraping and data gathering scripts
├── model-training/           # Data processing and BERT model training
│   └── BERT_Embeddings_and_KNN_Recommendation_System.ipynb
└── recommendation-api/       # FastAPI application
    ├── main.py
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd steam-game-recommender/recommendation-api
   ```

2. **Ensure you have the model files**
   Make sure these files are in the `recommendation-api/` directory:
   - `game_embeddings.npy`
   - `knn_model.pkl`
   - `games_with_embeddings.csv`

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the API**
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Local Development

1. **Set up virtual environment**
   ```bash
   cd recommendation-api/
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## API Usage

### Get Recommendations

**Endpoint:** `POST /recommend`

**Request:**
```json
{
  "description": "tactical shooter with economy system and rounds",
  "num_recommendations": 5
}
```

**Response:**
```json
{
    "recommendations": [
        {
            "name": "Sniper Ghost Warrior Contracts 2",
            "genres": "Action",
            "price": 3.99,
            "similarity_score": 0.5080599188804626
        },
        {
            "name": "Sniper Fury",
            "genres": "Action, Free To Play",
            "price": 0.0,
            "similarity_score": 0.5064460635185242
        },
        {
            "name": "Buckshot Roulette",
            "genres": "Action, Indie, Simulation",
            "price": 1.79,
            "similarity_score": 0.4956326484680176
        },
        {
            "name": "Sniper Elite",
            "genres": "Action",
            "price": 3.19,
            "similarity_score": 0.4905034899711609
        },
        {
            "name": "Masked Shooters 2",
            "genres": "Action, Indie",
            "price": 0.79,
            "similarity_score": 0.4892879128456116
        }
    ],
    "processing_time_ms": 265.54203033447266
}
```

### Example Queries

```bash
# Tactical shooter
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"description": "tactical shooter with economy system", "num_recommendations": 5}'

# Building game
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"description": "sandbox building with physics", "num_recommendations": 5}'

# Racing game
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"description": "fast cars racing simulation", "num_recommendations": 5}'
```

## How It Works

1. **Data Collection**: Gather Steam game data including descriptions and metadata
2. **Text Processing**: Clean and combine game information into rich text representations
3. **BERT Embeddings**: Generate semantic embeddings using sentence-transformers
4. **Similarity Search**: Use KNN with cosine similarity to find similar games
5. **API Service**: FastAPI application serves recommendations via REST endpoint

## Development

### Requirements

- Python 3.9+
- Docker (for containerized deployment)
- 4GB+ RAM (for model loading)

### Key Dependencies

- `fastapi` - Web framework
- `sentence-transformers` - BERT embeddings
- `scikit-learn` - KNN similarity search
- `pandas` - Data processing
- `uvicorn` - ASGI server

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Test recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"description": "puzzle game with logic", "num_recommendations": 3}'
```

## Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up --build

# Production deployment
docker build -t steam-recommender .
docker run -p 8000:8000 steam-recommender
```

### Cloud Deployment

The Docker container can be deployed to AWS ECS, Google Cloud Run, Azure Container Instances, or any Kubernetes cluster.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Troubleshooting

### Common Issues

1. **Model files not found**: Ensure model files are in the `recommendation-api/` directory
2. **Memory errors**: Increase Docker memory limit or system RAM
3. **Dependency conflicts**: Use exact versions in requirements.txt
4. **Port conflicts**: Change port in docker-compose.yml if 8000 is in use