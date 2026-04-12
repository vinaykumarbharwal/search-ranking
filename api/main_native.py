"""
FastAPI backend with native text features
No Kaggle files needed - works immediately!
"""
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import xgboost as xgb
import pickle
import os
import time
import sys

# Update path to import from src if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features_native import TextFeatureExtractor

app = FastAPI(title="RankSmart Search", description="Text-based search ranking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crucial: Mount the frontend static files so the UI works
static_path = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Load model and components
model = None
extractor = None
documents = []

# Sample documents (your content)
DOCUMENTS = [
    "Learn Python programming from basics to advanced with examples and projects",
    "Complete machine learning course with Python and scikit-learn library",
    "Data science bootcamp: pandas, numpy, matplotlib tutorial for beginners",
    "Web development with React and Node.js full stack guide 2024",
    "Deep learning specialization: neural networks and TensorFlow masterclass",
    "Artificial intelligence fundamentals: search algorithms and logic programming",
    "Python for beginners: getting started with Python programming language",
    "Java programming: object-oriented programming masterclass for developers",
    "AWS cloud computing: architecture and deployment guide for startups",
    "Machine learning algorithms explained with Python code examples",
]

@app.on_event("startup")
async def startup_event():
    global model, extractor, documents
    
    documents = DOCUMENTS
    
    # Try to load trained model
    root_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(root_dir, 'models', 'ranker_native.json')
    extractor_path = os.path.join(root_dir, 'models', 'extractor_native.pkl')
    
    if os.path.exists(model_path) and os.path.exists(extractor_path):
        # Load trained model
        model = xgb.Booster()
        model.load_model(model_path)
        
        with open(extractor_path, 'rb') as f:
            extractor = pickle.load(f)
        
        print(f"✅ Loaded trained model with {len(documents)} documents")
    else:
        # Create simple fallback model
        print("⚠️ No trained model found, using simple ranking")
        model = None
        extractor = None

def simple_rank(query, documents):
    """Simple keyword matching as fallback"""
    scores = []
    query_words = set(query.lower().split())
    
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words & doc_words)
        scores.append(overlap / max(len(query_words), 1))
    
    return scores

@app.get("/")
async def root():
    return {"message": "RankSmart Search API", "status": "running"}

@app.get("/search")
async def search(query: str = Query(..., min_length=1), top_k: int = 10):
    start_time = time.time()
    
    if model and extractor:
        # Use ML ranking
        results = []
        for doc_id, doc in enumerate(documents):
            features = extractor.extract_features(query, doc)
            dmatrix = xgb.DMatrix([features])
            score = model.predict(dmatrix)[0]
            results.append((doc, float(score), doc_id))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        formatted_results = [
            {"document": doc, "score": float(score), "id": doc_id}
            for doc, score, doc_id in results
        ]
    else:
        # Use simple ranking
        scores = simple_rank(query, documents)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        formatted_results = [
            {"document": doc, "score": float(score), "id": i}
            for i, (doc, score) in enumerate(ranked[:top_k])
        ]
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        "query": query,
        "results": formatted_results,
        "total_results": len(formatted_results),
        "response_time_ms": round(response_time, 2)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "documents": len(documents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
