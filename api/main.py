"""
FastAPI backend for search ranking system
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import SearchRanker

# Initialize FastAPI
app = FastAPI(
    title="Search Ranking System API",
    description="Learning-to-Rank based search engine with LambdaMART",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ranker
ranker = None

# Sample documents (in production, load from database)
DOCUMENTS = [
    "Learn Python programming from basics to advanced with examples",
    "Complete machine learning course with Python and scikit-learn",
    "Data science bootcamp: pandas, numpy, matplotlib tutorial",
    "Web development with React and Node.js full stack guide",
    "Deep learning specialization: neural networks and TensorFlow",
    "Artificial intelligence fundamentals: search algorithms and logic",
    "Python for beginners: getting started with Python programming",
    "Java programming: object-oriented programming masterclass",
    "AWS cloud computing: architecture and deployment guide",
    "Machine learning algorithms explained with Python code",
    "Data visualization with Python and matplotlib",
    "React hooks tutorial: modern React development",
    "Docker containerization for developers",
    "Kubernetes orchestration guide",
    "SQL database design and optimization"
]

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global ranker
    try:
        ranker = SearchRanker()
        ranker.index_documents(DOCUMENTS)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Model not found: {e}")
        print("Please run: python src/train.py first")

# Models
class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int
    response_time_ms: float

class Document(BaseModel):
    id: int
    content: str
    metadata: Optional[Dict] = None

class IndexRequest(BaseModel):
    documents: List[Document]

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Search Ranking System API",
        "version": "1.0.0",
        "endpoints": [
            "/search",
            "/health",
            "/docs",
            "/metrics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ranker is not None,
        "documents_indexed": len(DOCUMENTS) if ranker else 0
    }

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, max_length=200, description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    include_features: bool = Query(False, description="Include feature details")
):
    """
    Search endpoint that returns ranked results
    
    Example:
    GET /search?query=python%20programming&top_k=5
    """
    if not ranker:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")
    
    start_time = time.time()
    
    try:
        results = ranker.search(query, top_k)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted = {
                "document": result['document'],
                "score": result['score']
            }
            if include_features:
                # Show top 5 most important features
                features = result['features']
                top_features = {
                    k: v for k, v in sorted(features.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                }
                formatted['top_features'] = top_features
            
            formatted_results.append(formatted)
        
        response_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            results=formatted_results,
            total_results=len(formatted_results),
            response_time_ms=round(response_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_documents(request: IndexRequest):
    """Index new documents (for dynamic indexing)"""
    global ranker, DOCUMENTS
    
    if not ranker:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    new_docs = [doc.content for doc in request.documents]
    DOCUMENTS.extend(new_docs)
    ranker.index_documents(DOCUMENTS)
    
    return {
        "message": f"Indexed {len(new_docs)} documents",
        "total_documents": len(DOCUMENTS)
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "model_status": "loaded" if ranker else "not_loaded",
        "num_documents": len(DOCUMENTS) if ranker else 0,
        "num_features": len(ranker.feature_extractor.feature_names) if ranker else 0
    }

# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
