"""
FastAPI backend with native text features.
"""
from collections import defaultdict, deque
import logging
import os
import pickle
import sys
import threading
import time

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import xgboost as xgb

try:
    from api.settings import AppSettings, configure_logging
except ImportError:  # pragma: no cover
    from settings import AppSettings, configure_logging

# Update path to import from src if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from features_native import TextFeatureExtractor

settings = AppSettings()
configure_logging(settings.log_level)
logger = logging.getLogger("ranksmart.main_native")

_rate_limit_lock = threading.Lock()
_rate_limit_buckets = defaultdict(deque)

app = FastAPI(title="RankSmart Search", description="Text-based search ranking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
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


def _validate_query_text(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    if len(cleaned) > settings.max_query_length:
        raise HTTPException(
            status_code=422,
            detail=f"Query too long. Max length is {settings.max_query_length} characters",
        )
    return cleaned


def _validate_top_k(top_k: int) -> int:
    if top_k > settings.max_top_k:
        raise HTTPException(
            status_code=422,
            detail=f"top_k too high. Max value is {settings.max_top_k}",
        )
    return top_k


def require_api_key(request: Request) -> None:
    if not settings.require_api_key:
        return

    configured_key = settings.api_key.strip()
    if not configured_key:
        logger.error("api_key_required_but_missing_configuration")
        raise HTTPException(status_code=503, detail="API key authentication misconfigured")

    provided_key = request.headers.get(settings.api_key_header_name, "")
    if provided_key != configured_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.middleware("http")
async def ip_rate_limit(request: Request, call_next):
    if request.url.path == "/search":
        bucket_key = request.client.host if request.client else "unknown"
        now = time.time()
        window = settings.rate_limit_window_seconds
        max_requests = settings.rate_limit_max_requests

        with _rate_limit_lock:
            bucket = _rate_limit_buckets[bucket_key]
            while bucket and bucket[0] <= now - window:
                bucket.popleft()

            if len(bucket) >= max_requests:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

            bucket.append(now)

    return await call_next(request)

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

        logger.info("model_loaded documents=%s", len(documents))
    else:
        if settings.strict_model_loading:
            raise RuntimeError(
                "Required model artifacts not found. Set STRICT_MODEL_LOADING=false "
                "to allow degraded startup."
            )

        logger.warning("model_missing_degraded_mode_enabled")
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
async def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=settings.default_top_k, ge=1),
    _: None = Depends(require_api_key),
):
    query = _validate_query_text(query)
    top_k = _validate_top_k(top_k)
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

    logger.info(
        "search_completed query_len=%s top_k=%s result_count=%s latency_ms=%.2f degraded=%s",
        len(query),
        top_k,
        len(formatted_results),
        response_time,
        not bool(model and extractor),
    )
    
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
        "documents": len(documents),
        "degraded_mode": not bool(model and extractor),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
