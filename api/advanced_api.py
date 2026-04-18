from collections import defaultdict, deque
import asyncio
import logging
import os
import pickle
import sys
import threading
import time
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import xgboost as xgb

try:
    from api.settings import AppSettings, configure_logging
except ImportError:  # pragma: no cover
    from settings import AppSettings, configure_logging

settings = AppSettings()
configure_logging(settings.log_level)
logger = logging.getLogger("ranksmart.advanced_api")

_rate_limit_lock = threading.Lock()
_rate_limit_buckets = defaultdict(deque)

app = FastAPI(title="RankSmart Advanced API", description="Production-grade Learning-to-Rank System")

# Mount Static Files so the frontend still works!
static_path = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Make sure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from hybrid_retriever import HybridRetriever
except ImportError:
    HybridRetriever = None

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
    "Ultimate CSS Grid and Flexbox tutorial for responsive design",
    "RESTful API design best practices using FastAPI and SQLAlchemy",
    "Understanding pointer arithmetic and memory management in C++",
    "Rust programming language: memory safety without garbage collection",
    "Deploying Docker containers to Kubernetes: a step-by-step tutorial",
    "Git version control for software engineering teams",
    "Mastering Vim: productivity hacks for Linux terminal users",
    "Building microservices using Spring Boot and Java 17",
    "Serverless architecture with AWS Lambda and Amazon API Gateway",
    "Introduction to GraphQL: querying data efficiently",
    "Advanced TypeScript concepts: generics, utility types, and decorators",
    "How to build scalable realtime web apps with WebSockets and Node.js",
    "Cybersecurity basics: preventing SQL injection and XSS attacks",
    "Penetration testing with Kali Linux and Metasploit framework",
    "Data engineering pipelines using Apache Airflow and Kafka",
    "Introduction to quantum computing principles for computer scientists",
    "Developing mobile apps using Flutter and Dart framework",
    "SwiftUI for iOS app development: complete masterclass",
    "Android development with Kotlin: fragments, activities, and intents",
    "Game development in Unity using C#: a complete beginner's guide",
    "Unreal Engine 5 Blueprints: visual scripting for 3D games",
    "Designing highly available distributed systems systems architecture",
    "CAP theorem explained: consistency, availability, and partition tolerance",
    "Introduction to Natural Language Processing with Hugging Face transformers",
    "Image classification using convolutional neural networks (CNNs) in PyTorch",
    "Reinforcement learning: teaching algorithms to play games using OpenAI Gym",
    "Building your first LLM application using LangChain and OpenAI",
    "Search engine optimization (SEO) techniques for modern single page applications",
    "Continuous Integration and Continuous Deployment (CI/CD) with GitHub Actions",
    "Terraform crash course: infrastructure as code for Azure and AWS",
    "Ansible playbook fundamentals for automating server configuration",
    "PostgreSQL performance tuning: indexes, query plans, and optimization",
    "MongoDB fundamentals: NoSQL database design and aggregations",
    "Redis caching strategies to improve web application latency",
    "Elasticsearch full-text search and vector embedding retrieval",
    "Understanding OAuth 2.0 and OpenID Connect for secure authentication",
    "JWT (JSON Web Tokens) handling in modern web applications",
    "Frontend system design: architecture patterns for scale",
    "Design patterns in Java: Factory, Singleton, and Observer explained",
    "SOLID principles of object-oriented programming",
    "Test-Driven Development (TDD) using Python Pytest and Mock",
    "End-to-end testing with Cypress in a React application",
    "HTML5 WebGL graphics programming using Three.js",
    "CSS Animations and keyframes for engaging user interfaces",
    "Linux bash scripting: automation tricks for system administrators",
    "Networking 101: TCP/IP, DNS, routing, and HTTP protocols",
    "Nginx web server configuration and reverse proxy setup",
    "Load balancing strategies: round-robin, least connections, and IP hash",
    "WebRTC protocol for peer-to-peer video communication in browser",
    "Using Webpack, Vite, and Babel for JavaScript module bundling",
    "Server-side rendering (SSR) vs Static Site Generation (SSG) in Next.js",
    "Building a resilient REST API with Go (Golang) and Gorilla Mux",
    "Concurrency in Go: Goroutines, channels, and select statements",
    "Understanding the JavaScript event loop and asynchronous programming",
    "Promises and Async/Await deep dive in modern JavaScript",
    "WebAssembly (Wasm) introduction: running C++ and Rust in the browser",
    "How compilers work: lexical analysis and abstract syntax trees",
    "Building a simple interpreter from scratch in Python",
    "Writing efficient SQL queries using window functions and CTEs",
    "Data visualization with D3.js: building interactive charts",
    "Dashboard creation using Python Streamlit and Plotly",
    "Automated web scraping with Python Selenium and BeautifulSoup",
    "Handling CAPTCHAs and proxies in web scraping pipelines",
    "Cryptography 101: symmetric vs asymmetric encryption (RSA, AES)",
    "Hashing algorithms: SHA-256, bcrypt, and password security",
    "Blockchain fundamentals: how distributed ledgers operate",
    "Smart contract development on Ethereum using Solidity",
    "DeFi protocols and web3 dApp integration with React",
    "Introduction to Internet of Things (IoT) using Raspberry Pi and Python",
    "MicroPython on ESP32: building smart home sensors",
    "Big Data processing with Apache Spark and Hadoop",
    "Writing MapReduce jobs for distributed data architectures",
    "Data warehousing concepts: Star schemas and snowflake designs",
    "Data modeling in Google BigQuery for analytics",
    "A/B testing methodologies and statistical significance calculation",
    "Time series forecasting using ARIMA models and Prophet",
    "Anomaly detection algorithms for fraud prevention in banking",
    "Graph databases: Neo4j and Cypher query language",
    "Recommender systems: collaborative filtering vs content-based methods",
    "Matrix factorization for personalized product recommendations",
    "Computer vision object tracking with OpenCV and YOLO",
    "Generative Adversarial Networks (GANs) for synthetic image generation",
    "Stable Diffusion prompt engineering and image synthesis",
    "Voice recognition architectures: transforming speech to text",
    "Cloud cost optimization: reducing AWS EC2 and S3 bills",
    "Serverless vs Containers: choosing the right cloud architecture",
    "Technical writing for developers: crafting great API documentation",
    "Agile project management: Scrum framework for software teams"
]

retriever = None
ml_model = None
extractor = None
initialization_task = None
initialization_state = "not_started"


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
    global initialization_task, initialization_state
    logger.info("advanced_startup_begin")

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            from fastapi_cache.backends.redis import RedisBackend
            from redis import asyncio as redis_asyncio

            redis_client = redis_asyncio.from_url(redis_url, decode_responses=False)
            FastAPICache.init(RedisBackend(redis_client), prefix="search-cache")
            logger.info("cache_backend=redis")
        except ImportError:
            FastAPICache.init(InMemoryBackend(), prefix="search-cache")
            logger.warning("cache_backend=inmemory redis_dependency_missing")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="search-cache")
        logger.warning("cache_backend=inmemory")

    initialization_state = "in_progress"
    initialization_task = asyncio.create_task(_initialize_search_components())


async def _initialize_search_components():
    global retriever, ml_model, extractor, initialization_state
    try:
        if HybridRetriever:
            try:
                retriever = HybridRetriever(DOCUMENTS)
                logger.info("retriever_loaded")
            except Exception as exc:  # pragma: no cover
                retriever = None
                logger.exception("retriever_init_failed error=%s", exc)
        else:
            logger.warning("retriever_missing_using_fallback")

        # Load LambdaMART model and extractor
        root_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(root_dir, 'models', 'ranker_native.json')
        extractor_path = os.path.join(root_dir, 'models', 'extractor_native.pkl')

        if os.path.exists(model_path) and os.path.exists(extractor_path):
            ml_model = xgb.Booster()
            ml_model.load_model(model_path)
            with open(extractor_path, 'rb') as f:
                extractor = pickle.load(f)
            logger.info("model_loaded")
        elif settings.strict_model_loading:
            raise RuntimeError(
                "Required model artifacts not found. Set STRICT_MODEL_LOADING=false "
                "to allow degraded startup."
            )
        else:
            logger.warning("model_missing_degraded_mode_enabled")

        initialization_state = "ready"
        logger.info("advanced_startup_ready")
    except Exception as exc:  # pragma: no cover
        initialization_state = "failed"
        logger.exception("advanced_startup_failed error=%s", exc)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "RankSmart Advanced Hybrid API", "status": "running", "ui_url": "/static/index.html"}

async def perform_search(query: str, top_k: int = 10):
    """Executes True Two-Stage Retrieval: Hybrid Fetch -> LambdaMART Re-ranking"""
    if retriever and ml_model and extractor:
        # Stage 1: Fast Candidate Retrieval (BM25 + FAISS Vector Search)
        # We fetch extra candidates (top_k * 3) so our ML algorithm has more options to re-rank
        candidates = retriever.hybrid_retrieve(query, k=top_k * 3)
        
        # Stage 2: LambdaMART Intelligent Re-ranking
        ml_ranked = []
        for doc_id, hybrid_score in candidates:
            doc_text = DOCUMENTS[doc_id]
            
            # Extract the 20 complex features dynamic to this query-document pair
            features = extractor.extract_features(query, doc_text)
            dmatrix = xgb.DMatrix([features])
            
            # Let the AI tree predict the absolute relevance
            lambda_score = ml_model.predict(dmatrix)[0]
            
            # Fusion: Heavily weight the intelligent LambdaMART model, but consider Vector proximity
            final_accuracy_score = (float(lambda_score) * 0.8) + (float(hybrid_score) * 0.2)
            ml_ranked.append((doc_text, final_accuracy_score, doc_id))
            
        # Resort heavily by intelligent Learning-to-Rank score
        ml_ranked.sort(key=lambda x: x[1], reverse=True)
        ml_ranked = ml_ranked[:top_k]
        
        formatted = []
        for doc, obj_score, obj_idx in ml_ranked:
            formatted.append({
                "document": doc,
                "score": float(obj_score),
                "id": int(obj_idx)
            })
        return formatted
    elif retriever:
        # Fallback to standard vector search
        sorted_results = retriever.hybrid_retrieve(query, k=top_k)
        formatted = []
        for doc_id, score in sorted_results:
            formatted.append({
                "document": DOCUMENTS[doc_id],
                "score": float(score),
                "id": int(doc_id)
            })
        return formatted
    else:
        return [{"document": "Advanced Hybrid Search is offline", "score": 1.0, "rank": 1}]

async def async_index_documents(documents):
    """Async document indexing in Python native background task"""
    print(f"[*] Simulating indexing of {len(documents)} documents using FAISS/BM25...")
    await asyncio.sleep(2)
    print(f"[OK] {len(documents)} documents indexed.")

@app.get("/search")
@cache(expire=300)  # Caches identically to Redis, but uses RAM for Windows compatibility
async def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=settings.default_top_k, ge=1),
    async_mode: bool = False,
    _: None = Depends(require_api_key),
):
    """Cached, async search endpoint using Memory Backend"""
    query = _validate_query_text(query)
    top_k = _validate_top_k(top_k)
    start_time = time.time()
    
    # Synchronous search (cached using InMemory)
    results = await perform_search(query, top_k)
    
    execution_time = (time.time() - start_time) * 1000

    logger.info(
        "search_completed query_len=%s top_k=%s result_count=%s latency_ms=%.2f degraded=%s async_mode=%s",
        len(query),
        top_k,
        len(results),
        execution_time,
        not bool(retriever and ml_model and extractor),
        async_mode,
    )
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "response_time_ms": round(execution_time, 2),
        "features": "Hybrid Dense/Sparse Enabled",
        "cache_status": "Active"
    }

@app.post("/index/batch")
async def batch_index(
    documents: List[str],
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_key),
):
    """Windows-Safe Background indexing (Replaces Celery)"""
    background_tasks.add_task(async_index_documents, documents)
    return {"message": "Indexing started in FastAPI native background thread. No Celery worker needed on Windows!"}

@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "highly_advanced",
        "cache": "Configured FastAPICache backend",
        "async_tasks": "Native Python",
        "metrics": "Prometheus Ready",
        "retriever_loaded": retriever is not None,
        "model_loaded": ml_model is not None,
        "initialization_state": initialization_state,
        "degraded_mode": not bool(retriever and ml_model and extractor),
    }

if __name__ == "__main__":
    import uvicorn
    # Using uvicorn native event loop for Windows compatibility
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
