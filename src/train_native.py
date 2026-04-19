"""
Train native model using your own documents
No external datasets needed!
"""
import numpy as np
import xgboost as xgb
import pickle
import os
import re
from features_native import TextFeatureExtractor, FEATURE_NAMES

# Sample documents (your searchable content)
documents = [
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
    "Data visualization with Python and matplotlib for data science",
    "React hooks tutorial: modern React development for web apps",
    "Docker containerization for developers: complete guide",
    "Kubernetes orchestration: deploying microservices at scale",
    "SQL database design and optimization for backend developers",
    "Git version control: collaborative development best practices",
    "REST API design: building scalable web services",
    "Cybersecurity fundamentals: protecting your applications",
    "Mobile app development with React Native cross-platform",
    "Blockchain technology: understanding cryptocurrencies and smart contracts",
    "System design interview prep: load balancing and horizontal scaling patterns",
    "GraphQL API development with Apollo Server and React clients",
    "Linux command line essentials: bash scripting and process management",
    "Microservices architecture: service discovery and distributed tracing",
    "CI CD pipelines with GitHub Actions for automated deployment",
    "PostgreSQL performance tuning: indexing strategies and query planning",
    "NoSQL databases with MongoDB: schema design and aggregation framework",
    "Natural language processing with transformers and attention mechanisms",
    "Computer vision fundamentals using OpenCV and convolutional neural networks",
    "MLOps workflow: model versioning monitoring and feature stores",
    "Cloud security best practices for AWS IAM and network policies",
    "TypeScript for large JavaScript applications with strict typing",
    "Advanced CSS techniques: grid layout flexbox and responsive design systems",
    "Node.js backend engineering: event loop streams and async patterns",
    "FastAPI production guide: dependency injection validation and middleware",
    "Redis caching strategies for low latency APIs and session storage",
    "Elasticsearch basics: full text search analyzers and relevance tuning",
    "Data engineering with Apache Spark for batch and streaming pipelines",
    "Kubernetes observability: Prometheus Grafana and alerting setup",
    "Software testing pyramid: unit integration and end to end testing",
    "Secure authentication with OAuth2 JWT tokens and refresh token rotation",
    "Domain driven design: bounded contexts aggregates and ubiquitous language",
    "Event driven architecture with Kafka topics partitions and consumers",
    "Async Python programming with asyncio task scheduling and concurrency",
    "C++ memory management: smart pointers RAII and performance optimization",
    "Go language concurrency model with goroutines channels and context",
    "Data structures and algorithms interview preparation with practical examples",
    "SRE fundamentals: SLIs SLOs error budgets and incident response",
    "Frontend performance optimization: code splitting lazy loading and caching"
]

# Create training data by simulating queries
def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_queries_for_doc(doc):
    tokens = _tokenize(doc)
    if not tokens:
        return []

    # Build multiple query styles per document to improve generalization.
    unigram_query = " ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0]
    bigram_query = " ".join(tokens[2:5]) if len(tokens) >= 5 else " ".join(tokens[: min(len(tokens), 3)])
    tail_query = " ".join(tokens[-3:]) if len(tokens) >= 3 else " ".join(tokens)

    queries = [unigram_query, bigram_query, tail_query]
    return [q.strip() for q in queries if q.strip()]


def _graded_relevance(query, doc, is_primary):
    query_terms = set(_tokenize(query))
    doc_terms = set(_tokenize(doc))
    overlap = len(query_terms & doc_terms)
    coverage = overlap / max(len(query_terms), 1)
    exact_phrase = query.lower() in doc.lower()

    if is_primary:
        return 4 if exact_phrase else 3
    if exact_phrase and coverage >= 0.67:
        return 3
    if coverage >= 0.67:
        return 2
    if coverage >= 0.34:
        return 1
    return 0


def generate_training_data(documents):
    """Generate pseudo-labeled graded ranking data from your documents."""
    query_doc_pairs = []
    for doc_id, doc in enumerate(documents):
        for query in _build_queries_for_doc(doc):
            query_doc_pairs.append((query, doc_id))

    # Deduplicate while preserving order.
    seen = set()
    deduped_pairs = []
    for pair in query_doc_pairs:
        if pair[0] not in seen:
            deduped_pairs.append(pair)
            seen.add(pair[0])
    
    X_list = []
    y_list = []
    qid_list = []
    
    extractor = TextFeatureExtractor()
    extractor.fit(documents)
    
    for qid, (query, primary_doc_id) in enumerate(deduped_pairs):
        for doc_id, doc in enumerate(documents):
            features = extractor.extract_features(query, doc)
            X_list.append(features)

            relevance = _graded_relevance(query, doc, is_primary=(doc_id == primary_doc_id))
            y_list.append(relevance)
            qid_list.append(qid)
    
    return np.array(X_list), np.array(y_list), np.array(qid_list), extractor

print("[*] Generating training data...")
X, y, qid, extractor = generate_training_data(documents)

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(repo_root, 'models')

print(f"[OK] Generated {len(X)} training samples")
print(f"   Features: {X.shape[1]}")
print(f"   Queries: {len(np.unique(qid))}")

# Train LambdaMART
print("[*] Training LambdaMART model...")
model = xgb.XGBRanker(
    objective='rank:ndcg',
    eval_metric='ndcg@10',
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=2,
    reg_lambda=1.5,
    n_estimators=220,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y, qid=qid, verbose=True)

# Save model
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'ranker_native.json')
model.save_model(model_path)
print(f"[OK] Saved: {model_path}")

# Save feature extractor
extractor_path = os.path.join(models_dir, 'extractor_native.pkl')
with open(extractor_path, 'wb') as f:
    pickle.dump(extractor, f)
print(f"[OK] Saved: {extractor_path}")

# Save feature names
features_path = os.path.join(models_dir, 'features_native.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(FEATURE_NAMES, f)
print(f"[OK] Saved: {features_path}")

# Test prediction
test_features = extractor.extract_features("python", documents[0])
score = model.predict([test_features])
print(f"\n[Test] Test prediction: {score[0]:.4f}")

print("\n[OK] Training complete! Now run the API.")
