"""
Train native model using your own documents
No external datasets needed!
"""
import numpy as np
import xgboost as xgb
import pickle
import os
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
    "Blockchain technology: understanding cryptocurrencies and smart contracts"
]

# Create training data by simulating queries
def generate_training_data(documents):
    """Generate synthetic training data from your documents"""
    queries = [
        ("python programming", [0, 3]),  # Doc 0 and 3 are relevant
        ("machine learning", [1, 4, 9]),  # Docs 1,4,9 relevant
        ("data science", [2, 10]),  # Docs 2,10 relevant
        ("web development", [3, 11, 16]),  # Web-related docs
        ("deep learning", [4, 0]),  # Deep learning relevant
        ("artificial intelligence", [5, 4]),  # AI relevant
        ("python tutorial", [0, 6]),  # Python tutorials
        ("cloud computing", [8, 12]),  # Cloud-related
        ("database design", [14]),  # Database docs
        ("git version control", [15])  # Git docs
    ]
    
    X_list = []
    y_list = []
    qid_list = []
    
    extractor = TextFeatureExtractor()
    extractor.fit(documents)
    
    for qid, (query, relevant_docs) in enumerate(queries):
        for doc_id, doc in enumerate(documents):
            features = extractor.extract_features(query, doc)
            X_list.append(features)
            
            # Relevance: 3 if relevant, 0 otherwise
            relevance = 3 if doc_id in relevant_docs else 0
            y_list.append(relevance)
            qid_list.append(qid)
    
    return np.array(X_list), np.array(y_list), np.array(qid_list), extractor

print("[*] Generating training data...")
X, y, qid, extractor = generate_training_data(documents)

print(f"[OK] Generated {len(X)} training samples")
print(f"   Features: {X.shape[1]}")
print(f"   Queries: {len(np.unique(qid))}")

# Train LambdaMART
print("[*] Training LambdaMART model...")
model = xgb.XGBRanker(
    objective='rank:ndcg',
    eval_metric='ndcg@10',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y, qid=qid, verbose=True)

# Save model
os.makedirs('../models', exist_ok=True)
model.save_model('../models/ranker_native.json')
print("[OK] Saved: ../models/ranker_native.json")

# Save feature extractor
with open('../models/extractor_native.pkl', 'wb') as f:
    pickle.dump(extractor, f)
print("[OK] Saved: ../models/extractor_native.pkl")

# Save feature names
with open('../models/features_native.pkl', 'wb') as f:
    pickle.dump(FEATURE_NAMES, f)
print("[OK] Saved: ../models/features_native.pkl")

# Test prediction
test_features = extractor.extract_features("python", documents[0])
score = model.predict([test_features])
print(f"\n[Test] Test prediction: {score[0]:.4f}")

print("\n[OK] Training complete! Now run the API.")
