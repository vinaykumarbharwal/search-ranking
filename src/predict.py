"""
Inference module for ranking predictions
"""
import numpy as np
import xgboost as xgb
import pickle
from typing import List, Tuple, Dict
from retrieval import HybridRetriever
from features import FeatureExtractor

class SearchRanker:
    """Main search ranking system"""
    
    def __init__(self, model_path: str = 'models/ranker.json',
                 features_path: str = 'models/features.pkl',
                 scaler_path: str = 'models/scaler.pkl'):
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load feature extractor
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.load(features_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Initialize retriever
        self.retriever = HybridRetriever(use_semantic=True)
    
    def index_documents(self, documents: List[str]):
        """Index documents for retrieval"""
        self.retriever.fit(documents)
        self.feature_extractor.fit(documents)
        self.documents = documents
    
    def rank(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Rank documents for a query"""
        
        # Step 1: Retrieve candidates
        candidates = self.retriever.hybrid_search(query, top_k=50)
        
        if not candidates:
            return []
        
        # Step 2: Extract features
        doc_ids = [c[0] for c in candidates]
        candidate_docs = [self.documents[doc_id] for doc_id in doc_ids]
        
        features_list = []
        for doc_id, doc in zip(doc_ids, candidate_docs):
            features = self.feature_extractor.extract_query_doc_features(
                query, doc, doc_id
            )
            features_list.append(features)
        
        # Convert to matrix
        X = np.array([[f[name] for name in self.feature_extractor.feature_names] 
                     for f in features_list])
        X_scaled = self.scaler.transform(X)
        
        # Step 3: Predict scores
        dmatrix = xgb.DMatrix(X_scaled)
        scores = self.model.predict(dmatrix)
        
        # Step 4: Sort by score
        ranked_results = sorted(zip(candidate_docs, scores, features_list), 
                               key=lambda x: x[1], reverse=True)
        
        return ranked_results[:top_k]
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search with detailed results"""
        results = self.rank(query, top_k)
        
        return [
            {
                'document': doc,
                'score': float(score),
                'features': features
            }
            for doc, score, features in results
        ]

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Learn Python programming from basics to advanced with examples",
        "Complete machine learning course with Python and scikit-learn",
        "Data science bootcamp: pandas, numpy, matplotlib tutorial",
        "Web development with React and Node.js full stack guide",
        "Deep learning specialization: neural networks and TensorFlow",
        "Artificial intelligence fundamentals: search algorithms and logic",
        "Python for beginners: getting started with Python programming",
        "Java programming: object-oriented programming masterclass",
        "AWS cloud computing: architecture and deployment guide",
    ]
    
    # Initialize ranker
    ranker = SearchRanker()
    ranker.index_documents(documents)
    
    # Search
    query = "python programming tutorial"
    results = ranker.search(query, top_k=5)
    
    print(f"\nQuery: {query}\n")
    print("Top Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['document'][:80]}...")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Features: TF-IDF Cosine={result['features']['tfidf_cosine']:.3f}, "
              f"Overlap={result['features']['word_overlap_ratio']:.3f}\n")
