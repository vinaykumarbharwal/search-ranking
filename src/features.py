"""
Feature engineering module for Learning-to-Rank
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
from typing import List, Tuple, Dict
import pickle
import os

class FeatureExtractor:
    """Extract ranking features for query-document pairs"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF vectorizer on documents"""
        self.tfidf_vectorizer.fit(documents)
        self.is_fitted = True
    
    def extract_query_doc_features(self, query: str, document: str, 
                                   doc_id: int = None) -> Dict[str, float]:
        """Extract all features for a query-document pair"""
        features = {}
        
        # 1. TF-IDF Features
        if self.is_fitted:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            doc_tfidf = self.tfidf_vectorizer.transform([document])
            features['tfidf_cosine'] = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
            features['tfidf_euclidean'] = np.linalg.norm(query_tfidf.toarray() - doc_tfidf.toarray())
        
        # 2. Length features
        features['query_len'] = len(query)
        features['doc_len'] = len(document)
        features['doc_len_log'] = np.log1p(len(document))
        features['len_ratio'] = len(query) / (len(document) + 1)
        
        # 3. Word overlap features
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        features['word_overlap_count'] = len(query_words & doc_words)
        features['word_overlap_ratio'] = len(query_words & doc_words) / (len(query_words) + 1)
        features['jaccard_similarity'] = len(query_words & doc_words) / (len(query_words | doc_words) + 1)
        
        # 4. BM25-like features (simplified)
        query_terms = query.lower().split()
        doc_term_freq = Counter(document.lower().split())
        bm25_score = 0
        doc_len = len(doc_words)
        avg_doc_len = 100  # approximate
        
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                idf = np.log((1000 + 1) / (1 + 1))  # simplified IDF
                bm25_score += idf * (tf * (1.2 + 1)) / (tf + 1.2 * (1 - 0.75 + 0.75 * doc_len / avg_doc_len))
        features['bm25_simple'] = bm25_score
        
        # 5. Position-based features (if doc_id provided)
        if doc_id is not None:
            features['doc_position'] = doc_id
        
        # 6. Query term frequency in document
        features['query_term_tf_sum'] = sum(doc_term_freq.get(term, 0) for term in query_terms)
        features['query_term_tf_avg'] = features['query_term_tf_sum'] / (len(query_terms) + 1)
        
        # 7. Character-level features
        features['char_overlap'] = len(set(query) & set(document)) / (len(set(query)) + 1)
        
        # 8. N-gram overlap (bigrams)
        query_bigrams = set([query[i:i+2] for i in range(len(query)-1)])
        doc_bigrams = set([document[i:i+2] for i in range(len(document)-1)])
        features['bigram_overlap'] = len(query_bigrams & doc_bigrams) / (len(query_bigrams) + 1)
        
        # 9. Document quality signals
        features['has_capital'] = 1 if any(c.isupper() for c in document) else 0
        features['has_numbers'] = 1 if any(c.isdigit() for c in document) else 0
        features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', document)) / (len(document) + 1)
        
        # 10. Query-document embedding similarity (placeholder - can be enhanced)
        features['semantic_similarity'] = features['tfidf_cosine']  # placeholder
        
        return features
    
    def extract_batch(self, query: str, documents: List[str], 
                      doc_ids: List[int] = None) -> np.ndarray:
        """Extract features for multiple documents"""
        features_list = []
        
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else None
            features = self.extract_query_doc_features(query, doc, doc_id)
            features_list.append(features)
        
        return self._dict_to_matrix(features_list)
    
    def _dict_to_matrix(self, features_list: List[Dict]) -> np.ndarray:
        """Convert list of feature dicts to numpy matrix"""
        if not features_list:
            return np.array([])
        
        # Get all feature names
        feature_names = sorted(features_list[0].keys())
        self.feature_names = feature_names
        
        # Create matrix
        matrix = np.array([[f[name] for name in feature_names] 
                          for f in features_list])
        return matrix
    
    def save(self, path: str):
        """Save feature extractor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'is_fitted': self.is_fitted,
                'feature_names': getattr(self, 'feature_names', [])
            }, f)
    
    def load(self, path: str):
        """Load feature extractor from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.is_fitted = data['is_fitted']
        self.feature_names = data.get('feature_names', [])
