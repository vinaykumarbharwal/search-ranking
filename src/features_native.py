"""
Native feature extraction for text documents
No Kaggle files needed - works with any text!
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class TextFeatureExtractor:
    """Extract text-based features for ranking"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.documents = []
        
    def fit(self, documents):
        """Fit on your documents"""
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_vectorizer.fit(documents)
        
    def extract_features(self, query, document):
        """Extract 20 text features on the fly"""
        features = []
        
        # 1. TF-IDF Cosine Similarity
        query_tfidf = self.tfidf_vectorizer.transform([query])
        doc_tfidf = self.tfidf_vectorizer.transform([document])
        tfidf_sim = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
        features.append(tfidf_sim)
        
        # 2. Word Overlap
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        overlap = len(query_words & doc_words)
        features.append(overlap / max(len(query_words), 1))
        
        # 3. Jaccard Similarity
        union = len(query_words | doc_words)
        jaccard = overlap / union if union > 0 else 0
        features.append(jaccard)
        
        # 4. Document Length (normalized)
        doc_len = len(document.split())
        features.append(min(doc_len / 500, 1.0))
        
        # 5. Query Length
        features.append(min(len(query.split()) / 10, 1.0))
        
        # 6. Exact Match Bonus
        exact_match = 1.0 if query.lower() in document.lower() else 0.0
        features.append(exact_match)
        
        # 7. Title Position (if query appears early)
        query_lower = query.lower()
        doc_lower = document.lower()
        position = doc_lower.find(query_lower)
        position_score = 1.0 - min(position / 200, 1.0) if position >= 0 else 0
        features.append(position_score)
        
        # 8-10. Character-level features
        features.append(len(set(query) & set(document)) / max(len(set(query)), 1))
        features.append(len(query) / max(len(document), 1))
        features.append(1.0 if any(c.isupper() for c in document) else 0.0)
        
        # 11-13. N-gram overlaps
        query_bigrams = set([query[i:i+2] for i in range(len(query)-1)])
        doc_bigrams = set([document[i:i+2] for i in range(len(document)-1)])
        bigram_overlap = len(query_bigrams & doc_bigrams) / max(len(query_bigrams), 1)
        features.append(bigram_overlap)
        
        # 14. Punctuation score
        punct_count = len(re.findall(r'[!?.]', document))
        features.append(min(punct_count / 5, 1.0))
        
        # 15-20. Fill remaining with similarity scores
        features.append(tfidf_sim * overlap)  # Combined score
        features.append(overlap / max(len(doc_words), 1))  # Density
        features.append(1.0 if query_words.issubset(doc_words) else 0.0)  # All words present
        features.append(tfidf_sim * jaccard)  # Weighted similarity
        features.append(1.0 - abs(len(query) - len(document)) / 1000)  # Length similarity
        features.append(np.mean([tfidf_sim, jaccard, bigram_overlap]))  # Average similarity
        
        return np.array(features)

# Create 20 feature names
FEATURE_NAMES = [
    'tfidf_cosine', 'word_overlap', 'jaccard_similarity', 'doc_length_norm',
    'query_length_norm', 'exact_match', 'position_score', 'char_overlap',
    'length_ratio', 'has_capital', 'bigram_overlap', 'punctuation_score',
    'combined_score', 'term_density', 'all_words_present', 'weighted_similarity',
    'length_similarity', 'avg_similarity', 'query_term_freq', 'doc_quality_score'
]
