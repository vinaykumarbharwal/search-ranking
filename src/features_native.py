"""
Native feature extraction for text documents
No Kaggle files needed - works with any text!
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class TextFeatureExtractor:
    """Extract text-based features for ranking"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.documents = []
        
    def fit(self, documents):
        """Fit on your documents"""
        self.documents = documents
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), lowercase=True)
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
        query_tokens = _tokenize(query)
        doc_tokens = _tokenize(document)
        query_words = set(query_tokens)
        doc_words = set(doc_tokens)
        overlap = len(query_words & doc_words)
        features.append(overlap / max(len(query_words), 1))
        
        # 3. Jaccard Similarity
        union = len(query_words | doc_words)
        jaccard = overlap / union if union > 0 else 0
        features.append(jaccard)
        
        # 4. Document Length (normalized)
        doc_len = len(doc_tokens)
        features.append(min(doc_len / 500, 1.0))
        
        # 5. Query Length
        features.append(min(len(query_tokens) / 10, 1.0))
        
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
        features.append(len(set(query_tokens) & set(doc_tokens)) / max(len(set(query_tokens)), 1))
        features.append(len(query) / max(len(document), 1))
        features.append(1.0 if any(c.isupper() for c in document) else 0.0)
        
        # 11. N-gram overlap
        query_bigrams = set([query_tokens[i:i+2][0] + " " + query_tokens[i:i+2][1] for i in range(len(query_tokens)-1)])
        doc_bigrams = set([doc_tokens[i:i+2][0] + " " + doc_tokens[i:i+2][1] for i in range(len(doc_tokens)-1)])
        bigram_overlap = len(query_bigrams & doc_bigrams) / max(len(query_bigrams), 1)
        features.append(bigram_overlap)
        
        # 12. Punctuation score
        punct_count = len(re.findall(r'[!?.]', document))
        features.append(min(punct_count / 5, 1.0))
        
        # 13-18. Similarity and coverage features
        features.append(tfidf_sim * overlap)  # Combined score
        features.append(overlap / max(len(doc_words), 1))  # Density
        features.append(1.0 if query_words.issubset(doc_words) else 0.0)  # All words present
        features.append(tfidf_sim * jaccard)  # Weighted similarity
        features.append(1.0 - abs(len(query) - len(document)) / 1000)  # Length similarity
        features.append(np.mean([tfidf_sim, jaccard, bigram_overlap]))  # Average similarity

        # 19. Query term frequency in document
        query_term_count = sum(doc_tokens.count(term) for term in query_words)
        query_term_freq = query_term_count / max(len(doc_tokens), 1)
        features.append(query_term_freq)

        # 20. Lightweight document quality heuristic
        lexical_diversity = len(set(doc_tokens)) / max(len(doc_tokens), 1)
        doc_quality_score = np.mean([
            lexical_diversity,
            min(len(doc_tokens) / 40, 1.0),
            1.0 - min(punct_count / 12, 1.0),
        ])
        features.append(doc_quality_score)
        
        return np.array(features)

# Create 20 feature names
FEATURE_NAMES = [
    'tfidf_cosine', 'word_overlap', 'jaccard_similarity', 'doc_length_norm',
    'query_length_norm', 'exact_match', 'position_score', 'char_overlap',
    'length_ratio', 'has_capital', 'bigram_overlap', 'punctuation_score',
    'combined_score', 'term_density', 'all_words_present', 'weighted_similarity',
    'length_similarity', 'avg_similarity', 'query_term_freq', 'doc_quality_score'
]
