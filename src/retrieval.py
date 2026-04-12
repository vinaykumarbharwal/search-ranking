"""
Candidate retrieval module with BM25 and optional semantic search
"""
import pickle
import os
from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss

class HybridRetriever:
    """Hybrid retrieval combining BM25 and semantic search"""
    
    def __init__(self, use_semantic: bool = True):
        self.bm25 = None
        self.documents = []
        self.use_semantic = use_semantic
        
        if use_semantic:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.semantic_index = None
    
    def fit(self, documents: List[str]):
        """Fit retriever on documents"""
        self.documents = documents
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        if self.use_semantic:
            embeddings = self.semantic_model.encode(documents)
            dimension = embeddings.shape[1]
            self.semantic_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.semantic_index.add(embeddings.astype('float32'))
    
    def retrieve_bm25(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Retrieve using BM25"""
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
    
    def retrieve_semantic(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Retrieve using semantic search"""
        if not self.use_semantic:
            return []
        
        query_embedding = self.semantic_model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.semantic_index.search(query_embedding.astype('float32'), top_k)
        return [(idx, float(scores[0][i])) for i, idx in enumerate(indices[0])]
    
    def hybrid_search(self, query: str, top_k: int = 50, 
                      bm25_weight: float = 0.5) -> List[Tuple[int, float]]:
        """Hybrid retrieval combining BM25 and semantic search"""
        bm25_results = dict(self.retrieve_bm25(query, top_k * 2))
        
        if self.use_semantic:
            semantic_results = dict(self.retrieve_semantic(query, top_k * 2))
            
            all_docs = set(bm25_results.keys()) | set(semantic_results.keys())
            final_scores = {}
            
            for doc_id in all_docs:
                bm25_score = bm25_results.get(doc_id, 0)
                semantic_score = semantic_results.get(doc_id, 0)
                final_scores[doc_id] = (bm25_weight * bm25_score + 
                                       (1 - bm25_weight) * semantic_score)
        else:
            final_scores = bm25_results
        
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def save(self, path: str):
        """Save retriever to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'use_semantic': self.use_semantic,
                'bm25': self.bm25
            }, f)
    
    def load(self, path: str):
        """Load retriever from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.use_semantic = data['use_semantic']
        self.bm25 = data['bm25']
        
        if self.use_semantic:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = self.semantic_model.encode(self.documents)
            dimension = embeddings.shape[1]
            self.semantic_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.semantic_index.add(embeddings.astype('float32'))
