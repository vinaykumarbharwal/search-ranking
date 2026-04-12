import numpy as np

class HybridRetriever:
    """Multi-stage retrieval: BM25 + Dense + Reranking"""
    
    def __init__(self, documents):
        self.documents = documents
        
        try:
            import faiss
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
            
            # Build BM25 index (Case-insensitive)
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
            
            # Build FAISS dense index
            self.encoder = SentenceTransformer('msmarco-distilbert-base-v4')
            embeddings = self.encoder.encode(documents, show_progress_bar=True)
            self.dense_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.dense_index.add(embeddings.astype('float32'))
        except ImportError:
            print("Warning: Install faiss-cpu and sentence-transformers to use HybridRetriever.")
            self.bm25_index = None
            self.dense_index = None
    
    def retrieve_bm25(self, query, k=100):
        """Stage 1: BM25 retrieval"""
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        valid_indices = np.where(scores > 0)[0]
        if len(valid_indices) == 0:
            return []
            
        sorted_valid = valid_indices[np.argsort(scores[valid_indices])[-k:][::-1]]
        return [(idx, scores[idx]) for idx in sorted_valid]
    
    def retrieve_dense(self, query, k=100):
        """Stage 2: Dense retrieval"""
        import faiss
        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.dense_index.search(query_emb.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # Filter out weak semantic matches (Cosine similarity < 0.15 is basically noise)
            if float(scores[0][i]) > 0.15:
                results.append((idx, scores[0][i]))
        return results
    
    def hybrid_retrieve(self, query, k=100, bm25_weight=0.3, dense_weight=0.7):
        """Stage 3: Hybrid fusion via Reciprocal Rank Fusion"""
        bm25_results = dict(self.retrieve_bm25(query, k*2))
        dense_results = dict(self.retrieve_dense(query, k*2))
        
        all_docs = set(bm25_results.keys()) | set(dense_results.keys())
        rrf_scores = {}
        
        for doc_id in all_docs:
            bm25_rank = list(bm25_results.keys()).index(doc_id) + 1 if doc_id in bm25_results else 1e6
            dense_rank = list(dense_results.keys()).index(doc_id) + 1 if doc_id in dense_results else 1e6
            
            # RRF formula
            rrf_score = (1 / (10 + bm25_rank)) + (1 / (10 + dense_rank))
            
            # Add a slight boost from the raw dense score for exact distinguishing
            raw_dense = dense_results.get(doc_id, 0)
            final_score = (rrf_score * 5) + (raw_dense * 0.1)
            
            rrf_scores[doc_id] = final_score
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
