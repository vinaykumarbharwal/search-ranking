import numpy as np
import re


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class HybridRetriever:
    """Multi-stage retrieval: BM25 + Dense + Reranking"""
    
    def __init__(self, documents):
        self.documents = documents
        
        try:
            import faiss
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
            
            # Build BM25 index (Case-insensitive)
            tokenized_docs = [_tokenize(doc) for doc in documents]
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
        tokenized_query = _tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        valid_indices = np.where(scores > 0)[0]
        if len(valid_indices) == 0:
            return []
            
        sorted_valid = valid_indices[np.argsort(scores[valid_indices])[-k:][::-1]]
        sorted_scores = scores[sorted_valid]
        max_score = float(np.max(sorted_scores)) if len(sorted_scores) else 1.0
        min_score = float(np.min(sorted_scores)) if len(sorted_scores) else 0.0
        score_span = max(max_score - min_score, 1e-9)

        normalized = [
            (int(idx), float((scores[idx] - min_score) / score_span))
            for idx in sorted_valid
        ]
        return normalized
    
    def retrieve_dense(self, query, k=100):
        """Stage 2: Dense retrieval"""
        import faiss
        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.dense_index.search(query_emb.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # Filter out weak semantic matches (Cosine similarity < 0.15 is basically noise)
            if int(idx) < 0:
                continue
            score = float(scores[0][i])
            if score > 0.15:
                # Inner product is roughly in [-1,1] after normalization; map into [0,1].
                results.append((int(idx), (score + 1.0) / 2.0))
        return results
    
    def hybrid_retrieve(self, query, k=100, bm25_weight=0.3, dense_weight=0.7):
        """Stage 3: Hybrid fusion with weighted normalized scores + rank smoothing."""
        bm25_ranked = self.retrieve_bm25(query, k * 3)
        dense_ranked = self.retrieve_dense(query, k * 3)

        bm25_scores = {doc_id: score for doc_id, score in bm25_ranked}
        dense_scores = {doc_id: score for doc_id, score in dense_ranked}

        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_ranked)}
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranked)}

        all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
        rrf_scores = {}

        for doc_id in all_docs:
            bm25_rank = bm25_ranks.get(doc_id, 1e6)
            dense_rank = dense_ranks.get(doc_id, 1e6)

            # RRF component stabilizes ranking quality under noisy scores.
            rrf_score = (1 / (10 + bm25_rank)) + (1 / (10 + dense_rank))

            lexical_semantic = (
                bm25_weight * bm25_scores.get(doc_id, 0.0)
                + dense_weight * dense_scores.get(doc_id, 0.0)
            )

            final_score = (lexical_semantic * 0.85) + (rrf_score * 0.15)
            rrf_scores[doc_id] = final_score

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
