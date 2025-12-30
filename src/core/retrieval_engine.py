from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.config.config_manager import config

class RetrievalEngine:
    """
    Hybrid Retrieval Engine (Dense + Sparse) with Persistence.
    
    Combines Dense Passage Retrieval (FAISS) with BM25 (Sparse) for robust retrieval.
    Supports saving and loading indices to/from disk.
    """
    
    def __init__(self, documents: List[str] = None, index_path: Optional[str] = None):
        """
        Initialize retrieval engine.
        
        Args:
            documents: List of document strings to index (if creating new)
            index_path: Path to load/save index (optional)
        """
        self.documents = []
        self.index = None
        self.bm25 = None
        self.passage_embeddings = None
        self.index_path = index_path
        
        # Get model configuration
        model_name = config.get("retrieval.dense_model_id", "sentence-transformers/all-MiniLM-L6-v2")
        self.hybrid_alpha = config.get("retrieval.hybrid_alpha", 0.5)  # 0.5 = equal weight
        
        try:
            # Query encoder (always needed for dense/hybrid)
            self.query_encoder = SentenceTransformer(model_name)
            # Passage encoder (can be different model for better performance)
            self.passage_encoder = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load model {model_name}. Retrieval will fail: {e}")
            self.query_encoder = None
            self.passage_encoder = None

        # Load from disk if path provided and exists
        if index_path and os.path.exists(os.path.join(index_path, "docs.pkl")):
            self.load_index(index_path)
        elif documents:
            self.index_documents(documents)
            if index_path:
                self.save_index(index_path)

    def index_documents(self, documents: List[str]):
        """
        Index documents using both Dense (FAISS) and Sparse (BM25) methods.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        
        # 1. Sparse Indexing (BM25)
        print(f"Building BM25 index for {len(documents)} documents...")
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Dense Indexing (FAISS)
        if not self.passage_encoder:
            return
            
        print(f"Building Dense index for {len(documents)} documents...")
        
        # Encode passages using passage encoder
        embeddings = self.passage_encoder.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.passage_embeddings = embeddings
        dimension = embeddings.shape[1]
        
        # Initialize FAISS index (L2 distance for cosine similarity after normalization)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"Indexed {self.index.ntotal} documents")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve documents using Hybrid Search (Dense + Sparse).
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
        
        Returns:
            List of retrieved document strings
        """
        if top_k is None:
            top_k = config.get("retrieval.similarity_top_k", 3)
            
        # Get Dense scores
        dense_scores, dense_indices = self._retrieve_dense(query, top_k * 2) # Get more candidates
        
        # Get Sparse scores
        sparse_scores, sparse_indices = self._retrieve_sparse(query, top_k * 2)
        
        # Normalize scores (0-1 range approx)
        if dense_scores:
            d_max, d_min = max(dense_scores.values()), min(dense_scores.values())
            if d_max != d_min:
                dense_scores = {k: (v - d_min) / (d_max - d_min) for k, v in dense_scores.items()}
        
        if sparse_scores:
            s_max, s_min = max(sparse_scores.values()), min(sparse_scores.values())
            if s_max != s_min:
                sparse_scores = {k: (v - s_min) / (s_max - s_min) for k, v in sparse_scores.items()}
        
        # Combine scores (Reciprocal Rank Fusion or Linear Combination)
        # Using Linear Combination here
        combined_scores = {}
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for idx in all_indices:
            d_score = dense_scores.get(idx, 0.0)
            s_score = sparse_scores.get(idx, 0.0)
            combined_scores[idx] = (1 - self.hybrid_alpha) * d_score + self.hybrid_alpha * s_score
            
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k docs
        results = []
        for idx, score in sorted_indices[:top_k]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
                
        return results

    def _retrieve_dense(self, query: str, top_k: int) -> Tuple[Dict[int, float], List[int]]:
        """Dense retrieval helper."""
        if not self.index or not self.query_encoder:
            return {}, []
            
        query_embedding = self.query_encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Convert distances (L2) to similarity (approx)
        # For normalized vectors, L2 = 2(1 - cos_sim), so cos_sim = 1 - L2/2
        scores = {}
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                scores[idx] = 1 - (dist / 2)
                
        return scores, indices[0]

    def _retrieve_sparse(self, query: str, top_k: int) -> Tuple[Dict[int, float], List[int]]:
        """Sparse retrieval helper (BM25)."""
        if not self.bm25:
            return {}, []
            
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        scores = {idx: doc_scores[idx] for idx in top_n_indices}
        return scores, top_n_indices

    def save_index(self, path: str):
        """Save index and documents to disk."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        # Save documents and BM25
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(path, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)
            
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(path, "dense.index"))
        print(f"Index saved to {path}")

    def load_index(self, path: str):
        """Load index from disk."""
        try:
            with open(os.path.join(path, "docs.pkl"), "rb") as f:
                self.documents = pickle.load(f)
            with open(os.path.join(path, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)
            
            if os.path.exists(os.path.join(path, "dense.index")):
                self.index = faiss.read_index(os.path.join(path, "dense.index"))
            
            print(f"Index loaded from {path} ({len(self.documents)} docs)")
        except Exception as e:
            print(f"Error loading index: {e}")

    def get_statistics(self) -> Dict:
        """Get retrieval engine statistics."""
        return {
            "num_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "has_query_encoder": self.query_encoder is not None,
            "has_bm25": self.bm25 is not None,
            "hybrid_alpha": self.hybrid_alpha
        }
