from src.llm.llm_client import llm_client
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config_manager import config

@dataclass
class GroundingScore:
    """Comprehensive grounding score."""
    relevance_score: float  # Query-passage alignment
    consistency_score: float  # Inter-passage agreement
    contradiction_score: float  # Contradiction detection (lower is better)
    overall_score: float
    is_sufficient: bool
    reasoning: str

class GroundingModule:
    def __init__(self):
        # Initialize encoder for semantic similarity
        try:
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            self.encoder = None
            print("Warning: Could not load encoder for grounding. Some features may be limited.")
        
        self.relevance_threshold = config.get("grounding.relevance_threshold", 0.7)
        self.consistency_threshold = config.get("grounding.consistency_threshold", 0.8)
        self.contradiction_threshold = 0.3  # Lower is better
        self.use_llm_for_contradiction = config.get("grounding.use_llm_for_contradiction", True)
    
    def check_grounding(self, query: str, answer: str, context: List[str]) -> float:
        """
        Checks if the answer is grounded in the context.
        Returns a score between 0.0 and 1.0 (overall score).
        """
        if not context:
            return 0.0
        
        # Convert context to passage format for comprehensive scoring
        passages = [{"text": ctx} for ctx in context]
        
        # Get comprehensive grounding score
        score = self.score(query, answer, passages)
        
        return score.overall_score
    
    def score(
        self, 
        query: str, 
        answer: str, 
        retrieved_passages: List[Dict[str, any]]
    ) -> GroundingScore:
        """
        Comprehensive grounding scoring with relevance, consistency, and contradiction.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_passages: List of passage dictionaries with 'text' key
        
        Returns:
            GroundingScore object
        """
        if not retrieved_passages:
            return GroundingScore(
                relevance_score=0.0,
                consistency_score=0.0,
                contradiction_score=1.0,
                overall_score=0.0,
                is_sufficient=False,
                reasoning="No passages retrieved"
            )
        
        # Compute individual scores
        relevance_score = self._compute_relevance(query, answer, retrieved_passages)
        consistency_score = self._compute_consistency(retrieved_passages)
        
        # Contradiction detection can be expensive - make it optional
        if self.use_llm_for_contradiction:
            contradiction_score = self._compute_contradiction(query, answer, retrieved_passages)
        else:
            # Simple heuristic: assume no contradiction if encoder-based scores are high
            contradiction_score = 0.0 if (relevance_score > 0.7 and consistency_score > 0.7) else 0.3
        
        # Compute overall score (weighted average)
        overall_score = (
            0.4 * relevance_score +
            0.4 * consistency_score +
            0.2 * (1.0 - contradiction_score)  # Lower contradiction is better
        )
        
        # Determine if sufficient
        is_sufficient = (
            relevance_score >= self.relevance_threshold and
            consistency_score >= self.consistency_threshold and
            contradiction_score <= self.contradiction_threshold
        )
        
        reasoning = (
            f"Relevance: {relevance_score:.2f}, "
            f"Consistency: {consistency_score:.2f}, "
            f"Contradiction: {contradiction_score:.2f}"
        )
        
        return GroundingScore(
            relevance_score=relevance_score,
            consistency_score=consistency_score,
            contradiction_score=contradiction_score,
            overall_score=overall_score,
            is_sufficient=is_sufficient,
            reasoning=reasoning
        )
    
    def _compute_relevance(
        self, 
        query: str, 
        answer: str, 
        passages: List[Dict[str, any]]
    ) -> float:
        """Compute relevance score (query-passage alignment)."""
        if not self.encoder:
            # Fallback to LLM-based scoring
            return self._llm_based_relevance(query, answer, passages)
        
        try:
            # Encode query and passages
            query_embedding = self.encoder.encode(query, convert_to_numpy=True)
            passage_texts = [p.get('text', '') for p in passages]
            passage_embeddings = self.encoder.encode(passage_texts, convert_to_numpy=True)
            
            # Compute cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            similarities = []
            
            for passage_emb in passage_embeddings:
                passage_norm = np.linalg.norm(passage_emb)
                if passage_norm > 0 and query_norm > 0:
                    similarity = np.dot(query_embedding, passage_emb) / (query_norm * passage_norm)
                    similarities.append(similarity)
            
            # Average similarity, normalized to [0, 1]
            if similarities:
                avg_similarity = np.mean(similarities)
                return (avg_similarity + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
            
            return 0.0
        except Exception as e:
            print(f"Error computing relevance: {e}")
            return self._llm_based_relevance(query, answer, passages)
    
    def _compute_consistency(self, passages: List[Dict[str, any]]) -> float:
        """Compute consistency score (inter-passage agreement)."""
        if len(passages) < 2:
            return 1.0  # Single passage is trivially consistent
        
        if not self.encoder:
            # Fallback to LLM-based scoring
            return self._llm_based_consistency(passages)
        
        try:
            # Encode passages
            passage_texts = [p.get('text', '') for p in passages]
            embeddings = self.encoder.encode(passage_texts, convert_to_numpy=True)
            
            # Compute pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    emb_i = embeddings[i]
                    emb_j = embeddings[j]
                    
                    norm_i = np.linalg.norm(emb_i)
                    norm_j = np.linalg.norm(emb_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                        similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                return (avg_similarity + 1.0) / 2.0  # Normalize to [0, 1]
            
            return 0.0
        except Exception as e:
            print(f"Error computing consistency: {e}")
            return self._llm_based_consistency(passages)
    
    def _compute_contradiction(
        self, 
        query: str, 
        answer: str, 
        passages: List[Dict[str, any]]
    ) -> float:
        """Compute contradiction score (lower is better)."""
        passage_texts = "\n\n".join([
            f"Passage {i+1}: {p.get('text', '')}" 
            for i, p in enumerate(passages)
        ])
        
        prompt = f"""Analyze if the following answer contradicts the retrieved passages.

Query: {query}

Retrieved Passages:
{passage_texts}

Answer: {answer}

Rate the level of contradiction on a scale of 0.0 to 1.0, where:
- 0.0 = No contradiction, answer is fully supported
- 0.5 = Some contradiction or weak support
- 1.0 = Strong contradiction, answer contradicts passages

Respond with only a number between 0.0 and 1.0."""
        
        try:
            response = llm_client.generate(prompt)
            contradiction_score = float(response.strip())
            return max(0.0, min(1.0, contradiction_score))
        except (ValueError, Exception) as e:
            print(f"Contradiction detection failed: {e}")
            return 0.5  # Default to medium contradiction
    
    def _llm_based_relevance(self, query: str, answer: str, passages: List[Dict]) -> float:
        """Fallback LLM-based relevance scoring."""
        context_text = "\n".join([p.get('text', '') for p in passages])
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n\n"
            "Rate from 0.0 to 1.0 how relevant the context is to the query. "
            "Return only the score."
        )
        
        try:
            response = llm_client.generate(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception):
            return 0.5
    
    def _llm_based_consistency(self, passages: List[Dict]) -> float:
        """Fallback LLM-based consistency scoring."""
        passage_texts = "\n\n".join([f"Passage {i+1}: {p.get('text', '')}" for i, p in enumerate(passages)])
        prompt = (
            f"Passages:\n{passage_texts}\n\n"
            "Rate from 0.0 to 1.0 how consistent these passages are with each other. "
            "Return only the score."
        )
        
        try:
            response = llm_client.generate(prompt)
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, Exception):
            return 0.5
    
    def validate_answer(
        self, 
        query: str, 
        answer: str, 
        retrieved_passages: List[Dict[str, any]]
    ) -> tuple:
        """
        Validate if answer has sufficient grounding.
        
        Returns:
            Tuple of (is_valid, grounding_score)
        """
        score = self.score(query, answer, retrieved_passages)
        return score.is_sufficient, score
