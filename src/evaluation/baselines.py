from typing import List, Dict
import time
from src.core.retrieval_engine import RetrievalEngine
from src.core.answer_generation import AnswerGenerator
from src.llm.llm_client import llm_client

class BaselineRunner:
    """Runs baseline methods for comparison."""
    
    def __init__(self, documents: List[str] = None):
        """
        Initialize baseline runner.
        
        Args:
            documents: Documents for retrieval (if None, no retrieval available)
        """
        self.retrieval_engine = RetrievalEngine(documents) if documents else None
        self.answer_generator = AnswerGenerator()
    
    def run_llm_only(self, query: str) -> Dict:
        """
        Run LLM-only baseline (no retrieval).
        
        Args:
            query: User query
        
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        answer = self.answer_generator.generate_answer(query, context=None)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer,
            "evidence": [],
            "retrieval_calls": 0,
            "latency": latency,
            "method": "llm_only"
        }
    
    def run_static_rag(self, query: str) -> Dict:
        """
        Run static RAG baseline (always retrieve).
        
        Args:
            query: User query
        
        Returns:
            Dictionary with answer and metadata
        """
        if not self.retrieval_engine:
            raise ValueError("Retrieval engine required for static RAG baseline")
        
        start_time = time.time()
        
        # Always retrieve
        retrieved_passages = self.retrieval_engine.retrieve(query)
        
        # Generate answer
        answer = self.answer_generator.generate_answer(query, context=retrieved_passages)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer,
            "evidence": retrieved_passages,
            "retrieval_calls": 1,
            "latency": latency,
            "method": "static_rag"
        }
    
    def run_react_always_retrieve(self, query: str) -> Dict:
        """
        Run ReAct baseline with always-retrieve policy.
        
        Args:
            query: User query
        
        Returns:
            Dictionary with answer and metadata
        """
        if not self.retrieval_engine:
            raise ValueError("Retrieval engine required for ReAct baseline")
        
        start_time = time.time()
        retrieval_calls = 0
        
        # Simple ReAct loop: always retrieve, then generate
        retrieved_passages = self.retrieval_engine.retrieve(query)
        retrieval_calls += 1
        
        # Generate answer
        answer = self.answer_generator.generate_answer(query, context=retrieved_passages)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer,
            "evidence": retrieved_passages,
            "retrieval_calls": retrieval_calls,
            "latency": latency,
            "method": "react_always_retrieve"
        }
    
    def run_all_baselines(self, query: str) -> Dict[str, Dict]:
        """
        Run all baseline methods.
        
        Args:
            query: User query
        
        Returns:
            Dictionary mapping baseline name to results
        """
        results = {}
        
        # LLM-only (always available)
        results["llm_only"] = self.run_llm_only(query)
        
        # Retrieval-based baselines (if retrieval engine available)
        if self.retrieval_engine:
            results["static_rag"] = self.run_static_rag(query)
            results["react_always_retrieve"] = self.run_react_always_retrieve(query)
        
        return results

