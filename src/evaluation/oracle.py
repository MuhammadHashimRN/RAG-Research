"""
Oracle Policy for determining optimal retrieval decisions.
Used for calculating decision quality metrics.
"""

from typing import List, Dict, Optional
from src.core.retrieval_engine import RetrievalEngine
from src.core.answer_generation import AnswerGenerator
from src.llm.llm_client import llm_client

class OraclePolicy:
    """
    Oracle policy that determines optimal retrieval decisions.
    
    Uses ground truth to determine if retrieval was necessary.
    """
    
    def __init__(self, documents: Optional[List[str]] = None):
        """
        Initialize oracle policy.
        
        Args:
            documents: Documents for retrieval (optional)
        """
        self.retrieval_engine = RetrievalEngine(documents) if documents else None
        self.answer_generator = AnswerGenerator()
    
    def get_oracle_decision(
        self,
        query: str,
        ground_truth: str,
        documents: Optional[List[str]] = None
    ) -> bool:
        """
        Determine if retrieval was necessary (oracle decision).
        
        Args:
            query: User query
            ground_truth: Ground truth answer
            documents: Optional documents for retrieval
        
        Returns:
            True if retrieval was necessary, False otherwise
        """
        if not self.retrieval_engine and not documents:
            # No retrieval available, decision is always False
            return False
        
        # Strategy: Try answering without retrieval, then with retrieval
        # If answer improves significantly with retrieval, it was necessary
        
        # Answer without retrieval
        answer_no_retrieval = self.answer_generator.generate_answer(query, context=None)
        
        # Answer with retrieval
        if documents:
            temp_engine = RetrievalEngine(documents)
            retrieved = temp_engine.retrieve(query, top_k=3)
        elif self.retrieval_engine:
            retrieved = self.retrieval_engine.retrieve(query, top_k=3)
        else:
            retrieved = []
        
        answer_with_retrieval = self.answer_generator.generate_answer(query, context=retrieved)
        
        # Compare both answers to ground truth
        from src.evaluation.metrics import f1_score
        
        f1_no_retrieval = f1_score(answer_no_retrieval, ground_truth)
        f1_with_retrieval = f1_score(answer_with_retrieval, ground_truth)
        
        # If retrieval significantly improves answer quality, it was necessary
        improvement = f1_with_retrieval - f1_no_retrieval
        
        # Threshold: if improvement > 0.1, retrieval was necessary
        return improvement > 0.1
    
    def get_oracle_decisions(
        self,
        queries: List[str],
        ground_truths: List[str],
        documents: Optional[List[str]] = None
    ) -> List[bool]:
        """
        Get oracle decisions for multiple queries.
        
        Args:
            queries: List of queries
            ground_truths: List of ground truth answers
            documents: Optional documents for retrieval
        
        Returns:
            List of oracle decisions (True/False)
        """
        decisions = []
        
        for query, ground_truth in zip(queries, ground_truths):
            decision = self.get_oracle_decision(query, ground_truth, documents)
            decisions.append(decision)
        
        return decisions

