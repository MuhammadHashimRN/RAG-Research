from src.llm.llm_client import llm_client
from src.config.config_manager import config
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RetrievalDecision:
    """Represents a retrieval decision with metadata."""
    should_retrieve: bool
    confidence: float
    reasoning: str
    expected_benefit: float

class FailureMemory:
    """Memory of past retrieval failures for learning."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.failures: List[Dict] = []
    
    def add_failure(self, query: str, reason: str, context: Optional[Dict] = None):
        """Add a failure to memory."""
        failure = {
            "query": query,
            "reason": reason,
            "context": context or {}
        }
        self.failures.append(failure)
        
        # Trim if exceeds max size
        if len(self.failures) > self.max_size:
            self.failures = self.failures[-self.max_size:]
    
    def get_similar_failures(self, query: str, top_k: int = 3) -> List[Dict]:
        """Get similar past failures (simple keyword-based)."""
        # Return recent failures (could be enhanced with semantic similarity)
        return self.failures[-top_k:] if self.failures else []
    
    def clear(self):
        """Clear failure memory."""
        self.failures = []

class RetrievalDecisionModule:
    def __init__(self):
        self.template_path = "templates/retrieval_decision.txt"
        self._load_template()
        self.failure_memory = FailureMemory(max_size=100)

    def _load_template(self):
        if os.path.exists(self.template_path):
            with open(self.template_path, 'r') as f:
                self.template = f.read()
        else:
            self.template = "Query: {query}\nNeed retrieval? (YES/NO)"

    def decide(
        self, 
        query: str, 
        model_confidence: Optional[float] = None,
        context: Optional[Dict] = None
    ) -> RetrievalDecision:
        """
        Decide whether retrieval is necessary.
        
        Args:
            query: User query
            model_confidence: LLM's confidence in answering without retrieval
            context: Additional context
        
        Returns:
            RetrievalDecision object
        """
        if not config.get("retrieval.use_selective_retrieval", True):
            return RetrievalDecision(
                should_retrieve=True,
                confidence=1.0,
                reasoning="Selective retrieval disabled",
                expected_benefit=1.0
            )
        
        # Get model confidence if not provided
        # Skip confidence estimation to save LLM calls (can use simple heuristic)
        if model_confidence is None:
            # Simple heuristic: assume medium confidence for common queries
            # Can enable full estimation if needed: model_confidence = llm_client.estimate_confidence(query)
            model_confidence = 0.5  # Default to medium confidence
        
        # Get similar past failures
        similar_failures = self.failure_memory.get_similar_failures(query)
        
        # Build decision prompt
        prompt = self._build_decision_prompt(query, model_confidence, similar_failures)
        
        # Get LLM decision
        response = llm_client.generate(prompt)
        
        # Parse response
        should_retrieve = "YES" in response.upper()
        
        # Calculate expected benefit (inverse of confidence)
        expected_benefit = 1.0 - model_confidence if should_retrieve else 0.0
        
        # Generate reasoning
        reasoning = self._extract_reasoning(response, model_confidence)
        
        return RetrievalDecision(
            should_retrieve=should_retrieve,
            confidence=abs(model_confidence - 0.5) * 2,  # Normalize to 0-1
            reasoning=reasoning,
            expected_benefit=expected_benefit
        )
    
    def _build_decision_prompt(
        self, 
        query: str, 
        model_confidence: float,
        similar_failures: List[Dict]
    ) -> str:
        """Build prompt for retrieval decision."""
        prompt = self.template.format(query=query)
        
        # Add confidence information
        prompt += f"\n\nModel Confidence (without retrieval): {model_confidence:.2f}"
        
        # Add failure history if available
        if similar_failures:
            prompt += "\n\nRecent similar queries that required retrieval:"
            for failure in similar_failures[:3]:
                prompt += f"\n- {failure['query']}: {failure['reason']}"
        
        return prompt
    
    def _extract_reasoning(self, response: str, confidence: float) -> str:
        """Extract reasoning from LLM response."""
        if len(response) > 200:
            return response[:200] + "..."
        return response if response else f"Confidence: {confidence:.2f}"
    
    def record_failure(self, query: str, reason: str, context: Optional[Dict] = None):
        """Record a retrieval failure for future learning."""
        self.failure_memory.add_failure(query, reason, context)
