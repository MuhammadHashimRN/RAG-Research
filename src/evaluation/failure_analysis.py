"""
Failure Analysis Framework for categorizing and analyzing system failures.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class FailureType(str, Enum):
    """Types of failures."""
    RETRIEVAL_FAILURE = "retrieval_failure"
    DECISION_FAILURE = "decision_failure"
    GROUNDING_FAILURE = "grounding_failure"
    REFINEMENT_FAILURE = "refinement_failure"
    ANSWER_GENERATION_FAILURE = "answer_generation_failure"

@dataclass
class Failure:
    """Represents a system failure."""
    failure_type: FailureType
    query: str
    description: str
    context: Dict[str, Any]
    severity: float  # 0.0 to 1.0

class FailureAnalyzer:
    """
    Analyzes and categorizes system failures.
    
    Categories failures according to PRD Section 13:
    - Retrieval failure
    - Decision failure
    - Grounding failure
    - Refinement failure
    """
    
    def __init__(self):
        """Initialize failure analyzer."""
        self.failures: List[Failure] = []
    
    def analyze_result(
        self,
        query: str,
        prediction: str,
        reference: str,
        evidence: List[str],
        retrieval_calls: int,
        grounding_score: float,
        grounding_threshold: float = 0.7,
        metadata: Optional[Dict] = None
    ) -> List[Failure]:
        """
        Analyze a single result and identify failures.
        
        Args:
            query: User query
            prediction: Predicted answer
            reference: Ground truth answer
            evidence: Retrieved evidence
            retrieval_calls: Number of retrieval calls
            grounding_score: Final grounding score
            grounding_threshold: Grounding threshold
            metadata: Additional metadata
        
        Returns:
            List of identified failures
        """
        failures = []
        metadata = metadata or {}
        
        # 1. Retrieval Failure
        if self._is_retrieval_failure(query, prediction, reference, evidence, retrieval_calls):
            failures.append(Failure(
                failure_type=FailureType.RETRIEVAL_FAILURE,
                query=query,
                description="Failed to retrieve relevant information",
                context={
                    "retrieval_calls": retrieval_calls,
                    "evidence_count": len(evidence),
                    "prediction": prediction,
                    "reference": reference
                },
                severity=self._calculate_retrieval_severity(query, prediction, reference, evidence)
            ))
        
        # 2. Decision Failure
        if self._is_decision_failure(query, prediction, reference, evidence, retrieval_calls):
            failures.append(Failure(
                failure_type=FailureType.DECISION_FAILURE,
                query=query,
                description="Incorrect retrieval decision made",
                context={
                    "retrieval_calls": retrieval_calls,
                    "should_have_retrieved": len(evidence) == 0 and retrieval_calls == 0,
                    "prediction": prediction,
                    "reference": reference
                },
                severity=self._calculate_decision_severity(query, prediction, reference, evidence, retrieval_calls)
            ))
        
        # 3. Grounding Failure
        if self._is_grounding_failure(grounding_score, grounding_threshold, prediction, reference):
            failures.append(Failure(
                failure_type=FailureType.GROUNDING_FAILURE,
                query=query,
                description="Answer lacks sufficient grounding in evidence",
                context={
                    "grounding_score": grounding_score,
                    "threshold": grounding_threshold,
                    "evidence_count": len(evidence),
                    "prediction": prediction
                },
                severity=1.0 - grounding_score
            ))
        
        # 4. Refinement Failure
        if self._is_refinement_failure(metadata, grounding_score, grounding_threshold):
            failures.append(Failure(
                failure_type=FailureType.REFINEMENT_FAILURE,
                query=query,
                description="Refinement did not improve answer quality",
                context={
                    "grounding_score": grounding_score,
                    "threshold": grounding_threshold,
                    "refinement_iterations": metadata.get("refinement_iterations", 0)
                },
                severity=0.5  # Medium severity
            ))
        
        # 5. Answer Generation Failure
        if self._is_answer_generation_failure(prediction, reference):
            failures.append(Failure(
                failure_type=FailureType.ANSWER_GENERATION_FAILURE,
                query=query,
                description="Generated answer is incorrect or incomplete",
                context={
                    "prediction": prediction,
                    "reference": reference,
                    "grounding_score": grounding_score
                },
                severity=self._calculate_answer_severity(prediction, reference)
            ))
        
        self.failures.extend(failures)
        return failures
    
    def _is_retrieval_failure(
        self,
        query: str,
        prediction: str,
        reference: str,
        evidence: List[str],
        retrieval_calls: int
    ) -> bool:
        """Check if retrieval failed."""
        # Failure if: retrieved but no relevant evidence, or should have retrieved but didn't
        if retrieval_calls > 0 and len(evidence) == 0:
            return True
        
        # Check if answer is wrong and might have been helped by retrieval
        if retrieval_calls == 0 and not self._answer_matches(prediction, reference):
            # Simple heuristic: if answer is wrong, might be retrieval failure
            return True
        
        return False
    
    def _is_decision_failure(
        self,
        query: str,
        prediction: str,
        reference: str,
        evidence: List[str],
        retrieval_calls: int
    ) -> bool:
        """Check if retrieval decision was wrong."""
        # Oracle: if answer is wrong without retrieval, should have retrieved
        if retrieval_calls == 0 and not self._answer_matches(prediction, reference):
            # Check if query seems to need external knowledge
            if self._needs_external_knowledge(query):
                return True
        
        # Oracle: if retrieved but answer is still wrong, might not have needed retrieval
        if retrieval_calls > 0 and not self._answer_matches(prediction, reference):
            # Could be unnecessary retrieval (but this is harder to determine)
            pass
        
        return False
    
    def _is_grounding_failure(
        self,
        grounding_score: float,
        threshold: float,
        prediction: str,
        reference: str
    ) -> bool:
        """Check if grounding failed."""
        return grounding_score < threshold
    
    def _is_refinement_failure(
        self,
        metadata: Dict,
        grounding_score: float,
        threshold: float
    ) -> bool:
        """Check if refinement failed."""
        refinement_iterations = metadata.get("refinement_iterations", 0)
        if refinement_iterations > 0 and grounding_score < threshold:
            return True
        return False
    
    def _is_answer_generation_failure(
        self,
        prediction: str,
        reference: str
    ) -> bool:
        """Check if answer generation failed."""
        return not self._answer_matches(prediction, reference)
    
    def _answer_matches(self, prediction: str, reference: str) -> bool:
        """Check if prediction matches reference."""
        pred_normalized = prediction.strip().lower()
        ref_normalized = reference.strip().lower()
        return pred_normalized == ref_normalized
    
    def _needs_external_knowledge(self, query: str) -> bool:
        """Heuristic to determine if query needs external knowledge."""
        # Simple keyword-based heuristic
        knowledge_keywords = [
            "what is", "who is", "when did", "where is", "how many",
            "capital", "population", "founded", "located"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in knowledge_keywords)
    
    def _calculate_retrieval_severity(
        self,
        query: str,
        prediction: str,
        reference: str,
        evidence: List[str]
    ) -> float:
        """Calculate severity of retrieval failure."""
        if not self._answer_matches(prediction, reference):
            return 1.0  # High severity if answer is wrong
        return 0.5  # Medium severity otherwise
    
    def _calculate_decision_severity(
        self,
        query: str,
        prediction: str,
        reference: str,
        evidence: List[str],
        retrieval_calls: int
    ) -> float:
        """Calculate severity of decision failure."""
        if not self._answer_matches(prediction, reference):
            return 1.0
        return 0.3
    
    def _calculate_answer_severity(
        self,
        prediction: str,
        reference: str
    ) -> float:
        """Calculate severity of answer generation failure."""
        if not self._answer_matches(prediction, reference):
            return 1.0
        return 0.0
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about failures.
        
        Returns:
            Dictionary with failure statistics
        """
        if not self.failures:
            return {}
        
        stats = {
            "total_failures": len(self.failures),
            "by_type": {},
            "average_severity": 0.0,
            "severity_by_type": {}
        }
        
        # Count by type
        for failure_type in FailureType:
            type_failures = [f for f in self.failures if f.failure_type == failure_type]
            stats["by_type"][failure_type.value] = len(type_failures)
            
            if type_failures:
                avg_severity = sum(f.severity for f in type_failures) / len(type_failures)
                stats["severity_by_type"][failure_type.value] = avg_severity
        
        # Overall average severity
        if self.failures:
            stats["average_severity"] = sum(f.severity for f in self.failures) / len(self.failures)
        
        return stats
    
    def print_statistics(self):
        """Print failure statistics."""
        stats = self.get_failure_statistics()
        
        print("\n" + "="*80)
        print("FAILURE ANALYSIS")
        print("="*80)
        
        print(f"\nTotal Failures: {stats.get('total_failures', 0)}")
        print(f"Average Severity: {stats.get('average_severity', 0.0):.2f}")
        
        print("\nFailures by Type:")
        for failure_type, count in stats.get("by_type", {}).items():
            severity = stats.get("severity_by_type", {}).get(failure_type, 0.0)
            print(f"  {failure_type}: {count} (avg severity: {severity:.2f})")
    
    def save_analysis(self, filepath: str):
        """Save failure analysis to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "failures": [
                {
                    "type": f.failure_type.value,
                    "query": f.query,
                    "description": f.description,
                    "context": f.context,
                    "severity": f.severity
                }
                for f in self.failures
            ],
            "statistics": self.get_failure_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nFailure analysis saved to {filepath}")

