from typing import List, Dict, Optional
import re

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Calculate Exact Match (EM) score."""
    return prediction.strip().lower() == ground_truth.strip().lower()

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score (token overlap)."""
    pred_tokens = set(prediction.strip().lower().split())
    truth_tokens = set(ground_truth.strip().lower().split())
    
    if not truth_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    common = pred_tokens & truth_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def evidence_precision(
    prediction: str,
    evidence: List[str],
    ground_truth: str
) -> float:
    """
    Calculate evidence precision (how well evidence supports answer).
    
    Args:
        prediction: Predicted answer
        evidence: List of evidence passages
        ground_truth: Ground truth answer
    
    Returns:
        Evidence precision score between 0.0 and 1.0
    """
    if not evidence:
        return 0.0
    
    # Check if key terms from prediction appear in evidence
    pred_tokens = set(_normalize_text(prediction).split())
    evidence_text = " ".join(evidence)
    evidence_tokens = set(_normalize_text(evidence_text).split())
    
    # Calculate overlap
    overlap = len(pred_tokens & evidence_tokens)
    if len(pred_tokens) == 0:
        return 0.0
    
    return overlap / len(pred_tokens)

def hallucination_rate(
    predictions: List[str],
    evidence_list: List[List[str]]
) -> float:
    """
    Estimate hallucination rate.
    
    Args:
        predictions: List of predicted answers
        evidence_list: List of evidence lists for each prediction
    
    Returns:
        Estimated hallucination rate between 0.0 and 1.0
    """
    if len(predictions) != len(evidence_list):
        return 1.0  # Mismatch indicates potential issues
    
    hallucination_count = 0
    
    for pred, evidence in zip(predictions, evidence_list):
        if not evidence:
            # No evidence = potential hallucination
            hallucination_count += 1
            continue
        
        # Check if key entities/facts in answer appear in evidence
        pred_tokens = set(_normalize_text(pred).split())
        evidence_text = " ".join(evidence)
        evidence_tokens = set(_normalize_text(evidence_text).split())
        
        # Simple heuristic: if >50% of answer tokens not in evidence
        overlap = len(pred_tokens & evidence_tokens)
        if len(pred_tokens) > 0:
            overlap_ratio = overlap / len(pred_tokens)
            if overlap_ratio < 0.5:
                hallucination_count += 1
    
    return hallucination_count / len(predictions) if predictions else 0.0

def retrieval_decision_accuracy(
    decisions: List[bool],
    oracle_decisions: List[bool]
) -> float:
    """
    Calculate retrieval decision accuracy compared to oracle.
    
    Args:
        decisions: List of retrieval decisions (True/False)
        oracle_decisions: List of oracle decisions (True/False)
    
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    if len(decisions) != len(oracle_decisions):
        return 0.0
    
    correct = sum(1 for d, o in zip(decisions, oracle_decisions) if d == o)
    return correct / len(decisions) if decisions else 0.0

def unnecessary_retrieval_rate(
    decisions: List[bool],
    oracle_decisions: List[bool]
) -> float:
    """
    Calculate rate of unnecessary retrievals.
    
    Returns:
        Rate of retrievals when oracle says not needed
    """
    if len(decisions) != len(oracle_decisions):
        return 0.0
    
    unnecessary = sum(
        1 for d, o in zip(decisions, oracle_decisions) 
        if d and not o
    )
    total_retrievals = sum(decisions)
    
    return unnecessary / total_retrievals if total_retrievals > 0 else 0.0

def missed_retrieval_rate(
    decisions: List[bool],
    oracle_decisions: List[bool]
) -> float:
    """
    Calculate rate of missed retrievals.
    
    Returns:
        Rate of not retrieving when oracle says needed
    """
    if len(decisions) != len(oracle_decisions):
        return 0.0
    
    missed = sum(
        1 for d, o in zip(decisions, oracle_decisions) 
        if not d and o
    )
    total_should_retrieve = sum(oracle_decisions)
    
    return missed / total_should_retrieve if total_should_retrieve > 0 else 0.0

def calculate_all_metrics(
    predictions: List[str],
    references: List[str],
    evidence_list: Optional[List[List[str]]] = None,
    decisions: Optional[List[bool]] = None,
    oracle_decisions: Optional[List[bool]] = None,
    retrieval_calls: Optional[List[int]] = None,
    latencies: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Accuracy metrics
    em_scores = [exact_match_score(p, r) for p, r in zip(predictions, references)]
    f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]
    
    metrics["exact_match"] = sum(em_scores) / len(em_scores) if em_scores else 0.0
    metrics["f1_score"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Evidence metrics
    if evidence_list:
        evidence_precisions = [
            evidence_precision(p, e, r) 
            for p, e, r in zip(predictions, evidence_list, references)
        ]
        metrics["evidence_precision"] = (
            sum(evidence_precisions) / len(evidence_precisions) 
            if evidence_precisions else 0.0
        )
        metrics["hallucination_rate"] = hallucination_rate(predictions, evidence_list)
    
    # Decision quality metrics
    if decisions is not None and oracle_decisions is not None:
        metrics["retrieval_decision_accuracy"] = retrieval_decision_accuracy(
            decisions, oracle_decisions
        )
        metrics["unnecessary_retrieval_rate"] = unnecessary_retrieval_rate(
            decisions, oracle_decisions
        )
        metrics["missed_retrieval_rate"] = missed_retrieval_rate(
            decisions, oracle_decisions
        )
    
    # Efficiency metrics
    if retrieval_calls:
        metrics["avg_retrieval_calls"] = sum(retrieval_calls) / len(retrieval_calls)
        metrics["total_retrieval_calls"] = sum(retrieval_calls)
    
    if latencies:
        metrics["avg_latency"] = sum(latencies) / len(latencies)
        metrics["total_latency"] = sum(latencies)

    # Token usage metrics
    # We expect 'tokens' to be passed in some way or we need to add it to the signature.
    # For now, let's assume it might be in 'kwargs' or we add it to the signature in next step.
    # To avoid breaking signature now, we will wait for a dedicated call or update signature below.
    pass
    
    return metrics

def calculate_all_metrics_with_tokens(
    predictions: List[str],
    references: List[str],
    evidence_list: Optional[List[List[str]]] = None,
    decisions: Optional[List[bool]] = None,
    oracle_decisions: Optional[List[bool]] = None,
    retrieval_calls: Optional[List[int]] = None,
    latencies: Optional[List[float]] = None,
    token_counts: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics including tokens.
    """
    metrics = calculate_all_metrics(
        predictions, references, evidence_list, decisions, 
        oracle_decisions, retrieval_calls, latencies
    )
    
    if token_counts:
        metrics["avg_tokens"] = sum(token_counts) / len(token_counts)
        metrics["total_tokens"] = sum(token_counts)
        
    return metrics

def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text
