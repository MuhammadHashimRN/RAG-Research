from src.core.agent_controller import AgentController
from src.evaluation.metrics import calculate_all_metrics, exact_match_score, f1_score
from src.evaluation.baselines import BaselineRunner
from src.evaluation.failure_analysis import FailureAnalyzer
from src.evaluation.oracle import OraclePolicy
from typing import List, Dict, Optional
import numpy as np
import time
import json
from pathlib import Path

class Evaluator:
    def __init__(
        self, 
        agent: AgentController, 
        baseline_runner: Optional[BaselineRunner] = None,
        enable_failure_analysis: bool = True,
        enable_oracle: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            agent: Agent controller to evaluate
            baseline_runner: Optional baseline runner for comparison
            enable_failure_analysis: Whether to perform failure analysis
            enable_oracle: Whether to use oracle policy for decision quality
        """
        self.agent = agent
        self.baseline_runner = baseline_runner
        self.failure_analyzer = FailureAnalyzer() if enable_failure_analysis else None
        self.oracle_policy = None
        if enable_oracle and hasattr(agent, 'retrieval_engine'):
            retrieval_engine = agent.retrieval_engine
            if retrieval_engine and hasattr(retrieval_engine, 'documents'):
                documents = retrieval_engine.documents
                self.oracle_policy = OraclePolicy(documents) if documents else None
    
    def evaluate(
        self, 
        dataset: List[Dict],
        query_key: str = "question",
        answer_key: str = "answer",
        run_baselines: bool = True
    ) -> Dict:
        """
        Evaluate agent on dataset.
        
        Args:
            dataset: List of examples with query and answer
            query_key: Key for query in dataset items
            answer_key: Key for answer in dataset items
            run_baselines: Whether to run baseline methods
        
        Returns:
            Dictionary with evaluation results
        """
        results = {
            "agent_results": [],
            "agent_metrics": {},
            "baseline_results": {}
        }
        
        # Agent evaluation
        predictions = []
        references = []
        evidence_list = []
        decisions = []
        retrieval_calls = []
        latencies = []
        oracle_decisions = []
        
        for i, item in enumerate(dataset):
            query = item[query_key]
            reference = item[answer_key]
            
            print(f"Processing {i+1}/{len(dataset)}: {query[:50]}...")
            
            start_time = time.time()
            output = self.agent.run(query)
            latency = time.time() - start_time
            
            predictions.append(output['answer'])
            references.append(reference)
            evidence_list.append(output.get('context', []))
            decisions.append(output.get('retrieval_calls', 0) > 0)
            retrieval_calls.append(output.get('retrieval_calls', 0))
            latencies.append(latency)
            
            # Get oracle decision if available
            if self.oracle_policy:
                oracle_decision = self.oracle_policy.get_oracle_decision(query, reference)
                oracle_decisions.append(oracle_decision)
            
            # Perform failure analysis
            if self.failure_analyzer:
                self.failure_analyzer.analyze_result(
                    query=query,
                    prediction=output['answer'],
                    reference=reference,
                    evidence=output.get('context', []),
                    retrieval_calls=output.get('retrieval_calls', 0),
                    grounding_score=output.get('final_grounding_score', 0.0),
                    grounding_threshold=0.7,
                    metadata=output
                )
            
            results["agent_results"].append({
                "query": query,
                "prediction": output['answer'],
                "reference": reference,
                "evidence": output.get('context', []),
                "retrieval_calls": output.get('retrieval_calls', 0),
                "latency": latency,
                "grounding_score": output.get('final_grounding_score', 0.0)
            })
        
        # Calculate agent metrics
        results["agent_metrics"] = calculate_all_metrics(
            predictions=predictions,
            references=references,
            evidence_list=evidence_list,
            decisions=decisions,
            oracle_decisions=oracle_decisions if oracle_decisions else None,
            retrieval_calls=retrieval_calls,
            latencies=latencies
        )
        
        # Add failure analysis if enabled
        if self.failure_analyzer:
            results["failure_analysis"] = self.failure_analyzer.get_failure_statistics()
        
        # Baseline evaluation
        if run_baselines and self.baseline_runner:
            print("\nRunning baseline methods...")
            baseline_results = self._run_baselines(dataset, query_key, answer_key)
            results["baseline_results"] = baseline_results
        
        return results
    
    def _run_baselines(
        self,
        dataset: List[Dict],
        query_key: str,
        answer_key: str
    ) -> Dict[str, Dict]:
        """Run baseline methods."""
        baseline_results = {}
        
        baseline_methods = ["llm_only", "static_rag", "react_always_retrieve"]
        
        for method_name in baseline_methods:
            print(f"Running {method_name} baseline...")
            
            predictions = []
            references = []
            evidence_list = []
            retrieval_calls = []
            latencies = []
            
            for item in dataset:
                query = item[query_key]
                reference = item[answer_key]
                
                try:
                    if method_name == "llm_only":
                        result = self.baseline_runner.run_llm_only(query)
                    elif method_name == "static_rag":
                        result = self.baseline_runner.run_static_rag(query)
                    elif method_name == "react_always_retrieve":
                        result = self.baseline_runner.run_react_always_retrieve(query)
                    else:
                        continue
                    
                    predictions.append(result["answer"])
                    references.append(reference)
                    evidence_list.append(result.get("evidence", []))
                    retrieval_calls.append(result.get("retrieval_calls", 0))
                    latencies.append(result.get("latency", 0))
                except Exception as e:
                    print(f"Error in {method_name} for query '{query[:50]}...': {e}")
                    predictions.append("")
                    references.append(reference)
                    evidence_list.append([])
                    retrieval_calls.append(0)
                    latencies.append(0)
            
            # Calculate metrics
            metrics = calculate_all_metrics(
                predictions=predictions,
                references=references,
                evidence_list=evidence_list,
                retrieval_calls=retrieval_calls,
                latencies=latencies
            )
            
            baseline_results[method_name] = {
                "predictions": predictions,
                "metrics": metrics
            }
        
        return baseline_results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print("\nAgent Metrics:")
        for metric, value in results["agent_metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        if results.get("baseline_results"):
            print("\nBaseline Comparisons:")
            for baseline_name, baseline_data in results["baseline_results"].items():
                print(f"\n  {baseline_name}:")
                for metric, value in baseline_data["metrics"].items():
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")
    
    def save_results(self, results: Dict, filepath: str):
        """Save evaluation results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {filepath}")
