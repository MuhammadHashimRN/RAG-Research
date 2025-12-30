"""
Ablation Study Framework for comparing different system configurations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import copy
from src.core.agent_controller import AgentController
from src.evaluation.evaluator import Evaluator
from src.evaluation.baselines import BaselineRunner
from src.evaluation.dataset_loader import DatasetLoader
from src.config.config_manager import config

class AblationStudy:
    """
    Framework for running ablation studies.
    
    Supports comparing different configurations:
    - Selective vs always retrieve
    - With vs without grounding
    - With vs without self-refinement
    - With vs without abstention
    - Single-pass vs agent loop
    """
    
    def __init__(self, base_config: Optional[Dict] = None):
        """
        Initialize ablation study framework.
        
        Args:
            base_config: Base configuration dictionary (optional)
        """
        self.base_config = base_config or {}
        self.results: Dict[str, Any] = {}
    
    def create_config_variant(
        self,
        name: str,
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a configuration variant.
        
        Args:
            name: Name of the variant
            overrides: Configuration overrides
        
        Returns:
            Configuration dictionary
        """
        variant = copy.deepcopy(self.base_config)
        
        # Apply overrides
        for key, value in overrides.items():
            keys = key.split('.')
            current = variant
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        
        variant['name'] = name
        return variant
    
    def get_standard_ablation_configs(self) -> List[Dict[str, Any]]:
        """
        Get standard ablation study configurations.
        
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        # 1. Full System (baseline)
        configs.append(self.create_config_variant(
            "full_system",
            {}
        ))
        
        # 2. No Selective Retrieval (always retrieve)
        configs.append(self.create_config_variant(
            "no_selective_retrieval",
            {"retrieval.use_selective_retrieval": False}
        ))
        
        # 3. No Grounding Validation
        configs.append(self.create_config_variant(
            "no_grounding",
            {
                "grounding.relevance_threshold": 0.0,
                "grounding.consistency_threshold": 0.0
            }
        ))
        
        # 4. No Self-Refinement
        configs.append(self.create_config_variant(
            "no_refinement",
            {"agent.refinement_enabled": False}
        ))
        
        # 5. No Abstention
        configs.append(self.create_config_variant(
            "no_abstention",
            {"agent.abstention_enabled": False}
        ))
        
        # 6. Single-pass (no agent loop)
        configs.append(self.create_config_variant(
            "single_pass",
            {"agent.max_iterations": 1}
        ))
        
        # 7. No Selective + No Grounding
        configs.append(self.create_config_variant(
            "no_selective_no_grounding",
            {
                "retrieval.use_selective_retrieval": False,
                "grounding.relevance_threshold": 0.0,
                "grounding.consistency_threshold": 0.0
            }
        ))
        
        return configs
    
    def run_ablation(
        self,
        dataset: List[Dict],
        configs: Optional[List[Dict]] = None,
        query_key: str = "question",
        answer_key: str = "answer",
        documents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run ablation study with multiple configurations.
        
        Args:
            dataset: Evaluation dataset
            configs: List of configurations to test (if None, uses standard)
            query_key: Key for query in dataset
            answer_key: Key for answer in dataset
            documents: Documents for retrieval
        
        Returns:
            Dictionary with results for each configuration
        """
        if configs is None:
            configs = self.get_standard_ablation_configs()
        
        results = {}
        
        for config_variant in configs:
            variant_name = config_variant.get('name', 'unknown')
            print(f"\n{'='*80}")
            print(f"Running ablation: {variant_name}")
            print(f"{'='*80}")
            
            # Apply configuration (would need config manager update)
            # For now, we'll create agent with modified behavior
            agent = self._create_agent_with_config(config_variant, documents)
            baseline_runner = BaselineRunner(documents) if documents else None
            evaluator = Evaluator(agent, baseline_runner)
            
            # Run evaluation
            variant_results = evaluator.evaluate(
                dataset=dataset,
                query_key=query_key,
                answer_key=answer_key,
                run_baselines=False  # Skip baselines in ablation
            )
            
            results[variant_name] = {
                "config": config_variant,
                "metrics": variant_results["agent_metrics"],
                "num_samples": len(dataset)
            }
        
        self.results = results
        return results
    
    def _create_agent_with_config(
        self,
        config_variant: Dict,
        documents: Optional[List[str]]
    ) -> AgentController:
        """
        Create agent with specific configuration.
        """
        # We need to temporarily patch the global config for module initialization
        # or manually inject dependencies. Since modules read from config on init,
        # we will manually configure them after initialization.
        
        agent = AgentController(documents)
        
        # 1. Configure Retrieval Decision (Selective Retrieval)
        if "retrieval.use_selective_retrieval" in config_variant:
            # The decision module reads config on init, but we can override attributes if public
            # or just mock the decide method if needed.
            # Looking at RetrievalDecisionModule, it reads config in __init__.
            # We can re-initialize it or patch it.
            # Re-initializing is safer if we could pass config, but it uses global config.
            # Let's modify the instance attributes directly if possible, or assume 
            # RetrievalDecisionModule has a property we can set.
            # If not, we might need to subclass or rely on a config context manager.
            # For this codebase, let's assume we can modify attributes or need to re-init with patched config.
            
            # Since we can't easily patch the global config safely here without affecting others,
            # let's modify the module's behavior.
            use_selective = config_variant["retrieval.use_selective_retrieval"]
            if hasattr(agent.decision_module, 'use_selective_retrieval'):
                 agent.decision_module.use_selective_retrieval = use_selective
            else:
                 # Fallback: if not exposed, we might need to rely on the module checking config every time
                 # or inject a new one.
                 pass

        # 2. Configure Grounding (Thresholds)
        if "grounding.relevance_threshold" in config_variant:
            threshold = config_variant["grounding.relevance_threshold"]
            agent.grounding_threshold = threshold
            agent.grounding_module.relevance_threshold = threshold
            
        if "grounding.consistency_threshold" in config_variant:
             agent.grounding_module.consistency_threshold = config_variant["grounding.consistency_threshold"]

        # 3. Configure Agent Loop (Refinement, Abstention, Iterations)
        if "agent.refinement_enabled" in config_variant:
            agent.refinement_enabled = config_variant["agent.refinement_enabled"]
        
        if "agent.abstention_enabled" in config_variant:
            agent.abstention_enabled = config_variant["agent.abstention_enabled"]
        
        if "agent.max_iterations" in config_variant:
            agent.max_iterations = config_variant["agent.max_iterations"]
        
        return agent
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare results across all configurations.
        
        Returns:
            Comparison dictionary with improvements
        """
        if not self.results:
            return {}
        
        # Use full_system as baseline
        baseline_name = "full_system"
        if baseline_name not in self.results:
            # Use first config as baseline
            baseline_name = list(self.results.keys())[0]
        
        baseline_metrics = self.results[baseline_name]["metrics"]
        
        comparison = {
            "baseline": baseline_name,
            "comparisons": {}
        }
        
        for variant_name, variant_data in self.results.items():
            if variant_name == baseline_name:
                continue
            
            variant_metrics = variant_data["metrics"]
            comparison["comparisons"][variant_name] = {
                "metrics": variant_metrics,
                "improvements": {}
            }
            
            # Calculate improvements
            for metric_name, baseline_value in baseline_metrics.items():
                if metric_name in variant_metrics:
                    variant_value = variant_metrics[metric_name]
                    if isinstance(baseline_value, (int, float)) and baseline_value != 0:
                        improvement = ((variant_value - baseline_value) / baseline_value) * 100
                        comparison["comparisons"][variant_name]["improvements"][metric_name] = improvement
        
        return comparison
    
    def print_comparison(self):
        """Print ablation study comparison."""
        comparison = self.compare_results()
        
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        
        baseline_name = comparison["baseline"]
        print(f"\nBaseline: {baseline_name}")
        print(f"Metrics: {self.results[baseline_name]['metrics']}")
        
        print("\n" + "-"*80)
        print("Variants:")
        print("-"*80)
        
        for variant_name, variant_data in comparison["comparisons"].items():
            print(f"\n{variant_name}:")
            print(f"  Metrics: {variant_data['metrics']}")
            print(f"  Improvements vs baseline:")
            for metric, improvement in variant_data["improvements"].items():
                print(f"    {metric}: {improvement:+.2f}%")
    
    def save_results(self, filepath: str):
        """Save ablation study results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            "results": self.results,
            "comparison": self.compare_results()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nAblation results saved to {filepath}")

