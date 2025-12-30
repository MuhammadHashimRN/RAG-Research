"""
Script to run full ablation experiments for the research paper.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from src.evaluation.dataset_loader import DatasetLoader
from src.evaluation.ablation import AblationStudy
from src.core.retrieval_engine import RetrievalEngine
from src.config.config_manager import config

def main():
    print("================================================================================")
    print("STARTING RESEARCH EXPERIMENTS (ABLATION STUDY)")
    print("================================================================================")
    
    # 1. Load Dataset
    loader = DatasetLoader()
    # Using 1000 samples as requested for robust comparison
    SAMPLE_SIZE = 1000
    print(f"Loading HotpotQA dataset (sample size: {SAMPLE_SIZE})...")
    
    # Try loading from local cache first if efficient loader implemented, else load fresh
    dataset = loader.load_hotpotqa(split="train", max_samples=SAMPLE_SIZE)
    if not dataset:
        print("❌ Failed to load dataset. Exiting.")
        return

    # 2. Index Documents (Persistent)
    index_path = "data/indices/hotpotqa_1000"
    
    if os.path.exists(os.path.join(index_path, "dense.index")):
        print(f"✅ Found existing index at {index_path}. Using it.")
        # We don't need to load it here, the agents will load it
        # But we need the documents list for the AblationStudy interface
        # Let's load just the docs to pass them along
        temp_engine = RetrievalEngine(index_path=index_path)
        passages = temp_engine.documents
    else:
        print("Creating new index (this may take a while)...")
        passages = loader.prepare_passages_from_hotpotqa(dataset)
        passages = list(set(passages)) # Deduplicate
        print(f"Total unique passages to index: {len(passages)}")
        
        # Create and save index
        engine = RetrievalEngine(documents=passages)
        engine.save_index(index_path)
        print(f"✅ Index created and saved to {index_path}")

    # 3. Configure Ablation Study
    print("Initializing Ablation Study...")
    study = AblationStudy()
    
    # Standard configs
    configs = study.get_standard_ablation_configs()
    
    print(f"Running {len(configs)} configurations:")
    for c in configs:
        print(f" - {c['name']}")
    
    # 4. Run Experiments
    # Check for existing partial results to resume
    output_path = "experiments/results/ablation_results_1000.json"
    existing_results = {}
    if os.path.exists(output_path):
        import json
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
                existing_results = data.get("results", {})
            print(f"🔄 Resuming from existing results. Found {len(existing_results)} completed configs.")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}")

    # We pass 'passages' as 'documents' so the agents can index them.
    # We will modify run_ablation to accept existing results and skip them
    # Since we can't easily modify the method signature in this turn, we'll implement the loop here manually
    # leveraging the AblationStudy helper but managing the loop ourselves.
    
    final_results = existing_results.copy()
    study.results = final_results
    
    for config_variant in configs:
        variant_name = config_variant.get('name', 'unknown')
        
        if variant_name in final_results:
            print(f"⏩ Skipping {variant_name} (already completed)")
            continue
            
        print(f"\n{'='*80}")
        print(f"Running ablation: {variant_name}")
        print(f"{'='*80}")
        
        # Create agent
        agent = study._create_agent_with_config(config_variant, passages)
        from src.evaluation.baselines import BaselineRunner
        from src.evaluation.evaluator import Evaluator
        
        baseline_runner = BaselineRunner(passages)
        evaluator = Evaluator(agent, baseline_runner)
        
        # Run evaluation with progress saving
        try:
            # We run in smaller batches if needed, but Evaluator runs full dataset.
            # Ideally we'd batch inside evaluator, but for now let's rely on the LLM cache 
            # to make restarts fast.
            variant_results = evaluator.evaluate(
                dataset=dataset,
                query_key="question",
                answer_key="answer",
                run_baselines=False
            )
            
            final_results[variant_name] = {
                "config": config_variant,
                "metrics": variant_results["agent_metrics"],
                "num_samples": len(dataset)
            }
            
            # Save progress immediately after each config variant
            study.results = final_results
            study.save_results(output_path)
            
            # Sleep a bit to let API cool down between variants
            print("💤 Cooling down API for 10 seconds...")
            time.sleep(10)
            
        except Exception as e:
            print(f"❌ Error running {variant_name}: {e}")
            # Try to save what we have
            study.save_results(output_path)
    
    # 5. Output Results
    print("\n================================================================================")
    print("EXPERIMENT COMPLETE")
    print("================================================================================")
    
    study.print_comparison()
    
    # Save to file
    output_path = "experiments/results/ablation_results_1000.json"
    study.save_results(output_path)
    print(f"Full results saved to {output_path}")

if __name__ == "__main__":
    main()
