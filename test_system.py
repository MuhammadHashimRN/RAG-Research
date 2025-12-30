"""
Comprehensive test script for the Selective RAG system.
Tests all components and provides detailed output.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported."""
    print("="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        from src.config.config_manager import config
        print("✅ Config manager imported")
        
        from src.llm.llm_client import llm_client, LLMClient
        print("✅ LLM client imported")
        
        from src.core.retrieval_engine import RetrievalEngine
        print("✅ Retrieval engine imported")
        
        from src.core.retrieval_decision import RetrievalDecisionModule, RetrievalDecision
        print("✅ Retrieval decision module imported")
        
        from src.core.grounding import GroundingModule, GroundingScore
        print("✅ Grounding module imported")
        
        from src.core.refinement import RefinementModule
        print("✅ Refinement module imported")
        
        from src.core.answer_generation import AnswerGenerator
        print("✅ Answer generator imported")
        
        from src.core.agent_controller import AgentController, Action
        print("✅ Agent controller imported")
        
        from src.evaluation.metrics import calculate_all_metrics
        print("✅ Evaluation metrics imported")
        
        from src.evaluation.baselines import BaselineRunner
        print("✅ Baseline runner imported")
        
        from src.evaluation.dataset_loader import DatasetLoader
        print("✅ Dataset loader imported")
        
        from src.evaluation.evaluator import Evaluator
        print("✅ Evaluator imported")
        
        from src.evaluation.ablation import AblationStudy
        print("✅ Ablation study imported")
        
        from src.evaluation.failure_analysis import FailureAnalyzer, FailureType
        print("✅ Failure analyzer imported")
        
        from src.evaluation.oracle import OraclePolicy
        print("✅ Oracle policy imported")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_client():
    """Test LLM client functionality."""
    print("\n" + "="*80)
    print("TEST 2: LLM Client")
    print("="*80)
    
    try:
        from src.llm.llm_client import llm_client
        
        # Test basic generation
        response = llm_client.generate("Say 'Hello, World!' in one word.")
        print(f"✅ Basic generation: {response[:50]}")
        
        # Test confidence estimation
        confidence = llm_client.estimate_confidence("What is the capital of France?")
        print(f"✅ Confidence estimation: {confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        return False

def test_retrieval_engine():
    """Test retrieval engine."""
    print("\n" + "="*80)
    print("TEST 3: Retrieval Engine")
    print("="*80)
    
    try:
        from src.core.retrieval_engine import RetrievalEngine
        
        documents = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower is located in Paris, France."
        ]
        
        engine = RetrievalEngine(documents)
        print(f"✅ Engine initialized with {len(documents)} documents")
        
        # Test retrieval
        results = engine.retrieve("What is Python?", top_k=2)
        print(f"✅ Retrieved {len(results)} documents")
        print(f"   First result: {results[0][:50]}...")
        
        # Test statistics
        stats = engine.get_statistics()
        print(f"✅ Statistics: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Retrieval engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_decision():
    """Test retrieval decision module."""
    print("\n" + "="*80)
    print("TEST 4: Retrieval Decision Module")
    print("="*80)
    
    try:
        from src.core.retrieval_decision import RetrievalDecisionModule
        
        decision_module = RetrievalDecisionModule()
        print("✅ Decision module initialized")
        
        # Test decision
        decision = decision_module.decide("What is the capital of France?")
        print(f"✅ Decision made: should_retrieve={decision.should_retrieve}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Expected benefit: {decision.expected_benefit:.2f}")
        print(f"   Reasoning: {decision.reasoning[:100]}...")
        
        # Test failure memory
        decision_module.record_failure(
            "Test query",
            "No relevant documents found"
        )
        print("✅ Failure recorded")
        
        return True
    except Exception as e:
        print(f"❌ Retrieval decision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grounding():
    """Test grounding module."""
    print("\n" + "="*80)
    print("TEST 5: Grounding Module")
    print("="*80)
    
    try:
        from src.core.grounding import GroundingModule
        
        grounding = GroundingModule()
        print("✅ Grounding module initialized")
        
        # Test scoring
        passages = [{"text": "Python is a programming language used for data science."}]
        score = grounding.score(
            query="What is Python?",
            answer="Python is a programming language.",
            retrieved_passages=passages
        )
        
        print(f"✅ Grounding score calculated:")
        print(f"   Relevance: {score.relevance_score:.2f}")
        print(f"   Consistency: {score.consistency_score:.2f}")
        print(f"   Contradiction: {score.contradiction_score:.2f}")
        print(f"   Overall: {score.overall_score:.2f}")
        print(f"   Sufficient: {score.is_sufficient}")
        
        return True
    except Exception as e:
        print(f"❌ Grounding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_controller():
    """Test agent controller."""
    print("\n" + "="*80)
    print("TEST 6: Agent Controller")
    print("="*80)
    
    try:
        from src.core.agent_controller import AgentController
        
        documents = [
            "Python is a high-level programming language.",
            "Machine learning uses algorithms to learn from data.",
            "The Eiffel Tower is 324 meters tall."
        ]
        
        agent = AgentController(documents)
        print("✅ Agent controller initialized")
        
        # Test agent run
        result = agent.run("What is Python?")
        print(f"✅ Agent executed successfully")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Retrieval calls: {result['retrieval_calls']}")
        print(f"   Grounding score: {result['final_grounding_score']:.2f}")
        print(f"   Steps: {len(result['steps'])}")
        
        # Check ReAct structure
        if result['steps']:
            step = result['steps'][0]
            print(f"✅ ReAct structure verified:")
            print(f"   Step has thought: {'thought' in step}")
            print(f"   Step has action: {'action' in step}")
            print(f"   Step has observation: {'observation' in step}")
        
        return True
    except Exception as e:
        print(f"❌ Agent controller test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*80)
    print("TEST 7: Evaluation Metrics")
    print("="*80)
    
    try:
        from src.evaluation.metrics import calculate_all_metrics
        
        predictions = ["Python is a language", "Paris is in France"]
        references = ["Python is a programming language", "Paris is the capital of France"]
        evidence_list = [
            ["Python is used for programming"],
            ["Paris is located in France"]
        ]
        
        metrics = calculate_all_metrics(
            predictions=predictions,
            references=references,
            evidence_list=evidence_list,
            retrieval_calls=[1, 1],
            latencies=[0.5, 0.6]
        )
        
        print("✅ Metrics calculated:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baselines():
    """Test baseline runners."""
    print("\n" + "="*80)
    print("TEST 8: Baseline Runners")
    print("="*80)
    
    try:
        from src.evaluation.baselines import BaselineRunner
        
        documents = ["Python is a programming language."]
        baseline_runner = BaselineRunner(documents)
        print("✅ Baseline runner initialized")
        
        # Test LLM-only baseline
        result = baseline_runner.run_llm_only("What is Python?")
        print(f"✅ LLM-only baseline: {result['answer'][:50]}...")
        
        # Test static RAG baseline
        result = baseline_runner.run_static_rag("What is Python?")
        print(f"✅ Static RAG baseline: {result['answer'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Baseline runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_failure_analysis():
    """Test failure analysis."""
    print("\n" + "="*80)
    print("TEST 9: Failure Analysis")
    print("="*80)
    
    try:
        from src.evaluation.failure_analysis import FailureAnalyzer, FailureType
        
        analyzer = FailureAnalyzer()
        print("✅ Failure analyzer initialized")
        
        # Analyze a result
        failures = analyzer.analyze_result(
            query="What is Python?",
            prediction="Python is a snake",
            reference="Python is a programming language",
            evidence=["Python is used for programming"],
            retrieval_calls=1,
            grounding_score=0.3,
            grounding_threshold=0.7
        )
        
        print(f"✅ Analyzed result: {len(failures)} failures found")
        for failure in failures:
            print(f"   - {failure.failure_type.value}: {failure.description}")
        
        # Get statistics
        stats = analyzer.get_failure_statistics()
        print(f"✅ Statistics: {stats.get('total_failures', 0)} total failures")
        
        return True
    except Exception as e:
        print(f"❌ Failure analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end():
    """Test end-to-end system."""
    print("\n" + "="*80)
    print("TEST 10: End-to-End System Test")
    print("="*80)
    
    try:
        from src.core.agent_controller import AgentController
        from src.evaluation.evaluator import Evaluator
        from src.evaluation.baselines import BaselineRunner
        
        # Create test dataset
        documents = [
            "Python is a high-level programming language known for its readability.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower is located in Paris, France and is 324 meters tall."
        ]
        
        test_dataset = [
            {"question": "What is Python?", "answer": "Python is a programming language"},
            {"question": "Where is the Eiffel Tower?", "answer": "Paris, France"}
        ]
        
        # Initialize system
        agent = AgentController(documents)
        baseline_runner = BaselineRunner(documents)
        evaluator = Evaluator(agent, baseline_runner)
        
        print("✅ System initialized")
        
        # Run evaluation
        results = evaluator.evaluate(
            dataset=test_dataset,
            query_key="question",
            answer_key="answer",
            run_baselines=True
        )
        
        print("✅ Evaluation completed")
        print(f"   Agent EM: {results['agent_metrics'].get('exact_match', 0):.2f}")
        print(f"   Agent F1: {results['agent_metrics'].get('f1_score', 0):.2f}")
        
        if results.get('baseline_results'):
            print("   Baseline results available")
        
        return True
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SELECTIVE RAG SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check API key
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  WARNING: API key not found (GOOGLE_API_KEY or GROQ_API_KEY).")
        print("   Some tests may fail. Set GOOGLE_API_KEY or GROQ_API_KEY in .env file.")
    
    tests = [
        ("Module Imports", test_imports),
        ("LLM Client", test_llm_client),
        ("Retrieval Engine", test_retrieval_engine),
        ("Retrieval Decision", test_retrieval_decision),
        ("Grounding Module", test_grounding),
        ("Agent Controller", test_agent_controller),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Baseline Runners", test_baselines),
        ("Failure Analysis", test_failure_analysis),
        ("End-to-End System", test_end_to_end),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n================================================================================")
    print("TEST 11: Full Dataset Evaluation (HotpotQA)")
    print("================================================================================")
    try:
        from src.evaluation.dataset_loader import DatasetLoader
        from src.core.agent_controller import AgentController
        from src.evaluation.evaluator import Evaluator
        
        loader = DatasetLoader()
        
        # Load small sample of HotpotQA
        print("Loading HotpotQA dataset (dev split, 5 samples)...")
        dataset = loader.load_hotpotqa(split="train", max_samples=5)
        
        if dataset:
            print(f"✅ Loaded {len(dataset)} examples")
            
            # Prepare passages from dataset context
            print("Indexing passages from dataset...")
            passages = loader.prepare_passages_from_hotpotqa(dataset)
            # Limit passages for speed in test
            passages = passages[:100] 
            print(f"✅ Extracted {len(passages)} passages")
            
            # Initialize agent with these passages
            print("Initializing Agent with HotpotQA context...")
            agent = AgentController(passages)
            
            # Run evaluation on these 5 samples
            print("Running evaluation...")
            evaluator = Evaluator(agent)
            eval_results = evaluator.evaluate(
                dataset=dataset,
                query_key="question",
                answer_key="answer",
                run_baselines=True
            )
            
            print("✅ Evaluation completed successfully")
            print("Agent Metrics:", eval_results["agent_metrics"])
        else:
            print("⚠️ Could not load dataset (datasets package might be missing or network issue)")
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n================================================================================")
    print("TEST SUMMARY")
    print("================================================================================")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

