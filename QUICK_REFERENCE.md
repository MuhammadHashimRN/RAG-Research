# Quick Reference Guide

## 🚀 Running Tests

```bash
# Run comprehensive test suite
python test_system.py
```

## 📊 Basic Usage Examples

### 1. Simple Query
```python
from src.core.agent_controller import AgentController

documents = [
    "Python is a programming language.",
    "Machine learning uses algorithms."
]

agent = AgentController(documents)
result = agent.run("What is Python?")
print(result['answer'])
```

### 2. Full Evaluation
```python
from src.core.agent_controller import AgentController
from src.evaluation.evaluator import Evaluator
from src.evaluation.baselines import BaselineRunner
from src.evaluation.dataset_loader import DatasetLoader

# Load data
loader = DatasetLoader()
examples = loader.load_hotpotqa(split="dev", max_samples=10)
passages = loader.prepare_passages_from_hotpotqa(examples)

# Evaluate
agent = AgentController(documents=passages)
baseline_runner = BaselineRunner(documents=passages)
evaluator = Evaluator(agent, baseline_runner)

results = evaluator.evaluate(examples, run_baselines=True)
evaluator.print_summary(results)
```

### 3. Ablation Study
```python
from src.evaluation.ablation import AblationStudy
from src.evaluation.dataset_loader import DatasetLoader

loader = DatasetLoader()
examples = loader.load_hotpotqa(split="dev", max_samples=10)
passages = loader.prepare_passages_from_hotpotqa(examples)

ablation = AblationStudy()
results = ablation.run_ablation(examples, documents=passages)
ablation.print_comparison()
ablation.save_results("results/ablation.json")
```

### 4. Failure Analysis
```python
from src.evaluation.failure_analysis import FailureAnalyzer

analyzer = FailureAnalyzer()

# Automatically integrated in evaluator, or use manually:
failures = analyzer.analyze_result(
    query="...",
    prediction="...",
    reference="...",
    evidence=["..."],
    retrieval_calls=1,
    grounding_score=0.5
)

analyzer.print_statistics()
analyzer.save_analysis("results/failures.json")
```

## 🔧 Configuration

Edit `config/default_config.yaml`:

```yaml
retrieval:
  use_selective_retrieval: true
  similarity_top_k: 3
  dense_model_id: "sentence-transformers/all-MiniLM-L6-v2"

agent:
  max_iterations: 3
  refinement_enabled: true
  abstention_enabled: true

grounding:
  relevance_threshold: 0.7
  consistency_threshold: 0.8

llm:
  model_name: "gemini-2.5-pro"
  temperature: 0.0
```

## 📈 Key Metrics

- **Exact Match (EM)**: Exact answer match
- **F1 Score**: Token overlap
- **Evidence Precision**: Evidence support quality
- **Hallucination Rate**: Estimated hallucination
- **Retrieval Decision Accuracy**: vs oracle
- **Unnecessary Retrieval Rate**: Efficiency metric
- **Missed Retrieval Rate**: Coverage metric

## 🎯 Failure Categories

1. **Retrieval Failure**: Failed to retrieve relevant info
2. **Decision Failure**: Incorrect retrieval decision
3. **Grounding Failure**: Insufficient grounding
4. **Refinement Failure**: Refinement didn't help
5. **Answer Generation Failure**: Incorrect answer

## 📝 File Locations

- **Core Modules**: `src/core/`
- **Evaluation**: `src/evaluation/`
- **Configuration**: `config/default_config.yaml`
- **Tests**: `test_system.py`
- **Main Demo**: `main.py`

