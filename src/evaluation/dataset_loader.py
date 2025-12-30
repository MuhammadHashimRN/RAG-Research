import pandas as pd
from typing import List, Dict, Optional
import json

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets package not available. Install with: pip install datasets")

class DatasetLoader:
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize dataset loader."""
        self.cache_dir = cache_dir
    
    def load_hotpotqa(
        self, 
        split: str = "dev",
        max_samples: Optional[int] = None,
        path: Optional[str] = None
    ) -> List[Dict]:
        """
        Load HotpotQA dataset.
        
        Args:
            split: Dataset split (train, dev, test)
            max_samples: Maximum number of samples to load
            path: Optional path to local file
        
        Returns:
            List of examples with 'question', 'answer', 'context', etc.
        """
        if path:
            # Load from local file
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[:max_samples] if max_samples else data
        
        if not DATASETS_AVAILABLE:
            print("Warning: datasets package not available. Returning empty list.")
            return []
        
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split=split, cache_dir=self.cache_dir)
            
            examples = []
            for item in dataset:
                example = {
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "context": item.get("context", {}),
                    "supporting_facts": item.get("supporting_facts", []),
                    "type": item.get("type", ""),
                    "level": item.get("level", "")
                }
                examples.append(example)
                
                if max_samples and len(examples) >= max_samples:
                    break
            
            return examples
        except Exception as e:
            print(f"Error loading HotpotQA: {e}")
            return []
    
    def load_fever(
        self, 
        split: str = "dev",
        max_samples: Optional[int] = None,
        path: Optional[str] = None
    ) -> List[Dict]:
        """
        Load FEVER dataset.
        
        Args:
            split: Dataset split (train, dev, test)
            max_samples: Maximum number of samples to load
            path: Optional path to local file
        
        Returns:
            List of examples with 'claim', 'label', 'evidence', etc.
        """
        if path:
            # Load from local file
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[:max_samples] if max_samples else data
        
        if not DATASETS_AVAILABLE:
            print("Warning: datasets package not available. Returning empty list.")
            return []
        
        try:
            dataset = load_dataset("fever", split=split, cache_dir=self.cache_dir)
            
            examples = []
            for item in dataset:
                example = {
                    "id": item.get("id", 0),
                    "claim": item.get("claim", ""),
                    "label": item.get("label", ""),  # SUPPORTS, REFUTES, NOT_ENOUGH_INFO
                    "evidence": item.get("evidence", []),
                    "annotated_evidence": item.get("annotated_evidence", [])
                }
                examples.append(example)
                
                if max_samples and len(examples) >= max_samples:
                    break
            
            return examples
        except Exception as e:
            print(f"Error loading FEVER: {e}")
            return []
    
    def prepare_passages_from_hotpotqa(self, examples: List[Dict]) -> List[str]:
        """
        Extract passages from HotpotQA examples for indexing.
        
        Returns:
            List of passage strings
        """
        passages = []
        
        for example in examples:
            context = example.get("context", {})
            
            # HuggingFace datasets structure: context is dict with 'title' (list) and 'sentences' (list of lists)
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            
            if len(titles) == len(sentences_list):
                for title, sentences in zip(titles, sentences_list):
                    # sentences is a list of strings
                    text = " ".join(sentences)
                    if text:
                        passages.append(f"{title}: {text}")
            else:
                # Fallback or older format handling if needed, though standard HF is consistent
                pass
        
        return passages
    
    def prepare_passages_from_fever(self, examples: List[Dict]) -> List[str]:
        """
        Extract passages from FEVER examples for indexing.
        
        Returns:
            List of passage strings
        """
        passages = []
        
        for example in examples:
            evidence = example.get("evidence", [])
            
            for evid_group in evidence:
                for evid_item in evid_group:
                    if isinstance(evid_item, dict):
                        sentence = evid_item.get("sentence", "")
                        if sentence:
                            passages.append(sentence)
        
        return passages
