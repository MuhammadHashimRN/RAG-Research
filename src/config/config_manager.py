import yaml
import os
from typing import Any, Dict

class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self, config_path: str = "config/default_config.yaml"):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            # Fallback default if file missing
            self._config = {
                "retrieval": {"use_selective_retrieval": True, "similarity_top_k": 3},
                "agent": {"max_iterations": 3},
                "llm": {"model_name": "gemini-pro", "temperature": 0.0}  # Default model
            }

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

config = ConfigManager()
