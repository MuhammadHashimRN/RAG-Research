import os
import json
import hashlib
from typing import Optional, Any
from pathlib import Path

class CacheManager:
    """Simple file-based cache manager for LLM responses."""
    
    def __init__(self, cache_dir: str = ".cache/llm"):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        file_path = self._get_file_path(key)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
        
    def set(self, key: str, value: Any):
        """Set value in cache."""
        file_path = self._get_file_path(key)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write to cache: {e}")
            
    def _get_file_path(self, key: str) -> str:
        """Get file path for key."""
        # Create SHA256 hash of key
        hash_obj = hashlib.sha256(key.encode('utf-8'))
        filename = f"{hash_obj.hexdigest()}.json"
        return os.path.join(self.cache_dir, filename)

    def clear(self):
        """Clear all cache."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
