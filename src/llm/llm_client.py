from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.config.config_manager import config
from src.utils.cache_manager import CacheManager
import os
import time
import json
from dotenv import load_dotenv

# Import rate limiter
try:
    from src.utils.rate_limiter import get_rate_limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

load_dotenv()

class LLMClient:
    def __init__(self):
        # Determine preferred provider and model
        requested_model = config.get("llm.model_name", "llama-3.3-70b-versatile")
        self.temperature = config.get("llm.temperature", 0.0)
        
        # Cache configuration
        self.use_cache = config.get("llm.use_cache", True)
        self.cache = CacheManager() if self.use_cache else None
        
        # Check API keys
        groq_api_key = os.getenv("GROQ_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        self.llm = None
        self.model_name = None
        self.provider = None

        # Determine which provider to use
        use_groq = False
        use_ollama = False
        
        # Explicit provider config or implicit from model name
        provider_config = config.get("llm.provider", "").lower()
        if provider_config == "groq":
            use_groq = True
        elif provider_config == "ollama":
            use_ollama = True
        elif provider_config == "gemini" or provider_config == "google":
            use_groq = False
        else:
            # Infer from model name or available keys
            is_groq_model = any(k in requested_model.lower() for k in ["llama-3", "mixtral", "gemma", "groq"])
            is_ollama_model = any(k in requested_model.lower() for k in ["llama3", "llama2", "mistral", "phi"])
            
            if is_groq_model and groq_api_key:
                use_groq = True
            elif is_ollama_model:
                use_ollama = True
            elif groq_api_key and not google_api_key:
                use_groq = True
            elif not google_api_key and not groq_api_key:
                # Default to Ollama if no keys
                use_ollama = True

        if use_ollama:
            self.provider = "ollama"
            # Fallback models for Ollama
            fallback_models = [
                requested_model,
                "llama3",
                "mistral",
                "llama2",
                "phi3"
            ]
        elif use_groq:
            if not groq_api_key:
                 # Fallback to Ollama if Groq key missing
                 print("⚠️ GROQ_API_KEY missing, falling back to Ollama")
                 self.provider = "ollama"
                 fallback_models = ["llama3", "mistral"]
            else:
                self.provider = "groq"
                # Fallback models for Groq
                fallback_models = [
                    requested_model,
                    "llama-3.3-70b-versatile",
                    "mixtral-8x7b-32768",
                    "gemma2-9b-it",
                    "llama3-70b-8192"
                ]
        else:
            # Default to Gemini
            if not google_api_key:
                if use_ollama: # If we decided on ollama earlier
                     pass 
                elif groq_api_key:
                    print("⚠️ GOOGLE_API_KEY not found, but GROQ_API_KEY is present. Switching to Groq.")
                    use_groq = True
                    self.provider = "groq"
                    fallback_models = ["llama-3.3-70b-versatile"]
                else:
                    # No keys available -> Ollama
                    print("⚠️ No API keys found. Defaulting to Ollama (Local).")
                    self.provider = "ollama"
                    fallback_models = ["llama3", "mistral"]
            
            if self.provider == "gemini" or (not self.provider and google_api_key):
                self.provider = "gemini"
                # Fallback models for Gemini
                fallback_models = [
                    requested_model,
                    "gemini-pro",
                    "gemini-1.5-pro",
                    "models/gemini-pro"
                ]

        # Remove duplicates
        seen = set()
        fallback_models = [m for m in fallback_models if not (m in seen or seen.add(m))]
        
        last_error = None
        for model_name in fallback_models:
            try:
                if self.provider == "ollama":
                    test_llm = ChatOllama(
                        model=model_name,
                        temperature=self.temperature,
                        num_ctx=1024
                    )
                elif self.provider == "groq":
                    test_llm = ChatGroq(
                        model_name=model_name,
                        temperature=self.temperature,
                        groq_api_key=groq_api_key
                    )
                else:
                    test_llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=self.temperature,
                        google_api_key=google_api_key,
                        convert_system_message_to_human=True 
                    )
                
                # Check if we can instantiate (lightweight check)
                self.llm = test_llm
                self.model_name = model_name
                print(f"✅ Initialized LLM with provider: {self.provider}, model: {model_name}")
                break
                    
            except Exception as e:
                last_error = e
                print(f"⚠️ Failed to initialize {model_name}: {e}")
                continue
        
        if not self.llm or not self.model_name:
            # Last ditch effort for Ollama
            try:
                print("⚠️  All configured models failed. Trying default 'llama3' on Ollama...")
                self.llm = ChatOllama(model="llama3", num_ctx=1024)
                self.model_name = "llama3"
                self.provider = "ollama"
                print("✅ Initialized fallback LLM: Ollama/llama3")
            except:
                raise ValueError("Could not initialize any LLM. Please install Ollama or set API keys.")
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        
        # Rate limiting configuration
        self.enable_rate_limiting = config.get("llm.enable_rate_limiting", True)
        self.rate_limit_calls = config.get("llm.rate_limit_calls", 10)
        self.rate_limit_window = config.get("llm.rate_limit_window", 60.0)
        
        # Token usage tracking
        self.total_tokens = 0
        
        if not self.llm:
            raise ValueError("LLM not initialized. Check API key and model availability.")

    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        retry_on_quota: bool = True
    ) -> str:
        """
        Generate text from prompt with retry logic for rate limits.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            retry_on_quota: Whether to retry on quota errors
        
        Returns:
            Generated text
        """
        # Check cache first
        if self.use_cache and self.cache:
            cache_key = f"{self.model_name}:{self.temperature}:{system_prompt}:{prompt}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                # Update tokens from cache if stored, else estimate
                if isinstance(cached_response, dict):
                    self.total_tokens += cached_response.get('tokens', 0)
                    return cached_response.get('content', '')
                return cached_response

        # Apply rate limiting if enabled
        if self.enable_rate_limiting and RATE_LIMITER_AVAILABLE:
            rate_limiter = get_rate_limiter(self.rate_limit_calls, self.rate_limit_window)
            rate_limiter.wait_if_needed()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                content = response.content.strip()
                tokens_used = 0
                
                # Track token usage if available
                if hasattr(response, 'response_metadata') and response.response_metadata:
                    token_usage = response.response_metadata.get('token_usage', {})
                    if not token_usage:
                        # Try finding it in 'usage_metadata' or other keys depending on provider
                        token_usage = response.response_metadata.get('usage_metadata', {})
                    
                    if token_usage:
                        total = token_usage.get('total_tokens', 0)
                        self.total_tokens += total
                        tokens_used = total
                    else:
                        # Fallback: estimate tokens (rough char count / 4)
                        est = len(prompt) // 4 + len(content) // 4
                        self.total_tokens += est
                        tokens_used = est

                # Save to cache
                if self.use_cache and self.cache:
                    cache_key = f"{self.model_name}:{self.temperature}:{system_prompt}:{prompt}"
                    self.cache.set(cache_key, {
                        'content': content,
                        'tokens': tokens_used
                    })

                return content
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a quota/rate limit error
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    if not retry_on_quota or attempt == self.max_retries - 1:
                        print(f"⚠️  Quota/Rate limit error. Consider:")
                        print(f"   1. Using a different model (e.g., gemini-pro)")
                        print(f"   2. Waiting before retrying")
                        print(f"   3. Checking your API quota: https://ai.dev/usage")
                        return ""
                    
                    # Extract retry delay from error if available
                    delay = self._extract_retry_delay(error_str)
                    if delay:
                        print(f"⏳ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(delay)
                    else:
                        # Exponential backoff
                        delay = self.base_delay * (2 ** attempt)
                        print(f"⏳ Rate limit hit. Waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                        time.sleep(delay)
                elif "NOT_FOUND" in error_str or "404" in error_str:
                    # Model not found - don't retry, this won't help
                    print(f"❌ Model '{self.model_name}' not found (404).")
                    print(f"   The system should have tried fallback models during initialization.")
                    print(f"   Please check your config or run: python src/utils/model_checker.py")
                    return ""
                else:
                    # Other errors - don't retry
                    print(f"Error calling LLM: {e}")
                    return ""
        
        # All retries exhausted
        print(f"❌ Failed after {self.max_retries} attempts: {last_error}")
        return ""
    
    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        """Extract retry delay from error message."""
        import re
        # Look for "Please retry in X.XXs" pattern
        match = re.search(r'retry in ([\d.]+)s', error_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def estimate_confidence(self, query: str, context: Optional[str] = None) -> float:
        """
        Estimate model's confidence in answering without retrieval.
        Returns confidence score between 0.0 and 1.0.
        """
        prompt = f"""Estimate your confidence (0.0 to 1.0) in answering this query accurately without external knowledge retrieval.

Query: {query}

{f'Context: {context}' if context else ''}

Consider:
- Do you have sufficient knowledge in your training data?
- Is the query about factual information that may be outdated?
- Would external information significantly improve your answer?

Respond with only a number between 0.0 and 1.0."""
        
        try:
            # Temperature is already set in LLM initialization, no need to pass it
            response = self.generate(prompt)
            if not response:
                return 0.5  # Default on error
            
            confidence = float(response.strip())
            return max(0.0, min(1.0, confidence))
        except (ValueError, Exception) as e:
            print(f"Confidence estimation failed: {e}")
            return 0.5  # Default to medium confidence

# Create singleton instance for easy import
llm_client = LLMClient()
