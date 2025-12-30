"""
Utility to check available Gemini models.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def list_available_models():
    """
    List available Gemini models.
    
    Returns:
        List of available model names
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️  No API key found. Set GOOGLE_API_KEY in .env file")
            return []
        
        # Try to get models list (if API supports it)
        # For now, return common model names
        common_models = [
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.5-pro",
            "models/gemini-pro",
            "models/gemini-1.0-pro",
            "models/gemini-1.5-pro"
        ]
        
        print("Common Gemini model names to try:")
        for model in common_models:
            print(f"  - {model}")
        
        print("\nTo check which models work, try initializing ChatGoogleGenerativeAI with each.")
        return common_models
        
    except ImportError:
        print("⚠️  langchain_google_genai not installed")
        return []

def test_model(model_name: str) -> bool:
    """
    Test if a model name works.
    
    Args:
        model_name: Model name to test
    
    Returns:
        True if model works, False otherwise
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False
        
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # Try a simple call
        response = llm.invoke("Say 'test'")
        return True
    except Exception as e:
        print(f"❌ Model '{model_name}' failed: {e}")
        return False

if __name__ == "__main__":
    print("Checking available Gemini models...\n")
    models = list_available_models()
    
    print("\nTesting models:")
    for model in models:
        result = test_model(model)
        if result:
            print(f"✅ {model} - WORKS")
        else:
            print(f"❌ {model} - FAILED")

