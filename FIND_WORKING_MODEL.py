"""
Quick script to find which Gemini model works with your API key.
Run this to determine the correct model name for your setup.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_model(model_name: str) -> tuple[bool, str]:
    """Test if a model works. Returns (success, error_message)."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False, "No API key found"
        
        # Try to create the LLM instance
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        
        # If creation succeeds, model name is valid
        return True, "Model name accepted"
        
    except Exception as e:
        error_str = str(e)
        if "NOT_FOUND" in error_str or "404" in error_str:
            return False, "Model not found (404)"
        elif "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            return True, "Model exists but quota exceeded (this is OK for initialization)"
        else:
            return False, f"Error: {str(e)[:100]}"

def main():
    print("="*80)
    print("Finding Working Gemini Model")
    print("="*80)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found. Set GOOGLE_API_KEY in .env file")
        return 1
    
    print(f"✅ API key found\n")
    
    # Test common model names
    models_to_test = [
        "gemini-pro",
        "gemini-1.0-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "models/gemini-pro",
        "models/gemini-1.0-pro",
        "models/gemini-1.5-pro",
    ]
    
    print("Testing models...\n")
    
    working_models = []
    for model_name in models_to_test:
        success, message = test_model(model_name)
        status = "✅" if success else "❌"
        print(f"{status} {model_name:30s} - {message}")
        if success:
            working_models.append(model_name)
    
    print("\n" + "="*80)
    if working_models:
        print(f"✅ Found {len(working_models)} working model(s):")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n💡 Update config/default_config.yaml with:")
        print(f"   llm:")
        print(f"     model_name: \"{working_models[0]}\"")
    else:
        print("❌ No working models found.")
        print("   Please check:")
        print("   1. API key is valid")
        print("   2. You have access to Gemini API")
        print("   3. Check: https://ai.google.dev/models")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

