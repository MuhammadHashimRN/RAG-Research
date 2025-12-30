import requests
import json

def check_ollama():
    print("Checking Ollama API...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running!")
            if not models:
                print("⚠️  No models found. You need to run: ollama pull llama3")
            else:
                print("Available models:")
                for m in models:
                    print(f" - {m['name']}")
            return True
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to Ollama: {e}")
        print("Make sure Ollama is running in your system tray.")
    return False

if __name__ == "__main__":
    check_ollama()
