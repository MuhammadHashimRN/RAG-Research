import sys
import importlib

required_packages = [
    "langchain",
    "langchain_google_genai",
    "faiss",  # usually 'faiss-cpu' or 'faiss-gpu' but module is 'faiss'
    "sentence_transformers",
    "torch",
    "yaml",
    "numpy",
    "pandas"
]

def check_imports():
    print("Verifying installation...")
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} NOT found")
            missing.append(package)
            
    if missing:
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\nAll dependencies appear to be installed.")
        print("You can run the demo using: python main.py")

if __name__ == "__main__":
    check_imports()
