from src.core.agent_controller import AgentController
import os

def main():
    # Dummy knowledge base
    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "The height of the Eiffel Tower is 324 meters.",
        "The Great Wall of China is visible from space under certain conditions.",
        "Python is a high-level programming language known for its readability.",
        "Photosynthesis is the process used by plants to convert light energy into chemical energy."
    ]
    
    # Initialize Agent
    agent = AgentController(documents)
    
    # Test Queries
    queries = [
        "Where is the Eiffel Tower?",
        "How tall is the Eiffel Tower?",
        "What is Python?",
        "What is the capital of Mars?" # Should abstain or fail retrieval
    ]
    
    print("--- Agentic RAG System Demo ---\n")
    
    for q in queries:
        print(f"Query: {q}")
        result = agent.run(q)
        print(f"Answer: {result['answer']}")
        print(f"Logs: {result['logs']}")
        print("-" * 30)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure Google API key is set for this to work
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        print("WARNING: No valid API key (GOOGLE_API_KEY or GROQ_API_KEY) found. LLM calls will fail.")
    
    main()
