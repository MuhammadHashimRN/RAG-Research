# Agentic RAG Research Framework

This repository contains the implementation for the "Selective Retrieval and Grounded Self-Refinement" research paper.

## 🚀 Optimized for NVIDIA RTX 3050 (6GB VRAM)

This guide provides step-by-step instructions to set up and run the full 1000-sample ablation study on your new hardware.

---

### 1. Prerequisites

Before copying the code, ensure your new laptop has the following installed:

1.  **Python 3.10+**: [Download Here](https://www.python.org/downloads/)
2.  **Git**: [Download Here](https://git-scm.com/download/win)
3.  **Ollama** (for local LLM inference): [Download Here](https://ollama.com/)
4.  **NVIDIA Drivers & CUDA**: Ensure your GPU drivers are up to date via GeForce Experience.

---

### 2. Initial Setup

1.  **Clone/Copy the Repository** to your new laptop.
2.  **Open a Terminal** (PowerShell or Command Prompt) in the project folder.
3.  **Create a Virtual Environment**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

---

### 3. GPU Model Setup (Ollama)

Your RTX 3050 (6GB) is powerful enough to run **Llama 3 (8B)**, which is significantly smarter than the TinyLlama model used previously.

1.  **Pull the Model**:
    Run this command in your terminal:
    ```powershell
    ollama pull llama3
    ```
    *Note: This downloads approx 4.7GB. It fits perfectly into your 6GB VRAM.*

2.  **Verify GPU Acceleration**:
    Run a quick test to ensure Ollama is using your Nvidia GPU and not the CPU:
    ```powershell
    ollama run llama3 "Hello, are you running on GPU?"
    ```
    *Open Task Manager > Performance > GPU 0 to confirm usage spikes.*

---

### 4. Unlocking Performance (CRITICAL STEP)

The code was previously optimized for a low-RAM system. You must revert these settings to utilize your GPU capabilities.

#### A. Edit `config/default_config.yaml`
Open this file and update the following values:

```yaml
llm:
  provider: "ollama"
  model_name: "llama3"       # CHANGED from tinyllama
  temperature: 0.0

retrieval:
  use_selective_retrieval: true
  similarity_top_k: 3        # INCREASED from 1 to 3

agent:
  max_iterations: 2          # INCREASED from 1 to 2
  enable_thought: true       # ENABLED for better reasoning
```

#### B. Edit `src/llm/llm_client.py`
We need to increase the "Context Window" so the model can read more documents.

1.  Open `src/llm/llm_client.py`.
2.  Search for `num_ctx=1024`.
3.  Change it to `num_ctx=4096` (or remove the parameter entirely to use default).

**Look for this block (around line 90-100):**
```python
if self.provider == "ollama":
    test_llm = ChatOllama(
        model=model_name,
        temperature=self.temperature,
        num_ctx=4096  # <--- UPDATE THIS VALUE
    )
```

---

### 5. Running the Full Experiment

Now you are ready to generate the final data for your research paper.

1.  **Edit the Experiment Script**:
    Open `run_research_paper_experiments.py`.
    
    Change the sample size to 1000:
    ```python
    # Set to 1000 for the final paper results
    SAMPLE_SIZE = 1000 
    
    # Update index path to keep it separate
    index_path = "data/indices/hotpotqa_final_gpu"
    ```

2.  **Run the Script**:
    ```powershell
    python run_research_paper_experiments.py
    ```

### 6. Expected Performance

*   **Speed:** You should see processing speeds of roughly 2-5 seconds per query (vs 20+ seconds on CPU).
*   **Accuracy:** F1 scores and Evidence Precision should increase significantly with Llama 3.
*   **Memory:** If you get "Exit Status 2" errors again, it means 4096 context is slightly too large for 6GB VRAM alongside the Windows display.
    *   *Fix:* Lower `num_ctx` in `src/llm/llm_client.py` to `2048`.

---

### 7. Troubleshooting

*   **"Ollama runner process has terminated"**: This is an Out-Of-Memory error.
    *   Close other apps (Chrome, etc.) to free up VRAM.
    *   Reduce `num_ctx` to 2048.
*   **Slow Speed**: Ensure your laptop is plugged in and in "Performance Mode" (click the battery icon in Windows taskbar).