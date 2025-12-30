from src.llm.llm_client import llm_client
from typing import List

class RefinementModule:
    def refine_answer(self, query: str, current_answer: str, feedback: str, context: List[str]) -> str:
        context_text = "\n".join(context) if context else ""
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Current Answer: {current_answer}\n"
            f"Critique/Issue: {feedback}\n\n"
            "Refine the answer to address the critique while remaining grounded in the context."
        )
        return llm_client.generate(prompt)
