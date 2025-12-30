from src.llm.llm_client import llm_client
from typing import List

class AnswerGenerator:
    def generate_answer(self, query: str, context: List[str] = None) -> str:
        if context:
            context_text = "\n".join(context)
            prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n"
                "Answer the question using the provided context."
            )
        else:
            prompt = f"Question: {query}\nAnswer the question."
            
        return llm_client.generate(prompt)
