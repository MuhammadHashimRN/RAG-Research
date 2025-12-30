from src.core.retrieval_decision import RetrievalDecisionModule, RetrievalDecision
from src.core.retrieval_engine import RetrievalEngine
from src.core.grounding import GroundingModule, GroundingScore
from src.core.answer_generation import AnswerGenerator
from src.core.refinement import RefinementModule
from src.config.config_manager import config
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class Action(str, Enum):
    """Available agent actions following ReAct pattern."""
    RETRIEVE = "retrieve"
    REFINE_QUERY = "refine_query"
    GENERATE = "generate"
    ABSTAIN = "abstain"
    OBSERVE = "observe"

@dataclass
class AgentStep:
    """Represents a single agent step in ReAct pattern."""
    step_number: int
    thought: str
    action: Action
    observation: str
    state: Dict = None

class AgentController:
    def __init__(self, documents: List[str] = None):
        self.decision_module = RetrievalDecisionModule()
        self.retrieval_engine = RetrievalEngine(documents)
        self.grounding_module = GroundingModule()
        self.generator = AnswerGenerator()
        self.refiner = RefinementModule()
        
        self.max_iterations = config.get("agent.max_iterations", 3)
        self.grounding_threshold = config.get("grounding.relevance_threshold", 0.7)
        self.refinement_enabled = config.get("agent.refinement_enabled", True)
        self.abstention_enabled = config.get("agent.abstention_enabled", True)
        self.enable_thought = config.get("agent.enable_thought", False)  # Disable by default to save calls
        
        # State tracking
        self.steps: List[AgentStep] = []
        self.retrieval_calls = 0

    def run(self, query: str) -> Dict:
        """
        Execute agent loop following ReAct pattern: Thought → Action → Observation
        
        Args:
            query: User query
        
        Returns:
            Dictionary with answer, context, logs, and metadata
        """
        # Reset state
        self.steps = []
        self.retrieval_calls = 0
        
        # Track tokens
        from src.llm.llm_client import llm_client
        start_tokens = getattr(llm_client, 'total_tokens', 0)
        
        current_query = query
        context: List[str] = []
        answer: Optional[str] = None
        grounding_score = 0.0
        iteration = 0
        
        # ReAct Loop
        while iteration < self.max_iterations:
            iteration += 1
            step_num = len(self.steps) + 1
            
            # THOUGHT: Agent reasons about current state
            thought = self._think(current_query, answer, context, grounding_score, step_num)
            
            # ACTION: Decide what to do
            action = self._decide_action(thought, current_query, answer, context, grounding_score)
            
            # OBSERVATION: Execute action and observe result
            observation, state_update = self._act(
                action, current_query, answer, context, grounding_score
            )
            
            # Record step
            step = AgentStep(
                step_number=step_num,
                thought=thought,
                action=action,
                observation=observation,
                state=state_update
            )
            self.steps.append(step)
            
            # Update state from action
            if "context" in state_update:
                context = state_update["context"]
            if "answer" in state_update:
                answer = state_update["answer"]
            if "grounding_score" in state_update:
                grounding_score = state_update["grounding_score"]
            if "abstained" in state_update:
                break
            
            # Check termination conditions
            if self._should_terminate(answer, grounding_score, action):
                break
        
        # Final answer generation if not done
        if not answer:
            answer = self._generate_final_answer(current_query, context)
            if answer and context:
                _, score_obj = self.grounding_module.validate_answer(
                    current_query, answer, [{"text": ctx} for ctx in context]
                )
                grounding_score = score_obj.overall_score
        
        # Abstention check
        if (grounding_score < self.grounding_threshold and 
            self.abstention_enabled and 
            answer and 
            not answer.startswith("I cannot")):
            answer = "I cannot answer this question with sufficient confidence based on the available information."
            self.steps.append(AgentStep(
                step_number=len(self.steps) + 1,
                thought="Final grounding score is insufficient.",
                action=Action.ABSTAIN,
                observation="Abstained from answering",
                state={"abstained": True}
            ))
        
        # Build logs from steps
        logs = []
        for step in self.steps:
            logs.append(f"Thought: {step.thought}")
            logs.append(f"Action: {step.action.value}")
            logs.append(f"Observation: {step.observation}")
            
        # Calculate tokens used
        end_tokens = getattr(llm_client, 'total_tokens', 0)
        tokens_used = end_tokens - start_tokens
        
        return {
            "query": query,
            "answer": answer or "[NO ANSWER]",
            "context": context,
            "logs": logs,
            "final_grounding_score": grounding_score,
            "retrieval_calls": self.retrieval_calls,
            "tokens_used": tokens_used,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action.value,
                    "observation": s.observation
                }
                for s in self.steps
            ]
        }
    
    def _think(
        self, 
        query: str, 
        answer: Optional[str], 
        context: List[str],
        grounding_score: float,
        step_num: int
    ) -> str:
        """Generate thought/reasoning about current state."""
        # Skip thought generation if disabled (saves LLM calls)
        if not self.enable_thought:
            return f"Step {step_num}: Answer={bool(answer)}, Context={len(context)}, Score={grounding_score:.2f}"
        
        from src.llm.llm_client import llm_client
        
        prompt = f"""You are an expert research agent. Your goal is to answer a complex query that may require multiple steps of reasoning.

Original Query: "{query}"

Current Status (Step {step_num}/{self.max_iterations}):
- Current Answer Candidates: {answer if answer else 'None'}
- Retrieved Passages: {len(context)}
- Grounding Score: {grounding_score:.2f} (Target: {self.grounding_threshold})

Context Summary:
{chr(10).join([f"- {c[:150]}..." for c in context[:3]])}

CRITICAL ANALYSIS:
1. Is this a multi-hop question? (e.g., "Who is the mother of the founder of X?" -> Need to find founder first).
2. Do I have the *specific* bridging entities mentioned in the query?
3. Is the current answer fully supported by the text, or am I hallucinating details?
4. If I need to search again, what *exact* specific term should I search for?

Produce a concise thought. If you need more info, explicitly state the new search term."""
        
        try:
            thought = llm_client.generate(prompt)
            return thought.strip()[:600]
        except Exception as e:
            return f"Error generating thought: {e}"
    
    def _decide_action(
        self,
        thought: str,
        query: str,
        answer: Optional[str],
        context: List[str],
        grounding_score: float
    ) -> Action:
        """Decide next action based on current state."""
        # If no answer yet, decide on retrieval
        if not answer:
            decision = self.decision_module.decide(query)
            if decision.should_retrieve:
                return Action.RETRIEVE
            else:
                return Action.GENERATE
        
        # If answer exists, check grounding
        if context and grounding_score < self.grounding_threshold:
            # If we already refined or retrieved and it's still bad, maybe refine query?
            # Simple heuristic: if we haven't refined the query yet (check history), try that.
            # Otherwise, if refinement is enabled, try generating again (refine answer).
            
            # Check if we recently refined the query to avoid loops
            recent_actions = [s.action for s in self.steps[-2:]]
            if Action.REFINE_QUERY not in recent_actions:
                 return Action.REFINE_QUERY
            
            if self.refinement_enabled:
                return Action.GENERATE  # Will trigger refinement
            else:
                return Action.RETRIEVE
        
        # If grounding is sufficient, we're done
        if grounding_score >= self.grounding_threshold:
            return Action.OBSERVE  # Just observe, no action needed
        
        # Default: generate answer
        return Action.GENERATE
    
    def _act(
        self,
        action: Action,
        query: str,
        answer: Optional[str],
        context: List[str],
        grounding_score: float
    ) -> tuple[str, Dict]:
        """Execute action and return observation."""
        state_update = {}
        
        if action == Action.RETRIEVE:
            # Retrieve passages
            retrieved = self.retrieval_engine.retrieve(query)
            self.retrieval_calls += 1
            context = retrieved
            
            observation = f"Retrieved {len(retrieved)} passages"
            state_update["context"] = context
            
        elif action == Action.REFINE_QUERY:
            # Rewrite query and retrieve
            new_query = self._rewrite_query(query, answer)
            retrieved = self.retrieval_engine.retrieve(new_query)
            self.retrieval_calls += 1
            context = retrieved
            
            observation = f"Refined query to '{new_query}' and retrieved {len(retrieved)} passages"
            state_update["context"] = context
            # We don't update 'query' state variable to preserve original user query intent,
            # but we use the new context for next steps.
        
        elif action == Action.GENERATE:
            # Generate or refine answer
            if answer and context and self.refinement_enabled:
                # Refine existing answer
                feedback = "The answer is not sufficiently supported by the provided context. Ensure all claims are directly backed by the retrieved text."
                new_answer = self.refiner.refine_answer(query, answer, feedback, context)
                
                # Check grounding of refined answer
                passages = [{"text": ctx} for ctx in context]
                is_valid, score_obj = self.grounding_module.validate_answer(
                    query, new_answer, passages
                )
                
                # Only update if improved
                if score_obj.overall_score > grounding_score:
                    answer = new_answer
                    grounding_score = score_obj.overall_score
                    observation = f"Refined answer. New grounding score: {grounding_score:.2f}"
                else:
                    observation = f"Refinement did not improve (score: {score_obj.overall_score:.2f})"
            else:
                # Generate new answer
                answer = self.generator.generate_answer(query, context)
                
                # Check grounding
                if context:
                    passages = [{"text": ctx} for ctx in context]
                    is_valid, score_obj = self.grounding_module.validate_answer(
                        query, answer, passages
                    )
                    grounding_score = score_obj.overall_score
                    observation = f"Generated answer. Grounding score: {grounding_score:.2f}"
                else:
                    observation = "Generated answer without context"
            
            state_update["answer"] = answer
            state_update["grounding_score"] = grounding_score
        
        elif action == Action.ABSTAIN:
            observation = "Insufficient grounding. Abstaining from answering."
            state_update["abstained"] = True
        
        else:  # OBSERVE
            observation = f"Observing current state: answer ready, grounding score {grounding_score:.2f}"
        
        return observation, state_update
    
    def _rewrite_query(self, query: str, current_answer: Optional[str]) -> str:
        """Rewrite query to be more search-friendly based on current failure."""
        from src.llm.llm_client import llm_client
        prompt = f"""The previous retrieval for the query '{query}' did not yield sufficient results to ground the answer: "{current_answer or 'N/A'}".
        
        Please generate a single, improved search query that is more likely to retrieve relevant facts.
        Focus on key entities and removing conversational filler.
        
        New Query:"""
        try:
            new_query = llm_client.generate(prompt).strip().strip('"')
            return new_query
        except Exception:
            return query
    
    def _should_terminate(
        self, 
        answer: Optional[str], 
        grounding_score: float,
        action: Action
    ) -> bool:
        """Check if agent should terminate."""
        # Terminate if we have a good answer
        if answer and grounding_score >= self.grounding_threshold:
            return True
        
        # Terminate if we abstained
        if action == Action.ABSTAIN:
            return True
        
        # Terminate if we're just observing (nothing more to do)
        if action == Action.OBSERVE:
            return True
        
        return False
    
    def _generate_final_answer(self, query: str, context: List[str]) -> str:
        """Generate final answer as fallback."""
        return self.generator.generate_answer(query, context)
