# Rate Limit Analysis

## Why Rate Limits Are Hit

### LLM Calls Per Query (Current Implementation)

For a **single query**, the system makes:

1. **Retrieval Decision**: 1-2 calls
   - Confidence estimation: 1 call
   - Decision prompt: 1 call

2. **Thought Generation**: 1 call per step
   - Up to 3 steps = **3 calls**

3. **Answer Generation**: 1 call

4. **Grounding Scoring**: 1-3 calls
   - Relevance: Encoder-based (0 calls) OR LLM fallback (1 call)
   - Consistency: Encoder-based (0 calls) OR LLM fallback (1 call)
   - Contradiction: Always LLM (1 call)

5. **Refinement**: 1 call per iteration
   - Up to 3 iterations = **3 calls**

6. **Oracle (if enabled)**: 2 calls
   - With retrieval: 1 call
   - Without retrieval: 1 call

**Total per query: 5-15 LLM calls**

### For Evaluation

- 10 queries × 10 calls = **100 LLM calls**
- Free tier limit: ~15 requests/minute
- **Result**: Rate limit hit quickly!

## Solutions

### Solution 1: Disable Expensive Features (Quick Fix)

Edit `config/default_config.yaml`:

```yaml
agent:
  max_iterations: 1  # Reduce from 3 to 1
  refinement_enabled: false  # Disable refinement
  # Thought generation can be disabled in code

grounding:
  # Ensure encoder is used (not LLM fallback)
  # Encoder-based scoring uses 0 LLM calls
```

### Solution 2: Use Encoder-Based Scoring (Recommended)

The grounding module should use the encoder (sentence-transformers) instead of LLM:
- Relevance: Encoder-based ✅ (0 LLM calls)
- Consistency: Encoder-based ✅ (0 LLM calls)
- Contradiction: LLM-based (1 call) - can be made optional

### Solution 3: Disable Thought Generation

Thought generation adds 1 call per step. Can be made optional.

### Solution 4: Add Rate Limiting

Add delays between calls to stay under limits.

### Solution 5: Disable Oracle in Evaluation

Oracle adds 2 calls per query. Disable for evaluation.

