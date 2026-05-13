# Optimizing for Rate Limits

## Problem

The system makes **5-15 LLM calls per query**, which quickly hits free tier limits (15 requests/minute).

## Quick Fixes (Apply Now)

### 1. Disable Thought Generation

Edit `config/default_config.yaml`:
```yaml
agent:
  enable_thought: false  # Saves 3 calls per query
```

### 2. Reduce Max Iterations

```yaml
agent:
  max_iterations: 1  # Instead of 3, saves 2-4 calls
```

### 3. Disable Refinement

```yaml
agent:
  refinement_enabled: false  # Saves 3 calls per query
```

### 4. Use Encoder-Based Grounding

Ensure encoder is loaded (it is by default). This saves 2-3 calls per query.

### 5. Disable Contradiction Detection

```yaml
grounding:
  use_llm_for_contradiction: false  # Saves 1 call per query
```

### 6. Disable Oracle in Evaluation

When running evaluation, disable oracle:
```python
evaluator = Evaluator(agent, baseline_runner, enable_oracle=False)
```

## Optimized Configuration

```yaml
# config/default_config.yaml
retrieval:
  use_selective_retrieval: true
  similarity_top_k: 3
  dense_model_id: "sentence-transformers/all-MiniLM-L6-v2"

agent:
  max_iterations: 1  # Reduced
  refinement_enabled: false  # Disabled
  abstention_enabled: true
  enable_thought: false  # Disabled

grounding:
  relevance_threshold: 0.7
  consistency_threshold: 0.8
  use_llm_for_contradiction: false  # Disabled

llm:
  model_name: "gemini-pro"
  temperature: 0.0
  enable_rate_limiting: true
  rate_limit_calls: 10  # Adjust for your tier
  rate_limit_window: 60.0
```

## Call Reduction Summary

| Feature | Calls Saved | Impact |
|---------|-------------|--------|
| Disable thought | 3 | High |
| Reduce iterations to 1 | 2-4 | High |
| Disable refinement | 3 | High |
| Use encoder grounding | 2-3 | High |
| Disable contradiction | 1 | Medium |
| Disable oracle | 2 | Medium |

**Total savings: 13-15 calls per query → 2-3 calls per query**

## Rate Limiting

The system now includes automatic rate limiting:
- Default: 10 calls per 60 seconds
- Automatically waits between calls
- Configurable in config file

## Testing Configuration

For testing, use minimal calls:

```yaml
agent:
  max_iterations: 1
  refinement_enabled: false
  enable_thought: false

grounding:
  use_llm_for_contradiction: false
```

This reduces to **~3 calls per query**:
1. Retrieval decision (optional, can be simplified)
2. Answer generation
3. Grounding (encoder-based, 0 LLM calls)

## Monitoring

Check your API usage:
- https://ai.dev/usage?tab=rate-limit
- Monitor call frequency
- Adjust rate limits accordingly

