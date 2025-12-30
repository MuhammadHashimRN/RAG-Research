# Troubleshooting Guide

## Common Issues and Solutions

### 1. API Quota/Rate Limit Errors

**Error**: `429 RESOURCE_EXHAUSTED` or `Quota exceeded`

**Solutions**:
1. **Switch to free tier model**:
   ```yaml
   # config/default_config.yaml
   llm:
     model_name: "gemini-1.5-flash"  # Free tier compatible
   ```

2. **Wait and retry**: System now automatically retries with delays

3. **Check quota**: Visit https://ai.dev/usage?tab=rate-limit

4. **Reduce request frequency**: Add delays between requests

### 2. Model Not Found

**Error**: `Model not found` or `Invalid model name`

**Solution**:
- Use one of these models:
  - `gemini-1.5-flash` (recommended for free tier)
  - `gemini-1.5-pro`
  - `gemini-1.0-pro`

### 3. Import Errors

**Error**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
pip install -r requirements.txt
```

### 4. Configuration Not Loading

**Error**: Config file not found or values not working

**Solution**:
- Check `config/default_config.yaml` exists
- Verify YAML syntax is correct
- Restart application after config changes

### 5. Retrieval Not Working

**Error**: Empty results or no documents retrieved

**Solutions**:
1. **Check documents are indexed**:
   ```python
   engine = RetrievalEngine(documents)
   print(engine.get_statistics())
   ```

2. **Verify FAISS index**:
   - Ensure documents list is not empty
   - Check encoder model loaded successfully

3. **Test retrieval directly**:
   ```python
   results = engine.retrieve("test query", top_k=3)
   print(f"Retrieved: {len(results)} documents")
   ```

### 6. Grounding Scores Always Low

**Issue**: Grounding scores consistently below threshold

**Solutions**:
1. **Adjust thresholds** in config:
   ```yaml
   grounding:
     relevance_threshold: 0.6  # Lower if too strict
     consistency_threshold: 0.7
   ```

2. **Check evidence quality**: Ensure retrieved documents are relevant

3. **Review grounding calculation**: Check if encoder loaded correctly

### 7. Agent Stuck in Loop

**Issue**: Agent runs too many iterations

**Solutions**:
1. **Reduce max iterations**:
   ```yaml
   agent:
     max_iterations: 2  # Reduce from 3
   ```

2. **Increase grounding threshold**: Makes termination easier

3. **Check termination logic**: Verify `_should_terminate()` works

### 8. Memory Issues

**Error**: Out of memory or slow performance

**Solutions**:
1. **Use smaller encoder model**:
   ```yaml
   retrieval:
     dense_model_id: "sentence-transformers/all-MiniLM-L6-v2"  # Smaller model
   ```

2. **Reduce batch size**: Modify in retrieval_engine.py

3. **Process in smaller chunks**: Evaluate in batches

### 9. Test Failures

**Issue**: Tests fail with various errors

**Solutions**:
1. **Check API key**: Ensure `GOOGLE_API_KEY` is set in `.env`

2. **Run tests individually**:
   ```python
   # In test_system.py, comment out other tests
   test_imports()  # Start with this
   ```

3. **Check dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### 10. Evaluation Takes Too Long

**Issue**: Evaluation runs very slowly

**Solutions**:
1. **Reduce dataset size**: Use `max_samples=10` for testing

2. **Disable baselines**: Set `run_baselines=False`

3. **Use faster model**: `gemini-1.5-flash` is faster than pro models

4. **Cache results**: Save intermediate results

## Quick Fixes

### Reset Everything
```bash
# Clear cache
rm -rf .cache/

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check config
cat config/default_config.yaml
```

### Verify Setup
```bash
# Run verification
python verify_installation.py

# Run tests
python test_system.py
```

### Check Environment
```bash
# Verify API key
echo $GOOGLE_API_KEY

# Or check .env file
cat .env
```

## Getting Help

1. **Check logs**: Look for error messages in console output
2. **Review documentation**: See `README.md` and other guides
3. **Test components**: Run individual component tests
4. **Check API status**: Verify API is operational

## Emergency Fallbacks

If API is completely unavailable:

1. **Use mock responses**: Modify LLM client to return dummy data
2. **Test without LLM**: Focus on retrieval and grounding components
3. **Use local models**: Switch to Ollama or similar if available

