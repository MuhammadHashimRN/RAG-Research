# API Quota and Rate Limit Guide

## 🚨 Current Issue: Quota Exceeded

You're encountering quota limits with `gemini-2.5-pro`. This model may not be available on the free tier.

## ✅ Solutions

### Solution 1: Use Compatible Model (Recommended)

Update `config/default_config.yaml`:

```yaml
llm:
  model_name: "gemini-pro"  # Default, will try fallbacks automatically
  temperature: 0.0
```

**Common Model Names:**
- ✅ `gemini-pro` - Standard model (most compatible)
- ✅ `gemini-1.0-pro` - Version 1.0
- ✅ `gemini-1.5-pro` - Version 1.5
- ✅ `models/gemini-pro` - With models/ prefix
- ✅ `models/gemini-1.0-pro` - With models/ prefix

**Note**: The system now automatically tries fallback models if the requested one fails.

### Solution 2: Check Your Quota

1. Visit: https://ai.dev/usage?tab=rate-limit
2. Check your current usage
3. Verify which models are available on your plan

### Solution 3: Wait and Retry

The system now includes automatic retry logic with exponential backoff. It will:
- Automatically retry on rate limit errors
- Wait for the suggested delay from the API
- Use exponential backoff if no delay is specified

### Solution 4: Use Different Provider

You can modify the LLM client to use:
- OpenAI (if you have API key)
- Anthropic (if you have API key)
- Local models (Ollama, etc.)

## 🔧 Configuration Options

### Model Selection

Edit `config/default_config.yaml`:

```yaml
llm:
  # Common options (system will try fallbacks automatically):
  model_name: "gemini-pro"           # Default, most compatible
  # model_name: "gemini-1.0-pro"     # Version 1.0
  # model_name: "gemini-1.5-pro"     # Version 1.5 (if available)
  # model_name: "models/gemini-pro"  # With prefix
  
  temperature: 0.0
```

**Note**: If your specified model doesn't work, the system automatically tries:
1. Your specified model
2. `gemini-pro`
3. `gemini-1.0-pro`
4. `gemini-1.5-pro`
5. `models/gemini-pro`
6. `models/gemini-1.0-pro`

### Retry Configuration

The LLM client now includes:
- **Max retries**: 3 attempts
- **Automatic delay**: Extracts wait time from API errors
- **Exponential backoff**: 1s, 2s, 4s delays

## 📊 Rate Limits

### Free Tier Limits (Typical)
- **Requests per minute**: ~15
- **Requests per day**: ~1,500
- **Tokens per day**: Varies by model

### Paid Tier Limits
- Higher limits available
- Check your plan details

## 🛠️ Troubleshooting

### Error: "Quota exceeded for metric"
**Solution**: Switch to a free tier model or upgrade your plan

### Error: "429 RESOURCE_EXHAUSTED"
**Solution**: 
1. Wait for the suggested time
2. Use a different model
3. Reduce request frequency

### Error: "Model not found"
**Solution**: 
1. Check model name spelling
2. Verify model availability in your region
3. Try `gemini-1.5-flash` as fallback

## 💡 Best Practices

1. **Use Flash for Testing**: `gemini-1.5-flash` is fast and free
2. **Batch Requests**: Group multiple queries when possible
3. **Cache Results**: Store results to avoid redundant calls
4. **Monitor Usage**: Check quota regularly
5. **Handle Errors Gracefully**: System now retries automatically

## 🔄 Automatic Retry

The system now automatically:
- Detects quota/rate limit errors
- Extracts retry delay from API response
- Waits and retries up to 3 times
- Falls back gracefully on failure

## 📝 Example: Switching Models

```python
# In config/default_config.yaml
llm:
  model_name: "gemini-1.5-flash"  # Change this
  temperature: 0.0
```

Then restart your application.

## 🎯 Recommended Setup

For development/testing:
```yaml
llm:
  model_name: "gemini-pro"  # Most compatible
  temperature: 0.0
```

For production:
```yaml
llm:
  model_name: "gemini-1.5-pro"  # If available in your region/plan
  temperature: 0.0
```

**Check Available Models**:
Run `python src/utils/model_checker.py` to test which models work with your API key.

