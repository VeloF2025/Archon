# Dynamic Provider Switching for External Validator

## Overview

The External Validator supports **automatic dynamic switching** between different LLM providers. When you change the API key marked with the Shield icon in the UI, the validator automatically detects the provider and configures itself accordingly.

## How It Works

### 1. User Changes API Key in UI
- Navigate to Settings → API Keys
- Click the Shield icon next to desired API key
- Save changes

### 2. Automatic Provider Detection
The system detects providers using two methods:

#### A. Key Name Detection (Highest Priority)
- `DEEPSEEK_API_KEY` → DeepSeek
- `OPENAI_API_KEY` or `GPT_API_KEY` → OpenAI
- `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` → Anthropic
- `GROQ_API_KEY` → Groq
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` → Google
- `MISTRAL_API_KEY` → Mistral

#### B. API Key Format Detection (Fallback)
- `sk-proj-...` → OpenAI
- `sk-ant-...` → Anthropic
- `gsk_...` → Groq
- `AIza...` → Google Gemini
- Unknown formats → Default to DeepSeek

### 3. Automatic Model Selection
Each provider gets its optimal model automatically:

| Provider | Model | Base URL |
|----------|-------|----------|
| DeepSeek | deepseek-chat | https://api.deepseek.com |
| OpenAI | gpt-4o | Default OpenAI URL |
| Anthropic | claude-3-5-sonnet-20241022 | Default Anthropic URL |
| Groq | llama-3.3-70b-versatile | https://api.groq.com/openai/v1 |
| Google | gemini-1.5-pro | Google AI API |
| Mistral | mistral-large-latest | https://api.mistral.ai/v1 |

### 4. Dynamic Refresh Options

#### Option A: Automatic (Restart)
The validator loads the new API key on startup:
```bash
docker compose --profile validator restart archon-validator
```

#### Option B: Manual Refresh (No Restart)
Call the refresh endpoint to reload without restart:
```bash
curl -X POST http://localhost:8053/refresh-api-key
```

## Switching Providers - Step by Step

### Example: Switch from DeepSeek to OpenAI

1. **Add OpenAI Key in UI**:
   - Go to Settings → API Keys
   - Add key named `OPENAI_API_KEY`
   - Enter your OpenAI API key
   - Save

2. **Mark for Validator Use**:
   - Click Shield icon next to OpenAI key (turns purple)
   - Previous DeepSeek Shield automatically deselects
   - Save changes

3. **Refresh Validator**:
   ```bash
   curl -X POST http://localhost:8053/refresh-api-key
   ```

4. **Verify Switch**:
   ```bash
   curl http://localhost:8053/config
   ```
   Response shows:
   ```json
   {
     "llm": {
       "provider": "openai",
       "model": "gpt-4o",
       "has_api_key": true
     }
   }
   ```

## Provider Capabilities

### OpenAI-Compatible APIs
These providers work seamlessly with the existing validator:
- ✅ **DeepSeek** - Low cost, high performance
- ✅ **OpenAI** - GPT-4o for best quality
- ✅ **Groq** - Fastest inference with Llama 3.3
- ✅ **Mistral** - European alternative

### Requires Additional SDK (Future)
These providers need SDK installation:
- ⚠️ **Anthropic** - Falls back to Claude Code
- ⚠️ **Google** - Falls back to Claude Code

## Fallback Behavior

If a provider fails or no API key is configured:
1. **Primary Fallback**: Claude Code with anti-bias guardrails
2. **Guardrails Applied**:
   - Must find at least 1 issue
   - Confidence capped at 70% for self-work
   - 80% skepticism factor applied
   - Adversarial validation approach

## Testing Different Providers

### Test Validation Request
```bash
curl -X POST http://localhost:8053/validate \
  -H "Content-Type: application/json" \
  -d '{
    "output": "def test(): return True",
    "validation_type": "code",
    "context": {"description": "Test function"}
  }'
```

### Check Active Provider
```bash
curl http://localhost:8053/config | jq '.llm.provider'
```

### Monitor Logs
```bash
docker logs archon-validator --tail 20
```

## Performance Comparison

| Provider | Speed | Cost | Quality | Best For |
|----------|-------|------|---------|----------|
| DeepSeek | Fast | $0.14/M | Good | Budget-conscious validation |
| OpenAI | Medium | $5/M | Excellent | Critical code review |
| Groq | Fastest | $0.10/M | Good | High-volume validation |
| Mistral | Fast | $2/M | Very Good | European compliance |
| Claude Fallback | Fast | Free | Good* | No API key scenarios |

*With anti-bias guardrails to prevent self-approval

## Troubleshooting

### Provider Not Switching
1. Check API key is saved in database
2. Verify Shield icon is selected
3. Check validator logs for errors
4. Try manual refresh endpoint

### Provider Falls Back to Claude
- Check API key format is correct
- Verify provider is supported
- Check network connectivity
- Review validator logs

### Getting Provider Status
```bash
# Full configuration
curl http://localhost:8053/config

# Health check
curl http://localhost:8053/health

# Refresh from database
curl -X POST http://localhost:8053/refresh-api-key
```

## Security Notes

- API keys are encrypted in database
- Keys never logged or exposed in responses
- Provider detection happens server-side
- Automatic key rotation supported

## Future Enhancements

1. **Additional Providers**:
   - Cohere
   - AI21 Labs
   - Hugging Face Inference

2. **Advanced Features**:
   - Multi-provider consensus validation
   - Provider-specific prompt optimization
   - Cost tracking per provider
   - A/B testing between providers

3. **UI Improvements**:
   - Provider dropdown in Settings
   - Real-time provider status
   - Cost estimates per provider
   - Performance metrics dashboard

## Summary

The External Validator's dynamic provider switching ensures:
- ✅ **Zero downtime** when switching providers
- ✅ **Automatic configuration** based on API key
- ✅ **Optimal model selection** per provider
- ✅ **Fallback protection** with Claude Code
- ✅ **Cost optimization** with provider choice

Simply change the Shield icon in the UI, and the validator automatically adapts!