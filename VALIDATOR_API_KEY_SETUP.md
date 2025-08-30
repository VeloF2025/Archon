# External Validator API Key Setup Guide

## Where to Configure API Keys

The External Validator supports multiple configuration methods:

### 1. Environment File (.env) - **RECOMMENDED**

Edit the `.env` file in the Archon root directory:

```bash
C:\Jarvis\AI Workspace\Archon\.env
```

#### For DeepSeek (Recommended - Lower Cost):

1. **Get API Key**: 
   - Go to https://platform.deepseek.com/
   - Sign up or log in
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key

2. **Add to .env**:
```env
# External Validator Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
VALIDATOR_LLM_PROVIDER=deepseek
VALIDATOR_MODEL=deepseek-chat
VALIDATOR_TEMPERATURE=0.1
```

#### For OpenAI:

1. **Get API Key**:
   - Go to https://platform.openai.com/
   - Sign up or log in
   - Navigate to API Keys
   - Create new secret key
   - Copy the key

2. **Add to .env**:
```env
# External Validator Configuration
OPENAI_API_KEY=your_openai_api_key_here
VALIDATOR_LLM_PROVIDER=openai
VALIDATOR_MODEL=gpt-4o
VALIDATOR_TEMPERATURE=0.1
```

### 2. Runtime Configuration via API

You can also configure the API key at runtime using the configuration endpoint:

```bash
curl -X POST http://localhost:8053/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "deepseek",
    "api_key": "your_api_key_here",
    "model": "deepseek-chat",
    "temperature": 0.1
  }'
```

### 3. Configuration File (validator_config.json)

The validator automatically creates a config file at:
```
C:\Jarvis\AI Workspace\Archon\python\src\agents\external_validator\validator_config.json
```

You can edit this file directly (note: API key is not stored here for security):
```json
{
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "temperature": 0.1,
    "max_tokens": 4096
  },
  "validation": {
    "enable_deterministic": true,
    "enable_cross_check": true,
    "confidence_threshold": 0.9,
    "max_context_tokens": 5000,
    "enable_proactive_triggers": true
  }
}
```

### 4. Docker Environment Variables

When using Docker, pass the API key as an environment variable:

```bash
# Using docker run
docker run -e DEEPSEEK_API_KEY=your_key_here archon-validator

# Using docker-compose
docker compose --profile validator up -d
```

The docker-compose.yml already includes the environment variable mapping.

## Priority Order

The External Validator loads configuration in this priority order:
1. Environment variables (highest priority)
2. .env file
3. validator_config.json file
4. Default values (lowest priority)

## Testing Your API Key

After configuring your API key, test it:

### 1. Check Health Status:
```bash
curl http://localhost:8053/health
```

Expected response should show `"llm_connected": true`

### 2. Test Validation:
```bash
curl -X POST http://localhost:8053/validate \
  -H "Content-Type: application/json" \
  -d '{
    "output": "def test(): return True",
    "validation_type": "code"
  }'
```

### 3. Check Configuration:
```bash
curl http://localhost:8053/config
```

Should show your configured provider and model.

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use .env file** which is already in .gitignore
3. **Rotate keys regularly** for production use
4. **Set usage limits** in your API provider dashboard
5. **Monitor usage** to detect any anomalies

## Cost Considerations

### DeepSeek (Recommended)
- **Cost**: ~$0.14 per million input tokens
- **Performance**: Excellent for validation tasks
- **Speed**: Fast response times
- **Recommendation**: Best value for validation

### OpenAI GPT-4o
- **Cost**: ~$5 per million input tokens
- **Performance**: Superior reasoning
- **Speed**: Moderate response times
- **Recommendation**: Use for critical validations

## Troubleshooting

### API Key Not Working:

1. **Check logs**:
```bash
docker logs archon-validator
```

2. **Verify environment**:
```python
import os
print(os.getenv('DEEPSEEK_API_KEY'))  # Should show your key
```

3. **Test connection**:
```python
from src.agents.external_validator import ValidatorConfig, LLMClient
config = ValidatorConfig()
client = LLMClient(config)
await client.check_connection()  # Should return True
```

### Common Issues:

- **"No API key configured"**: Check .env file location and format
- **"Invalid API key"**: Verify key is correct and active
- **"Rate limited"**: Check API usage limits
- **"Connection refused"**: Ensure validator service is running

## Quick Start

1. **Add your API key to .env**:
```bash
echo "DEEPSEEK_API_KEY=your_key_here" >> .env
```

2. **Start the validator**:
```bash
docker compose --profile validator up -d
```

3. **Verify it's working**:
```bash
curl http://localhost:8053/health | grep llm_connected
```

You should see `"llm_connected": true` in the response!