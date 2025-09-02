# ⚙️ Configuration Guide

This guide covers all configuration options for Archon, from basic setup to advanced customization.

## 📋 Quick Configuration Checklist

- [ ] Set up Supabase connection
- [ ] Configure API keys for LLM providers
- [ ] Set embedding provider and model
- [ ] Configure RAG strategies
- [ ] Set up MCP client connections
- [ ] Configure optional features (agents, projects)

## 🔑 Environment Variables

### Required Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

#### Essential Variables

```bash
# Supabase Configuration (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here

# Service Ports (Optional - defaults shown)
HOST=localhost
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
ARCHON_UI_PORT=3737
```

#### Optional Variables

```bash
# Logging and Monitoring
LOGFIRE_TOKEN=your-logfire-token
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Production Mode
PROD=false  # Set to true for production deployments

# Frontend Configuration
VITE_ALLOWED_HOSTS=192.168.1.100,myhost.local
```

### Supabase Setup

1. **Create Project**: Go to [supabase.com](https://supabase.com/) and create a new project

2. **Get Credentials**: Navigate to Settings → API
   - Copy your project URL
   - Copy the **service_role** key (NOT the anon key)

3. **Database Setup**: Run the setup SQL:
   ```sql
   -- In Supabase SQL Editor, execute:
   -- Contents of migration/complete_setup.sql
   ```

⚠️ **Critical**: Always use the **service_role** key. The anon key will cause permission errors.

## 🤖 LLM Provider Configuration

### Supported Providers

Archon supports multiple LLM and embedding providers:

- **OpenAI** (GPT-4, GPT-3.5-turbo, text-embedding-3-small)
- **Google Gemini** (gemini-pro, text-embedding-004)
- **Ollama** (Local models)
- **Anthropic Claude** (via MCP integration)

### Configuration via UI

1. Open Archon at http://localhost:3737
2. Navigate to **Settings** → **API Keys**
3. Select your preferred provider
4. Enter your API key
5. Choose model variants

### Configuration via Database

API keys and model settings are stored encrypted in Supabase:

```sql
-- View current configuration
SELECT * FROM credentials WHERE key LIKE 'OPENAI%' OR key LIKE 'MODEL%';

-- Update OpenAI API key (encrypted automatically)
INSERT INTO credentials (key, value) VALUES ('OPENAI_API_KEY', 'your-api-key-here')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
```

### Provider-Specific Setup

#### OpenAI

```bash
# Environment (optional - prefer UI configuration)
OPENAI_API_KEY=your-openai-key

# Models supported:
# - gpt-4o (recommended)
# - gpt-4-turbo
# - gpt-3.5-turbo
# - text-embedding-3-small (embeddings)
```

#### Google Gemini

```bash
# Configure via Settings UI or database
# Models supported:
# - gemini-1.5-pro
# - gemini-1.5-flash
# - text-embedding-004 (embeddings)
```

#### Ollama (Local)

```bash
# Install Ollama locally
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2
ollama pull nomic-embed-text

# Configure Archon to use Ollama
# Provider: ollama
# Model: llama2
# Embedding Model: nomic-embed-text
```

## 🔍 RAG Configuration

### Search Strategies

Configure retrieval strategies via Settings UI:

#### Hybrid Search (Recommended)
- Combines keyword and semantic search
- Best accuracy for most use cases
- Moderate resource usage

```json
{
  "use_hybrid_search": true,
  "keyword_weight": 0.3,
  "semantic_weight": 0.7
}
```

#### Contextual Embeddings
- Enhanced context awareness
- Higher accuracy for complex queries
- Increased processing time

```json
{
  "use_contextual_embeddings": true,
  "context_window": 512
}
```

#### Reranking
- Post-retrieval result optimization
- Best accuracy but requires agents service
- Higher resource usage

```json
{
  "use_reranking": true,
  "rerank_top_k": 10,
  "final_top_k": 5
}
```

### Embedding Configuration

```bash
# Vector dimensions (must match your embedding model)
EMBEDDING_DIMENSIONS=1536  # OpenAI text-embedding-3-small
EMBEDDING_DIMENSIONS=768   # For other models
```

### Performance Tuning

```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_concurrent_requests": 10,
  "timeout_seconds": 30
}
```

## 🕸️ Web Crawling Configuration

### Crawler Settings

Configure via Settings UI or database:

```json
{
  "crawl_max_concurrent": 10,
  "crawl_batch_size": 50,
  "memory_threshold_percent": 80,
  "dispatcher_check_interval": 0.5,
  "max_depth": 3,
  "respect_robots_txt": true,
  "user_agent": "Archon-Crawler/1.0"
}
```

### Site-Specific Configuration

```json
{
  "allowed_domains": ["docs.example.com", "wiki.example.com"],
  "blocked_paths": ["/admin", "/private"],
  "custom_selectors": {
    "content": "main, article, .content",
    "title": "h1, .page-title",
    "exclude": ".navigation, .sidebar, footer"
  }
}
```

### Rate Limiting

```json
{
  "requests_per_second": 2,
  "concurrent_requests": 5,
  "delay_between_requests": 1.0,
  "respect_crawl_delay": true
}
```

## 🔌 MCP Integration

### Client Configuration

Connect AI coding assistants to Archon's MCP server:

#### Claude Code Configuration

```json
{
  "mcpServers": {
    "archon": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-archon",
        "http://localhost:8051"
      ]
    }
  }
}
```

#### Cursor Configuration

```json
{
  "mcp": {
    "servers": {
      "archon": {
        "command": "curl",
        "args": ["-X", "POST", "http://localhost:8051/tools/execute"]
      }
    }
  }
}
```

### Available MCP Tools

```bash
# List available tools
curl http://localhost:8051/tools

# Test tool execution
curl -X POST http://localhost:8051/tools/archon:perform_rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "FastAPI setup", "match_count": 5}'
```

### Connection Security

```bash
# Enable authentication (optional)
MCP_AUTH_TOKEN=your-secure-token

# Configure CORS for web clients
MCP_ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 📊 AI Agents Configuration

### Agent System Setup

Enable the agents service for advanced AI features:

```bash
# Start with agents enabled
docker compose --profile agents up -d

# Or permanently enable
echo "AGENTS_ENABLED=true" >> .env
```

### Specialized Agents

Configure individual agent capabilities:

```json
{
  "agents": {
    "system_architect": {
      "enabled": true,
      "model": "gpt-4o",
      "temperature": 0.1
    },
    "code_implementer": {
      "enabled": true,
      "model": "gpt-4o",
      "max_tokens": 4000
    },
    "security_auditor": {
      "enabled": true,
      "model": "gpt-4o",
      "security_level": "strict"
    }
  }
}
```

### DeepConf Confidence Engine

```json
{
  "deepconf": {
    "enabled": true,
    "confidence_threshold": 0.8,
    "uncertainty_handling": "flag",
    "consensus_algorithm": "weighted_voting"
  }
}
```

### SCWT Metrics

```json
{
  "scwt": {
    "enabled": true,
    "real_time_monitoring": true,
    "quality_gates": {
      "coverage_threshold": 0.95,
      "complexity_limit": 10,
      "performance_budget": 1500
    }
  }
}
```

## 📋 Project Management

### Project Features

Enable/disable project management features:

```json
{
  "projects": {
    "enabled": true,
    "auto_create_tasks": true,
    "ai_assisted_planning": true,
    "version_control": true
  }
}
```

### Task Management

```json
{
  "tasks": {
    "default_priority": "medium",
    "auto_assignment": true,
    "progress_tracking": true,
    "deadline_notifications": true
  }
}
```

## 🔐 Security Configuration

### Authentication

```bash
# Enable authentication (future feature)
AUTH_ENABLED=false
AUTH_PROVIDER=supabase
```

### API Security

```bash
# Rate limiting
API_RATE_LIMIT=100  # requests per minute
API_RATE_LIMIT_BURST=10

# CORS configuration
CORS_ORIGINS=http://localhost:3737,https://yourdomain.com
CORS_ALLOW_CREDENTIALS=true
```

### Data Protection

```bash
# Encryption keys (auto-generated if not provided)
ENCRYPTION_KEY=your-32-character-encryption-key

# Backup settings
BACKUP_ENABLED=true
BACKUP_INTERVAL=24h
BACKUP_RETENTION=30d
```

## 📊 Monitoring and Logging

### Logfire Integration

```bash
# Optional: Enable Logfire for observability
LOGFIRE_TOKEN=your-logfire-token

# Configure logging levels
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_DESTINATION=stdout
```

### Health Monitoring

```bash
# Health check endpoints
curl http://localhost:8181/health    # Main server
curl http://localhost:8051/health    # MCP server
curl http://localhost:8052/health    # Agents service
```

### Performance Monitoring

```json
{
  "monitoring": {
    "metrics_enabled": true,
    "performance_tracking": true,
    "error_reporting": true,
    "usage_analytics": false
  }
}
```

## 🎛️ Advanced Configuration

### Docker Compose Overrides

Create `docker-compose.override.yml` for local customization:

```yaml
services:
  archon-server:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./custom-config:/app/config

  archon-ui:
    ports:
      - "4000:3737"
```

### Custom Extensions

```bash
# Custom agent configurations
mkdir -p custom/agents
cp python/src/agents/configs/custom_agent.json custom/agents/

# Custom UI themes
mkdir -p custom/themes
cp archon-ui-main/src/styles/custom-theme.css custom/themes/
```

### Database Migrations

```bash
# Run custom migrations
docker exec archon-server python -m alembic upgrade head

# Create custom migration
docker exec archon-server python -m alembic revision --autogenerate -m "custom changes"
```

## 🔧 Environment-Specific Configurations

### Development

```bash
# .env.development
LOG_LEVEL=DEBUG
AGENTS_ENABLED=true
HOT_RELOAD=true
DEBUG_MODE=true
```

### Staging

```bash
# .env.staging
PROD=false
LOG_LEVEL=INFO
BACKUP_ENABLED=true
MONITORING_ENABLED=true
```

### Production

```bash
# .env.production
PROD=true
LOG_LEVEL=WARNING
SECURITY_ENHANCED=true
BACKUP_ENABLED=true
MONITORING_ENABLED=true
CACHE_ENABLED=true
```

## 🔍 Configuration Validation

### Validation Script

```bash
# Check configuration
node scripts/validate-config.js

# Or using Python
python scripts/validate_config.py
```

### Common Issues

1. **Wrong Supabase key type**: Use service_role, not anon
2. **Port conflicts**: Check for services using default ports
3. **Missing API keys**: Configure at least one LLM provider
4. **Insufficient resources**: Ensure adequate memory/CPU

## 📚 Next Steps

After configuration:

1. **[User Guide](docs/user-guide/)** - Learn how to use Archon
2. **[API Documentation](docs/api/)** - Integrate with Archon's APIs
3. **[Development Guide](CONTRIBUTING.md)** - Contribute to Archon
4. **[Troubleshooting](TROUBLESHOOTING.md)** - Solve common issues

---

**Need help with configuration?** Check our [troubleshooting guide](TROUBLESHOOTING.md) or join our community discussions!