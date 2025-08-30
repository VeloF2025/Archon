# Phase 5: External Validator Agent

## Overview

The External Validator Agent is a standalone referee service that monitors and validates all communications, prompts, and outputs from the Archon system. It operates independently using external LLMs (DeepSeek or OpenAI) to ensure impartial validation.

## Key Features

- **Independent Validation**: Operates outside Archon's Claude Code framework
- **External LLM Integration**: Uses DeepSeek (preferred) or OpenAI models
- **Deterministic Checks**: Runs pytest, ruff, mypy, eslint, semgrep
- **Cross-Validation**: Validates against PRP context, Graphiti entities, REF docs
- **MCP Integration**: Seamless integration via Model Context Protocol
- **Real-time Monitoring**: Proactive validation triggers on Archon events
- **JSON Verdicts**: Standardized validation responses with evidence

## Architecture

```
┌─────────────────────────────────────────┐
│            Archon System                 │
│  ┌────────┐  ┌────────┐  ┌──────────┐  │
│  │ Server │  │  MCP   │  │  Agents  │  │
│  └────┬───┘  └────┬───┘  └────┬─────┘  │
│       └───────────┼───────────┘         │
└───────────────────┼─────────────────────┘
                    │
                    ▼ MCP/API
        ┌──────────────────────┐
        │  External Validator  │
        │    (Port 8053)       │
        ├──────────────────────┤
        │ • FastAPI Service    │
        │ • Validation Engine  │
        │ • LLM Client        │
        │ • Deterministic     │
        │ • Cross-Check       │
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │   External LLM       │
        │ (DeepSeek/OpenAI)    │
        └──────────────────────┘
```

## Quick Start

### 1. Configure Environment

Create or update `.env` file:

```bash
# For DeepSeek (recommended)
DEEPSEEK_API_KEY=your_deepseek_api_key
VALIDATOR_LLM_PROVIDER=deepseek
VALIDATOR_MODEL=deepseek-chat
VALIDATOR_TEMPERATURE=0.1

# OR for OpenAI
OPENAI_API_KEY=your_openai_api_key
VALIDATOR_LLM_PROVIDER=openai
VALIDATOR_MODEL=gpt-4o
VALIDATOR_TEMPERATURE=0.1

# Validation settings
VALIDATOR_CONFIDENCE_THRESHOLD=0.9
```

### 2. Start the Validator

#### With Docker:
```bash
# Start validator with other services
docker compose --profile validator up -d

# Or start validator only
docker compose up archon-validator -d
```

#### Standalone:
```bash
cd python/src/agents/external_validator
pip install -r ../../requirements.validator.txt
uvicorn main:app --host 0.0.0.0 --port 8053
```

### 3. Verify Health

```bash
curl http://localhost:8053/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_provider": "deepseek",
  "llm_connected": true,
  "deterministic_available": true,
  "uptime_seconds": 120,
  "total_validations": 0
}
```

## API Endpoints

### Validation

**POST /validate**
```json
{
  "output": "Code or content to validate",
  "prompt": "Original prompt (optional)",
  "context": {
    "prp": "Project requirements",
    "files": ["file1.py", "file2.js"],
    "entities": [{"name": "UserService"}],
    "docs": "API documentation"
  },
  "validation_type": "code",
  "temperature_override": 0.05
}
```

Response:
```json
{
  "status": "PASS",
  "issues": [],
  "evidence": [
    {
      "source": "deterministic",
      "content": "All checks passed",
      "confidence": 1.0
    }
  ],
  "metrics": {
    "hallucination_rate": 0.0,
    "confidence_score": 0.95,
    "validation_time_ms": 1234
  },
  "summary": "Validation passed with 5 supporting evidence points."
}
```

### Configuration

**POST /configure**
```json
{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "temperature": 0.1,
  "confidence_threshold": 0.9
}
```

### Health Check

**GET /health**
Returns validator status and configuration.

### Metrics

**GET /metrics**
Returns validation metrics and statistics.

## MCP Tools

The validator registers these MCP tools with Archon:

### validate
Validate content using External Validator
```
mcp:validate output="code to check" validation_type="code"
```

### configure_validator
Configure validator settings
```
mcp:configure_validator provider="deepseek" temperature=0.05
```

### validator_health
Check validator status
```
mcp:validator_health
```

## Validation Types

1. **code**: Validates code syntax, patterns, and quality
2. **documentation**: Validates documentation accuracy
3. **prompt**: Validates prompt/response pairs
4. **output**: Validates agent outputs
5. **full**: Comprehensive validation (default)

## Deterministic Checks

The validator runs these tools when available:

- **pytest**: Test execution and validation
- **ruff**: Python linting and formatting
- **mypy**: Python type checking
- **eslint**: JavaScript/TypeScript linting
- **semgrep**: Security and pattern analysis

## Cross-Validation

### PRP Context
Validates output against project requirements:
- Requirement coverage
- Implementation completeness
- Specification compliance

### Graphiti Entities
Validates entity references:
- Entity existence
- Relationship accuracy
- Naming consistency

### REF Documentation
Validates against documentation:
- API compliance
- Technical accuracy
- Terminology consistency

### DeepConf Filtering
Applies confidence thresholds:
- Filters low-confidence claims
- Adjusts severity based on evidence
- Requires 0.9+ confidence for critical decisions

## Gaming Detection (DGTS)

The validator detects and blocks gaming patterns:

```python
# DETECTED PATTERNS:
assert True  # Meaningless test
return "mock_data"  # Fake implementation
# validation_required = False  # Disabled validation
if False:  # Unreachable code
```

## Success Metrics

Target metrics from SCWT benchmark:

- **Hallucination Reduction**: ≥50%
- **Verdict Accuracy**: ≥90%
- **Precision**: ≥85%
- **Knowledge Reuse**: ≥30%
- **Token Savings**: 70-85%
- **Task Efficiency**: ≥30% reduction
- **Communication Efficiency**: ≥20% fewer iterations

## Proactive Triggers

The validator automatically validates on these events:

- `agent_output`: When agents produce output
- `code_change`: When code is modified
- `prompt_submission`: When prompts are submitted
- `sub_agent_response`: When sub-agents respond
- `pre_deployment`: Before deployment

## Configuration UI

Access validator configuration in Archon UI:

1. Navigate to Settings → Validator API
2. Enter API key and select provider
3. Adjust temperature (0.0-0.2)
4. Set confidence threshold
5. Enable/disable proactive triggers

## Troubleshooting

### LLM Connection Issues
```bash
# Check API key
echo $DEEPSEEK_API_KEY

# Test connection
curl http://localhost:8053/health

# Check logs
docker logs archon-validator
```

### Deterministic Tools Missing
```bash
# Install tools
pip install pytest ruff mypy
npm install -g eslint

# Verify installation
pytest --version
ruff --version
```

### Validation Failures
```bash
# Check validation details
curl -X POST http://localhost:8053/validate \
  -H "Content-Type: application/json" \
  -d '{"output": "test code", "validation_type": "code"}'

# Adjust confidence threshold
curl -X POST http://localhost:8053/configure \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.8}'
```

## Development

### Running Tests
```bash
cd python/src/agents/external_validator
pytest tests/ -v
```

### Adding Validation Rules
Edit `deterministic.py` to add patterns:
```python
gaming_patterns = [
    (r"pattern_regex", "Description", ValidationSeverity.ERROR),
    # Add new patterns here
]
```

### Extending Cross-Checks
Edit `cross_check.py` to add validators:
```python
async def _check_custom(self, output, context):
    # Add custom validation logic
    pass
```

## Integration with Existing Validation

The External Validator complements:

1. **Claude Validator** (`validation/`): Internal gaming detection
2. **DGTS Validator** (`validation/dgts_validator.py`): System gaming prevention
3. **Doc-Driven Validator** (`validation/doc_driven_validator.py`): Documentation compliance

All validators work together for comprehensive coverage.

## Next Steps

1. **Configure API Keys**: Set up DeepSeek or OpenAI credentials
2. **Start Validator**: Launch with Docker or standalone
3. **Test Validation**: Run sample validations
4. **Enable Triggers**: Configure proactive validation
5. **Monitor Metrics**: Track improvement via SCWT

## Support

For issues or questions:
- Check logs: `docker logs archon-validator`
- Review metrics: `http://localhost:8053/metrics`
- Test health: `http://localhost:8053/health`
- See PRD: `/PRDs/Validator PRD.md`