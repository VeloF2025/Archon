# Phase 5: External Validator Agent Implementation PRP

**Phase**: 5  
**Component**: External Validator Agent  
**Status**: In Development  
**Version**: 1.0  
**Date**: August 30, 2025

## Overview

Phase 5 introduces the External Validator Agent - a standalone referee service that monitors and validates all communications, prompts, and outputs from the Archon system. This agent operates independently from Archon's internal Claude Code-based framework, using external LLMs (DeepSeek or OpenAI) to ensure impartial validation.

## Key Objectives

1. **Independent Validation**: Create a standalone service that acts as an external referee for Archon's agents
2. **Hallucination Reduction**: Achieve ≥50% reduction in hallucinations through deterministic checks
3. **External LLM Integration**: Use DeepSeek (preferred) or OpenAI models with low temperature (0-0.2)
4. **Seamless Integration**: Connect via MCP/API without modifying Archon's core
5. **Measurable Improvements**: Track via Standard Coding Workflow Test (SCWT)

## Architecture Components

### 1. External Validator Service
- **Location**: `/python/src/agents/external_validator/`
- **Technology**: FastAPI + Python
- **Port**: 8053 (dedicated validator port)
- **Docker**: Separate container for isolation

### 2. LLM Configuration
- **Primary**: DeepSeek API (low cost, high accuracy)
- **Fallback**: OpenAI GPT-4o
- **Temperature**: 0.0-0.2 for deterministic responses
- **UI Config**: Archon Settings → Validator API

### 3. Validation Engine
- **Deterministic Checks**: pytest, ruff, mypy, semgrep
- **Cross-Check Logic**: DeepConf (≥0.9 confidence), Graphiti entities, REF docs
- **PRP Context**: Structured prompts ≤5k tokens
- **JSON Verdicts**: Standardized validation responses

### 4. Integration Points
- **MCP Tools**: `validate`, `configure_validator`
- **API Endpoints**: `/validate`, `/configure_llm`, `/health`
- **Webhooks**: Auto-validation triggers
- **UI Dashboard**: Validation summary and metrics

## Implementation Phases

### Phase 5.1: Core Service Setup (Week 1-2)
- [ ] Create FastAPI service structure
- [ ] Implement basic validation endpoint
- [ ] Set up Docker container
- [ ] Add health check endpoint
- [ ] Create validator configuration system

### Phase 5.2: LLM Integration (Week 2-3)
- [ ] DeepSeek API integration
- [ ] OpenAI fallback support
- [ ] Temperature configuration
- [ ] API key management (encrypted)
- [ ] Model selection logic

### Phase 5.3: Validation Logic (Week 3-4)
- [ ] Deterministic check runners
- [ ] Cross-check implementation
- [ ] Confidence filtering (DeepConf)
- [ ] Entity validation (Graphiti)
- [ ] Documentation grounding (REF)

### Phase 5.4: Archon Integration (Week 4-5)
- [ ] MCP tool registration
- [ ] Webhook setup for triggers
- [ ] UI configuration panel
- [ ] Validation dashboard
- [ ] Metrics tracking

### Phase 5.5: SCWT Benchmark (Week 5-6)
- [ ] Implement SCWT test suite
- [ ] Baseline measurements
- [ ] Performance optimization
- [ ] Documentation
- [ ] Release preparation

## Success Metrics

### Primary Goals
- **Hallucination Reduction**: ≥50%
- **Verdict Accuracy**: ≥90%
- **Precision**: ≥85%
- **Knowledge Reuse**: ≥30%
- **Token Savings**: 70-85%

### Performance Targets
- **Validation Speed**: <2s per check
- **Setup Time**: ≤10 minutes
- **Task Efficiency**: ≥30% reduction
- **Communication Efficiency**: ≥20% fewer iterations

## File Structure

```
/python/src/agents/external_validator/
├── __init__.py
├── main.py                 # FastAPI app
├── config.py              # Configuration management
├── llm_client.py          # DeepSeek/OpenAI interface
├── validation_engine.py   # Core validation logic
├── deterministic.py       # Deterministic checks
├── cross_check.py         # Cross-validation logic
├── models.py              # Pydantic models
├── mcp_integration.py    # MCP tool handlers
├── webhooks.py           # Event triggers
└── scwt/                  # SCWT benchmark
    ├── __init__.py
    ├── runner.py
    ├── metrics.py
    └── test_cases.py

/archon-ui-main/src/pages/ValidatorConfig.tsx  # UI configuration
/archon-ui-main/src/pages/ValidatorDashboard.tsx  # Metrics dashboard
```

## Integration with Existing Validation

The External Validator complements existing validation systems:

1. **Claude Validator** (internal): Detects gaming behaviors
2. **DGTS Validator** (internal): Prevents system gaming
3. **External Validator** (Phase 5): Independent verification using external LLMs

All three work together to ensure comprehensive validation coverage.

## Risk Mitigation

1. **API Cost Management**: Track token usage, implement caching
2. **Latency Issues**: Async validation, batch processing
3. **Integration Complexity**: Clean MCP interface, minimal coupling
4. **False Positives**: Tunable confidence thresholds

## Next Steps

1. Create validator service directory structure
2. Implement basic FastAPI service
3. Add DeepSeek integration
4. Create MCP tools
5. Build UI configuration panel
6. Implement SCWT benchmark

## References

- Original PRD: `/PRDs/Validator PRD.md`
- Existing Validation: `/python/src/agents/validation/`
- MCP Server: `/python/src/mcp_server/`
- Archon UI: `/archon-ui-main/`