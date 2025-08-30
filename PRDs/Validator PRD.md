# Product Requirements Document (PRD)

**Project: External Validator Agent for Archon System**  
**Version: 1.0**  
**Author: [Your Name]**  
**Date: August 30, 2025**

## 1. Executive Summary
The External Validator Agent is a standalone "referee" service designed to monitor and validate communications, prompts, and outputs from the Archon system and its agents, ensuring no hallucinations or errors while maintaining independence from Archon’s internal Claude Code-based framework. It stands outside Archon as an external component but integrates seamlessly via Archon’s MCP (Model Context Protocol) or API calls, allowing Archon to send data for validation. The agent uses an external LLM (preferably DeepSeek or OpenAI models like GPT-4o, configurable via Archon UI as the "Validator API") with low temperature (0-0.2) for deterministic, fact-based validation. This PRD defines the agent's features, functions, architecture, and integration with Archon, focusing on truth-seeking, transparency, and efficiency to achieve ≥50% hallucination reduction, ≥90% verdict accuracy, and ≥85% precision in monitored workflows, as tested in the Standard Coding Workflow Test (SCWT).

## 2. Goals & Objectives
- Create a standalone external agent that acts as a referee to police Archon’s internal agents (Claude Code-based), monitoring prompts, communications, and outputs for accuracy, consistency, and policy compliance.
- Use external LLMs (DeepSeek preferred, OpenAI as alternative) to avoid Claude Code biases, ensuring impartial validation.
- Enable LLM API setup from Archon UI, labeled as "Validator API," for user-configurable integration (e.g., API key, model selection, temperature).
- Reduce hallucinations and errors in Archon workflows by performing deterministic checks, cross-checks against context, and confidence filtering.
- Integrate with Archon’s MCP for seamless data flow, without modifying Archon’s core (e.g., primary agent, sub-agents).
- Achieve measurable improvements: ≥50% hallucination reduction, ≥30% knowledge reuse in verdicts, 70-85% token/compute savings, and ≥20% fewer iterations in Archon tasks.
- Maintain simplicity (setup ≤10 minutes) and scalability, with no dependency on Claude Code for validation.

## 3. Scope
**In Scope:**
- Development of the External Validator Agent as a separate service (e.g., Python/FastAPI backend) that integrates with Archon via MCP/API.
- LLM configuration via Archon UI ("Validator API" section) for DeepSeek/OpenAI, with temp adjustment (0-0.2).
- Features: Deterministic checks, cross-checks, JSON verdicts, proactive triggers, integration with DeepConf, Graphiti, REF, PRP context.
- Monitoring of Archon prompts/comms/outputs (e.g., sent via MCP calls).
- Testing via SCWT to benchmark improvements in Archon workflows.

**Out of Scope:**
- Internal modifications to Archon’s Claude Code-based agents (validator stands outside).
- Use of Claude Code as LLM for validator (external only: DeepSeek/OpenAI).
- The "other API function" mentioned (assumed separate from Validator API).
- Non-validation tasks (e.g., code generation); focus on referee role.

## 4. System Architecture
The External Validator Agent is a standalone service (e.g., Dockerized Python/FastAPI app) that runs outside Archon, communicating via MCP/API endpoints (e.g., POST `/validate` with payload including prompts/outputs/context). Archon sends data for validation, and the agent returns JSON verdicts. The LLM API (DeepSeek/OpenAI) is configured via Archon UI, stored securely (e.g., encrypted env variables).

**Key Components:**
1. **API Endpoint**: FastAPI server with MCP-compatible routes (e.g., `/validate`, `/configure_llm`).
2. **LLM Interface**: External LLM (DeepSeek default, OpenAI alternative) with temp config (0-0.2) for deterministic validation.
3. **Validation Engine**: Deterministic tools (pytest, ruff, mypy, semgrep) + cross-check logic using DeepConf (confidence ≥0.9), Graphiti (entity validation), REF Tools (docs grounding), PRP context (structured prompts ≤5k tokens).
4. **Proactive Triggers**: Auto-validation on Archon events (e.g., sub-agent output, code change) via webhooks/MCP hooks.
5. **Archon UI Integration**: "Validator API" section in Archon UI for LLM setup (API key, model, temp).
6. **Logging & Monitoring**: Track validation requests and costs (e.g., token usage).

**Data Flows**: Archon → MCP/API call to Validator (prompt/output/context) → Validation (checks + LLM cross-check) → JSON verdict back to Archon → Archon integration/reprompt.

## 5. Functional Requirements
**5.1 LLM Configuration**
- Archon UI section ("Validator API") for setup: Enter API key, select model (DeepSeek default, OpenAI alternative), set temp (0-0.2).
- MCP tool: `configure_validator [model=deepseek] [temp=0.1]`.

**5.2 Monitoring & Input**
- Monitor Archon prompts, comms, outputs via MCP/API (e.g., POST `/validate` with JSON payload: {"prompt": "...", "output": "...", "context": PRP pack ≤5k tokens}).
- Support proactive triggers (e.g., auto-validate on Archon events like sub-agent output).

**5.3 Deterministic Checks**
- Run build/test (e.g., pytest), lint (ruff/eslint), type (mypy), security (semgrep), no-op detection.

**5.4 Cross-Check & Validation**
- Verify outputs against PRP context (file paths, examples, criteria), Graphiti entities (e.g., `Module: WebApp`), REF docs (e.g., OAuth2 specs).
- Use DeepConf for confidence filtering (≥0.9); reject uncited claims or inconsistencies.

**5.5 Output & Feedback**
- JSON verdicts: {"status": "PASS/FAIL/UNSURE", "issues": [], "evidence": [], "fixes": []}.
- Provenance tracking: path@commit:span for all references.

**5.6 UI Enhancements**
- Validator API setup in Archon UI.
- Validation summary dashboard with verdicts, costs, DeepConf scores.

**5.7 CLI/MCP Tools**
- `validate <payload> [temp=0.1]` (MCP call to external agent).

## 6. Non-Functional Requirements
- **Simplicity**: Setup ≤10 minutes (Docker/FastAPI app); UI config in Archon.
- **Performance**: Validation <2s; 70-85% token/compute savings.
- **Scalability**: Handle multi-user Archon workloads.
- **Transparency**: Log verdicts/costs; UI for visibility.
- **Security**: Encrypt API keys; local/offline default.

## 7. Standard Coding Workflow Test (SCWT)
The SCWT is a single, repeatable benchmark test to track improvements across phases, simulating a project-agnostic coding workflow to stress the validator's monitoring.

**SCWT Definition**:
- **Task**: Monitor Archon workflow for building a web app feature (e.g., `auth_endpoint.py`).
- **Setup**: Mock Archon repo; send prompts/outputs to validator via MCP/API.
- **Execution**:
  - Archon query: “Build secure auth endpoint.”
  - Validator monitors prompts/outputs, performs checks, returns verdicts.
- **Metrics**:
  - Hallucination Rate: ≤10%.
  - Knowledge Reuse: ≥30%.
  - Task Efficiency: ≥30% reduction.
  - Communication Efficiency: ≥20% fewer iterations.
  - Precision: ≥85%.
  - Verdict Accuracy: ≥90%.
  - UI Usability: ≥10% CLI reduction.
- **Tools**: Scripts for execution, scorer, counter. Manual audits.
- **Baseline**: Run on unvalidated Archon.

**SCWT Evolution**:
- **Phase 1**: Basic checks.
- **Phase 2**: Cross-check with PRP.
- **Phase 3**: Full integration with DeepConf/Graphiti/REF.

## 8. Phased Roadmap
Development is gated: Implement → Test → Run SCWT → Improve → Proceed.

**Phase 1: Setup & Deterministic Checks (2-4 weeks)**
- **Implement**: External FastAPI service; UI config for Validator API (DeepSeek/OpenAI, temp).
- **Test**: Deterministic checks (pytest, ruff).
- **SCWT Benchmark**: Task efficiency (≥15% reduction), hallucination rate (~30%).
- **Improve**: Tune checks if efficiency <15%.
- **Gate**: Proceed if SCWT shows ≥10% gain.

**Phase 2: Cross-Check & Integration (3-5 weeks)**
- **Implement**: PRP context handling, Graphiti/REF integration, DeepConf filtering.
- **Test**: Cross-checks, MCP/API with Archon.
- **SCWT Benchmark**: Hallucination reduction (≥50%), precision (≥85%).
- **Improve**: Tune temp/thresholds if reduction <50%.
- **Gate**: Proceed if SCWT shows ≥40% error drop.

**Phase 3: Proactive Triggers & Polish (3-5 weeks)**
- **Implement**: Auto-triggers, JSON verdicts, UI validation summary.
- **Test**: Proactive monitoring, end-to-end with Archon.
- **SCWT Benchmark**: Knowledge reuse (≥30%), efficiency (≥30% reduction).
- **Improve**: Optimize triggers if reuse <30%.
- **Gate**: Proceed if SCWT meets all benchmarks.

## 9. Success Metrics (Via SCWT)
- Hallucination rate: ≤10%.
- Knowledge reuse: ≥30%.
- Efficiency: 70-85% token/compute savings; ≥30% task time reduction; ≥20% fewer iterations.
- Precision: ≥85%.
- Verdict accuracy: ≥90%.
- Setup time: ≤10 minutes.

## 10. Risks & Mitigations
- **Integration Friction**: Mitigation: MCP compatibility; test with SCWT.
- **Cost Overruns**: Mitigation: Track in SCWT/UI; optimize with DeepConf.
- **Scalability Issues**: Mitigation: Pluggable backends; test in Phase 3.
- **Hallucination Persistence**: Mitigation: Low-temp LLM; Graphiti/REF/PRP provenance.

## 11. Deliverables
- External Validator service (FastAPI repo).
- Archon UI updates for Validator API config.
- MCP/API schemas (e.g., `/validate` payload).
- SCWT: Scripts/docs for setup, execution, metrics.
- Docs: Setup guide, architecture diagrams, examples.
- Samples: JSON verdicts, PRP context, validation logs.
