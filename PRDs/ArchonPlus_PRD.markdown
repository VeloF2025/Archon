📋 Product Requirements Document (PRD)

**Project: Enhanced Archon AI Coding System (Archon+)**  
**Version: 2.4**  
**Author: [Your Name]**  
**Date: August 29, 2025**

## 1. Executive Summary
This PRD outlines a phased enhancement of Archon—an open-source AI Operating System for coding assistants—by forking the repository and integrating features inspired by ForgeFlow-v2-FF2 (e.g., adaptive memory, multi-agent orchestration), `context-engineering-intro` (e.g., specialized sub-agent orchestration, Product Requirements Prompts (PRPs), example-driven development, proactive triggers), DeepConf for confidence-based reasoning, a prompt enhancement mechanism, Graphiti for real-time knowledge graphs, REF Tools MCP for token-efficient docs search, and HRM for hierarchical reasoning as an optional specialized agent. Archon+ will serve as a general-purpose coding system for diverse projects, with **global sub-agents** (20 specialized roles—e.g., Python Backend Coder, TS Frontend Linter—scalable to more) with focused memory scopes, reusable across projects. A meta-agent dynamically spawns and manages these agents, with the primary agent (Claude Code) orchestrating via prompt enhancement and validation. The validator is the **only external agent**, configurable with Archon's options for external LLMs (e.g., DeepSeek as a key option) and adjustable temperature (temp configurable to 0-0.2 for deterministic, fact-based "policing" to eliminate lying/hallucinations). All other agents use Claude Code’s framework internally. The React-based UI exposes agent roles, statuses, prompts, validation, and metrics. Development is gated by implementation, testing, and benchmarking using a single **Standard Coding Workflow Test (SCWT)** to track hallucination reduction, knowledge reuse, efficiency, and communication improvements.

Phases prioritize: (1) Specialized global sub-agent enhancements (with HRM, PRPs, proactive triggers), (2) Meta-agent for dynamic scaling, (3) Validator and prompt enhancers (with REF, external LLM options like DeepSeek), followed by memory/retrieval and optimizations.

## 2. Goals & Objectives
- Fork Archon and extend it into a robust, adaptive AI coding system for any coding project.
- Develop **global sub-agents** (specialized, role-based, starting with 20, scalable) reusable across projects, with optional HRM for reasoning tasks.
- Improve memory with temporal knowledge graphs (Graphiti), REF for docs access, and PRPs for structured prompts.
- Enhance orchestration with Claude Code as the primary agent managing even more agents via meta-agent, prompt enhancement, and validation loops.
- Implement validator as the only external agent, using Archon's external LLM options (e.g., DeepSeek) with configurable temp (0-0.2) to ensure no lying/hallucinations while "policing" Claude Code agents.
- Enhance UI for visibility/control (e.g., agent dashboard, SCWT metrics).
- Reduce hallucinations (≥50%) via validator, DeepConf, Graphiti/REF provenance, and prompt refinement.
- Achieve efficiency: ≥30% knowledge reuse, 70-85% token/compute savings, ≥20% fewer iterations.
- Maintain simplicity (CLI/MCP primary, UI optional, setup ≤10 min) with scalability (Postgres/Neo4j).

## 3. Scope
**In Scope:**
- Forking Archon; adding dynamic specialized global sub-agents (20+, HRM option), meta-agent, validator (external agent with LLM options like DeepSeek, temp config), policy engine, prompt enhancement, memory layers, adaptive retrieval, PRP-like context packs, DeepConf, Graphiti, REF Tools MCP, proactive triggers, and UI enhancements.
- Workflow: User → Primary → Prompt Enhancement → Sub-Agents (via meta-agent, scalable) → Prompt Enhancement → Validator (external agent with DeepConf/Graphiti/REF) → Feedback/Output.
- CLI/MCP tools, project-agnostic; UI for visibility/control.
- Single SCWT for benchmarking, testing UI/REF/HRM/validator improvements.

**Out of Scope:**
- External LLMs beyond validator (all other agents use Claude Code framework).
- Full LLM fine-tuning or heavyweight ontologies.
- External services beyond Postgres/Qdrant/Neo4j/Supabase/Neon.
- Non-coding use cases.

## 4. System Architecture
Base: Archon’s microservices (React UI, FastAPI backend, MCP server, Agents service, Supabase/Postgres).  
Enhancements: Modular services/adapters, pluggable for local (SQLite) or scaled (Postgres/pgvector, Neo4j) modes, with UI enhancements.

**Key Components:**
1. **Specialized Global Sub-Agent System**: Dynamic/unbounded agents (20 roles—e.g., Python Backend Coder, TS Frontend Linter—scalable) with focused memory scopes; optional HRM; PRP-based prompts; proactive triggers.
2. **Meta-Agent**: Spawns/manages specialized sub-agents based on project/task needs.
3. **Prompt Enhancer Service**: Refines prompts bidirectionally using Graphiti/REF/PRPs.
4. **Validator Service**: External agent (only one) using Archon's external LLM options (e.g., DeepSeek as default) with configurable temp (0-0.2 for no lying); deterministic checks + DeepConf + Graphiti/REF provenance.
5. **Memory Service**: Layered storage with Graphiti; PRP-driven examples.
6. **Adaptive Retriever**: Hybrid search (FTS + vectors + Graphiti + REF) with bandit weights/re-ranker.
7. **Context Assembler**: PRP-like Markdown packs per task, leveraging Graphiti/REF.
8. **Orchestrator**: Primary (Claude Code) delegates to sub-agents via meta-agent, using PRP prompts.
9. **Policy Engine**: YAML rules for thresholds.
10. **DeepConf Wrapper**: Confidence filtering for agents/validator/enhancer.
11. **Graphiti Service**: Knowledge graph for real-time updates, temporal queries.
12. **REF Tools MCP**: Token-efficient docs search for public/private documentation.
13. **Local Adapter**: SQLite/JSON fallback; Graphiti with Kuzu database.
14. **Enhanced UI**: Agent dashboard, prompt viewer, validation summary, Graphiti explorer, REF search, HRM visualizer, SCWT metrics, task progress/debug.

**Data Flows**: User query → Context assembly (with Graphiti/REF/PRPs) → Prompt enhancement → Primary orchestration → Meta-agent (spawn sub-agents) → Prompt enhancement → Sub execution (HRM option, PRP prompts) → Prompt enhancement → Validation (external agent with rules + DeepConf + Graphiti/REF) → Integration/Reprompt → Verified output (UI visibility).

## 5. Functional Requirements
**5.1 Specialized Global Sub-Agent System:**
- Enhance agents: Dynamic/unbounded specialized roles (20—e.g., Python Backend Coder, TS Frontend Linter, Unit Test Generator, Security Auditor, Doc Writer, API Integrator, HRM Reasoning Agent—scalable) with focused memory scopes via JSON configs.
- Parallel execution via agent pool, conflict resolution (Redis or Git worktrees).
- PRP-based prompts: Structured with file paths, examples, test patterns.
- Proactive triggers: Auto-invoke agents (e.g., Security Auditor on code changes).
- HRM integration: Optional reasoning agent for algorithmic tasks.

**5.2 Meta-Agent:**
- Dynamic creation: Spawn specialized sub-agents (unbounded) based on task/project needs; access memory/retrieval (e.g., ADRs, standards).
- MCP tool: `meta_spawn <task_description> [project_type=web|data|infra] [max_agents=30]`.

**5.3 Prompt Enhancement:**
- Bidirectional refinement: Rewrite for clarity; inject Graphiti/REF/PRP entities/context/provenance; adapt from verdicts.
- MCP tool: `enhance_prompt <original> [direction=to-sub/from-sub] [project_context]`.
- Validator check: Post-enhancement quality/hallucination risk via Graphiti/REF/PRPs.

**5.4 Validator & DeepConf:**
- External agent only: Configurable with Archon's external LLM options (e.g., DeepSeek as default, others like Grok, GPT-4o); temp adjustable (0-0.2 for deterministic "policing" to eliminate lying/hallucinations).
- Deterministic checks: Build/test/lint/type/security/no-op detection (pytest, ruff, mypy, semgrep, eslint).
- Cross-check: Verify against packs/criteria; use Graphiti/REF/PRPs for entity/docs validation.
- DeepConf: Online/offline filtering; confidence thresholds (≥0.9).
- Output: JSON verdicts (status, issues, evidence, fixes).
- Policy enforcement: YAML rules (e.g., min_coverage: 75%).

**5.5 Memory & Knowledge:**
- Global/project knowledge as Markdown cards/ADRs; job memory as JSON (gotchas, decisions, contracts), scoped to roles.
- Graphiti: Store entities/relationships (e.g., `Module: WebApp`); bi-temporal queries.
- REF: Docs search for public/private sources.
- PRPs: Structured prompts with examples, file paths, test patterns.
- Auto-promote recurring elements (≥3 times); use Graphiti/REF/PRPs for provenance.

**5.6 Retrieval & Learning:**
- Hybrid retrieval: SQLite FTS5 (default) or Postgres/pgvector/Qdrant + Graphiti + REF; embeddings via all-MiniLM-L6-v2 (upgradable).
- Online bandit weights/re-ranker; incorporate Graphiti/REF signals.
- Graphiti search: Semantic, keyword, graph-based queries (e.g., `NODE_HYBRID_SEARCH_RRF`).
- REF search: Token-efficient docs access (e.g., `ref_search_documentation`, `ref_read_url`).

**5.7 Context Packs:**
- Generated per task: ≤5k tokens, with provenance (path@commit:span via Graphiti/REF/PRPs), checklists, Graphiti-derived entities, project-agnostic.
- Include role-specific job memory, acceptance criteria, PRP examples.

**5.8 Orchestration Workflow:**
- Primary (Claude Code) breaks tasks, uses meta-agent to spawn specialized subs (scalable), prompts via enhancer/PRPs, processes validated/enhanced feedback.
- Sub-agents execute (HRM option, PRP prompts); outputs routed to enhancer then validator (external agent).
- Proactive triggers auto-invoke agents (e.g., Security Auditor on code changes).
- On FAIL/UNSURE, primary reprompts or escalates.

**5.9 UI Enhancements:**
- **Agent Dashboard**: Real-time view of 20+ sub-agents (roles, statuses, PIDs); meta-agent controls.
- **Prompt Viewer**: Original vs. enhanced/PRP prompts, with Graphiti/REF/PRP context and validator feedback.
- **Validation Summary**: JSON verdicts, DeepConf scores, Graphiti/REF/PRP provenance, policy toggles.
- **Graphiti Explorer**: Interactive graph viewer for entities/relationships, with temporal filters.
- **REF Search Interface**: Search bar for docs, with markdown results and filters.
- **HRM Visualizer (Optional)**: Reasoning traces and confidence scores for HRM tasks.
- **SCWT Metrics Dashboard**: Live metrics (hallucination, reuse, efficiency, precision, verdict accuracy).
- **Task Progress/Debug**: Task overview, debug tools (`why`, `top`, context previews).

**5.10 CLI/MCP Tools:**
- Identical surface: `reindex`, `retrieve`, `assemble-context`, `learn`, `promote`, `why`, `top`, `validate`, `policy-check`, `extract-claims`, `deepconf-reason`, `enhance-prompt`, `meta_spawn`, `graphiti-add-episode`, `graphiti-search`, `ref_search_documentation`, `ref_read_url`, `generate_prp_context`.

## 6. Non-Functional Requirements
- **Simplicity**: Local setup ≤10 minutes (pipx/Docker); single binary/container (SQLite/Kuzu); UI optional.
- **Performance**: Retrieval <500ms; validation <2s; 70-85% token/compute savings; ≥20% fewer iterations.
- **Scalability**: Local (SQLite) to multi-user (Postgres/Neo4j, Archon UI) across project types.
- **Transparency**: Debug tools; Graphiti/REF/PRP provenance; UI for visibility.
- **Security**: Local/offline default; optional cloud (Supabase/Neon).

## 7. Standard Coding Workflow Test (SCWT)
The SCWT is a single, repeatable benchmark test to track improvements across phases, simulating a project-agnostic coding workflow (e.g., building a web app feature) to stress all components (20+ sub-agents, meta-agent, memory, retrieval, orchestration, prompt enhancement, validation (external agent), Graphiti, REF, HRM, UI, PRPs).

**SCWT Definition**:
- **Task**: Develop a new feature for a web app—e.g., a user authentication endpoint (`auth_endpoint.py` in Python/Flask), TypeScript/React frontend (`Login.tsx`), unit tests (≥75% coverage), and documentation (README update). Inputs: Module code, project specs, coding standards, prior gotchas, ADRs, REF docs (e.g., OAuth2 specs).
- **Setup**: Mock repo with `backend/auth_endpoint.py`, `frontend/Login.tsx`, `tests/test_auth.py`, `docs/standards.md`, `knowledge/project/api_design.md`, `adr/ADR-2025-08-01-oauth.md`. Initialize with 2-3 job memories (JSON, role-specific).
- **Execution**:
  - User query: “Build a secure auth endpoint with frontend integration, tests, and docs for a web app.”
  - Primary (Claude Code) delegates to sub-agents (e.g., Python Backend Coder, TS Frontend Linter) via meta-agent and PRP-enhanced prompts.
  - Outputs pass through enhancer, validator (external agent with DeepConf/Graphiti/REF/PRPs), and feedback loops.
  - Proactive triggers auto-invoke agents (e.g., Security Auditor).
  - UI displays agent statuses, prompts, verdicts, Graphiti entities, REF results, SCWT metrics.
  - Deliverables: Endpoint code, frontend component, tests, README update, PR description.
- **Metrics** (Automated via scripts, manual audits):
  - **Hallucination Rate**: % uncited claims/errors (≤10%).
  - **Knowledge Reuse**: % context pack from memory/Graphiti/REF/PRPs (≥30%).
  - **Task Efficiency**: End-to-end time (≥30% reduction); token usage (≥70% savings).
  - **Communication Efficiency**: Primary-sub iterations (≥20% reduction).
  - **Precision**: Cited sources relevant to task (≥85%).
  - **Verdict Accuracy**: Validator verdicts vs. human review (≥90%).
  - **UI Usability**: CLI usage reduction (≥10% task time savings).
- **Tools**: Python scripts for execution, hallucination scorer, token counter, PR diff analyzer. Manual audits for 5-10 runs.
- **Baseline**: Run SCWT on vanilla Archon (e.g., ~30% hallucination rate, no reuse, higher iterations).

**SCWT Evolution**:
- **Phase 1**: Sub-agents (20 roles, HRM, PRPs); efficiency/iterations/precision.
- **Phase 2**: Meta-agent; dynamic scaling.
- **Phase 3**: Validator/enhancers/REF; hallucination/verdicts.
- **Phase 4**: Memory/retrieval/Graphiti; reuse/precision.
- **Phase 5**: DeepConf; savings/accuracy.
- **Phase 6**: UI polish; holistic validation.

## 8. Phased Roadmap
Development is gated: Implement → Test → Run SCWT → Improve → Proceed. Baselines from vanilla Archon via SCWT.

**Phase 1: Fork & Specialized Global Sub-Agent System Enhancements (2-4 weeks)**
- **Implement**: Clone/fork Archon; enhance global sub-agents (20 roles—e.g., Python Backend Coder, TS Frontend Linter—scalable, focused memory scopes via JSON configs); parallel execution; conflict resolution; basic orchestration; minimal memory/retrieval (SQLite, Markdown cards); HRM option; PRP-based prompts with examples (`examples/`); proactive triggers (e.g., Security Auditor on code changes); UI agent dashboard (view of 20+ roles, statuses, PIDs).
- **Test**: Unit/integration for sub-agent execution, role configs, feedback loops, HRM, PRPs, triggers, UI.
- **SCWT Benchmark**: Run SCWT; measure task efficiency (≥15% time reduction), communication efficiency (≥10% fewer iterations), precision (≥85%), hallucination rate (~30%), UI usability (≥5% CLI reduction). Compare to vanilla Archon.
- **Improve**: Tune roles/memory/PRPs/triggers/UI if iterations > baseline or precision <85%.
- **Gate**: Proceed if SCWT shows ≥10% efficiency gain, ≥85% precision, no major bugs.

**Phase 2: Meta-Agent Integration (3-5 weeks)**
- **Implement**: Meta-agent (dynamic spawning/management of sub-agents—unbounded); orchestration integration; UI meta-agent controls (spawn/adjust agents).
- **Test**: Dynamic creation; SCWT flows across projects; UI controls.
- **SCWT Benchmark**: Task efficiency (≥20% reduction), communication efficiency (≥15% fewer iterations), knowledge reuse (≥20%), precision (≥85%), UI usability (≥7% CLI reduction). Vs. Phase 1.
- **Improve**: Refine meta-knowledge/UI if reuse <20%.
- **Gate**: Proceed if SCWT shows ≥15% scaling improvements.

**Phase 3: Validator and Prompt Enhancers with REF Integration (4-6 weeks)**
- **Implement**: Validator as external agent (Archon's LLM options—e.g., DeepSeek default—with configurable temp 0-0.2 for no lying); deterministic checks (pytest, ruff, mypy, semgrep, eslint); policy YAML; JSON verdicts; prompt enhancer (bidirectional, PRP-based); REF Tools MCP (`ref_search_documentation`, `ref_read_url`); UI validation summary (verdicts, DeepConf, REF/PRP provenance), prompt viewer (original vs. enhanced).
- **Test**: SCWT with hallucination injection; loop handling with enhanced prompts/REF; UI displays.
- **SCWT Benchmark**: Hallucination reduction (≥50%), verdict accuracy (≥90%), communication efficiency (≥20% fewer reprompts), precision (≥85%), REF impact (≥10% hallucination drop), UI usability (≥10% CLI reduction). Vs. prior phases.
- **Improve**: Adjust temp/thresholds/enhancer/REF/UI if over-blocking or verdicts <90%.
- **Gate**: Proceed if SCWT shows ≥40% error drop, ≤10% false positives.

**Phase 4: Memory/Retrieval Foundation with Graphiti (3-5 weeks)**
- **Implement**: Memory service (global/project/job/runtime, role-specific); adaptive retriever (hybrid, bandit weights); Graphiti prototype (Kuzu database, entity/relationship ingestion); integrate with meta-agent/validator; UI Graphiti explorer (graph viewer, temporal filters).
- **Test**: Unit/integration for memory storage, retrieval, Graphiti; UI interaction.
- **SCWT Benchmark**: Knowledge reuse (≥30%), retrieval precision (≥85%), Graphiti impact (≥10% precision boost), UI usability (≥10% CLI reduction). Vs. prior phases.
- **Improve**: Tune embeddings/graph queries/UI if precision <85% or reuse <30%.
- **Gate**: Proceed if SCWT shows ≥20% reuse/precision gain.

**Phase 5: DeepConf Integration & Optimization (3-5 weeks)**
- **Implement**: DeepConf wrapper (online/offline); tie to validator/agents/enhancer/memory distillation; optimize enhancer with DeepConf; Graphiti scalability (Neo4j); UI HRM visualizer (optional), REF search interface.
- **Test**: SCWT with reasoning paths; confidence in loops with enhanced prompts/graph queries; UI displays.
- **SCWT Benchmark**: Token/compute savings (70-85%), accuracy gains (2-5%), efficiency (≥30% time reduction), precision (≥85%), UI usability (≥10% CLI reduction). Vs. prior phases.
- **Improve**: Fine-tune modes/enhancer/graph/UI if savings <70%.
- **Gate**: Proceed if SCWT meets all prior benchmarks + ≥70% efficiency.

**Phase 6: Final Polish & Deployment (2-4 weeks)**
- **Implement**: Full CLI/MCP suite; local adapter; UI polish (SCWT metrics dashboard, task progress/debug tools).
- **Test**: SCWT in multi-user, multi-project scenarios; edge cases; UI usability.
- **SCWT Benchmark**: System-wide metrics (≥50% hallucination reduction, ≥30% reuse, setup ≤10 min, ≥20% iteration reduction, ≥85% precision, ≥10% UI usability gain).
- **Improve**: Based on holistic review.
- **Release**: Open-source forked repo.

## 9. Success Metrics (Via SCWT)
- Hallucination rate: ≤10% (audits/PR reviews).
- Knowledge reuse: ≥30% in packs from memory/Graphiti/REF/PRPs.
- Efficiency: 70-85% token/compute savings; ≥30% task time reduction; ≥20% fewer iterations.
- Precision: Cited sources ≥85%; Graphiti/REF/PRPs boost ≥10%.
- Verdict accuracy: ≥90% human agreement.
- Setup time: ≤10 minutes; UI usability: ≥10% CLI reduction.
- User satisfaction (surveys).

## 10. Risks & Mitigations
- **Complexity Creep**: Mitigation: Phase gates; local-first SQLite/Kuzu; Graphiti/REF/HRM/PRPs optional; UI optional; cap agents at 30/task if needed.
- **SCWT Failures**: Mitigation: Iterative improvements; fallback to baselines; refine test scripts.
- **Graphiti/DeepConf/Prompt Latency**: Mitigation: Offline mode; parallelize; toggleable enhancer/UI.
- **Hallucination Persistence**: Mitigation: Graphiti/REF/PRP provenance; focused memory; human escalation.
- **Scalability Issues**: Mitigation: Pluggable backends; test multi-project scenarios early.

## 11. Deliverables
- Forked Archon repo with phased commits/branches.
- CLI/MCP tools and schemas (e.g., verdict JSON, enhance-prompt, meta_spawn, Graphiti/REF/PRP APIs).
- SCWT: Python scripts/repo/docs for setup, execution, metrics (hallucination scorer, token counter, PR analyzer).
- Docs: Setup guide, architecture diagrams, workflow examples, SCWT reports.
- UI: Agent dashboard, prompt viewer, validation summary, Graphiti explorer, REF search, HRM visualizer (optional), SCWT metrics dashboard, task progress/debug.
- Samples: Context packs, PRP-enhanced prompts, validation outputs, Graphiti entities, REF results, DeepConf traces, HRM examples.

