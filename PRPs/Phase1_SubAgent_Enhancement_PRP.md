# Product Requirements Prompt (PRP): Phase 1 - Specialized Global Sub-Agent System Enhancement

## 📋 **Metadata**
- **PRP ID**: ARCH-P1-001
- **Phase**: 1 of 6
- **Project**: Archon+ Enhanced AI Coding System
- **Task ID**: 1db9dc81-be13-4374-bc7b-b86c16ca2885
- **Priority**: High
- **Estimated Duration**: 2-4 weeks
- **Dependencies**: Fork Archon repository to VeloF2025/Archon

## 🎯 **Objective**
Fork Archon and implement a specialized global sub-agent system with 20+ role-based agents (Python Backend Coder, TS Frontend Linter, Unit Test Generator, Security Auditor, Doc Writer, API Integrator, HRM Reasoning Agent, etc.) with focused memory scopes, parallel execution, conflict resolution, PRP-based prompts, proactive triggers, and UI agent dashboard.

## 🔄 **Context & Background**
Base Archon provides foundational MCP server architecture and basic agent orchestration. Phase 1 enhances this with:
- **Dynamic/unbounded specialized agents** (20 roles, scalable to more)
- **Focused memory scopes** via JSON configurations
- **Parallel execution** with conflict resolution (Redis/Git worktrees)
- **PRP-based structured prompts** with examples and file paths
- **Proactive triggers** for automatic agent invocation
- **UI agent dashboard** for real-time visibility

## 🏗️ **Implementation Requirements**

### **Sub-Agent System Architecture**
```
Global Sub-Agent Pool (20+ Roles):
├── Python Backend Coder
├── TypeScript Frontend Linter  
├── Unit Test Generator
├── Security Auditor
├── Documentation Writer
├── API Integrator
├── HRM Reasoning Agent (Optional)
├── Database Designer
├── Performance Optimizer
├── Code Reviewer
├── DevOps Engineer
├── UI/UX Designer
├── Error Handler
├── Refactoring Specialist
├── Configuration Manager
├── Deployment Coordinator
├── Monitoring Agent
├── Quality Assurance
├── Technical Writer
└── Integration Tester
```

### **Core Components to Implement**

#### **1. Agent Configuration System**
- **Path**: `python/src/agents/configs/`
- **Format**: JSON role definitions with memory scopes
- **Example**:
```json
{
  "role": "python_backend_coder",
  "name": "Python Backend Coder",
  "description": "Specialized in Python/FastAPI backend development",
  "memory_scope": ["backend", "api", "database", "authentication"],
  "skills": ["python", "fastapi", "sqlalchemy", "pytest", "async"],
  "proactive_triggers": ["*.py", "requirements.txt", "alembic/*"],
  "prp_template": "python_backend_prp.md"
}
```

#### **2. Parallel Execution Engine**
- **Path**: `python/src/agents/orchestration/`
- **Features**: Agent pool management, task queue, conflict resolution
- **Conflict Resolution**: Redis locks or Git worktrees for file-level conflicts

#### **3. PRP-Based Prompt System**
- **Path**: `python/src/agents/prompts/prp/`
- **Structure**: Role-specific PRP templates with examples
- **Example Template**:
```markdown
# Python Backend Development PRP

## Context
- Project: {project_name}
- Files: {file_paths}
- Dependencies: {dependencies}

## Requirements
{requirements}

## Examples
{examples}

## Standards
{coding_standards}

## Output Format
- Implementation files
- Tests (≥75% coverage)  
- Documentation
```

#### **4. Proactive Trigger System**
- **Path**: `python/src/agents/triggers/`
- **Implementation**: File watcher + pattern matching
- **Triggers**:
  - Code changes → Security Auditor
  - New dependencies → Security Auditor
  - Test files → Unit Test Generator
  - Documentation changes → Technical Writer

#### **5. UI Agent Dashboard**
- **Path**: `archon-ui-main/src/components/agents/`
- **Features**:
  - Real-time agent status (Active, Idle, Processing)
  - Agent roles and current tasks
  - Process IDs and resource usage
  - Agent spawn/kill controls
  - Memory scope visualization

### **Memory & Retrieval Enhancement**
- **SQLite-based** job memory (JSON format)
- **Role-specific memory scopes** (prevent cross-contamination)
- **Markdown cards** for global/project knowledge
- **Auto-promotion** of recurring patterns (≥3 occurrences)

## 🧪 **SCWT Benchmark Integration**

### **SCWT Test Setup**
Create mock repository structure for testing:
```
scwt-test-repo/
├── backend/
│   └── auth_endpoint.py (stub)
├── frontend/  
│   └── Login.tsx (stub)
├── tests/
│   └── test_auth.py (stub)
├── docs/
│   └── standards.md
├── knowledge/project/
│   └── api_design.md
└── adr/
    └── ADR-2025-08-01-oauth.md
```

### **Benchmark Metrics (Phase 1 Targets)**
- **Task Efficiency**: ≥15% time reduction vs baseline
- **Communication Efficiency**: ≥10% fewer Claude Code ↔ Sub-agent iterations
- **Precision**: ≥85% relevant context in agent responses
- **Hallucination Rate**: ~30% (baseline measurement for future phases)
- **UI Usability**: ≥5% reduction in CLI usage time

### **Test Execution**
- **Input**: "Build a secure auth endpoint with frontend integration, tests, and docs for a web app"
- **Agents Invoked**: Python Backend Coder, TS Frontend Linter, Unit Test Generator, Security Auditor, Doc Writer
- **Expected Output**: Working auth endpoint + tests + docs + UI integration
- **Measurement**: Automated scripts + manual audit

## 📊 **Success Criteria & Gates**

### **Phase 1 Gate Requirements**
✅ **Must Pass All:**
- [ ] 20+ specialized agents configured and functional
- [ ] Parallel execution without deadlocks
- [ ] PRP-based prompts generating relevant context
- [ ] Proactive triggers correctly invoking agents
- [ ] UI dashboard showing real-time agent status
- [ ] SCWT benchmark showing ≥10% efficiency gain
- [ ] SCWT precision ≥85%
- [ ] No major bugs or crashes during testing

### **Quality Assurance**
- **Unit Tests**: Agent configuration, execution engine, triggers
- **Integration Tests**: Multi-agent workflows, UI interactions
- **SCWT Validation**: Full end-to-end testing with metrics collection
- **Manual Review**: Code quality, documentation, UI usability

## 🚀 **Deliverables**

1. **Forked Repository**: `github.com/VeloF2025/Archon`
2. **Agent System**: 20+ role configurations with JSON schemas
3. **Execution Engine**: Parallel orchestration with conflict resolution
4. **PRP Templates**: Role-specific prompt templates with examples
5. **Trigger System**: File-based proactive agent invocation
6. **UI Dashboard**: Real-time agent monitoring and controls
7. **SCWT Framework**: Benchmark testing scripts and metrics
8. **Documentation**: Setup guide, architecture diagrams, examples

## 🔗 **References & Dependencies**

### **External Dependencies**
- Base Archon repository (coleam00/Archon)
- Claude Code MCP integration
- FastAPI backend framework
- React/TypeScript frontend
- SQLite for local memory storage

### **Integration Points**
- Archon's existing MCP server
- Task management system
- Knowledge base integration
- Socket.IO for real-time updates

## 📝 **Implementation Notes**

### **Development Approach**
1. **Fork Repository**: Clone coleam00/Archon → VeloF2025/Archon
2. **Agent Configs**: Define 20+ role JSON configurations
3. **Execution Engine**: Implement parallel agent orchestration
4. **PRP System**: Create role-specific prompt templates
5. **Proactive Triggers**: File watcher + pattern matching
6. **UI Dashboard**: Real-time agent status monitoring
7. **SCWT Setup**: Benchmark framework and test repository
8. **Testing**: Unit, integration, and SCWT validation

### **Risk Mitigation**
- **Agent Conflicts**: Redis locks + Git worktrees for file-level isolation
- **Memory Bloat**: Role-specific scopes prevent cross-contamination
- **UI Performance**: Optimize real-time updates with Socket.IO throttling
- **SCWT Reliability**: Multiple test runs + manual audit validation

---

**Next Phase**: Meta-Agent Integration (Phase 2) - Dynamic agent spawning and management system