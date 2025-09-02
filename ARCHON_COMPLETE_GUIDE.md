# 🤖 ARCHON - Complete Development Guide & Manifest

## 📋 Table of Contents
- [Archon Manifest](#archon-manifest)
- [System Overview](#system-overview)  
- [Global Rules & Protocols](#global-rules--protocols)
- [Development Standards](#development-standards)
- [AI Agent System](#ai-agent-system)
- [Security & Quality Gates](#security--quality-gates)
- [Installation & Setup](#installation--setup)
- [MCP Integration](#mcp-integration)
- [Architecture Details](#architecture-details)

---

## 🎯 ARCHON MANIFEST

**Mission**: Transform software development with AI-powered assistance, comprehensive knowledge management, and zero-tolerance quality enforcement.

### Core Principles
1. **AI-First Development** - 21+ specialized AI agents for every development task
2. **Zero-Tolerance Quality** - No TypeScript errors, no console.log, >95% test coverage
3. **Knowledge-Driven** - RAG-powered search across all project documentation and code
4. **Real-Time Collaboration** - Socket.IO updates, live project management
5. **Security by Default** - Built-in security auditing and vulnerability prevention
6. **Performance Obsessed** - <1.5s page loads, <200ms API responses
7. **Truth-First Development** - NLNH (No Lies, No Hallucination) protocol

### Value Proposition
- **80% faster development** through AI agent automation
- **99% fewer bugs** through zero-tolerance quality gates
- **100% test coverage** through TDD enforcement
- **Real-time knowledge** through advanced RAG system
- **Production-ready code** through specialized agent validation

---

## 🏗️ SYSTEM OVERVIEW

### Archon Modes

#### 🪶 **Archon Light Mode** (Individual Developers)
- **Setup**: 3 minutes (Node.js + API key)
- **Agents**: 5 core AI agents
- **Storage**: File-based (portable)
- **Best For**: Evaluation, learning, individual projects

#### 🏗️ **Archon Full Mode** (Teams & Production)
- **Setup**: 15 minutes (Docker + Supabase)
- **Agents**: 21+ specialized AI agents
- **Storage**: PostgreSQL + pgvector
- **Best For**: Team collaboration, production systems

### Core Architecture
```
┌─ Frontend (React + TypeScript) ────────── Port 3737
├─ Main Server (FastAPI + Socket.IO) ────── Port 8181  
├─ MCP Server (Model Context Protocol) ──── Port 8051
├─ Agents Service (PydanticAI) ──────────── Port 8052
└─ Database (Supabase PostgreSQL + pgvector)
```

---

## 🛡️ GLOBAL RULES & PROTOCOLS

### 🚨 UNIVERSAL ACTIVATION TRIGGERS

#### **RYR Command - Rules Enforcement**
**Triggers**: "Remember Your Rules", "RYR", "ryr"
**Action**: Load all rules, display status, activate NLNH protocol

#### **ForgeFlow (FF) - AI Orchestration**
**Triggers**: "ForgeFlow", "FF", "@FF"
**Action**: Launch multi-agent parallel development system

#### **Archon Global - Project Management**
**Triggers**: "Archon", "@Archon", "archon"
**Action**: Activate project management, knowledge base, specialized agents

### 🚫 CRITICAL SAFETY PROTOCOLS

#### **NLNH Protocol (No Lies, No Hallucination)**
- Absolute truthfulness - zero tolerance for AI hallucinations
- Say "I don't know" when uncertain
- Report real errors, admit all failures
- Show actual error messages, not fabricated ones

#### **DGTS Protocol (Don't Game The System)**
- Prevent fake tests that always pass
- Block commented validation rules
- Detect mock implementations instead of real features
- Real-time agent behavior monitoring

#### **AntiHall Validator**
- Mandatory validation before suggesting ANY code
- Verify all methods/components exist in codebase
- 100% prevention of non-existent code references

### 📋 DOCUMENTATION-DRIVEN DEVELOPMENT (MANDATORY)
**Core Rule**: Tests MUST be created from PRD/PRP/ADR documentation BEFORE any implementation

**Workflow**:
1. Parse requirements from documentation
2. Create test specifications from requirements
3. Write tests first that validate documented behavior
4. Implement minimal code to pass tests
5. Validate all requirements have corresponding tests

---

## 🔧 DEVELOPMENT STANDARDS

### **Zero-Tolerance Quality Gates (MANDATORY)**

#### **Automatic Blocking Patterns**
- ❌ **Console.log statements** - Use proper logging service
- ❌ **TypeScript/Build errors** - Zero compilation errors
- ❌ **ESLint violations** - Zero errors, zero warnings
- ❌ **Undefined error references** - All catch blocks need error parameters
- ❌ **Bundle size violations** - Max 500kB per chunk
- ❌ **Void error anti-patterns** - No error silencing

#### **Enhanced Validation Command**
```bash
# MANDATORY before any code changes:
node scripts/zero-tolerance-check.js
```

#### **Quality Standards**
- **Test Coverage**: >95% (pytest/vitest)
- **File Size**: Max 300 lines per file
- **Type Coverage**: 100% TypeScript (no 'any' types)
- **Performance**: Page load <1.5s, API <200ms
- **Security**: Input validation, proper error handling

### **Code Quality Requirements**
```javascript
// ✅ REQUIRED - Proper error handling:
try {
  const result = await apiCall();
  return result;
} catch (error: unknown) {
  log.error('API call failed', { error }, 'ComponentName');
  throw new Error(`Failed to fetch data: ${error}`);
}

// ✅ REQUIRED - Proper logging:
import { log } from '@/lib/logger';
log.info('User action', { userId, action }, 'ComponentName');

// ❌ BLOCKED - Console statements:
console.log('debug info'); // This will block commits

// ❌ BLOCKED - Undefined errors:
catch (error) {  // Missing type, causes undefined references
  // Handle error
}
```

---

## 🤖 AI AGENT SYSTEM

### **21+ Specialized Agents Available**

#### **🏗️ Development Specialists**
- **system-architect** - Architecture design and planning
- **code-implementer** - Zero-error code implementation  
- **test-coverage-validator** - Test creation and >95% coverage
- **code-quality-reviewer** - Code review and validation
- **security-auditor** - Vulnerability scanning and fixes
- **performance-optimizer** - Performance analysis and tuning
- **ui-ux-optimizer** - Interface design and accessibility

#### **📊 Analysis & Planning**  
- **strategic-planner** - Task breakdown and project planning
- **database-architect** - Data modeling and optimization
- **api-design-architect** - RESTful API design
- **antihallucination-validator** - Code existence verification

#### **🛠️ Operations Specialists**
- **deployment-automation** - CI/CD and deployment
- **code-refactoring-optimizer** - Code improvement
- **documentation-generator** - Documentation creation
- **devops-automation** - Infrastructure management

### **Agent Activation via Task Tool**
```javascript
// Launch specialized agents for complex tasks
@FF <task>  // Standard orchestration  
@FF! <task> // Emergency mode (bypass prompts)
@FF --pattern=<type> // Use specific execution pattern
@FF --agents=<list> // Custom agent pipeline
```

### **Agent Quality Enforcement**
Every agent must:
1. Run validation before development
2. Follow zero-tolerance quality gates  
3. Create documentation-driven tests
4. Pass DGTS anti-gaming checks
5. Validate code existence with AntiHall

---

## 🔒 SECURITY & QUALITY GATES

### **Pre-Commit Validation (ENHANCED)**
```bash
# Primary validation (replaces basic checks)
node scripts/zero-tolerance-check.js

# Fallback validation
python "C:\Jarvis\UNIVERSAL_RULES_CHECKER.py" --path "." --pre-commit
```

### **Security Auditing**
- **Automatic vulnerability scanning** on code changes
- **Input validation enforcement** for all user inputs  
- **Authentication/authorization validation** for protected routes
- **Dependency vulnerability checking** for npm/pip packages
- **Secret detection** to prevent credential commits

### **Performance Gates**
- **Bundle size limits** - Max 500kB per chunk
- **API response times** - <200ms average
- **Page load targets** - <1.5s initial load
- **Memory usage limits** - Monitored and enforced
- **Database query optimization** - Automatic slow query detection

---

## 🚀 INSTALLATION & SETUP

### **Archon Light Mode (3 minutes)**
```bash
# 1. Clone repository
git clone https://github.com/VeloF2025/Archon.git
cd Archon

# 2. Configure environment
cp .env.light .env
# Edit .env with your API key

# 3. Start Light Mode
npm run light:start

# Access at: http://localhost:3000
```

### **Archon Full Mode (15 minutes)**
```bash
# 1. Clone repository  
git clone https://github.com/VeloF2025/Archon.git
cd Archon

# 2. Setup Supabase
# Create account at supabase.com
# Create new project, copy credentials

# 3. Configure environment
cp .env.example .env
# Add Supabase URL and service key

# 4. Start all services
docker-compose up -d --build

# 5. Access UI
# Main UI: http://localhost:3737
# API: http://localhost:8181  
# MCP: http://localhost:8051
```

### **Requirements**
- **Light Mode**: Node.js 18+, API key (OpenAI/Claude/Gemini)
- **Full Mode**: Docker, Docker Compose, Supabase account

---

## 🔌 MCP INTEGRATION

### **Model Context Protocol Support**
Archon integrates with AI coding assistants (Cursor, Windsurf, etc.) via MCP:

### **Available MCP Tools**
```javascript
// Core tools (both modes)
archon:chat                 // Chat with AI agents
archon:search_knowledge     // Search knowledge base  
archon:add_knowledge        // Add documents to knowledge

// Full mode additional tools
archon:manage_project       // Project operations
archon:manage_task          // Task management  
archon:execute_agent        // Deploy specialized agents
archon:get_system_metrics   // System health monitoring
archon:export_data          // Data export capabilities
```

### **MCP Configuration**
Add to your AI assistant's MCP settings:
```json
{
  "archon": {
    "command": "node",
    "args": ["path/to/archon/mcp-server.js"],
    "env": {
      "ARCHON_URL": "http://localhost:8181"
    }
  }
}
```

---

## 🏛️ ARCHITECTURE DETAILS

### **Microservices Architecture**
```
Frontend Service (React + TypeScript + Vite)
├── Real-time UI updates via Socket.IO
├── Component library with TypeScript
├── State management with React Context
└── TailwindCSS for styling

Main Server (FastAPI + Python)
├── RESTful API endpoints
├── Socket.IO for real-time updates  
├── Background task management
├── Authentication and authorization
└── Database operations

MCP Server (HTTP-based)
├── Model Context Protocol implementation
├── Tool execution and validation
├── AI assistant integration
└── Lightweight HTTP server

Agents Service (PydanticAI)
├── 21+ specialized AI agents
├── Parallel execution engine
├── Task routing and orchestration
├── Quality gate enforcement
└── Agent behavior monitoring

Database (Supabase PostgreSQL)
├── Vector embeddings (pgvector)
├── Full-text search capabilities
├── Real-time subscriptions
├── Row-level security
└── Automatic backups
```

### **Data Flow**
```
User Request → Frontend → Main Server → Agents Service
                    ↓           ↓              ↓
              Socket.IO ← Database ← MCP Server
```

### **Technology Stack**
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS
- **Backend**: FastAPI, Python 3.12, Socket.IO
- **Database**: PostgreSQL 15, pgvector, Supabase
- **AI**: PydanticAI, OpenAI, Anthropic, Google Gemini
- **Infrastructure**: Docker, Docker Compose
- **Testing**: Vitest (frontend), pytest (backend), Playwright (E2E)

---

## 📊 METRICS & MONITORING

### **Development Metrics**
- **Code Quality Score**: Based on ESLint, TypeScript, test coverage
- **Agent Performance**: Task completion rates, error frequencies
- **Knowledge Base Health**: Document freshness, search relevance
- **System Performance**: Response times, memory usage, uptime

### **Quality Dashboards**
- **Real-time code quality** monitoring
- **Agent activity tracking** and performance metrics  
- **Test coverage reports** with trend analysis
- **Security vulnerability** scanning and resolution tracking

---

## 🎯 USAGE SCENARIOS

### **Individual Developers**
- **Code assistance** via specialized AI agents
- **Knowledge management** for personal projects
- **Quality enforcement** through automated gates
- **Performance optimization** and security auditing

### **Development Teams**
- **Real-time collaboration** on shared projects
- **Knowledge sharing** across team members
- **Standardized quality** gates and coding standards
- **Project management** with task tracking

### **Enterprise Organizations**  
- **Scalable infrastructure** with microservices
- **Advanced security** and compliance features
- **Performance monitoring** and optimization
- **Custom agent development** for specific workflows

---

## 🚀 GETTING STARTED CHECKLIST

### **For Individual Developers**
- [ ] Install Node.js 18+
- [ ] Get API key (OpenAI/Claude/Gemini)
- [ ] Clone Archon repository
- [ ] Start with Light Mode (`npm run light:start`)
- [ ] Try chat with different AI agents
- [ ] Upgrade to Full Mode when ready

### **For Development Teams**
- [ ] Setup Docker and Docker Compose
- [ ] Create Supabase account and project
- [ ] Configure team environment variables
- [ ] Deploy Full Mode (`docker-compose up -d`)
- [ ] Configure MCP for team AI assistants
- [ ] Setup project management and task tracking

### **For Organizations**
- [ ] Review security and compliance requirements
- [ ] Plan infrastructure deployment (cloud/on-premise)
- [ ] Configure enterprise authentication
- [ ] Setup monitoring and logging systems
- [ ] Train development teams on Archon workflows
- [ ] Customize agents for organizational needs

---

## 📞 SUPPORT & COMMUNITY

- **GitHub Repository**: https://github.com/VeloF2025/Archon
- **Documentation**: See `/docs` folder for detailed guides
- **Issues & Bugs**: GitHub Issues
- **Feature Requests**: GitHub Discussions
- **Community**: Discord/Slack (links in repository)

---

**Version**: Archon v2.0 Alpha
**Last Updated**: January 2025
**License**: See LICENSE file in repository

*This guide represents the complete development framework for Archon - the AI-powered development platform that transforms how software is built.*