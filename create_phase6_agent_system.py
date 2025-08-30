#!/usr/bin/env python3
"""
Add Phase 6: Agent System Integration to Archon project
Move existing phases down to accommodate the new critical phase
"""

import requests
import json
from datetime import datetime

# Archon project ID
PROJECT_ID = "85bc9bf7-465e-4235-9990-969adac869e5"
BASE_URL = "http://localhost:8181/api"

def update_task(task_id, updates):
    """Update a task with given data"""
    url = f"{BASE_URL}/tasks/{task_id}"
    response = requests.put(url, json=updates)
    if response.status_code == 200:
        print(f"[OK] Updated task {task_id}")
        return True
    else:
        print(f"[FAIL] Failed to update task {task_id}: {response.text}")
        return False

def create_task(task_data):
    """Create a new task"""
    url = f"{BASE_URL}/projects/{PROJECT_ID}/tasks"
    response = requests.post(url, json=task_data)
    if response.status_code in [200, 201]:
        print(f"[OK] Created task: {task_data['title']}")
        return response.json()
    else:
        print(f"[FAIL] Failed to create task: {response.text}")
        return None

def main():
    print(f"Adding Phase 6: Agent System Integration to project {PROJECT_ID}...")
    
    # Update existing Phase 6 to Phase 7
    print("Updating Phase 6 (DeepConf) -> Phase 7...")
    update_task("437fd305-2947-4299-ba6f-009f7cc4827d", {
        "title": "Phase 7: DeepConf Integration & Final Polish",
        "description": """🤖 ASSIGNED TO: Archon Meta-Agent

📝 NOTE: Moved from Phase 6 to Phase 7 to accommodate Agent System Integration

IMPLEMENT: DeepConf wrapper (online/offline); confidence scoring system; probabilistic output validation; multi-model consensus mechanisms; uncertainty quantification; tie to validator/agents/enhancer/memory distillation; optimize enhancer with DeepConf; Graphiti scalability (Neo4j); UI HRM visualizer, REF search interface.

KEY DELIVERABLES:
• DeepConf scoring engine
• Confidence thresholds configuration
• Multi-model voting system
• Uncertainty reporting in UI
• Confidence-based fallback strategies
• Full CLI/MCP suite
• Local adapter
• UI polish (SCWT metrics dashboard, task progress/debug tools)

SCWT BENCHMARK: Token/compute savings (70-85%), accuracy gains (2-5%), efficiency (≥30% time reduction), precision (≥85%), UI usability (≥10% CLI reduction). System-wide metrics (≥50% hallucination reduction, ≥30% reuse, setup ≥10 min, ≥20% iteration reduction, ≥85% precision, ≥10% UI usability gain).

GATE: Proceed if SCWT meets all prior benchmarks + ≥70% efficiency.

🏗️ This task will be executed by the Archon Meta-Agent, not the user."""
    })
    
    # Update existing Phase 7 to Phase 8
    print("Updating Phase 7 (Production) -> Phase 8...")
    update_task("663379ba-6ca9-4977-9a33-c815439aa27e", {
        "title": "Phase 8: Production Optimization & Enterprise Features",
        "description": """🤖 ASSIGNED TO: Archon Meta-Agent

📝 NOTE: Moved from Phase 7 to Phase 8 to accommodate Agent System Integration

IMPLEMENT: Production deployment configuration, monitoring/observability setup, comprehensive documentation, performance optimization, security hardening.

ENTERPRISE FEATURES:
• Multi-tenant architecture
• Role-based access control (enterprise)
• SSO integration
• Audit logging
• Performance monitoring
• Load balancing
• Auto-scaling

SCWT BENCHMARK TARGETS:
• System reliability: ≥99.9% uptime
• Response time: <200ms p95
• Error rate: <0.1%
• Security: Zero critical vulnerabilities
• Documentation: 100% API coverage
• Scalability: 1000+ concurrent users

GATE: Proceed to production if all benchmarks pass.

🏗️ This task will be executed by the Archon Meta-Agent, not the user."""
    })
    
    # Create new Phase 6: Agent System Integration
    print("Creating new Phase 6: Agent System Integration...")
    phase6_description = """🤖 ASSIGNED TO: Archon Meta-Agent

🔧 CRITICAL ARCHITECTURAL PHASE: TRUE SUB-AGENT SYSTEM

PROBLEM IDENTIFIED:
• 22 specialized agents exist only as JSON configs - not operational
• No integration with Claude Code's Task tool system
• Agents cannot operate independently with their own tools/context
• Missing Anthropic-style sub-agent architecture
• No autonomous agent spawning or workflow management

IMPLEMENTATION REQUIREMENTS:

**1. Sub-Agent Architecture (Anthropic-Style)**
• Each agent has isolated tool access and context
• Agents operate within defined scopes and guardrails
• Independent workflow execution without Claude Code intervention
• Agent-to-agent communication protocols
• Hierarchical agent management (Meta-Agent → Sub-Agents)

**2. Claude Code Integration Bridge**
• Connect 22 specialized agents to Task tool system
• Auto-spawn agents based on code changes/context
• Seamless handoff between Claude Code and sub-agents
• Preserve Claude Code as primary orchestrator
• Agent results flow back to main conversation

**3. Meta-Agent System**
• Intelligent agent selection and spawning
• Cross-agent task coordination
• Resource management and conflict resolution
• Agent performance monitoring
• Dynamic scaling based on workload

**4. Tool & Context Isolation**
• Each agent gets scoped tool access (file ops, API calls, etc.)
• Isolated working directories for agent execution
• Agent-specific knowledge bases and memory
• Guardrails prevent scope violations
• Secure inter-agent communication

**5. Autonomous Workflows**
• File watcher triggers (*.py → test_generator, security_auditor)
• Git hooks integration (pre-commit → code_reviewer, quality_assurance)
• API endpoint changes → api_integrator, documentation_writer
• Database schema changes → database_designer, migration_agent
• UI changes → ui_ux_designer, test_generator

**TECHNICAL DELIVERABLES:**
• Agent Runtime Environment (isolated execution)
• Task Distribution System (meta-agent orchestration)
• Tool Access Control Framework
• Agent Communication Protocol
• Integration with existing Claude Code workflow
• Real-time agent monitoring dashboard
• Agent performance metrics and SCWT benchmarks

**SCWT BENCHMARK TARGETS:**
• Agent autonomy: ≥80% tasks completed without Claude Code intervention
• Response time: <5s for agent selection and spawning
• Scope adherence: 100% - no agent scope violations
• Tool isolation: 100% - proper access control
• Integration success: ≥90% seamless handoffs to/from Claude Code
• Concurrent agents: Support 8+ simultaneous specialized agents
• Auto-spawn accuracy: ≥85% correct agent selection for triggers

**GATE CRITERIA:**
• All 22 agents operational and integrated
• Meta-agent successfully coordinates multi-agent workflows
• Claude Code maintains primary control while delegating specialized tasks
• Agent system operates autonomously within defined boundaries
• Full integration with existing Archon infrastructure (MCP, Memory, Graphiti)

**ARCHITECTURE PRINCIPLES:**
• Claude Code = Primary Orchestrator (project oversight, complex decisions)
• Meta-Agent = Sub-Agent Coordinator (task distribution, resource management)
• Specialized Agents = Scoped Workers (focused expertise, isolated execution)
• Seamless handoff between layers based on task complexity and scope

This phase is CRITICAL for Archon's evolution from a tool system to a true multi-agent development environment.

🎯 **Status**: Ready for implementation - highest priority
📅 **Dependencies**: Phases 1-5 complete, Agent configs and orchestrator code exist
🔧 **Complexity**: High - requires deep integration with Claude Code architecture

⏳ This task will be executed by the Archon Meta-Agent with Claude Code oversight."""
    
    create_task({
        "title": "Phase 6: Agent System Integration & Sub-Agent Architecture",
        "description": phase6_description,
        "status": "todo",
        "assignee": "User",
        "task_order": 5
    })
    
    # Update project description
    project_update = {
        "description": """Fork and enhance Archon with specialized global sub-agents (22 roles), meta-agent orchestration, external validator (DeepSeek), prompt enhancement, Graphiti knowledge graphs, REF Tools MCP, Agent System Integration, DeepConf confidence-based reasoning, and comprehensive UI dashboard.

PHASES COMPLETED:
[OK] Phase 1: Sub-agent enhancement - COMPLETE
[OK] Phase 2: Meta-agent orchestration - COMPLETE  
[OK] Phase 3: External validation & prompt enhancement - COMPLETE
[OK] Phase 4: Advanced Memory System with Graphiti - COMPLETE (92.3% SCWT)
[OK] Phase 5: External Validator Agent - COMPLETE

CURRENT: Phase 6 - Agent System Integration & Sub-Agent Architecture (CRITICAL)

UPCOMING:
Phase 7: DeepConf Integration & Final Polish
Phase 8: Production Optimization & Enterprise Features

Overall Progress: 62% (5/8 phases complete)
Benchmarked via Standard Coding Workflow Test (SCWT)."""
    }
    
    response = requests.put(f"{BASE_URL}/projects/{PROJECT_ID}", json=project_update)
    if response.status_code == 200:
        print("[OK] Updated project description with new phase structure")
    else:
        print(f"[FAIL] Failed to update project: {response.text}")
    
    print(f"\n✅ Phase structure updated successfully!")
    print(f"   Project ID: {PROJECT_ID}")
    print("   NEW PHASE 6: Agent System Integration & Sub-Agent Architecture")
    print("   Phase 6 (DeepConf) → Phase 7")
    print("   Phase 7 (Production) → Phase 8")
    print("   Overall Progress: 62% (5/8 phases)")

if __name__ == "__main__":
    main()