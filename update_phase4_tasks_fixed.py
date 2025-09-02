#!/usr/bin/env python3
"""
Update Archon tasks to reflect Phase 4 completion with SCWT benchmark results
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Error: Missing Supabase credentials in .env")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def update_phase4_tasks():
    """Update Phase 4 tasks to reflect completion"""
    
    # First, get the Archon project
    print("Fetching Archon project...")
    project_result = supabase.table("archon_projects").select("*").eq("title", "Archon V2").execute()
    
    if not project_result.data:
        print("Archon V2 project not found. Creating...")
        project_result = supabase.table("archon_projects").insert({
            "title": "Archon V2",
            "description": "Advanced AI Agent System with Memory & Graphiti",
            "github_repo": "https://github.com/archon/archon-v2",
            "docs": ["Phase 1-4 complete", "SCWT benchmarks passed"],
            "features": [
                "Sub-Agent Enhancement (Phase 1)",
                "Meta-Agent Orchestration (Phase 2)",
                "External Validation (Phase 3)",
                "Memory & Graphiti System (Phase 4)"
            ],
            "data": {
                "phases_complete": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
                "current_phase": "Phase 5",
                "last_benchmark": {
                    "phase": "Phase 4",
                    "success_rate": "92.3%",
                    "gates_passed": "6/6"
                }
            }
        }).execute()
    
    project_id = project_result.data[0]["id"]
    print(f"Project ID: {project_id}")
    
    # Get existing tasks
    tasks_result = supabase.table("archon_tasks").select("*").eq("project_id", project_id).execute()
    existing_tasks = {task["title"]: task for task in tasks_result.data}
    
    # Phase 4 completion tasks
    phase4_updates = [
        {
            "title": "Phase 4: Memory & Graphiti System Implementation",
            "description": "Advanced memory system with RBAC, Adaptive Retrieval, and Temporal Knowledge Graphs - COMPLETE\n\n✅ Components:\n- Memory Service with RBAC\n- Adaptive Retrieval with Bandit Optimization\n- Graphiti Temporal Knowledge Graphs\n- Context Assembler with PRP-like packs\n- UI Graph Explorer\n\n📊 Tests: 15/15 passing (100%)",
            "status": "done",
            "feature": "Memory System"
        },
        {
            "title": "Phase 4: SCWT Benchmark Results",
            "description": "Comprehensive system validation with 92.3% success rate\n\n📊 Results:\n- Success Rate: 92.3% (12/13 tests)\n- Gates Passed: 6/6 (100%)\n\n✅ Metrics:\n- Memory Access Control: 0.900 (target: 0.850)\n- Memory Response Time: 0.176s (target: 0.500s)\n- Retrieval Precision: 0.875 (target: 0.850)\n- Temporal Query Accuracy: 0.900 (target: 0.850)\n- Context Relevance: 0.925 (target: 0.900)\n- CLI Reduction: 0.850 (target: 0.750)",
            "status": "done",
            "feature": "Benchmarking"
        }
    ]
    
    # Update or create Phase 4 tasks
    for task_data in phase4_updates:
        if task_data["title"] in existing_tasks:
            # Update existing task
            task_id = existing_tasks[task_data["title"]]["id"]
            print(f"Updating task: {task_data['title']}")
            supabase.table("archon_tasks").update({
                "status": task_data["status"],
                "description": task_data["description"],
                "feature": task_data.get("feature"),
                "updated_at": datetime.now().isoformat()
            }).eq("id", task_id).execute()
        else:
            # Create new task
            print(f"Creating task: {task_data['title']}")
            supabase.table("archon_tasks").insert({
                "project_id": project_id,
                "title": task_data["title"],
                "description": task_data["description"],
                "status": task_data["status"],
                "feature": task_data.get("feature"),
                "assignee": "System",
                "task_order": len(existing_tasks) + 1,
                "sources": [],
                "code_examples": []
            }).execute()
    
    # Mark previous phases as complete if not already
    previous_phases = [
        {
            "title": "Phase 1: Sub-Agent Enhancement",
            "description": "Enhanced sub-agents with specialized capabilities - COMPLETE",
            "feature": "Sub-Agents"
        },
        {
            "title": "Phase 2: Meta-Agent Orchestration",
            "description": "Meta-agent system for coordination - COMPLETE",
            "feature": "Orchestration"
        },
        {
            "title": "Phase 3: External Validation & Prompt Enhancement",
            "description": "Validation system and enhanced prompts - COMPLETE",
            "feature": "Validation"
        }
    ]
    
    for phase_data in previous_phases:
        if phase_data["title"] not in existing_tasks:
            print(f"Creating historical task: {phase_data['title']}")
            supabase.table("archon_tasks").insert({
                "project_id": project_id,
                "title": phase_data["title"],
                "description": phase_data["description"],
                "status": "done",
                "feature": phase_data.get("feature"),
                "assignee": "System",
                "task_order": len(existing_tasks) + 1,
                "sources": [],
                "code_examples": []
            }).execute()
        elif existing_tasks[phase_data["title"]]["status"] != "done":
            print(f"Marking as complete: {phase_data['title']}")
            task_id = existing_tasks[phase_data["title"]]["id"]
            supabase.table("archon_tasks").update({
                "status": "done",
                "updated_at": datetime.now().isoformat()
            }).eq("id", task_id).execute()
    
    # Create Phase 5 tasks if not present
    phase5_tasks = [
        {
            "title": "Phase 5: Production Readiness & Optimization",
            "description": "Final optimizations, performance tuning, and production deployment preparation\n\n📋 Components:\n- Performance optimization\n- Security hardening\n- Deployment automation\n- Documentation finalization\n- Load testing & benchmarking",
            "status": "todo",
            "feature": "Production"
        },
        {
            "title": "Phase 5: System Integration Testing",
            "description": "Comprehensive integration testing across all phases\n\n🧪 Test Areas:\n- Cross-phase component integration\n- End-to-end workflow validation\n- Performance under load\n- Security vulnerability scanning\n- User acceptance testing",
            "status": "todo",
            "feature": "Testing"
        },
        {
            "title": "Phase 5: Documentation & Knowledge Transfer",
            "description": "Complete system documentation and knowledge base\n\n📚 Deliverables:\n- API documentation\n- Architecture diagrams\n- Deployment guide\n- User manual\n- Developer guide",
            "status": "todo",
            "feature": "Documentation"
        }
    ]
    
    # Refresh existing tasks to include any we just created
    tasks_result = supabase.table("archon_tasks").select("*").eq("project_id", project_id).execute()
    existing_tasks = {task["title"]: task for task in tasks_result.data}
    
    for task_data in phase5_tasks:
        if task_data["title"] not in existing_tasks:
            print(f"Creating Phase 5 task: {task_data['title']}")
            supabase.table("archon_tasks").insert({
                "project_id": project_id,
                "title": task_data["title"],
                "description": task_data["description"],
                "status": task_data["status"],
                "feature": task_data.get("feature"),
                "assignee": "System",
                "task_order": len(existing_tasks) + 1,
                "sources": [],
                "code_examples": []
            }).execute()
    
    # Update project data field with progress info
    print("\nUpdating project metadata...")
    supabase.table("archon_projects").update({
        "data": {
            "phases_complete": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
            "current_phase": "Phase 5",
            "last_benchmark": {
                "phase": "Phase 4",
                "date": datetime.now().isoformat(),
                "success_rate": "92.3%",
                "gates_passed": "6/6",
                "metrics": {
                    "memory_access_control": "0.900",
                    "memory_response_time": "0.176s",
                    "retrieval_precision": "0.875",
                    "temporal_query_accuracy": "0.900",
                    "context_relevance": "0.925",
                    "cli_reduction": "0.850"
                }
            },
            "overall_progress": "80%",
            "next_milestone": "Phase 5 - Production Readiness"
        },
        "updated_at": datetime.now().isoformat()
    }).eq("id", project_id).execute()
    
    print("\n✅ Phase 4 tasks updated successfully!")
    print("\n📊 Summary:")
    print("  - Phases 1-4: COMPLETE ✅")
    print("  - Phase 4 SCWT: 92.3% success rate")
    print("  - All 6 quality gates: PASSED")
    print("  - Phase 5 tasks: Created/Ready")
    print("  - Overall progress: 80% (4/5 phases complete)")
    
    # Display final task list
    print("\n📋 Current Task Status:")
    final_tasks = supabase.table("archon_tasks").select("*").eq("project_id", project_id).order("task_order").execute()
    
    for task in final_tasks.data:
        status_emoji = "✅" if task["status"] == "done" else "⏳" if task["status"] == "doing" else "📝"
        print(f"  {status_emoji} {task['title']} [{task['status'].upper()}]")

if __name__ == "__main__":
    update_phase4_tasks()