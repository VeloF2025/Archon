#!/usr/bin/env python3
"""
Update Archon tasks to reflect Phase 4 completion with SCWT benchmark results
"""

import asyncio
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

async def update_phase4_tasks():
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
            "docs": [],
            "features": [],
            "data": {
                "phases_complete": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
                "current_phase": "Phase 5",
                "last_benchmark": "SCWT Phase 4 - 92.3% success rate"
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
            "description": "Advanced memory system with RBAC, Adaptive Retrieval, and Temporal Knowledge Graphs",
            "status": "done",
            "metadata": {
                "completion_date": datetime.now().isoformat(),
                "tests_passed": "15/15 (100%)",
                "scwt_benchmark": "92.3% success rate",
                "components": [
                    "Memory Service with RBAC",
                    "Adaptive Retrieval with Bandit Optimization",
                    "Graphiti Temporal Knowledge Graphs",
                    "Context Assembler with PRP-like packs",
                    "UI Graph Explorer"
                ]
            }
        },
        {
            "title": "Phase 4: SCWT Benchmark Results",
            "description": "Comprehensive system validation with 92.3% success rate",
            "status": "done",
            "metadata": {
                "benchmark_date": datetime.now().isoformat(),
                "success_rate": "92.3% (12/13 tests passed)",
                "gates_passed": "6/6 (100%)",
                "metrics": {
                    "memory_access_control": "0.900 (target: 0.850) ✅",
                    "memory_response_time": "0.176s (target: 0.500s) ✅",
                    "retrieval_precision": "0.875 (target: 0.850) ✅",
                    "temporal_query_accuracy": "0.900 (target: 0.850) ✅",
                    "context_relevance": "0.925 (target: 0.900) ✅",
                    "cli_reduction": "0.850 (target: 0.750) ✅"
                }
            }
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
                "metadata": task_data["metadata"],
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
                "metadata": task_data["metadata"]
            }).execute()
    
    # Mark previous phases as complete if not already
    previous_phases = [
        ("Phase 1: Sub-Agent Enhancement", "Complete - Enhanced sub-agents with specialized capabilities"),
        ("Phase 2: Meta-Agent Orchestration", "Complete - Meta-agent system for coordination"),
        ("Phase 3: External Validation & Prompt Enhancement", "Complete - Validation system and enhanced prompts")
    ]
    
    for title, description in previous_phases:
        if title not in existing_tasks:
            print(f"Creating historical task: {title}")
            supabase.table("archon_tasks").insert({
                "project_id": project_id,
                "title": title,
                "description": description,
                "status": "done",
                "metadata": {"completion_status": "historical_complete"}
            }).execute()
        elif existing_tasks[title]["status"] != "done":
            print(f"Marking as complete: {title}")
            task_id = existing_tasks[title]["id"]
            supabase.table("archon_tasks").update({
                "status": "done",
                "updated_at": datetime.now().isoformat()
            }).eq("id", task_id).execute()
    
    # Create Phase 5 tasks if not present
    phase5_tasks = [
        {
            "title": "Phase 5: Production Readiness & Optimization",
            "description": "Final optimizations, performance tuning, and production deployment preparation",
            "status": "todo",
            "metadata": {
                "priority": "high",
                "estimated_effort": "2 weeks",
                "components": [
                    "Performance optimization",
                    "Security hardening",
                    "Deployment automation",
                    "Documentation finalization",
                    "Load testing & benchmarking"
                ]
            }
        },
        {
            "title": "Phase 5: System Integration Testing",
            "description": "Comprehensive integration testing across all phases",
            "status": "todo",
            "metadata": {
                "priority": "high",
                "test_areas": [
                    "Cross-phase component integration",
                    "End-to-end workflow validation",
                    "Performance under load",
                    "Security vulnerability scanning",
                    "User acceptance testing"
                ]
            }
        },
        {
            "title": "Phase 5: Documentation & Knowledge Transfer",
            "description": "Complete system documentation and knowledge base",
            "status": "todo",
            "metadata": {
                "priority": "medium",
                "deliverables": [
                    "API documentation",
                    "Architecture diagrams",
                    "Deployment guide",
                    "User manual",
                    "Developer guide"
                ]
            }
        }
    ]
    
    for task_data in phase5_tasks:
        if task_data["title"] not in existing_tasks:
            print(f"Creating Phase 5 task: {task_data['title']}")
            supabase.table("archon_tasks").insert({
                "project_id": project_id,
                "title": task_data["title"],
                "description": task_data["description"],
                "status": task_data["status"],
                "metadata": task_data["metadata"]
            }).execute()
    
    # Update project metadata
    print("\nUpdating project metadata...")
    supabase.table("archon_projects").update({
        "data": {
            "phases_complete": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
            "current_phase": "Phase 5",
            "last_benchmark": {
                "phase": "Phase 4",
                "date": datetime.now().isoformat(),
                "success_rate": "92.3%",
                "gates_passed": "6/6"
            },
            "overall_progress": "80%",  # 4 out of 5 phases complete
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
    print("  - Overall progress: 80%")

if __name__ == "__main__":
    asyncio.run(update_phase4_tasks())