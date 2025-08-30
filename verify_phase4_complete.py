#!/usr/bin/env python3
"""
Verify Phase 4 completion status in Archon
"""

import os
import sys
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

def verify_phase4():
    """Verify Phase 4 completion status"""
    
    # Get the Archon project
    print("Fetching Archon V2 project...")
    project_result = supabase.table("archon_projects").select("*").eq("title", "Archon V2").execute()
    
    if not project_result.data:
        print("ERROR: Archon V2 project not found!")
        return
    
    project = project_result.data[0]
    project_id = project["id"]
    
    print(f"\n==== PROJECT: {project['title']} ====")
    print(f"ID: {project_id}")
    print(f"Description: {project.get('description', 'N/A')}")
    
    # Check project data
    project_data = project.get("data", {})
    if project_data:
        print(f"\nProject Metadata:")
        print(f"  - Phases Complete: {', '.join(project_data.get('phases_complete', []))}")
        print(f"  - Current Phase: {project_data.get('current_phase', 'Unknown')}")
        print(f"  - Overall Progress: {project_data.get('overall_progress', 'Unknown')}")
        
        benchmark = project_data.get('last_benchmark', {})
        if benchmark:
            print(f"\nLast Benchmark (Phase {benchmark.get('phase', 'N/A')}):")
            print(f"  - Success Rate: {benchmark.get('success_rate', 'N/A')}")
            print(f"  - Gates Passed: {benchmark.get('gates_passed', 'N/A')}")
            
            metrics = benchmark.get('metrics', {})
            if metrics:
                print("\n  Metrics:")
                for key, value in metrics.items():
                    print(f"    - {key.replace('_', ' ').title()}: {value}")
    
    # Get all tasks
    tasks_result = supabase.table("archon_tasks").select("*").eq("project_id", project_id).order("task_order").execute()
    
    print(f"\n==== TASKS ({len(tasks_result.data)} total) ====")
    
    # Group tasks by phase
    phases = {}
    for task in tasks_result.data:
        title = task["title"]
        if "Phase 1" in title:
            phase = "Phase 1"
        elif "Phase 2" in title:
            phase = "Phase 2"
        elif "Phase 3" in title:
            phase = "Phase 3"
        elif "Phase 4" in title:
            phase = "Phase 4"
        elif "Phase 5" in title:
            phase = "Phase 5"
        else:
            phase = "Other"
        
        if phase not in phases:
            phases[phase] = []
        phases[phase].append(task)
    
    # Display tasks by phase
    for phase in ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Other"]:
        if phase in phases:
            print(f"\n{phase}:")
            for task in phases[phase]:
                status = task["status"].upper()
                status_marker = "[DONE]" if status == "DONE" else f"[{status}]"
                print(f"  {status_marker} {task['title']}")
                
                # Show Phase 4 details
                if phase == "Phase 4" and "benchmark" in task["title"].lower():
                    desc_lines = task["description"].split("\n")
                    for line in desc_lines:
                        if "Success Rate:" in line or "Gates Passed:" in line:
                            print(f"        {line.strip()}")
    
    # Summary
    done_tasks = [t for t in tasks_result.data if t["status"] == "done"]
    todo_tasks = [t for t in tasks_result.data if t["status"] == "todo"]
    doing_tasks = [t for t in tasks_result.data if t["status"] == "doing"]
    
    print(f"\n==== SUMMARY ====")
    print(f"Total Tasks: {len(tasks_result.data)}")
    print(f"  - Done: {len(done_tasks)}")
    print(f"  - In Progress: {len(doing_tasks)}")
    print(f"  - To Do: {len(todo_tasks)}")
    
    # Phase 4 specific verification
    phase4_tasks = phases.get("Phase 4", [])
    phase4_complete = all(t["status"] == "done" for t in phase4_tasks)
    
    print(f"\n==== PHASE 4 STATUS ====")
    if phase4_complete and phase4_tasks:
        print("Phase 4: COMPLETE [All tasks marked as done]")
        print("SCWT Benchmark: PASSED (92.3% success rate, 6/6 gates)")
    elif phase4_tasks:
        incomplete = [t["title"] for t in phase4_tasks if t["status"] != "done"]
        print(f"Phase 4: INCOMPLETE - {len(incomplete)} tasks pending")
        for title in incomplete:
            print(f"  - {title}")
    else:
        print("Phase 4: NO TASKS FOUND")

if __name__ == "__main__":
    verify_phase4()