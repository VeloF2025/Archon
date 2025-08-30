#!/usr/bin/env python3
"""
Mark Phase 1 Subtasks as Complete
Updates all completed Phase 1 subtasks to reflect current progress
"""

import requests
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mark_subtasks_complete():
    """Mark completed Phase 1 subtasks as done"""
    api_base = "http://localhost:8181/api"
    project_id = "85bc9bf7-465e-4235-9990-969adac869e5"
    
    # Subtasks we've completed
    completed_subtasks = [
        {
            "title": "Phase 1 - Fork Archon Repository", 
            "reason": "Successfully forked and pushed to VeloF2025/Archon"
        },
        {
            "title": "Phase 1 - Design 20+ Sub-Agent Roles",
            "reason": "Created 22+ specialized agent configurations with JSON schemas"
        },
        {
            "title": "Phase 1 - Implement Parallel Execution Engine", 
            "reason": "Built comprehensive parallel execution with conflict resolution"
        },
        {
            "title": "Phase 1 - Create PRP-Based Prompt System",
            "reason": "Created detailed PRP templates and management system"
        },
        {
            "title": "Phase 1 - Build Proactive Trigger System",
            "reason": "Implemented advanced file monitoring and agent invocation"
        }
    ]
    
    try:
        # Get all tasks
        response = requests.get(f"{api_base}/projects/{project_id}/tasks")
        if response.status_code != 200:
            logger.error(f"Failed to get tasks: {response.status_code}")
            return
        
        tasks = response.json()
        logger.info(f"Found {len(tasks)} total tasks")
        
        # Mark completed subtasks
        for completed in completed_subtasks:
            task_title = completed["title"]
            reason = completed["reason"]
            
            # Find matching task
            matching_task = None
            for task in tasks:
                if task["title"] == task_title:
                    matching_task = task
                    break
            
            if matching_task:
                # Update task to completed
                update_data = {
                    "status": "done",
                    "description": matching_task["description"] + f"\n\n✅ COMPLETED: {reason}\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                
                update_response = requests.put(
                    f"{api_base}/tasks/{matching_task['id']}",
                    json=update_data
                )
                
                if update_response.status_code == 200:
                    logger.info(f"✅ Marked '{task_title}' as COMPLETED")
                else:
                    logger.error(f"❌ Failed to update '{task_title}': {update_response.status_code}")
            else:
                logger.warning(f"⚠️ Could not find task: '{task_title}'")
        
        # Mark main Phase 1 task as completed
        main_phase_1_task = None
        for task in tasks:
            if "Phase 1: Fork & Specialized Global Sub-Agent System Enhancements" in task["title"]:
                main_phase_1_task = task
                break
        
        if main_phase_1_task:
            completion_summary = """
✅ PHASE 1 COMPLETED SUCCESSFULLY

🎯 **Major Accomplishments:**
• Forked Archon to VeloF2025/Archon repository  
• Created 22+ specialized agent configurations
• Built parallel execution engine with conflict resolution
• Implemented PRP-based prompt template system
• Created proactive trigger system with file monitoring
• All agent assignments properly configured for AI execution

📊 **Deliverables:**
• Agent configurations: 22 roles with JSON schemas
• Execution engine: Parallel orchestration system
• PRP templates: 5+ detailed prompt templates
• Trigger system: Advanced file monitoring and agent invocation
• Agent registry: Master configuration system

🚀 **Ready for Phase 2:** Meta-Agent Integration and dynamic spawning system

⏰ **Completed:** """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            update_data = {
                "status": "done",
                "description": main_phase_1_task["description"] + completion_summary
            }
            
            update_response = requests.put(
                f"{api_base}/tasks/{main_phase_1_task['id']}",
                json=update_data
            )
            
            if update_response.status_code == 200:
                logger.info("🎉 PHASE 1 MARKED AS COMPLETED!")
            else:
                logger.error(f"Failed to update main Phase 1 task: {update_response.status_code}")
        
        logger.info("Phase 1 completion marking finished!")
        
    except Exception as e:
        logger.error(f"Error marking tasks complete: {e}")

if __name__ == "__main__":
    mark_subtasks_complete()