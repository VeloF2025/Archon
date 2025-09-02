#!/usr/bin/env python3
"""
Verify Task Assignments Script
Checks that all tasks have proper AI agent assignments
"""

import requests
import json

def verify_task_assignments():
    """Verify all tasks have proper AI agent assignments"""
    api_base = "http://localhost:8181/api"
    project_id = "85bc9bf7-465e-4235-9990-969adac869e5"
    
    try:
        response = requests.get(f"{api_base}/projects/{project_id}/tasks")
        if response.status_code != 200:
            print(f"Failed to get tasks: {response.status_code}")
            return
        
        tasks = response.json()
        print(f"VERIFYING {len(tasks)} TASKS:")
        print("=" * 60)
        
        for task in tasks:
            title = task["title"]
            description = task["description"]
            
            # Check if task has agent assignment
            if "🤖 ASSIGNED TO:" in description:
                # Extract agent name
                lines = description.split("\n")
                agent_line = [line for line in lines if "🤖 ASSIGNED TO:" in line][0]
                agent_name = agent_line.replace("🤖 ASSIGNED TO:", "").strip()
                
                print(f"PASS {title}")
                print(f"   Agent: {agent_name}")
                
                # Check if user assignment note is present
                if "This task will be executed by" in description and "not the user" in description:
                    print(f"   Execution: AI Agent (not user)")
                else:
                    print(f"   WARNING: Missing execution note")
                    
            else:
                print(f"FAIL {title}")
                print(f"   WARNING: NO AGENT ASSIGNMENT FOUND")
            
            print()
        
        print("VERIFICATION COMPLETE")
        
    except Exception as e:
        print(f"Error verifying assignments: {e}")

if __name__ == "__main__":
    verify_task_assignments()