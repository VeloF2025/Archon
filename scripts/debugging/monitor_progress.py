#!/usr/bin/env python3
"""
Archon Self-Enhancement Progress Monitor
Real-time monitoring and reporting of project progress
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import Dict, List, Any

class ArchonProgressMonitor:
    def __init__(self, project_id: str = "422c4712-8619-4788-804c-3016cbc37478"):
        self.project_id = project_id
        self.base_url = "http://localhost:8181"
        
    async def get_project_status(self) -> Dict[str, Any]:
        """Get current project and task status"""
        async with httpx.AsyncClient() as client:
            try:
                # Get project details
                project_response = await client.get(f"{self.base_url}/api/projects/{self.project_id}")
                project_data = project_response.json() if project_response.status_code == 200 else {}
                
                # Get all projects to find tasks
                projects_response = await client.get(f"{self.base_url}/api/projects")
                projects_data = projects_response.json() if projects_response.status_code == 200 else []
                
                # Find our project in the list (it should have tasks)
                our_project = None
                for project in projects_data:
                    if project.get('id') == self.project_id:
                        our_project = project
                        break
                
                return {
                    "project": project_data,
                    "project_with_tasks": our_project,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def generate_progress_report(self) -> str:
        """Generate a detailed progress report"""
        status_data = await self.get_project_status()
        
        if "error" in status_data:
            return f"ERROR: Could not fetch project status - {status_data['error']}"
        
        project = status_data.get("project", {})
        project_with_tasks = status_data.get("project_with_tasks")
        
        # Extract task information
        tasks = []
        if project_with_tasks and isinstance(project_with_tasks, dict):
            # Tasks might be in different locations depending on API structure
            tasks = project_with_tasks.get("tasks", [])
            if not tasks and "data" in project_with_tasks:
                tasks = project_with_tasks["data"].get("tasks", [])
        
        # Generate report
        report = f"""
ARCHON SELF-ENHANCEMENT PROJECT STATUS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

PROJECT OVERVIEW:
  Title: {project.get('title', 'Unknown')}
  Description: {project.get('description', 'No description')[:100]}...
  Created: {project.get('created_at', 'Unknown')[:19].replace('T', ' ')}
  Updated: {project.get('updated_at', 'Unknown')[:19].replace('T', ' ')}

TASK SUMMARY:
  Total Tasks: {len(tasks)}
"""
        
        if tasks:
            # Analyze task status
            status_counts = {}
            priority_counts = {}
            
            for task in tasks:
                status = task.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Try to extract priority from description
                desc = task.get('description', '').upper()
                if 'CRITICAL:' in desc:
                    priority = 'critical'
                elif 'HIGH:' in desc:
                    priority = 'high'
                elif 'MEDIUM:' in desc:
                    priority = 'medium'
                else:
                    priority = 'unknown'
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Add status breakdown
            report += "\nTASK STATUS BREAKDOWN:\n"
            for status, count in status_counts.items():
                percentage = (count / len(tasks)) * 100
                report += f"  {status.title()}: {count} tasks ({percentage:.1f}%)\n"
            
            # Add priority breakdown
            report += "\nPRIORITY BREAKDOWN:\n"
            for priority, count in priority_counts.items():
                percentage = (count / len(tasks)) * 100
                report += f"  {priority.title()}: {count} tasks ({percentage:.1f}%)\n"
            
            # List all tasks with status
            report += f"\nDETAILED TASK LIST:\n"
            for i, task in enumerate(tasks, 1):
                title = task.get('title', 'Untitled Task')
                status = task.get('status', 'unknown')
                assignee = task.get('assignee', 'unassigned')
                
                # Extract priority from description
                desc = task.get('description', '')
                if 'CRITICAL:' in desc.upper():
                    priority_marker = '[CRITICAL]'
                elif 'HIGH:' in desc.upper():
                    priority_marker = '[HIGH]'
                elif 'MEDIUM:' in desc.upper():
                    priority_marker = '[MEDIUM]'
                else:
                    priority_marker = ''
                
                report += f"  {i:2d}. {priority_marker} {title}\n"
                report += f"      Status: {status.title()} | Assignee: {assignee}\n"
        
        else:
            report += "\n  No tasks found - this might be a data structure issue\n"
            report += f"\n  Debug info - Project keys: {list(project.keys())}\n"
            if project_with_tasks:
                report += f"  Project with tasks keys: {list(project_with_tasks.keys())}\n"
        
        # Add next steps
        report += f"""

NEXT STEPS:
1. Start with CRITICAL priority tasks (can run in parallel):
   - Implement DeepConf Lazy Loading (performance-optimizer)
   - Simplify Meta-Agent Orchestration (system-architect)

2. Then HIGH priority integration fixes:
   - Connect DeepConf to Main Workflow (code-implementer)
   - Complete Claude Code Task Tool Integration (code-implementer)
   - Activate TDD Enforcement Gate (code-implementer)

3. Complete with MEDIUM priority improvements:
   - UI/UX optimizations
   - Memory usage optimization
   - Documentation

ESTIMATED TIMELINE:
  With 2-3 agents working in parallel: 2-3 weeks
  Sequential execution: 4-6 weeks

PROJECT URL: http://localhost:3737/projects/{self.project_id}
{'='*60}
"""
        
        return report

    async def monitor_continuously(self, interval_seconds: int = 30):
        """Monitor project progress continuously"""
        print("Starting continuous monitoring of Archon Self-Enhancement project...")
        print(f"Monitoring interval: {interval_seconds} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                report = await self.generate_progress_report()
                
                # Clear screen and show report
                print("\033[2J\033[H", end="")  # Clear screen
                print(report)
                
                # Wait for next check
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nMonitoring error: {str(e)}")

async def main():
    """Main function to run progress monitoring"""
    monitor = ArchonProgressMonitor()
    
    print("Archon Self-Enhancement Progress Monitor")
    print("=" * 50)
    
    # Generate initial report
    report = await monitor.generate_progress_report()
    print(report)
    
    # Ask if user wants continuous monitoring
    try:
        choice = input("\nStart continuous monitoring? (y/N): ").strip().lower()
        if choice in ['y', 'yes']:
            await monitor.monitor_continuously()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    asyncio.run(main())