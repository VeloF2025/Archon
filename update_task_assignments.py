#!/usr/bin/env python3
"""
Update Task Assignments Script
Updates all existing Archon+ tasks to reflect AI agent assignments instead of user assignments
"""

from dynamic_task_tracker import ArchonTaskTracker
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Update all task assignments to reflect AI agents"""
    logger.info("Starting task assignment update process...")
    
    # Initialize tracker
    tracker = ArchonTaskTracker()
    
    # Update all task assignments
    success = tracker.update_all_task_assignments()
    
    if success:
        print("TASK ASSIGNMENTS UPDATED SUCCESSFULLY")
        print("All tasks now show proper AI agent assignments instead of 'User'")
        
        # Show updated project status
        status = tracker.get_project_status()
        print(f"Project Status: {status['phases_completed']} completed, {status['phases_in_progress']} in progress, {status['phases_pending']} pending")
        
    else:
        print("FAILED TO UPDATE TASK ASSIGNMENTS")
        print("Check logs for error details")

if __name__ == "__main__":
    main()