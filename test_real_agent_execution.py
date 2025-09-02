#!/usr/bin/env python3
"""
Test real agent execution through HTTP API
Verifies that the parallel executor can call actual agents
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "python" / "src"))

from agents.orchestration.parallel_executor import ParallelExecutor, AgentTask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_agent_execution():
    """Test that we can execute real agents through HTTP API"""
    
    logger.info("=== TESTING REAL AGENT EXECUTION ===")
    
    # Initialize executor
    executor = ParallelExecutor(
        max_concurrent=2,
        config_path="python/src/agents/configs"
    )
    
    # Create a simple test task
    test_task = AgentTask(
        task_id="test_agent_call",
        agent_role="python_backend_coder",
        description="Write a simple Python function that adds two numbers",
        input_data={
            "project_name": "Test Project", 
            "requirements": "Create a function called add_numbers(a, b) that returns a + b"
        },
        priority=1,
        timeout_minutes=5
    )
    
    # Add task to executor
    success = executor.add_task(test_task)
    if not success:
        logger.error("Failed to add task to executor")
        return False
    
    logger.info(f"Added task {test_task.task_id} to executor")
    
    # Execute the task
    try:
        results = await executor.execute_batch(timeout_minutes=10)
        
        logger.info("=== EXECUTION RESULTS ===")
        logger.info(f"Completed: {len(results['completed'])}")
        logger.info(f"Failed: {len(results['failed'])}")
        logger.info(f"Timeout: {len(results['timeout'])}")
        
        # Debug: Check executor state
        logger.info(f"Active tasks: {len(executor.active_tasks)}")
        logger.info(f"Running futures: {len(executor.running_futures)}")
        logger.info(f"Completed tasks: {len(executor.completed_tasks)}")
        logger.info(f"Task queue: {len(executor.task_queue)}")
        
        # Check results
        if results['completed']:
            task = results['completed'][0]
            logger.info(f"Task {task.task_id} completed successfully!")
            logger.info(f"Agent result: {json.dumps(task.result, indent=2)}")
            return True
        elif results['failed']:
            task = results['failed'][0]
            logger.error(f"Task {task.task_id} failed: {task.error_message}")
            return False
        elif executor.completed_tasks:
            # Check if tasks are in the executor's completed list but not in results
            task = executor.completed_tasks[0]
            logger.info(f"Found completed task in executor: {task.task_id}")
            logger.info(f"Task status: {task.status}")
            logger.info(f"Agent result: {json.dumps(task.result, indent=2)}")
            return True
        else:
            logger.error("No results returned")
            # Show detailed state for debugging
            for task_id, task in executor.active_tasks.items():
                logger.error(f"Active task {task_id}: status={task.status}, error={task.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False
    finally:
        executor.shutdown()

async def main():
    """Main test function"""
    success = await test_real_agent_execution()
    
    if success:
        print("\n✅ REAL AGENT EXECUTION TEST: PASSED")
        print("The parallel executor successfully called real agents via HTTP API!")
        sys.exit(0)
    else:
        print("\n❌ REAL AGENT EXECUTION TEST: FAILED") 
        print("Could not execute real agents - check service status and logs")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())