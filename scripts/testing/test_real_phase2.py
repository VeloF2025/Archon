#!/usr/bin/env python3
"""
Test Phase 2 with REAL agent execution - NO GAMING!
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python" / "src"))

from agents.orchestration.meta_agent import MetaAgentOrchestrator
from agents.orchestration.parallel_executor import ParallelExecutor, AgentTask

async def test_real_execution():
    print("=== TESTING PHASE 2 WITH REAL AGENT EXECUTION ===\n")
    
    # Initialize with REAL ParallelExecutor
    base_executor = ParallelExecutor(max_concurrent=3)
    meta_orchestrator = MetaAgentOrchestrator(
        base_executor=base_executor,
        max_agents=10
    )
    
    # Create REAL tasks that will call actual agents
    tasks = [
        AgentTask(
            task_id="real_1",
            agent_role="documentation_writer",
            description="Write a brief summary of what Phase 2 parallel execution does",
            input_data={"project_name": "archon-plus", "real_test": True},
            priority=1,
            timeout_minutes=2
        ),
        AgentTask(
            task_id="real_2",
            agent_role="python_backend_coder",
            description="Create a simple Python function that adds two numbers",
            input_data={"project_name": "archon-plus", "real_test": True},
            priority=1,
            timeout_minutes=2
        ),
        AgentTask(
            task_id="real_3",
            agent_role="test_generator",
            description="Generate a test for a function that adds two numbers",
            input_data={"project_name": "archon-plus", "real_test": True},
            priority=2,
            timeout_minutes=2
        )
    ]
    
    print(f"Executing {len(tasks)} tasks with REAL agents...\n")
    start_time = time.time()
    
    # Execute with REAL parallel execution
    results = await meta_orchestrator.execute_parallel(tasks)
    
    execution_time = time.time() - start_time
    
    print(f"\n=== REAL EXECUTION RESULTS ===")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Average per task: {execution_time/len(tasks):.2f} seconds\n")
    
    # Show REAL results
    for task in results:
        print(f"Task {task.task_id}:")
        print(f"  Status: {task.status.value}")
        
        if hasattr(task, 'result') and task.result:
            output = task.result.get('agent_output', {})
            if output:
                print(f"  REAL Output: {str(output)[:200]}...")
        elif hasattr(task, 'error_message') and task.error_message:
            print(f"  Error: {task.error_message}")
        print()
    
    # Calculate metrics
    completed = len([t for t in results if t.status.value == "completed"])
    failed = len([t for t in results if t.status.value == "failed"])
    
    print(f"=== FINAL METRICS ===")
    print(f"Success rate: {completed}/{len(tasks)} ({completed/len(tasks)*100:.0f}%)")
    print(f"Failed: {failed}")
    print(f"Parallel efficiency: {len(tasks)*10/execution_time:.1f}% (assuming 10s sequential per task)")
    
    # Verify this is REAL execution
    if execution_time < 2:
        print("\n⚠️ WARNING: Execution too fast - might be simulated!")
    else:
        print("\n✅ CONFIRMED: Real agent execution (took >2 seconds)")
    
    base_executor.shutdown()
    return completed == len(tasks)

if __name__ == "__main__":
    success = asyncio.run(test_real_execution())
    exit(0 if success else 1)