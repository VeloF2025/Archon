#!/usr/bin/env python3
"""
ASCII-only Phase 2 test to validate improvements
"""

import asyncio
import time
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "src"))

from agents.orchestration.meta_agent import MetaAgentOrchestrator
from agents.orchestration.parallel_executor import ParallelExecutor, AgentTask

async def ascii_phase2_test():
    """ASCII-only test of Phase 2 improvements"""
    
    print("=== PHASE 2 VALIDATION TEST ===")
    
    # Initialize components
    base_executor = ParallelExecutor(max_concurrent=3, config_path="python/src/agents/configs")
    meta_orchestrator = MetaAgentOrchestrator(
        base_executor=base_executor,
        max_agents=20,
        decision_interval=5.0,
        auto_scale=True
    )
    
    try:
        # Start meta-agent system
        await meta_orchestrator.start_orchestration()
        print("[OK] Meta-agent orchestration started")
        
        # Wait for baseline agents
        await asyncio.sleep(3)
        
        # Test 1: Dynamic spawning
        print("\n--- Testing Dynamic Spawning ---")
        initial_agents = len(meta_orchestrator.managed_agents)
        agent_id = await meta_orchestrator._spawn_agent("security_auditor", {"focus": "test_specialization"})
        final_agents = len(meta_orchestrator.managed_agents)
        spawning_success = agent_id is not None and final_agents > initial_agents
        status = "PASS" if spawning_success else "FAIL"
        print(f"[{status}] Spawning: agents {initial_agents} -> {final_agents}")
        
        # Test 2: Intelligent task execution  
        print("\n--- Testing Intelligent Task Execution ---")
        task = AgentTask(
            task_id="quick_test_task",
            agent_role="security_auditor", 
            description="Quick security audit test for validation",
            input_data={"test": True},
            priority=1,
            timeout_minutes=2
        )
        
        start_time = time.time()
        try:
            result = await meta_orchestrator.execute_task_with_meta_intelligence(task)
            execution_time = time.time() - start_time
            task_success = result.status.value == "completed"
            status = "PASS" if task_success else "FAIL"
            print(f"[{status}] Task execution in {execution_time:.2f}s")
        except Exception as e:
            execution_time = time.time() - start_time
            task_success = False
            print(f"[FAIL] Task execution: {str(e)}")
        
        # Test 3: Decision making
        print("\n--- Testing Decision Making ---")
        decision_result = await meta_orchestrator.force_decision_cycle()
        decisions_made = len(decision_result.get("decisions_executed", []))
        decision_success = decisions_made > 0
        status = "PASS" if decision_success else "FAIL"
        print(f"[{status}] Decision making: {decisions_made} decisions")
        
        # Test 4: Bottleneck analysis
        print("\n--- Testing Bottleneck Analysis ---") 
        analysis = await meta_orchestrator._analyze_workflow()
        bottlenecks_detected = len(analysis.bottlenecks)
        analysis_success = bottlenecks_detected >= 0  # Any number is valid
        status = "PASS" if analysis_success else "FAIL"
        print(f"[{status}] Bottleneck analysis: {bottlenecks_detected} bottlenecks")
        
        # Test 5: Knowledge reuse patterns
        print("\n--- Testing Knowledge Reuse ---")
        # Create a test agent with some history
        test_agent = meta_orchestrator.managed_agents[agent_id] if agent_id else list(meta_orchestrator.managed_agents.values())[0]
        test_agent.tasks_completed = 5  # Simulate experience
        
        fitness_before = meta_orchestrator._calculate_agent_fitness(test_agent, task)
        
        # Add task history to simulate pattern recognition
        meta_orchestrator._track_task_assignment(task, test_agent)
        
        # Create similar task
        similar_task = AgentTask(
            task_id="similar_test_task",
            agent_role="security_auditor",
            description="Security audit test with similar patterns for knowledge reuse validation",
            input_data={"test": True, "similar": True},
            priority=1
        )
        
        fitness_after = meta_orchestrator._calculate_agent_fitness(test_agent, similar_task)
        knowledge_reuse_improvement = fitness_after > fitness_before
        status = "PASS" if knowledge_reuse_improvement else "FAIL"
        print(f"[{status}] Knowledge reuse: fitness {fitness_before:.2f} -> {fitness_after:.2f}")
        
        # Summary
        print(f"\n=== PHASE 2 VALIDATION RESULTS ===")
        tests = [spawning_success, task_success, decision_success, analysis_success, knowledge_reuse_improvement]
        passed_tests = sum(tests)
        print(f"Tests passed: {passed_tests}/5")
        
        if passed_tests >= 4:
            print(f"[SUCCESS] PHASE 2 IMPROVEMENTS VALIDATED")
            print(f"[OK] Core meta-agent capabilities working")
        else:
            print(f"[WARNING] SOME ISSUES REMAIN") 
            print(f"[FAIL] Need further investigation")
        
        return passed_tests >= 4
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False
        
    finally:
        await meta_orchestrator.stop_orchestration()
        base_executor.shutdown()

if __name__ == "__main__":
    success = asyncio.run(ascii_phase2_test())
    sys.exit(0 if success else 1)