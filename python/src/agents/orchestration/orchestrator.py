#!/usr/bin/env python3
"""
Main Orchestrator for Archon+ Agent System
Coordinates parallel execution, agent pool management, and task scheduling

MANDATORY: All orchestration operations must comply with ARCHON OPERATIONAL MANIFEST (MANIFEST.md)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .parallel_executor import ParallelExecutor, AgentTask, AgentStatus
from .agent_pool import AgentPool, AgentState

# MANDATORY: Import manifest integration
try:
    from ..configs.MANIFEST_INTEGRATION import get_archon_manifest, enforce_manifest_compliance
except ImportError:
    # Handle relative import issues
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from configs.MANIFEST_INTEGRATION import get_archon_manifest, enforce_manifest_compliance

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationResult:
    """Result of orchestration execution"""
    execution_id: str
    start_time: float
    end_time: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    timeout_tasks: int
    agent_utilization: Dict[str, float]
    execution_metrics: Dict[str, Any]

class ArchonOrchestrator:
    """
    Main orchestrator that combines parallel execution with agent pool management
    Provides high-level interface for Archon+ agent system
    """
    
    def __init__(self, 
                 config_path: str = "python/src/agents/configs",
                 max_concurrent_tasks: int = 8,
                 max_agents_per_role: int = 3,
                 max_total_agents: int = 20,
                 auto_scale: bool = True):
        
        # MANDATORY: Enforce manifest compliance before initialization
        if not enforce_manifest_compliance("ArchonOrchestrator", "initialization"):
            raise RuntimeError("MANIFEST COMPLIANCE FAILURE: ArchonOrchestrator cannot initialize without manifest")
        
        self.config_path = Path(config_path)
        
        # MANDATORY: Load manifest for orchestration rules
        self.manifest = get_archon_manifest()
        self.orchestration_rules = self.manifest.get_agent_orchestration_rules()
        
        # Initialize components
        self.executor = ParallelExecutor(
            max_concurrent=max_concurrent_tasks,
            config_path=str(config_path)
        )
        
        self.agent_pool = AgentPool(
            config_path=str(config_path),
            max_agents_per_role=max_agents_per_role,
            max_total_agents=max_total_agents,
            auto_scale=auto_scale
        )
        
        # Orchestration state
        self.execution_history: List[OrchestrationResult] = []
        self.active_executions: Dict[str, Dict] = {}
        
        # Initialize minimum agents
        self.agent_pool.ensure_minimum_agents()
        
        logger.info("✅ ArchonOrchestrator initialized with MANIFEST compliance")
        logger.info(f"📋 Loaded orchestration rules: {len(self.orchestration_rules)} categories")
    
    def create_task(self, 
                   agent_role: str,
                   description: str,
                   input_data: Dict,
                   dependencies: List[str] = None,
                   priority: int = 1,
                   timeout_minutes: int = 30) -> AgentTask:
        """Create a new agent task"""
        
        task_id = f"task_{agent_role}_{int(time.time() * 1000)}"
        
        # Get agent config to determine isolation requirements
        agent_config = self.executor.agent_configs.get(agent_role)
        requires_isolation = False
        if agent_config:
            requires_isolation = agent_config.execution_context.get("requires_isolation", False)
        
        task = AgentTask(
            task_id=task_id,
            agent_role=agent_role,
            description=description,
            input_data=input_data,
            dependencies=dependencies or [],
            priority=priority,
            timeout_minutes=timeout_minutes,
            requires_isolation=requires_isolation
        )
        
        logger.info(f"Created task {task_id} for agent role {agent_role}")
        return task
    
    def add_task(self, task: AgentTask) -> bool:
        """Add task to execution queue"""
        return self.executor.add_task(task)
    
    def create_workflow(self, workflow_definition: Dict) -> List[AgentTask]:
        """
        Create a workflow of tasks from definition
        
        Workflow format:
        {
            "name": "workflow_name",
            "tasks": [
                {
                    "agent_role": "python_backend_coder",
                    "description": "Task description",
                    "input_data": {...},
                    "depends_on": ["task_name1", "task_name2"],
                    "priority": 1
                }
            ]
        }
        """
        tasks = []
        task_map = {}  # name -> task_id mapping
        
        workflow_name = workflow_definition.get("name", "unnamed_workflow")
        
        # First pass: create all tasks
        for task_def in workflow_definition.get("tasks", []):
            task = self.create_task(
                agent_role=task_def["agent_role"],
                description=task_def["description"],
                input_data=task_def.get("input_data", {}),
                priority=task_def.get("priority", 1),
                timeout_minutes=task_def.get("timeout_minutes", 30)
            )
            
            tasks.append(task)
            task_name = task_def.get("name", task.task_id)
            task_map[task_name] = task.task_id
        
        # Second pass: resolve dependencies
        for i, task_def in enumerate(workflow_definition.get("tasks", [])):
            task = tasks[i]
            depends_on = task_def.get("depends_on", [])
            
            # Convert task names to task IDs
            task.dependencies = [task_map.get(dep_name, dep_name) for dep_name in depends_on]
        
        logger.info(f"Created workflow '{workflow_name}' with {len(tasks)} tasks")
        return tasks
    
    async def execute_workflow(self, 
                              workflow_definition: Dict,
                              timeout_minutes: int = 60) -> OrchestrationResult:
        """Execute a complete workflow"""
        
        execution_id = f"exec_{int(time.time() * 1000)}"
        start_time = time.time()
        
        logger.info(f"Starting workflow execution {execution_id}")
        
        # Create workflow tasks
        tasks = self.create_workflow(workflow_definition)
        
        # Add all tasks to executor
        for task in tasks:
            self.add_task(task)
        
        # Execute with enhanced monitoring
        results = await self._execute_with_monitoring(execution_id, timeout_minutes)
        
        end_time = time.time()
        
        # Create orchestration result
        result = OrchestrationResult(
            execution_id=execution_id,
            start_time=start_time,
            end_time=end_time,
            total_tasks=len(tasks),
            completed_tasks=len(results["completed"]),
            failed_tasks=len(results["failed"]),
            timeout_tasks=len(results["timeout"]),
            agent_utilization=self._calculate_agent_utilization(),
            execution_metrics=self._gather_execution_metrics(results)
        )
        
        self.execution_history.append(result)
        
        logger.info(f"Workflow execution {execution_id} completed in {end_time - start_time:.2f}s")
        return result
    
    async def _execute_with_monitoring(self, execution_id: str, timeout_minutes: int) -> Dict:
        """Execute tasks with enhanced monitoring and agent management"""
        
        self.active_executions[execution_id] = {
            "start_time": time.time(),
            "status": "running"
        }
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(
            self._monitor_execution(execution_id)
        )
        
        try:
            # Execute batch with timeout
            results = await self.executor.execute_batch(timeout_minutes)
            
            self.active_executions[execution_id]["status"] = "completed"
            return results
            
        except Exception as e:
            self.active_executions[execution_id]["status"] = "failed"
            self.active_executions[execution_id]["error"] = str(e)
            logger.error(f"Execution {execution_id} failed: {e}")
            raise
        
        finally:
            monitoring_task.cancel()
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _monitor_execution(self, execution_id: str):
        """Monitor execution progress and manage agents"""
        try:
            while execution_id in self.active_executions:
                # Scale agents based on load
                self.agent_pool.scale_agents()
                
                # Log progress
                executor_status = self.executor.get_status()
                pool_status = self.agent_pool.get_pool_status()
                
                logger.debug(f"Execution {execution_id} progress: "
                           f"Running: {executor_status['running_tasks']}, "
                           f"Queued: {executor_status['queued_tasks']}, "
                           f"Agents: {pool_status['total_agents']}")
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for execution {execution_id}")
    
    def _calculate_agent_utilization(self) -> Dict[str, float]:
        """Calculate agent utilization metrics"""
        pool_status = self.agent_pool.get_pool_status()
        utilization = {}
        
        for role, role_data in pool_status["role_distribution"].items():
            total = role_data["count"]
            busy = role_data["busy"]
            
            if total > 0:
                utilization[role] = busy / total
            else:
                utilization[role] = 0.0
        
        return utilization
    
    def _gather_execution_metrics(self, results: Dict) -> Dict[str, Any]:
        """Gather comprehensive execution metrics"""
        metrics = {
            "success_rate": 0.0,
            "average_task_duration": 0.0,
            "total_execution_time": 0.0,
            "resource_efficiency": 0.0,
            "dependency_resolution_time": 0.0
        }
        
        all_tasks = results["completed"] + results["failed"] + results["timeout"]
        
        if all_tasks:
            # Success rate
            metrics["success_rate"] = len(results["completed"]) / len(all_tasks)
            
            # Average task duration
            completed_tasks = results["completed"]
            if completed_tasks:
                durations = []
                for task in completed_tasks:
                    if task.start_time and task.end_time:
                        durations.append(task.end_time - task.start_time)
                
                if durations:
                    metrics["average_task_duration"] = sum(durations) / len(durations)
        
        # Resource efficiency (placeholder - would calculate based on actual usage)
        pool_status = self.agent_pool.get_pool_status()
        if pool_status["total_agents"] > 0:
            busy_agents = pool_status["task_statistics"]["currently_busy"]
            metrics["resource_efficiency"] = busy_agents / pool_status["total_agents"]
        
        return metrics
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        executor_status = self.executor.get_status()
        pool_status = self.agent_pool.get_pool_status()
        
        return {
            "orchestrator": {
                "active_executions": len(self.active_executions),
                "total_executions": len(self.execution_history),
                "last_execution": self.execution_history[-1].execution_id if self.execution_history else None
            },
            "executor": executor_status,
            "agent_pool": pool_status,
            "system_health": {
                "healthy_agents": pool_status["agent_states"].get("idle", 0) + pool_status["agent_states"].get("active", 0) + pool_status["agent_states"].get("busy", 0),
                "error_agents": pool_status["agent_states"].get("error", 0),
                "total_memory_mb": pool_status["resource_usage"]["total_memory_mb"],
                "average_cpu_percent": pool_status["resource_usage"]["average_cpu_percent"]
            }
        }
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """Get execution history summary"""
        recent_executions = self.execution_history[-limit:] if limit else self.execution_history
        
        history = []
        for result in recent_executions:
            history.append({
                "execution_id": result.execution_id,
                "duration": result.end_time - result.start_time,
                "total_tasks": result.total_tasks,
                "success_rate": result.completed_tasks / result.total_tasks if result.total_tasks > 0 else 0,
                "agent_utilization": result.agent_utilization,
                "start_time": result.start_time
            })
        
        return history
    
    def shutdown(self):
        """Graceful shutdown of orchestrator"""
        logger.info("Shutting down ArchonOrchestrator")
        
        # Cancel active executions
        for execution_id in list(self.active_executions.keys()):
            self.active_executions[execution_id]["status"] = "cancelled"
        
        # Shutdown components
        self.executor.shutdown()
        self.agent_pool.shutdown()
        
        logger.info("ArchonOrchestrator shutdown complete")

# Example workflow definitions
SAMPLE_WORKFLOWS = {
    "auth_system_workflow": {
        "name": "Authentication System Implementation",
        "tasks": [
            {
                "name": "backend_auth",
                "agent_role": "python_backend_coder",
                "description": "Implement authentication API endpoints",
                "input_data": {"endpoints": ["/auth/login", "/auth/logout", "/auth/refresh"]},
                "priority": 1
            },
            {
                "name": "frontend_auth",
                "agent_role": "typescript_frontend_agent",
                "description": "Create authentication UI components",
                "input_data": {"components": ["LoginForm", "LogoutButton", "AuthGuard"]},
                "depends_on": ["backend_auth"],
                "priority": 1
            },
            {
                "name": "auth_tests",
                "agent_role": "test_generator",
                "description": "Generate comprehensive auth tests",
                "input_data": {"test_types": ["unit", "integration", "e2e"]},
                "depends_on": ["backend_auth", "frontend_auth"],
                "priority": 2
            },
            {
                "name": "security_audit",
                "agent_role": "security_auditor",
                "description": "Security audit of auth system",
                "input_data": {"audit_scope": ["endpoints", "tokens", "session_management"]},
                "depends_on": ["backend_auth"],
                "priority": 1
            },
            {
                "name": "auth_docs",
                "agent_role": "documentation_writer",
                "description": "Create auth system documentation",
                "input_data": {"doc_types": ["api", "user_guide", "security"]},
                "depends_on": ["backend_auth", "frontend_auth", "security_audit"],
                "priority": 3
            }
        ]
    }
}

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize orchestrator
        orchestrator = ArchonOrchestrator(max_concurrent_tasks=6, max_total_agents=15)
        
        try:
            # Execute sample workflow
            print("Starting authentication system workflow...")
            
            result = await orchestrator.execute_workflow(
                SAMPLE_WORKFLOWS["auth_system_workflow"],
                timeout_minutes=15
            )
            
            # Print results
            print(f"\nWorkflow Results:")
            print(f"Execution ID: {result.execution_id}")
            print(f"Duration: {result.end_time - result.start_time:.2f}s")
            print(f"Tasks: {result.completed_tasks}/{result.total_tasks} completed")
            print(f"Success Rate: {result.completed_tasks/result.total_tasks*100:.1f}%")
            print(f"Agent Utilization: {result.agent_utilization}")
            
            # System status
            status = orchestrator.get_system_status()
            print(f"\nSystem Status:")
            print(f"Total Agents: {status['agent_pool']['total_agents']}")
            print(f"Healthy Agents: {status['system_health']['healthy_agents']}")
            print(f"Memory Usage: {status['system_health']['total_memory_mb']:.1f} MB")
            
        except Exception as e:
            print(f"Workflow execution failed: {e}")
        
        finally:
            orchestrator.shutdown()
    
    # Run example
    asyncio.run(main())