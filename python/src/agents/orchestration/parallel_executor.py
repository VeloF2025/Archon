#!/usr/bin/env python3
"""
Parallel Execution Engine for Archon+ Sub-Agents
Handles concurrent agent execution with conflict resolution
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Callable
import redis
import threading
from contextlib import contextmanager
import httpx
import os

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class ConflictResolutionStrategy(Enum):
    REDIS_LOCKS = "redis_locks"
    GIT_WORKTREES = "git_worktrees"
    QUEUE_SERIALIZE = "queue_serialize"

@dataclass
class AgentTask:
    """Individual agent task representation"""
    task_id: str
    agent_role: str
    description: str
    input_data: Dict
    dependencies: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout_minutes: int = 30
    requires_isolation: bool = False
    status: AgentStatus = AgentStatus.IDLE
    result: Optional[Dict] = None
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class AgentConfig:
    """Agent configuration from JSON files"""
    role: str
    name: str
    description: str
    memory_scope: List[str]
    skills: List[str]
    proactive_triggers: List[str]
    prp_template: str
    priority: str
    execution_context: Dict
    dependencies: List[str]
    output_patterns: List[str]

class ConflictResolver:
    """Handles file-level conflicts between agents"""
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.REDIS_LOCKS):
        self.strategy = strategy
        self.redis_client = None
        self.file_locks: Dict[str, threading.Lock] = {}
        
        if strategy == ConflictResolutionStrategy.REDIS_LOCKS:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Redis conflict resolution initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to local locks: {e}")
                self.strategy = ConflictResolutionStrategy.QUEUE_SERIALIZE
    
    @contextmanager
    def acquire_file_lock(self, file_path: str, timeout: int = 30):
        """Acquire exclusive lock on file path"""
        lock_key = f"archon:file_lock:{file_path}"
        
        if self.strategy == ConflictResolutionStrategy.REDIS_LOCKS and self.redis_client:
            # Redis distributed lock
            lock_acquired = False
            try:
                lock_acquired = self.redis_client.set(lock_key, "locked", nx=True, ex=timeout)
                if lock_acquired:
                    logger.debug(f"Acquired Redis lock for {file_path}")
                    yield True
                else:
                    logger.warning(f"Failed to acquire Redis lock for {file_path}")
                    yield False
            finally:
                if lock_acquired and self.redis_client:
                    self.redis_client.delete(lock_key)
                    logger.debug(f"Released Redis lock for {file_path}")
        else:
            # Local thread lock fallback
            if file_path not in self.file_locks:
                self.file_locks[file_path] = threading.Lock()
            
            lock = self.file_locks[file_path]
            acquired = lock.acquire(timeout=timeout)
            try:
                if acquired:
                    logger.debug(f"Acquired local lock for {file_path}")
                    yield True
                else:
                    logger.warning(f"Failed to acquire local lock for {file_path}")
                    yield False
            finally:
                if acquired:
                    lock.release()
                    logger.debug(f"Released local lock for {file_path}")

class ParallelExecutor:
    """Main parallel execution engine for Archon+ agents"""
    
    def __init__(self, 
                 max_concurrent: int = 8,
                 conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.REDIS_LOCKS,
                 config_path: str = "python/src/agents/configs"):
        self.max_concurrent = max_concurrent
        self.conflict_resolver = ConflictResolver(conflict_strategy)
        self.config_path = Path(config_path)
        
        # Agent management
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.running_futures: Dict[str, asyncio.Future] = {}
        
        # Load agent configurations
        self._load_agent_configs()
        
        logger.info(f"ParallelExecutor initialized with {len(self.agent_configs)} agents, max_concurrent={max_concurrent}")
    
    async def _load_prp_template(self, template_name: str) -> str:
        """Load PRP template content from file"""
        if not template_name:
            return "Execute the given task using your specialized capabilities."
        
        template_path = self.config_path.parent / "prompts" / "prp" / f"{template_name}"
        
        try:
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"PRP template not found: {template_path}")
                return "Execute the given task using your specialized capabilities."
        except Exception as e:
            logger.error(f"Failed to load PRP template {template_name}: {e}")
            return "Execute the given task using your specialized capabilities."
    
    async def _format_prp_prompt(self, prp_template: str, task: AgentTask, agent_config: AgentConfig) -> str:
        """Format PRP template with task-specific data"""
        try:
            # Basic variable interpolation
            formatted = prp_template
            
            # Replace common variables
            variables = {
                "task_id": task.task_id,
                "task_description": task.description,
                "agent_role": task.agent_role,
                "agent_name": agent_config.name,
                "input_data": json.dumps(task.input_data, indent=2),
                "file_patterns": ", ".join(task.file_patterns) if task.file_patterns else "N/A",
                "priority": str(task.priority),
                "skills": ", ".join(agent_config.skills),
                "project_name": task.input_data.get("project_name", "Current Project"),
                "requirements": task.input_data.get("requirements", task.description)
            }
            
            # Replace variables in template
            for var_name, var_value in variables.items():
                formatted = formatted.replace(f"{{{var_name}}}", str(var_value))
                formatted = formatted.replace(f"{{{{var_name}}}}", str(var_value))  # Handle double braces
            
            return formatted
            
        except Exception as e:
            logger.error(f"Failed to format PRP template: {e}")
            return f"Execute the following task: {task.description}\n\nInput data: {json.dumps(task.input_data, indent=2)}"
    
    def _map_agent_role_to_type(self, agent_role: str) -> str:
        """Map agent role to agent service type"""
        # Map specialized roles to available agent types in the service
        role_mapping = {
            "python_backend_coder": "rag",
            "typescript_frontend_agent": "rag", 
            "test_generator": "rag",
            "security_auditor": "rag",
            "documentation_writer": "rag",
            "code_reviewer": "rag",
            "refactoring_specialist": "rag",
            "performance_optimizer": "rag",
            "error_handler": "rag",
            "system_architect": "rag",
            "database_designer": "rag",
            "ui_ux_designer": "rag",
            "api_integrator": "rag",
            "devops_engineer": "rag",
            "data_analyst": "rag",
            "configuration_manager": "rag",
            "integration_tester": "rag",
            "monitoring_agent": "rag",
            "quality_assurance": "rag",
            "deployment_coordinator": "rag",
            "technical_writer": "rag",
            "hrm_reasoning_agent": "rag"
        }
        
        return role_mapping.get(agent_role, "rag")  # Default to rag agent
    
    def _load_agent_configs(self):
        """Load all agent configurations from JSON files"""
        if not self.config_path.exists():
            logger.error(f"Agent config path does not exist: {self.config_path}")
            return
        
        config_files = list(self.config_path.glob("*.json"))
        config_files = [f for f in config_files if f.name != "agent_registry.json"]
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                agent_config = AgentConfig(**config_data)
                self.agent_configs[agent_config.role] = agent_config
                logger.debug(f"Loaded agent config: {agent_config.role}")
                
            except Exception as e:
                logger.error(f"Failed to load agent config {config_file}: {e}")
        
        logger.info(f"Loaded {len(self.agent_configs)} agent configurations")
    
    def add_task(self, task: AgentTask) -> bool:
        """Add task to execution queue"""
        # Validate agent exists
        if task.agent_role not in self.agent_configs:
            logger.error(f"Unknown agent role: {task.agent_role}")
            return False
        
        # Check dependencies are satisfied
        missing_deps = []
        for dep_task_id in task.dependencies:
            if dep_task_id not in [t.task_id for t in self.completed_tasks]:
                missing_deps.append(dep_task_id)
        
        if missing_deps:
            logger.warning(f"Task {task.task_id} has unsatisfied dependencies: {missing_deps}")
            task.status = AgentStatus.BLOCKED
        else:
            task.status = AgentStatus.QUEUED
        
        self.task_queue.append(task)
        self.active_tasks[task.task_id] = task
        logger.info(f"Added task {task.task_id} for agent {task.agent_role}")
        return True
    
    def _check_file_conflicts(self, task: AgentTask) -> List[str]:
        """Check for file-level conflicts with running tasks"""
        conflicts = []
        agent_config = self.agent_configs[task.agent_role]
        
        # Get files this task will modify
        task_files = set()
        for pattern in agent_config.output_patterns:
            # Simplified pattern matching - in real implementation would use glob
            task_files.add(pattern)
        
        # Check against running tasks
        for running_task_id, running_future in self.running_futures.items():
            if running_task_id == task.task_id:
                continue
                
            running_task = self.active_tasks[running_task_id]
            running_config = self.agent_configs[running_task.agent_role]
            
            # Get files running task is modifying
            running_files = set()
            for pattern in running_config.output_patterns:
                running_files.add(pattern)
            
            # Check for overlap
            overlapping_files = task_files.intersection(running_files)
            if overlapping_files:
                conflicts.extend(list(overlapping_files))
        
        return conflicts
    
    async def _execute_task(self, task: AgentTask) -> AgentTask:
        """Execute individual agent task"""
        logger.info(f"Starting execution of task {task.task_id} with agent {task.agent_role}")
        
        task.status = AgentStatus.RUNNING
        task.start_time = time.time()
        
        try:
            # Check for file conflicts
            conflicts = self._check_file_conflicts(task)
            agent_config = self.agent_configs[task.agent_role]
            
            if conflicts and agent_config.execution_context.get("requires_isolation", False):
                # Wait for file locks
                for file_path in conflicts:
                    async with self.conflict_resolver.acquire_file_lock(file_path):
                        logger.debug(f"Acquired lock for {file_path}")
            
            # Execute real agent through HTTP API
            agent_result = await self._execute_real_agent(task)
            
            task.status = AgentStatus.COMPLETED
            task.result = {
                "status": "success",
                "agent_output": agent_result.get("result"),
                "metadata": agent_result.get("metadata", {}),
                "files_modified": agent_config.output_patterns
            }
            
        except Exception as e:
            task.status = AgentStatus.FAILED
            task.error_message = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            task.end_time = time.time()
            duration = task.end_time - task.start_time if task.start_time else 0
            logger.info(f"Task {task.task_id} completed in {duration:.2f} seconds")
        
        return task
    
    async def _execute_real_agent(self, task: AgentTask) -> Dict:
        """Execute real agent through HTTP API call to agents service"""
        logger.info(f"Executing real agent {task.agent_role} for task {task.task_id}")
        
        # Get agent configuration
        agent_config = self.agent_configs[task.agent_role]
        
        # Load PRP template if available
        prp_content = await self._load_prp_template(agent_config.prp_template)
        
        # Interpolate PRP template with task data
        formatted_prompt = await self._format_prp_prompt(prp_content, task, agent_config)
        
        # Prepare agent request with proper dependencies
        agent_type = self._map_agent_role_to_type(task.agent_role)
        
        # Create context with required fields based on agent type
        context = {
            "task_id": task.task_id,
            "agent_role": task.agent_role,
            "input_data": task.input_data,
            "file_patterns": task.file_patterns,
            # Common dependency fields
            "project_id": task.input_data.get("project_id", "archon-plus"),
            "user_id": "system",
            "trace_id": f"task_{task.task_id}",
            "request_id": f"req_{task.task_id}"
        }
        
        # Add agent-specific dependency fields
        if agent_type == "rag":
            context.update({
                "source_filter": None,
                "match_count": 5
            })
        elif agent_type == "document":
            context.update({
                "current_document_id": f"doc_{task.task_id}",
                "document_type": "technical_documentation",
                "content": task.description,
                "metadata": {
                    "task_type": "documentation",
                    "project": task.input_data.get("project_name", "archon-plus"),
                    "component": task.agent_role
                }
            })
        
        agent_request = {
            "agent_type": agent_type,
            "prompt": formatted_prompt,
            "context": context,
            "options": {
                "timeout_minutes": task.timeout_minutes,
                "priority": task.priority
            }
        }
        
        # Call agents service
        try:
            agents_port = os.getenv("ARCHON_AGENTS_PORT", "8052")
            agents_url = f"http://localhost:{agents_port}"
            
            async with httpx.AsyncClient(timeout=task.timeout_minutes * 60) as client:
                response = await client.post(
                    f"{agents_url}/agents/run",
                    json=agent_request
                )
                
                if response.status_code != 200:
                    raise Exception(f"Agent service error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if not result.get("success", False):
                    raise Exception(f"Agent execution failed: {result.get('error', 'Unknown error')}")
                
                logger.info(f"Agent {task.agent_role} completed successfully for task {task.task_id}")
                return result
                
        except httpx.TimeoutException:
            raise Exception(f"Agent execution timed out after {task.timeout_minutes} minutes")
        except Exception as e:
            logger.error(f"Agent execution failed for {task.task_id}: {str(e)}")
            raise
    
    def _resolve_dependencies(self) -> List[AgentTask]:
        """Get tasks ready for execution (dependencies satisfied)"""
        ready_tasks = []
        
        for task in self.task_queue:
            if task.status != AgentStatus.QUEUED:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = True
            for dep_task_id in task.dependencies:
                dep_task = self.active_tasks.get(dep_task_id)
                if not dep_task or dep_task.status != AgentStatus.COMPLETED:
                    deps_satisfied = False
                    break
            
            if deps_satisfied:
                ready_tasks.append(task)
            else:
                task.status = AgentStatus.BLOCKED
        
        # Sort by priority (1=high, 2=medium, 3=low)
        ready_tasks.sort(key=lambda t: t.priority)
        
        return ready_tasks
    
    async def execute_batch(self, timeout_minutes: int = 60) -> Dict[str, List[AgentTask]]:
        """Execute all queued tasks in parallel with dependency resolution"""
        logger.info(f"Starting batch execution of {len(self.task_queue)} tasks")
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        results = {
            "completed": [],
            "failed": [],
            "timeout": []
        }
        
        while self.task_queue and (time.time() - start_time) < timeout_seconds:
            # Get tasks ready for execution
            ready_tasks = self._resolve_dependencies()
            
            if not ready_tasks:
                # No tasks ready - check if we're waiting for running tasks
                if self.running_futures:
                    await asyncio.sleep(1)
                    continue
                else:
                    # No running tasks and no ready tasks - deadlock or completion
                    break
            
            # Launch tasks up to concurrency limit
            available_slots = self.max_concurrent - len(self.running_futures)
            tasks_to_launch = ready_tasks[:available_slots]
            
            for task in tasks_to_launch:
                self.task_queue.remove(task)
                future = asyncio.create_task(self._execute_task(task))
                self.running_futures[task.task_id] = future
                logger.info(f"Launched task {task.task_id}")
            
            # Wait for at least one task to complete
            if self.running_futures:
                done_tasks = []
                for task_id, future in list(self.running_futures.items()):
                    if future.done():
                        done_tasks.append(task_id)
                
                if not done_tasks:
                    # Wait for any task to complete
                    done, pending = await asyncio.wait(
                        self.running_futures.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=1.0
                    )
                    
                    for future in done:
                        # Find task_id for this future
                        for task_id, f in list(self.running_futures.items()):
                            if f == future:
                                done_tasks.append(task_id)
                                break
                
                # Process completed tasks
                for task_id in done_tasks:
                    future = self.running_futures.pop(task_id)
                    try:
                        completed_task = await future
                        if completed_task.status == AgentStatus.COMPLETED:
                            results["completed"].append(completed_task)
                            self.completed_tasks.append(completed_task)
                        else:
                            results["failed"].append(completed_task)
                        
                        # Remove from active tasks
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
                            
                    except Exception as e:
                        logger.error(f"Future for task {task_id} failed: {e}")
                        task = self.active_tasks.get(task_id)
                        if task:
                            task.status = AgentStatus.FAILED
                            task.error_message = str(e)
                            results["failed"].append(task)
        
        # Handle timeout
        remaining_time = timeout_seconds - (time.time() - start_time)
        if remaining_time <= 0:
            logger.warning("Batch execution timed out")
            for task in self.task_queue:
                if task.status == AgentStatus.QUEUED:
                    task.status = AgentStatus.FAILED
                    task.error_message = "Execution timeout"
                    results["timeout"].append(task)
        
        # Wait for remaining running tasks and process their results
        if self.running_futures:
            logger.info(f"Waiting for {len(self.running_futures)} remaining tasks")
            
            # Process remaining futures
            remaining_tasks = list(self.running_futures.items())
            for task_id, future in remaining_tasks:
                try:
                    completed_task = await asyncio.wait_for(future, timeout=30)
                    
                    # Process the completed task
                    if completed_task.status == AgentStatus.COMPLETED:
                        results["completed"].append(completed_task)
                        self.completed_tasks.append(completed_task)
                    else:
                        results["failed"].append(completed_task)
                    
                    # Remove from active tasks and running futures
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    if task_id in self.running_futures:
                        del self.running_futures[task_id]
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task_id} timed out during cleanup")
                    task = self.active_tasks.get(task_id)
                    if task:
                        task.status = AgentStatus.FAILED
                        task.error_message = "Cleanup timeout"
                        results["timeout"].append(task)
                except Exception as e:
                    logger.error(f"Error processing completed task {task_id}: {e}")
                    task = self.active_tasks.get(task_id)
                    if task:
                        task.status = AgentStatus.FAILED
                        task.error_message = str(e)
                        results["failed"].append(task)
        
        total_time = time.time() - start_time
        logger.info(f"Batch execution completed in {total_time:.2f} seconds")
        logger.info(f"Results: {len(results['completed'])} completed, {len(results['failed'])} failed, {len(results['timeout'])} timeout")
        
        return results
    
    def get_status(self) -> Dict:
        """Get current executor status"""
        return {
            "total_agents": len(self.agent_configs),
            "queued_tasks": len([t for t in self.task_queue if t.status == AgentStatus.QUEUED]),
            "running_tasks": len(self.running_futures),
            "completed_tasks": len(self.completed_tasks),
            "blocked_tasks": len([t for t in self.task_queue if t.status == AgentStatus.BLOCKED]),
            "max_concurrent": self.max_concurrent,
            "conflict_strategy": self.conflict_resolver.strategy.value
        }
    
    def shutdown(self):
        """Graceful shutdown of executor"""
        logger.info("Shutting down ParallelExecutor")
        
        # Cancel running futures
        for future in self.running_futures.values():
            future.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("ParallelExecutor shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize executor
        executor = ParallelExecutor(max_concurrent=4)
        
        # Create sample tasks
        tasks = [
            AgentTask(
                task_id="task_1",
                agent_role="python_backend_coder",
                description="Implement user authentication API",
                input_data={"endpoint": "/auth/login"},
                priority=1
            ),
            AgentTask(
                task_id="task_2", 
                agent_role="typescript_frontend_agent",
                description="Create login form component",
                input_data={"component": "LoginForm"},
                dependencies=["task_1"],
                priority=1
            ),
            AgentTask(
                task_id="task_3",
                agent_role="test_generator",
                description="Generate tests for auth system",
                input_data={"test_target": "auth"},
                dependencies=["task_1", "task_2"],
                priority=2
            )
        ]
        
        # Add tasks to executor
        for task in tasks:
            executor.add_task(task)
        
        # Execute batch
        results = await executor.execute_batch(timeout_minutes=10)
        
        # Print results
        print(f"\nExecution Results:")
        print(f"Completed: {len(results['completed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Timeout: {len(results['timeout'])}")
        
        for task in results['completed']:
            print(f"  ✓ {task.task_id}: {task.description}")
        
        for task in results['failed']:
            print(f"  ✗ {task.task_id}: {task.error_message}")
        
        # Shutdown
        executor.shutdown()
    
    # Run example
    asyncio.run(main())