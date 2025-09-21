"""
Workflow Executor for Parallel and Sequential Task Execution
Manages the actual execution of workflow tasks with advanced features
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import json
import traceback
from collections import defaultdict
import inspect
import os
import sys

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Task execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAMING = "streaming"
    DISTRIBUTED = "distributed"


class ExecutionStatus(Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"
    SKIPPED = "skipped"


class ResourceType(Enum):
    """Resource types for allocation"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


@dataclass
class ResourceRequirement:
    """Resource requirements for task execution"""
    resource_type: ResourceType
    amount: float
    unit: str
    exclusive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.resource_type.value,
            "amount": self.amount,
            "unit": self.unit,
            "exclusive": self.exclusive
        }


@dataclass
class ExecutionContext:
    """Context for task execution"""
    task_id: str
    workflow_id: str
    attempt: int = 1
    max_attempts: int = 3
    timeout: Optional[float] = None
    resources: List[ResourceRequirement] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_env(self) -> Dict[str, str]:
        """Get combined environment variables"""
        env = os.environ.copy()
        env.update(self.environment)
        return env


@dataclass
class ExecutionResult:
    """Result of task execution"""
    task_id: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    resources_used: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "resources_used": self.resources_used,
            "artifacts": self.artifacts
        }


@dataclass
class TaskExecutor:
    """Individual task executor"""
    executor_id: str
    name: str
    function: Callable
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_parallel: int = 10
    batch_size: int = 100
    timeout: Optional[float] = 300.0
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the task"""
        result = ExecutionResult(
            task_id=context.task_id,
            status=ExecutionStatus.PENDING
        )
        
        try:
            result.start_time = datetime.now()
            result.status = ExecutionStatus.RUNNING
            
            # Apply timeout if specified
            if context.timeout or self.timeout:
                timeout = context.timeout or self.timeout
                output = await asyncio.wait_for(
                    self._run_function(context),
                    timeout=timeout
                )
            else:
                output = await self._run_function(context)
                
            result.output = output
            result.status = ExecutionStatus.SUCCESS
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = f"Task timed out after {timeout} seconds"
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.logs.append(traceback.format_exc())
            
        finally:
            result.end_time = datetime.now()
            if result.start_time:
                result.duration = (result.end_time - result.start_time).total_seconds()
                
        return result
        
    async def _run_function(self, context: ExecutionContext) -> Any:
        """Run the actual function"""
        # Check if function is async
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**context.parameters)
        else:
            # Run in thread pool for blocking functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.function,
                **context.parameters
            )


class WorkflowExecutor:
    """Main workflow executor managing task execution"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers // 2)
        self.executors: Dict[str, TaskExecutor] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.execution_history: Dict[str, List[ExecutionResult]] = defaultdict(list)
        self.resource_manager = ResourceManager()
        self.checkpoint_manager = CheckpointManager()
        self.metrics: Dict[str, int] = defaultdict(int)
        
    def register_executor(self, executor: TaskExecutor) -> None:
        """Register a task executor"""
        self.executors[executor.executor_id] = executor
        logger.info(f"Registered executor {executor.name}")
        
    async def execute_workflow(self, workflow_id: str, tasks: List[Dict[str, Any]],
                              mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> Dict[str, ExecutionResult]:
        """Execute a complete workflow"""
        logger.info(f"Starting workflow {workflow_id} with {len(tasks)} tasks in {mode.value} mode")
        
        results = {}
        
        if mode == ExecutionMode.SEQUENTIAL:
            results = await self._execute_sequential(workflow_id, tasks)
        elif mode == ExecutionMode.PARALLEL:
            results = await self._execute_parallel(workflow_id, tasks)
        elif mode == ExecutionMode.BATCH:
            results = await self._execute_batch(workflow_id, tasks)
        elif mode == ExecutionMode.STREAMING:
            results = await self._execute_streaming(workflow_id, tasks)
        elif mode == ExecutionMode.DISTRIBUTED:
            results = await self._execute_distributed(workflow_id, tasks)
            
        self.metrics["workflows_executed"] += 1
        return results
        
    async def _execute_sequential(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, ExecutionResult]:
        """Execute tasks sequentially"""
        results = {}
        
        for task in tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            
            # Check dependencies
            deps = task.get("dependencies", [])
            if not self._check_dependencies(deps, results):
                result = ExecutionResult(
                    task_id=task_id,
                    status=ExecutionStatus.SKIPPED,
                    error="Dependencies not met"
                )
                results[task_id] = result
                continue
                
            # Execute task
            result = await self._execute_task(workflow_id, task)
            results[task_id] = result
            
            # Stop on failure if required
            if result.status == ExecutionStatus.FAILED and task.get("stop_on_failure", False):
                logger.warning(f"Stopping workflow due to task {task_id} failure")
                break
                
        return results
        
    async def _execute_parallel(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, ExecutionResult]:
        """Execute tasks in parallel"""
        # Group tasks by dependencies
        task_groups = self._group_tasks_by_dependencies(tasks)
        results = {}
        
        for group in task_groups:
            # Execute each group in parallel
            group_tasks = []
            for task in group:
                task_future = asyncio.create_task(
                    self._execute_task(workflow_id, task)
                )
                group_tasks.append((task.get("id", str(uuid.uuid4())), task_future))
                
            # Wait for group to complete
            for task_id, task_future in group_tasks:
                result = await task_future
                results[task_id] = result
                
        return results
        
    async def _execute_batch(self, workflow_id: str, tasks: List[Dict[str, Any]],
                            batch_size: int = 10) -> Dict[str, ExecutionResult]:
        """Execute tasks in batches"""
        results = {}
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await self._execute_parallel(workflow_id, batch)
            results.update(batch_results)
            
            # Add delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(0.1)
                
        return results
        
    async def _execute_streaming(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, ExecutionResult]:
        """Execute tasks in streaming mode with pipeline"""
        results = {}
        pipeline_data = None
        
        for task in tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            
            # Pass output from previous task as input
            if pipeline_data is not None:
                task["parameters"] = task.get("parameters", {})
                task["parameters"]["input"] = pipeline_data
                
            result = await self._execute_task(workflow_id, task)
            results[task_id] = result
            
            if result.status == ExecutionStatus.SUCCESS:
                pipeline_data = result.output
            else:
                # Break pipeline on failure
                break
                
        return results
        
    async def _execute_distributed(self, workflow_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, ExecutionResult]:
        """Execute tasks in distributed mode across workers"""
        # This is a simplified version - real distributed execution would use
        # a message queue or distributed task system
        
        # Distribute tasks across available workers
        worker_count = self.max_workers
        task_queues = [[] for _ in range(worker_count)]
        
        for i, task in enumerate(tasks):
            worker_id = i % worker_count
            task_queues[worker_id].append(task)
            
        # Execute on each worker
        worker_futures = []
        for worker_id, worker_tasks in enumerate(task_queues):
            if worker_tasks:
                future = asyncio.create_task(
                    self._execute_worker_tasks(workflow_id, worker_id, worker_tasks)
                )
                worker_futures.append(future)
                
        # Collect results
        results = {}
        for future in worker_futures:
            worker_results = await future
            results.update(worker_results)
            
        return results
        
    async def _execute_worker_tasks(self, workflow_id: str, worker_id: int,
                                   tasks: List[Dict[str, Any]]) -> Dict[str, ExecutionResult]:
        """Execute tasks assigned to a specific worker"""
        logger.info(f"Worker {worker_id} executing {len(tasks)} tasks")
        results = {}
        
        for task in tasks:
            task_id = task.get("id", str(uuid.uuid4()))
            result = await self._execute_task(workflow_id, task)
            results[task_id] = result
            
        return results
        
    async def _execute_task(self, workflow_id: str, task: Dict[str, Any]) -> ExecutionResult:
        """Execute a single task"""
        task_id = task.get("id", str(uuid.uuid4()))
        executor_id = task.get("executor_id")
        
        if not executor_id or executor_id not in self.executors:
            return ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.FAILED,
                error=f"Executor {executor_id} not found"
            )
            
        executor = self.executors[executor_id]
        
        # Create execution context
        context = ExecutionContext(
            task_id=task_id,
            workflow_id=workflow_id,
            parameters=task.get("parameters", {}),
            environment=task.get("environment", {}),
            dependencies=task.get("dependencies", []),
            timeout=task.get("timeout"),
            max_attempts=task.get("max_attempts", 3)
        )
        
        # Check resource availability
        resources = task.get("resources", [])
        if resources and not await self.resource_manager.allocate(task_id, resources):
            return ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.FAILED,
                error="Required resources not available"
            )
            
        try:
            # Load checkpoint if exists
            checkpoint = await self.checkpoint_manager.load(task_id)
            if checkpoint:
                context.artifacts = checkpoint.get("artifacts", {})
                context.attempt = checkpoint.get("attempt", 1) + 1
                
            # Execute with retry logic
            result = None
            for attempt in range(context.max_attempts):
                context.attempt = attempt + 1
                result = await executor.execute(context)
                
                if result.status == ExecutionStatus.SUCCESS:
                    break
                elif result.status == ExecutionStatus.TIMEOUT:
                    # Don't retry timeouts
                    break
                elif attempt < context.max_attempts - 1:
                    # Exponential backoff for retries
                    await asyncio.sleep(2 ** attempt)
                    
            # Save checkpoint
            if result and result.status == ExecutionStatus.SUCCESS:
                await self.checkpoint_manager.save(task_id, {
                    "artifacts": result.artifacts,
                    "output": result.output,
                    "attempt": context.attempt
                })
                
            # Record execution
            self.execution_history[workflow_id].append(result)
            self.metrics[f"task_{result.status.value}"] += 1
            
            return result
            
        finally:
            # Release resources
            if resources:
                await self.resource_manager.release(task_id)
                
    def _check_dependencies(self, dependencies: List[str], results: Dict[str, ExecutionResult]) -> bool:
        """Check if task dependencies are met"""
        for dep in dependencies:
            if dep not in results:
                return False
            if results[dep].status != ExecutionStatus.SUCCESS:
                return False
        return True
        
    def _group_tasks_by_dependencies(self, tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group tasks that can be executed in parallel"""
        groups = []
        remaining = tasks.copy()
        completed = set()
        
        while remaining:
            group = []
            for task in remaining[:]:
                task_id = task.get("id", str(uuid.uuid4()))
                deps = set(task.get("dependencies", []))
                
                # Check if all dependencies are completed
                if deps.issubset(completed):
                    group.append(task)
                    remaining.remove(task)
                    
            if not group:
                # No tasks can be executed - circular dependency
                logger.error("Circular dependency detected in workflow")
                break
                
            groups.append(group)
            for task in group:
                completed.add(task.get("id", str(uuid.uuid4())))
                
        return groups
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.cancel()
            del self.running_tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        return False
        
    async def cancel_workflow(self, workflow_id: str) -> int:
        """Cancel all tasks in a workflow"""
        cancelled = 0
        for task_id, task in list(self.running_tasks.items()):
            if workflow_id in task_id:
                if await self.cancel_task(task_id):
                    cancelled += 1
        return cancelled
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        return {
            "total_executed": sum(self.metrics.values()),
            "success": self.metrics.get("task_success", 0),
            "failed": self.metrics.get("task_failed", 0),
            "timeout": self.metrics.get("task_timeout", 0),
            "cancelled": self.metrics.get("task_cancelled", 0),
            "running": len(self.running_tasks),
            "workflows_executed": self.metrics.get("workflows_executed", 0)
        }
        
    def get_execution_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get execution history for a workflow"""
        history = self.execution_history.get(workflow_id, [])
        return [result.to_dict() for result in history]
        
    async def cleanup(self) -> None:
        """Clean up resources"""
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
            
        # Shutdown pools
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        
        logger.info("Workflow executor cleaned up")


class ResourceManager:
    """Manages resource allocation for task execution"""
    
    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.DISK: 100.0,
            ResourceType.NETWORK: 100.0
        }
        self.allocated: Dict[str, List[ResourceRequirement]] = {}
        self.lock = asyncio.Lock()
        
    async def allocate(self, task_id: str, requirements: List[ResourceRequirement]) -> bool:
        """Allocate resources for a task"""
        async with self.lock:
            # Check availability
            for req in requirements:
                if req.resource_type in self.available_resources:
                    if self.available_resources[req.resource_type] < req.amount:
                        return False
                        
            # Allocate resources
            for req in requirements:
                if req.resource_type in self.available_resources:
                    self.available_resources[req.resource_type] -= req.amount
                    
            self.allocated[task_id] = requirements
            return True
            
    async def release(self, task_id: str) -> None:
        """Release resources allocated to a task"""
        async with self.lock:
            if task_id in self.allocated:
                for req in self.allocated[task_id]:
                    if req.resource_type in self.available_resources:
                        self.available_resources[req.resource_type] += req.amount
                del self.allocated[task_id]


class CheckpointManager:
    """Manages task checkpoints for recovery"""
    
    def __init__(self, checkpoint_dir: str = "/tmp/workflow_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    async def save(self, task_id: str, data: Dict[str, Any]) -> None:
        """Save checkpoint for a task"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{task_id}.checkpoint")
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {task_id}: {str(e)}")
            
    async def load(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint for a task"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{task_id}.checkpoint")
        
        if not os.path.exists(checkpoint_file):
            return None
            
        try:
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {task_id}: {str(e)}")
            return None
            
    async def delete(self, task_id: str) -> None:
        """Delete checkpoint for a task"""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{task_id}.checkpoint")
        
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
            except Exception as e:
                logger.error(f"Failed to delete checkpoint for {task_id}: {str(e)}")