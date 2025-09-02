"""
Parallel Execution Engine for Phase 2 Meta-Agent Orchestration
Implements async parallel task execution with worker pool pattern
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .parallel_executor import AgentTask, AgentStatus

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    agent_role: str
    status: AgentStatus
    output: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Aggregated results from batch execution"""
    total_tasks: int
    completed: List[TaskResult]
    failed: List[TaskResult]
    timeout: List[TaskResult]
    total_execution_time: float
    parallel_efficiency: float  # % of time saved vs sequential
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelExecutionEngine:
    """
    Manages concurrent task execution across multiple agents.
    Replaces sequential execution with async parallel processing.
    """
    
    def __init__(self, 
                 max_workers: int = 10,
                 task_timeout: float = 60.0,
                 enable_progress_tracking: bool = True):
        """
        Initialize parallel execution engine.
        
        Args:
            max_workers: Maximum concurrent workers
            task_timeout: Default timeout per task in seconds
            enable_progress_tracking: Enable WebSocket progress updates
        """
        self.max_workers = max_workers
        self.task_timeout = task_timeout
        self.enable_progress_tracking = enable_progress_tracking
        
        # Queues and state
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Worker management
        self.worker_pool: List[asyncio.Task] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Execution tracking
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Synchronization
        self._shutdown = False
        self._workers_started = False
        self._lock = asyncio.Lock()
        
        # Service connector for real execution (will be set by coordinator)
        self.service_connector = None
        
        logger.info(f"Initialized ParallelExecutionEngine with {max_workers} workers")
    
    async def start_workers(self):
        """Start worker pool for processing tasks"""
        if self._workers_started:
            return
            
        async with self._lock:
            if not self._workers_started:
                for i in range(self.max_workers):
                    worker = asyncio.create_task(self._worker(f"worker-{i}"))
                    self.worker_pool.append(worker)
                self._workers_started = True
                logger.info(f"Started {self.max_workers} worker tasks")
    
    async def stop_workers(self):
        """Stop all workers gracefully"""
        self._shutdown = True
        
        # Cancel all workers
        for worker in self.worker_pool:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_pool, return_exceptions=True)
        self.worker_pool.clear()
        self._workers_started = False
        logger.info("Stopped all worker tasks")
    
    async def submit_task(self, task: AgentTask, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        if not task.task_id:
            task.task_id = str(uuid.uuid4())
        
        # Ensure workers are started
        if not self._workers_started:
            await self.start_workers()
        
        # Add to appropriate queue
        if priority != TaskPriority.NORMAL:
            await self.priority_queue.put((priority.value, task))
        else:
            await self.task_queue.put(task)
        
        self.active_tasks[task.task_id] = task
        self.total_tasks_submitted += 1
        
        logger.debug(f"Submitted task {task.task_id} with priority {priority}")
        return task.task_id
    
    async def execute_batch(self, tasks: List[AgentTask], timeout_minutes: float = 5.0) -> BatchResult:
        """
        Execute multiple tasks in parallel.
        
        Args:
            tasks: List of tasks to execute
            timeout_minutes: Overall timeout for batch
            
        Returns:
            Aggregated batch results
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        
        logger.info(f"Starting batch execution of {len(tasks)} tasks (batch_id: {batch_id})")
        
        # Ensure workers are started
        if not self._workers_started:
            await self.start_workers()
        
        # Submit all tasks
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for completion or timeout
        timeout_seconds = timeout_minutes * 60
        completed = []
        failed = []
        timeout_tasks = []
        
        try:
            # Wait for all tasks with timeout
            await asyncio.wait_for(
                self._wait_for_tasks(task_ids, completed, failed),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Batch execution timed out after {timeout_minutes} minutes")
            # Collect whatever completed
            for task_id in task_ids:
                if task_id in self.completed_tasks:
                    result = self.completed_tasks[task_id]
                    if result.status == AgentStatus.COMPLETED:
                        completed.append(result)
                    else:
                        failed.append(result)
                else:
                    # Task didn't complete - mark as timeout
                    timeout_tasks.append(TaskResult(
                        task_id=task_id,
                        agent_role=self.active_tasks.get(task_id, AgentTask(task_id=task_id)).agent_role,
                        status=AgentStatus.TIMEOUT,
                        error_message="Task timed out",
                        execution_time=timeout_seconds
                    ))
        
        # Calculate metrics
        total_time = time.time() - start_time
        sequential_estimate = len(tasks) * 20  # Estimate 20s per task sequentially
        parallel_efficiency = max(0, (sequential_estimate - total_time) / sequential_estimate)
        
        return BatchResult(
            total_tasks=len(tasks),
            completed=completed,
            failed=failed,
            timeout=timeout_tasks,
            total_execution_time=total_time,
            parallel_efficiency=parallel_efficiency,
            metadata={
                "batch_id": batch_id,
                "worker_count": self.max_workers,
                "timeout_minutes": timeout_minutes
            }
        )
    
    async def _wait_for_tasks(self, task_ids: List[str], completed: List, failed: List):
        """Wait for specific tasks to complete"""
        remaining = set(task_ids)
        
        while remaining:
            # Check completed tasks
            for task_id in list(remaining):
                if task_id in self.completed_tasks:
                    result = self.completed_tasks[task_id]
                    if result.status == AgentStatus.COMPLETED:
                        completed.append(result)
                    else:
                        failed.append(result)
                    remaining.remove(task_id)
            
            if remaining:
                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
    
    async def _worker(self, worker_id: str):
        """
        Worker coroutine for processing tasks.
        
        Args:
            worker_id: Unique worker identifier
        """
        logger.debug(f"Worker {worker_id} started")
        
        while not self._shutdown:
            task = None
            try:
                # Try priority queue first
                if not self.priority_queue.empty():
                    priority, task = await asyncio.wait_for(
                        self.priority_queue.get(), 
                        timeout=0.1
                    )
                # Then regular queue
                elif not self.task_queue.empty():
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=0.1
                    )
                else:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                if task:
                    # Execute the task
                    result = await self._execute_task(task, worker_id)
                    
                    # Store result
                    self.completed_tasks[task.task_id] = result
                    
                    # Update counters
                    if result.status == AgentStatus.COMPLETED:
                        self.total_tasks_completed += 1
                    else:
                        self.total_tasks_failed += 1
                    
                    # Remove from active tasks
                    self.active_tasks.pop(task.task_id, None)
                    
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except asyncio.TimeoutError:
                continue  # No task available, continue loop
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                if task and task.task_id:
                    # Mark task as failed
                    self.completed_tasks[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        agent_role=task.agent_role,
                        status=AgentStatus.FAILED,
                        error_message=str(e)
                    )
                    self.active_tasks.pop(task.task_id, None)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, task: AgentTask, worker_id: str) -> TaskResult:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            worker_id: ID of worker executing the task
            
        Returns:
            Task execution result
        """
        start_time = time.time()
        logger.info(f"Worker {worker_id} executing task {task.task_id} ({task.agent_role})")
        
        try:
            # Check if we have a service connector set
            if hasattr(self, 'service_connector') and self.service_connector:
                # Use the service connector for real execution
                task.start_time = datetime.now()
                
                # Get routed agent from task metadata
                agent_id = task.metadata.get("routed_to") if hasattr(task, 'metadata') and task.metadata else task.agent_role
                
                result = await asyncio.wait_for(
                    self.service_connector.execute_task(agent_id, task),
                    timeout=self.task_timeout
                )
                task.end_time = datetime.now()
                
                execution_time = time.time() - start_time
                
                return TaskResult(
                    task_id=task.task_id,
                    agent_role=task.agent_role,
                    status=AgentStatus.COMPLETED if result.get("status") == "completed" else AgentStatus.FAILED,
                    output=result.get("output", ""),
                    error_message=result.get("error", ""),
                    execution_time=execution_time,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    metadata={"worker_id": worker_id, "result": result}
                )
            else:
                # No service connector - this is an error, not a fallback!
                raise RuntimeError(
                    f"No service connector available for task {task.task_id}. "
                    "Real execution required - no simulation allowed!"
                )
            
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.task_id,
                agent_role=task.agent_role,
                status=AgentStatus.TIMEOUT,
                error_message=f"Task timed out after {self.task_timeout}s",
                execution_time=self.task_timeout
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_role=task.agent_role,
                status=AgentStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_result(self, task_id: str, timeout: float = 60.0) -> Optional[TaskResult]:
        """
        Get result for a specific task.
        
        Args:
            task_id: Task ID to get result for
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            await asyncio.sleep(0.1)
        
        return None
    
    async def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate all results from completed tasks.
        
        Returns:
            Aggregated statistics and results
        """
        completed = [r for r in self.completed_tasks.values() if r.status == AgentStatus.COMPLETED]
        failed = [r for r in self.completed_tasks.values() if r.status == AgentStatus.FAILED]
        timeout = [r for r in self.completed_tasks.values() if r.status == AgentStatus.TIMEOUT]
        
        total_execution_time = sum(r.execution_time for r in self.completed_tasks.values())
        avg_execution_time = total_execution_time / len(self.completed_tasks) if self.completed_tasks else 0
        
        return {
            "total_submitted": self.total_tasks_submitted,
            "total_completed": len(completed),
            "total_failed": len(failed),
            "total_timeout": len(timeout),
            "success_rate": len(completed) / self.total_tasks_submitted if self.total_tasks_submitted > 0 else 0,
            "average_execution_time": avg_execution_time,
            "total_execution_time": total_execution_time,
            "active_tasks": len(self.active_tasks),
            "completed_results": completed,
            "failed_results": failed,
            "timeout_results": timeout
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "workers": self.max_workers,
            "workers_active": len([w for w in self.worker_pool if not w.done()]),
            "queue_depth": self.task_queue.qsize() + self.priority_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_submitted": self.total_tasks_submitted,
            "total_completed": self.total_tasks_completed,
            "total_failed": self.total_tasks_failed,
            "shutdown": self._shutdown
        }