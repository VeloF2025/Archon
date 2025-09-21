"""
Task Scheduler Module
Advanced task scheduling and execution management
"""

import asyncio
import heapq
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from croniter import croniter
import pytz


class ScheduleType(Enum):
    """Types of schedules"""
    ONCE = "once"
    RECURRING = "recurring"
    CRON = "cron"
    INTERVAL = "interval"
    RATE = "rate"
    EVENT_BASED = "event_based"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class RecurrencePattern(Enum):
    """Recurrence patterns"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class Schedule:
    """Schedule configuration"""
    schedule_type: ScheduleType
    start_time: datetime
    end_time: Optional[datetime] = None
    timezone: str = "UTC"
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    recurrence_pattern: Optional[RecurrencePattern] = None
    recurrence_interval: int = 1
    days_of_week: List[int] = field(default_factory=list)  # 0=Monday, 6=Sunday
    days_of_month: List[int] = field(default_factory=list)
    months: List[int] = field(default_factory=list)
    max_occurrences: Optional[int] = None
    occurrences: int = 0


@dataclass
class ScheduledTask:
    """Scheduled task definition"""
    task_id: str
    name: str
    description: str
    schedule: Schedule
    action: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: Optional[int] = None  # seconds
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Task execution record"""
    execution_id: str
    task_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.RUNNING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """
    Advanced task scheduling and execution system
    """
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.executions: Dict[str, TaskExecution] = {}
        self.execution_history: List[TaskExecution] = []
        self.active_tasks: Set[str] = set()
        
        # Priority queue for scheduled tasks
        self.task_queue: List[Tuple[datetime, str]] = []
        
        # Task dependencies graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Metrics
        self.metrics = {
            "total_tasks": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0
        }
        
        # Start scheduler
        self._running = True
        asyncio.create_task(self._scheduler_loop())
        asyncio.create_task(self._executor_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            try:
                await self._schedule_tasks()
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                print(f"Scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _executor_loop(self):
        """Task executor loop"""
        while self._running:
            try:
                await self._execute_ready_tasks()
                await asyncio.sleep(0.1)  # Check frequently
            except Exception as e:
                print(f"Executor error: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_loop(self):
        """Cleanup old executions"""
        while self._running:
            try:
                cutoff = datetime.now() - timedelta(days=7)
                
                # Clean old execution history
                self.execution_history = [
                    e for e in self.execution_history
                    if e.started_at > cutoff
                ]
                
                # Keep only last 10000 executions
                if len(self.execution_history) > 10000:
                    self.execution_history = self.execution_history[-10000:]
                
                await asyncio.sleep(3600)  # Cleanup hourly
                
            except Exception as e:
                print(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def add_task(self, task: ScheduledTask) -> bool:
        """Add a scheduled task"""
        if task.task_id in self.tasks:
            return False
        
        self.tasks[task.task_id] = task
        
        # Calculate next run time
        task.next_run = self._calculate_next_run(task)
        
        # Add to queue if enabled
        if task.enabled and task.next_run:
            heapq.heappush(self.task_queue, (task.next_run, task.task_id))
        
        # Update dependencies
        if task.dependencies:
            self.dependency_graph[task.task_id] = set(task.dependencies)
        
        self.metrics["total_tasks"] += 1
        return True
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task"""
        if task_id not in self.tasks:
            return False
        
        del self.tasks[task_id]
        
        # Remove from queue
        self.task_queue = [(t, tid) for t, tid in self.task_queue if tid != task_id]
        heapq.heapify(self.task_queue)
        
        # Remove from dependencies
        if task_id in self.dependency_graph:
            del self.dependency_graph[task_id]
        
        # Remove as dependency from other tasks
        for deps in self.dependency_graph.values():
            deps.discard(task_id)
        
        self.metrics["total_tasks"] -= 1
        return True
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update task configuration"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.updated_at = datetime.now()
        
        # Recalculate next run
        task.next_run = self._calculate_next_run(task)
        
        # Update queue
        self.task_queue = [(t, tid) for t, tid in self.task_queue if tid != task_id]
        
        if task.enabled and task.next_run:
            heapq.heappush(self.task_queue, (task.next_run, task.task_id))
        else:
            heapq.heapify(self.task_queue)
        
        return True
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task"""
        return self.update_task(task_id, {"enabled": True})
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task"""
        return self.update_task(task_id, {"enabled": False})
    
    async def execute_task_now(self, task_id: str) -> str:
        """Execute a task immediately"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        return await self._execute_task(task)
    
    async def _schedule_tasks(self):
        """Schedule tasks for execution"""
        now = datetime.now(pytz.UTC)
        
        for task in self.tasks.values():
            if not task.enabled:
                continue
            
            # Calculate next run if not set
            if not task.next_run:
                task.next_run = self._calculate_next_run(task)
                if task.next_run:
                    heapq.heappush(self.task_queue, (task.next_run, task.task_id))
    
    async def _execute_ready_tasks(self):
        """Execute tasks that are ready"""
        now = datetime.now(pytz.UTC)
        
        while self.task_queue and self.task_queue[0][0] <= now:
            next_run, task_id = heapq.heappop(self.task_queue)
            
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            if not task.enabled:
                continue
            
            # Check dependencies
            if not self._check_dependencies(task_id):
                # Reschedule for later
                heapq.heappush(self.task_queue, 
                             (now + timedelta(seconds=60), task_id))
                continue
            
            # Execute task
            asyncio.create_task(self._execute_task_async(task))
            
            # Schedule next occurrence
            if task.schedule.schedule_type != ScheduleType.ONCE:
                task.next_run = self._calculate_next_run(task)
                if task.next_run:
                    heapq.heappush(self.task_queue, (task.next_run, task.task_id))
    
    async def _execute_task_async(self, task: ScheduledTask):
        """Execute task asynchronously"""
        execution_id = await self._execute_task(task)
        
        # Wait for completion and handle result
        execution = self.executions.get(execution_id)
        if execution:
            # Trigger events
            if execution.status == TaskStatus.COMPLETED:
                await self._trigger_event("task_completed", {
                    "task_id": task.task_id,
                    "execution_id": execution_id,
                    "result": execution.result
                })
            elif execution.status == TaskStatus.FAILED:
                await self._trigger_event("task_failed", {
                    "task_id": task.task_id,
                    "execution_id": execution_id,
                    "error": execution.error
                })
    
    async def _execute_task(self, task: ScheduledTask) -> str:
        """Execute a task"""
        import uuid
        
        execution_id = str(uuid.uuid4())
        
        execution = TaskExecution(
            execution_id=execution_id,
            task_id=task.task_id,
            started_at=datetime.now()
        )
        
        self.executions[execution_id] = execution
        self.active_tasks.add(task.task_id)
        
        try:
            # Apply timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    task.action(**task.parameters),
                    timeout=task.timeout
                )
            else:
                result = await task.action(**task.parameters)
            
            execution.status = TaskStatus.COMPLETED
            execution.result = result
            
            # Update task
            task.last_run = execution.started_at
            task.schedule.occurrences += 1
            
            # Update metrics
            self.metrics["successful_executions"] += 1
            
        except asyncio.TimeoutError:
            execution.status = TaskStatus.FAILED
            execution.error = "Task execution timeout"
            
            # Retry if configured
            if execution.retry_count < task.max_retries:
                await self._retry_task(task, execution)
            else:
                self.metrics["failed_executions"] += 1
                
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error = str(e)
            
            # Retry if configured
            if execution.retry_count < task.max_retries:
                await self._retry_task(task, execution)
            else:
                self.metrics["failed_executions"] += 1
        
        finally:
            execution.completed_at = datetime.now()
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            self.active_tasks.discard(task.task_id)
            self.execution_history.append(execution)
            
            # Update metrics
            self.metrics["total_executions"] += 1
            self._update_average_duration(execution.duration_seconds)
        
        return execution_id
    
    async def _retry_task(self, task: ScheduledTask, execution: TaskExecution):
        """Retry a failed task"""
        execution.retry_count += 1
        execution.status = TaskStatus.RETRYING
        
        # Wait before retry
        await asyncio.sleep(task.retry_delay * execution.retry_count)
        
        try:
            if task.timeout:
                result = await asyncio.wait_for(
                    task.action(**task.parameters),
                    timeout=task.timeout
                )
            else:
                result = await task.action(**task.parameters)
            
            execution.status = TaskStatus.COMPLETED
            execution.result = result
            self.metrics["successful_executions"] += 1
            
        except Exception as e:
            if execution.retry_count >= task.max_retries:
                execution.status = TaskStatus.FAILED
                execution.error = f"Max retries exceeded: {str(e)}"
                self.metrics["failed_executions"] += 1
            else:
                # Retry again
                await self._retry_task(task, execution)
    
    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate next run time for a task"""
        schedule = task.schedule
        now = datetime.now(pytz.timezone(schedule.timezone))
        
        # Check if schedule has ended
        if schedule.end_time and now >= schedule.end_time:
            return None
        
        # Check max occurrences
        if schedule.max_occurrences and schedule.occurrences >= schedule.max_occurrences:
            return None
        
        if schedule.schedule_type == ScheduleType.ONCE:
            if not task.last_run and schedule.start_time > now:
                return schedule.start_time
            return None
            
        elif schedule.schedule_type == ScheduleType.CRON:
            if schedule.cron_expression:
                cron = croniter(schedule.cron_expression, now)
                return cron.get_next(datetime)
            return None
            
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            if schedule.interval_seconds:
                if task.last_run:
                    return task.last_run + timedelta(seconds=schedule.interval_seconds)
                else:
                    return max(schedule.start_time, now)
            return None
            
        elif schedule.schedule_type == ScheduleType.RECURRING:
            return self._calculate_recurring_next_run(schedule, task.last_run, now)
        
        return None
    
    def _calculate_recurring_next_run(self, schedule: Schedule,
                                    last_run: Optional[datetime],
                                    now: datetime) -> Optional[datetime]:
        """Calculate next run for recurring schedule"""
        if schedule.recurrence_pattern == RecurrencePattern.DAILY:
            if last_run:
                next_run = last_run + timedelta(days=schedule.recurrence_interval)
            else:
                next_run = max(schedule.start_time, now)
            
        elif schedule.recurrence_pattern == RecurrencePattern.WEEKLY:
            if schedule.days_of_week:
                # Find next occurrence on specified days
                current_day = now.weekday()
                days_ahead = None
                
                for day in sorted(schedule.days_of_week):
                    if day > current_day:
                        days_ahead = day - current_day
                        break
                
                if days_ahead is None:
                    # Next week
                    days_ahead = 7 - current_day + min(schedule.days_of_week)
                
                next_run = now + timedelta(days=days_ahead)
            else:
                if last_run:
                    next_run = last_run + timedelta(weeks=schedule.recurrence_interval)
                else:
                    next_run = max(schedule.start_time, now)
                    
        elif schedule.recurrence_pattern == RecurrencePattern.MONTHLY:
            if schedule.days_of_month:
                # Find next occurrence on specified days
                next_run = self._next_monthly_occurrence(now, schedule.days_of_month)
            else:
                if last_run:
                    # Add months
                    month = last_run.month + schedule.recurrence_interval
                    year = last_run.year + (month - 1) // 12
                    month = ((month - 1) % 12) + 1
                    next_run = last_run.replace(year=year, month=month)
                else:
                    next_run = max(schedule.start_time, now)
                    
        elif schedule.recurrence_pattern == RecurrencePattern.YEARLY:
            if last_run:
                next_run = last_run.replace(
                    year=last_run.year + schedule.recurrence_interval
                )
            else:
                next_run = max(schedule.start_time, now)
        else:
            next_run = None
        
        return next_run
    
    def _next_monthly_occurrence(self, now: datetime,
                                days_of_month: List[int]) -> datetime:
        """Find next monthly occurrence on specified days"""
        current_day = now.day
        current_month = now.month
        current_year = now.year
        
        # Check remaining days in current month
        for day in sorted(days_of_month):
            if day > current_day:
                try:
                    return now.replace(day=day)
                except ValueError:
                    # Invalid day for this month
                    continue
        
        # Next month
        next_month = current_month + 1
        next_year = current_year
        
        if next_month > 12:
            next_month = 1
            next_year += 1
        
        for day in sorted(days_of_month):
            try:
                return now.replace(year=next_year, month=next_month, day=day)
            except ValueError:
                continue
        
        return now + timedelta(days=30)  # Fallback
    
    def _check_dependencies(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        if task_id not in self.dependency_graph:
            return True
        
        dependencies = self.dependency_graph[task_id]
        
        for dep_task_id in dependencies:
            # Check if dependency task has run successfully recently
            recent_executions = [
                e for e in self.execution_history
                if e.task_id == dep_task_id and
                   e.status == TaskStatus.COMPLETED and
                   (datetime.now() - e.completed_at).total_seconds() < 3600
            ]
            
            if not recent_executions:
                return False
        
        return True
    
    def _update_average_duration(self, duration: float):
        """Update average task duration metric"""
        total = self.metrics["total_executions"]
        if total > 0:
            current_avg = self.metrics["average_duration"]
            self.metrics["average_duration"] = (
                (current_avg * (total - 1) + duration) / total
            )
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    print(f"Event handler error: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[TaskExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def get_upcoming_tasks(self, hours: int = 24) -> List[Tuple[datetime, str]]:
        """Get upcoming tasks in next N hours"""
        cutoff = datetime.now(pytz.UTC) + timedelta(hours=hours)
        
        upcoming = [
            (next_run, task_id)
            for next_run, task_id in self.task_queue
            if next_run <= cutoff
        ]
        
        return sorted(upcoming)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return {
            **self.metrics,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue)
        }
    
    async def shutdown(self):
        """Shutdown the scheduler"""
        self._running = False
        
        # Cancel active tasks
        for task_id in list(self.active_tasks):
            # Wait for tasks to complete
            await asyncio.sleep(0.1)
        
        # Clear queues
        self.task_queue.clear()