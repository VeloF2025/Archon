#!/usr/bin/env python3
"""
Agent Task Planning & Execution Engine Module

This module provides comprehensive task planning and execution capabilities for autonomous AI agents.
It implements hierarchical task planning, dynamic scheduling, resource allocation, and execution
monitoring to enable efficient task management and execution coordination.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.0.0
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be planned and executed"""
    COMPUTATION = auto()           # Computational tasks
    DATA_PROCESSING = auto()       # Data manipulation tasks
    COMMUNICATION = auto()         # Agent communication tasks
    COORDINATION = auto()          # Multi-agent coordination tasks
    LEARNING = auto()             # Machine learning tasks
    MONITORING = auto()           # System monitoring tasks
    MAINTENANCE = auto()          # System maintenance tasks
    OPTIMIZATION = auto()         # Performance optimization tasks
    ANALYSIS = auto()             # Data analysis tasks
    DECISION_MAKING = auto()      # Decision support tasks


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class TaskStatus(Enum):
    """Task execution status"""
    PLANNED = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    SKIPPED = auto()


class PlanningStrategy(Enum):
    """Task planning strategies"""
    HIERARCHICAL = auto()         # Top-down hierarchical planning
    REACTIVE = auto()             # Reactive planning based on events
    HYBRID = auto()               # Combination of hierarchical and reactive
    GOAL_ORIENTED = auto()        # Goal-driven planning
    TEMPORAL = auto()             # Time-based planning
    RESOURCE_OPTIMAL = auto()     # Resource-optimized planning


class ExecutionStrategy(Enum):
    """Task execution strategies"""
    SEQUENTIAL = auto()           # Execute tasks in sequence
    PARALLEL = auto()             # Execute tasks in parallel
    PIPELINE = auto()             # Pipeline execution
    ADAPTIVE = auto()             # Adaptive execution based on conditions
    PRIORITY_BASED = auto()       # Priority-driven execution
    DEADLINE_DRIVEN = auto()      # Deadline-driven execution


class ResourceType(Enum):
    """Types of resources required for task execution"""
    CPU = auto()
    MEMORY = auto()
    STORAGE = auto()
    NETWORK = auto()
    GPU = auto()
    API_QUOTA = auto()
    DATABASE_CONNECTION = auto()
    CUSTOM = auto()


@dataclass
class TaskResource:
    """Represents a resource required for task execution"""
    resource_type: ResourceType
    amount: float
    unit: str
    is_consumable: bool = True
    availability: float = 1.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConstraint:
    """Represents a constraint on task execution"""
    constraint_type: str
    condition: str
    value: Any
    is_hard: bool = True  # Hard vs soft constraint
    penalty: float = 0.0  # Penalty for violating soft constraints
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDependency:
    """Represents a dependency between tasks"""
    dependent_task_id: str
    dependency_task_id: str
    dependency_type: str = "finish_to_start"  # Types: finish_to_start, start_to_start, etc.
    delay: float = 0.0
    is_mandatory: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecutionContext:
    """Context for task execution"""
    agent_id: str
    execution_id: str
    environment: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, TaskResource] = field(default_factory=dict)
    configurations: Dict[str, Any] = field(default_factory=dict)
    state_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentTask:
    """Represents a task in the planning and execution system"""
    task_id: str
    agent_id: str
    task_type: TaskType
    name: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PLANNED
    
    # Task content and parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    actual_output: Optional[Dict[str, Any]] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    planned_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    deadline: Optional[datetime] = None
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    
    # Resources and constraints
    required_resources: Dict[str, TaskResource] = field(default_factory=dict)
    allocated_resources: Dict[str, TaskResource] = field(default_factory=dict)
    constraints: List[TaskConstraint] = field(default_factory=list)
    dependencies: List[TaskDependency] = field(default_factory=list)
    
    # Execution details
    execution_context: Optional[TaskExecutionContext] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class TaskPlan:
    """Represents a complete task execution plan"""
    plan_id: str
    name: str
    description: str
    agent_id: str
    planning_strategy: PlanningStrategy
    execution_strategy: ExecutionStrategy
    
    # Task organization
    tasks: Dict[str, AgentTask] = field(default_factory=dict)
    task_hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # Parent -> Children
    execution_order: List[str] = field(default_factory=list)
    
    # Planning details
    created_at: datetime = field(default_factory=datetime.now)
    planned_start: Optional[datetime] = None
    planned_end: Optional[datetime] = None
    estimated_total_duration: float = 0.0
    
    # Resources and constraints
    total_resource_requirements: Dict[str, TaskResource] = field(default_factory=dict)
    global_constraints: List[TaskConstraint] = field(default_factory=list)
    
    # Execution tracking
    status: TaskStatus = TaskStatus.PLANNED
    progress: float = 0.0
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Metrics for task execution monitoring"""
    total_tasks_planned: int = 0
    total_tasks_executed: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    average_execution_time: float = 0.0
    average_planning_time: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    deadline_adherence_rate: float = 0.0
    retry_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseTaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    async def execute_task(self, task: AgentTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """Execute a task"""
        pass
    
    @abstractmethod
    async def validate_task(self, task: AgentTask) -> bool:
        """Validate if task can be executed"""
        pass
    
    @abstractmethod
    async def estimate_duration(self, task: AgentTask) -> float:
        """Estimate task execution duration"""
        pass


class ComputationTaskExecutor(BaseTaskExecutor):
    """Executor for computational tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 4)
        )
    
    async def execute_task(self, task: AgentTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """Execute a computational task"""
        try:
            # Extract computation parameters
            computation_type = task.parameters.get('computation_type', 'default')
            data = task.parameters.get('data', {})
            algorithm = task.parameters.get('algorithm', 'default')
            
            # Simulate computation
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._perform_computation,
                computation_type,
                data,
                algorithm
            )
            
            return {
                'result': result,
                'computation_type': computation_type,
                'processed_items': len(data) if isinstance(data, (list, dict)) else 1,
                'execution_time': time.time() - context.created_at.timestamp()
            }
            
        except Exception as e:
            logger.error(f"Computation task execution failed: {e}")
            raise
    
    def _perform_computation(self, computation_type: str, data: Any, algorithm: str) -> Any:
        """Perform the actual computation"""
        # Simulate computation based on type
        if computation_type == 'data_analysis':
            # Simulate data analysis
            time.sleep(0.1)  # Simulate processing time
            return {
                'analysis_result': f"Analyzed {len(data) if hasattr(data, '__len__') else 1} items",
                'algorithm_used': algorithm
            }
        elif computation_type == 'optimization':
            # Simulate optimization
            time.sleep(0.2)
            return {
                'optimized_value': 0.95,
                'iterations': 100,
                'convergence': True
            }
        else:
            # Default computation
            time.sleep(0.05)
            return {'processed': True, 'result': 'success'}
    
    async def validate_task(self, task: AgentTask) -> bool:
        """Validate if computational task can be executed"""
        required_params = ['computation_type']
        return all(param in task.parameters for param in required_params)
    
    async def estimate_duration(self, task: AgentTask) -> float:
        """Estimate computational task duration"""
        computation_type = task.parameters.get('computation_type', 'default')
        data_size = len(task.parameters.get('data', []))
        
        # Simple estimation based on type and data size
        base_times = {
            'data_analysis': 0.1,
            'optimization': 0.2,
            'machine_learning': 0.5,
            'default': 0.05
        }
        
        base_time = base_times.get(computation_type, base_times['default'])
        return base_time * (1 + data_size * 0.001)


class DataProcessingTaskExecutor(BaseTaskExecutor):
    """Executor for data processing tasks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def execute_task(self, task: AgentTask, context: TaskExecutionContext) -> Dict[str, Any]:
        """Execute a data processing task"""
        try:
            operation = task.parameters.get('operation', 'transform')
            input_data = task.parameters.get('input_data', {})
            processing_config = task.parameters.get('config', {})
            
            # Simulate data processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            processed_data = await self._process_data(operation, input_data, processing_config)
            
            return {
                'processed_data': processed_data,
                'operation': operation,
                'input_size': len(input_data) if hasattr(input_data, '__len__') else 1,
                'output_size': len(processed_data) if hasattr(processed_data, '__len__') else 1
            }
            
        except Exception as e:
            logger.error(f"Data processing task execution failed: {e}")
            raise
    
    async def _process_data(self, operation: str, data: Any, config: Dict[str, Any]) -> Any:
        """Process data based on operation type"""
        if operation == 'transform':
            # Simulate data transformation
            return {'transformed': True, 'original_size': len(data) if hasattr(data, '__len__') else 1}
        elif operation == 'filter':
            # Simulate data filtering
            return {'filtered': True, 'items_kept': int(len(data) * 0.8) if hasattr(data, '__len__') else 1}
        elif operation == 'aggregate':
            # Simulate data aggregation
            return {'aggregated': True, 'summary': 'aggregation_complete'}
        else:
            return {'processed': True, 'operation': operation}
    
    async def validate_task(self, task: AgentTask) -> bool:
        """Validate if data processing task can be executed"""
        required_params = ['operation', 'input_data']
        return all(param in task.parameters for param in required_params)
    
    async def estimate_duration(self, task: AgentTask) -> float:
        """Estimate data processing task duration"""
        operation = task.parameters.get('operation', 'transform')
        data_size = len(task.parameters.get('input_data', []))
        
        operation_times = {
            'transform': 0.05,
            'filter': 0.03,
            'aggregate': 0.08,
            'sort': 0.1,
            'join': 0.15
        }
        
        base_time = operation_times.get(operation, 0.05)
        return base_time * (1 + data_size * 0.0005)


class TaskPlanner:
    """Advanced task planner with multiple planning strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planning_strategies = {
            PlanningStrategy.HIERARCHICAL: self._hierarchical_planning,
            PlanningStrategy.REACTIVE: self._reactive_planning,
            PlanningStrategy.HYBRID: self._hybrid_planning,
            PlanningStrategy.GOAL_ORIENTED: self._goal_oriented_planning,
            PlanningStrategy.TEMPORAL: self._temporal_planning,
            PlanningStrategy.RESOURCE_OPTIMAL: self._resource_optimal_planning
        }
    
    async def create_plan(self, goals: List[Dict[str, Any]], strategy: PlanningStrategy,
                         constraints: Optional[List[TaskConstraint]] = None,
                         agent_id: str = None) -> TaskPlan:
        """Create a task execution plan"""
        try:
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
            
            # Create base plan
            plan = TaskPlan(
                plan_id=plan_id,
                name=f"Plan for {len(goals)} goals",
                description=f"Generated using {strategy.name} strategy",
                agent_id=agent_id,
                planning_strategy=strategy,
                execution_strategy=ExecutionStrategy.ADAPTIVE,
                global_constraints=constraints or []
            )
            
            # Apply planning strategy
            if strategy in self.planning_strategies:
                await self.planning_strategies[strategy](plan, goals)
            else:
                await self._default_planning(plan, goals)
            
            # Calculate plan metrics
            await self._calculate_plan_metrics(plan)
            
            logger.info(f"Created plan {plan_id} with {len(plan.tasks)} tasks")
            return plan
            
        except Exception as e:
            logger.error(f"Plan creation failed: {e}")
            raise
    
    async def _hierarchical_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Hierarchical task planning"""
        for goal_idx, goal in enumerate(goals):
            # Create high-level task
            main_task_id = f"task_main_{goal_idx}"
            main_task = AgentTask(
                task_id=main_task_id,
                agent_id=plan.agent_id,
                task_type=TaskType.COORDINATION,
                name=goal.get('name', f"Goal {goal_idx}"),
                description=goal.get('description', ''),
                priority=TaskPriority(goal.get('priority', 3)),
                parameters=goal.get('parameters', {})
            )
            
            plan.tasks[main_task_id] = main_task
            plan.execution_order.append(main_task_id)
            
            # Create subtasks
            subtasks = goal.get('subtasks', [])
            subtask_ids = []
            
            for subtask_idx, subtask in enumerate(subtasks):
                subtask_id = f"task_{goal_idx}_{subtask_idx}"
                subtask_obj = AgentTask(
                    task_id=subtask_id,
                    agent_id=plan.agent_id,
                    task_type=TaskType(subtask.get('type', 1)),
                    name=subtask.get('name', f"Subtask {subtask_idx}"),
                    description=subtask.get('description', ''),
                    priority=TaskPriority(subtask.get('priority', 3)),
                    parameters=subtask.get('parameters', {})
                )
                
                plan.tasks[subtask_id] = subtask_obj
                subtask_ids.append(subtask_id)
                
                # Add dependency to main task
                dependency = TaskDependency(
                    dependent_task_id=main_task_id,
                    dependency_task_id=subtask_id,
                    dependency_type="finish_to_start"
                )
                main_task.dependencies.append(dependency)
            
            # Update hierarchy
            plan.task_hierarchy[main_task_id] = subtask_ids
    
    async def _goal_oriented_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Goal-oriented task planning"""
        for goal_idx, goal in enumerate(goals):
            goal_type = goal.get('type', 'general')
            
            if goal_type == 'optimization':
                await self._create_optimization_tasks(plan, goal, goal_idx)
            elif goal_type == 'data_processing':
                await self._create_data_processing_tasks(plan, goal, goal_idx)
            elif goal_type == 'learning':
                await self._create_learning_tasks(plan, goal, goal_idx)
            else:
                await self._create_general_tasks(plan, goal, goal_idx)
    
    async def _create_optimization_tasks(self, plan: TaskPlan, goal: Dict[str, Any], goal_idx: int) -> None:
        """Create tasks for optimization goals"""
        # Analysis task
        analysis_task = AgentTask(
            task_id=f"opt_analysis_{goal_idx}",
            agent_id=plan.agent_id,
            task_type=TaskType.ANALYSIS,
            name="Optimization Analysis",
            description="Analyze current state for optimization",
            parameters={'analysis_type': 'optimization', 'target': goal.get('target', {})}
        )
        plan.tasks[analysis_task.task_id] = analysis_task
        
        # Optimization task
        opt_task = AgentTask(
            task_id=f"opt_exec_{goal_idx}",
            agent_id=plan.agent_id,
            task_type=TaskType.OPTIMIZATION,
            name="Execute Optimization",
            description="Perform optimization algorithm",
            parameters={'algorithm': goal.get('algorithm', 'default'), 'constraints': goal.get('constraints', [])}
        )
        opt_task.dependencies.append(TaskDependency(
            dependent_task_id=opt_task.task_id,
            dependency_task_id=analysis_task.task_id
        ))
        plan.tasks[opt_task.task_id] = opt_task
        
        plan.execution_order.extend([analysis_task.task_id, opt_task.task_id])
    
    async def _create_data_processing_tasks(self, plan: TaskPlan, goal: Dict[str, Any], goal_idx: int) -> None:
        """Create tasks for data processing goals"""
        operations = goal.get('operations', ['load', 'process', 'save'])
        
        previous_task_id = None
        for op_idx, operation in enumerate(operations):
            task_id = f"data_{goal_idx}_{op_idx}"
            task = AgentTask(
                task_id=task_id,
                agent_id=plan.agent_id,
                task_type=TaskType.DATA_PROCESSING,
                name=f"Data {operation.title()}",
                description=f"Perform {operation} operation",
                parameters={'operation': operation, 'config': goal.get('config', {})}
            )
            
            if previous_task_id:
                task.dependencies.append(TaskDependency(
                    dependent_task_id=task_id,
                    dependency_task_id=previous_task_id
                ))
            
            plan.tasks[task_id] = task
            plan.execution_order.append(task_id)
            previous_task_id = task_id
    
    async def _create_learning_tasks(self, plan: TaskPlan, goal: Dict[str, Any], goal_idx: int) -> None:
        """Create tasks for learning goals"""
        # Data preparation
        prep_task = AgentTask(
            task_id=f"learn_prep_{goal_idx}",
            agent_id=plan.agent_id,
            task_type=TaskType.DATA_PROCESSING,
            name="Prepare Learning Data",
            description="Prepare data for learning",
            parameters={'operation': 'prepare_learning_data', 'config': goal.get('data_config', {})}
        )
        plan.tasks[prep_task.task_id] = prep_task
        
        # Training task
        train_task = AgentTask(
            task_id=f"learn_train_{goal_idx}",
            agent_id=plan.agent_id,
            task_type=TaskType.LEARNING,
            name="Execute Learning",
            description="Perform learning algorithm",
            parameters={'algorithm': goal.get('algorithm', 'default'), 'config': goal.get('learning_config', {})}
        )
        train_task.dependencies.append(TaskDependency(
            dependent_task_id=train_task.task_id,
            dependency_task_id=prep_task.task_id
        ))
        plan.tasks[train_task.task_id] = train_task
        
        plan.execution_order.extend([prep_task.task_id, train_task.task_id])
    
    async def _create_general_tasks(self, plan: TaskPlan, goal: Dict[str, Any], goal_idx: int) -> None:
        """Create general tasks for unspecified goals"""
        task_id = f"general_{goal_idx}"
        task = AgentTask(
            task_id=task_id,
            agent_id=plan.agent_id,
            task_type=TaskType.COMPUTATION,
            name=goal.get('name', f"General Task {goal_idx}"),
            description=goal.get('description', ''),
            parameters=goal.get('parameters', {})
        )
        plan.tasks[task_id] = task
        plan.execution_order.append(task_id)
    
    async def _reactive_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Reactive task planning based on current state"""
        # Create immediate response tasks
        for goal_idx, goal in enumerate(goals):
            task_id = f"reactive_{goal_idx}"
            task = AgentTask(
                task_id=task_id,
                agent_id=plan.agent_id,
                task_type=TaskType.COORDINATION,
                name=f"Reactive Response {goal_idx}",
                description="Immediate response to current state",
                priority=TaskPriority.HIGH,
                parameters=goal.get('parameters', {})
            )
            plan.tasks[task_id] = task
            plan.execution_order.append(task_id)
    
    async def _hybrid_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Hybrid planning combining hierarchical and reactive approaches"""
        # Apply both strategies
        await self._hierarchical_planning(plan, goals[:len(goals)//2])
        await self._reactive_planning(plan, goals[len(goals)//2:])
    
    async def _temporal_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Time-based task planning"""
        # Sort goals by deadline/priority
        sorted_goals = sorted(goals, key=lambda g: (
            g.get('deadline', datetime.max),
            g.get('priority', 5)
        ))
        
        await self._goal_oriented_planning(plan, sorted_goals)
    
    async def _resource_optimal_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Resource-optimized task planning"""
        # Group goals by resource requirements
        resource_groups = defaultdict(list)
        
        for goal in goals:
            resources = goal.get('required_resources', {})
            resource_key = tuple(sorted(resources.keys()))
            resource_groups[resource_key].append(goal)
        
        # Plan each resource group
        for resource_key, group_goals in resource_groups.items():
            await self._goal_oriented_planning(plan, group_goals)
    
    async def _default_planning(self, plan: TaskPlan, goals: List[Dict[str, Any]]) -> None:
        """Default planning strategy"""
        await self._goal_oriented_planning(plan, goals)
    
    async def _calculate_plan_metrics(self, plan: TaskPlan) -> None:
        """Calculate plan metrics and estimates"""
        total_duration = 0.0
        critical_path_duration = 0.0
        
        for task in plan.tasks.values():
            total_duration += task.estimated_duration
            # Simple critical path calculation (would be more complex in reality)
            if not task.dependencies:
                critical_path_duration = max(critical_path_duration, task.estimated_duration)
        
        plan.estimated_total_duration = total_duration
        plan.planned_start = datetime.now() + timedelta(minutes=1)
        plan.planned_end = plan.planned_start + timedelta(seconds=critical_path_duration)


class TaskExecutor:
    """Advanced task executor with multiple execution strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.executors: Dict[TaskType, BaseTaskExecutor] = {}
        self._initialize_executors()
        
        # Execution management
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.resource_pool = ResourcePool(config.get('resources', {}))
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
    
    def _initialize_executors(self) -> None:
        """Initialize task type-specific executors"""
        self.executors[TaskType.COMPUTATION] = ComputationTaskExecutor(
            self.config.get('computation', {})
        )
        self.executors[TaskType.DATA_PROCESSING] = DataProcessingTaskExecutor(
            self.config.get('data_processing', {})
        )
        # Add more executors as needed
    
    async def execute_plan(self, plan: TaskPlan, strategy: ExecutionStrategy = None) -> bool:
        """Execute a complete task plan"""
        try:
            execution_strategy = strategy or plan.execution_strategy
            
            if execution_strategy == ExecutionStrategy.SEQUENTIAL:
                return await self._execute_sequential(plan)
            elif execution_strategy == ExecutionStrategy.PARALLEL:
                return await self._execute_parallel(plan)
            elif execution_strategy == ExecutionStrategy.PIPELINE:
                return await self._execute_pipeline(plan)
            elif execution_strategy == ExecutionStrategy.ADAPTIVE:
                return await self._execute_adaptive(plan)
            elif execution_strategy == ExecutionStrategy.PRIORITY_BASED:
                return await self._execute_priority_based(plan)
            elif execution_strategy == ExecutionStrategy.DEADLINE_DRIVEN:
                return await self._execute_deadline_driven(plan)
            else:
                return await self._execute_sequential(plan)
                
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = TaskStatus.FAILED
            return False
    
    async def _execute_sequential(self, plan: TaskPlan) -> bool:
        """Execute tasks sequentially"""
        plan.status = TaskStatus.RUNNING
        
        for task_id in plan.execution_order:
            task = plan.tasks[task_id]
            
            # Check dependencies
            if not await self._check_dependencies(task, plan):
                continue
            
            # Execute task
            success = await self.execute_task(task)
            if success:
                plan.completed_tasks.add(task_id)
            else:
                plan.failed_tasks.add(task_id)
                if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                    plan.status = TaskStatus.FAILED
                    return False
        
        plan.status = TaskStatus.COMPLETED
        return True
    
    async def _execute_parallel(self, plan: TaskPlan) -> bool:
        """Execute tasks in parallel where possible"""
        plan.status = TaskStatus.RUNNING
        
        # Group tasks by dependency level
        dependency_levels = self._calculate_dependency_levels(plan)
        
        for level, task_ids in dependency_levels.items():
            # Execute all tasks at this level in parallel
            tasks_to_execute = []
            for task_id in task_ids:
                task = plan.tasks[task_id]
                if await self._check_dependencies(task, plan):
                    tasks_to_execute.append(self.execute_task(task))
            
            if tasks_to_execute:
                results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
                
                for task_id, result in zip(task_ids, results):
                    if isinstance(result, bool) and result:
                        plan.completed_tasks.add(task_id)
                    else:
                        plan.failed_tasks.add(task_id)
        
        plan.status = TaskStatus.COMPLETED if plan.failed_tasks else TaskStatus.COMPLETED
        return len(plan.failed_tasks) == 0
    
    async def _execute_adaptive(self, plan: TaskPlan) -> bool:
        """Execute tasks adaptively based on conditions"""
        # Start with parallel execution, fall back to sequential on failures
        try:
            return await self._execute_parallel(plan)
        except Exception as e:
            logger.warning(f"Parallel execution failed, falling back to sequential: {e}")
            return await self._execute_sequential(plan)
    
    async def _execute_priority_based(self, plan: TaskPlan) -> bool:
        """Execute tasks based on priority"""
        plan.status = TaskStatus.RUNNING
        
        # Sort tasks by priority
        sorted_tasks = sorted(
            plan.tasks.values(),
            key=lambda t: (t.priority.value, t.deadline or datetime.max)
        )
        
        for task in sorted_tasks:
            if await self._check_dependencies(task, plan):
                success = await self.execute_task(task)
                if success:
                    plan.completed_tasks.add(task.task_id)
                else:
                    plan.failed_tasks.add(task.task_id)
        
        plan.status = TaskStatus.COMPLETED
        return len(plan.failed_tasks) == 0
    
    async def _execute_deadline_driven(self, plan: TaskPlan) -> bool:
        """Execute tasks driven by deadlines"""
        plan.status = TaskStatus.RUNNING
        
        # Sort tasks by deadline
        sorted_tasks = sorted(
            [task for task in plan.tasks.values() if task.deadline],
            key=lambda t: t.deadline
        )
        
        # Execute deadline-sensitive tasks first
        for task in sorted_tasks:
            if await self._check_dependencies(task, plan):
                success = await self.execute_task(task)
                if success:
                    plan.completed_tasks.add(task.task_id)
                else:
                    plan.failed_tasks.add(task.task_id)
        
        # Execute remaining tasks
        remaining_tasks = [
            task for task in plan.tasks.values()
            if task.task_id not in plan.completed_tasks and task.task_id not in plan.failed_tasks
        ]
        
        for task in remaining_tasks:
            if await self._check_dependencies(task, plan):
                success = await self.execute_task(task)
                if success:
                    plan.completed_tasks.add(task.task_id)
                else:
                    plan.failed_tasks.add(task.task_id)
        
        plan.status = TaskStatus.COMPLETED
        return len(plan.failed_tasks) == 0
    
    async def _execute_pipeline(self, plan: TaskPlan) -> bool:
        """Execute tasks in pipeline fashion"""
        # Simplified pipeline execution
        return await self._execute_sequential(plan)
    
    def _calculate_dependency_levels(self, plan: TaskPlan) -> Dict[int, List[str]]:
        """Calculate dependency levels for parallel execution"""
        levels = defaultdict(list)
        task_levels = {}
        
        # Calculate level for each task
        for task_id, task in plan.tasks.items():
            level = self._get_task_level(task, plan.tasks, task_levels)
            task_levels[task_id] = level
            levels[level].append(task_id)
        
        return dict(levels)
    
    def _get_task_level(self, task: AgentTask, all_tasks: Dict[str, AgentTask], 
                       task_levels: Dict[str, int]) -> int:
        """Get dependency level for a task"""
        if task.task_id in task_levels:
            return task_levels[task.task_id]
        
        if not task.dependencies:
            task_levels[task.task_id] = 0
            return 0
        
        max_dep_level = 0
        for dep in task.dependencies:
            dep_task = all_tasks.get(dep.dependency_task_id)
            if dep_task:
                dep_level = self._get_task_level(dep_task, all_tasks, task_levels)
                max_dep_level = max(max_dep_level, dep_level)
        
        level = max_dep_level + 1
        task_levels[task.task_id] = level
        return level
    
    async def _check_dependencies(self, task: AgentTask, plan: TaskPlan) -> bool:
        """Check if task dependencies are satisfied"""
        for dep in task.dependencies:
            if dep.dependency_task_id not in plan.completed_tasks:
                return False
        return True
    
    async def execute_task(self, task: AgentTask) -> bool:
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.actual_start = datetime.now()
            
            # Get appropriate executor
            executor = self.executors.get(task.task_type)
            if not executor:
                logger.error(f"No executor for task type {task.task_type}")
                task.status = TaskStatus.FAILED
                task.error_message = f"No executor for task type {task.task_type}"
                return False
            
            # Validate task
            if not await executor.validate_task(task):
                logger.error(f"Task validation failed for {task.task_id}")
                task.status = TaskStatus.FAILED
                task.error_message = "Task validation failed"
                return False
            
            # Create execution context
            context = TaskExecutionContext(
                agent_id=task.agent_id,
                execution_id=f"exec_{uuid.uuid4().hex[:8]}"
            )
            
            # Execute task
            result = await executor.execute_task(task, context)
            
            # Update task with results
            task.actual_output = result
            task.actual_end = datetime.now()
            task.actual_duration = (task.actual_end - task.actual_start).total_seconds()
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            
            logger.info(f"Task {task.task_id} completed successfully")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.actual_end = datetime.now()
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.QUEUED
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                return await self.execute_task(task)
            
            return False


class ResourcePool:
    """Resource pool for managing execution resources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.available_resources: Dict[str, float] = config.get('initial_resources', {
            'cpu': 100.0,
            'memory': 1000.0,
            'storage': 10000.0,
            'network': 100.0
        })
        self.allocated_resources: Dict[str, float] = defaultdict(float)
        self.lock = threading.Lock()
    
    async def allocate_resources(self, required: Dict[str, TaskResource]) -> bool:
        """Allocate resources for task execution"""
        with self.lock:
            # Check availability
            for resource_name, resource in required.items():
                available = self.available_resources.get(resource_name, 0)
                if available < resource.amount:
                    return False
            
            # Allocate resources
            for resource_name, resource in required.items():
                self.available_resources[resource_name] -= resource.amount
                self.allocated_resources[resource_name] += resource.amount
            
            return True
    
    async def release_resources(self, allocated: Dict[str, TaskResource]) -> None:
        """Release allocated resources"""
        with self.lock:
            for resource_name, resource in allocated.items():
                self.available_resources[resource_name] += resource.amount
                self.allocated_resources[resource_name] -= resource.amount


class AgentTaskPlanningExecution:
    """Main task planning and execution system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = f"task_sys_{uuid.uuid4().hex[:8]}"
        
        # Initialize components
        self.planner = TaskPlanner(config.get('planning', {}))
        self.executor = TaskExecutor(config.get('execution', {}))
        
        # System state
        self.active_plans: Dict[str, TaskPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.metrics = ExecutionMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
    
    async def start(self) -> None:
        """Start the task planning and execution system"""
        try:
            self.is_running = True
            
            # Start background monitoring
            self.background_tasks.add(
                asyncio.create_task(self._metrics_collector())
            )
            
            logger.info(f"Task planning system {self.system_id} started")
            
        except Exception as e:
            logger.error(f"System start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the task planning and execution system"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            logger.info(f"Task planning system {self.system_id} stopped")
            
        except Exception as e:
            logger.error(f"System stop failed: {e}")
    
    async def create_and_execute_plan(self, goals: List[Dict[str, Any]], 
                                    planning_strategy: PlanningStrategy = PlanningStrategy.GOAL_ORIENTED,
                                    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
                                    agent_id: str = None) -> str:
        """Create and execute a task plan"""
        try:
            # Create plan
            plan = await self.planner.create_plan(goals, planning_strategy, agent_id=agent_id)
            plan.execution_strategy = execution_strategy
            
            self.active_plans[plan.plan_id] = plan
            self.metrics.total_tasks_planned += len(plan.tasks)
            
            # Execute plan
            start_time = time.time()
            success = await self.executor.execute_plan(plan, execution_strategy)
            execution_time = time.time() - start_time
            
            # Update metrics
            if success:
                self.metrics.successful_executions += len(plan.completed_tasks)
                self.metrics.average_execution_time = (
                    (self.metrics.average_execution_time * self.metrics.total_tasks_executed + execution_time) /
                    (self.metrics.total_tasks_executed + len(plan.tasks))
                )
            else:
                self.metrics.failed_executions += len(plan.failed_tasks)
            
            self.metrics.total_tasks_executed += len(plan.tasks)
            
            # Record execution
            self.execution_history.append({
                'plan_id': plan.plan_id,
                'success': success,
                'execution_time': execution_time,
                'tasks_completed': len(plan.completed_tasks),
                'tasks_failed': len(plan.failed_tasks),
                'timestamp': datetime.now()
            })
            
            return plan.plan_id
            
        except Exception as e:
            logger.error(f"Plan creation and execution failed: {e}")
            raise
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plan"""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return None
        
        return {
            'plan_id': plan.plan_id,
            'status': plan.status.name,
            'progress': plan.progress,
            'total_tasks': len(plan.tasks),
            'completed_tasks': len(plan.completed_tasks),
            'failed_tasks': len(plan.failed_tasks),
            'created_at': plan.created_at,
            'planned_start': plan.planned_start,
            'planned_end': plan.planned_end,
            'estimated_duration': plan.estimated_total_duration
        }
    
    def get_metrics(self) -> ExecutionMetrics:
        """Get current system metrics"""
        self.metrics.last_updated = datetime.now()
        if self.metrics.total_tasks_executed > 0:
            self.metrics.deadline_adherence_rate = (
                self.metrics.successful_executions / self.metrics.total_tasks_executed
            ) * 100
        return self.metrics
    
    def get_active_plans(self) -> List[Dict[str, Any]]:
        """Get list of active plans"""
        return [
            {
                'plan_id': plan.plan_id,
                'name': plan.name,
                'status': plan.status.name,
                'progress': plan.progress,
                'total_tasks': len(plan.tasks),
                'agent_id': plan.agent_id
            }
            for plan in self.active_plans.values()
        ]
    
    async def _metrics_collector(self) -> None:
        """Background task for collecting metrics"""
        while self.is_running:
            try:
                # Update resource utilization
                # This would collect real resource usage in a production system
                
                # Calculate retry rate
                total_retries = sum(
                    task.retry_count
                    for plan in self.active_plans.values()
                    for task in plan.tasks.values()
                )
                
                if self.metrics.total_tasks_executed > 0:
                    self.metrics.retry_rate = (total_retries / self.metrics.total_tasks_executed) * 100
                
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(30)


async def example_task_planning_execution_usage():
    """Comprehensive example of task planning and execution system usage"""
    
    print("\n🎯 Agent Task Planning & Execution System Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'planning': {
            'default_strategy': 'goal_oriented',
            'enable_optimization': True
        },
        'execution': {
            'default_strategy': 'adaptive',
            'max_parallel_tasks': 5,
            'computation': {
                'max_workers': 4
            },
            'data_processing': {
                'buffer_size': 1000
            },
            'resources': {
                'initial_resources': {
                    'cpu': 100.0,
                    'memory': 2000.0,
                    'storage': 50000.0,
                    'network': 200.0
                }
            }
        }
    }
    
    # Initialize system
    task_system = AgentTaskPlanningExecution(config)
    await task_system.start()
    
    print(f"✅ Task system {task_system.system_id} started")
    
    try:
        # Example 1: Data Processing Pipeline
        print("\n1. Data Processing Pipeline")
        print("-" * 40)
        
        data_goals = [
            {
                'name': 'Process Customer Data',
                'type': 'data_processing',
                'priority': 2,
                'operations': ['load', 'clean', 'transform', 'analyze', 'save'],
                'config': {
                    'source': 'customer_database',
                    'format': 'csv',
                    'output_format': 'json'
                },
                'parameters': {
                    'batch_size': 1000,
                    'validation_rules': ['not_null', 'valid_email']
                }
            }
        ]
        
        plan_id1 = await task_system.create_and_execute_plan(
            goals=data_goals,
            planning_strategy=PlanningStrategy.TEMPORAL,
            execution_strategy=ExecutionStrategy.PIPELINE
        )
        
        print(f"✅ Data processing plan created and executed: {plan_id1}")
        
        # Example 2: Optimization Goals
        print("\n2. Optimization Tasks")
        print("-" * 40)
        
        optimization_goals = [
            {
                'name': 'Resource Allocation Optimization',
                'type': 'optimization',
                'priority': 1,
                'algorithm': 'genetic_algorithm',
                'target': {
                    'objective': 'minimize_cost',
                    'variables': ['cpu_allocation', 'memory_allocation'],
                    'constraints': ['max_latency', 'min_availability']
                },
                'constraints': [
                    {'type': 'resource', 'limit': 1000},
                    {'type': 'time', 'deadline': '2025-01-09T18:00:00Z'}
                ]
            }
        ]
        
        plan_id2 = await task_system.create_and_execute_plan(
            goals=optimization_goals,
            planning_strategy=PlanningStrategy.RESOURCE_OPTIMAL,
            execution_strategy=ExecutionStrategy.PRIORITY_BASED
        )
        
        print(f"✅ Optimization plan created and executed: {plan_id2}")
        
        # Example 3: Learning Tasks
        print("\n3. Machine Learning Pipeline")
        print("-" * 40)
        
        learning_goals = [
            {
                'name': 'Predictive Model Training',
                'type': 'learning',
                'priority': 2,
                'algorithm': 'random_forest',
                'data_config': {
                    'training_data': 'historical_data.csv',
                    'validation_split': 0.2,
                    'preprocessing': ['normalize', 'feature_selection']
                },
                'learning_config': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'cross_validation': 5
                }
            }
        ]
        
        plan_id3 = await task_system.create_and_execute_plan(
            goals=learning_goals,
            planning_strategy=PlanningStrategy.HIERARCHICAL,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        print(f"✅ Learning plan created and executed: {plan_id3}")
        
        # Example 4: Mixed Goals with Dependencies
        print("\n4. Complex Multi-Goal Plan")
        print("-" * 40)
        
        complex_goals = [
            {
                'name': 'Data Preparation',
                'type': 'data_processing',
                'priority': 1,
                'operations': ['extract', 'validate', 'clean'],
                'config': {'source': 'multiple_sources'}
            },
            {
                'name': 'Feature Engineering',
                'type': 'computation',
                'priority': 2,
                'parameters': {
                    'computation_type': 'data_analysis',
                    'data': list(range(1000)),  # Sample data
                    'algorithm': 'feature_extraction'
                },
                'depends_on': ['Data Preparation']
            },
            {
                'name': 'Model Training',
                'type': 'learning',
                'priority': 2,
                'algorithm': 'neural_network',
                'depends_on': ['Feature Engineering']
            },
            {
                'name': 'Performance Optimization',
                'type': 'optimization',
                'priority': 3,
                'algorithm': 'hyperparameter_tuning',
                'depends_on': ['Model Training']
            }
        ]
        
        plan_id4 = await task_system.create_and_execute_plan(
            goals=complex_goals,
            planning_strategy=PlanningStrategy.HYBRID,
            execution_strategy=ExecutionStrategy.ADAPTIVE
        )
        
        print(f"✅ Complex plan created and executed: {plan_id4}")
        
        # Allow some execution time
        await asyncio.sleep(2)
        
        # Example 5: Plan Status Monitoring
        print("\n5. Plan Status Monitoring")
        print("-" * 40)
        
        for plan_id in [plan_id1, plan_id2, plan_id3, plan_id4]:
            status = await task_system.get_plan_status(plan_id)
            if status:
                print(f"Plan {plan_id}:")
                print(f"  Status: {status['status']}")
                print(f"  Progress: {status['progress']:.1f}%")
                print(f"  Tasks: {status['completed_tasks']}/{status['total_tasks']} completed")
                print(f"  Failed: {status['failed_tasks']}")
        
        # Example 6: System Metrics
        print("\n6. System Metrics")
        print("-" * 40)
        
        metrics = task_system.get_metrics()
        print(f"✅ Total tasks planned: {metrics.total_tasks_planned}")
        print(f"✅ Total tasks executed: {metrics.total_tasks_executed}")
        print(f"✅ Successful executions: {metrics.successful_executions}")
        print(f"✅ Failed executions: {metrics.failed_executions}")
        print(f"✅ Average execution time: {metrics.average_execution_time:.3f}s")
        print(f"✅ Average planning time: {metrics.average_planning_time:.3f}s")
        print(f"✅ Deadline adherence rate: {metrics.deadline_adherence_rate:.1f}%")
        print(f"✅ Retry rate: {metrics.retry_rate:.1f}%")
        
        # Example 7: Active Plans Overview
        print("\n7. Active Plans Overview")
        print("-" * 40)
        
        active_plans = task_system.get_active_plans()
        print(f"✅ Active plans: {len(active_plans)}")
        
        for plan_info in active_plans:
            print(f"   - {plan_info['plan_id']}: {plan_info['name']}")
            print(f"     Status: {plan_info['status']}")
            print(f"     Progress: {plan_info['progress']:.1f}%")
            print(f"     Tasks: {plan_info['total_tasks']}")
            print(f"     Agent: {plan_info['agent_id']}")
        
        # Allow background tasks to run
        await asyncio.sleep(1)
        
    finally:
        # Cleanup
        await task_system.stop()
        print(f"\n✅ Task planning and execution system stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_task_planning_execution_usage())