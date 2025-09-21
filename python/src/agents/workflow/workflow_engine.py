"""
Workflow Engine Module
Core workflow orchestration and execution engine
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict


class WorkflowStatus(Enum):
    """Workflow execution status"""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class StepType(Enum):
    """Types of workflow steps"""
    ACTION = "action"
    DECISION = "decision"
    PARALLEL = "parallel"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"
    WAIT = "wait"
    APPROVAL = "approval"
    NOTIFICATION = "notification"
    TRANSFORM = "transform"
    VALIDATION = "validation"


class TriggerType(Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    WEBHOOK = "webhook"
    API = "api"
    FILE = "file"
    DATABASE = "database"
    CONDITION = "condition"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    step_id: str
    name: str
    step_type: StepType
    action: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    error_handler: Optional[str] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None  # seconds
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowTrigger:
    """Workflow trigger configuration"""
    trigger_id: str
    trigger_type: TriggerType
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    schedule: Optional[Dict[str, Any]] = None
    event_pattern: Optional[Dict[str, Any]] = None
    webhook_config: Optional[Dict[str, Any]] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: int = 1
    triggers: List[WorkflowTrigger] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.RUNNING
    current_step: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """
    Advanced workflow orchestration engine
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_executions: Set[str] = set()
        self.step_handlers: Dict[str, Callable] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.execution_lock = asyncio.Lock()
        
        # Workflow graph for dependency analysis
        self.workflow_graph = nx.DiGraph()
        
        # Execution history
        self.execution_history: List[WorkflowExecution] = []
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0
        }
        
        self._register_default_handlers()
        self._start_engine()
    
    def _register_default_handlers(self):
        """Register default step handlers"""
        self.register_handler("log", self._handle_log)
        self.register_handler("wait", self._handle_wait)
        self.register_handler("http_request", self._handle_http_request)
        self.register_handler("transform", self._handle_transform)
        self.register_handler("validate", self._handle_validate)
        self.register_handler("notify", self._handle_notify)
    
    def _start_engine(self):
        """Start the workflow engine"""
        asyncio.create_task(self._process_event_queue())
        asyncio.create_task(self._monitor_executions())
        asyncio.create_task(self._cleanup_completed())
    
    async def _process_event_queue(self):
        """Process workflow events"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._handle_event(event)
            except Exception as e:
                print(f"Event processing error: {e}")
    
    async def _monitor_executions(self):
        """Monitor active workflow executions"""
        while True:
            try:
                await self._check_timeouts()
                await self._update_metrics()
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Execution monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_completed(self):
        """Clean up completed executions"""
        while True:
            try:
                cutoff = datetime.now() - timedelta(hours=24)
                
                # Move completed executions to history
                completed = [
                    exec_id for exec_id, execution in self.executions.items()
                    if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
                    and execution.completed_at and execution.completed_at < cutoff
                ]
                
                for exec_id in completed:
                    execution = self.executions.pop(exec_id)
                    self.execution_history.append(execution)
                    self.active_executions.discard(exec_id)
                
                # Keep only recent history
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-1000:]
                
                await asyncio.sleep(3600)  # Cleanup hourly
                
            except Exception as e:
                print(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def create_workflow(self, name: str, description: str) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        
        self.workflows[workflow.workflow_id] = workflow
        return workflow
    
    def add_step(self, workflow_id: str, step: WorkflowStep) -> bool:
        """Add a step to workflow"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.steps.append(step)
        workflow.updated_at = datetime.now()
        
        # Update workflow graph
        self._update_workflow_graph(workflow)
        
        return True
    
    def _update_workflow_graph(self, workflow: Workflow):
        """Update workflow dependency graph"""
        # Clear existing nodes for this workflow
        workflow_nodes = [n for n in self.workflow_graph.nodes() 
                         if n.startswith(workflow.workflow_id)]
        self.workflow_graph.remove_nodes_from(workflow_nodes)
        
        # Add nodes and edges
        for step in workflow.steps:
            node_id = f"{workflow.workflow_id}:{step.step_id}"
            self.workflow_graph.add_node(node_id, step=step)
            
            for next_step_id in step.next_steps:
                next_node_id = f"{workflow.workflow_id}:{next_step_id}"
                self.workflow_graph.add_edge(node_id, next_node_id)
    
    def validate_workflow(self, workflow_id: str) -> Tuple[bool, List[str]]:
        """Validate workflow for execution"""
        if workflow_id not in self.workflows:
            return False, ["Workflow not found"]
        
        workflow = self.workflows[workflow_id]
        errors = []
        
        # Check for start step
        if not workflow.steps:
            errors.append("Workflow has no steps")
        
        # Check for cycles
        workflow_nodes = [n for n in self.workflow_graph.nodes() 
                         if n.startswith(workflow_id)]
        subgraph = self.workflow_graph.subgraph(workflow_nodes)
        
        if not nx.is_directed_acyclic_graph(subgraph):
            errors.append("Workflow contains cycles")
        
        # Check for unreachable steps
        if workflow.steps:
            start_node = f"{workflow_id}:{workflow.steps[0].step_id}"
            reachable = nx.descendants(self.workflow_graph, start_node)
            reachable.add(start_node)
            
            for step in workflow.steps:
                node_id = f"{workflow_id}:{step.step_id}"
                if node_id not in reachable:
                    errors.append(f"Step {step.name} is unreachable")
        
        # Check handlers exist
        for step in workflow.steps:
            if step.action and step.action not in self.step_handlers:
                errors.append(f"Handler not found for action: {step.action}")
        
        return len(errors) == 0, errors
    
    async def execute_workflow(self, workflow_id: str,
                              inputs: Optional[Dict[str, Any]] = None) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        # Validate workflow
        is_valid, errors = self.validate_workflow(workflow_id)
        if not is_valid:
            raise ValueError(f"Workflow validation failed: {errors}")
        
        # Create execution
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            started_at=datetime.now(),
            context={
                **workflow.variables,
                **(inputs or {}),
                "workflow_id": workflow_id,
                "execution_id": execution_id
            }
        )
        
        self.executions[execution_id] = execution
        self.active_executions.add(execution_id)
        
        # Start execution
        asyncio.create_task(self._execute_workflow_async(execution_id))
        
        return execution_id
    
    async def _execute_workflow_async(self, execution_id: str):
        """Execute workflow asynchronously"""
        try:
            async with self.execution_lock:
                execution = self.executions[execution_id]
                workflow = self.workflows[execution.workflow_id]
                
                # Execute steps
                current_step_index = 0
                while current_step_index < len(workflow.steps):
                    step = workflow.steps[current_step_index]
                    execution.current_step = step.step_id
                    
                    # Check if execution was cancelled
                    if execution.status == WorkflowStatus.CANCELLED:
                        break
                    
                    # Execute step
                    try:
                        result = await self._execute_step(step, execution.context)
                        execution.execution_path.append(step.step_id)
                        execution.results[step.step_id] = result
                        
                        # Update context with result
                        if isinstance(result, dict):
                            execution.context.update(result)
                        
                        # Determine next step
                        if step.step_type == StepType.DECISION:
                            next_step_id = self._evaluate_decision(step, execution.context)
                            if next_step_id:
                                # Find index of next step
                                for i, s in enumerate(workflow.steps):
                                    if s.step_id == next_step_id:
                                        current_step_index = i
                                        break
                                else:
                                    current_step_index += 1
                            else:
                                current_step_index += 1
                        elif step.next_steps:
                            # Go to first next step
                            next_step_id = step.next_steps[0]
                            for i, s in enumerate(workflow.steps):
                                if s.step_id == next_step_id:
                                    current_step_index = i
                                    break
                            else:
                                current_step_index += 1
                        else:
                            current_step_index += 1
                            
                    except Exception as e:
                        # Handle step error
                        error_info = {
                            "step_id": step.step_id,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        execution.errors.append(error_info)
                        
                        if step.error_handler:
                            # Execute error handler
                            await self._execute_error_handler(step.error_handler, 
                                                             execution.context, e)
                        elif step.retry_policy:
                            # Retry step
                            retry_success = await self._retry_step(step, execution)
                            if not retry_success:
                                execution.status = WorkflowStatus.FAILED
                                break
                        else:
                            execution.status = WorkflowStatus.FAILED
                            break
                
                # Workflow completed
                if execution.status != WorkflowStatus.FAILED:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.completed_at = datetime.now()
                
                # Update metrics
                self.metrics["total_executions"] += 1
                if execution.status == WorkflowStatus.COMPLETED:
                    self.metrics["successful_executions"] += 1
                else:
                    self.metrics["failed_executions"] += 1
                    
        except Exception as e:
            print(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.errors.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        finally:
            self.active_executions.discard(execution_id)
    
    async def _execute_step(self, step: WorkflowStep,
                          context: Dict[str, Any]) -> Any:
        """Execute a single workflow step"""
        # Apply timeout if specified
        if step.timeout:
            return await asyncio.wait_for(
                self._execute_step_internal(step, context),
                timeout=step.timeout
            )
        else:
            return await self._execute_step_internal(step, context)
    
    async def _execute_step_internal(self, step: WorkflowStep,
                                   context: Dict[str, Any]) -> Any:
        """Internal step execution"""
        if step.step_type == StepType.ACTION:
            if step.action in self.step_handlers:
                handler = self.step_handlers[step.action]
                return await handler(step.parameters, context)
            else:
                raise ValueError(f"Unknown action: {step.action}")
                
        elif step.step_type == StepType.WAIT:
            wait_time = step.parameters.get("seconds", 1)
            await asyncio.sleep(wait_time)
            return {"waited": wait_time}
            
        elif step.step_type == StepType.PARALLEL:
            # Execute parallel steps
            tasks = []
            for parallel_step_id in step.parameters.get("steps", []):
                # Find and execute parallel step
                tasks.append(self._execute_parallel_step(parallel_step_id, context))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {"parallel_results": results}
            
        elif step.step_type == StepType.LOOP:
            # Execute loop
            results = []
            items = self._evaluate_expression(step.parameters.get("items", []), context)
            
            for item in items:
                loop_context = {**context, "item": item}
                # Execute loop body
                result = await self._execute_loop_body(step, loop_context)
                results.append(result)
            
            return {"loop_results": results}
            
        elif step.step_type == StepType.APPROVAL:
            # Wait for approval
            return await self._wait_for_approval(step, context)
            
        elif step.step_type == StepType.TRANSFORM:
            # Transform data
            return self._transform_data(step.parameters, context)
            
        elif step.step_type == StepType.VALIDATION:
            # Validate data
            return self._validate_data(step.parameters, context)
            
        else:
            return {}
    
    def _evaluate_decision(self, step: WorkflowStep,
                         context: Dict[str, Any]) -> Optional[str]:
        """Evaluate decision step"""
        for condition in step.conditions:
            if self._evaluate_condition(condition, context):
                return condition.get("next_step")
        
        # Default next step
        return step.next_steps[0] if step.next_steps else None
    
    def _evaluate_condition(self, condition: Dict[str, Any],
                          context: Dict[str, Any]) -> bool:
        """Evaluate a condition"""
        operator = condition.get("operator", "eq")
        left = self._evaluate_expression(condition.get("left"), context)
        right = self._evaluate_expression(condition.get("right"), context)
        
        if operator == "eq":
            return left == right
        elif operator == "ne":
            return left != right
        elif operator == "gt":
            return left > right
        elif operator == "gte":
            return left >= right
        elif operator == "lt":
            return left < right
        elif operator == "lte":
            return left <= right
        elif operator == "in":
            return left in right
        elif operator == "not_in":
            return left not in right
        elif operator == "contains":
            return right in left
        elif operator == "regex":
            import re
            return bool(re.match(right, str(left)))
        else:
            return False
    
    def _evaluate_expression(self, expression: Any,
                           context: Dict[str, Any]) -> Any:
        """Evaluate an expression in context"""
        if isinstance(expression, str) and expression.startswith("{{") and expression.endswith("}}"):
            # Variable reference
            var_name = expression[2:-2].strip()
            return self._get_nested_value(context, var_name)
        else:
            return expression
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary"""
        keys = path.split(".")
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        
        return value
    
    async def _retry_step(self, step: WorkflowStep,
                        execution: WorkflowExecution) -> bool:
        """Retry a failed step"""
        max_retries = step.retry_policy.get("max_retries", 3)
        retry_delay = step.retry_policy.get("retry_delay", 1)
        
        for attempt in range(max_retries):
            await asyncio.sleep(retry_delay * (attempt + 1))
            
            try:
                result = await self._execute_step(step, execution.context)
                execution.results[step.step_id] = result
                return True
            except Exception as e:
                continue
        
        return False
    
    async def _execute_error_handler(self, handler_name: str,
                                   context: Dict[str, Any],
                                   error: Exception):
        """Execute error handler"""
        if handler_name in self.step_handlers:
            handler = self.step_handlers[handler_name]
            await handler({"error": str(error)}, context)
    
    async def _check_timeouts(self):
        """Check for execution timeouts"""
        for execution_id in list(self.active_executions):
            execution = self.executions.get(execution_id)
            if not execution:
                continue
            
            # Check overall timeout (1 hour default)
            if (datetime.now() - execution.started_at).seconds > 3600:
                execution.status = WorkflowStatus.FAILED
                execution.errors.append({
                    "error": "Workflow execution timeout",
                    "timestamp": datetime.now().isoformat()
                })
                self.active_executions.discard(execution_id)
    
    async def _update_metrics(self):
        """Update execution metrics"""
        if self.execution_history:
            durations = [
                (e.completed_at - e.started_at).total_seconds()
                for e in self.execution_history
                if e.completed_at
            ]
            
            if durations:
                self.metrics["average_duration"] = sum(durations) / len(durations)
    
    def register_handler(self, action: str, handler: Callable):
        """Register a step handler"""
        self.step_handlers[action] = handler
    
    async def pause_execution(self, execution_id: str):
        """Pause workflow execution"""
        if execution_id in self.executions:
            self.executions[execution_id].status = WorkflowStatus.PAUSED
    
    async def resume_execution(self, execution_id: str):
        """Resume workflow execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                asyncio.create_task(self._execute_workflow_async(execution_id))
    
    async def cancel_execution(self, execution_id: str):
        """Cancel workflow execution"""
        if execution_id in self.executions:
            self.executions[execution_id].status = WorkflowStatus.CANCELLED
            self.active_executions.discard(execution_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status"""
        return self.executions.get(execution_id)
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics"""
        return {
            **self.metrics,
            "active_executions": len(self.active_executions),
            "total_workflows": len(self.workflows)
        }
    
    # Default handlers
    async def _handle_log(self, parameters: Dict[str, Any],
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Log message handler"""
        message = parameters.get("message", "")
        level = parameters.get("level", "info")
        print(f"[{level.upper()}] {message}")
        return {"logged": True}
    
    async def _handle_wait(self, parameters: Dict[str, Any],
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Wait handler"""
        seconds = parameters.get("seconds", 1)
        await asyncio.sleep(seconds)
        return {"waited": seconds}
    
    async def _handle_http_request(self, parameters: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP request handler"""
        # Placeholder for HTTP request
        return {"status": 200, "body": {}}
    
    async def _handle_transform(self, parameters: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data handler"""
        return self._transform_data(parameters, context)
    
    async def _handle_validate(self, parameters: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data handler"""
        return self._validate_data(parameters, context)
    
    async def _handle_notify(self, parameters: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification handler"""
        return {"notified": True}
    
    def _transform_data(self, parameters: Dict[str, Any],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data based on rules"""
        transformations = parameters.get("transformations", [])
        result = {}
        
        for transform in transformations:
            source = self._evaluate_expression(transform.get("source"), context)
            target = transform.get("target")
            operation = transform.get("operation", "copy")
            
            if operation == "copy":
                result[target] = source
            elif operation == "uppercase":
                result[target] = str(source).upper()
            elif operation == "lowercase":
                result[target] = str(source).lower()
            elif operation == "concat":
                separator = transform.get("separator", "")
                result[target] = separator.join(map(str, source))
            elif operation == "split":
                separator = transform.get("separator", ",")
                result[target] = str(source).split(separator)
            elif operation == "json_parse":
                result[target] = json.loads(source)
            elif operation == "json_stringify":
                result[target] = json.dumps(source)
        
        return result
    
    def _validate_data(self, parameters: Dict[str, Any],
                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data based on rules"""
        validations = parameters.get("validations", [])
        errors = []
        
        for validation in validations:
            field = validation.get("field")
            value = self._evaluate_expression(f"{{{{{field}}}}}", context)
            rule = validation.get("rule")
            
            if rule == "required" and not value:
                errors.append(f"{field} is required")
            elif rule == "email":
                import re
                if not re.match(r"[^@]+@[^@]+\.[^@]+", str(value)):
                    errors.append(f"{field} is not a valid email")
            elif rule == "min_length":
                min_len = validation.get("value", 0)
                if len(str(value)) < min_len:
                    errors.append(f"{field} must be at least {min_len} characters")
            elif rule == "max_length":
                max_len = validation.get("value", 100)
                if len(str(value)) > max_len:
                    errors.append(f"{field} must be at most {max_len} characters")
            elif rule == "regex":
                import re
                pattern = validation.get("pattern")
                if not re.match(pattern, str(value)):
                    errors.append(f"{field} does not match required pattern")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _execute_parallel_step(self, step_id: str,
                                   context: Dict[str, Any]) -> Any:
        """Execute a parallel step"""
        # Placeholder for parallel step execution
        return {"step_id": step_id, "result": "completed"}
    
    async def _execute_loop_body(self, step: WorkflowStep,
                               context: Dict[str, Any]) -> Any:
        """Execute loop body"""
        # Placeholder for loop body execution
        return {"loop_iteration": "completed"}
    
    async def _wait_for_approval(self, step: WorkflowStep,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for approval"""
        # Placeholder for approval mechanism
        return {"approved": True, "approver": "system"}
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle workflow event"""
        # Process event and trigger appropriate workflows
        event_type = event.get("type")
        
        for workflow in self.workflows.values():
            for trigger in workflow.triggers:
                if trigger.trigger_type == TriggerType.EVENT:
                    pattern = trigger.event_pattern or {}
                    if self._match_event_pattern(event, pattern):
                        await self.execute_workflow(workflow.workflow_id, event)