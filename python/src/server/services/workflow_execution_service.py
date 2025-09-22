"""
Workflow Execution Engine Service

Handles the actual execution of workflows including:
- Step-by-step workflow execution
- Agent assignment and task execution
- Parallel execution support
- Error handling and retry logic
- Real-time status updates
- Integration with Agency Swarm components

Following Archon server patterns and error handling standards
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ...auth.utils.dependencies import get_db_session as get_db
from ...database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution,
    ExecutionStatus, StepType, AgentType, ModelTier
)
from ...database.agent_models import AgentV3, AgentState
from ..config.config import get_config

# Import Socket.IO broadcast functions
try:
    from ..api_routes.workflow_socketio import (
        broadcast_execution_started,
        broadcast_execution_progress,
        broadcast_execution_step_started,
        broadcast_execution_step_completed,
        broadcast_execution_step_failed,
        broadcast_execution_completed,
        broadcast_execution_failed,
        broadcast_execution_paused,
        broadcast_execution_resumed,
        broadcast_execution_cancelled
    )
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExecutionState(str, Enum):
    """Execution state for the engine"""
    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ExecutionContext:
    """Context for workflow execution"""
    execution_id: str
    workflow_id: str
    workflow: WorkflowDefinition
    execution: WorkflowExecution
    parameters: Dict[str, Any]
    step_results: Dict[str, Any]
    errors: List[str]
    state: ExecutionState = ExecutionState.IDLE

class WorkflowExecutionEngine:
    """Main workflow execution engine"""

    def __init__(self):
        self.settings = get_config()
        self.active_contexts: Dict[str, ExecutionContext] = {}
        self.execution_queue = asyncio.Queue()
        self.running = False
        self.max_concurrent_executions = 10
        self.agent_service_url = "http://archon-agents:8052"

    async def start(self):
        """Start the execution engine"""
        if self.running:
            return

        self.running = True
        logger.info("Starting workflow execution engine")

        # Start worker tasks
        for i in range(3):  # 3 worker tasks
            asyncio.create_task(self._execution_worker(f"worker-{i}"))

        # Start status updater
        asyncio.create_task(self._status_updater())

    async def stop(self):
        """Stop the execution engine"""
        self.running = False
        logger.info("Stopping workflow execution engine")

        # Wait for active executions to complete or timeout
        timeout = 30  # 30 seconds
        start_time = time.time()

        while len(self.active_contexts) > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)

        if len(self.active_contexts) > 0:
            logger.warning(f"{len(self.active_contexts)} executions still active after timeout")

    async def execute_workflow(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        parameters: Dict[str, Any],
        db: Session
    ) -> WorkflowExecution:
        """Execute a workflow"""
        try:
            # Get or create execution context
            if execution_id not in self.active_contexts:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.execution_id == execution_id
                ).first()

                if not execution:
                    raise ValueError(f"Execution not found: {execution_id}")

                context = ExecutionContext(
                    execution_id=execution_id,
                    workflow_id=str(workflow.id),
                    workflow=workflow,
                    execution=execution,
                    parameters=parameters,
                    step_results={},
                    errors=[]
                )
                self.active_contexts[execution_id] = context

            else:
                context = self.active_contexts[execution_id]

            # Add to execution queue
            await self.execution_queue.put(("execute", execution_id))

            logger.info(f"Queued workflow execution: {execution_id}")
            return execution

        except Exception as e:
            logger.error(f"Failed to queue workflow execution {execution_id}: {e}")
            raise

    async def _execution_worker(self, worker_name: str):
        """Worker task for processing execution queue"""
        logger.info(f"Started execution worker: {worker_name}")

        while self.running:
            try:
                # Get next execution from queue
                task = await asyncio.wait_for(self.execution_queue.get(), timeout=1.0)
                command, execution_id = task

                if command == "execute":
                    await self._process_execution(execution_id)

            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue

        logger.info(f"Stopped execution worker: {worker_name}")

    async def _process_execution(self, execution_id: str):
        """Process a workflow execution"""
        db = next(get_db())
        try:
            context = self.active_contexts.get(execution_id)
            if not context:
                logger.error(f"Execution context not found: {execution_id}")
                return

            # Update execution status
            context.state = ExecutionState.EXECUTING
            context.execution.status = ExecutionStatus.RUNNING
            context.execution.started_at = datetime.now()
            db.commit()

            # Broadcast execution started event
            if SOCKETIO_AVAILABLE:
                try:
                    asyncio.create_task(broadcast_execution_started({
                        "id": execution_id,
                        "workflow_id": context.workflow_id,
                        "status": ExecutionStatus.RUNNING.value,
                        "started_at": context.execution.started_at.isoformat(),
                        "parameters": context.parameters
                    }))
                except Exception as e:
                    logger.warning(f"Failed to broadcast execution started: {e}")

            logger.info(f"Starting workflow execution: {execution_id}")

            # Execute workflow steps
            await self._execute_workflow_steps(context, db)

            # Complete execution
            if context.execution.status == ExecutionStatus.RUNNING:
                context.execution.status = ExecutionStatus.COMPLETED
                context.execution.completed_at = datetime.now()

                # Calculate final progress
                total_steps = len(context.workflow.flow_data.get("nodes", []))
                context.execution.progress = 1.0

                logger.info(f"Workflow execution completed: {execution_id}")

            elif context.execution.status == ExecutionStatus.PAUSED:
                logger.info(f"Workflow execution paused: {execution_id}")

            # Update execution results
            context.execution.results = context.step_results
            context.execution.errors = context.errors

            db.commit()

            # Clean up context
            if context.execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                del self.active_contexts[execution_id]

        except Exception as e:
            logger.error(f"Error processing execution {execution_id}: {e}")
            await self._handle_execution_error(execution_id, str(e), db)

        finally:
            db.close()

    async def _execute_workflow_steps(self, context: ExecutionContext, db: Session):
        """Execute all steps in the workflow"""
        try:
            # Get workflow steps from ReactFlow data
            nodes = context.workflow.flow_data.get("nodes", [])
            edges = context.workflow.flow_data.get("edges", [])

            # Build execution graph
            execution_graph = self._build_execution_graph(nodes, edges)

            # Execute steps in order
            executed_steps = set()
            total_steps = len(nodes)

            while len(executed_steps) < total_steps:
                # Find steps ready for execution
                ready_steps = self._find_ready_steps(execution_graph, executed_steps)

                if not ready_steps:
                    # Check if we're stuck (circular dependency or missing dependencies)
                    if self._has_unresolved_dependencies(execution_graph, executed_steps):
                        raise RuntimeError("Workflow has unresolved dependencies")
                    else:
                        break  # No more steps to execute

                # Execute ready steps (potentially in parallel)
                await self._execute_steps_parallel(context, ready_steps, execution_graph, db)

                # Update progress
                context.execution.progress = len(executed_steps) / total_steps
                db.commit()

                # Check if execution should pause or stop
                if context.state != ExecutionState.EXECUTING:
                    break

                # Add small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error executing workflow steps: {e}")
            raise

    def _build_execution_graph(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build execution graph from ReactFlow data"""
        graph = {}

        # Create nodes
        for node in nodes:
            node_id = node["id"]
            node_data = node.get("data", {})

            graph[node_id] = {
                "id": node_id,
                "type": node.get("type"),
                "data": node_data,
                "position": node.get("position", {"x": 0, "y": 0}),
                "dependencies": [],
                "next_steps": [],
                "executed": False,
                "result": None
            }

        # Add edges
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")

            if source in graph and target in graph:
                graph[source]["next_steps"].append(target)
                graph[target]["dependencies"].append(source)

        return graph

    def _find_ready_steps(self, graph: Dict[str, Dict[str, Any]], executed_steps: Set[str]) -> List[str]:
        """Find steps ready for execution (all dependencies completed)"""
        ready_steps = []

        for step_id, step_data in graph.items():
            if step_data["executed"]:
                continue

            # Check if all dependencies are executed
            dependencies_met = all(
                dep in executed_steps
                for dep in step_data["dependencies"]
            )

            if dependencies_met:
                ready_steps.append(step_id)

        return ready_steps

    def _has_unresolved_dependencies(self, graph: Dict[str, Dict[str, Any]], executed_steps: Set[str]) -> bool:
        """Check if there are unresolved dependencies"""
        for step_id, step_data in graph.items():
            if step_data["executed"]:
                continue

            # Check if any dependency is not executed and not in graph (missing)
            for dep in step_data["dependencies"]:
                if dep not in executed_steps and dep not in graph:
                    return True

        return False

    async def _execute_steps_parallel(
        self,
        context: ExecutionContext,
        step_ids: List[str],
        graph: Dict[str, Dict[str, Any]],
        db: Session
    ):
        """Execute multiple steps in parallel"""
        if not step_ids:
            return

        # Create tasks for parallel execution
        tasks = []
        for step_id in step_ids:
            task = asyncio.create_task(self._execute_single_step(context, step_id, graph[step_id], db))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for step_id, result in zip(step_ids, results):
            if isinstance(result, Exception):
                # Step failed
                logger.error(f"Step {step_id} failed: {result}")
                context.errors.append(f"Step {step_id} failed: {str(result)}")
                graph[step_id]["result"] = {"error": str(result)}
            else:
                # Step completed successfully
                graph[step_id]["executed"] = True
                graph[step_id]["result"] = result
                context.step_results[step_id] = result

    async def _execute_single_step(
        self,
        context: ExecutionContext,
        step_id: str,
        step_data: Dict[str, Any],
        db: Session
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            # Create step execution record
            step_execution = StepExecution(
                workflow_execution_id=context.execution.id,
                step_id=step_id,
                status=ExecutionStatus.PENDING,
                input_data=context.parameters,
                started_at=datetime.now()
            )

            db.add(step_execution)
            db.flush()

            # Update current step in execution
            context.execution.current_step_id = step_id
            db.commit()

            # Execute based on step type
            step_type = self._map_node_type_to_step_type(step_data["type"])

            if step_type == StepType.AGENT_TASK:
                result = await self._execute_agent_task(context, step_data, step_execution, db)
            elif step_type == StepType.DECISION:
                result = await self._execute_decision_step(context, step_data, step_execution, db)
            elif step_type == StepType.PARALLEL:
                result = await self._execute_parallel_step(context, step_data, step_execution, db)
            elif step_type == StepType.API_CALL:
                result = await self._execute_api_call(context, step_data, step_execution, db)
            elif step_type == StepType.DELAY:
                result = await self._execute_delay_step(context, step_data, step_execution, db)
            elif step_type == StepType.NOTIFICATION:
                result = await self._execute_notification_step(context, step_data, step_execution, db)
            else:
                # Default agent task
                result = await self._execute_agent_task(context, step_data, step_execution, db)

            # Update step execution record
            step_execution.status = ExecutionStatus.COMPLETED
            step_execution.output_data = result
            step_execution.completed_at = datetime.now()
            step_execution.execution_time_seconds = int((step_execution.completed_at - step_execution.started_at).total_seconds())

            db.commit()

            return result

        except Exception as e:
            # Update step execution record with error
            step_execution.status = ExecutionStatus.FAILED
            step_execution.error_message = str(e)
            step_execution.completed_at = datetime.now()
            db.commit()

            raise

    def _map_node_type_to_step_type(self, node_type: str) -> StepType:
        """Map ReactFlow node type to workflow step type"""
        type_mapping = {
            "agent": StepType.AGENT_TASK,
            "decision": StepType.DECISION,
            "parallel": StepType.PARALLEL,
            "delay": StepType.DELAY,
            "api": StepType.API_CALL,
            "transform": StepType.DATA_TRANSFORM,
            "input": StepType.HUMAN_INPUT,
            "notification": StepType.NOTIFICATION,
            "subworkflow": StepType.SUB_WORKFLOW
        }
        return type_mapping.get(node_type, StepType.AGENT_TASK)

    async def _execute_agent_task(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute an agent task"""
        try:
            node_data = step_data["data"]
            task_description = node_data.get("description", f"Execute {step_data['id']}")
            task_prompt = node_data.get("prompt", task_description)

            # Determine agent type and model tier
            agent_type = self._get_agent_type_from_step(node_data) or context.workflow.default_agent_type
            model_tier = self._get_model_tier_from_step(node_data) or context.workflow.default_model_tier

            # Prepare agent request
            agent_request = {
                "agent_type": agent_type.value if agent_type else "GENERAL_PURPOSE",
                "prompt": task_prompt,
                "context": {
                    "workflow_id": context.workflow_id,
                    "execution_id": context.execution_id,
                    "step_id": step_data["id"],
                    "parameters": context.parameters,
                    "step_results": context.step_results
                }
            }

            # Call agent service
            async with httpx.AsyncClient(timeout=300) as client:  # 5 minute timeout
                response = await client.post(
                    f"{self.agent_service_url}/agents/run",
                    json=agent_request,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    agent_result = response.json()

                    # Update step execution with agent info
                    step_execution.assigned_agent_id = agent_result.get("agent_id")
                    step_execution.agent_type = agent_type
                    step_execution.model_tier = model_tier
                    step_execution.tokens_used = agent_result.get("tokens_used", 0)
                    step_execution.cost_usd = agent_result.get("cost_usd", 0.0)

                    return agent_result.get("result", {})
                else:
                    raise Exception(f"Agent service returned {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Failed to execute agent task {step_data['id']}: {e}")
            raise

    async def _execute_decision_step(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute a decision step"""
        try:
            node_data = step_data["data"]
            conditions = node_data.get("conditions", {})
            branches = node_data.get("branches", {})

            # Evaluate conditions
            evaluation_result = self._evaluate_conditions(conditions, context.step_results)

            # Determine next path
            next_path = "default"  # Default path
            for condition, path in branches.items():
                if self._evaluate_condition(condition, context.step_results):
                    next_path = path
                    break

            return {
                "decision": evaluation_result,
                "next_path": next_path,
                "branches": branches
            }

        except Exception as e:
            logger.error(f"Failed to execute decision step {step_data['id']}: {e}")
            raise

    async def _execute_parallel_step(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute a parallel step (for now, just pass through)"""
        try:
            node_data = step_data["data"]
            parallel_steps = node_data.get("parallel_steps", [])

            # For now, return step configuration
            # In a full implementation, this would execute sub-steps in parallel
            return {
                "parallel_steps": parallel_steps,
                "executed": len(parallel_steps),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Failed to execute parallel step {step_data['id']}: {e}")
            raise

    async def _execute_api_call(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute an API call step"""
        try:
            node_data = step_data["data"]
            url = node_data.get("url")
            method = node_data.get("method", "GET").upper()
            headers = node_data.get("headers", {})
            body = node_data.get("body", {})

            if not url:
                raise ValueError("API call step requires a URL")

            # Make API call
            async with httpx.AsyncClient(timeout=30) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, json=body, headers=headers)
                elif method == "PUT":
                    response = await client.put(url, json=body, headers=headers)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code >= 400:
                    raise Exception(f"API call failed: {response.status_code} - {response.text}")

                return {
                    "status_code": response.status_code,
                    "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "headers": dict(response.headers)
                }

        except Exception as e:
            logger.error(f"Failed to execute API call {step_data['id']}: {e}")
            raise

    async def _execute_delay_step(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute a delay step"""
        try:
            node_data = step_data["data"]
            delay_seconds = node_data.get("delay_seconds", 10)

            await asyncio.sleep(delay_seconds)

            return {
                "delayed_seconds": delay_seconds,
                "completed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to execute delay step {step_data['id']}: {e}")
            raise

    async def _execute_notification_step(
        self,
        context: ExecutionContext,
        step_data: Dict[str, Any],
        step_execution: StepExecution,
        db: Session
    ) -> Dict[str, Any]:
        """Execute a notification step"""
        try:
            node_data = step_data["data"]
            message = node_data.get("message", "Workflow notification")
            recipients = node_data.get("recipients", [])
            notification_type = node_data.get("type", "info")

            # For now, just log the notification
            # In a full implementation, this would send emails, Slack messages, etc.
            logger.info(f"Notification: {message} to {recipients} (type: {notification_type})")

            return {
                "message": message,
                "recipients": recipients,
                "type": notification_type,
                "sent_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to execute notification step {step_data['id']}: {e}")
            raise

    def _get_agent_type_from_step(self, node_data: Dict[str, Any]) -> Optional[AgentType]:
        """Get agent type from step data"""
        agent_type = node_data.get("agentType")
        if agent_type and agent_type in AgentType.__members__:
            return AgentType(agent_type)
        return None

    def _get_model_tier_from_step(self, node_data: Dict[str, Any]) -> Optional[ModelTier]:
        """Get model tier from step data"""
        model_tier = node_data.get("modelTier")
        if model_tier and model_tier in ModelTier.__members__:
            return ModelTier(model_tier)
        return None

    def _evaluate_conditions(self, conditions: Dict[str, Any], step_results: Dict[str, Any]) -> bool:
        """Evaluate multiple conditions with AND logic"""
        for condition in conditions.get("conditions", []):
            if not self._evaluate_condition(condition, step_results):
                return False
        return True

    def _evaluate_condition(self, condition: Union[str, Dict[str, Any]], step_results: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        if isinstance(condition, str):
            # Simple string condition - check if step exists and has truthy result
            step_result = step_results.get(condition)
            return bool(step_result)

        elif isinstance(condition, dict):
            # Complex condition
            field = condition.get("field")
            operator = condition.get("operator", "equals")
            value = condition.get("value")

            if field.startswith("steps."):
                # Reference to step result
                step_id = field.split(".", 1)[1]
                step_result = step_results.get(step_id, {})
                actual_value = step_result.get("value", step_result)
            else:
                # Direct value from parameters
                actual_value = context.parameters.get(field)

            # Apply operator
            if operator == "equals":
                return actual_value == value
            elif operator == "not_equals":
                return actual_value != value
            elif operator == "contains":
                return value in str(actual_value) if actual_value else False
            elif operator == "greater_than":
                return float(actual_value or 0) > float(value)
            elif operator == "less_than":
                return float(actual_value or 0) < float(value)
            elif operator == "exists":
                return actual_value is not None
            elif operator == "not_exists":
                return actual_value is None

        return False

    async def _handle_execution_error(self, execution_id: str, error: str, db: Session):
        """Handle execution errors"""
        try:
            context = self.active_contexts.get(execution_id)
            if context:
                context.execution.status = ExecutionStatus.FAILED
                context.execution.completed_at = datetime.now()
                context.execution.errors = context.errors + [error]
                context.state = ExecutionState.ERROR

                db.commit()

                # Clean up context
                del self.active_contexts[execution_id]

            logger.error(f"Execution failed: {execution_id} - {error}")

        except Exception as e:
            logger.error(f"Error handling execution error: {e}")

    async def _status_updater(self):
        """Periodic status update task"""
        while self.running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                # Update status for all active executions
                for execution_id, context in self.active_contexts.items():
                    if context.state == ExecutionState.EXECUTING:
                        # Calculate runtime
                        if context.execution.started_at:
                            runtime = (datetime.now() - context.execution.started_at).total_seconds()

                            # Check timeout
                            if context.workflow.timeout_seconds and runtime > context.workflow.timeout_seconds:
                                await self._handle_execution_error(execution_id, "Execution timeout", next(get_db()))

            except Exception as e:
                logger.error(f"Error in status updater: {e}")

    async def pause_execution(self, execution_id: str):
        """Pause a running execution"""
        context = self.active_contexts.get(execution_id)
        if context and context.state == ExecutionState.EXECUTING:
            context.state = ExecutionState.PAUSED

            # Broadcast pause event
            if SOCKETIO_AVAILABLE:
                try:
                    asyncio.create_task(broadcast_execution_paused(execution_id, context.workflow_id))
                except Exception as e:
                    logger.warning(f"Failed to broadcast execution paused: {e}")

            logger.info(f"Paused execution: {execution_id}")

    async def resume_execution(self, execution_id: str):
        """Resume a paused execution"""
        context = self.active_contexts.get(execution_id)
        if context and context.state == ExecutionState.PAUSED:
            context.state = ExecutionState.EXECUTING
            # Re-add to execution queue
            await self.execution_queue.put(("execute", execution_id))

            # Broadcast resume event
            if SOCKETIO_AVAILABLE:
                try:
                    asyncio.create_task(broadcast_execution_resumed(execution_id, context.workflow_id))
                except Exception as e:
                    logger.warning(f"Failed to broadcast execution resumed: {e}")

            logger.info(f"Resumed execution: {execution_id}")

    async def cancel_execution(self, execution_id: str):
        """Cancel an execution"""
        context = self.active_contexts.get(execution_id)
        if context:
            workflow_id = context.workflow_id
            context.state = ExecutionState.STOPPING

            # Broadcast cancel event
            if SOCKETIO_AVAILABLE:
                try:
                    asyncio.create_task(broadcast_execution_cancelled(execution_id, workflow_id))
                except Exception as e:
                    logger.warning(f"Failed to broadcast execution cancelled: {e}")

            await self._handle_execution_error(execution_id, "Execution cancelled", next(get_db()))

# Global engine instance
execution_engine: Optional[WorkflowExecutionEngine] = None

async def get_workflow_execution_engine() -> WorkflowExecutionEngine:
    """Get or create workflow execution engine instance"""
    global execution_engine
    if execution_engine is None:
        execution_engine = WorkflowExecutionEngine()
        await execution_engine.start()
    return execution_engine

async def stop_workflow_execution_engine():
    """Stop the workflow execution engine"""
    global execution_engine
    if execution_engine:
        await execution_engine.stop()
        execution_engine = None

# Service wrapper for compatibility
class WorkflowExecutionService:
    """Service wrapper for workflow execution"""

    def __init__(self):
        self.engine = None

    async def execute_workflow(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        parameters: Dict[str, Any]
    ) -> WorkflowExecution:
        """Execute a workflow using the engine"""
        if not self.engine:
            self.engine = await get_workflow_execution_engine()

        db = next(get_db())
        try:
            return await self.engine.execute_workflow(execution_id, workflow, parameters, db)
        finally:
            db.close()

    async def pause_execution(self, execution_id: str):
        """Pause a workflow execution"""
        if self.engine:
            await self.engine.pause_execution(execution_id)

    async def resume_execution(self, execution_id: str):
        """Resume a workflow execution"""
        if self.engine:
            await self.engine.resume_execution(execution_id)

    async def cancel_execution(self, execution_id: str):
        """Cancel a workflow execution"""
        if self.engine:
            await self.engine.cancel_execution(execution_id)