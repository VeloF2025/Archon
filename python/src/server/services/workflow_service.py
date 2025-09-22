"""
Workflow Management Service

Provides business logic for workflow management including:
- CRUD operations for workflows
- ReactFlow data format compatibility
- Workflow validation and versioning
- Integration with agent management system
- Real-time workflow status tracking

Following Archon server patterns and error handling standards
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import httpx
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from pydantic import ValidationError

from ...auth.utils.dependencies import get_db_session
from ...database.workflow_models import (
    WorkflowDefinition, WorkflowVersion, WorkflowStep,
    WorkflowExecution, StepExecution, WorkflowMetrics,
    WorkflowAnalytics, WorkflowSchedule, WebhookTrigger, EventTrigger,
    WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecutionRequest,
    WorkflowExecutionResponse, WorkflowStatusResponse, WorkflowAnalyticsResponse,
    ReactFlowData, validate_reactflow_data, WorkflowStatus, ExecutionStatus, StepType,
    TriggerType, AgentType, ModelTier, identify_workflow_bottlenecks
)
from ...database.agent_models import AgentV3, AgentState
from ..config.config import get_config

logger = logging.getLogger(__name__)

class WorkflowService:
    """Main workflow management service"""

    def __init__(self):
        self.settings = get_config()
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_locks: Dict[str, asyncio.Lock] = {}

    async def create_workflow(
        self,
        request: WorkflowCreateRequest,
        created_by: str,
        project_id: str,
        db: Session
    ) -> Tuple[bool, Union[WorkflowDefinition, str]]:
        """Create a new workflow definition"""
        try:
            # Validate ReactFlow data
            is_valid, errors = validate_reactflow_data(request.flow_data)
            if not is_valid:
                return False, f"Invalid ReactFlow data: {', '.join(errors)}"

            # Create workflow definition
            workflow = WorkflowDefinition(
                name=request.name,
                description=request.description,
                flow_data=request.flow_data,
                status=WorkflowStatus.DRAFT,
                trigger_type=request.trigger_type,
                default_agent_type=request.default_agent_type,
                default_model_tier=request.default_model_tier,
                timeout_seconds=request.timeout_seconds,
                tags=request.tags,
                project_id=uuid.UUID(project_id),
                created_by=created_by,
                is_public=request.is_public
            )

            db.add(workflow)
            db.flush()  # Get workflow ID

            # Extract and create workflow steps from ReactFlow nodes
            await self._create_workflow_steps(workflow, request.flow_data, db)

            # Create initial version
            self._create_workflow_version(workflow, "Initial version", created_by, db)

            db.commit()
            logger.info(f"Created workflow: {workflow.name} ({workflow.id})")

            return True, workflow

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create workflow: {e}")
            return False, f"Failed to create workflow: {str(e)}"

    async def get_workflow(self, workflow_id: str, db: Session) -> Optional[WorkflowDefinition]:
        """Get workflow by ID"""
        try:
            workflow = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.id == uuid.UUID(workflow_id)
            ).first()

            if workflow:
                # Update accessed_at timestamp
                workflow.updated_at = datetime.now()
                db.commit()

            return workflow

        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            return None

    async def list_workflows(
        self,
        project_id: str,
        status: Optional[WorkflowStatus] = None,
        trigger_type: Optional[TriggerType] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = None
    ) -> Tuple[List[WorkflowDefinition], int]:
        """List workflows with filtering"""
        try:
            if not db:
                db = next(get_db_session())

            query = db.query(WorkflowDefinition).filter(
                WorkflowDefinition.project_id == uuid.UUID(project_id)
            )

            # Apply filters
            if status:
                query = query.filter(WorkflowDefinition.status == status)
            if trigger_type:
                query = query.filter(WorkflowDefinition.trigger_type == trigger_type)

            # Get total count
            total = query.count()

            # Apply pagination
            workflows = query.order_by(desc(WorkflowDefinition.created_at)).offset(skip).limit(limit).all()

            return workflows, total

        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return [], 0

    async def update_workflow(
        self,
        workflow_id: str,
        request: WorkflowUpdateRequest,
        updated_by: str,
        db: Session
    ) -> Tuple[bool, Union[WorkflowDefinition, str]]:
        """Update workflow definition"""
        try:
            workflow = await self.get_workflow(workflow_id, db)
            if not workflow:
                return False, "Workflow not found"

            # Store original data for version tracking
            original_data = workflow.flow_data.copy()
            original_name = workflow.name

            # Update fields
            if request.name:
                workflow.name = request.name
            if request.description is not None:
                workflow.description = request.description
            if request.flow_data:
                # Validate ReactFlow data
                is_valid, errors = validate_reactflow_data(request.flow_data)
                if not is_valid:
                    return False, f"Invalid ReactFlow data: {', '.join(errors)}"
                workflow.flow_data = request.flow_data
            if request.status:
                workflow.status = request.status
            if request.tags is not None:
                workflow.tags = request.tags
            if request.timeout_seconds:
                workflow.timeout_seconds = request.timeout_seconds

            workflow.updated_at = datetime.now()

            # Update workflow steps if flow_data changed
            if request.flow_data and request.flow_data != original_data:
                # Delete existing steps
                db.query(WorkflowStep).filter(WorkflowStep.workflow_id == workflow.id).delete()
                # Create new steps
                await self._create_workflow_steps(workflow, request.flow_data, db)

            # Create new version if significant changes
            if (request.flow_data and request.flow_data != original_data) or \
               (request.name and request.name != original_name):
                change_reason = []
                if request.name and request.name != original_name:
                    change_reason.append(f"Renamed from '{original_name}' to '{request.name}'")
                if request.flow_data and request.flow_data != original_data:
                    change_reason.append("Flow structure updated")

                self._create_workflow_version(
                    workflow,
                    ", ".join(change_reason),
                    updated_by,
                    db
                )

            db.commit()
            logger.info(f"Updated workflow: {workflow.name} ({workflow.id})")

            return True, workflow

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            return False, f"Failed to update workflow: {str(e)}"

    async def delete_workflow(self, workflow_id: str, db: Session) -> Tuple[bool, str]:
        """Delete workflow definition"""
        try:
            workflow = await self.get_workflow(workflow_id, db)
            if not workflow:
                return False, "Workflow not found"

            # Check if workflow is currently running
            active_executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == workflow.id,
                    WorkflowExecution.status.in_([ExecutionStatus.RUNNING, ExecutionStatus.PENDING])
                )
            ).count()

            if active_executions > 0:
                return False, f"Cannot delete workflow: {active_executions} active executions"

            # Delete workflow (cascade will handle related records)
            db.delete(workflow)
            db.commit()

            logger.info(f"Deleted workflow: {workflow.name} ({workflow.id})")
            return True, "Workflow deleted successfully"

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            return False, f"Failed to delete workflow: {str(e)}"

    async def execute_workflow(
        self,
        request: WorkflowExecutionRequest,
        triggered_by: str,
        db: Session
    ) -> Tuple[bool, Union[WorkflowExecutionResponse, str]]:
        """Execute a workflow"""
        try:
            # Get workflow definition
            workflow = await self.get_workflow(str(request.workflow_id), db)
            if not workflow:
                return False, "Workflow not found"

            # Check if workflow can be executed
            if workflow.status != WorkflowStatus.PUBLISHED:
                return False, f"Workflow must be PUBLISHED to execute (current: {workflow.status})"

            # Create execution instance
            execution_id = f"wf_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            execution = WorkflowExecution(
                workflow_id=workflow.id,
                execution_id=execution_id,
                triggered_by=triggered_by,
                trigger_data=request.trigger_data,
                status=ExecutionStatus.PENDING,
                estimated_duration=self._estimate_workflow_duration(workflow)
            )

            db.add(execution)
            db.flush()

            # Store in active executions
            self.active_executions[execution_id] = execution
            self.execution_locks[execution_id] = asyncio.Lock()

            db.commit()

            # Start execution asynchronously
            if request.async_execution:
                asyncio.create_task(self._execute_workflow_async(execution_id, workflow, request.parameters))
                return True, WorkflowExecutionResponse(
                    execution_id=execution_id,
                    workflow_id=workflow.id,
                    status=ExecutionStatus.PENDING,
                    started_at=datetime.now(),
                    estimated_duration=execution.estimated_duration,
                    message="Workflow execution started"
                )
            else:
                # Execute synchronously
                result = await self._execute_workflow_sync(execution_id, workflow, request.parameters)
                return True, result

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to execute workflow: {e}")
            return False, f"Failed to execute workflow: {str(e)}"

    async def get_execution_status(self, execution_id: str, db: Session) -> Optional[WorkflowStatusResponse]:
        """Get workflow execution status"""
        try:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution_id
            ).first()

            if not execution:
                return None

            # Get step executions
            step_executions = db.query(StepExecution).filter(
                StepExecution.workflow_execution_id == execution.id
            ).all()

            # Calculate metrics
            metrics = self._calculate_execution_metrics(execution, step_executions)

            return WorkflowStatusResponse(
                execution_id=execution.execution_id,
                workflow_id=execution.workflow_id,
                status=execution.status,
                progress=execution.progress,
                current_step_id=execution.current_step_id,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                results=execution.results,
                errors=execution.errors,
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Failed to get execution status for {execution_id}: {e}")
            return None

    async def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        skip: int = 0,
        limit: int = 100,
        db: Session = None
    ) -> Tuple[List[WorkflowExecution], int]:
        """List workflow executions"""
        try:
            if not db:
                db = next(get_db_session())

            query = db.query(WorkflowExecution)

            # Apply filters
            if workflow_id:
                query = query.filter(WorkflowExecution.workflow_id == uuid.UUID(workflow_id))
            if status:
                query = query.filter(WorkflowExecution.status == status)

            # Get total count
            total = query.count()

            # Apply pagination
            executions = query.order_by(desc(WorkflowExecution.created_at)).offset(skip).limit(limit).all()

            return executions, total

        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return [], 0

    async def get_workflow_analytics(
        self,
        workflow_id: str,
        period_days: int = 30,
        db: Session = None
    ) -> Optional[WorkflowAnalyticsResponse]:
        """Get workflow analytics for the specified period"""
        try:
            if not db:
                db = next(get_db_session())

            period_end = datetime.now()
            period_start = period_end - timedelta(days=period_days)

            # Get execution metrics
            executions = db.query(WorkflowExecution).filter(
                and_(
                    WorkflowExecution.workflow_id == uuid.UUID(workflow_id),
                    WorkflowExecution.created_at >= period_start,
                    WorkflowExecution.created_at <= period_end
                )
            ).all()

            if not executions:
                return None

            # Calculate analytics
            total_executions = len(executions)
            successful_executions = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
            failed_executions = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)

            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0

            # Average execution time
            completed_executions = [e for e in executions if e.execution_time_seconds]
            avg_execution_time = sum(e.execution_time_seconds or 0 for e in completed_executions) / len(completed_executions) if completed_executions else 0.0

            # Total cost
            total_cost = sum(
                sum(se.cost_usd or 0 for se in db.query(StepExecution).filter(
                    StepExecution.workflow_execution_id == e.id
                ).all())
                for e in executions
            )

            # Get step executions for bottleneck analysis
            all_step_executions = []
            for execution in executions:
                step_execs = db.query(StepExecution).filter(
                    StepExecution.workflow_execution_id == execution.id
                ).all()
                all_step_executions.extend(step_execs)

            bottlenecks = identify_workflow_bottlenecks(all_step_executions)

            # Generate recommendations
            recommendations = self._generate_workflow_recommendations(
                executions, all_step_executions, success_rate, avg_execution_time
            )

            return WorkflowAnalyticsResponse(
                workflow_id=uuid.UUID(workflow_id),
                period_start=period_start,
                period_end=period_end,
                total_executions=total_executions,
                success_rate=success_rate,
                avg_execution_time=avg_execution_time,
                total_cost=total_cost,
                performance_score=self._calculate_performance_score(
                    success_rate, avg_execution_time, total_cost / total_executions if total_executions > 0 else 0
                ),
                bottleneck_steps=bottlenecks,
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Failed to get workflow analytics for {workflow_id}: {e}")
            return None

    async def pause_execution(self, execution_id: str, db: Session) -> Tuple[bool, str]:
        """Pause a running workflow execution"""
        try:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution_id
            ).first()

            if not execution:
                return False, "Execution not found"

            if execution.status != ExecutionStatus.RUNNING:
                return False, f"Cannot pause execution with status: {execution.status}"

            execution.status = ExecutionStatus.PAUSED
            db.commit()

            # Notify execution engine
            await self._notify_execution_control(execution_id, "pause")

            logger.info(f"Paused execution: {execution_id}")
            return True, "Execution paused successfully"

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to pause execution {execution_id}: {e}")
            return False, f"Failed to pause execution: {str(e)}"

    async def resume_execution(self, execution_id: str, db: Session) -> Tuple[bool, str]:
        """Resume a paused workflow execution"""
        try:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution_id
            ).first()

            if not execution:
                return False, "Execution not found"

            if execution.status != ExecutionStatus.PAUSED:
                return False, f"Cannot resume execution with status: {execution.status}"

            execution.status = ExecutionStatus.RUNNING
            db.commit()

            # Notify execution engine
            await self._notify_execution_control(execution_id, "resume")

            logger.info(f"Resumed execution: {execution_id}")
            return True, "Execution resumed successfully"

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to resume execution {execution_id}: {e}")
            return False, f"Failed to resume execution: {str(e)}"

    async def cancel_execution(self, execution_id: str, db: Session) -> Tuple[bool, str]:
        """Cancel a workflow execution"""
        try:
            execution = db.query(WorkflowExecution).filter(
                WorkflowExecution.execution_id == execution_id
            ).first()

            if not execution:
                return False, "Execution not found"

            if execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                return False, f"Cannot cancel execution with status: {execution.status}"

            execution.status = ExecutionStatus.CANCELLED
            execution.completed_at = datetime.now()
            db.commit()

            # Notify execution engine
            await self._notify_execution_control(execution_id, "cancel")

            logger.info(f"Cancelled execution: {execution_id}")
            return True, "Execution cancelled successfully"

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cancel execution {execution_id}: {e}")
            return False, f"Failed to cancel execution: {str(e)}"

    # Private helper methods

    async def _create_workflow_steps(self, workflow: WorkflowDefinition, flow_data: Dict[str, Any], db: Session):
        """Create workflow steps from ReactFlow nodes"""
        nodes = flow_data.get("nodes", [])

        for node in nodes:
            node_data = node.get("data", {})
            node_type = node.get("type")

            # Determine step type from node type
            step_type = self._map_node_type_to_step_type(node_type)

            step = WorkflowStep(
                workflow_id=workflow.id,
                step_id=node["id"],
                name=node_data.get("label", node["id"]),
                step_type=step_type,
                description=node_data.get("description"),
                parameters=node_data.get("parameters", {}),
                conditions=node_data.get("conditions", {}),
                agent_type=self._get_agent_type_from_node(node_data),
                model_tier=self._get_model_tier_from_node(node_data),
                position=node.get("position", {"x": 0, "y": 0}),
                depends_on=self._get_node_dependencies(node["id"], flow_data.get("edges", [])),
                next_steps=self._get_next_steps(node["id"], flow_data.get("edges", []))
            )

            db.add(step)

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

    def _get_agent_type_from_node(self, node_data: Dict[str, Any]) -> Optional[AgentType]:
        """Extract agent type from node data"""
        agent_type = node_data.get("agentType")
        if agent_type and agent_type in AgentType.__members__:
            return AgentType(agent_type)
        return None

    def _get_model_tier_from_node(self, node_data: Dict[str, Any]) -> Optional[ModelTier]:
        """Extract model tier from node data"""
        model_tier = node_data.get("modelTier")
        if model_tier and model_tier in ModelTier.__members__:
            return ModelTier(model_tier)
        return None

    def _get_node_dependencies(self, node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
        """Get node dependencies from edges"""
        dependencies = []
        for edge in edges:
            if edge.get("target") == node_id:
                dependencies.append(edge["source"])
        return dependencies

    def _get_next_steps(self, node_id: str, edges: List[Dict[str, Any]]) -> List[str]:
        """Get next steps from edges"""
        next_steps = []
        for edge in edges:
            if edge.get("source") == node_id:
                next_steps.append(edge["target"])
        return next_steps

    def _create_workflow_version(self, workflow: WorkflowDefinition, changelog: str, created_by: str, db: Session):
        """Create a new workflow version"""
        version = WorkflowVersion(
            workflow_id=workflow.id,
            version=workflow.version,
            flow_data=workflow.flow_data.copy(),
            changelog=changelog,
            created_by=created_by
        )
        db.add(version)

    def _estimate_workflow_duration(self, workflow: WorkflowDefinition) -> Optional[int]:
        """Estimate workflow execution duration in seconds"""
        # Simple estimation based on number of steps
        steps = workflow.flow_data.get("nodes", [])
        base_time = 30  # 30 seconds base
        step_time = 60  # 60 seconds per step
        return base_time + (len(steps) * step_time)

    async def _execute_workflow_async(self, execution_id: str, workflow: WorkflowDefinition, parameters: Dict[str, Any]):
        """Execute workflow asynchronously"""
        try:
            # Import here to avoid circular dependency
            from .workflow_execution_service import WorkflowExecutionService

            execution_service = WorkflowExecutionService()
            await execution_service.execute_workflow(execution_id, workflow, parameters)

        except Exception as e:
            logger.error(f"Async workflow execution failed for {execution_id}: {e}")
            # Update execution status to failed
            db = next(get_db_session())
            try:
                execution = db.query(WorkflowExecution).filter(
                    WorkflowExecution.execution_id == execution_id
                ).first()
                if execution:
                    execution.status = ExecutionStatus.FAILED
                    execution.completed_at = datetime.now()
                    execution.errors = [str(e)]
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update failed execution status: {db_error}")

    async def _execute_workflow_sync(self, execution_id: str, workflow: WorkflowDefinition, parameters: Dict[str, Any]) -> WorkflowExecutionResponse:
        """Execute workflow synchronously"""
        # Import here to avoid circular dependency
        from .workflow_execution_service import WorkflowExecutionService

        execution_service = WorkflowExecutionService()
        result = await execution_service.execute_workflow(execution_id, workflow, parameters)

        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id=workflow.id,
            status=result.status if hasattr(result, 'status') else ExecutionStatus.COMPLETED,
            started_at=result.started_at if hasattr(result, 'started_at') else datetime.now(),
            estimated_duration=result.estimated_duration if hasattr(result, 'estimated_duration') else None,
            message="Workflow execution completed"
        )

    def _calculate_execution_metrics(self, execution: WorkflowExecution, step_executions: List[StepExecution]) -> Dict[str, Any]:
        """Calculate execution metrics"""
        total_steps = len(step_executions)
        completed_steps = sum(1 for se in step_executions if se.status == ExecutionStatus.COMPLETED)
        failed_steps = sum(1 for se in step_executions if se.status == ExecutionStatus.FAILED)

        total_tokens = sum(se.tokens_used or 0 for se in step_executions)
        total_cost = sum(se.cost_usd or 0 for se in step_executions)

        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": completed_steps / total_steps if total_steps > 0 else 0.0,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_step_time": sum(se.execution_time_seconds or 0 for se in step_executions) / total_steps if total_steps > 0 else 0.0
        }

    def _calculate_performance_score(self, success_rate: float, avg_execution_time: float, avg_cost: float) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        # Normalize metrics
        success_score = success_rate
        time_score = max(0, 1 - (avg_execution_time / 1800))  # Normalize by 30 minutes
        cost_score = max(0, 1 - (avg_cost / 10.0))  # Normalize by $10

        # Weighted average
        return (success_score * 0.5) + (time_score * 0.3) + (cost_score * 0.2)

    def _generate_workflow_recommendations(
        self,
        executions: List[WorkflowExecution],
        step_executions: List[StepExecution],
        success_rate: float,
        avg_execution_time: float
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.8:
            recommendations.append("Low success rate detected. Review error handling and input validation.")
        elif success_rate < 0.9:
            recommendations.append("Success rate could be improved. Consider adding retry logic.")

        # Performance recommendations
        if avg_execution_time > 300:  # 5 minutes
            recommendations.append("Execution time is high. Consider optimizing slow steps or running in parallel.")

        # Bottleneck recommendations
        bottlenecks = identify_workflow_bottlenecks(step_executions)
        if bottlenecks:
            slowest = bottlenecks[0]
            recommendations.append(f"Step '{slowest['step_id']}' is a bottleneck. Consider optimization.")

        # Cost recommendations
        total_cost = sum(se.cost_usd or 0 for se in step_executions)
        if total_cost > 5.0:  # $5
            recommendations.append("High execution cost detected. Consider using lower-tier models for simple steps.")

        return recommendations

    async def _notify_execution_control(self, execution_id: str, action: str):
        """Notify execution engine about control actions"""
        # This would integrate with the execution engine via message queue or direct call
        logger.info(f"Notifying execution engine to {action} execution {execution_id}")
        # Implementation would depend on execution engine architecture

# Global service instance
workflow_service: Optional[WorkflowService] = None

async def get_workflow_service() -> WorkflowService:
    """Get or create workflow service instance"""
    global workflow_service
    if workflow_service is None:
        workflow_service = WorkflowService()
        await workflow_service.__init__()  # Ensure initialization
    return workflow_service