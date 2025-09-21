"""
Agency Workflow Service - Business logic for agency workflow management.

This service provides the core business logic for managing agency workflows,
integrating with the Phase 1 Agency Swarm orchestration system and providing
comprehensive workflow lifecycle management.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_

from ...agents.orchestration.parallel_executor import ParallelExecutor, WorkflowStatus
from ...agents.orchestration.archon_agency import ArchonAgency
from ...database.models import (
    AgencyWorkflow,
    WorkflowExecution,
    AgentCapability,
    ConversationThread,
    CommunicationMessage
)
from ...server.config.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class WorkflowType(str, Enum):
    """Supported workflow types."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"
    CHAIN = "chain"


class WorkflowPriority(str, Enum):
    """Workflow priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CommunicationPattern(str, Enum):
    """Supported communication patterns."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    CHAIN = "chain"
    MULTICAST = "multicast"
    ASYNC = "async"


class CreateWorkflowRequest(BaseModel):
    """Request model for creating a new workflow."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    workflow_type: WorkflowType
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    agent_ids: List[str] = Field(..., min_items=1)
    workflow_config: Dict[str, Any] = Field(default_factory=dict)
    communication_pattern: CommunicationPattern = CommunicationPattern.DIRECT
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowExecutionRequest(BaseModel):
    """Request model for executing a workflow."""
    workflow_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    execution_config: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: Optional[int] = Field(default=300)
    async_execution: bool = Field(default=False)


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime]
    result: Dict[str, Any]
    error_message: Optional[str]
    execution_time_ms: Optional[int]


class AgencyWorkflowService:
    """Service for managing agency workflows with comprehensive business logic."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.parallel_executor = ParallelExecutor()
        self.archon_agency = ArchonAgency()
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._workflow_locks: Dict[str, asyncio.Lock] = {}

    async def create_workflow(self, request: CreateWorkflowRequest) -> AgencyWorkflow:
        """Create a new agency workflow with validation and configuration."""
        try:
            # Validate agent IDs exist and have required capabilities
            await self._validate_agent_capabilities(request.agent_ids, request.workflow_config)

            # Generate unique workflow ID
            workflow_id = str(uuid.uuid4())

            # Create workflow in database
            workflow = AgencyWorkflow(
                id=workflow_id,
                name=request.name,
                description=request.description,
                workflow_type=request.workflow_type.value,
                priority=request.priority.value,
                agent_ids=request.agent_ids,
                workflow_config=request.workflow_config,
                communication_pattern=request.communication_pattern.value,
                tags=request.tags,
                metadata=request.metadata,
                status=WorkflowStatus.CREATED,
                created_at=datetime.utcnow()
            )

            self.db.add(workflow)
            await self.db.commit()

            logger.info(f"Created workflow: {workflow_id} - {request.name}")
            return workflow

        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            raise

    async def execute_workflow(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
        """Execute a workflow with comprehensive monitoring and error handling."""
        try:
            # Get workflow from database
            workflow = await self._get_workflow(request.workflow_id)

            # Create execution record
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=request.workflow_id,
                status=WorkflowStatus.RUNNING,
                input_data=request.input_data,
                execution_config=request.execution_config,
                start_time=datetime.utcnow()
            )

            self.db.add(execution)
            await self.db.commit()

            # Execute workflow based on type
            if request.async_execution:
                # Start async execution
                task = asyncio.create_task(
                    self._execute_workflow_async(workflow, execution, request)
                )
                self._active_executions[execution_id] = task

                return WorkflowExecutionResponse(
                    execution_id=execution_id,
                    workflow_id=request.workflow_id,
                    status=WorkflowStatus.RUNNING,
                    start_time=execution.start_time,
                    result={"message": "Workflow execution started"},
                    execution_time_ms=None
                )
            else:
                # Execute synchronously
                result = await self._execute_workflow_sync(workflow, execution, request)
                return result

        except Exception as e:
            logger.error(f"Error executing workflow {request.workflow_id}: {e}")
            raise

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status with execution history."""
        try:
            workflow = await self._get_workflow(workflow_id)

            # Get latest execution
            stmt = select(WorkflowExecution).where(
                WorkflowExecution.workflow_id == workflow_id
            ).order_by(WorkflowExecution.created_at.desc()).limit(1)

            result = await self.db.execute(stmt)
            latest_execution = result.scalar_one_or_none()

            # Get execution history
            stmt = select(WorkflowExecution).where(
                WorkflowExecution.workflow_id == workflow_id
            ).order_by(WorkflowExecution.created_at.desc())

            result = await self.db.execute(stmt)
            execution_history = result.scalars().all()

            return {
                "workflow": workflow,
                "latest_execution": latest_execution,
                "execution_history": execution_history,
                "agent_status": await self._get_agent_status(workflow.agent_ids)
            }

        except Exception as e:
            logger.error(f"Error getting workflow status {workflow_id}: {e}")
            raise

    async def list_workflows(self,
                           workflow_type: Optional[WorkflowType] = None,
                           status: Optional[WorkflowStatus] = None,
                           tags: Optional[List[str]] = None,
                           limit: int = 50,
                           offset: int = 0) -> List[AgencyWorkflow]:
        """List workflows with filtering and pagination."""
        try:
            query = select(AgencyWorkflow)

            # Apply filters
            if workflow_type:
                query = query.where(AgencyWorkflow.workflow_type == workflow_type.value)

            if status:
                query = query.where(AgencyWorkflow.status == status.value)

            if tags:
                for tag in tags:
                    query = query.where(AgencyWorkflow.tags.contains([tag]))

            # Apply pagination
            query = query.order_by(AgencyWorkflow.created_at.desc())
            query = query.offset(offset).limit(limit)

            result = await self.db.execute(query)
            return result.scalars().all()

        except Exception as e:
            logger.error(f"Error listing workflows: {e}")
            raise

    async def list_agent_capabilities(self, agent_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List agent capabilities with filtering."""
        try:
            query = select(AgentCapability)

            if agent_ids:
                query = query.where(AgentCapability.agent_id.in_(agent_ids))

            result = await self.db.execute(query)
            capabilities = result.scalars().all()

            return [
                {
                    "agent_id": cap.agent_id,
                    "capability_type": cap.capability_type,
                    "capability_name": cap.capability_name,
                    "description": cap.description,
                    "parameters": cap.parameters,
                    "confidence_level": cap.confidence_level,
                    "last_used": cap.last_used
                }
                for cap in capabilities
            ]

        except Exception as e:
            logger.error(f"Error listing agent capabilities: {e}")
            raise

    async def create_conversation_thread(self,
                                      workflow_id: str,
                                      thread_name: str,
                                      participant_ids: List[str]) -> ConversationThread:
        """Create a conversation thread for a workflow."""
        try:
            thread_id = str(uuid.uuid4())
            thread = ConversationThread(
                id=thread_id,
                workflow_id=workflow_id,
                name=thread_name,
                participant_ids=participant_ids,
                created_at=datetime.utcnow()
            )

            self.db.add(thread)
            await self.db.commit()

            logger.info(f"Created conversation thread: {thread_id} for workflow: {workflow_id}")
            return thread

        except Exception as e:
            logger.error(f"Error creating conversation thread: {e}")
            raise

    async def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow analytics."""
        try:
            # Get workflow
            workflow = await self._get_workflow(workflow_id)

            # Get execution statistics
            stmt = select(WorkflowExecution).where(
                WorkflowExecution.workflow_id == workflow_id
            )

            result = await self.db.execute(stmt)
            executions = result.scalars().all()

            # Calculate analytics
            total_executions = len(executions)
            successful_executions = sum(1 for e in executions if e.status == WorkflowStatus.COMPLETED)
            failed_executions = sum(1 for e in executions if e.status == WorkflowStatus.FAILED)

            avg_execution_time = None
            if executions:
                execution_times = [
                    (e.end_time - e.start_time).total_seconds() * 1000
                    for e in executions if e.end_time
                ]
                if execution_times:
                    avg_execution_time = sum(execution_times) / len(execution_times)

            return {
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "average_execution_time_ms": avg_execution_time,
                "latest_execution": executions[0] if executions else None,
                "execution_history": [
                    {
                        "execution_id": e.id,
                        "status": e.status.value,
                        "start_time": e.start_time,
                        "end_time": e.end_time,
                        "execution_time_ms": (e.end_time - e.start_time).total_seconds() * 1000 if e.end_time else None
                    }
                    for e in executions
                ]
            }

        except Exception as e:
            logger.error(f"Error getting workflow analytics {workflow_id}: {e}")
            raise

    async def cancel_workflow_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        try:
            # Get execution
            stmt = select(WorkflowExecution).where(
                WorkflowExecution.id == execution_id
            )

            result = await self.db.execute(stmt)
            execution = result.scalar_one_or_none()

            if not execution:
                raise ValueError(f"Execution {execution_id} not found")

            if execution.status != WorkflowStatus.RUNNING:
                raise ValueError(f"Execution {execution_id} is not running")

            # Cancel async task if exists
            if execution_id in self._active_executions:
                task = self._active_executions[execution_id]
                if not task.done():
                    task.cancel()
                    del self._active_executions[execution_id]

            # Update execution status
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.utcnow()
            await self.db.commit()

            logger.info(f"Cancelled workflow execution: {execution_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling workflow execution {execution_id}: {e}")
            raise

    # Helper methods
    async def _validate_agent_capabilities(self, agent_ids: List[str], workflow_config: Dict[str, Any]) -> None:
        """Validate that agents have required capabilities."""
        # This would integrate with the agent registry to validate capabilities
        # For now, we'll assume validation passes
        pass

    async def _get_workflow(self, workflow_id: str) -> AgencyWorkflow:
        """Get workflow by ID."""
        stmt = select(AgencyWorkflow).where(AgencyWorkflow.id == workflow_id)
        result = await self.db.execute(stmt)
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        return workflow

    async def _execute_workflow_sync(self,
                                   workflow: AgencyWorkflow,
                                   execution: WorkflowExecution,
                                   request: WorkflowExecutionRequest) -> WorkflowExecutionResponse:
        """Execute workflow synchronously."""
        start_time = datetime.utcnow()

        try:
            # Execute based on workflow type
            if workflow.workflow_type == WorkflowType.PARALLEL:
                result = await self.parallel_executor.execute_parallel_workflow(
                    workflow.agent_ids,
                    request.input_data,
                    workflow.workflow_config
                )
            elif workflow.workflow_type == WorkflowType.SEQUENTIAL:
                result = await self.parallel_executor.execute_sequential_workflow(
                    workflow.agent_ids,
                    request.input_data,
                    workflow.workflow_config
                )
            else:
                # Default to agency execution
                result = await self.archon_agency.execute_workflow(
                    workflow.id,
                    request.input_data,
                    workflow.workflow_config
                )

            # Update execution record
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.utcnow()
            execution.result = result
            await self.db.commit()

            return WorkflowExecutionResponse(
                execution_id=execution.id,
                workflow_id=workflow.id,
                status=execution.status,
                start_time=start_time,
                end_time=execution.end_time,
                result=result,
                execution_time_ms=int((execution.end_time - start_time).total_seconds() * 1000)
            )

        except Exception as e:
            # Update execution record with error
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.error_message = str(e)
            await self.db.commit()

            return WorkflowExecutionResponse(
                execution_id=execution.id,
                workflow_id=workflow.id,
                status=execution.status,
                start_time=start_time,
                end_time=execution.end_time,
                result={"error": str(e)},
                error_message=str(e),
                execution_time_ms=int((execution.end_time - start_time).total_seconds() * 1000)
            )

    async def _execute_workflow_async(self,
                                     workflow: AgencyWorkflow,
                                     execution: WorkflowExecution,
                                     request: WorkflowExecutionRequest) -> None:
        """Execute workflow asynchronously."""
        try:
            result = await self._execute_workflow_sync(workflow, execution, request)
            # Task completed, clean up
            if execution.id in self._active_executions:
                del self._active_executions[execution.id]

        except Exception as e:
            logger.error(f"Async workflow execution failed: {e}")
            if execution.id in self._active_executions:
                del self._active_executions[execution.id]

    async def _get_agent_status(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Get current status of agents."""
        # This would integrate with the agent registry to get real-time status
        # For now, return placeholder data
        return {
            agent_id: {
                "status": "active",
                "last_seen": datetime.utcnow(),
                "current_workload": 0
            }
            for agent_id in agent_ids
        }