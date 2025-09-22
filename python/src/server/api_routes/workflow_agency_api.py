"""
Agency Workflow API Routes
RESTful API for ReactFlow-compatible workflow management with Agency Swarm integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import asyncio
import uuid

from ...database.workflow_models import (
    WorkflowDefinition, WorkflowExecution, StepExecution, WorkflowAnalytics,
    WorkflowCreateRequest, WorkflowUpdateRequest, WorkflowExecutionRequest,
    ReactFlowNode, ReactFlowEdge, ReactFlowData, ExecutionStatus
)
from ..services.workflow_service import WorkflowService
from ..services.workflow_execution_service import WorkflowExecutionService
from ..services.workflow_analytics_service import WorkflowAnalyticsService

router = APIRouter(prefix="/api/workflow-agency", tags=["workflow-agency"])

# Initialize services
workflow_service = WorkflowService()
execution_service = WorkflowExecutionService()
analytics_service = WorkflowAnalyticsService()


# Pydantic models for API requests/responses
class WorkflowCreateRequest(BaseModel):
    """Workflow creation request with ReactFlow compatibility"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_template: bool = False


class WorkflowUpdateRequest(BaseModel):
    """Workflow update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    variables: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None


class WorkflowExecuteRequest(BaseModel):
    """Workflow execution request"""
    workflow_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = True
    timeout: Optional[int] = Field(None, ge=1, le=3600)


class StepExecutionRequest(BaseModel):
    """Step execution request"""
    step_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class WorkflowValidationRequest(BaseModel):
    """Workflow validation request"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class WorkflowTemplateRequest(BaseModel):
    """Workflow template creation request"""
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    category: str = Field(..., min_length=1, max_length=100)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


# Workflow Management Endpoints
@router.post("/workflows")
async def create_workflow(request: WorkflowCreateRequest) -> Dict[str, Any]:
    """Create a new workflow with ReactFlow compatibility"""
    try:
        # Convert ReactFlow data to internal format
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in request.nodes],
            edges=[ReactFlowEdge(**edge) for edge in request.edges]
        )

        workflow = await workflow_service.create_workflow(
            name=request.name,
            description=request.description,
            react_flow_data=react_flow_data,
            variables=request.variables,
            tags=request.tags,
            is_template=request.is_template
        )

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "version": workflow.version,
            "created_at": workflow.created_at.isoformat(),
            "nodes": [node.model_dump() for node in workflow.nodes],
            "edges": [edge.model_dump() for edge in workflow.edges],
            "variables": workflow.variables,
            "tags": workflow.tags
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows")
async def list_workflows(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    search: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """List workflows with filtering and pagination"""
    try:
        workflows = await workflow_service.list_workflows(
            skip=skip,
            limit=limit,
            status=status,
            tags=tags,
            search=search
        )

        return {
            "total": workflows["total"],
            "workflows": [
                {
                    "workflow_id": w.id,
                    "name": w.name,
                    "description": w.description,
                    "status": w.status,
                    "version": w.version,
                    "tags": w.tags,
                    "created_at": w.created_at.isoformat(),
                    "updated_at": w.updated_at.isoformat(),
                    "execution_count": w.execution_count,
                    "last_executed": w.last_executed.isoformat() if w.last_executed else None
                }
                for w in workflows["workflows"]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get workflow details with ReactFlow data"""
    try:
        workflow = await workflow_service.get_workflow(workflow_id)

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "version": workflow.version,
            "tags": workflow.tags,
            "variables": workflow.variables,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "execution_count": workflow.execution_count,
            "last_executed": workflow.last_executed.isoformat() if workflow.last_executed else None,
            "react_flow_data": {
                "nodes": [node.model_dump() for node in workflow.nodes],
                "edges": [edge.model_dump() for edge in workflow.edges]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/workflows/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    request: WorkflowUpdateRequest
) -> Dict[str, Any]:
    """Update workflow"""
    try:
        # Build update data
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.variables is not None:
            update_data["variables"] = request.variables
        if request.tags is not None:
            update_data["tags"] = request.tags
        if request.status is not None:
            update_data["status"] = request.status

        # Handle ReactFlow data updates
        if request.nodes is not None or request.edges is not None:
            react_flow_data = ReactFlowData(
                nodes=[ReactFlowNode(**node) for node in (request.nodes or [])],
                edges=[ReactFlowEdge(**edge) for edge in (request.edges or [])]
            )
            update_data["react_flow_data"] = react_flow_data

        workflow = await workflow_service.update_workflow(workflow_id, update_data)

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "version": workflow.version,
            "updated_at": workflow.updated_at.isoformat(),
            "message": "Workflow updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """Delete workflow"""
    try:
        success = await workflow_service.delete_workflow(workflow_id)

        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {"message": "Workflow deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflow Execution Endpoints
@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Execute a workflow"""
    try:
        workflow = await workflow_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        if request.async_execution:
            # Execute asynchronously
            execution_id = await execution_service.execute_workflow(
                workflow_id=workflow_id,
                inputs=request.inputs,
                timeout=request.timeout
            )

            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "started",
                "message": "Workflow execution started asynchronously"
            }
        else:
            # Execute synchronously (wait for completion or timeout)
            execution_id = await execution_service.execute_workflow(
                workflow_id=workflow_id,
                inputs=request.inputs,
                timeout=request.timeout
            )

            # Wait for completion (with timeout)
            max_wait = request.timeout or 300  # Default 5 minutes
            start_time = datetime.now()

            while (datetime.now() - start_time).seconds < max_wait:
                execution = await execution_service.get_execution_status(execution_id)
                if execution and execution.status in [
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED
                ]:
                    return {
                        "execution_id": execution_id,
                        "workflow_id": workflow_id,
                        "status": execution.status.value,
                        "results": execution.results,
                        "errors": execution.errors,
                        "metrics": execution.metrics,
                        "duration": (execution.completed_at - execution.started_at).total_seconds() if execution.completed_at else None
                    }
                await asyncio.sleep(1)

            return {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": "timeout",
                "message": "Execution timeout - check status separately"
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str) -> Dict[str, Any]:
    """Get workflow execution status"""
    try:
        execution = await execution_service.get_execution_status(execution_id)

        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")

        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "current_step": execution.current_step_id,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "results": execution.results,
            "errors": execution.errors,
            "metrics": execution.metrics,
            "step_executions": [
                {
                    "step_id": step.id,
                    "step_name": step.step_name,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat(),
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "result": step.result,
                    "error": step.error,
                    "duration": (step.completed_at - step.started_at).total_seconds() if step.completed_at else None
                }
                for step in execution.step_executions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workflows/{workflow_id}/executions")
async def get_workflow_executions(
    workflow_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
) -> Dict[str, Any]:
    """Get execution history for a workflow"""
    try:
        executions = await execution_service.get_execution_history(
            workflow_id=workflow_id,
            skip=skip,
            limit=limit
        )

        return {
            "total": executions["total"],
            "executions": [
                {
                    "execution_id": exec.id,
                    "status": exec.status.value,
                    "started_at": exec.started_at.isoformat(),
                    "completed_at": exec.completed_at.isoformat() if exec.completed_at else None,
                    "duration": (exec.completed_at - exec.started_at).total_seconds() if exec.completed_at else None,
                    "has_errors": len(exec.errors) > 0
                }
                for exec in executions["executions"]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/executions/{execution_id}/pause")
async def pause_execution(execution_id: str) -> Dict[str, Any]:
    """Pause workflow execution"""
    try:
        success = await execution_service.pause_execution(execution_id)

        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be paused")

        return {"message": "Execution paused successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str) -> Dict[str, Any]:
    """Resume workflow execution"""
    try:
        success = await execution_service.resume_execution(execution_id)

        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be resumed")

        return {"message": "Execution resumed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """Cancel workflow execution"""
    try:
        success = await execution_service.cancel_execution(execution_id)

        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")

        return {"message": "Execution cancelled successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflow Validation Endpoints
@router.post("/workflows/validate")
async def validate_workflow(request: WorkflowValidationRequest) -> Dict[str, Any]:
    """Validate workflow structure and ReactFlow data"""
    try:
        react_flow_data = ReactFlowData(
            nodes=[ReactFlowNode(**node) for node in request.nodes],
            edges=[ReactFlowEdge(**edge) for edge in request.edges]
        )

        is_valid, errors = await workflow_service.validate_workflow(react_flow_data)

        return {
            "valid": is_valid,
            "errors": errors,
            "node_count": len(react_flow_data.nodes),
            "edge_count": len(react_flow_data.edges)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/workflows/{workflow_id}/validate")
async def validate_existing_workflow(workflow_id: str) -> Dict[str, Any]:
    """Validate existing workflow"""
    try:
        workflow = await workflow_service.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        react_flow_data = ReactFlowData(
            nodes=workflow.nodes,
            edges=workflow.edges
        )

        is_valid, errors = await workflow_service.validate_workflow(react_flow_data)

        return {
            "valid": is_valid,
            "errors": errors,
            "node_count": len(react_flow_data.nodes),
            "edge_count": len(react_flow_data.edges)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflow Analytics Endpoints
@router.get("/workflows/{workflow_id}/analytics")
async def get_workflow_analytics(
    workflow_id: str,
    days: int = Query(30, ge=1, le=365)
) -> Dict[str, Any]:
    """Get workflow analytics and insights"""
    try:
        analytics = await analytics_service.get_workflow_analytics(
            workflow_id=workflow_id,
            days=days
        )

        if not analytics:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics/dashboard")
async def get_dashboard_analytics(
    days: int = Query(7, ge=1, le=90)
) -> Dict[str, Any]:
    """Get dashboard analytics for all workflows"""
    try:
        analytics = await analytics_service.get_dashboard_analytics(days=days)
        return analytics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics/performance")
async def get_performance_analytics(
    workflow_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
) -> Dict[str, Any]:
    """Get performance analytics and bottlenecks"""
    try:
        if workflow_id:
            analytics = await analytics_service.get_performance_analytics(
                workflow_id=workflow_id,
                days=days
            )
        else:
            analytics = await analytics_service.get_system_performance_analytics(days=days)

        return analytics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analytics/costs")
async def get_cost_analytics(
    workflow_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365)
) -> Dict[str, Any]:
    """Get cost analytics and optimization recommendations"""
    try:
        if workflow_id:
            analytics = await analytics_service.get_cost_analytics(
                workflow_id=workflow_id,
                days=days
            )
        else:
            analytics = await analytics_service.get_system_cost_analytics(days=days)

        return analytics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflow Templates Endpoints
@router.get("/templates")
async def list_workflow_templates(
    category: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200)
) -> Dict[str, Any]:
    """List workflow templates"""
    try:
        templates = await workflow_service.list_templates(
            category=category,
            skip=skip,
            limit=limit
        )

        return {
            "total": templates["total"],
            "templates": [
                {
                    "template_id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "tags": t.tags,
                    "usage_count": t.usage_count,
                    "created_at": t.created_at.isoformat()
                }
                for t in templates["templates"]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates/{template_id}")
async def get_workflow_template(template_id: str) -> Dict[str, Any]:
    """Get workflow template details"""
    try:
        template = await workflow_service.get_template(template_id)

        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return {
            "template_id": template.id,
            "name": template.name,
            "description": template.description,
            "category": template.category,
            "parameters": template.parameters,
            "react_flow_data": {
                "nodes": [node.model_dump() for node in template.nodes],
                "edges": [edge.model_dump() for edge in template.edges]
            },
            "tags": template.tags,
            "usage_count": template.usage_count,
            "created_at": template.created_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/templates/{template_id}/instantiate")
async def instantiate_workflow_from_template(
    template_id: str,
    name: str,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create workflow from template"""
    try:
        workflow = await workflow_service.create_workflow_from_template(
            template_id=template_id,
            name=name,
            description=description,
            parameters=parameters or {}
        )

        if not workflow:
            raise HTTPException(status_code=404, detail="Template not found")

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "message": "Workflow created from template successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Real-time Metrics Endpoints
@router.get("/metrics/real-time")
async def get_real_time_metrics() -> Dict[str, Any]:
    """Get real-time workflow metrics"""
    try:
        metrics = await analytics_service.get_real_time_metrics()
        return metrics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system-wide workflow metrics"""
    try:
        metrics = await analytics_service.get_system_metrics()
        return metrics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Health Check Endpoint
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for workflow agency services"""
    try:
        # Check database connectivity
        workflow_count = await workflow_service.get_workflow_count()

        return {
            "status": "healthy",
            "services": {
                "workflow_service": "active",
                "execution_service": "active",
                "analytics_service": "active"
            },
            "database": {
                "connected": True,
                "workflow_count": workflow_count
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


