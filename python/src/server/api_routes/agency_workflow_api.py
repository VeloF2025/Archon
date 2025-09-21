"""
Agency Workflow API Routes for Archon Server

This module provides REST API endpoints for agency workflow management including:
- Workflow CRUD operations
- Workflow execution management
- Communication flow management
- Thread and conversation management
- Performance monitoring and analytics

Integration with Phase 1 Agency Swarm orchestration system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from fastapi import APIRouter, HTTPException, Query, Path, Body, BackgroundTasks
from pydantic import BaseModel, Field

# Add the project root to Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import agency workflow components
from src.mcp_server.modules.agency_workflow_tools import (
    AgencyWorkflowManager,
    WorkflowType,
    WorkflowStatus
)
from src.mcp_server.modules.agency_communication_tools import (
    CommunicationManager,
    CommunicationType,
    MessagePriority
)
from src.mcp_server.modules.agency_monitoring_tools import (
    AgencyMonitor,
    MonitoringLevel,
    AlertSeverity
)
from src.mcp_server.modules.agency_thread_tools import (
    ThreadManager,
    ThreadType,
    ThreadStatus
)
from src.server.config.logfire_config import api_logger

logger = logging.getLogger(__name__)

# Initialize managers
_workflow_manager = AgencyWorkflowManager()
_communication_manager = CommunicationManager()
_monitor = AgencyMonitor()
_thread_manager = ThreadManager()

# Create router
router = APIRouter(prefix="/api/agency", tags=["agency-workflows"])


# Pydantic models for request/response
class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of the workflow")
    workflow_type: str = Field(..., description="Type of workflow (sequential, parallel, hierarchical, collaborative, adaptive)")
    entry_point_agents: List[str] = Field(..., description="List of entry point agent names")
    communication_flows: List[Dict[str, Any]] = Field(..., description="Communication flow configurations")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional workflow configuration")


class WorkflowExecuteRequest(BaseModel):
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow execution")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional execution context")


class AgentMessageRequest(BaseModel):
    sender: str = Field(..., description="Name of the sending agent")
    recipient: Union[str, List[str]] = Field(..., description="Name or list of names of receiving agents")
    content: str = Field(..., description="Message content")
    message_type: str = Field("direct", description="Type of communication")
    priority: str = Field("medium", description="Message priority")
    thread_id: Optional[str] = Field(None, description="Optional thread ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    agency_id: Optional[str] = Field(None, description="Optional agency ID")


class ThreadCreateRequest(BaseModel):
    sender: str = Field(..., description="Name of the sending agent")
    recipient: str = Field(..., description="Name of the receiving agent")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial context")
    thread_name: Optional[str] = Field(None, description="Optional thread name")
    thread_type: str = Field("general", description="Type of thread")
    description: Optional[str] = Field(None, description="Thread description")
    expires_in_hours: Optional[int] = Field(None, description="Hours until thread expires")
    tags: Optional[List[str]] = Field(None, description="Thread tags")


class ThreadMessageRequest(BaseModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    sender: str = Field(..., description="Name of the message sender")
    content: str = Field(..., description="Message content")
    recipient: Optional[str] = Field(None, description="Optional recipient name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Message metadata")


class MonitoringConfigRequest(BaseModel):
    enabled: bool = Field(..., description="Whether monitoring should be enabled")
    collection_interval: Optional[int] = Field(None, description="Metrics collection interval in seconds")
    health_check_interval: Optional[int] = Field(None, description="Health check interval in seconds")
    alert_thresholds: Optional[Dict[str, float]] = Field(None, description="Alert threshold configuration")


# Workflow endpoints
@router.post("/workflows", response_model=Dict[str, Any])
async def create_workflow(request: WorkflowCreateRequest):
    """Create a new agency workflow."""
    try:
        workflow_type_enum = WorkflowType(request.workflow_type.lower())

        workflow_id = await _workflow_manager.create_workflow(
            name=request.name,
            description=request.description,
            workflow_type=workflow_type_enum,
            entry_point_agents=request.entry_point_agents,
            communication_flows=request.communication_flows,
            config=request.config or {}
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "name": request.name,
            "workflow_type": request.workflow_type,
            "created_at": datetime.utcnow().isoformat(),
            "message": f"Workflow '{request.name}' created successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/workflows", response_model=Dict[str, Any])
async def list_workflows(
    filter_type: Optional[str] = Query(None, description="Filter by workflow type"),
    filter_status: Optional[str] = Query(None, description="Filter by workflow status")
):
    """List all available agency workflows."""
    try:
        workflows = _workflow_manager.list_workflows()

        # Apply filters
        if filter_type:
            workflows = [wf for wf in workflows if wf["workflow_type"] == filter_type]
        if filter_status:
            workflows = [wf for wf in workflows if wf["status"] == filter_status]

        return {
            "success": True,
            "workflows": workflows,
            "total_count": len(workflows),
            "filters": {
                "type": filter_type,
                "status": filter_status
            }
        }

    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(workflow_id: str = Path(..., description="Workflow ID")):
    """Get details of a specific workflow."""
    try:
        workflows = _workflow_manager.list_workflows()
        workflow = next((wf for wf in workflows if wf["workflow_id"] == workflow_id), None)

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return {
            "success": True,
            "workflow": workflow
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/workflows/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: WorkflowExecuteRequest = Body(...)
):
    """Execute a predefined agency workflow."""
    try:
        execution_id = await _workflow_manager.execute_workflow(
            workflow_id=workflow_id,
            input_data=request.input_data,
            context=request.context
        )

        return {
            "success": True,
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "message": f"Workflow execution started: {execution_id}"
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/workflows/{workflow_id}/executions", response_model=Dict[str, Any])
async def get_workflow_executions(
    workflow_id: str = Path(..., description="Workflow ID"),
    status_filter: Optional[str] = Query(None, description="Filter by execution status")
):
    """Get execution history for a specific workflow."""
    try:
        executions = _workflow_manager.get_workflow_executions(workflow_id)

        # Apply status filter
        if status_filter:
            executions = [ex for ex in executions if ex["status"] == status_filter]

        return {
            "success": True,
            "executions": executions,
            "total_count": len(executions),
            "workflow_id": workflow_id
        }

    except Exception as e:
        logger.error(f"Error getting workflow executions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions/{execution_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(execution_id: str = Path(..., description="Execution ID")):
    """Get the status of a workflow execution."""
    try:
        status = _workflow_manager.get_workflow_status(execution_id)

        return {
            "success": True,
            "execution_status": status,
            "queried_at": datetime.utcnow().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Communication endpoints
@router.post("/communication/send-message", response_model=Dict[str, Any])
async def send_agent_message(request: AgentMessageRequest):
    """Send a message between agents."""
    try:
        # Parse message type and priority
        comm_type = CommunicationType(request.message_type.lower())
        msg_priority = MessagePriority(request.priority.lower())

        result = await _communication_manager.send_agent_message(
            sender=request.sender,
            recipient=request.recipient,
            content=request.content,
            message_type=comm_type,
            priority=msg_priority,
            thread_id=request.thread_id,
            context=request.context or {},
            agency_id=request.agency_id
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error sending agent message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/communication/broadcast", response_model=Dict[str, Any])
async def broadcast_message(request: AgentMessageRequest):
    """Broadcast a message to multiple agents."""
    try:
        if not isinstance(request.recipient, list):
            raise HTTPException(status_code=400, detail="Broadcast requires a list of recipients")

        msg_priority = MessagePriority(request.priority.lower())

        result = await _communication_manager.send_agent_message(
            sender=request.sender,
            recipient=request.recipient,
            content=request.content,
            message_type=CommunicationType.BROADCAST,
            priority=msg_priority,
            thread_id=request.thread_id,
            context=request.context or {},
            agency_id=request.agency_id
        )

        result["action"] = "broadcast"
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/communication/stats", response_model=Dict[str, Any])
async def get_communication_stats(
    time_range_hours: int = Query(24, description="Time range in hours for statistics")
):
    """Get communication statistics and analytics."""
    try:
        stats = await _communication_manager.get_communication_stats()

        # Add time-based filtering
        cutoff_time = datetime.utcnow().timestamp() - (time_range_hours * 3600)
        time_filtered_messages = [
            msg for msg in _communication_manager.message_history
            if msg.timestamp.timestamp() > cutoff_time
        ]

        stats.update({
            "time_range_hours": time_range_hours,
            "messages_in_range": len(time_filtered_messages),
            "messages_per_hour": len(time_filtered_messages) / max(time_range_hours, 1)
        })

        return {
            "success": True,
            "communication_statistics": stats,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting communication stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Thread management endpoints
@router.post("/threads", response_model=Dict[str, Any])
async def create_conversation_thread(request: ThreadCreateRequest):
    """Create a new conversation thread between agents."""
    try:
        thread_type_enum = ThreadType(request.thread_type.lower())

        thread_id = await _thread_manager.create_thread(
            sender=request.sender,
            recipient=request.recipient,
            initial_context=request.initial_context or {},
            thread_name=request.thread_name,
            thread_type=thread_type_enum,
            description=request.description,
            expires_in_hours=request.expires_in_hours,
            tags=request.tags or []
        )

        return {
            "success": True,
            "thread_id": thread_id,
            "sender": request.sender,
            "recipient": request.recipient,
            "thread_type": request.thread_type,
            "created_at": datetime.utcnow().isoformat(),
            "message": f"Conversation thread '{request.thread_name or thread_id[:8]}' created successfully"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating conversation thread: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/threads", response_model=Dict[str, Any])
async def list_threads(
    thread_type: Optional[str] = Query(None, description="Filter by thread type"),
    status: Optional[str] = Query(None, description="Filter by thread status"),
    participant: Optional[str] = Query(None, description="Filter by participant"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    limit: int = Query(100, description="Maximum number of threads to return"),
    include_expired: bool = Query(False, description="Whether to include expired threads")
):
    """List conversation threads with filtering."""
    try:
        # Parse filters
        thread_type_enum = None
        if thread_type:
            thread_type_enum = ThreadType(thread_type.lower())

        status_enum = None
        if status:
            status_enum = ThreadStatus(status.lower())

        threads = await _thread_manager.list_threads(
            thread_type=thread_type_enum,
            status=status_enum,
            participant=participant,
            tag=tag,
            limit=limit,
            include_expired=include_expired
        )

        return {
            "success": True,
            "threads": threads,
            "total_threads": len(threads),
            "filters": {
                "thread_type": thread_type,
                "status": status,
                "participant": participant,
                "tag": tag
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/threads/{thread_id}", response_model=Dict[str, Any])
async def get_thread(
    thread_id: str = Path(..., description="Thread ID"),
    limit: int = Query(50, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Number of messages to skip"),
    include_metadata: bool = Query(True, description="Whether to include message metadata")
):
    """Get a conversation thread and its messages."""
    try:
        messages = await _thread_manager.get_thread_messages(
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            include_metadata=include_metadata
        )

        metadata = await _thread_manager.get_thread_metadata(thread_id)

        return {
            "success": True,
            "thread_id": thread_id,
            "messages": messages,
            "total_messages": len(_thread_manager.thread_messages.get(thread_id, [])),
            "returned_messages": len(messages),
            "thread_info": {
                "name": metadata.name if metadata else None,
                "participants": metadata.participants if metadata else [],
                "thread_type": metadata.thread_type.value if metadata else None
            } if metadata else None,
            "retrieved_at": datetime.utcnow().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting thread: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/threads/{thread_id}/messages", response_model=Dict[str, Any])
async def add_thread_message(
    thread_id: str = Path(..., description="Thread ID"),
    request: ThreadMessageRequest = Body(...)
):
    """Add a message to a conversation thread."""
    try:
        message_id = await _thread_manager.add_message(
            thread_id=thread_id,
            role=request.role,
            sender=request.sender,
            content=request.content,
            recipient=request.recipient,
            metadata=request.metadata or {}
        )

        return {
            "success": True,
            "message_id": message_id,
            "thread_id": thread_id,
            "sender": request.sender,
            "role": request.role,
            "added_at": datetime.utcnow().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding thread message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/threads/{thread_id}", response_model=Dict[str, Any])
async def delete_thread(thread_id: str = Path(..., description="Thread ID")):
    """Delete a conversation thread."""
    try:
        metadata = await _thread_manager.get_thread_metadata(thread_id)
        thread_name = metadata.name if metadata else thread_id

        await _thread_manager.delete_thread(thread_id)

        return {
            "success": True,
            "thread_id": thread_id,
            "thread_name": thread_name,
            "deleted_at": datetime.utcnow().isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Monitoring endpoints
@router.get("/monitoring/summary", response_model=Dict[str, Any])
async def get_agency_summary(
    include_health: bool = Query(True, description="Whether to include health summary"),
    include_metrics: bool = Query(True, description="Whether to include metrics summary"),
    include_alerts: bool = Query(True, description="Whether to include alerts summary")
):
    """Get comprehensive agency monitoring summary."""
    try:
        summary = _monitor.get_agency_summary()

        # Customize summary based on parameters
        if not include_health:
            summary.pop("health_summary", None)
        if not include_metrics:
            summary.pop("latest_metrics", None)
        if not include_alerts:
            summary.pop("alert_summary", None)

        return {
            "success": True,
            "agency_summary": summary,
            "queried_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting agency summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    metric_name: Optional[str] = Query(None, description="Filter by specific metric name"),
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    time_range_hours: int = Query(1, description="Time range in hours for metrics"),
    limit: int = Query(100, description="Maximum number of metrics to return")
):
    """Get performance metrics for agencies and workflows."""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)

        metrics = _monitor.get_performance_metrics(
            metric_name=metric_name,
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        # Format metrics for response
        formatted_metrics = []
        for metric in metrics:
            formatted_metric = {
                "timestamp": metric.timestamp.isoformat(),
                "metric_name": metric.metric_name,
                "value": metric.value,
                "unit": metric.unit,
                "agent_name": metric.agent_name,
                "workflow_id": metric.workflow_id,
                "execution_id": metric.execution_id
            }
            if metric.metadata:
                formatted_metric["metadata"] = metric.metadata

            formatted_metrics.append(formatted_metric)

        # Calculate summary statistics
        if metrics:
            values = [m.value for m in metrics]
            summary = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "average": sum(values) / len(values),
                "latest": values[-1] if values else None
            }
        else:
            summary = {"count": 0}

        return {
            "success": True,
            "metrics": formatted_metrics,
            "summary": summary,
            "filters": {
                "metric_name": metric_name,
                "agent_name": agent_name,
                "time_range_hours": time_range_hours
            }
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/health", response_model=Dict[str, Any])
async def get_agent_health(
    agent_name: Optional[str] = Query(None, description="Filter by specific agent name"),
    include_history: bool = Query(False, description="Whether to include health status history")
):
    """Get health status information for agents."""
    try:
        # Ensure monitoring is running
        if not _monitor.monitoring_enabled:
            await _monitor.start_monitoring()

        health_status = _monitor.get_health_status(agent_name)

        # Format health status
        formatted_health = {}
        for name, health in health_status.items():
            health_info = {
                "agent_name": health.agent_name,
                "is_healthy": health.is_healthy,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "uptime_seconds": health.uptime_seconds
            }
            if health.error_message:
                health_info["error_message"] = health.error_message

            formatted_health[name] = health_info

        return {
            "success": True,
            "agent_health": formatted_health,
            "total_agents": len(formatted_health),
            "healthy_agents": sum(1 for h in formatted_health.values() if h["is_healthy"]),
            "queried_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting agent health: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/monitoring/alerts", response_model=Dict[str, Any])
async def get_monitoring_alerts(
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    source: Optional[str] = Query(None, description="Filter by alert source"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(50, description="Maximum number of alerts to return")
):
    """Get monitoring alerts and notifications."""
    try:
        # Parse severity filter
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity: {severity}. Valid values: {', '.join(s.value for s in AlertSeverity)}"
                )

        alerts = _monitor.get_alerts(
            severity=severity_enum,
            source=source,
            resolved=resolved,
            limit=limit
        )

        # Format alerts
        formatted_alerts = []
        for alert in alerts:
            alert_info = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "is_resolved": alert.resolved_at is not None
            }
            if alert.resolved_at:
                alert_info["resolved_at"] = alert.resolved_at.isoformat()
            if alert.metadata:
                alert_info["metadata"] = alert.metadata

            formatted_alerts.append(alert_info)

        return {
            "success": True,
            "alerts": formatted_alerts,
            "total_alerts": len(formatted_alerts),
            "filters": {
                "severity": severity,
                "source": source,
                "resolved": resolved
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting monitoring alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/monitoring/config", response_model=Dict[str, Any])
async def configure_monitoring(request: MonitoringConfigRequest):
    """Configure monitoring settings and thresholds."""
    try:
        # Update monitoring enabled state
        _monitor.monitoring_enabled = request.enabled

        # Update intervals if provided
        if request.collection_interval is not None:
            _monitor.collection_interval = max(10, request.collection_interval)

        if request.health_check_interval is not None:
            _monitor.health_check_interval = max(30, request.health_check_interval)

        # Update alert thresholds if provided
        if request.alert_thresholds:
            _monitor.alert_thresholds.update(request.alert_thresholds)

        # Start or stop monitoring based on enabled state
        if request.enabled:
            await _monitor.start_monitoring()
        else:
            await _monitor.stop_monitoring()

        return {
            "success": True,
            "monitoring_enabled": request.enabled,
            "collection_interval_seconds": _monitor.collection_interval,
            "health_check_interval_seconds": _monitor.health_check_interval,
            "alert_thresholds": _monitor.alert_thresholds,
            "configured_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error configuring monitoring: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Analytics endpoints
@router.get("/analytics/workflows", response_model=Dict[str, Any])
async def get_workflow_analytics(
    workflow_id: Optional[str] = Query(None, description="Filter by specific workflow ID"),
    time_range_hours: int = Query(24, description="Time range in hours for analytics"),
    include_metrics: bool = Query(True, description="Whether to include detailed performance metrics")
):
    """Get workflow performance metrics and analytics."""
    try:
        from src.mcp_server.modules.agency_workflow_tools import _workflow_manager

        # Get executions within time range
        cutoff_time = datetime.utcnow().timestamp() - (time_range_hours * 3600)

        executions = _workflow_manager.get_workflow_executions(workflow_id)
        recent_executions = [
            ex for ex in executions
            if ex["started_at"] and datetime.fromisoformat(ex["started_at"]).timestamp() > cutoff_time
        ]

        # Calculate analytics
        total_executions = len(recent_executions)
        successful_executions = len([ex for ex in recent_executions if ex["status"] == "completed"])
        failed_executions = len([ex for ex in recent_executions if ex["status"] == "failed"])

        # Calculate average execution time
        execution_times = []
        for ex in recent_executions:
            if ex["started_at"] and ex["completed_at"]:
                start = datetime.fromisoformat(ex["started_at"])
                end = datetime.fromisoformat(ex["completed_at"])
                execution_times.append((end - start).total_seconds())

        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        # Calculate success rate
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

        analytics = {
            "time_range_hours": time_range_hours,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate_percent": round(success_rate, 2),
            "average_execution_time_seconds": round(avg_execution_time, 2),
            "workflow_filter": workflow_id
        }

        if include_metrics:
            # Add detailed metrics
            status_distribution = {}
            for ex in recent_executions:
                status = ex["status"]
                status_distribution[status] = status_distribution.get(status, 0) + 1

            analytics.update({
                "status_distribution": status_distribution,
                "execution_time_details": {
                    "min_time": min(execution_times) if execution_times else 0,
                    "max_time": max(execution_times) if execution_times else 0,
                    "median_time": sorted(execution_times)[len(execution_times)//2] if execution_times else 0
                }
            })

        return {
            "success": True,
            "analytics": analytics,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting workflow analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/threads", response_model=Dict[str, Any])
async def get_thread_analytics(
    thread_id: Optional[str] = Query(None, description="Filter by specific thread ID"),
    time_range_hours: int = Query(24, description="Time range in hours for analytics")
):
    """Get conversation thread analytics and statistics."""
    try:
        analytics = await _thread_manager.get_thread_analytics(
            thread_id=thread_id,
            time_range_hours=time_range_hours
        )

        # If specific thread requested, add detailed info
        if thread_id:
            metadata = await _thread_manager.get_thread_metadata(thread_id)
            messages = await _thread_manager.get_thread_messages(
                thread_id=thread_id,
                limit=1000,
                include_metadata=False
            )

            if metadata:
                analytics["thread_details"] = {
                    "thread_id": thread_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "thread_type": metadata.thread_type.value,
                    "status": metadata.status.value,
                    "participants": metadata.participants,
                    "message_count": len(messages),
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat(),
                    "tags": metadata.tags
                }

                # Analyze message patterns
                if messages:
                    from collections import defaultdict
                    senders = defaultdict(int)
                    roles = defaultdict(int)

                    for msg in messages:
                        senders[msg["sender"]] += 1
                        roles[msg["role"]] += 1

                    analytics["message_patterns"] = {
                        "senders": dict(senders),
                        "roles": dict(roles),
                        "average_messages_per_hour": len(messages) / max(time_range_hours, 1)
                    }

        return {
            "success": True,
            "analytics": analytics,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting thread analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Agent capabilities endpoint
@router.get("/agents/capabilities", response_model=Dict[str, Any])
async def list_agent_capabilities(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    include_metadata: bool = Query(True, description="Whether to include detailed metadata")
):
    """Query agent capabilities and available agents."""
    try:
        # Get available agent types from the parallel executor
        try:
            from src.agents.orchestration.parallel_executor import ParallelExecutor

            # Create temporary executor to get agent configs
            executor = ParallelExecutor(max_concurrent=1)
            agent_configs = executor.agent_configs
            executor.shutdown()
        except Exception as e:
            logger.warning(f"Could not load agent configs: {e}")
            agent_configs = {}

        # Format agent capabilities
        capabilities = []
        for role, config in agent_configs.items():
            if agent_type and agent_type not in role.lower():
                continue

            capability_info = {
                "role": role,
                "name": getattr(config, 'name', role),
                "description": getattr(config, 'description', ''),
                "skills": getattr(config, 'skills', []),
                "priority": getattr(config, 'priority', 'medium')
            }

            if include_metadata:
                capability_info.update({
                    "memory_scope": getattr(config, 'memory_scope', []),
                    "proactive_triggers": getattr(config, 'proactive_triggers', []),
                    "execution_context": getattr(config, 'execution_context', {}),
                    "dependencies": getattr(config, 'dependencies', []),
                    "output_patterns": getattr(config, 'output_patterns', [])
                })

            capabilities.append(capability_info)

        # Add system-level capabilities
        from src.mcp_server.modules.agency_workflow_tools import WorkflowType, CommunicationFlowType
        system_capabilities = {
            "parallel_execution": True,
            "workflow_types": [t.value for t in WorkflowType],
            "communication_flows": [t.value for t in CommunicationFlowType],
            "thread_management": True,
            "confidence_scoring": True,
            "conflict_resolution": True
        }

        return {
            "success": True,
            "agent_capabilities": capabilities,
            "system_capabilities": system_capabilities,
            "total_agents": len(capabilities),
            "queried_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error listing agent capabilities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")