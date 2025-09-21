"""
Socket.IO Event Handlers for Agency Workflows

This module contains all Socket.IO event handlers for real-time workflow communication.
Provides live updates for workflow execution, status changes, and analytics.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional

from ..config.logfire_config import get_logger
from ..socketio_app import get_socketio_instance
from ..services.workflow_execution_service import WorkflowExecutionService, ExecutionStatus
from ..services.workflow_analytics_service import WorkflowAnalyticsService

logger = get_logger(__name__)

# Get Socket.IO instance
sio = get_socketio_instance()

# Initialize services
execution_service = WorkflowExecutionService()
analytics_service = WorkflowAnalyticsService()

# Rate limiting for Socket.IO broadcasts
_last_broadcast_times: Dict[str, float] = {}
_min_broadcast_interval = 0.1  # Minimum 100ms between broadcasts per room

# Room management for workflow events
_workflow_rooms: Dict[str, set] = {}  # workflow_id -> set of session_ids
_execution_rooms: Dict[str, set] = {}  # execution_id -> set of session_ids


# Helper functions for room management
async def join_workflow_room(workflow_id: str, sid: str):
    """Join a workflow room for real-time updates."""
    room = f"workflow_{workflow_id}"
    await sio.enter_room(sid, room)

    if workflow_id not in _workflow_rooms:
        _workflow_rooms[workflow_id] = set()
    _workflow_rooms[workflow_id].add(sid)

    logger.info(f"🔗 [WORKFLOW SOCKET] Session {sid} joined workflow room {room}")


async def leave_workflow_room(workflow_id: str, sid: str):
    """Leave a workflow room."""
    room = f"workflow_{workflow_id}"
    await sio.leave_room(sid, room)

    if workflow_id in _workflow_rooms:
        _workflow_rooms[workflow_id].discard(sid)
        if not _workflow_rooms[workflow_id]:
            del _workflow_rooms[workflow_id]

    logger.info(f"🔗 [WORKFLOW SOCKET] Session {sid} left workflow room {room}")


async def join_execution_room(execution_id: str, sid: str):
    """Join an execution room for real-time updates."""
    room = f"execution_{execution_id}"
    await sio.enter_room(sid, room)

    if execution_id not in _execution_rooms:
        _execution_rooms[execution_id] = set()
    _execution_rooms[execution_id].add(sid)

    logger.info(f"🔗 [WORKFLOW SOCKET] Session {sid} joined execution room {room}")


async def leave_execution_room(execution_id: str, sid: str):
    """Leave an execution room."""
    room = f"execution_{execution_id}"
    await sio.leave_room(sid, room)

    if execution_id in _execution_rooms:
        _execution_rooms[execution_id].discard(sid)
        if not _execution_rooms[execution_id]:
            del _execution_rooms[execution_id]

    logger.info(f"🔗 [WORKFLOW SOCKET] Session {sid} left execution room {room}")


# Rate limiting helper
async def can_broadcast(room: str) -> bool:
    """Check if we can broadcast to a room (rate limiting)."""
    current_time = time.time()
    last_broadcast = _last_broadcast_times.get(room, 0)

    if current_time - last_broadcast >= _min_broadcast_interval:
        _last_broadcast_times[room] = current_time
        return True
    return False


# Workflow Event Broadcast Functions
async def broadcast_workflow_created(workflow_data: Dict[str, Any]):
    """Broadcast workflow creation event."""
    room = "workflows_all"
    if await can_broadcast(room):
        await sio.emit("workflow_created", workflow_data, room=room)
        logger.info(f"📝 [WORKFLOW SOCKET] Broadcasted workflow_created: {workflow_data.get('name', 'Unknown')}")


async def broadcast_workflow_updated(workflow_data: Dict[str, Any]):
    """Broadcast workflow update event."""
    room = f"workflow_{workflow_data['id']}"
    if await can_broadcast(room):
        workflow_data["server_timestamp"] = time.time() * 1000
        await sio.emit("workflow_updated", workflow_data, room=room)
        logger.info(f"📝 [WORKFLOW SOCKET] Broadcasted workflow_updated: {workflow_data.get('name', 'Unknown')}")


async def broadcast_workflow_deleted(workflow_id: str):
    """Broadcast workflow deletion event."""
    room = "workflows_all"
    if await can_broadcast(room):
        await sio.emit("workflow_deleted", {"workflow_id": workflow_id}, room=room)
        logger.info(f"🗑️ [WORKFLOW SOCKET] Broadcasted workflow_deleted: {workflow_id}")


async def broadcast_workflow_status_changed(workflow_id: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Broadcast workflow status change event."""
    room = f"workflow_{workflow_id}"
    if await can_broadcast(room):
        data = {
            "workflow_id": workflow_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        await sio.emit("workflow_status_changed", data, room=room)
        logger.info(f"📊 [WORKFLOW SOCKET] Broadcasted workflow_status_changed: {workflow_id} -> {status}")


# Execution Event Broadcast Functions
async def broadcast_execution_started(execution_data: Dict[str, Any]):
    """Broadcast execution started event."""
    room = f"workflow_{execution_data['workflow_id']}"
    execution_room = f"execution_{execution_data['id']}"

    if await can_broadcast(room):
        await sio.emit("execution_started", execution_data, room=room)
        logger.info(f"🚀 [WORKFLOW SOCKET] Broadcasted execution_started: {execution_data['id']}")

    # Also broadcast to execution-specific room
    if await can_broadcast(execution_room):
        await sio.emit("execution_started", execution_data, room=execution_room)


async def broadcast_execution_progress(execution_id: str, progress_data: Dict[str, Any]):
    """Broadcast execution progress update."""
    room = f"execution_{execution_id}"
    if await can_broadcast(room):
        progress_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("execution_progress", progress_data, room=room)
        logger.info(f"📊 [WORKFLOW SOCKET] Broadcasted execution_progress: {execution_id} - {progress_data.get('progress_percent', 0)}%")


async def broadcast_execution_step_started(execution_id: str, step_data: Dict[str, Any]):
    """Broadcast step execution started event."""
    room = f"execution_{execution_id}"
    if await can_broadcast(room):
        step_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("execution_step_started", step_data, room=room)
        logger.info(f"🔧 [WORKFLOW SOCKET] Broadcasted execution_step_started: {step_data.get('step_name', 'Unknown')}")


async def broadcast_execution_step_completed(execution_id: str, step_data: Dict[str, Any]):
    """Broadcast step execution completed event."""
    room = f"execution_{execution_id}"
    if await can_broadcast(room):
        step_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("execution_step_completed", step_data, room=room)
        logger.info(f"✅ [WORKFLOW SOCKET] Broadcasted execution_step_completed: {step_data.get('step_name', 'Unknown')}")


async def broadcast_execution_step_failed(execution_id: str, step_data: Dict[str, Any]):
    """Broadcast step execution failed event."""
    room = f"execution_{execution_id}"
    if await can_broadcast(room):
        step_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("execution_step_failed", step_data, room=room)
        logger.warning(f"❌ [WORKFLOW SOCKET] Broadcasted execution_step_failed: {step_data.get('step_name', 'Unknown')}")


async def broadcast_execution_completed(execution_data: Dict[str, Any]):
    """Broadcast execution completed event."""
    room = f"workflow_{execution_data['workflow_id']}"
    execution_room = f"execution_{execution_data['id']}"

    if await can_broadcast(room):
        await sio.emit("execution_completed", execution_data, room=room)
        logger.info(f"🎉 [WORKFLOW SOCKET] Broadcasted execution_completed: {execution_data['id']}")

    # Also broadcast to execution-specific room
    if await can_broadcast(execution_room):
        await sio.emit("execution_completed", execution_data, room=execution_room)


async def broadcast_execution_failed(execution_data: Dict[str, Any]):
    """Broadcast execution failed event."""
    room = f"workflow_{execution_data['workflow_id']}"
    execution_room = f"execution_{execution_data['id']}"

    if await can_broadcast(room):
        await sio.emit("execution_failed", execution_data, room=room)
        logger.warning(f"❌ [WORKFLOW SOCKET] Broadcasted execution_failed: {execution_data['id']}")

    # Also broadcast to execution-specific room
    if await can_broadcast(execution_room):
        await sio.emit("execution_failed", execution_data, room=execution_room)


async def broadcast_execution_paused(execution_id: str, workflow_id: str):
    """Broadcast execution paused event."""
    room = f"workflow_{workflow_id}"
    execution_room = f"execution_{execution_id}"

    data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "timestamp": datetime.now().isoformat()
    }

    if await can_broadcast(room):
        await sio.emit("execution_paused", data, room=room)
        logger.info(f"⏸️ [WORKFLOW SOCKET] Broadcasted execution_paused: {execution_id}")

    if await can_broadcast(execution_room):
        await sio.emit("execution_paused", data, room=execution_room)


async def broadcast_execution_resumed(execution_id: str, workflow_id: str):
    """Broadcast execution resumed event."""
    room = f"workflow_{workflow_id}"
    execution_room = f"execution_{execution_id}"

    data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "timestamp": datetime.now().isoformat()
    }

    if await can_broadcast(room):
        await sio.emit("execution_resumed", data, room=room)
        logger.info(f"▶️ [WORKFLOW SOCKET] Broadcasted execution_resumed: {execution_id}")

    if await can_broadcast(execution_room):
        await sio.emit("execution_resumed", data, room=execution_room)


async def broadcast_execution_cancelled(execution_id: str, workflow_id: str):
    """Broadcast execution cancelled event."""
    room = f"workflow_{workflow_id}"
    execution_room = f"execution_{execution_id}"

    data = {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "timestamp": datetime.now().isoformat()
    }

    if await can_broadcast(room):
        await sio.emit("execution_cancelled", data, room=room)
        logger.info(f"🚫 [WORKFLOW SOCKET] Broadcasted execution_cancelled: {execution_id}")

    if await can_broadcast(execution_room):
        await sio.emit("execution_cancelled", data, room=execution_room)


# Analytics Event Broadcast Functions
async def broadcast_analytics_update(workflow_id: str, analytics_data: Dict[str, Any]):
    """Broadcast analytics update event."""
    room = f"workflow_{workflow_id}"
    if await can_broadcast(room):
        analytics_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("analytics_update", analytics_data, room=room)
        logger.info(f"📊 [WORKFLOW SOCKET] Broadcasted analytics_update: {workflow_id}")


async def broadcast_real_time_metrics(metrics_data: Dict[str, Any]):
    """Broadcast real-time metrics to all connected clients."""
    room = "workflows_metrics"
    if await can_broadcast(room):
        metrics_data["timestamp"] = datetime.now().isoformat()
        await sio.emit("real_time_metrics", metrics_data, room=room)
        logger.info(f"📈 [WORKFLOW SOCKET] Broadcasted real_time_metrics")


# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"🔗 [WORKFLOW SOCKET] Client connected: {sid}")

    # Send initial connection acknowledgment
    await sio.emit("connected", {
        "message": "Connected to workflow real-time updates",
        "timestamp": datetime.now().isoformat()
    }, to=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"🔗 [WORKFLOW SOCKET] Client disconnected: {sid}")

    # Clean up room memberships
    rooms_to_clean = []

    for workflow_id, sessions in _workflow_rooms.items():
        if sid in sessions:
            rooms_to_clean.append(("workflow", workflow_id))

    for execution_id, sessions in _execution_rooms.items():
        if sid in sessions:
            rooms_to_clean.append(("execution", execution_id))

    for room_type, room_id in rooms_to_clean:
        if room_type == "workflow":
            await leave_workflow_room(room_id, sid)
        elif room_type == "execution":
            await leave_execution_room(room_id, sid)


@sio.event
async def subscribe_workflow(sid, data):
    """Subscribe to workflow updates."""
    try:
        workflow_id = data.get("workflow_id")
        if not workflow_id:
            await sio.emit("error", {"message": "workflow_id required"}, to=sid)
            return

        await join_workflow_room(workflow_id, sid)

        await sio.emit("subscribed", {
            "type": "workflow",
            "workflow_id": workflow_id,
            "message": "Subscribed to workflow updates"
        }, to=sid)

        logger.info(f"📝 [WORKFLOW SOCKET] Client {sid} subscribed to workflow {workflow_id}")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in subscribe_workflow: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def unsubscribe_workflow(sid, data):
    """Unsubscribe from workflow updates."""
    try:
        workflow_id = data.get("workflow_id")
        if not workflow_id:
            await sio.emit("error", {"message": "workflow_id required"}, to=sid)
            return

        await leave_workflow_room(workflow_id, sid)

        await sio.emit("unsubscribed", {
            "type": "workflow",
            "workflow_id": workflow_id,
            "message": "Unsubscribed from workflow updates"
        }, to=sid)

        logger.info(f"📝 [WORKFLOW SOCKET] Client {sid} unsubscribed from workflow {workflow_id}")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in unsubscribe_workflow: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def subscribe_execution(sid, data):
    """Subscribe to execution updates."""
    try:
        execution_id = data.get("execution_id")
        if not execution_id:
            await sio.emit("error", {"message": "execution_id required"}, to=sid)
            return

        await join_execution_room(execution_id, sid)

        await sio.emit("subscribed", {
            "type": "execution",
            "execution_id": execution_id,
            "message": "Subscribed to execution updates"
        }, to=sid)

        logger.info(f"🚀 [WORKFLOW SOCKET] Client {sid} subscribed to execution {execution_id}")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in subscribe_execution: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def unsubscribe_execution(sid, data):
    """Unsubscribe from execution updates."""
    try:
        execution_id = data.get("execution_id")
        if not execution_id:
            await sio.emit("error", {"message": "execution_id required"}, to=sid)
            return

        await leave_execution_room(execution_id, sid)

        await sio.emit("unsubscribed", {
            "type": "execution",
            "execution_id": execution_id,
            "message": "Unsubscribed from execution updates"
        }, to=sid)

        logger.info(f"🚀 [WORKFLOW SOCKET] Client {sid} unsubscribed from execution {execution_id}")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in unsubscribe_execution: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def subscribe_metrics(sid, data):
    """Subscribe to real-time metrics."""
    try:
        await sio.enter_room(sid, "workflows_metrics")

        await sio.emit("subscribed", {
            "type": "metrics",
            "message": "Subscribed to real-time metrics"
        }, to=sid)

        logger.info(f"📊 [WORKFLOW SOCKET] Client {sid} subscribed to metrics")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in subscribe_metrics: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def unsubscribe_metrics(sid, data):
    """Unsubscribe from real-time metrics."""
    try:
        await sio.leave_room(sid, "workflows_metrics")

        await sio.emit("unsubscribed", {
            "type": "metrics",
            "message": "Unsubscribed from real-time metrics"
        }, to=sid)

        logger.info(f"📊 [WORKFLOW SOCKET] Client {sid} unsubscribed from metrics")

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in unsubscribe_metrics: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def get_workflow_status(sid, data):
    """Get current workflow status."""
    try:
        workflow_id = data.get("workflow_id")
        if not workflow_id:
            await sio.emit("error", {"message": "workflow_id required"}, to=sid)
            return

        # Get workflow status from execution service
        execution = await execution_service.get_latest_execution(workflow_id)

        status_data = {
            "workflow_id": workflow_id,
            "has_active_execution": execution is not None,
            "latest_execution": execution.model_dump() if execution else None,
            "timestamp": datetime.now().isoformat()
        }

        await sio.emit("workflow_status", status_data, to=sid)

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in get_workflow_status: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


@sio.event
async def get_execution_details(sid, data):
    """Get detailed execution information."""
    try:
        execution_id = data.get("execution_id")
        if not execution_id:
            await sio.emit("error", {"message": "execution_id required"}, to=sid)
            return

        execution = await execution_service.get_execution_status(execution_id)

        if not execution:
            await sio.emit("error", {"message": "Execution not found"}, to=sid)
            return

        await sio.emit("execution_details", execution.model_dump(), to=sid)

    except Exception as e:
        logger.error(f"❌ [WORKFLOW SOCKET] Error in get_execution_details: {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


# Background task for real-time metrics broadcasting
async def start_metrics_broadcast():
    """Start background task for broadcasting real-time metrics."""
    while True:
        try:
            metrics = await analytics_service.get_real_time_metrics()
            await broadcast_real_time_metrics(metrics)
        except Exception as e:
            logger.error(f"❌ [WORKFLOW SOCKET] Error in metrics broadcast: {e}")

        await asyncio.sleep(5)  # Broadcast every 5 seconds


# Initialize function to start background tasks
async def initialize_workflow_socketio():
    """Initialize workflow Socket.IO services."""
    logger.info("🔧 [WORKFLOW SOCKET] Initializing workflow Socket.IO services")

    # Start metrics broadcast task
    asyncio.create_task(start_metrics_broadcast())

    logger.info("✅ [WORKFLOW SOCKET] Workflow Socket.IO services initialized")


# Cleanup function
async def cleanup_workflow_socketio():
    """Cleanup workflow Socket.IO services."""
    logger.info("🧹 [WORKFLOW SOCKET] Cleaning up workflow Socket.IO services")

    # Clear room mappings
    _workflow_rooms.clear()
    _execution_rooms.clear()
    _last_broadcast_times.clear()

    logger.info("✅ [WORKFLOW SOCKET] Workflow Socket.IO services cleaned up")