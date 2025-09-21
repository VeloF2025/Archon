"""
Socket.IO handlers for agent collaboration events

This module provides real-time WebSocket handlers for agent collaboration,
allowing clients to monitor and interact with multi-agent coordination.
"""

import logging
import json
from typing import Dict, Any
from uuid import UUID

from ..socketio_app import get_socketio_instance

logger = logging.getLogger(__name__)

# Get Socket.IO instance
sio = get_socketio_instance()


@sio.event
async def join_collaboration_room(sid: str, data: Dict[str, Any]):
    """
    Join a collaboration room for real-time updates
    
    Args:
        sid: Socket session ID
        data: Contains 'project_id' or 'agent_id' to join
    """
    try:
        project_id = data.get("project_id")
        agent_id = data.get("agent_id")
        
        rooms_joined = []
        
        if project_id:
            room = f"project_{project_id}"
            sio.enter_room(sid, room)
            rooms_joined.append(room)
            logger.info(f"Client {sid} joined project room: {room}")
        
        if agent_id:
            room = f"agent_{agent_id}"
            sio.enter_room(sid, room)
            rooms_joined.append(room)
            logger.info(f"Client {sid} joined agent room: {room}")
        
        # Send confirmation
        await sio.emit("collaboration_room_joined", {
            "rooms": rooms_joined,
            "status": "success"
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error joining collaboration room: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "join_room"
        }, to=sid)


@sio.event
async def leave_collaboration_room(sid: str, data: Dict[str, Any]):
    """
    Leave a collaboration room
    
    Args:
        sid: Socket session ID
        data: Contains 'project_id' or 'agent_id' to leave
    """
    try:
        project_id = data.get("project_id")
        agent_id = data.get("agent_id")
        
        rooms_left = []
        
        if project_id:
            room = f"project_{project_id}"
            sio.leave_room(sid, room)
            rooms_left.append(room)
            logger.info(f"Client {sid} left project room: {room}")
        
        if agent_id:
            room = f"agent_{agent_id}"
            sio.leave_room(sid, room)
            rooms_left.append(room)
            logger.info(f"Client {sid} left agent room: {room}")
        
        # Send confirmation
        await sio.emit("collaboration_room_left", {
            "rooms": rooms_left,
            "status": "success"
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Error leaving collaboration room: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "leave_room"
        }, to=sid)


@sio.event
async def request_collaboration_status(sid: str, data: Dict[str, Any]):
    """
    Request current collaboration status for a project or agent
    
    Args:
        sid: Socket session ID
        data: Contains 'project_id' or 'agent_id'
    """
    try:
        from ..services.agent_collaboration_service import get_agent_collaboration_service
        collaboration_service = get_agent_collaboration_service()
        project_id = data.get("project_id")
        agent_id = data.get("agent_id")
        
        status = {}
        
        if project_id and UUID(project_id) in collaboration_service.active_sessions:
            session = collaboration_service.active_sessions[UUID(project_id)]
            status["session"] = {
                "id": str(session.id),
                "project_id": str(session.project_id),
                "pattern": session.pattern.value,
                "state": session.state,
                "participants": [str(p) for p in session.participants],
                "active_agents": [str(a) for a in session.active_agents],
                "created_at": session.created_at.isoformat()
            }
        
        if agent_id:
            # Get agent-specific collaboration info
            agent_uuid = UUID(agent_id)
            if agent_uuid in collaboration_service.agent_subscriptions:
                status["subscriptions"] = list(collaboration_service.agent_subscriptions[agent_uuid])
            
            # Check if agent is in any active sessions
            for session_id, session in collaboration_service.active_sessions.items():
                if agent_uuid in session.participants:
                    status["active_session"] = str(session_id)
                    break
        
        await sio.emit("collaboration_status", status, to=sid)
        
    except Exception as e:
        logger.error(f"Error getting collaboration status: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "get_status"
        }, to=sid)


@sio.event
async def send_collaboration_message(sid: str, data: Dict[str, Any]):
    """
    Send a collaboration message between agents or to a project
    
    Args:
        sid: Socket session ID
        data: Message data including type, source, target, payload
    """
    try:
        from ..services.agent_collaboration_service import (
            get_agent_collaboration_service,
            CollaborationEvent,
            CollaborationEventType
        )
        collaboration_service = get_agent_collaboration_service()
        
        # Create collaboration event from message
        event = CollaborationEvent(
            event_type=CollaborationEventType[data.get("type", "STATUS_UPDATE")],
            source_agent_id=UUID(data["source_agent_id"]) if "source_agent_id" in data else None,
            target_agent_id=UUID(data["target_agent_id"]) if "target_agent_id" in data else None,
            project_id=UUID(data["project_id"]) if "project_id" in data else None,
            task_id=UUID(data["task_id"]) if "task_id" in data else None,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {})
        )
        
        # Publish the event
        success = await collaboration_service.publish_event(event)
        
        if success:
            await sio.emit("collaboration_message_sent", {
                "event_id": str(event.id),
                "status": "success"
            }, to=sid)
        else:
            await sio.emit("collaboration_error", {
                "error": "Failed to send message",
                "action": "send_message"
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Error sending collaboration message: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "send_message"
        }, to=sid)


@sio.event
async def start_collaboration_session(sid: str, data: Dict[str, Any]):
    """
    Start a new collaboration session
    
    Args:
        sid: Socket session ID
        data: Session configuration including agents, pattern, context
    """
    try:
        from ..services.agent_collaboration_service import (
            get_agent_collaboration_service,
            CoordinationPattern
        )
        collaboration_service = get_agent_collaboration_service()
        
        # Create collaboration session
        session = await collaboration_service.create_collaboration_session(
            project_id=UUID(data["project_id"]),
            agents=[UUID(a) for a in data["agents"]],
            pattern=CoordinationPattern[data.get("pattern", "PEER_TO_PEER")],
            context=data.get("context", {})
        )
        
        # Send session created event
        session_data = {
            "session_id": str(session.id),
            "project_id": str(session.project_id),
            "pattern": session.pattern.value,
            "participants": [str(p) for p in session.participants],
            "status": "started"
        }
        
        # Notify all participants
        room = f"project_{session.project_id}"
        await sio.emit("collaboration_session_started", session_data, room=room)
        
        # Send confirmation to requester
        await sio.emit("collaboration_session_created", session_data, to=sid)
        
    except Exception as e:
        logger.error(f"Error starting collaboration session: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "start_session"
        }, to=sid)


@sio.event
async def stop_collaboration_session(sid: str, data: Dict[str, Any]):
    """
    Stop an active collaboration session
    
    Args:
        sid: Socket session ID
        data: Contains 'session_id' to stop
    """
    try:
        from ..services.agent_collaboration_service import get_agent_collaboration_service
        collaboration_service = get_agent_collaboration_service()
        session_id = UUID(data["session_id"])
        
        if session_id in collaboration_service.active_sessions:
            session = collaboration_service.active_sessions[session_id]
            
            # End the session
            await collaboration_service.end_collaboration_session(session_id)
            
            # Notify all participants
            room = f"project_{session.project_id}"
            await sio.emit("collaboration_session_stopped", {
                "session_id": str(session_id),
                "status": "stopped"
            }, room=room)
            
            # Send confirmation
            await sio.emit("collaboration_session_ended", {
                "session_id": str(session_id),
                "status": "success"
            }, to=sid)
        else:
            await sio.emit("collaboration_error", {
                "error": "Session not found",
                "action": "stop_session"
            }, to=sid)
            
    except Exception as e:
        logger.error(f"Error stopping collaboration session: {e}")
        await sio.emit("collaboration_error", {
            "error": str(e),
            "action": "stop_session"
        }, to=sid)


# Register handlers when module is imported
logger.info("Registered Socket.IO collaboration handlers")