"""
Agent Collaboration Service - Real-time pub/sub for multi-agent coordination

This service provides real-time communication and coordination capabilities
for agents using Redis pub/sub and Socket.IO for client updates.
"""

import logging
import json
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CollaborationEventType(Enum):
    """Types of collaboration events"""
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    KNOWLEDGE_SHARED = "knowledge_shared"
    HELP_REQUESTED = "help_requested"
    HELP_OFFERED = "help_offered"
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_GRANTED = "resource_granted"

class CoordinationPattern(Enum):
    """Multi-agent coordination patterns"""
    SEQUENTIAL = "sequential"  # Agents work in sequence
    PARALLEL = "parallel"  # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Manager-worker pattern
    PEER_TO_PEER = "peer_to_peer"  # Direct agent-to-agent
    BROADCAST = "broadcast"  # One-to-many communication
    CONSENSUS = "consensus"  # Group decision making
    DELEGATION = "delegation"  # Task handoff between agents

@dataclass
class CollaborationEvent:
    """Represents a collaboration event between agents"""
    id: UUID = field(default_factory=uuid4)
    event_type: CollaborationEventType = CollaborationEventType.STATUS_UPDATE
    source_agent_id: UUID = None
    target_agent_id: Optional[UUID] = None  # None for broadcast
    project_id: Optional[UUID] = None
    task_id: Optional[UUID] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    response_timeout: int = 30  # seconds

@dataclass
class AgentCollaborationSession:
    """Tracks an active collaboration session between agents"""
    session_id: UUID = field(default_factory=uuid4)
    project_id: UUID = None
    participating_agents: Set[UUID] = field(default_factory=set)
    coordination_pattern: CoordinationPattern = CoordinationPattern.PEER_TO_PEER
    session_context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True

@dataclass
class TaskCoordination:
    """Coordinates task execution across multiple agents"""
    coordination_id: UUID = field(default_factory=uuid4)
    primary_task_id: UUID = None
    subtasks: List[Dict[str, Any]] = field(default_factory=list)
    assigned_agents: Dict[UUID, UUID] = field(default_factory=dict)  # task_id -> agent_id
    dependencies: Dict[UUID, List[UUID]] = field(default_factory=dict)  # task_id -> [dependency_ids]
    status: Dict[UUID, str] = field(default_factory=dict)  # task_id -> status
    pattern: CoordinationPattern = CoordinationPattern.PARALLEL

class AgentCollaborationService:
    """Service for managing real-time agent collaboration"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        """
        Initialize the collaboration service
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.active_sessions: Dict[UUID, AgentCollaborationSession] = {}
        self.task_coordinations: Dict[UUID, TaskCoordination] = {}
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = {}
        self.agent_subscriptions: Dict[UUID, Set[str]] = {}  # agent_id -> set of channels
        self._running = False
        self._listener_task = None
        
        logger.info("Initialized Agent Collaboration Service")
    
    async def connect(self):
        """Connect to Redis and start listening for events"""
        try:
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to global collaboration channel
            await self.pubsub.subscribe("archon:collaboration:global")
            
            # Start the listener
            self._running = True
            self._listener_task = asyncio.create_task(self._listen_for_events())
            
            logger.info("Connected to Redis for agent collaboration")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup"""
        self._running = False
        
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Disconnected from Redis")
    
    async def _listen_for_events(self):
        """Listen for events from Redis pub/sub"""
        while self._running:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                
                if message and message['type'] == 'message':
                    await self._handle_redis_message(message)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
                await asyncio.sleep(1)
    
    async def _handle_redis_message(self, message: Dict[str, Any]):
        """Handle incoming Redis message"""
        try:
            data = json.loads(message['data'])
            event = CollaborationEvent(**data)
            
            # Process event based on type
            await self._process_event(event)
            
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")
    
    async def _process_event(self, event: CollaborationEvent):
        """Process a collaboration event"""
        # Call registered handlers
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
        
        # Log event for debugging
        logger.debug(f"Processed {event.event_type.value} event from agent {event.source_agent_id}")
    
    def register_event_handler(self, event_type: CollaborationEventType, handler: Callable):
        """Register a handler for specific event types"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} events")
    
    async def publish_event(self, event: CollaborationEvent) -> bool:
        """
        Publish a collaboration event
        
        Args:
            event: The event to publish
            
        Returns:
            True if published successfully
        """
        try:
            # Ensure we're connected
            if self.redis_client is None:
                await self.connect()
            # Determine channel based on target
            if event.target_agent_id:
                # Direct message to specific agent
                channel = f"archon:collaboration:agent:{event.target_agent_id}"
            elif event.project_id:
                # Project-wide broadcast
                channel = f"archon:collaboration:project:{event.project_id}"
            else:
                # Global broadcast
                channel = "archon:collaboration:global"
            
            # Serialize event
            event_data = {
                "id": str(event.id),
                "event_type": event.event_type.value,
                "source_agent_id": str(event.source_agent_id) if event.source_agent_id else None,
                "target_agent_id": str(event.target_agent_id) if event.target_agent_id else None,
                "project_id": str(event.project_id) if event.project_id else None,
                "task_id": str(event.task_id) if event.task_id else None,
                "payload": event.payload,
                "metadata": event.metadata,
                "timestamp": event.timestamp.isoformat(),
                "requires_response": event.requires_response,
                "response_timeout": event.response_timeout
            }
            
            # Publish to Redis
            await self.redis_client.publish(channel, json.dumps(event_data))
            
            # Store event in history (with TTL)
            history_key = f"archon:collaboration:history:{event.id}"
            await self.redis_client.setex(
                history_key,
                3600,  # 1 hour TTL
                json.dumps(event_data)
            )
            
            # Emit Socket.IO event for real-time client updates
            try:
                from ..socketio_app import get_socketio_instance
                sio = get_socketio_instance()
                socket_event_name = f"collaboration_{event.event_type.value}"
                
                # Emit to project room if project_id exists
                if event.project_id:
                    room = f"project_{event.project_id}"
                    await sio.emit(socket_event_name, event_data, room=room)
                    await sio.emit("collaboration_update", {
                        "type": event.event_type.value,
                        "data": event_data
                    }, room=room)
                else:
                    # Global broadcast
                    await sio.emit(socket_event_name, event_data)
                    await sio.emit("collaboration_update", {
                        "type": event.event_type.value,
                        "data": event_data
                    })
                
                logger.debug(f"Emitted Socket.IO event: {socket_event_name}")
            except ImportError:
                logger.debug("Socket.IO not available - skipping event emission")
            except Exception as e:
                logger.warning(f"Failed to emit Socket.IO event: {e}")
                # Don't fail the operation if Socket.IO emission fails
            
            logger.info(f"Published {event.event_type.value} event to {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    async def subscribe_agent(self, agent_id: UUID, channels: List[str] = None):
        """
        Subscribe an agent to collaboration channels
        
        Args:
            agent_id: The agent to subscribe
            channels: Additional channels to subscribe to
        """
        # Default channels for every agent
        default_channels = [
            f"archon:collaboration:agent:{agent_id}",  # Direct messages
            "archon:collaboration:global"  # Global broadcasts
        ]
        
        all_channels = default_channels + (channels or [])
        
        # Subscribe to channels
        for channel in all_channels:
            await self.pubsub.subscribe(channel)
        
        # Track subscriptions
        if agent_id not in self.agent_subscriptions:
            self.agent_subscriptions[agent_id] = set()
        self.agent_subscriptions[agent_id].update(all_channels)
        
        logger.info(f"Agent {agent_id} subscribed to {len(all_channels)} channels")
    
    async def unsubscribe_agent(self, agent_id: UUID):
        """Unsubscribe an agent from all channels"""
        if agent_id in self.agent_subscriptions:
            for channel in self.agent_subscriptions[agent_id]:
                await self.pubsub.unsubscribe(channel)
            
            del self.agent_subscriptions[agent_id]
            logger.info(f"Agent {agent_id} unsubscribed from all channels")
    
    async def create_collaboration_session(
        self,
        project_id: UUID,
        agents: List[UUID],
        pattern: CoordinationPattern = CoordinationPattern.PEER_TO_PEER,
        context: Dict[str, Any] = None
    ) -> AgentCollaborationSession:
        """
        Create a new collaboration session
        
        Args:
            project_id: Project ID for the session
            agents: List of participating agent IDs
            pattern: Coordination pattern to use
            context: Session context
            
        Returns:
            Created collaboration session
        """
        session = AgentCollaborationSession(
            project_id=project_id,
            participating_agents=set(agents),
            coordination_pattern=pattern,
            session_context=context or {}
        )
        
        self.active_sessions[session.session_id] = session
        
        # Notify agents about the session
        for agent_id in agents:
            await self.publish_event(CollaborationEvent(
                event_type=CollaborationEventType.AGENT_JOINED,
                source_agent_id=agent_id,
                project_id=project_id,
                payload={
                    "session_id": str(session.session_id),
                    "pattern": pattern.value,
                    "participants": [str(a) for a in agents]
                }
            ))
        
        logger.info(f"Created collaboration session {session.session_id} with {len(agents)} agents")
        return session
    
    async def coordinate_task_execution(
        self,
        primary_task: Dict[str, Any],
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, UUID],
        pattern: CoordinationPattern = CoordinationPattern.PARALLEL
    ) -> TaskCoordination:
        """
        Coordinate task execution across multiple agents
        
        Args:
            primary_task: The main task details
            subtasks: List of subtasks
            agent_assignments: Task ID to agent ID mapping
            pattern: Coordination pattern
            
        Returns:
            Task coordination object
        """
        coordination = TaskCoordination(
            primary_task_id=UUID(primary_task['id']),
            subtasks=subtasks,
            pattern=pattern
        )
        
        # Assign agents to tasks
        for task_id_str, agent_id in agent_assignments.items():
            task_id = UUID(task_id_str)
            coordination.assigned_agents[task_id] = agent_id
            coordination.status[task_id] = "pending"
        
        self.task_coordinations[coordination.coordination_id] = coordination
        
        # Start task execution based on pattern
        if pattern == CoordinationPattern.PARALLEL:
            # All tasks start simultaneously
            for task in subtasks:
                task_id = UUID(task['id'])
                agent_id = coordination.assigned_agents.get(task_id)
                if agent_id:
                    await self._dispatch_task_to_agent(task, agent_id, coordination.coordination_id)
                    
        elif pattern == CoordinationPattern.SEQUENTIAL:
            # Start first task only
            if subtasks:
                first_task = subtasks[0]
                task_id = UUID(first_task['id'])
                agent_id = coordination.assigned_agents.get(task_id)
                if agent_id:
                    await self._dispatch_task_to_agent(first_task, agent_id, coordination.coordination_id)
        
        logger.info(f"Started task coordination {coordination.coordination_id} with pattern {pattern.value}")
        return coordination
    
    async def _dispatch_task_to_agent(
        self,
        task: Dict[str, Any],
        agent_id: UUID,
        coordination_id: UUID
    ):
        """Dispatch a task to an agent"""
        await self.publish_event(CollaborationEvent(
            event_type=CollaborationEventType.COORDINATION_REQUEST,
            source_agent_id=UUID("00000000-0000-0000-0000-000000000000"),  # System
            target_agent_id=agent_id,
            task_id=UUID(task['id']),
            payload={
                "task": task,
                "coordination_id": str(coordination_id)
            },
            requires_response=True
        ))
    
    async def handle_task_completion(
        self,
        task_id: UUID,
        agent_id: UUID,
        result: Dict[str, Any]
    ):
        """
        Handle task completion from an agent
        
        Args:
            task_id: Completed task ID
            agent_id: Agent that completed the task
            result: Task result
        """
        # Find the coordination
        coordination = None
        for coord in self.task_coordinations.values():
            if task_id in coord.assigned_agents:
                coordination = coord
                break
        
        if not coordination:
            logger.warning(f"No coordination found for task {task_id}")
            return
        
        # Update status
        coordination.status[task_id] = "completed"
        
        # Check if we need to start next task (sequential pattern)
        if coordination.pattern == CoordinationPattern.SEQUENTIAL:
            # Find next pending task
            for i, subtask in enumerate(coordination.subtasks):
                subtask_id = UUID(subtask['id'])
                if coordination.status.get(subtask_id) == "pending":
                    agent_id = coordination.assigned_agents.get(subtask_id)
                    if agent_id:
                        await self._dispatch_task_to_agent(subtask, agent_id, coordination.coordination_id)
                    break
        
        # Check if all tasks are complete
        all_complete = all(
            status in ["completed", "failed"]
            for status in coordination.status.values()
        )
        
        if all_complete:
            # Notify about coordination completion
            await self.publish_event(CollaborationEvent(
                event_type=CollaborationEventType.TASK_COMPLETED,
                source_agent_id=UUID("00000000-0000-0000-0000-000000000000"),
                payload={
                    "coordination_id": str(coordination.coordination_id),
                    "primary_task_id": str(coordination.primary_task_id),
                    "results": result
                }
            ))
            
            logger.info(f"Task coordination {coordination.coordination_id} completed")
    
    async def request_agent_help(
        self,
        requesting_agent: UUID,
        task_context: Dict[str, Any],
        required_capabilities: List[str]
    ) -> Optional[UUID]:
        """
        Request help from other agents
        
        Args:
            requesting_agent: Agent requesting help
            task_context: Context about the task
            required_capabilities: Capabilities needed
            
        Returns:
            Agent ID that offered help, if any
        """
        # Publish help request
        await self.publish_event(CollaborationEvent(
            event_type=CollaborationEventType.HELP_REQUESTED,
            source_agent_id=requesting_agent,
            payload={
                "task_context": task_context,
                "required_capabilities": required_capabilities
            },
            requires_response=True,
            response_timeout=10
        ))
        
        # Wait for responses (simplified - in production would use proper async response handling)
        await asyncio.sleep(2)
        
        # Check for help offers in Redis
        offers_key = f"archon:collaboration:help_offers:{requesting_agent}"
        offers = await self.redis_client.lrange(offers_key, 0, -1)
        
        if offers:
            # Return first offer (could be more sophisticated selection)
            helper_agent = UUID(offers[0])
            logger.info(f"Agent {helper_agent} offered help to {requesting_agent}")
            return helper_agent
        
        return None
    
    async def share_knowledge_between_agents(
        self,
        source_agent: UUID,
        target_agents: List[UUID],
        knowledge: Dict[str, Any]
    ):
        """
        Share knowledge between agents
        
        Args:
            source_agent: Agent sharing knowledge
            target_agents: Agents to share with
            knowledge: Knowledge to share
        """
        for target in target_agents:
            await self.publish_event(CollaborationEvent(
                event_type=CollaborationEventType.KNOWLEDGE_SHARED,
                source_agent_id=source_agent,
                target_agent_id=target,
                payload={
                    "knowledge": knowledge,
                    "shared_at": datetime.now().isoformat()
                }
            ))
        
        logger.info(f"Agent {source_agent} shared knowledge with {len(target_agents)} agents")
    
    async def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get metrics about ongoing collaborations"""
        metrics = {
            "active_sessions": len(self.active_sessions),
            "active_coordinations": len(self.task_coordinations),
            "subscribed_agents": len(self.agent_subscriptions),
            "sessions_by_pattern": {},
            "coordinations_by_status": {"pending": 0, "in_progress": 0, "completed": 0}
        }
        
        # Count sessions by pattern
        for session in self.active_sessions.values():
            pattern = session.coordination_pattern.value
            metrics["sessions_by_pattern"][pattern] = metrics["sessions_by_pattern"].get(pattern, 0) + 1
        
        # Count coordinations by status
        for coordination in self.task_coordinations.values():
            for status in coordination.status.values():
                if status in metrics["coordinations_by_status"]:
                    metrics["coordinations_by_status"][status] += 1
        
        return metrics

# Global instance
_collaboration_service = None

async def get_collaboration_service() -> AgentCollaborationService:
    """Get or create the global collaboration service instance"""
    global _collaboration_service
    if _collaboration_service is None:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        _collaboration_service = AgentCollaborationService(redis_url=redis_url)
        await _collaboration_service.connect()
    return _collaboration_service

# Alias for compatibility
def get_agent_collaboration_service(redis_url: str = "redis://redis:6379") -> AgentCollaborationService:
    """Get or create the global agent collaboration service instance (sync wrapper)"""
    global _collaboration_service
    if _collaboration_service is None:
        _collaboration_service = AgentCollaborationService(redis_url)
    return _collaboration_service