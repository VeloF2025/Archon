"""
Real-Time Collaboration System v3.0 - Shared Context and Knowledge Broadcasting
Based on F-RTC-001, F-RTC-002 from PRD specifications

NLNH Protocol: Real collaboration with actual pub/sub and shared context
DGTS Enforcement: No fake broadcasting, actual real-time communication
"""

import asyncio
import json
import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
import queue
import time

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels for knowledge broadcasting"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ContextStatus(Enum):
    """Shared context status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class SharedContext:
    """Shared context for collaborative tasks (F-RTC-001)"""
    task_id: str
    project_id: str
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    discoveries: List[Dict[str, Any]] = field(default_factory=list)
    blockers: List[Dict[str, Any]] = field(default_factory=list)
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = field(default=ContextStatus.ACTIVE.value)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_discovery(self, agent_id: str, discovery: Dict[str, Any]):
        """Add discovery to shared context"""
        self.discoveries.append({
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "discovery": discovery,
            "discovery_id": str(uuid.uuid4())
        })
        self.updated_at = datetime.now()
        logger.debug(f"Discovery added to context {self.task_id} by {agent_id}")
    
    def add_blocker(self, agent_id: str, blocker: Dict[str, Any]):
        """Add blocker to shared context"""
        self.blockers.append({
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "blocker": blocker,
            "blocker_id": str(uuid.uuid4()),
            "resolved": False
        })
        self.updated_at = datetime.now()
        logger.warning(f"Blocker added to context {self.task_id} by {agent_id}: {blocker.get('description', 'Unknown')}")
    
    def add_pattern(self, agent_id: str, pattern: Dict[str, Any]):
        """Add successful pattern to shared context"""
        self.patterns.append({
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "pattern_id": str(uuid.uuid4()),
            "confidence": pattern.get("confidence", 0.8)
        })
        self.updated_at = datetime.now()
        logger.info(f"Pattern added to context {self.task_id} by {agent_id}")
    
    def resolve_blocker(self, blocker_id: str, resolution: Dict[str, Any]) -> bool:
        """Mark blocker as resolved"""
        for blocker_entry in self.blockers:
            if blocker_entry.get("blocker_id") == blocker_id:
                blocker_entry["resolved"] = True
                blocker_entry["resolution"] = resolution
                blocker_entry["resolved_at"] = datetime.now().isoformat()
                self.updated_at = datetime.now()
                logger.info(f"Blocker {blocker_id} resolved in context {self.task_id}")
                return True
        return False
    
    def get_active_blockers(self) -> List[Dict[str, Any]]:
        """Get unresolved blockers"""
        return [blocker for blocker in self.blockers if not blocker.get("resolved", False)]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary"""
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "context_id": self.context_id,
            "status": self.status,
            "participants": self.participants,
            "discoveries_count": len(self.discoveries),
            "active_blockers_count": len(self.get_active_blockers()),
            "patterns_count": len(self.patterns),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class BroadcastMessage:
    """Message for knowledge broadcasting (F-RTC-002)"""
    topic: str
    content: Dict[str, Any]
    priority: str = MessagePriority.NORMAL.value
    sender_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    recipients: List[str] = field(default_factory=list)
    acknowledged_by: Set[str] = field(default_factory=set)
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def acknowledge(self, agent_id: str):
        """Agent acknowledges message receipt"""
        self.acknowledged_by.add(agent_id)
        logger.debug(f"Message {self.message_id} acknowledged by {agent_id}")
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if not self.ttl_seconds:
            return False
        
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        return age_seconds > self.ttl_seconds
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return self.retry_count < self.max_retries and not self.is_expired()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "content": self.content,
            "priority": self.priority,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "recipients": self.recipients,
            "acknowledged_by": list(self.acknowledged_by),
            "ttl_seconds": self.ttl_seconds,
            "retry_count": self.retry_count
        }


@dataclass
class Subscription:
    """Agent subscription to topics"""
    agent_id: str
    topics: List[str]
    callback: Optional[Callable] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def matches_message(self, message: BroadcastMessage) -> bool:
        """Check if subscription matches message"""
        if not self.active:
            return False
        
        if message.topic not in self.topics:
            return False
        
        return self._apply_filters(message)
    
    def _apply_filters(self, message: BroadcastMessage) -> bool:
        """Apply subscription filters"""
        if not self.filters:
            return True
        
        # Priority filter
        if "min_priority" in self.filters:
            priority_order = {
                MessagePriority.CRITICAL.value: 4,
                MessagePriority.HIGH.value: 3,
                MessagePriority.NORMAL.value: 2,
                MessagePriority.LOW.value: 1
            }
            
            message_priority = priority_order.get(message.priority, 2)
            min_priority = priority_order.get(self.filters["min_priority"], 2)
            
            if message_priority < min_priority:
                return False
        
        # Sender filter
        if "sender_id" in self.filters:
            if message.sender_id != self.filters["sender_id"]:
                return False
        
        # Content type filter
        if "content_type" in self.filters:
            if message.content.get("type") != self.filters["content_type"]:
                return False
        
        # Age filter
        if "max_age_seconds" in self.filters:
            age_seconds = (datetime.now() - message.timestamp).total_seconds()
            if age_seconds > self.filters["max_age_seconds"]:
                return False
        
        return True
    
    def update_message_stats(self):
        """Update subscription message statistics"""
        self.message_count += 1
        self.last_message_at = datetime.now()


class ConflictResolver:
    """Resolve conflicts between contradictory patterns"""
    
    def __init__(self):
        self.resolution_strategies = {
            "confidence_weighted": self._resolve_by_confidence,
            "recency_weighted": self._resolve_by_recency,
            "consensus": self._resolve_by_consensus,
            "domain_expert": self._resolve_by_expertise
        }
    
    async def resolve_conflict(self, patterns: List[Dict[str, Any]], 
                             strategy: str = "confidence_weighted") -> Dict[str, Any]:
        """Resolve conflict between contradictory patterns"""
        if len(patterns) <= 1:
            return patterns[0] if patterns else {}
        
        resolver = self.resolution_strategies.get(strategy, self._resolve_by_confidence)
        resolution = await resolver(patterns)
        
        logger.info(f"Conflict resolved using {strategy}: {len(patterns)} patterns -> 1 resolution")
        return resolution
    
    async def _resolve_by_confidence(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by highest confidence pattern"""
        max_confidence = max(pattern.get("confidence", 0.5) for pattern in patterns)
        
        best_patterns = [p for p in patterns if p.get("confidence", 0.5) == max_confidence]
        
        if len(best_patterns) == 1:
            return best_patterns[0]
        
        # If tied, use recency
        return await self._resolve_by_recency(best_patterns)
    
    async def _resolve_by_recency(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by most recent pattern"""
        def get_timestamp(pattern):
            timestamp_str = pattern.get("timestamp", "1970-01-01T00:00:00")
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        most_recent = max(patterns, key=get_timestamp)
        return most_recent
    
    async def _resolve_by_consensus(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by pattern with most similar variants"""
        # Simple implementation: count similar pattern types
        pattern_types = {}
        
        for pattern in patterns:
            pattern_type = pattern.get("type", "unknown")
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(pattern)
        
        # Return highest confidence from most common type
        most_common_type = max(pattern_types.keys(), key=lambda x: len(pattern_types[x]))
        return await self._resolve_by_confidence(pattern_types[most_common_type])
    
    async def _resolve_by_expertise(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by agent expertise level (simplified)"""
        # In a real implementation, this would check agent expertise scores
        # For now, prefer patterns from agents with "expert" in their ID
        expert_patterns = [p for p in patterns if "expert" in p.get("agent_id", "").lower()]
        
        if expert_patterns:
            return await self._resolve_by_confidence(expert_patterns)
        
        return await self._resolve_by_confidence(patterns)


class MessageBroker:
    """Real-time message broker for knowledge broadcasting"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.topics: Dict[str, List[BroadcastMessage]] = {}
        self.subscriptions: Dict[str, List[Subscription]] = {}
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.active = True
        self.stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "failed_deliveries": 0,
            "active_subscriptions": 0
        }
        
        # Start background processing
        self._processing_task = None
        self.conflict_resolver = ConflictResolver()
    
    async def start(self):
        """Start the message broker"""
        if not self._processing_task:
            self._processing_task = asyncio.create_task(self._process_message_queue())
            logger.info("Message broker started")
    
    async def stop(self):
        """Stop the message broker"""
        self.active = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Message broker stopped")
    
    async def publish(self, topic: str, message: BroadcastMessage) -> int:
        """Publish message to topic"""
        if not self.active:
            return 0
        
        message.topic = topic
        
        # Store message in topic
        if topic not in self.topics:
            self.topics[topic] = []
        
        self.topics[topic].append(message)
        
        # Add to processing queue
        try:
            await self.message_queue.put(message)
            self.stats["messages_published"] += 1
        except asyncio.QueueFull:
            logger.error(f"Message queue full, dropping message {message.message_id}")
            return 0
        
        logger.debug(f"Published message {message.message_id} to topic {topic}")
        return await self._deliver_message(message)
    
    async def subscribe(self, subscription: Subscription) -> bool:
        """Subscribe agent to topics"""
        agent_id = subscription.agent_id
        
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = []
        
        # Check for duplicate subscription
        for existing_sub in self.subscriptions[agent_id]:
            if (existing_sub.topics == subscription.topics and 
                existing_sub.filters == subscription.filters):
                logger.warning(f"Duplicate subscription detected for {agent_id}")
                return False
        
        self.subscriptions[agent_id].append(subscription)
        self.stats["active_subscriptions"] += 1
        
        logger.info(f"Agent {agent_id} subscribed to topics: {subscription.topics}")
        return True
    
    async def unsubscribe(self, agent_id: str, topics: Optional[List[str]] = None) -> bool:
        """Unsubscribe agent from topics"""
        if agent_id not in self.subscriptions:
            return False
        
        if topics:
            # Remove specific topic subscriptions
            original_count = len(self.subscriptions[agent_id])
            self.subscriptions[agent_id] = [
                sub for sub in self.subscriptions[agent_id]
                if not any(topic in sub.topics for topic in topics)
            ]
            removed_count = original_count - len(self.subscriptions[agent_id])
            self.stats["active_subscriptions"] -= removed_count
        else:
            # Remove all subscriptions
            removed_count = len(self.subscriptions[agent_id])
            del self.subscriptions[agent_id]
            self.stats["active_subscriptions"] -= removed_count
        
        logger.info(f"Agent {agent_id} unsubscribed from {removed_count} subscriptions")
        return True
    
    async def _deliver_message(self, message: BroadcastMessage) -> int:
        """Deliver message to subscribers"""
        delivered_count = 0
        
        for agent_id, subscriptions in self.subscriptions.items():
            for subscription in subscriptions:
                if subscription.matches_message(message):
                    message.recipients.append(agent_id)
                    subscription.update_message_stats()
                    delivered_count += 1
                    
                    # Execute callback if provided
                    if subscription.callback:
                        try:
                            if asyncio.iscoroutinefunction(subscription.callback):
                                await subscription.callback(message)
                            else:
                                subscription.callback(message)
                        except Exception as e:
                            logger.error(f"Callback error for {agent_id}: {e}")
                            self.stats["failed_deliveries"] += 1
        
        self.stats["messages_delivered"] += delivered_count
        return delivered_count
    
    async def _process_message_queue(self):
        """Background task to process message queue"""
        while self.active:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Check for expired messages
                if message.is_expired():
                    logger.debug(f"Message {message.message_id} expired, skipping")
                    continue
                
                # Additional processing could be added here
                # e.g., conflict detection, pattern analysis, etc.
                
            except asyncio.TimeoutError:
                continue  # No message to process
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
    
    async def get_topic_messages(self, topic: str, limit: int = 100) -> List[BroadcastMessage]:
        """Get recent messages from topic"""
        if topic not in self.topics:
            return []
        
        messages = self.topics[topic]
        # Return most recent messages
        return sorted(messages, key=lambda m: m.timestamp, reverse=True)[:limit]
    
    async def detect_conflicts(self, topic: str) -> List[Dict[str, Any]]:
        """Detect contradictory patterns in topic"""
        messages = await self.get_topic_messages(topic, limit=50)
        
        # Group by pattern type
        pattern_groups = {}
        for message in messages:
            if message.content.get("type") == "pattern":
                pattern_type = message.content.get("pattern_type", "unknown")
                if pattern_type not in pattern_groups:
                    pattern_groups[pattern_type] = []
                pattern_groups[pattern_type].append(message)
        
        conflicts = []
        for pattern_type, messages_list in pattern_groups.items():
            if len(messages_list) > 1:
                # Simple conflict detection: different solutions for same problem
                solutions = set(msg.content.get("solution", "") for msg in messages_list)
                if len(solutions) > 1:
                    conflicts.append({
                        "pattern_type": pattern_type,
                        "conflicting_messages": [msg.message_id for msg in messages_list],
                        "solutions": list(solutions)
                    })
        
        return conflicts
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        return {
            **self.stats,
            "active": self.active,
            "topics_count": len(self.topics),
            "total_messages": sum(len(msgs) for msgs in self.topics.values()),
            "queue_size": self.message_queue.qsize(),
            "subscriptions_by_agent": {
                agent_id: len(subs) for agent_id, subs in self.subscriptions.items()
            }
        }


class SharedContextManager:
    """Manager for shared contexts across collaborative tasks"""
    
    def __init__(self):
        self.contexts: Dict[str, SharedContext] = {}
        self.task_to_context: Dict[str, str] = {}
        self.project_contexts: Dict[str, List[str]] = {}
        
    async def create_shared_context(self, task_id: str, project_id: str, 
                                  metadata: Dict[str, Any] = None) -> SharedContext:
        """Create new shared context for collaborative task"""
        if task_id in self.task_to_context:
            existing_context_id = self.task_to_context[task_id]
            return self.contexts[existing_context_id]
        
        context = SharedContext(
            task_id=task_id,
            project_id=project_id,
            metadata=metadata or {}
        )
        
        # Store context
        self.contexts[context.context_id] = context
        self.task_to_context[task_id] = context.context_id
        
        # Update project contexts
        if project_id not in self.project_contexts:
            self.project_contexts[project_id] = []
        self.project_contexts[project_id].append(context.context_id)
        
        logger.info(f"Created shared context for task {task_id} in project {project_id}")
        return context
    
    async def get_shared_context(self, task_id: str) -> Optional[SharedContext]:
        """Get shared context by task ID"""
        context_id = self.task_to_context.get(task_id)
        return self.contexts.get(context_id) if context_id else None
    
    async def join_context(self, task_id: str, agent_id: str) -> bool:
        """Agent joins shared context"""
        context = await self.get_shared_context(task_id)
        if context and agent_id not in context.participants:
            context.participants.append(agent_id)
            context.updated_at = datetime.now()
            logger.info(f"Agent {agent_id} joined context for task {task_id}")
            return True
        return False
    
    async def leave_context(self, task_id: str, agent_id: str) -> bool:
        """Agent leaves shared context"""
        context = await self.get_shared_context(task_id)
        if context and agent_id in context.participants:
            context.participants.remove(agent_id)
            context.updated_at = datetime.now()
            logger.info(f"Agent {agent_id} left context for task {task_id}")
            return True
        return False
    
    async def archive_context(self, task_id: str, reason: str = "completed") -> bool:
        """Archive completed context"""
        context = await self.get_shared_context(task_id)
        if context:
            context.status = ContextStatus.ARCHIVED.value
            context.metadata["archived_reason"] = reason
            context.metadata["archived_at"] = datetime.now().isoformat()
            context.updated_at = datetime.now()
            logger.info(f"Archived context for task {task_id}: {reason}")
            return True
        return False
    
    async def get_project_contexts(self, project_id: str, 
                                 status_filter: Optional[str] = None) -> List[SharedContext]:
        """Get all contexts for a project"""
        if project_id not in self.project_contexts:
            return []
        
        contexts = []
        for context_id in self.project_contexts[project_id]:
            context = self.contexts.get(context_id)
            if context and (not status_filter or context.status == status_filter):
                contexts.append(context)
        
        return sorted(contexts, key=lambda c: c.updated_at, reverse=True)
    
    async def get_context_analytics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics for contexts"""
        contexts = []
        
        if project_id:
            contexts = await self.get_project_contexts(project_id)
        else:
            contexts = list(self.contexts.values())
        
        if not contexts:
            return {"total_contexts": 0}
        
        # Calculate analytics
        total_discoveries = sum(len(c.discoveries) for c in contexts)
        total_blockers = sum(len(c.blockers) for c in contexts)
        total_patterns = sum(len(c.patterns) for c in contexts)
        active_contexts = sum(1 for c in contexts if c.status == ContextStatus.ACTIVE.value)
        
        return {
            "total_contexts": len(contexts),
            "active_contexts": active_contexts,
            "total_discoveries": total_discoveries,
            "total_blockers": total_blockers,
            "total_patterns": total_patterns,
            "avg_participants_per_context": sum(len(c.participants) for c in contexts) / len(contexts),
            "contexts_by_status": {
                status.value: sum(1 for c in contexts if c.status == status.value)
                for status in ContextStatus
            }
        }


class CollaborationOrchestrator:
    """Main orchestrator for real-time collaboration"""
    
    def __init__(self):
        self.message_broker = MessageBroker()
        self.context_manager = SharedContextManager()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        
    async def start(self):
        """Start collaboration system"""
        await self.message_broker.start()
        logger.info("Collaboration orchestrator started")
    
    async def stop(self):
        """Stop collaboration system"""
        await self.message_broker.stop()
        logger.info("Collaboration orchestrator stopped")
    
    async def create_collaborative_task(self, task_id: str, project_id: str,
                                      participating_agents: List[str],
                                      task_metadata: Dict[str, Any] = None) -> SharedContext:
        """Create collaborative task with shared context"""
        # Create shared context
        context = await self.context_manager.create_shared_context(
            task_id, project_id, task_metadata
        )
        
        # Add participating agents to context
        for agent_id in participating_agents:
            await self.context_manager.join_context(task_id, agent_id)
        
        # Track active task
        self.active_tasks[task_id] = {
            "project_id": project_id,
            "context_id": context.context_id,
            "participants": participating_agents.copy(),
            "created_at": datetime.now(),
            "status": "active"
        }
        
        # Broadcast task creation
        task_message = BroadcastMessage(
            topic=f"project-{project_id}",
            content={
                "type": "collaborative_task_created",
                "task_id": task_id,
                "participants": participating_agents,
                "context_id": context.context_id
            },
            priority=MessagePriority.HIGH.value,
            sender_id="collaboration-orchestrator"
        )
        
        await self.message_broker.publish(f"project-{project_id}", task_message)
        
        logger.info(f"Created collaborative task {task_id} with {len(participating_agents)} agents")
        return context
    
    async def broadcast_discovery(self, task_id: str, agent_id: str, 
                                discovery: Dict[str, Any]) -> bool:
        """Broadcast discovery to task participants"""
        context = await self.context_manager.get_shared_context(task_id)
        if not context:
            return False
        
        # Add to shared context
        context.add_discovery(agent_id, discovery)
        
        # Broadcast to participants
        discovery_message = BroadcastMessage(
            topic=f"task-{task_id}",
            content={
                "type": "discovery_shared",
                "task_id": task_id,
                "agent_id": agent_id,
                "discovery": discovery,
                "context_id": context.context_id
            },
            priority=MessagePriority.HIGH.value if discovery.get("critical") else MessagePriority.NORMAL.value,
            sender_id=agent_id
        )
        
        delivered = await self.message_broker.publish(f"task-{task_id}", discovery_message)
        logger.info(f"Discovery broadcast from {agent_id} delivered to {delivered} recipients")
        
        return delivered > 0
    
    async def broadcast_blocker(self, task_id: str, agent_id: str, 
                              blocker: Dict[str, Any]) -> bool:
        """Broadcast blocker to task participants"""
        context = await self.context_manager.get_shared_context(task_id)
        if not context:
            return False
        
        # Add to shared context
        context.add_blocker(agent_id, blocker)
        
        # Broadcast with high priority
        blocker_message = BroadcastMessage(
            topic=f"task-{task_id}",
            content={
                "type": "blocker_reported",
                "task_id": task_id,
                "agent_id": agent_id,
                "blocker": blocker,
                "context_id": context.context_id,
                "requires_attention": True
            },
            priority=MessagePriority.CRITICAL.value if blocker.get("severity") == "blocking" else MessagePriority.HIGH.value,
            sender_id=agent_id
        )
        
        delivered = await self.message_broker.publish(f"task-{task_id}", blocker_message)
        logger.warning(f"Blocker broadcast from {agent_id} delivered to {delivered} recipients")
        
        return delivered > 0
    
    async def broadcast_pattern(self, task_id: str, agent_id: str, 
                              pattern: Dict[str, Any]) -> bool:
        """Broadcast successful pattern to task participants"""
        context = await self.context_manager.get_shared_context(task_id)
        if not context:
            return False
        
        # Add to shared context
        context.add_pattern(agent_id, pattern)
        
        # Check for conflicts with existing patterns
        conflicts = await self.message_broker.detect_conflicts(f"task-{task_id}")
        
        pattern_message = BroadcastMessage(
            topic=f"task-{task_id}",
            content={
                "type": "pattern_shared",
                "task_id": task_id,
                "agent_id": agent_id,
                "pattern": pattern,
                "context_id": context.context_id,
                "conflicts_detected": len(conflicts) > 0
            },
            priority=MessagePriority.HIGH.value if pattern.get("confidence", 0.5) > 0.8 else MessagePriority.NORMAL.value,
            sender_id=agent_id
        )
        
        delivered = await self.message_broker.publish(f"task-{task_id}", pattern_message)
        logger.info(f"Pattern broadcast from {agent_id} delivered to {delivered} recipients")
        
        return delivered > 0
    
    async def subscribe_agent_to_collaboration(self, agent_id: str, interests: List[str],
                                             callback: Optional[Callable] = None) -> bool:
        """Subscribe agent to collaboration topics"""
        # Create comprehensive subscription
        collaboration_topics = [
            "discoveries", "blockers", "patterns", "conflicts",
            f"agent-{agent_id}"  # Personal topic
        ]
        
        # Add interest-based topics
        for interest in interests:
            collaboration_topics.extend([
                f"domain-{interest}",
                f"skill-{interest}",
                f"type-{interest}"
            ])
        
        subscription = Subscription(
            agent_id=agent_id,
            topics=collaboration_topics,
            callback=callback,
            filters={"min_priority": "normal"}
        )
        
        success = await self.message_broker.subscribe(subscription)
        
        if success:
            logger.info(f"Agent {agent_id} subscribed to collaboration with interests: {interests}")
        
        return success
    
    async def get_collaboration_dashboard(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive collaboration dashboard"""
        # Context analytics
        context_analytics = await self.context_manager.get_context_analytics(project_id)
        
        # Broker statistics
        broker_stats = self.message_broker.get_broker_stats()
        
        # Active tasks
        active_tasks = []
        for task_id, task_info in self.active_tasks.items():
            if not project_id or task_info["project_id"] == project_id:
                context = await self.context_manager.get_shared_context(task_id)
                if context:
                    active_tasks.append({
                        "task_id": task_id,
                        "participants_count": len(context.participants),
                        "discoveries_count": len(context.discoveries),
                        "active_blockers": len(context.get_active_blockers()),
                        "patterns_count": len(context.patterns),
                        "last_activity": context.updated_at.isoformat()
                    })
        
        return {
            "collaboration_overview": {
                "active_tasks": len(active_tasks),
                "total_participants": sum(task["participants_count"] for task in active_tasks),
                "total_discoveries": sum(task["discoveries_count"] for task in active_tasks),
                "total_blockers": sum(task["active_blockers"] for task in active_tasks),
                "total_patterns": sum(task["patterns_count"] for task in active_tasks)
            },
            "context_analytics": context_analytics,
            "messaging_stats": broker_stats,
            "active_tasks": active_tasks,
            "timestamp": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Real-Time Collaboration System"""
    print("🤝 Archon v3.0 Real-Time Collaboration System")
    print("=" * 55)
    
    # Initialize collaboration system
    orchestrator = CollaborationOrchestrator()
    await orchestrator.start()
    
    try:
        # Create collaborative task
        task_id = "implement-user-authentication"
        project_id = "secure-webapp"
        agents = ["security-expert-001", "backend-dev-002", "frontend-dev-003"]
        
        print(f"\n🎯 Creating collaborative task: {task_id}")
        context = await orchestrator.create_collaborative_task(
            task_id, project_id, agents,
            {"complexity": "high", "domain": "security"}
        )
        
        # Subscribe agents to collaboration
        for agent_id in agents:
            interests = ["security", "authentication", "api"] if "security" in agent_id else ["frontend", "ui"] if "frontend" in agent_id else ["backend", "database"]
            await orchestrator.subscribe_agent_to_collaboration(agent_id, interests)
        
        print(f"✅ Task created with {len(context.participants)} participants")
        
        # Simulate collaborative workflow
        print(f"\n🔍 Simulating collaborative discoveries...")
        
        # Security expert makes discovery
        await orchestrator.broadcast_discovery(task_id, "security-expert-001", {
            "type": "security_requirement",
            "description": "JWT tokens should use RS256 for asymmetric signing",
            "impact": "high",
            "actionable": True,
            "reference": "OWASP JWT Security Guidelines"
        })
        
        # Backend dev reports blocker
        await orchestrator.broadcast_blocker(task_id, "backend-dev-002", {
            "type": "dependency_issue",
            "description": "JWT library missing from package.json",
            "severity": "blocking",
            "estimated_fix_time": "30 minutes"
        })
        
        # Frontend dev shares pattern
        await orchestrator.broadcast_pattern(task_id, "frontend-dev-003", {
            "type": "ui_pattern",
            "pattern": "Token refresh handling with React hooks",
            "confidence": 0.9,
            "code_example": "const useTokenRefresh = () => { ... }",
            "reusable": True
        })
        
        # Get collaboration dashboard
        print(f"\n📊 Collaboration Dashboard:")
        dashboard = await orchestrator.get_collaboration_dashboard(project_id)
        
        overview = dashboard["collaboration_overview"]
        print(f"  Active Tasks: {overview['active_tasks']}")
        print(f"  Total Participants: {overview['total_participants']}")
        print(f"  Total Discoveries: {overview['total_discoveries']}")
        print(f"  Total Blockers: {overview['total_blockers']}")
        print(f"  Total Patterns: {overview['total_patterns']}")
        
        messaging_stats = dashboard["messaging_stats"]
        print(f"\n📡 Messaging Statistics:")
        print(f"  Messages Published: {messaging_stats['messages_published']}")
        print(f"  Messages Delivered: {messaging_stats['messages_delivered']}")
        print(f"  Active Subscriptions: {messaging_stats['active_subscriptions']}")
        
        print(f"\n✅ Real-Time Collaboration System demo completed!")
        
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())