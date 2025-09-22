#!/usr/bin/env python3
"""
Kafka Integration Service for Archon
Connects existing services to Kafka for real-time messaging and streaming.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
import uuid
import os

from src.agents.messaging.distributed_messaging_system import (
    DistributedMessagingSystem,
    MessagePattern,
    MessagePriority,
    DistributedMessage,
    MessageMetadata
)

logger = logging.getLogger(__name__)

@dataclass
class KafkaTopics:
    """Standard Kafka topics used in Archon"""
    # Agent communication
    AGENT_COMMANDS = "archon.agents.commands"
    AGENT_RESPONSES = "archon.agents.responses"
    AGENT_STATUS = "archon.agents.status"
    AGENT_METRICS = "archon.agents.metrics"
    
    # System events
    SYSTEM_EVENTS = "archon.system.events"
    SYSTEM_ALERTS = "archon.system.alerts"
    SYSTEM_HEALTH = "archon.system.health"
    
    # Knowledge management
    KNOWLEDGE_UPDATES = "archon.knowledge.updates"
    KNOWLEDGE_QUERIES = "archon.knowledge.queries"
    KNOWLEDGE_EMBEDDINGS = "archon.knowledge.embeddings"
    
    # Project management
    PROJECT_EVENTS = "archon.projects.events"
    TASK_UPDATES = "archon.tasks.updates"
    
    # Real-time analytics
    ANALYTICS_EVENTS = "archon.analytics.events"
    PERFORMANCE_METRICS = "archon.performance.metrics"
    USER_ACTIVITIES = "archon.user.activities"
    
    # Streaming data
    LIVE_STREAMS = "archon.streams.live"
    DATA_INGESTION = "archon.data.ingestion"

@dataclass 
class EventData:
    """Standard event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class KafkaIntegrationService:
    """Service to integrate Archon components with Kafka messaging"""
    
    def __init__(self):
        self.messaging_system: Optional[DistributedMessagingSystem] = None
        self.is_initialized = False
        self.subscriptions: Dict[str, str] = {}  # topic -> subscription_id
        self.event_handlers: Dict[str, List[Callable]] = {}  # event_type -> handlers
        
        # Configuration
        # Parse Redis URL if available
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        redis_host = 'redis'
        redis_port = 6379

        if redis_url.startswith('redis://'):
            # Parse redis://host:port format
            redis_parts = redis_url.replace('redis://', '').split(':')
            redis_host = redis_parts[0]
            redis_port = int(redis_parts[1]) if len(redis_parts) > 1 else 6379

        self.kafka_config = {
            'default_backend': 'kafka',
            'kafka': {
                'host': os.getenv('KAFKA_HOST', 'kafka'),
                'port': int(os.getenv('KAFKA_PORT', 9092)),
                'connection_timeout': 30.0,
                'retry_delay': 1.0
            },
            'redis': {
                'host': redis_host,
                'port': redis_port,
                'database': 0
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize Kafka messaging system"""
        try:
            self.messaging_system = DistributedMessagingSystem(self.kafka_config)
            await self.messaging_system.start()
            
            # Subscribe to core topics
            await self._setup_core_subscriptions()
            
            self.is_initialized = True
            logger.info("Kafka integration service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka integration service: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Kafka messaging system"""
        try:
            if self.messaging_system:
                await self.messaging_system.stop()
            self.is_initialized = False
            logger.info("Kafka integration service shut down")
        except Exception as e:
            logger.error(f"Error shutting down Kafka integration service: {e}")
    
    async def _setup_core_subscriptions(self) -> None:
        """Setup subscriptions to core Kafka topics"""
        core_topics = [
            (KafkaTopics.AGENT_COMMANDS, self._handle_agent_command),
            (KafkaTopics.AGENT_RESPONSES, self._handle_agent_response),
            (KafkaTopics.SYSTEM_EVENTS, self._handle_system_event),
            (KafkaTopics.SYSTEM_ALERTS, self._handle_system_alert),
            (KafkaTopics.KNOWLEDGE_UPDATES, self._handle_knowledge_update),
            (KafkaTopics.ANALYTICS_EVENTS, self._handle_analytics_event),
        ]
        
        for topic, handler in core_topics:
            try:
                subscription_id = await self.messaging_system.subscribe(
                    topic, 
                    handler,
                    backend='kafka'
                )
                if subscription_id:
                    self.subscriptions[topic] = subscription_id
                    logger.info(f"Subscribed to Kafka topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {topic}: {e}")
    
    # Event Publishing Methods
    
    async def publish_agent_command(self, agent_id: str, command: str, params: Dict[str, Any] = None) -> bool:
        """Publish agent command"""
        event = EventData(
            event_type="agent_command",
            source="archon_server",
            data={
                "agent_id": agent_id,
                "command": command,
                "params": params or {}
            }
        )
        return await self._publish_event(KafkaTopics.AGENT_COMMANDS, event)
    
    async def publish_agent_status(self, agent_id: str, status: str, details: Dict[str, Any] = None) -> bool:
        """Publish agent status update"""
        event = EventData(
            event_type="agent_status",
            source=f"agent_{agent_id}",
            data={
                "agent_id": agent_id,
                "status": status,
                "details": details or {}
            }
        )
        return await self._publish_event(KafkaTopics.AGENT_STATUS, event)
    
    async def publish_system_event(self, event_type: str, data: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Publish system event"""
        event = EventData(
            event_type=event_type,
            source="archon_system",
            data=data
        )
        return await self._publish_event(KafkaTopics.SYSTEM_EVENTS, event, priority)
    
    async def publish_system_alert(self, alert_type: str, message: str, severity: str = "info", details: Dict[str, Any] = None) -> bool:
        """Publish system alert"""
        event = EventData(
            event_type="system_alert",
            source="archon_system",
            data={
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "details": details or {}
            }
        )
        priority = MessagePriority.CRITICAL if severity == "critical" else MessagePriority.HIGH if severity == "error" else MessagePriority.NORMAL
        return await self._publish_event(KafkaTopics.SYSTEM_ALERTS, event, priority)
    
    async def publish_knowledge_update(self, update_type: str, knowledge_id: str, data: Dict[str, Any]) -> bool:
        """Publish knowledge base update"""
        event = EventData(
            event_type="knowledge_update",
            source="knowledge_service",
            data={
                "update_type": update_type,
                "knowledge_id": knowledge_id,
                **data
            }
        )
        return await self._publish_event(KafkaTopics.KNOWLEDGE_UPDATES, event)
    
    async def publish_task_update(self, task_id: str, project_id: str, status: str, data: Dict[str, Any] = None) -> bool:
        """Publish task status update"""
        event = EventData(
            event_type="task_update",
            source="task_service",
            data={
                "task_id": task_id,
                "project_id": project_id,
                "status": status,
                **(data or {})
            }
        )
        return await self._publish_event(KafkaTopics.TASK_UPDATES, event)
    
    async def publish_analytics_event(self, metric_type: str, value: Any, tags: Dict[str, str] = None) -> bool:
        """Publish analytics event"""
        event = EventData(
            event_type="analytics_metric",
            source="analytics_service",
            data={
                "metric_type": metric_type,
                "value": value,
                "tags": tags or {}
            }
        )
        return await self._publish_event(KafkaTopics.ANALYTICS_EVENTS, event)
    
    async def publish_performance_metrics(self, service: str, metrics: Dict[str, Any]) -> bool:
        """Publish performance metrics"""
        event = EventData(
            event_type="performance_metrics",
            source=service,
            data=metrics
        )
        return await self._publish_event(KafkaTopics.PERFORMANCE_METRICS, event)
    
    async def publish_user_activity(self, user_id: str, activity_type: str, data: Dict[str, Any]) -> bool:
        """Publish user activity event"""
        event = EventData(
            event_type="user_activity",
            source="web_server",
            data={
                "user_id": user_id,
                "activity_type": activity_type,
                **data
            }
        )
        return await self._publish_event(KafkaTopics.USER_ACTIVITIES, event)
    
    # Stream Publishing Methods
    
    async def publish_live_stream_data(self, stream_id: str, data: Any, stream_type: str = "default") -> bool:
        """Publish live streaming data"""
        event = EventData(
            event_type="live_stream_data",
            source=f"stream_{stream_id}",
            data={
                "stream_id": stream_id,
                "stream_type": stream_type,
                "payload": data
            }
        )
        return await self._publish_event(KafkaTopics.LIVE_STREAMS, event)
    
    async def publish_data_ingestion_event(self, source: str, data_type: str, records_count: int, metadata: Dict[str, Any] = None) -> bool:
        """Publish data ingestion event"""
        event = EventData(
            event_type="data_ingestion",
            source=source,
            data={
                "data_type": data_type,
                "records_count": records_count,
                "metadata": metadata or {}
            }
        )
        return await self._publish_event(KafkaTopics.DATA_INGESTION, event)
    
    # Event Subscription Methods
    
    def register_event_handler(self, event_type: str, handler: Callable[[EventData], None]) -> None:
        """Register handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def subscribe_to_topic(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> str:
        """Subscribe to custom Kafka topic"""
        if not self.messaging_system:
            logger.error("Messaging system not initialized")
            return ""
        
        def wrapped_handler(payload):
            try:
                if isinstance(payload, dict):
                    handler(payload)
                else:
                    logger.warning(f"Received non-dict payload for topic {topic}")
            except Exception as e:
                logger.error(f"Handler error for topic {topic}: {e}")
        
        subscription_id = await self.messaging_system.subscribe(
            topic,
            wrapped_handler,
            backend='kafka'
        )
        
        if subscription_id:
            self.subscriptions[topic] = subscription_id
            logger.info(f"Subscribed to custom topic: {topic}")
        
        return subscription_id
    
    # Internal Methods
    
    async def _publish_event(self, topic: str, event: EventData, priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Internal method to publish event"""
        if not self.messaging_system:
            logger.error("Messaging system not initialized")
            return False
        
        try:
            payload = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "source": event.source,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "metadata": event.metadata
            }
            
            success = await self.messaging_system.publish(
                topic,
                payload,
                pattern=MessagePattern.PUBLISH_SUBSCRIBE,
                backend='kafka',
                priority=priority
            )
            
            if success:
                logger.debug(f"Published event {event.event_type} to topic {topic}")
            else:
                logger.error(f"Failed to publish event {event.event_type} to topic {topic}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error publishing event to {topic}: {e}")
            return False
    
    # Event Handler Methods
    
    async def _handle_agent_command(self, payload: Dict[str, Any]) -> None:
        """Handle agent command events"""
        try:
            event_data = self._parse_event_data(payload)
            logger.info(f"Received agent command: {event_data.data.get('command')} for agent {event_data.data.get('agent_id')}")
            
            # Dispatch to registered handlers
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling agent command: {e}")
    
    async def _handle_agent_response(self, payload: Dict[str, Any]) -> None:
        """Handle agent response events"""
        try:
            event_data = self._parse_event_data(payload)
            logger.debug(f"Received agent response from {event_data.source}")
            
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling agent response: {e}")
    
    async def _handle_system_event(self, payload: Dict[str, Any]) -> None:
        """Handle system events"""
        try:
            event_data = self._parse_event_data(payload)
            logger.info(f"System event: {event_data.event_type}")
            
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling system event: {e}")
    
    async def _handle_system_alert(self, payload: Dict[str, Any]) -> None:
        """Handle system alerts"""
        try:
            event_data = self._parse_event_data(payload)
            severity = event_data.data.get('severity', 'info')
            message = event_data.data.get('message', 'No message')
            
            if severity in ['critical', 'error']:
                logger.error(f"System alert [{severity.upper()}]: {message}")
            else:
                logger.info(f"System alert [{severity.upper()}]: {message}")
            
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling system alert: {e}")
    
    async def _handle_knowledge_update(self, payload: Dict[str, Any]) -> None:
        """Handle knowledge base updates"""
        try:
            event_data = self._parse_event_data(payload)
            update_type = event_data.data.get('update_type')
            knowledge_id = event_data.data.get('knowledge_id')
            
            logger.info(f"Knowledge update: {update_type} for {knowledge_id}")
            
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling knowledge update: {e}")
    
    async def _handle_analytics_event(self, payload: Dict[str, Any]) -> None:
        """Handle analytics events"""
        try:
            event_data = self._parse_event_data(payload)
            logger.debug(f"Analytics event: {event_data.event_type}")
            
            await self._dispatch_event(event_data)
            
        except Exception as e:
            logger.error(f"Error handling analytics event: {e}")
    
    def _parse_event_data(self, payload: Dict[str, Any]) -> EventData:
        """Parse payload into EventData"""
        return EventData(
            event_id=payload.get('event_id', ''),
            event_type=payload.get('event_type', ''),
            source=payload.get('source', ''),
            timestamp=datetime.fromisoformat(payload.get('timestamp', datetime.now().isoformat())),
            data=payload.get('data', {}),
            metadata=payload.get('metadata', {})
        )
    
    async def _dispatch_event(self, event_data: EventData) -> None:
        """Dispatch event to registered handlers"""
        handlers = self.event_handlers.get(event_data.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    # Status and Monitoring
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        if not self.messaging_system:
            return {"status": "not_initialized"}
        
        backend_status = self.messaging_system.get_backend_status()
        system_metrics = self.messaging_system.get_system_metrics()
        
        return {
            "status": "running" if self.is_initialized else "stopped",
            "subscriptions": len(self.subscriptions),
            "event_handlers": {event_type: len(handlers) for event_type, handlers in self.event_handlers.items()},
            "backend_status": backend_status,
            "metrics": {
                "messages_sent": system_metrics.messages_sent,
                "messages_received": system_metrics.messages_received,
                "messages_failed": system_metrics.messages_failed,
                "throughput_per_second": system_metrics.throughput_per_second
            }
        }

# Global instance
kafka_service: Optional[KafkaIntegrationService] = None

def get_kafka_service() -> KafkaIntegrationService:
    """Get global Kafka service instance"""
    global kafka_service
    if kafka_service is None:
        kafka_service = KafkaIntegrationService()
    return kafka_service

async def initialize_kafka_service() -> bool:
    """Initialize global Kafka service"""
    service = get_kafka_service()
    return await service.initialize()

async def shutdown_kafka_service() -> None:
    """Shutdown global Kafka service"""
    global kafka_service
    if kafka_service:
        await kafka_service.shutdown()
        kafka_service = None