#!/usr/bin/env python3
"""
Agent Communication Protocols Module

This module provides comprehensive communication infrastructure for autonomous AI agents.
It implements multiple communication protocols, message handling, and network coordination
to enable effective agent-to-agent communication and collaboration.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.0.0
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import hashlib
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommunicationProtocol(Enum):
    """Supported communication protocols"""
    DIRECT_MESSAGE = auto()          # Point-to-point messaging
    BROADCAST = auto()               # One-to-many messaging
    MULTICAST = auto()               # Group-based messaging
    PUBLISH_SUBSCRIBE = auto()       # Topic-based messaging
    REQUEST_RESPONSE = auto()        # Synchronous communication
    GOSSIP = auto()                  # Distributed information sharing
    FEDERATION = auto()              # Cross-system communication
    SECURE_CHANNEL = auto()          # Encrypted communication


class MessageType(Enum):
    """Types of messages in agent communication"""
    TASK_REQUEST = auto()
    TASK_RESPONSE = auto()
    STATUS_UPDATE = auto()
    RESOURCE_QUERY = auto()
    RESOURCE_RESPONSE = auto()
    COORDINATION_REQUEST = auto()
    COORDINATION_RESPONSE = auto()
    HEARTBEAT = auto()
    DISCOVERY = auto()
    NEGOTIATION = auto()
    CONSENSUS = auto()
    ERROR = auto()
    SYSTEM_NOTIFICATION = auto()


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class CommunicationSecurity(Enum):
    """Security levels for communication"""
    NONE = auto()
    BASIC_AUTH = auto()
    TOKEN_BASED = auto()
    ENCRYPTED = auto()
    SIGNED = auto()
    MUTUAL_TLS = auto()


@dataclass
class CommunicationMessage:
    """Represents a communication message between agents"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    protocol: CommunicationProtocol
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    requires_response: bool = False
    conversation_id: Optional[str] = None
    security_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    delivery_attempts: int = 0
    max_delivery_attempts: int = 3


@dataclass
class CommunicationChannel:
    """Represents a communication channel between agents"""
    channel_id: str
    protocol: CommunicationProtocol
    participants: Set[str]
    security_level: CommunicationSecurity
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    is_active: bool = True


@dataclass
class CommunicationMetrics:
    """Communication system performance metrics"""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    average_latency: float = 0.0
    active_channels: int = 0
    peak_throughput: int = 0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class BaseCommunicationHandler(ABC):
    """Abstract base class for communication handlers"""
    
    @abstractmethod
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send a message"""
        pass
    
    @abstractmethod
    async def receive_message(self, timeout: float = None) -> Optional[CommunicationMessage]:
        """Receive a message"""
        pass
    
    @abstractmethod
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        """Create a communication channel"""
        pass
    
    @abstractmethod
    async def close_channel(self, channel_id: str) -> bool:
        """Close a communication channel"""
        pass


class DirectMessageHandler(BaseCommunicationHandler):
    """Handler for direct point-to-point messaging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        
    async def send_message(self, message: CommunicationMessage) -> bool:
        try:
            if message.receiver_id not in self.message_queues:
                self.message_queues[message.receiver_id] = asyncio.Queue()
            
            await self.message_queues[message.receiver_id].put(message)
            return True
        except Exception as e:
            logger.error(f"Direct message send failed: {e}")
            return False
    
    async def receive_message(self, agent_id: str, timeout: float = None) -> Optional[CommunicationMessage]:
        try:
            if agent_id not in self.message_queues:
                self.message_queues[agent_id] = asyncio.Queue()
            
            if timeout:
                return await asyncio.wait_for(
                    self.message_queues[agent_id].get(), 
                    timeout=timeout
                )
            else:
                return await self.message_queues[agent_id].get()
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Direct message receive failed: {e}")
            return None
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        channel_id = f"direct_{uuid.uuid4().hex[:8]}"
        self.active_connections[channel_id] = channel_config
        return channel_id
    
    async def close_channel(self, channel_id: str) -> bool:
        return self.active_connections.pop(channel_id, None) is not None


class BroadcastHandler(BaseCommunicationHandler):
    """Handler for broadcast messaging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subscribers: Set[str] = set()
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
    async def send_message(self, message: CommunicationMessage) -> bool:
        try:
            # Send to all subscribers
            for subscriber in self.subscribers:
                subscriber_message = CommunicationMessage(
                    message_id=f"{message.message_id}_{subscriber}",
                    sender_id=message.sender_id,
                    receiver_id=subscriber,
                    message_type=message.message_type,
                    protocol=message.protocol,
                    content=message.content,
                    priority=message.priority,
                    timestamp=message.timestamp,
                    expiry=message.expiry,
                    requires_response=message.requires_response,
                    conversation_id=message.conversation_id,
                    security_context=message.security_context,
                    metadata=message.metadata
                )
                await self.broadcast_queue.put(subscriber_message)
            return True
        except Exception as e:
            logger.error(f"Broadcast message send failed: {e}")
            return False
    
    async def receive_message(self, timeout: float = None) -> Optional[CommunicationMessage]:
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.broadcast_queue.get(), 
                    timeout=timeout
                )
            else:
                return await self.broadcast_queue.get()
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Broadcast message receive failed: {e}")
            return None
    
    async def subscribe(self, agent_id: str) -> bool:
        self.subscribers.add(agent_id)
        return True
    
    async def unsubscribe(self, agent_id: str) -> bool:
        self.subscribers.discard(agent_id)
        return True
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        return f"broadcast_{uuid.uuid4().hex[:8]}"
    
    async def close_channel(self, channel_id: str) -> bool:
        return True


class PublishSubscribeHandler(BaseCommunicationHandler):
    """Handler for topic-based publish-subscribe messaging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.topics: Dict[str, Set[str]] = defaultdict(set)
        self.topic_queues: Dict[str, Dict[str, asyncio.Queue]] = defaultdict(dict)
        
    async def send_message(self, message: CommunicationMessage) -> bool:
        try:
            topic = message.metadata.get('topic', 'default')
            subscribers = self.topics.get(topic, set())
            
            for subscriber in subscribers:
                if subscriber not in self.topic_queues[topic]:
                    self.topic_queues[topic][subscriber] = asyncio.Queue()
                
                await self.topic_queues[topic][subscriber].put(message)
            return True
        except Exception as e:
            logger.error(f"Publish-subscribe message send failed: {e}")
            return False
    
    async def receive_message(self, agent_id: str, topic: str, timeout: float = None) -> Optional[CommunicationMessage]:
        try:
            if agent_id not in self.topic_queues[topic]:
                self.topic_queues[topic][agent_id] = asyncio.Queue()
            
            if timeout:
                return await asyncio.wait_for(
                    self.topic_queues[topic][agent_id].get(),
                    timeout=timeout
                )
            else:
                return await self.topic_queues[topic][agent_id].get()
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Publish-subscribe message receive failed: {e}")
            return None
    
    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        self.topics[topic].add(agent_id)
        return True
    
    async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        self.topics[topic].discard(agent_id)
        if not self.topics[topic]:
            del self.topics[topic]
        return True
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        return f"pubsub_{uuid.uuid4().hex[:8]}"
    
    async def close_channel(self, channel_id: str) -> bool:
        return True


class RequestResponseHandler(BaseCommunicationHandler):
    """Handler for synchronous request-response communication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_handlers: Dict[str, Callable] = {}
        
    async def send_message(self, message: CommunicationMessage) -> bool:
        try:
            if message.requires_response:
                # Store future for response
                future = asyncio.Future()
                self.pending_requests[message.message_id] = future
                
                # Send message (implementation would depend on transport)
                # For now, simulate sending
                await asyncio.sleep(0.001)  # Simulate network delay
                
                return True
            else:
                # Regular message sending
                return True
        except Exception as e:
            logger.error(f"Request-response message send failed: {e}")
            return False
    
    async def send_request(self, message: CommunicationMessage, timeout: float = 30.0) -> Optional[CommunicationMessage]:
        """Send a request and wait for response"""
        try:
            message.requires_response = True
            future = asyncio.Future()
            self.pending_requests[message.message_id] = future
            
            await self.send_message(message)
            
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.pending_requests.pop(message.message_id, None)
            return None
        except Exception as e:
            logger.error(f"Request sending failed: {e}")
            self.pending_requests.pop(message.message_id, None)
            return None
    
    async def send_response(self, original_message: CommunicationMessage, response_content: Dict[str, Any]) -> bool:
        """Send a response to a request"""
        try:
            response_message = CommunicationMessage(
                message_id=f"resp_{uuid.uuid4().hex[:8]}",
                sender_id=original_message.receiver_id,
                receiver_id=original_message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                protocol=original_message.protocol,
                content=response_content,
                conversation_id=original_message.message_id
            )
            
            # Complete the pending request
            if original_message.message_id in self.pending_requests:
                self.pending_requests[original_message.message_id].set_result(response_message)
                del self.pending_requests[original_message.message_id]
            
            return True
        except Exception as e:
            logger.error(f"Response sending failed: {e}")
            return False
    
    async def receive_message(self, timeout: float = None) -> Optional[CommunicationMessage]:
        # Implementation would depend on transport layer
        return None
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        return f"reqresp_{uuid.uuid4().hex[:8]}"
    
    async def close_channel(self, channel_id: str) -> bool:
        return True


class GossipProtocolHandler(BaseCommunicationHandler):
    """Handler for gossip-based distributed communication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.known_agents: Set[str] = set()
        self.gossip_data: Dict[str, Dict[str, Any]] = {}
        self.gossip_interval = config.get('gossip_interval', 5.0)
        self.max_gossip_age = config.get('max_gossip_age', 60.0)
        
    async def send_message(self, message: CommunicationMessage) -> bool:
        try:
            # Add to gossip data
            self.gossip_data[message.message_id] = {
                'message': message,
                'timestamp': time.time(),
                'hops': 0
            }
            return True
        except Exception as e:
            logger.error(f"Gossip message send failed: {e}")
            return False
    
    async def receive_message(self, timeout: float = None) -> Optional[CommunicationMessage]:
        # Implementation would depend on gossip network
        return None
    
    async def spread_gossip(self) -> None:
        """Spread gossip to known agents"""
        try:
            current_time = time.time()
            
            # Clean old gossip data
            expired_keys = [
                key for key, data in self.gossip_data.items()
                if current_time - data['timestamp'] > self.max_gossip_age
            ]
            
            for key in expired_keys:
                del self.gossip_data[key]
            
            # Spread current gossip (implementation would depend on network)
            logger.info(f"Spreading {len(self.gossip_data)} gossip items")
            
        except Exception as e:
            logger.error(f"Gossip spreading failed: {e}")
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        return f"gossip_{uuid.uuid4().hex[:8]}"
    
    async def close_channel(self, channel_id: str) -> bool:
        return True


class AgentCommunicationProtocols:
    """Main communication protocols management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = f"comm_sys_{uuid.uuid4().hex[:8]}"
        
        # Initialize handlers
        self.handlers: Dict[CommunicationProtocol, BaseCommunicationHandler] = {}
        self._initialize_handlers()
        
        # Communication management
        self.active_channels: Dict[str, CommunicationChannel] = {}
        self.message_history: List[CommunicationMessage] = []
        self.metrics = CommunicationMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
    def _initialize_handlers(self) -> None:
        """Initialize communication protocol handlers"""
        try:
            self.handlers[CommunicationProtocol.DIRECT_MESSAGE] = DirectMessageHandler(
                self.config.get('direct_message', {})
            )
            
            self.handlers[CommunicationProtocol.BROADCAST] = BroadcastHandler(
                self.config.get('broadcast', {})
            )
            
            self.handlers[CommunicationProtocol.PUBLISH_SUBSCRIBE] = PublishSubscribeHandler(
                self.config.get('publish_subscribe', {})
            )
            
            self.handlers[CommunicationProtocol.REQUEST_RESPONSE] = RequestResponseHandler(
                self.config.get('request_response', {})
            )
            
            self.handlers[CommunicationProtocol.GOSSIP] = GossipProtocolHandler(
                self.config.get('gossip', {})
            )
            
            logger.info(f"Initialized {len(self.handlers)} communication handlers")
            
        except Exception as e:
            logger.error(f"Handler initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the communication system"""
        try:
            self.is_running = True
            
            # Start background tasks
            self.background_tasks.add(
                asyncio.create_task(self._metrics_collector())
            )
            
            self.background_tasks.add(
                asyncio.create_task(self._message_cleanup())
            )
            
            logger.info(f"Communication system {self.system_id} started")
            
        except Exception as e:
            logger.error(f"Communication system start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the communication system"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            # Close all channels
            for channel_id in list(self.active_channels.keys()):
                await self.close_channel(channel_id)
            
            logger.info(f"Communication system {self.system_id} stopped")
            
        except Exception as e:
            logger.error(f"Communication system stop failed: {e}")
    
    async def create_channel(self, protocol: CommunicationProtocol, participants: List[str], 
                           security_level: CommunicationSecurity = CommunicationSecurity.NONE,
                           config: Optional[Dict[str, Any]] = None) -> str:
        """Create a communication channel"""
        try:
            channel_id = f"chan_{uuid.uuid4().hex[:8]}"
            channel_config = config or {}
            
            # Create channel with handler
            if protocol in self.handlers:
                handler_channel_id = await self.handlers[protocol].create_channel(channel_config)
                
                channel = CommunicationChannel(
                    channel_id=channel_id,
                    protocol=protocol,
                    participants=set(participants),
                    security_level=security_level,
                    configuration=channel_config
                )
                
                self.active_channels[channel_id] = channel
                self.metrics.active_channels += 1
                
                logger.info(f"Created channel {channel_id} with protocol {protocol.name}")
                return channel_id
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
                
        except Exception as e:
            logger.error(f"Channel creation failed: {e}")
            raise
    
    async def close_channel(self, channel_id: str) -> bool:
        """Close a communication channel"""
        try:
            if channel_id not in self.active_channels:
                return False
            
            channel = self.active_channels[channel_id]
            handler = self.handlers.get(channel.protocol)
            
            if handler:
                success = await handler.close_channel(channel_id)
                if success:
                    del self.active_channels[channel_id]
                    self.metrics.active_channels -= 1
                    logger.info(f"Closed channel {channel_id}")
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Channel closure failed: {e}")
            return False
    
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send a message through the appropriate protocol"""
        try:
            handler = self.handlers.get(message.protocol)
            if not handler:
                logger.error(f"No handler for protocol {message.protocol}")
                return False
            
            # Update metrics
            self.metrics.total_messages_sent += 1
            message.delivery_attempts += 1
            
            # Send message
            start_time = time.time()
            success = await handler.send_message(message)
            latency = time.time() - start_time
            
            # Update metrics
            if success:
                self.metrics.successful_deliveries += 1
                self.metrics.average_latency = (
                    (self.metrics.average_latency * (self.metrics.successful_deliveries - 1) + latency) /
                    self.metrics.successful_deliveries
                )
            else:
                self.metrics.failed_deliveries += 1
            
            # Store in history
            self.message_history.append(message)
            
            return success
            
        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            self.metrics.failed_deliveries += 1
            return False
    
    async def receive_message(self, agent_id: str, protocol: CommunicationProtocol,
                            timeout: float = None, **kwargs) -> Optional[CommunicationMessage]:
        """Receive a message from the specified protocol"""
        try:
            handler = self.handlers.get(protocol)
            if not handler:
                logger.error(f"No handler for protocol {protocol}")
                return None
            
            # Protocol-specific receive logic
            if protocol == CommunicationProtocol.DIRECT_MESSAGE:
                message = await handler.receive_message(agent_id, timeout)
            elif protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                topic = kwargs.get('topic', 'default')
                message = await handler.receive_message(agent_id, topic, timeout)
            else:
                message = await handler.receive_message(timeout)
            
            if message:
                self.metrics.total_messages_received += 1
            
            return message
            
        except Exception as e:
            logger.error(f"Message receiving failed: {e}")
            return None
    
    async def send_request(self, message: CommunicationMessage, timeout: float = 30.0) -> Optional[CommunicationMessage]:
        """Send a request and wait for response"""
        try:
            if message.protocol != CommunicationProtocol.REQUEST_RESPONSE:
                logger.error("Request-response requires REQUEST_RESPONSE protocol")
                return None
            
            handler = self.handlers[CommunicationProtocol.REQUEST_RESPONSE]
            if hasattr(handler, 'send_request'):
                return await handler.send_request(message, timeout)
            
            return None
            
        except Exception as e:
            logger.error(f"Request sending failed: {e}")
            return None
    
    async def broadcast_message(self, message: CommunicationMessage) -> bool:
        """Broadcast a message to all subscribers"""
        try:
            message.protocol = CommunicationProtocol.BROADCAST
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Message broadcasting failed: {e}")
            return False
    
    async def publish_to_topic(self, topic: str, message: CommunicationMessage) -> bool:
        """Publish a message to a specific topic"""
        try:
            message.protocol = CommunicationProtocol.PUBLISH_SUBSCRIBE
            message.metadata['topic'] = topic
            return await self.send_message(message)
            
        except Exception as e:
            logger.error(f"Topic publishing failed: {e}")
            return False
    
    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe an agent to a topic"""
        try:
            handler = self.handlers.get(CommunicationProtocol.PUBLISH_SUBSCRIBE)
            if handler and hasattr(handler, 'subscribe_to_topic'):
                return await handler.subscribe_to_topic(agent_id, topic)
            return False
            
        except Exception as e:
            logger.error(f"Topic subscription failed: {e}")
            return False
    
    async def subscribe_to_broadcast(self, agent_id: str) -> bool:
        """Subscribe an agent to broadcasts"""
        try:
            handler = self.handlers.get(CommunicationProtocol.BROADCAST)
            if handler and hasattr(handler, 'subscribe'):
                return await handler.subscribe(agent_id)
            return False
            
        except Exception as e:
            logger.error(f"Broadcast subscription failed: {e}")
            return False
    
    def get_metrics(self) -> CommunicationMetrics:
        """Get current communication metrics"""
        self.metrics.last_updated = datetime.now()
        self.metrics.error_rate = (
            self.metrics.failed_deliveries / 
            max(self.metrics.total_messages_sent, 1)
        ) * 100
        return self.metrics
    
    def get_active_channels(self) -> List[CommunicationChannel]:
        """Get list of active communication channels"""
        return list(self.active_channels.values())
    
    def get_message_history(self, limit: Optional[int] = None) -> List[CommunicationMessage]:
        """Get communication message history"""
        if limit:
            return self.message_history[-limit:]
        return self.message_history.copy()
    
    async def _metrics_collector(self) -> None:
        """Background task for collecting metrics"""
        while self.is_running:
            try:
                # Update peak throughput
                current_throughput = len([
                    msg for msg in self.message_history
                    if (datetime.now() - msg.timestamp).seconds < 60
                ])
                
                self.metrics.peak_throughput = max(
                    self.metrics.peak_throughput,
                    current_throughput
                )
                
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(10)
    
    async def _message_cleanup(self) -> None:
        """Background task for cleaning up old messages"""
        while self.is_running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                # Remove old messages
                self.message_history = [
                    msg for msg in self.message_history
                    if msg.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Message cleanup failed: {e}")
                await asyncio.sleep(300)


async def example_agent_communication_usage():
    """Comprehensive example of agent communication protocols usage"""
    
    print("\n🔗 Agent Communication Protocols System Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'direct_message': {
            'max_queue_size': 1000,
            'timeout_seconds': 30
        },
        'broadcast': {
            'max_subscribers': 100
        },
        'publish_subscribe': {
            'max_topics': 50,
            'topic_ttl': 3600
        },
        'request_response': {
            'default_timeout': 30.0,
            'max_pending_requests': 100
        },
        'gossip': {
            'gossip_interval': 5.0,
            'max_gossip_age': 60.0,
            'fanout': 3
        }
    }
    
    # Initialize communication system
    comm_system = AgentCommunicationProtocols(config)
    await comm_system.start()
    
    print(f"✅ Communication system {comm_system.system_id} started")
    
    try:
        # Example 1: Direct Message Communication
        print("\n1. Direct Message Communication")
        print("-" * 40)
        
        # Create direct message channel
        channel_id = await comm_system.create_channel(
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            participants=["agent_1", "agent_2"],
            security_level=CommunicationSecurity.BASIC_AUTH
        )
        
        print(f"✅ Created direct message channel: {channel_id}")
        
        # Send direct message
        message = CommunicationMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.TASK_REQUEST,
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            content={
                "task": "process_data",
                "data": {"items": [1, 2, 3, 4, 5]},
                "deadline": "2025-01-09T18:00:00Z"
            },
            priority=MessagePriority.HIGH,
            requires_response=True
        )
        
        success = await comm_system.send_message(message)
        print(f"✅ Direct message sent: {success}")
        
        # Receive message
        received = await comm_system.receive_message(
            agent_id="agent_2",
            protocol=CommunicationProtocol.DIRECT_MESSAGE,
            timeout=1.0
        )
        
        if received:
            print(f"✅ Message received by agent_2: {received.content['task']}")
        
        # Example 2: Broadcast Communication
        print("\n2. Broadcast Communication")
        print("-" * 40)
        
        # Subscribe agents to broadcasts
        await comm_system.subscribe_to_broadcast("agent_1")
        await comm_system.subscribe_to_broadcast("agent_2")
        await comm_system.subscribe_to_broadcast("agent_3")
        
        # Broadcast message
        broadcast_msg = CommunicationMessage(
            message_id=f"broadcast_{uuid.uuid4().hex[:8]}",
            sender_id="system",
            receiver_id=None,  # Broadcast
            message_type=MessageType.SYSTEM_NOTIFICATION,
            protocol=CommunicationProtocol.BROADCAST,
            content={
                "announcement": "System maintenance scheduled",
                "time": "2025-01-09T20:00:00Z",
                "duration": "30 minutes"
            },
            priority=MessagePriority.HIGH
        )
        
        success = await comm_system.broadcast_message(broadcast_msg)
        print(f"✅ Broadcast message sent: {success}")
        
        # Example 3: Publish-Subscribe Communication
        print("\n3. Publish-Subscribe Communication")
        print("-" * 40)
        
        # Subscribe to topics
        await comm_system.subscribe_to_topic("agent_1", "task_updates")
        await comm_system.subscribe_to_topic("agent_2", "task_updates")
        await comm_system.subscribe_to_topic("agent_3", "resource_alerts")
        
        # Publish to topic
        topic_msg = CommunicationMessage(
            message_id=f"topic_{uuid.uuid4().hex[:8]}",
            sender_id="task_manager",
            receiver_id=None,
            message_type=MessageType.STATUS_UPDATE,
            protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
            content={
                "task_id": "task_123",
                "status": "completed",
                "result": {"processed_items": 100},
                "completion_time": "2025-01-09T15:30:00Z"
            }
        )
        
        success = await comm_system.publish_to_topic("task_updates", topic_msg)
        print(f"✅ Message published to topic: {success}")
        
        # Receive from topic
        received_topic = await comm_system.receive_message(
            agent_id="agent_1",
            protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
            topic="task_updates",
            timeout=1.0
        )
        
        if received_topic:
            print(f"✅ Topic message received: {received_topic.content['status']}")
        
        # Example 4: Request-Response Communication
        print("\n4. Request-Response Communication")
        print("-" * 40)
        
        # Send request with response expectation
        request_msg = CommunicationMessage(
            message_id=f"req_{uuid.uuid4().hex[:8]}",
            sender_id="agent_1",
            receiver_id="agent_2",
            message_type=MessageType.RESOURCE_QUERY,
            protocol=CommunicationProtocol.REQUEST_RESPONSE,
            content={
                "query": "get_available_resources",
                "resource_type": "compute",
                "min_capacity": 1000
            },
            requires_response=True
        )
        
        # This would normally wait for an actual response
        # For demo, we'll simulate the request
        print("✅ Request-response pattern configured")
        
        # Example 5: System Metrics and Status
        print("\n5. Communication Metrics")
        print("-" * 40)
        
        metrics = comm_system.get_metrics()
        print(f"✅ Total messages sent: {metrics.total_messages_sent}")
        print(f"✅ Total messages received: {metrics.total_messages_received}")
        print(f"✅ Successful deliveries: {metrics.successful_deliveries}")
        print(f"✅ Failed deliveries: {metrics.failed_deliveries}")
        print(f"✅ Active channels: {metrics.active_channels}")
        print(f"✅ Average latency: {metrics.average_latency:.4f}s")
        print(f"✅ Error rate: {metrics.error_rate:.2f}%")
        
        # Example 6: Channel Management
        print("\n6. Channel Management")
        print("-" * 40)
        
        active_channels = comm_system.get_active_channels()
        print(f"✅ Active channels: {len(active_channels)}")
        
        for channel in active_channels:
            print(f"   - {channel.channel_id}: {channel.protocol.name}")
            print(f"     Participants: {len(channel.participants)}")
            print(f"     Security: {channel.security_level.name}")
            print(f"     Messages: {channel.message_count}")
        
        # Example 7: Message History
        print("\n7. Message History")
        print("-" * 40)
        
        history = comm_system.get_message_history(limit=5)
        print(f"✅ Recent messages: {len(history)}")
        
        for msg in history:
            print(f"   - {msg.message_id}: {msg.message_type.name}")
            print(f"     From: {msg.sender_id} To: {msg.receiver_id}")
            print(f"     Protocol: {msg.protocol.name}")
            print(f"     Priority: {msg.priority.name}")
        
        # Allow background tasks to run briefly
        await asyncio.sleep(2)
        
    finally:
        # Cleanup
        await comm_system.stop()
        print(f"\n✅ Communication system stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_agent_communication_usage())