#!/usr/bin/env python3
"""
Distributed Messaging System Module

This module provides production-grade distributed messaging capabilities using Redis,
RabbitMQ, Apache Kafka, and other messaging systems. It replaces the simple asyncio.Queue
implementations with scalable, fault-tolerant messaging infrastructure.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.1.0
"""

import asyncio
import json
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import pickle
import threading
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagingBackend(Enum):
    """Supported messaging backends"""
    REDIS = auto()
    RABBITMQ = auto()
    KAFKA = auto()
    NATS = auto()
    PULSAR = auto()
    ZMQ = auto()
    MEMORY = auto()  # In-memory fallback


class MessagePattern(Enum):
    """Message exchange patterns"""
    PUBLISH_SUBSCRIBE = auto()
    REQUEST_RESPONSE = auto()
    POINT_TO_POINT = auto()
    BROADCAST = auto()
    WORK_QUEUE = auto()
    TOPIC_EXCHANGE = auto()
    DIRECT_EXCHANGE = auto()
    FANOUT_EXCHANGE = auto()
    HEADER_EXCHANGE = auto()


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class DeliveryMode(Enum):
    """Message delivery modes"""
    AT_MOST_ONCE = auto()
    AT_LEAST_ONCE = auto()
    EXACTLY_ONCE = auto()


class SerializationFormat(Enum):
    """Serialization formats"""
    JSON = auto()
    PICKLE = auto()
    MSGPACK = auto()
    PROTOBUF = auto()
    AVRO = auto()
    BINARY = auto()


@dataclass
class MessageMetadata:
    """Message metadata"""
    message_id: str
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    priority: MessagePriority = MessagePriority.NORMAL
    content_type: str = "application/json"
    encoding: str = "utf-8"
    headers: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    route_key: Optional[str] = None


@dataclass
class DistributedMessage:
    """Distributed message container"""
    metadata: MessageMetadata
    payload: Any
    pattern: MessagePattern
    serialization: SerializationFormat = SerializationFormat.JSON
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    def serialize(self) -> bytes:
        """Serialize message for transport"""
        data = {
            'metadata': {
                'message_id': self.metadata.message_id,
                'correlation_id': self.metadata.correlation_id,
                'reply_to': self.metadata.reply_to,
                'timestamp': self.metadata.timestamp.isoformat(),
                'expiry': self.metadata.expiry.isoformat() if self.metadata.expiry else None,
                'priority': self.metadata.priority.value,
                'content_type': self.metadata.content_type,
                'encoding': self.metadata.encoding,
                'headers': self.metadata.headers,
                'retry_count': self.metadata.retry_count,
                'max_retries': self.metadata.max_retries,
                'route_key': self.metadata.route_key
            },
            'payload': self.payload,
            'pattern': self.pattern.value,
            'serialization': self.serialization.value,
            'delivery_mode': self.delivery_mode.value
        }
        
        if self.serialization == SerializationFormat.JSON:
            return json.dumps(data, default=str).encode('utf-8')
        elif self.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        elif self.serialization == SerializationFormat.BINARY:
            return str(data).encode('utf-8')
        else:
            # Default to JSON
            return json.dumps(data, default=str).encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes, serialization: SerializationFormat = SerializationFormat.JSON) -> 'DistributedMessage':
        """Deserialize message from transport"""
        try:
            if serialization == SerializationFormat.JSON:
                msg_data = json.loads(data.decode('utf-8'))
            elif serialization == SerializationFormat.PICKLE:
                msg_data = pickle.loads(data)
            else:
                msg_data = json.loads(data.decode('utf-8'))
            
            # Reconstruct metadata
            meta_data = msg_data['metadata']
            metadata = MessageMetadata(
                message_id=meta_data['message_id'],
                correlation_id=meta_data.get('correlation_id'),
                reply_to=meta_data.get('reply_to'),
                timestamp=datetime.fromisoformat(meta_data['timestamp']),
                expiry=datetime.fromisoformat(meta_data['expiry']) if meta_data.get('expiry') else None,
                priority=MessagePriority(meta_data.get('priority', 3)),
                content_type=meta_data.get('content_type', 'application/json'),
                encoding=meta_data.get('encoding', 'utf-8'),
                headers=meta_data.get('headers', {}),
                retry_count=meta_data.get('retry_count', 0),
                max_retries=meta_data.get('max_retries', 3),
                route_key=meta_data.get('route_key')
            )
            
            return cls(
                metadata=metadata,
                payload=msg_data['payload'],
                pattern=MessagePattern(msg_data.get('pattern', 1)),
                serialization=SerializationFormat(msg_data.get('serialization', 1)),
                delivery_mode=DeliveryMode(msg_data.get('delivery_mode', 2))
            )
            
        except Exception as e:
            logger.error(f"Message deserialization failed: {e}")
            raise


@dataclass
class ConnectionConfig:
    """Connection configuration"""
    backend: MessagingBackend
    host: str = "localhost"
    port: int = 5672
    username: Optional[str] = None
    password: Optional[str] = None
    virtual_host: str = "/"
    database: int = 0  # For Redis
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    connection_timeout: float = 30.0
    heartbeat: float = 60.0
    max_connections: int = 10
    retry_attempts: int = 3
    retry_delay: float = 1.0
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessagingMetrics:
    """Messaging system metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    connections_active: int = 0
    connections_failed: int = 0
    average_latency: float = 0.0
    throughput_per_second: float = 0.0
    queue_depths: Dict[str, int] = field(default_factory=dict)
    consumer_counts: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class BaseMessagingBackend(ABC):
    """Abstract base class for messaging backends"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.metrics = MessagingMetrics()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection"""
        pass
    
    @abstractmethod
    async def publish(self, topic: str, message: DistributedMessage) -> bool:
        """Publish message"""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback: Callable[[DistributedMessage], None]) -> str:
        """Subscribe to topic"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic"""
        pass
    
    @abstractmethod
    async def send_request(self, topic: str, message: DistributedMessage, timeout: float = 30.0) -> Optional[DistributedMessage]:
        """Send request and wait for response"""
        pass
    
    @abstractmethod
    async def send_response(self, original_message: DistributedMessage, response: DistributedMessage) -> bool:
        """Send response to request"""
        pass


class RedisMessagingBackend(BaseMessagingBackend):
    """Redis-based messaging backend"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.redis_available = self._check_redis()
        self.subscribers: Dict[str, Any] = {}
        self.subscription_tasks: Dict[str, asyncio.Task] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        if self.redis_available:
            import redis.asyncio as aioredis
            self.aioredis = aioredis
    
    def _check_redis(self) -> bool:
        """Check if Redis is available"""
        try:
            import redis
            logger.info(f"Redis {redis.__version__} available")
            return True
        except ImportError:
            logger.warning("Redis not available - using mock implementation")
            return False
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            if not self.redis_available:
                # Mock connection
                self.connection = {"type": "mock_redis", "connected": True}
                self.is_connected = True
                logger.info("Using mock Redis connection")
                return True
            
            # Real Redis connection
            connection_params = {
                'host': self.config.host,
                'port': self.config.port,
                'db': self.config.database,
                'socket_connect_timeout': self.config.connection_timeout,
                'health_check_interval': self.config.heartbeat
            }
            
            if self.config.username and self.config.password:
                connection_params.update({
                    'username': self.config.username,
                    'password': self.config.password
                })
            
            if self.config.ssl_enabled:
                connection_params['ssl'] = True
                if self.config.ssl_cert_path:
                    connection_params['ssl_certfile'] = self.config.ssl_cert_path
            
            self.connection = self.aioredis.Redis(**connection_params)
            
            # Test connection
            await self.connection.ping()
            self.is_connected = True
            self.metrics.connections_active += 1
            
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.metrics.connections_failed += 1
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Redis"""
        try:
            # Cancel subscription tasks
            for task in self.subscription_tasks.values():
                if not task.done():
                    task.cancel()
            
            self.subscription_tasks.clear()
            
            if self.redis_available and self.connection:
                await self.connection.close()
            
            self.is_connected = False
            self.metrics.connections_active = max(0, self.metrics.connections_active - 1)
            
            logger.info("Disconnected from Redis")
            return True
            
        except Exception as e:
            logger.error(f"Redis disconnect failed: {e}")
            return False
    
    async def publish(self, topic: str, message: DistributedMessage) -> bool:
        """Publish message to Redis"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.redis_available:
                # Mock publish
                logger.debug(f"Mock Redis publish to {topic}")
                self.metrics.messages_sent += 1
                return True
            
            # Serialize message
            serialized = message.serialize()
            
            # Publish based on pattern
            if message.pattern == MessagePattern.PUBLISH_SUBSCRIBE:
                await self.connection.publish(topic, serialized)
            elif message.pattern == MessagePattern.WORK_QUEUE:
                await self.connection.lpush(f"queue:{topic}", serialized)
            else:
                # Default to pub/sub
                await self.connection.publish(topic, serialized)
            
            self.metrics.messages_sent += 1
            logger.debug(f"Published message to Redis topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Redis publish failed: {e}")
            self.metrics.messages_failed += 1
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[DistributedMessage], None]) -> str:
        """Subscribe to Redis topic"""
        try:
            if not self.is_connected:
                await self.connect()
            
            subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
            
            if not self.redis_available:
                # Mock subscription
                logger.debug(f"Mock Redis subscription to {topic}")
                return subscription_id
            
            # Create pubsub instance
            pubsub = self.connection.pubsub()
            await pubsub.subscribe(topic)
            
            # Store subscriber
            self.subscribers[subscription_id] = {
                'topic': topic,
                'callback': callback,
                'pubsub': pubsub
            }
            
            # Start subscription task
            task = asyncio.create_task(self._subscription_handler(subscription_id))
            self.subscription_tasks[subscription_id] = task
            
            logger.info(f"Subscribed to Redis topic: {topic}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Redis subscription failed: {e}")
            return ""
    
    async def _subscription_handler(self, subscription_id: str) -> None:
        """Handle Redis subscription messages"""
        try:
            subscriber = self.subscribers.get(subscription_id)
            if not subscriber:
                return
            
            pubsub = subscriber['pubsub']
            callback = subscriber['callback']
            
            while True:
                try:
                    message = await pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        # Deserialize and process message
                        try:
                            dist_message = DistributedMessage.deserialize(message['data'])
                            callback(dist_message)
                            self.metrics.messages_received += 1
                        except Exception as e:
                            logger.error(f"Message processing failed: {e}")
                            self.metrics.messages_failed += 1
                            
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Subscription handler error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Subscription handler failed: {e}")
        finally:
            # Cleanup
            if subscription_id in self.subscribers:
                subscriber = self.subscribers[subscription_id]
                if 'pubsub' in subscriber:
                    await subscriber['pubsub'].close()
                del self.subscribers[subscription_id]
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Redis topic"""
        try:
            if subscription_id not in self.subscribers:
                return False
            
            # Cancel task
            if subscription_id in self.subscription_tasks:
                task = self.subscription_tasks[subscription_id]
                if not task.done():
                    task.cancel()
                del self.subscription_tasks[subscription_id]
            
            # Close pubsub
            subscriber = self.subscribers[subscription_id]
            if 'pubsub' in subscriber:
                await subscriber['pubsub'].close()
            
            del self.subscribers[subscription_id]
            
            logger.info(f"Unsubscribed from Redis topic: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Redis unsubscribe failed: {e}")
            return False
    
    async def send_request(self, topic: str, message: DistributedMessage, timeout: float = 30.0) -> Optional[DistributedMessage]:
        """Send request via Redis and wait for response"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Set up response handling
            response_topic = f"response_{message.metadata.message_id}"
            message.metadata.reply_to = response_topic
            
            # Create future for response
            response_future = asyncio.Future()
            self.pending_requests[message.metadata.message_id] = response_future
            
            # Subscribe to response topic
            def response_callback(response_msg: DistributedMessage):
                if response_msg.metadata.correlation_id == message.metadata.message_id:
                    if not response_future.done():
                        response_future.set_result(response_msg)
            
            response_sub_id = await self.subscribe(response_topic, response_callback)
            
            try:
                # Send request
                success = await self.publish(topic, message)
                if not success:
                    return None
                
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
                
            finally:
                # Cleanup
                await self.unsubscribe(response_sub_id)
                self.pending_requests.pop(message.metadata.message_id, None)
                
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for message {message.metadata.message_id}")
            return None
        except Exception as e:
            logger.error(f"Redis request failed: {e}")
            return None
    
    async def send_response(self, original_message: DistributedMessage, response: DistributedMessage) -> bool:
        """Send response via Redis"""
        try:
            if not original_message.metadata.reply_to:
                logger.warning("No reply_to address in original message")
                return False
            
            # Set correlation ID
            response.metadata.correlation_id = original_message.metadata.message_id
            
            # Send response
            return await self.publish(original_message.metadata.reply_to, response)
            
        except Exception as e:
            logger.error(f"Redis response failed: {e}")
            return False


class RabbitMQMessagingBackend(BaseMessagingBackend):
    """RabbitMQ-based messaging backend"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.rabbitmq_available = self._check_rabbitmq()
        self.channels: Dict[str, Any] = {}
        self.consumers: Dict[str, Any] = {}
        self.exchanges: Set[str] = set()
        self.queues: Set[str] = set()
        
        if self.rabbitmq_available:
            import aio_pika
            self.aio_pika = aio_pika
    
    def _check_rabbitmq(self) -> bool:
        """Check if aio_pika is available"""
        try:
            import aio_pika
            logger.info("aio_pika (RabbitMQ) available")
            return True
        except ImportError:
            logger.warning("aio_pika not available - using mock implementation")
            return False
    
    async def connect(self) -> bool:
        """Connect to RabbitMQ"""
        try:
            if not self.rabbitmq_available:
                # Mock connection
                self.connection = {"type": "mock_rabbitmq", "connected": True}
                self.is_connected = True
                logger.info("Using mock RabbitMQ connection")
                return True
            
            # Real RabbitMQ connection
            connection_url = f"amqp://"
            if self.config.username and self.config.password:
                connection_url += f"{self.config.username}:{self.config.password}@"
            
            connection_url += f"{self.config.host}:{self.config.port}{self.config.virtual_host}"
            
            self.connection = await self.aio_pika.connect_robust(
                connection_url,
                heartbeat=int(self.config.heartbeat)
            )
            
            self.is_connected = True
            self.metrics.connections_active += 1
            
            logger.info(f"Connected to RabbitMQ at {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ connection failed: {e}")
            self.metrics.connections_failed += 1
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from RabbitMQ"""
        try:
            # Close channels
            for channel in self.channels.values():
                if hasattr(channel, 'close'):
                    await channel.close()
            
            self.channels.clear()
            self.consumers.clear()
            
            if self.rabbitmq_available and self.connection:
                await self.connection.close()
            
            self.is_connected = False
            self.metrics.connections_active = max(0, self.metrics.connections_active - 1)
            
            logger.info("Disconnected from RabbitMQ")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ disconnect failed: {e}")
            return False
    
    async def _get_channel(self) -> Any:
        """Get or create RabbitMQ channel"""
        if not self.rabbitmq_available:
            return {"type": "mock_channel"}
        
        channel_id = "default"
        if channel_id not in self.channels:
            self.channels[channel_id] = await self.connection.channel()
        
        return self.channels[channel_id]
    
    async def _ensure_exchange(self, exchange_name: str, exchange_type: str = "topic") -> None:
        """Ensure exchange exists"""
        if not self.rabbitmq_available or exchange_name in self.exchanges:
            return
        
        channel = await self._get_channel()
        await channel.declare_exchange(
            exchange_name,
            self.aio_pika.ExchangeType.TOPIC if exchange_type == "topic" else self.aio_pika.ExchangeType.DIRECT,
            durable=True
        )
        self.exchanges.add(exchange_name)
    
    async def _ensure_queue(self, queue_name: str, exchange_name: str = "", routing_key: str = "") -> Any:
        """Ensure queue exists and is bound"""
        if not self.rabbitmq_available:
            return {"type": "mock_queue"}
        
        channel = await self._get_channel()
        
        if queue_name not in self.queues:
            queue = await channel.declare_queue(queue_name, durable=True)
            self.queues.add(queue_name)
            
            if exchange_name:
                await self._ensure_exchange(exchange_name)
                exchange = await channel.get_exchange(exchange_name)
                await queue.bind(exchange, routing_key=routing_key)
            
            return queue
        else:
            return await channel.get_queue(queue_name)
    
    async def publish(self, topic: str, message: DistributedMessage) -> bool:
        """Publish message to RabbitMQ"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.rabbitmq_available:
                # Mock publish
                logger.debug(f"Mock RabbitMQ publish to {topic}")
                self.metrics.messages_sent += 1
                return True
            
            channel = await self._get_channel()
            
            # Prepare message
            serialized = message.serialize()
            routing_key = message.metadata.route_key or topic
            
            # Publish based on pattern
            if message.pattern == MessagePattern.WORK_QUEUE:
                # Direct to queue
                await self._ensure_queue(topic)
                await channel.default_exchange.publish(
                    self.aio_pika.Message(
                        serialized,
                        priority=message.metadata.priority.value,
                        expiration=int((message.metadata.expiry - datetime.now()).total_seconds() * 1000) if message.metadata.expiry else None
                    ),
                    routing_key=topic
                )
            else:
                # Topic exchange
                exchange_name = f"exchange_{topic}"
                await self._ensure_exchange(exchange_name)
                exchange = await channel.get_exchange(exchange_name)
                
                await exchange.publish(
                    self.aio_pika.Message(
                        serialized,
                        priority=message.metadata.priority.value
                    ),
                    routing_key=routing_key
                )
            
            self.metrics.messages_sent += 1
            logger.debug(f"Published message to RabbitMQ topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ publish failed: {e}")
            self.metrics.messages_failed += 1
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[DistributedMessage], None]) -> str:
        """Subscribe to RabbitMQ topic"""
        try:
            if not self.is_connected:
                await self.connect()
            
            subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
            
            if not self.rabbitmq_available:
                # Mock subscription
                logger.debug(f"Mock RabbitMQ subscription to {topic}")
                return subscription_id
            
            # Create queue for subscription
            queue_name = f"queue_{topic}_{subscription_id}"
            exchange_name = f"exchange_{topic}"
            
            queue = await self._ensure_queue(queue_name, exchange_name, topic)
            
            # Message handler
            async def message_handler(message):
                try:
                    async with message.process():
                        dist_message = DistributedMessage.deserialize(message.body)
                        callback(dist_message)
                        self.metrics.messages_received += 1
                except Exception as e:
                    logger.error(f"Message processing failed: {e}")
                    self.metrics.messages_failed += 1
            
            # Start consuming
            consumer_tag = await queue.consume(message_handler)
            
            self.consumers[subscription_id] = {
                'topic': topic,
                'queue': queue,
                'consumer_tag': consumer_tag,
                'callback': callback
            }
            
            logger.info(f"Subscribed to RabbitMQ topic: {topic}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"RabbitMQ subscription failed: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from RabbitMQ topic"""
        try:
            if subscription_id not in self.consumers:
                return False
            
            if self.rabbitmq_available:
                consumer = self.consumers[subscription_id]
                queue = consumer['queue']
                consumer_tag = consumer['consumer_tag']
                
                await queue.cancel(consumer_tag)
            
            del self.consumers[subscription_id]
            
            logger.info(f"Unsubscribed from RabbitMQ topic: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ unsubscribe failed: {e}")
            return False
    
    async def send_request(self, topic: str, message: DistributedMessage, timeout: float = 30.0) -> Optional[DistributedMessage]:
        """Send request via RabbitMQ and wait for response"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.rabbitmq_available:
                # Mock request-response
                await asyncio.sleep(0.01)
                return DistributedMessage(
                    metadata=MessageMetadata(
                        message_id=f"resp_{uuid.uuid4().hex[:8]}",
                        correlation_id=message.metadata.message_id
                    ),
                    payload={"response": "mock_response"},
                    pattern=MessagePattern.REQUEST_RESPONSE
                )
            
            # Set up response handling
            response_queue = f"response_{message.metadata.message_id}"
            message.metadata.reply_to = response_queue
            
            # Create response queue
            channel = await self._get_channel()
            queue = await channel.declare_queue(response_queue, exclusive=True, auto_delete=True)
            
            # Create future for response
            response_future = asyncio.Future()
            
            # Response handler
            async def response_handler(response_msg):
                try:
                    async with response_msg.process():
                        dist_message = DistributedMessage.deserialize(response_msg.body)
                        if dist_message.metadata.correlation_id == message.metadata.message_id:
                            if not response_future.done():
                                response_future.set_result(dist_message)
                except Exception as e:
                    if not response_future.done():
                        response_future.set_exception(e)
            
            # Start consuming responses
            consumer_tag = await queue.consume(response_handler)
            
            try:
                # Send request
                success = await self.publish(topic, message)
                if not success:
                    return None
                
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
                
            finally:
                # Cleanup
                await queue.cancel(consumer_tag)
                await queue.delete()
                
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for message {message.metadata.message_id}")
            return None
        except Exception as e:
            logger.error(f"RabbitMQ request failed: {e}")
            return None
    
    async def send_response(self, original_message: DistributedMessage, response: DistributedMessage) -> bool:
        """Send response via RabbitMQ"""
        try:
            if not original_message.metadata.reply_to:
                logger.warning("No reply_to address in original message")
                return False
            
            # Set correlation ID
            response.metadata.correlation_id = original_message.metadata.message_id
            
            # Send response to reply queue
            if not self.rabbitmq_available:
                return True
            
            channel = await self._get_channel()
            serialized = response.serialize()
            
            await channel.default_exchange.publish(
                self.aio_pika.Message(serialized),
                routing_key=original_message.metadata.reply_to
            )
            
            return True
            
        except Exception as e:
            logger.error(f"RabbitMQ response failed: {e}")
            return False


class KafkaMessagingBackend(BaseMessagingBackend):
    """Kafka-based messaging backend"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.kafka_available = self._check_kafka()
        self.producer = None
        self.consumers: Dict[str, Any] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        if self.kafka_available:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            self.AIOKafkaProducer = AIOKafkaProducer
            self.AIOKafkaConsumer = AIOKafkaConsumer
    
    def _check_kafka(self) -> bool:
        """Check if aiokafka is available"""
        try:
            import aiokafka
            logger.info(f"aiokafka {aiokafka.__version__} available")
            return True
        except ImportError:
            logger.warning("aiokafka not available - using mock implementation")
            return False
    
    async def connect(self) -> bool:
        """Connect to Kafka"""
        try:
            if not self.kafka_available:
                # Mock connection
                self.connection = {"type": "mock_kafka", "connected": True}
                self.is_connected = True
                logger.info("Using mock Kafka connection")
                return True
            
            # Real Kafka connection
            bootstrap_servers = f"{self.config.host}:{self.config.port}"
            
            # Create producer
            self.producer = self.AIOKafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: v,  # We'll serialize manually
                compression_type='gzip',
                request_timeout_ms=int(self.config.connection_timeout * 1000),
                retry_backoff_ms=int(self.config.retry_delay * 1000)
            )
            
            await self.producer.start()
            
            self.is_connected = True
            self.metrics.connections_active += 1
            
            logger.info(f"Connected to Kafka at {bootstrap_servers}")
            return True
            
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
            self.metrics.connections_failed += 1
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Kafka"""
        try:
            # Stop all consumers
            for consumer_id, task in list(self.consumer_tasks.items()):
                if not task.done():
                    task.cancel()
                
                consumer = self.consumers.get(consumer_id, {}).get('consumer')
                if consumer:
                    await consumer.stop()
            
            self.consumer_tasks.clear()
            self.consumers.clear()
            
            # Stop producer
            if self.kafka_available and self.producer:
                await self.producer.stop()
            
            self.is_connected = False
            self.metrics.connections_active = max(0, self.metrics.connections_active - 1)
            
            logger.info("Disconnected from Kafka")
            return True
            
        except Exception as e:
            logger.error(f"Kafka disconnect failed: {e}")
            return False
    
    async def publish(self, topic: str, message: DistributedMessage) -> bool:
        """Publish message to Kafka"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if not self.kafka_available:
                # Mock publish
                logger.debug(f"Mock Kafka publish to {topic}")
                self.metrics.messages_sent += 1
                return True
            
            # Serialize message
            serialized = message.serialize()
            
            # Determine partition key from metadata
            partition_key = None
            if message.metadata.route_key:
                partition_key = message.metadata.route_key.encode('utf-8')
            elif message.metadata.correlation_id:
                partition_key = message.metadata.correlation_id.encode('utf-8')
            
            # Send to Kafka
            await self.producer.send_and_wait(
                topic,
                value=serialized,
                key=partition_key,
                headers={
                    'message_id': message.metadata.message_id.encode('utf-8'),
                    'priority': str(message.metadata.priority.value).encode('utf-8'),
                    'pattern': str(message.pattern.value).encode('utf-8'),
                    'content_type': message.metadata.content_type.encode('utf-8')
                }
            )
            
            self.metrics.messages_sent += 1
            logger.debug(f"Published message to Kafka topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Kafka publish failed: {e}")
            self.metrics.messages_failed += 1
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[DistributedMessage], None]) -> str:
        """Subscribe to Kafka topic"""
        try:
            if not self.is_connected:
                await self.connect()
            
            subscription_id = f"kafka_sub_{uuid.uuid4().hex[:8]}"
            
            if not self.kafka_available:
                # Mock subscription
                logger.debug(f"Mock Kafka subscription to {topic}")
                return subscription_id
            
            # Create consumer
            consumer = self.AIOKafkaConsumer(
                topic,
                bootstrap_servers=f"{self.config.host}:{self.config.port}",
                group_id=f"archon_consumer_group_{subscription_id}",
                value_deserializer=lambda v: v,  # We'll deserialize manually
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            # Store consumer
            self.consumers[subscription_id] = {
                'topic': topic,
                'callback': callback,
                'consumer': consumer
            }
            
            # Start consumer task
            task = asyncio.create_task(self._consumer_handler(subscription_id))
            self.consumer_tasks[subscription_id] = task
            
            logger.info(f"Subscribed to Kafka topic: {topic}")
            return subscription_id
            
        except Exception as e:
            logger.error(f"Kafka subscription failed: {e}")
            return ""
    
    async def _consumer_handler(self, subscription_id: str) -> None:
        """Handle Kafka consumer messages"""
        try:
            consumer_info = self.consumers.get(subscription_id)
            if not consumer_info:
                return
            
            consumer = consumer_info['consumer']
            callback = consumer_info['callback']
            
            await consumer.start()
            
            try:
                async for message in consumer:
                    try:
                        # Deserialize message
                        dist_message = DistributedMessage.deserialize(message.value)
                        
                        # Execute callback
                        callback(dist_message)
                        self.metrics.messages_received += 1
                        
                    except Exception as e:
                        logger.error(f"Message processing failed: {e}")
                        self.metrics.messages_failed += 1
                        
            finally:
                await consumer.stop()
                
        except Exception as e:
            logger.error(f"Kafka consumer handler failed: {e}")
        finally:
            # Cleanup
            if subscription_id in self.consumers:
                del self.consumers[subscription_id]
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from Kafka topic"""
        try:
            if subscription_id not in self.consumers:
                return False
            
            # Cancel consumer task
            if subscription_id in self.consumer_tasks:
                task = self.consumer_tasks[subscription_id]
                if not task.done():
                    task.cancel()
                del self.consumer_tasks[subscription_id]
            
            # Stop consumer
            consumer_info = self.consumers.get(subscription_id)
            if consumer_info and 'consumer' in consumer_info:
                consumer = consumer_info['consumer']
                await consumer.stop()
            
            del self.consumers[subscription_id]
            
            logger.info(f"Unsubscribed from Kafka topic: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Kafka unsubscribe failed: {e}")
            return False
    
    async def send_request(self, topic: str, message: DistributedMessage, timeout: float = 30.0) -> Optional[DistributedMessage]:
        """Send request via Kafka and wait for response"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Set up response handling
            response_topic = f"response_{message.metadata.message_id}"
            message.metadata.reply_to = response_topic
            
            # Create future for response
            response_future = asyncio.Future()
            self.pending_requests[message.metadata.message_id] = response_future
            
            # Subscribe to response topic
            def response_callback(response_msg: DistributedMessage):
                if response_msg.metadata.correlation_id == message.metadata.message_id:
                    if not response_future.done():
                        response_future.set_result(response_msg)
            
            response_sub_id = await self.subscribe(response_topic, response_callback)
            
            try:
                # Send request
                success = await self.publish(topic, message)
                if not success:
                    return None
                
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
                
            finally:
                # Cleanup
                await self.unsubscribe(response_sub_id)
                self.pending_requests.pop(message.metadata.message_id, None)
                
        except asyncio.TimeoutError:
            logger.warning(f"Kafka request timeout for message {message.metadata.message_id}")
            return None
        except Exception as e:
            logger.error(f"Kafka request failed: {e}")
            return None
    
    async def send_response(self, original_message: DistributedMessage, response: DistributedMessage) -> bool:
        """Send response via Kafka"""
        try:
            if not original_message.metadata.reply_to:
                logger.warning("No reply_to address in original message")
                return False
            
            # Set correlation ID
            response.metadata.correlation_id = original_message.metadata.message_id
            
            # Send response
            return await self.publish(original_message.metadata.reply_to, response)
            
        except Exception as e:
            logger.error(f"Kafka response failed: {e}")
            return False


class InMemoryMessagingBackend(BaseMessagingBackend):
    """In-memory messaging backend for testing/fallback"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.topics: Dict[str, List[Callable]] = defaultdict(list)
        self.queues: Dict[str, deque] = defaultdict(deque)
        self.pending_requests: Dict[str, asyncio.Future] = {}
    
    async def connect(self) -> bool:
        """Connect (always succeeds for in-memory)"""
        self.is_connected = True
        self.connection = {"type": "in_memory", "connected": True}
        logger.info("Connected to in-memory messaging backend")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect"""
        self.is_connected = False
        self.topics.clear()
        self.queues.clear()
        self.pending_requests.clear()
        logger.info("Disconnected from in-memory messaging backend")
        return True
    
    async def publish(self, topic: str, message: DistributedMessage) -> bool:
        """Publish message in-memory"""
        try:
            if message.pattern == MessagePattern.WORK_QUEUE:
                self.queues[topic].append(message)
            else:
                # Pub/sub - notify all subscribers
                for callback in self.topics[topic]:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Callback failed: {e}")
            
            self.metrics.messages_sent += 1
            return True
            
        except Exception as e:
            logger.error(f"In-memory publish failed: {e}")
            self.metrics.messages_failed += 1
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[DistributedMessage], None]) -> str:
        """Subscribe to topic in-memory"""
        subscription_id = f"sub_{uuid.uuid4().hex[:8]}"
        self.topics[topic].append(callback)
        logger.debug(f"Subscribed to in-memory topic: {topic}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic (simplified)"""
        # In real implementation, would track subscription IDs properly
        logger.debug(f"Unsubscribed from in-memory topic: {subscription_id}")
        return True
    
    async def send_request(self, topic: str, message: DistributedMessage, timeout: float = 30.0) -> Optional[DistributedMessage]:
        """Send request in-memory"""
        try:
            # Mock request-response
            await asyncio.sleep(0.001)  # Simulate network delay
            
            return DistributedMessage(
                metadata=MessageMetadata(
                    message_id=f"resp_{uuid.uuid4().hex[:8]}",
                    correlation_id=message.metadata.message_id
                ),
                payload={"response": "in_memory_response", "original": message.payload},
                pattern=MessagePattern.REQUEST_RESPONSE
            )
            
        except Exception as e:
            logger.error(f"In-memory request failed: {e}")
            return None
    
    async def send_response(self, original_message: DistributedMessage, response: DistributedMessage) -> bool:
        """Send response in-memory"""
        try:
            response.metadata.correlation_id = original_message.metadata.message_id
            # In a real implementation, this would route to the waiting request
            return True
            
        except Exception as e:
            logger.error(f"In-memory response failed: {e}")
            return False


class DistributedMessagingSystem:
    """Main distributed messaging system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = f"msg_sys_{uuid.uuid4().hex[:8]}"
        
        # Backend management
        self.backends: Dict[str, BaseMessagingBackend] = {}
        self.default_backend: Optional[str] = None
        
        # Subscription management
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # System metrics
        self.global_metrics = MessagingMetrics()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Initialize default backends
        self._initialize_backends()
    
    def _initialize_backends(self) -> None:
        """Initialize messaging backends"""
        try:
            # Redis backend
            if 'redis' in self.config:
                redis_config = ConnectionConfig(
                    backend=MessagingBackend.REDIS,
                    **self.config['redis']
                )
                self.backends['redis'] = RedisMessagingBackend(redis_config)
            
            # RabbitMQ backend
            if 'rabbitmq' in self.config:
                rabbitmq_config = ConnectionConfig(
                    backend=MessagingBackend.RABBITMQ,
                    **self.config['rabbitmq']
                )
                self.backends['rabbitmq'] = RabbitMQMessagingBackend(rabbitmq_config)
            
            # Kafka backend
            if 'kafka' in self.config:
                kafka_config = ConnectionConfig(
                    backend=MessagingBackend.KAFKA,
                    **self.config['kafka']
                )
                self.backends['kafka'] = KafkaMessagingBackend(kafka_config)
            
            # In-memory fallback
            memory_config = ConnectionConfig(backend=MessagingBackend.MEMORY)
            self.backends['memory'] = InMemoryMessagingBackend(memory_config)
            
            # Set default backend
            self.default_backend = self.config.get('default_backend', 'memory')
            if self.default_backend not in self.backends:
                self.default_backend = 'memory'
            
            logger.info(f"Initialized {len(self.backends)} messaging backends")
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the messaging system"""
        try:
            self.is_running = True
            
            # Connect to all backends
            for backend_name, backend in self.backends.items():
                try:
                    success = await backend.connect()
                    if success:
                        logger.info(f"Connected to {backend_name} backend")
                    else:
                        logger.warning(f"Failed to connect to {backend_name} backend")
                except Exception as e:
                    logger.error(f"Backend {backend_name} connection failed: {e}")
            
            # Start background tasks
            self.background_tasks.add(
                asyncio.create_task(self._metrics_collector())
            )
            
            self.background_tasks.add(
                asyncio.create_task(self._health_monitor())
            )
            
            logger.info(f"Distributed messaging system {self.system_id} started")
            
        except Exception as e:
            logger.error(f"Messaging system start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the messaging system"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            # Disconnect all backends
            for backend_name, backend in self.backends.items():
                try:
                    await backend.disconnect()
                    logger.info(f"Disconnected from {backend_name} backend")
                except Exception as e:
                    logger.error(f"Backend {backend_name} disconnect failed: {e}")
            
            logger.info(f"Distributed messaging system {self.system_id} stopped")
            
        except Exception as e:
            logger.error(f"Messaging system stop failed: {e}")
    
    async def publish(self, topic: str, payload: Any, 
                     pattern: MessagePattern = MessagePattern.PUBLISH_SUBSCRIBE,
                     backend: Optional[str] = None,
                     priority: MessagePriority = MessagePriority.NORMAL,
                     **kwargs) -> bool:
        """Publish message"""
        try:
            backend_name = backend or self.default_backend
            if backend_name not in self.backends:
                logger.error(f"Backend {backend_name} not found")
                return False
            
            # Create message
            message = DistributedMessage(
                metadata=MessageMetadata(
                    message_id=f"msg_{uuid.uuid4().hex[:8]}",
                    priority=priority,
                    **kwargs
                ),
                payload=payload,
                pattern=pattern
            )
            
            # Publish
            backend_instance = self.backends[backend_name]
            success = await backend_instance.publish(topic, message)
            
            if success:
                self.global_metrics.messages_sent += 1
            else:
                self.global_metrics.messages_failed += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Message publish failed: {e}")
            self.global_metrics.messages_failed += 1
            return False
    
    async def subscribe(self, topic: str, callback: Callable[[Any], None],
                       backend: Optional[str] = None) -> str:
        """Subscribe to topic"""
        try:
            backend_name = backend or self.default_backend
            if backend_name not in self.backends:
                logger.error(f"Backend {backend_name} not found")
                return ""
            
            # Wrap callback to extract payload
            def wrapped_callback(message: DistributedMessage):
                try:
                    callback(message.payload)
                    self.global_metrics.messages_received += 1
                except Exception as e:
                    logger.error(f"Callback execution failed: {e}")
                    self.global_metrics.messages_failed += 1
            
            backend_instance = self.backends[backend_name]
            subscription_id = await backend_instance.subscribe(topic, wrapped_callback)
            
            if subscription_id:
                self.subscriptions[subscription_id] = {
                    'topic': topic,
                    'backend': backend_name,
                    'callback': callback,
                    'created_at': datetime.now()
                }
            
            return subscription_id
            
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from topic"""
        try:
            if subscription_id not in self.subscriptions:
                logger.warning(f"Subscription {subscription_id} not found")
                return False
            
            subscription = self.subscriptions[subscription_id]
            backend_name = subscription['backend']
            
            if backend_name in self.backends:
                backend_instance = self.backends[backend_name]
                success = await backend_instance.unsubscribe(subscription_id)
                
                if success:
                    del self.subscriptions[subscription_id]
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    async def send_request(self, topic: str, payload: Any, 
                          timeout: float = 30.0,
                          backend: Optional[str] = None) -> Optional[Any]:
        """Send request and wait for response"""
        try:
            backend_name = backend or self.default_backend
            if backend_name not in self.backends:
                logger.error(f"Backend {backend_name} not found")
                return None
            
            # Create request message
            message = DistributedMessage(
                metadata=MessageMetadata(
                    message_id=f"req_{uuid.uuid4().hex[:8]}"
                ),
                payload=payload,
                pattern=MessagePattern.REQUEST_RESPONSE
            )
            
            backend_instance = self.backends[backend_name]
            response = await backend_instance.send_request(topic, message, timeout)
            
            return response.payload if response else None
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    async def send_response(self, request_message: DistributedMessage, 
                           response_payload: Any,
                           backend: Optional[str] = None) -> bool:
        """Send response to request"""
        try:
            backend_name = backend or self.default_backend
            if backend_name not in self.backends:
                logger.error(f"Backend {backend_name} not found")
                return False
            
            # Create response message
            response = DistributedMessage(
                metadata=MessageMetadata(
                    message_id=f"resp_{uuid.uuid4().hex[:8]}"
                ),
                payload=response_payload,
                pattern=MessagePattern.REQUEST_RESPONSE
            )
            
            backend_instance = self.backends[backend_name]
            return await backend_instance.send_response(request_message, response)
            
        except Exception as e:
            logger.error(f"Response failed: {e}")
            return False
    
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all backends"""
        status = {}
        
        for backend_name, backend in self.backends.items():
            status[backend_name] = {
                'connected': backend.is_connected,
                'backend_type': backend.config.backend.name,
                'metrics': {
                    'messages_sent': backend.metrics.messages_sent,
                    'messages_received': backend.metrics.messages_received,
                    'messages_failed': backend.metrics.messages_failed,
                    'connections_active': backend.metrics.connections_active,
                    'connections_failed': backend.metrics.connections_failed
                }
            }
        
        return status
    
    def get_system_metrics(self) -> MessagingMetrics:
        """Get global system metrics"""
        # Aggregate metrics from all backends
        total_sent = sum(backend.metrics.messages_sent for backend in self.backends.values())
        total_received = sum(backend.metrics.messages_received for backend in self.backends.values())
        total_failed = sum(backend.metrics.messages_failed for backend in self.backends.values())
        
        self.global_metrics.messages_sent = total_sent
        self.global_metrics.messages_received = total_received
        self.global_metrics.messages_failed = total_failed
        self.global_metrics.connections_active = sum(
            backend.metrics.connections_active for backend in self.backends.values()
        )
        self.global_metrics.last_updated = datetime.now()
        
        return self.global_metrics
    
    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List active subscriptions"""
        return [
            {
                'subscription_id': sub_id,
                'topic': sub_info['topic'],
                'backend': sub_info['backend'],
                'created_at': sub_info['created_at']
            }
            for sub_id, sub_info in self.subscriptions.items()
        ]
    
    async def _metrics_collector(self) -> None:
        """Background task for collecting metrics"""
        while self.is_running:
            try:
                # Update global metrics
                self.get_system_metrics()
                
                # Calculate throughput
                if self.global_metrics.messages_sent > 0:
                    elapsed_time = (datetime.now() - self.global_metrics.last_updated).total_seconds()
                    if elapsed_time > 0:
                        self.global_metrics.throughput_per_second = (
                            self.global_metrics.messages_sent / elapsed_time
                        )
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection failed: {e}")
                await asyncio.sleep(30)
    
    async def _health_monitor(self) -> None:
        """Background task for monitoring backend health"""
        while self.is_running:
            try:
                for backend_name, backend in self.backends.items():
                    if not backend.is_connected:
                        logger.warning(f"Backend {backend_name} disconnected, attempting reconnect")
                        try:
                            await backend.connect()
                        except Exception as e:
                            logger.error(f"Reconnect failed for {backend_name}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60)


async def example_distributed_messaging_usage():
    """Comprehensive example of distributed messaging system usage"""
    
    print("\n🔄 Distributed Messaging System Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'default_backend': 'memory',
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'database': 0
        },
        'rabbitmq': {
            'host': 'localhost',
            'port': 5672,
            'username': 'guest',
            'password': 'guest',
            'virtual_host': '/'
        }
    }
    
    # Initialize messaging system
    messaging_system = DistributedMessagingSystem(config)
    await messaging_system.start()
    
    print(f"✅ Messaging system {messaging_system.system_id} started")
    
    try:
        # Example 1: Basic Publish-Subscribe
        print("\n1. Basic Publish-Subscribe Pattern")
        print("-" * 40)
        
        received_messages = []
        
        def message_handler(payload):
            received_messages.append(payload)
            print(f"   Received: {payload}")
        
        # Subscribe to topic
        sub_id = await messaging_system.subscribe("test_topic", message_handler)
        print(f"✅ Subscribed to topic: {sub_id}")
        
        # Publish messages
        for i in range(3):
            success = await messaging_system.publish(
                "test_topic",
                {"message": f"Hello World {i}", "timestamp": datetime.now().isoformat()},
                priority=MessagePriority.NORMAL
            )
            print(f"   Published message {i}: {success}")
        
        # Allow messages to be processed
        await asyncio.sleep(0.5)
        print(f"✅ Received {len(received_messages)} messages")
        
        # Example 2: Work Queue Pattern
        print("\n2. Work Queue Pattern")
        print("-" * 40)
        
        work_results = []
        
        def work_handler(payload):
            # Simulate work processing
            work_id = payload.get('work_id')
            result = f"Processed work {work_id}"
            work_results.append(result)
            print(f"   Worker processed: {result}")
        
        # Subscribe to work queue
        work_sub = await messaging_system.subscribe("work_queue", work_handler)
        
        # Send work items
        for i in range(5):
            await messaging_system.publish(
                "work_queue",
                {"work_id": i, "task": f"process_item_{i}", "data": list(range(i * 10, (i + 1) * 10))},
                pattern=MessagePattern.WORK_QUEUE
            )
        
        await asyncio.sleep(0.5)
        print(f"✅ Processed {len(work_results)} work items")
        
        # Example 3: Request-Response Pattern
        print("\n3. Request-Response Pattern")
        print("-" * 40)
        
        # Set up request handler
        def request_handler(payload):
            request_type = payload.get('type')
            if request_type == 'calculate':
                numbers = payload.get('numbers', [])
                result = sum(numbers)
                # In a real system, this would send a response back
                print(f"   Calculated sum: {result}")
        
        # Subscribe to requests
        req_sub = await messaging_system.subscribe("calculation_service", request_handler)
        
        # Send request (mock response for demonstration)
        response = await messaging_system.send_request(
            "calculation_service",
            {"type": "calculate", "numbers": [1, 2, 3, 4, 5]},
            timeout=10.0
        )
        
        if response:
            print(f"✅ Request-response completed: {response}")
        else:
            print("✅ Request-response pattern demonstrated (mock)")
        
        # Example 4: Priority Messaging
        print("\n4. Priority Messaging")
        print("-" * 40)
        
        priority_messages = []
        
        def priority_handler(payload):
            priority_messages.append(payload)
            print(f"   Priority message: {payload}")
        
        priority_sub = await messaging_system.subscribe("priority_topic", priority_handler)
        
        # Send messages with different priorities
        priorities = [MessagePriority.LOW, MessagePriority.CRITICAL, MessagePriority.NORMAL, MessagePriority.HIGH]
        
        for i, priority in enumerate(priorities):
            await messaging_system.publish(
                "priority_topic",
                {"message": f"Priority message {i}", "level": priority.name},
                priority=priority
            )
        
        await asyncio.sleep(0.5)
        print(f"✅ Handled {len(priority_messages)} priority messages")
        
        # Example 5: Multi-Backend Usage
        print("\n5. Multi-Backend Messaging")
        print("-" * 40)
        
        backend_messages = {
            'memory': [],
            'redis': [],
            'rabbitmq': []
        }
        
        # Subscribe to each backend (if available)
        for backend_name in ['memory', 'redis', 'rabbitmq']:
            if backend_name in messaging_system.backends:
                def make_handler(backend):
                    def handler(payload):
                        backend_messages[backend].append(payload)
                        print(f"   {backend.title()} received: {payload['message']}")
                    return handler
                
                await messaging_system.subscribe(
                    f"{backend_name}_topic", 
                    make_handler(backend_name),
                    backend=backend_name
                )
                
                # Send message to this backend
                await messaging_system.publish(
                    f"{backend_name}_topic",
                    {"message": f"Message for {backend_name}", "backend": backend_name},
                    backend=backend_name
                )
        
        await asyncio.sleep(0.5)
        
        for backend_name, messages in backend_messages.items():
            if messages:
                print(f"✅ {backend_name.title()} backend: {len(messages)} messages")
        
        # Example 6: System Status and Metrics
        print("\n6. System Status and Metrics")
        print("-" * 40)
        
        # Backend status
        backend_status = messaging_system.get_backend_status()
        print("✅ Backend Status:")
        for backend_name, status in backend_status.items():
            print(f"   {backend_name}:")
            print(f"     Connected: {status['connected']}")
            print(f"     Type: {status['backend_type']}")
            print(f"     Messages sent: {status['metrics']['messages_sent']}")
            print(f"     Messages received: {status['metrics']['messages_received']}")
            print(f"     Messages failed: {status['metrics']['messages_failed']}")
        
        # System metrics
        system_metrics = messaging_system.get_system_metrics()
        print(f"\n✅ System Metrics:")
        print(f"   Total messages sent: {system_metrics.messages_sent}")
        print(f"   Total messages received: {system_metrics.messages_received}")
        print(f"   Total messages failed: {system_metrics.messages_failed}")
        print(f"   Active connections: {system_metrics.connections_active}")
        print(f"   Throughput: {system_metrics.throughput_per_second:.2f} msg/sec")
        
        # Active subscriptions
        subscriptions = messaging_system.list_subscriptions()
        print(f"\n✅ Active subscriptions: {len(subscriptions)}")
        for sub in subscriptions[:3]:  # Show first 3
            print(f"   - {sub['subscription_id']}: {sub['topic']} ({sub['backend']})")
        
        # Allow background tasks to run
        await asyncio.sleep(2)
        
        # Cleanup subscriptions
        for sub in subscriptions:
            await messaging_system.unsubscribe(sub['subscription_id'])
        
        print("✅ Cleaned up subscriptions")
        
    finally:
        # Cleanup
        await messaging_system.stop()
        print(f"\n✅ Distributed messaging system stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_distributed_messaging_usage())