"""
Advanced Distributed Message Queue
High-performance message queuing with guaranteed delivery, routing, and scaling
"""

import asyncio
import logging
import time
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import heapq
import threading

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


class DeliveryMode(Enum):
    """Message delivery modes"""
    AT_MOST_ONCE = "at_most_once"     # Fire and forget
    AT_LEAST_ONCE = "at_least_once"   # Guaranteed delivery
    EXACTLY_ONCE = "exactly_once"     # No duplicates


class ExchangeType(Enum):
    """Message exchange types"""
    DIRECT = "direct"       # Direct routing by key
    FANOUT = "fanout"      # Broadcast to all queues
    TOPIC = "topic"        # Topic-based routing with wildcards
    HEADERS = "headers"    # Route by message headers


@dataclass
class Message:
    """Message in the queue system"""
    message_id: str
    content: Any
    routing_key: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    # Timing information
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    scheduled_at: Optional[datetime] = None  # For delayed messages
    processed_at: Optional[datetime] = None
    
    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: timedelta = timedelta(seconds=30)
    
    # Tracing
    trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Metadata
    producer_id: Optional[str] = None
    consumer_id: Optional[str] = None
    queue_name: Optional[str] = None
    exchange_name: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if message is expired"""
        return self.expires_at and datetime.now() > self.expires_at
    
    @property
    def is_ready_for_delivery(self) -> bool:
        """Check if message is ready for delivery"""
        if self.scheduled_at and datetime.now() < self.scheduled_at:
            return False
        return not self.is_expired and self.status == MessageStatus.PENDING
    
    @property
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return (self.status == MessageStatus.FAILED and 
                self.retry_count < self.max_retries)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "content": self.content,
            "routing_key": self.routing_key,
            "headers": self.headers,
            "priority": self.priority.value,
            "status": self.status.value,
            "delivery_mode": self.delivery_mode.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "producer_id": self.producer_id,
            "consumer_id": self.consumer_id,
            "queue_name": self.queue_name,
            "exchange_name": self.exchange_name
        }


@dataclass
class QueueConfig:
    """Queue configuration"""
    name: str
    max_length: Optional[int] = None
    max_size_bytes: Optional[int] = None
    ttl: Optional[timedelta] = None
    auto_delete: bool = False
    durable: bool = True
    exclusive: bool = False
    dead_letter_exchange: Optional[str] = None
    dead_letter_routing_key: Optional[str] = None
    max_priority: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_length": self.max_length,
            "max_size_bytes": self.max_size_bytes,
            "ttl_seconds": self.ttl.total_seconds() if self.ttl else None,
            "auto_delete": self.auto_delete,
            "durable": self.durable,
            "exclusive": self.exclusive,
            "dead_letter_exchange": self.dead_letter_exchange,
            "dead_letter_routing_key": self.dead_letter_routing_key,
            "max_priority": self.max_priority
        }


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    exchange_type: ExchangeType
    durable: bool = True
    auto_delete: bool = False
    internal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.exchange_type.value,
            "durable": self.durable,
            "auto_delete": self.auto_delete,
            "internal": self.internal
        }


@dataclass
class Binding:
    """Queue to exchange binding"""
    queue_name: str
    exchange_name: str
    routing_key: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Consumer:
    """Message consumer configuration"""
    consumer_id: str
    queue_name: str
    callback: Callable[[Message], bool]
    auto_ack: bool = True
    prefetch_count: int = 1
    exclusive: bool = False
    active: bool = True
    
    # Statistics
    messages_consumed: int = 0
    messages_acked: int = 0
    messages_rejected: int = 0
    last_activity: Optional[datetime] = None


class PriorityQueue:
    """Priority queue for messages"""
    
    def __init__(self, max_priority: int = 5):
        self.max_priority = max_priority
        self.queues: Dict[int, deque] = {i: deque() for i in range(1, max_priority + 1)}
        self.lock = threading.RLock()
        self._size = 0
    
    def put(self, message: Message) -> None:
        """Add message to priority queue"""
        with self.lock:
            priority = min(message.priority.value, self.max_priority)
            self.queues[priority].append(message)
            self._size += 1
    
    def get(self) -> Optional[Message]:
        """Get highest priority message"""
        with self.lock:
            # Check from highest to lowest priority
            for priority in range(self.max_priority, 0, -1):
                if self.queues[priority]:
                    message = self.queues[priority].popleft()
                    self._size -= 1
                    return message
            return None
    
    def peek(self) -> Optional[Message]:
        """Peek at highest priority message without removing"""
        with self.lock:
            for priority in range(self.max_priority, 0, -1):
                if self.queues[priority]:
                    return self.queues[priority][0]
            return None
    
    def size(self) -> int:
        """Get total number of messages"""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self._size == 0


class MessageQueue:
    """Individual message queue"""
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.messages = PriorityQueue(config.max_priority)
        self.processing_messages: Dict[str, Message] = {}
        self.dead_letter_messages: List[Message] = []
        
        # Statistics
        self.total_messages = 0
        self.processed_messages = 0
        self.failed_messages = 0
        self.current_size_bytes = 0
        
        # Consumer management
        self.consumers: Dict[str, Consumer] = {}
        
        # Locks
        self.lock = threading.RLock()
    
    def enqueue(self, message: Message) -> bool:
        """Add message to queue"""
        with self.lock:
            # Check queue limits
            if self.config.max_length and self.messages.size() >= self.config.max_length:
                logger.warning(f"Queue {self.config.name} is full")
                return False
            
            if self.config.max_size_bytes:
                message_size = len(str(message.content))
                if self.current_size_bytes + message_size > self.config.max_size_bytes:
                    logger.warning(f"Queue {self.config.name} size limit exceeded")
                    return False
                self.current_size_bytes += message_size
            
            # Set queue name
            message.queue_name = self.config.name
            
            # Add to queue
            self.messages.put(message)
            self.total_messages += 1
            
            return True
    
    def dequeue(self) -> Optional[Message]:
        """Get next message from queue"""
        with self.lock:
            message = self.messages.get()
            if message:
                # Move to processing
                message.status = MessageStatus.PROCESSING
                self.processing_messages[message.message_id] = message
            return message
    
    def ack_message(self, message_id: str, consumer_id: str) -> bool:
        """Acknowledge message processing"""
        with self.lock:
            message = self.processing_messages.get(message_id)
            if not message:
                return False
            
            # Update message status
            message.status = MessageStatus.PROCESSED
            message.processed_at = datetime.now()
            message.consumer_id = consumer_id
            
            # Remove from processing
            del self.processing_messages[message_id]
            self.processed_messages += 1
            
            # Update size
            if self.config.max_size_bytes:
                self.current_size_bytes -= len(str(message.content))
            
            return True
    
    def nack_message(self, message_id: str, requeue: bool = True) -> bool:
        """Negative acknowledge message"""
        with self.lock:
            message = self.processing_messages.get(message_id)
            if not message:
                return False
            
            # Remove from processing
            del self.processing_messages[message_id]
            
            if requeue and message.should_retry:
                # Increment retry count and requeue
                message.retry_count += 1
                message.status = MessageStatus.RETRY
                message.scheduled_at = datetime.now() + message.retry_delay
                
                self.messages.put(message)
            else:
                # Send to dead letter queue
                message.status = MessageStatus.DEAD_LETTER
                self.dead_letter_messages.append(message)
                self.failed_messages += 1
                
                # Update size
                if self.config.max_size_bytes:
                    self.current_size_bytes -= len(str(message.content))
            
            return True
    
    def add_consumer(self, consumer: Consumer) -> None:
        """Add consumer to queue"""
        with self.lock:
            self.consumers[consumer.consumer_id] = consumer
            logger.info(f"Added consumer {consumer.consumer_id} to queue {self.config.name}")
    
    def remove_consumer(self, consumer_id: str) -> bool:
        """Remove consumer from queue"""
        with self.lock:
            if consumer_id in self.consumers:
                del self.consumers[consumer_id]
                logger.info(f"Removed consumer {consumer_id} from queue {self.config.name}")
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                "name": self.config.name,
                "total_messages": self.total_messages,
                "pending_messages": self.messages.size(),
                "processing_messages": len(self.processing_messages),
                "processed_messages": self.processed_messages,
                "failed_messages": self.failed_messages,
                "dead_letter_messages": len(self.dead_letter_messages),
                "current_size_bytes": self.current_size_bytes,
                "active_consumers": len([c for c in self.consumers.values() if c.active]),
                "total_consumers": len(self.consumers)
            }


class MessageExchange:
    """Message exchange for routing"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.bindings: List[Binding] = []
        
    def add_binding(self, binding: Binding) -> None:
        """Add queue binding"""
        self.bindings.append(binding)
        logger.info(f"Added binding: {binding.queue_name} -> {self.config.name}")
    
    def remove_binding(self, queue_name: str, routing_key: str = "") -> bool:
        """Remove queue binding"""
        for i, binding in enumerate(self.bindings):
            if binding.queue_name == queue_name and binding.routing_key == routing_key:
                del self.bindings[i]
                logger.info(f"Removed binding: {queue_name} -> {self.config.name}")
                return True
        return False
    
    def route_message(self, message: Message) -> List[str]:
        """Route message to appropriate queues"""
        target_queues = []
        
        if self.config.exchange_type == ExchangeType.DIRECT:
            # Direct routing by exact routing key match
            for binding in self.bindings:
                if binding.routing_key == message.routing_key:
                    target_queues.append(binding.queue_name)
        
        elif self.config.exchange_type == ExchangeType.FANOUT:
            # Broadcast to all bound queues
            target_queues = [binding.queue_name for binding in self.bindings]
        
        elif self.config.exchange_type == ExchangeType.TOPIC:
            # Topic routing with wildcards
            target_queues = self._topic_route(message)
        
        elif self.config.exchange_type == ExchangeType.HEADERS:
            # Header-based routing
            target_queues = self._header_route(message)
        
        return target_queues
    
    def _topic_route(self, message: Message) -> List[str]:
        """Route message using topic patterns"""
        target_queues = []
        routing_key = message.routing_key
        
        for binding in self.bindings:
            if self._topic_matches(binding.routing_key, routing_key):
                target_queues.append(binding.queue_name)
        
        return target_queues
    
    def _topic_matches(self, pattern: str, routing_key: str) -> bool:
        """Check if routing key matches topic pattern"""
        pattern_parts = pattern.split('.')
        key_parts = routing_key.split('.')
        
        i = j = 0
        
        while i < len(pattern_parts) and j < len(key_parts):
            if pattern_parts[i] == '#':
                # '#' matches zero or more words
                if i == len(pattern_parts) - 1:
                    return True  # '#' at end matches everything remaining
                
                # Find next non-wildcard pattern
                next_pattern = None
                for k in range(i + 1, len(pattern_parts)):
                    if pattern_parts[k] != '#' and pattern_parts[k] != '*':
                        next_pattern = pattern_parts[k]
                        break
                
                if not next_pattern:
                    return True
                
                # Skip until we find the next pattern
                while j < len(key_parts) and key_parts[j] != next_pattern:
                    j += 1
                
                if j >= len(key_parts):
                    return False
                
                i += 1
                
            elif pattern_parts[i] == '*':
                # '*' matches exactly one word
                i += 1
                j += 1
                
            elif pattern_parts[i] == key_parts[j]:
                # Exact match
                i += 1
                j += 1
                
            else:
                return False
        
        # Check if we've consumed all patterns and keys
        while i < len(pattern_parts) and pattern_parts[i] == '#':
            i += 1
        
        return i == len(pattern_parts) and j == len(key_parts)
    
    def _header_route(self, message: Message) -> List[str]:
        """Route message based on headers"""
        target_queues = []
        
        for binding in self.bindings:
            if self._headers_match(binding.headers, message.headers):
                target_queues.append(binding.queue_name)
        
        return target_queues
    
    def _headers_match(self, binding_headers: Dict[str, Any], 
                      message_headers: Dict[str, Any]) -> bool:
        """Check if message headers match binding headers"""
        for key, value in binding_headers.items():
            if key not in message_headers or message_headers[key] != value:
                return False
        return True


class DistributedMessageQueue:
    """Main distributed message queue system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.node_id = self.config.get("node_id", str(uuid.uuid4()))
        
        # Core components
        self.queues: Dict[str, MessageQueue] = {}
        self.exchanges: Dict[str, MessageExchange] = {}
        
        # Consumer management
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        
        # Scheduled messages (for delayed delivery)
        self.scheduled_messages: List[Tuple[datetime, Message]] = []
        
        # System state
        self.running = False
        self.metrics = {
            "messages_published": 0,
            "messages_consumed": 0,
            "messages_acked": 0,
            "messages_nacked": 0,
            "messages_failed": 0
        }
        
        # Create default exchange
        default_exchange = ExchangeConfig("", ExchangeType.DIRECT)
        self.exchanges[""] = MessageExchange(default_exchange)
    
    async def start(self) -> None:
        """Start message queue system"""
        if self.running:
            return
        
        self.running = True
        
        # Start scheduled message processor
        asyncio.create_task(self._process_scheduled_messages())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_messages())
        
        logger.info(f"Message queue system started (node: {self.node_id})")
    
    async def stop(self) -> None:
        """Stop message queue system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop all consumer tasks
        for task in self.consumer_tasks.values():
            task.cancel()
        
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks.values(), return_exceptions=True)
        
        self.consumer_tasks.clear()
        
        logger.info("Message queue system stopped")
    
    def create_queue(self, config: QueueConfig) -> bool:
        """Create a new queue"""
        if config.name in self.queues:
            return False
        
        self.queues[config.name] = MessageQueue(config)
        logger.info(f"Created queue: {config.name}")
        return True
    
    def delete_queue(self, queue_name: str, if_unused: bool = False) -> bool:
        """Delete a queue"""
        if queue_name not in self.queues:
            return False
        
        queue = self.queues[queue_name]
        
        if if_unused and (queue.messages.size() > 0 or len(queue.consumers) > 0):
            return False
        
        # Stop all consumers for this queue
        consumers_to_remove = [cid for cid, consumer in queue.consumers.items()]
        for consumer_id in consumers_to_remove:
            self.stop_consuming(consumer_id)
        
        del self.queues[queue_name]
        logger.info(f"Deleted queue: {queue_name}")
        return True
    
    def create_exchange(self, config: ExchangeConfig) -> bool:
        """Create a new exchange"""
        if config.name in self.exchanges:
            return False
        
        self.exchanges[config.name] = MessageExchange(config)
        logger.info(f"Created exchange: {config.name} (type: {config.exchange_type.value})")
        return True
    
    def delete_exchange(self, exchange_name: str) -> bool:
        """Delete an exchange"""
        if exchange_name not in self.exchanges or exchange_name == "":
            return False  # Cannot delete default exchange
        
        del self.exchanges[exchange_name]
        logger.info(f"Deleted exchange: {exchange_name}")
        return True
    
    def bind_queue(self, queue_name: str, exchange_name: str, 
                   routing_key: str = "", headers: Dict[str, Any] = None) -> bool:
        """Bind queue to exchange"""
        if queue_name not in self.queues or exchange_name not in self.exchanges:
            return False
        
        binding = Binding(
            queue_name=queue_name,
            exchange_name=exchange_name,
            routing_key=routing_key,
            headers=headers or {}
        )
        
        self.exchanges[exchange_name].add_binding(binding)
        return True
    
    def unbind_queue(self, queue_name: str, exchange_name: str, 
                     routing_key: str = "") -> bool:
        """Unbind queue from exchange"""
        if exchange_name not in self.exchanges:
            return False
        
        return self.exchanges[exchange_name].remove_binding(queue_name, routing_key)
    
    async def publish(self, exchange_name: str, message: Message) -> bool:
        """Publish message to exchange"""
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not found")
            return False
        
        # Set exchange name
        message.exchange_name = exchange_name
        
        # Route message to appropriate queues
        exchange = self.exchanges[exchange_name]
        target_queues = exchange.route_message(message)
        
        if not target_queues:
            logger.warning(f"No queues bound for message routing key: {message.routing_key}")
            return False
        
        # Enqueue message to target queues
        success_count = 0
        for queue_name in target_queues:
            if queue_name in self.queues:
                # Create copy of message for each queue
                queue_message = Message(
                    message_id=f"{message.message_id}_{queue_name}",
                    content=message.content,
                    routing_key=message.routing_key,
                    headers=message.headers.copy(),
                    priority=message.priority,
                    delivery_mode=message.delivery_mode,
                    expires_at=message.expires_at,
                    scheduled_at=message.scheduled_at,
                    retry_count=message.retry_count,
                    max_retries=message.max_retries,
                    trace_id=message.trace_id,
                    correlation_id=message.correlation_id,
                    producer_id=message.producer_id
                )
                
                if message.scheduled_at and message.scheduled_at > datetime.now():
                    # Add to scheduled messages
                    heapq.heappush(self.scheduled_messages, (message.scheduled_at, queue_message))
                else:
                    # Immediate delivery
                    if self.queues[queue_name].enqueue(queue_message):
                        success_count += 1
        
        self.metrics["messages_published"] += success_count
        return success_count > 0
    
    async def consume(self, queue_name: str, callback: Callable[[Message], bool],
                     consumer_id: str = None, auto_ack: bool = True, 
                     prefetch_count: int = 1) -> str:
        """Start consuming messages from queue"""
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} not found")
        
        if not consumer_id:
            consumer_id = str(uuid.uuid4())
        
        consumer = Consumer(
            consumer_id=consumer_id,
            queue_name=queue_name,
            callback=callback,
            auto_ack=auto_ack,
            prefetch_count=prefetch_count
        )
        
        # Add consumer to queue
        self.queues[queue_name].add_consumer(consumer)
        
        # Start consumer task
        task = asyncio.create_task(self._consumer_loop(consumer))
        self.consumer_tasks[consumer_id] = task
        
        logger.info(f"Started consumer {consumer_id} for queue {queue_name}")
        return consumer_id
    
    def stop_consuming(self, consumer_id: str) -> bool:
        """Stop message consumer"""
        if consumer_id not in self.consumer_tasks:
            return False
        
        # Cancel consumer task
        task = self.consumer_tasks[consumer_id]
        task.cancel()
        del self.consumer_tasks[consumer_id]
        
        # Remove consumer from queue
        for queue in self.queues.values():
            if consumer_id in queue.consumers:
                queue.remove_consumer(consumer_id)
                break
        
        logger.info(f"Stopped consumer {consumer_id}")
        return True
    
    async def _consumer_loop(self, consumer: Consumer) -> None:
        """Main consumer processing loop"""
        queue = self.queues[consumer.queue_name]
        
        while self.running and consumer.active:
            try:
                # Get message from queue
                message = queue.dequeue()
                
                if not message:
                    await asyncio.sleep(0.1)  # Brief pause if no messages
                    continue
                
                # Skip expired messages
                if message.is_expired:
                    queue.nack_message(message.message_id, requeue=False)
                    continue
                
                # Process message
                consumer.last_activity = datetime.now()
                
                try:
                    # Call consumer callback
                    success = await self._call_consumer_callback(consumer, message)
                    
                    if consumer.auto_ack:
                        if success:
                            queue.ack_message(message.message_id, consumer.consumer_id)
                            consumer.messages_acked += 1
                            self.metrics["messages_acked"] += 1
                        else:
                            queue.nack_message(message.message_id, requeue=True)
                            consumer.messages_rejected += 1
                            self.metrics["messages_nacked"] += 1
                    
                    consumer.messages_consumed += 1
                    self.metrics["messages_consumed"] += 1
                    
                except Exception as e:
                    logger.error(f"Consumer {consumer.consumer_id} error: {str(e)}")
                    queue.nack_message(message.message_id, requeue=True)
                    consumer.messages_rejected += 1
                    self.metrics["messages_failed"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _call_consumer_callback(self, consumer: Consumer, message: Message) -> bool:
        """Call consumer callback function"""
        if asyncio.iscoroutinefunction(consumer.callback):
            return await consumer.callback(message)
        else:
            return consumer.callback(message)
    
    async def _process_scheduled_messages(self) -> None:
        """Process scheduled messages for delayed delivery"""
        while self.running:
            try:
                now = datetime.now()
                
                # Process all ready scheduled messages
                while (self.scheduled_messages and 
                       self.scheduled_messages[0][0] <= now):
                    
                    scheduled_time, message = heapq.heappop(self.scheduled_messages)
                    
                    # Find target queue and enqueue
                    if message.queue_name and message.queue_name in self.queues:
                        message.scheduled_at = None  # Clear scheduled time
                        self.queues[message.queue_name].enqueue(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled message processing error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages periodically"""
        while self.running:
            try:
                # Clean up expired messages from all queues
                for queue in self.queues.values():
                    expired_messages = []
                    
                    # Check processing messages for expiration
                    for msg_id, message in list(queue.processing_messages.items()):
                        if message.is_expired:
                            expired_messages.append(msg_id)
                    
                    # Remove expired processing messages
                    for msg_id in expired_messages:
                        queue.nack_message(msg_id, requeue=False)
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(60)
    
    def ack_message(self, queue_name: str, message_id: str, consumer_id: str) -> bool:
        """Manually acknowledge message"""
        if queue_name not in self.queues:
            return False
        
        success = self.queues[queue_name].ack_message(message_id, consumer_id)
        if success:
            self.metrics["messages_acked"] += 1
        return success
    
    def nack_message(self, queue_name: str, message_id: str, requeue: bool = True) -> bool:
        """Manually negative acknowledge message"""
        if queue_name not in self.queues:
            return False
        
        success = self.queues[queue_name].nack_message(message_id, requeue)
        if success:
            self.metrics["messages_nacked"] += 1
        return success
    
    def get_queue_info(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get queue information"""
        if queue_name not in self.queues:
            return None
        
        queue = self.queues[queue_name]
        stats = queue.get_stats()
        stats["config"] = queue.config.to_dict()
        return stats
    
    def get_exchange_info(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get exchange information"""
        if exchange_name not in self.exchanges:
            return None
        
        exchange = self.exchanges[exchange_name]
        return {
            "config": exchange.config.to_dict(),
            "bindings": len(exchange.bindings),
            "binding_details": [
                {
                    "queue_name": binding.queue_name,
                    "routing_key": binding.routing_key,
                    "headers": binding.headers
                }
                for binding in exchange.bindings
            ]
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get message queue system status"""
        total_pending = sum(queue.messages.size() for queue in self.queues.values())
        total_processing = sum(len(queue.processing_messages) for queue in self.queues.values())
        total_consumers = sum(len(queue.consumers) for queue in self.queues.values())
        
        return {
            "node_id": self.node_id,
            "running": self.running,
            "total_queues": len(self.queues),
            "total_exchanges": len(self.exchanges),
            "total_consumers": total_consumers,
            "total_pending_messages": total_pending,
            "total_processing_messages": total_processing,
            "scheduled_messages": len(self.scheduled_messages),
            "metrics": self.metrics.copy()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        queue_metrics = {}
        for name, queue in self.queues.items():
            queue_metrics[name] = queue.get_stats()
        
        return {
            "system_metrics": self.metrics.copy(),
            "queue_metrics": queue_metrics,
            "consumer_count": len(self.consumer_tasks),
            "scheduled_messages": len(self.scheduled_messages)
        }