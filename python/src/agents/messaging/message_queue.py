"""
High-Performance Message Queue for Agency Swarm

Implements a scalable, high-throughput message queuing system:
- Distributed message queuing with multiple backends
- Priority-based message routing
- Message persistence and recovery
- Real-time message processing
- Performance monitoring and optimization

Target Performance:
- 10,000+ messages/second throughput
- <100ms message processing latency
- 99.9% message delivery reliability
- Automatic load balancing
- Horizontal scaling support
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)

# Optional backend support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    aio_pika = None

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1    # System-critical messages
    HIGH = 2        # High-priority user messages
    NORMAL = 3      # Normal processing
    LOW = 4         # Background tasks
    BATCH = 5       # Batch processing

class MessageType(Enum):
    """Types of messages"""
    TASK_ASSIGNMENT = "task_assignment"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_REPORT = "error_report"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    COLLABORATION = "collaboration"
    KNOWLEDGE_UPDATE = "knowledge_update"
    METRICS = "metrics"

class QueueBackend(Enum):
    """Message queue backends"""
    MEMORY = "memory"          # In-memory queue
    REDIS = "redis"            # Redis-based queue
    RABBITMQ = "rabbitmq"      # RabbitMQ queue
    KAFKA = "kafka"            # Apache Kafka (not implemented)
    DATABASE = "database"      # Database-backed queue

@dataclass
class Message:
    """Message structure"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    sender_id: str
    recipient_id: Optional[str] = None
    topic: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    requires_ack: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get message age in seconds"""
        return time.time() - self.created_at

    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1

    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries

@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    queue_name: str
    total_messages: int
    pending_messages: int
    processing_messages: int
    completed_messages: int
    failed_messages: int
    avg_processing_time_ms: float
    throughput_per_sec: float
    error_rate_percent: float
    last_updated: float = field(default_factory=time.time)

class MemoryQueue:
    """In-memory message queue implementation"""

    def __init__(self, queue_name: str, max_size: int = 10000):
        self.queue_name = queue_name
        self.max_size = max_size

        # Priority queues
        self.priority_queues = {
            MessagePriority.CRITICAL: deque(maxlen=max_size // 5),
            MessagePriority.HIGH: deque(maxlen=max_size // 3),
            MessagePriority.NORMAL: deque(maxlen=max_size // 2),
            MessagePriority.LOW: deque(maxlen=max_size),
            MessagePriority.BATCH: deque(maxlen=max_size * 2)
        }

        # Message tracking
        self.processing_messages = {}  # message_id -> Message
        self.completed_messages = set()  # message_id set
        self.failed_messages = set()  # message_id set

        # Metrics
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.total_completed = 0
        self.total_failed = 0
        self.total_processing_time = 0.0
        self.lock = threading.RLock()

    def enqueue(self, message: Message) -> bool:
        """Add message to queue"""
        with self.lock:
            # Check queue capacity
            total_size = sum(len(q) for q in self.priority_queues.values())
            if total_size >= self.max_size:
                logger.warning(f"Queue {self.queue_name} is full, dropping message")
                return False

            # Add to appropriate priority queue
            queue = self.priority_queues[message.priority]
            queue.append(message)
            self.total_enqueued += 1

            return True

    def dequeue(self) -> Optional[Message]:
        """Get next message from queue (priority-based)"""
        with self.lock:
            # Check queues in priority order
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, MessagePriority.NORMAL, MessagePriority.LOW, MessagePriority.BATCH]:
                queue = self.priority_queues[priority]
                if queue:
                    message = queue.popleft()
                    self.processing_messages[message.message_id] = message
                    self.total_dequeued += 1
                    return message

            return None

    def complete(self, message_id: str, processing_time_ms: float):
        """Mark message as completed"""
        with self.lock:
            if message_id in self.processing_messages:
                message = self.processing_messages.pop(message_id)
                self.completed_messages.add(message_id)
                self.total_completed += 1
                self.total_processing_time += processing_time_ms

    def fail(self, message_id: str, error: Optional[str] = None):
        """Mark message as failed"""
        with self.lock:
            if message_id in self.processing_messages:
                message = self.processing_messages.pop(message_id)

                # Retry if possible
                if message.can_retry():
                    message.increment_retry()
                    self.priority_queues[message.priority].append(message)
                    logger.info(f"Retrying message {message_id} (attempt {message.retry_count})")
                else:
                    self.failed_messages.add(message_id)
                    self.total_failed += 1
                    logger.error(f"Message {message_id} failed permanently: {error}")

    def get_metrics(self) -> QueueMetrics:
        """Get queue metrics"""
        with self.lock:
            pending = sum(len(q) for q in self.priority_queues.values())
            processing = len(self.processing_messages)
            completed = len(self.completed_messages)
            failed = len(self.failed_messages)

            avg_processing_time = (self.total_processing_time / max(self.total_completed, 1)) * 1000

            # Calculate throughput (messages per second)
            total_processed = self.total_completed + self.total_failed
            uptime = time.time() - getattr(self, 'start_time', time.time())
            throughput = total_processed / max(uptime, 1)

            error_rate = (self.total_failed / max(total_processed, 1)) * 100

            return QueueMetrics(
                queue_name=self.queue_name,
                total_messages=self.total_enqueued,
                pending_messages=pending,
                processing_messages=processing,
                completed_messages=completed,
                failed_messages=failed,
                avg_processing_time_ms=avg_processing_time,
                throughput_per_sec=throughput,
                error_rate_percent=error_rate
            )

class RedisQueue:
    """Redis-based distributed message queue"""

    def __init__(self, queue_name: str, redis_config: Dict[str, Any]):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")

        self.queue_name = queue_name
        self.redis_client = redis.Redis(**redis_config)

        # Key patterns
        self.pending_key = f"queue:{queue_name}:pending"
        self.processing_key = f"queue:{queue_name}:processing"
        self.completed_key = f"queue:{queue_name}:completed"
        self.failed_key = f"queue:{queue_name}:failed"
        self.metrics_key = f"queue:{queue_name}:metrics"

    def enqueue(self, message: Message) -> bool:
        """Add message to Redis queue"""
        try:
            # Serialize message
            message_data = json.dumps({
                "message_id": message.message_id,
                "message_type": message.message_type.value,
                "priority": message.priority.value,
                "payload": message.payload,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "topic": message.topic,
                "created_at": message.created_at,
                "expires_at": message.expires_at,
                "retry_count": message.retry_count,
                "max_retries": message.max_retries,
                "requires_ack": message.requires_ack,
                "metadata": message.metadata
            })

            # Add to priority-ordered list
            score = self._calculate_priority_score(message)
            self.redis_client.zadd(self.pending_key, {message_data: score})

            return True

        except Exception as e:
            logger.error(f"Failed to enqueue message to Redis: {e}")
            return False

    def dequeue(self) -> Optional[Message]:
        """Get next message from Redis queue"""
        try:
            # Get highest priority message
            result = self.redis_client.zrange(self.pending_key, 0, 0, withscores=True)

            if not result:
                return None

            message_data, score = result[0]

            # Move to processing queue
            pipe = self.redis_client.pipeline()
            pipe.zrem(self.pending_key, message_data)
            pipe.lpush(self.processing_key, message_data)
            pipe.execute()

            # Deserialize message
            data = json.loads(message_data)
            return Message(**data)

        except Exception as e:
            logger.error(f"Failed to dequeue message from Redis: {e}")
            return None

    def complete(self, message_id: str, processing_time_ms: float):
        """Mark message as completed"""
        try:
            # Remove from processing and add to completed
            pipe = self.redis_client.pipeline()

            # Find and remove from processing queue
            processing_messages = self.redis_client.lrange(self.processing_key, 0, -1)
            for msg_data in processing_messages:
                data = json.loads(msg_data)
                if data.get("message_id") == message_id:
                    pipe.lrem(self.processing_key, 1, msg_data)
                    pipe.lpush(self.completed_key, msg_data)
                    break

            pipe.execute()

            # Update metrics
            self._update_metrics({"completed": 1, "processing_time": processing_time_ms})

        except Exception as e:
            logger.error(f"Failed to complete message in Redis: {e}")

    def fail(self, message_id: str, error: Optional[str] = None):
        """Mark message as failed"""
        try:
            # Find message in processing queue
            processing_messages = self.redis_client.lrange(self.processing_key, 0, -1)
            message_data = None

            for msg_data in processing_messages:
                data = json.loads(msg_data)
                if data.get("message_id") == message_id:
                    message_data = msg_data
                    break

            if message_data:
                data = json.loads(message_data)
                message = Message(**data)

                # Retry if possible
                if message.can_retry():
                    message.increment_retry()
                    self.enqueue(message)
                    logger.info(f"Retrying message {message_id} (attempt {message.retry_count})")
                else:
                    # Move to failed queue
                    pipe = self.redis_client.pipeline()
                    pipe.lrem(self.processing_key, 1, message_data)
                    pipe.lpush(self.failed_key, message_data)
                    pipe.execute()

                # Update metrics
                self._update_metrics({"failed": 1})

        except Exception as e:
            logger.error(f"Failed to mark message as failed in Redis: {e}")

    def _calculate_priority_score(self, message: Message) -> float:
        """Calculate Redis priority score (lower = higher priority)"""
        base_score = message.priority.value * 1000
        age_penalty = message.age_seconds * 10
        retry_penalty = message.retry_count * 100
        return base_score + age_penalty + retry_penalty

    def _update_metrics(self, updates: Dict[str, Any]):
        """Update queue metrics"""
        try:
            current_metrics = self.redis_client.hgetall(self.metrics_key)
            metrics = {k.decode(): json.loads(v.decode()) for k, v in current_metrics.items()}

            for key, value in updates.items():
                if key in metrics:
                    if isinstance(metrics[key], (int, float)):
                        metrics[key] += value
                    else:
                        metrics[key] = value
                else:
                    metrics[key] = value

            # Convert metrics back to strings for Redis
            string_metrics = {k: json.dumps(v) for k, v in metrics.items()}
            self.redis_client.hmset(self.metrics_key, string_metrics)

        except Exception as e:
            logger.error(f"Failed to update Redis metrics: {e}")

    def get_metrics(self) -> QueueMetrics:
        """Get queue metrics from Redis"""
        try:
            metrics_data = self.redis_client.hgetall(self.metrics_key)
            metrics = {k.decode(): json.loads(v.decode()) for k, v in metrics_data.items()}

            pending = self.redis_client.zcard(self.pending_key)
            processing = self.redis_client.llen(self.processing_key)
            completed = self.redis_client.llen(self.completed_key)
            failed = self.redis_client.llen(self.failed_key)

            return QueueMetrics(
                queue_name=self.queue_name,
                total_messages=metrics.get("total", 0),
                pending_messages=pending,
                processing_messages=processing,
                completed_messages=completed,
                failed_messages=failed,
                avg_processing_time_ms=metrics.get("avg_processing_time", 0),
                throughput_per_sec=metrics.get("throughput", 0),
                error_rate_percent=metrics.get("error_rate", 0)
            )

        except Exception as e:
            logger.error(f"Failed to get Redis queue metrics: {e}")
            return QueueMetrics(
                queue_name=self.queue_name,
                total_messages=0,
                pending_messages=0,
                processing_messages=0,
                completed_messages=0,
                failed_messages=0,
                avg_processing_time_ms=0,
                throughput_per_sec=0,
                error_rate_percent=0
            )

class MessageQueueManager:
    """Manages multiple message queues with intelligent routing"""

    def __init__(self, backend: QueueBackend = QueueBackend.MEMORY, **backend_config):
        self.backend = backend
        self.backend_config = backend_config

        # Queue management
        self.queues: Dict[str, Union[MemoryQueue, RedisQueue]] = {}
        self.queue_configs: Dict[str, Dict] = {}
        self.message_handlers: Dict[str, Callable] = {}

        # Processing management
        self.processors: Dict[str, ThreadPoolExecutor] = {}
        self.running = False
        self.processor_threads = {}

        # Performance tracking
        self.global_metrics = {
            "total_messages_processed": 0,
            "total_processing_time": 0.0,
            "error_rate": 0.0,
            "throughput_per_sec": 0.0,
            "start_time": time.time()
        }

        # Start the manager
        self.start()

    def start(self):
        """Start message queue manager"""
        logger.info(f"Starting Message Queue Manager with {self.backend.value} backend")
        self.running = True

        # Create default queues
        self.create_queue("default", {"max_size": 10000})
        self.create_queue("critical", {"max_size": 1000, "priority_only": True})
        self.create_queue("batch", {"max_size": 50000})

        logger.info("Message Queue Manager started")

    def stop(self):
        """Stop message queue manager"""
        logger.info("Stopping Message Queue Manager")
        self.running = False

        # Stop all processors
        for queue_name, processor in self.processors.items():
            processor.shutdown(wait=True)

        # Stop processor threads
        for thread in self.processor_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)

        logger.info("Message Queue Manager stopped")

    def create_queue(self, queue_name: str, config: Dict[str, Any]) -> bool:
        """Create a new message queue"""
        try:
            if self.backend == QueueBackend.MEMORY:
                queue = MemoryQueue(queue_name, max_size=config.get("max_size", 10000))
            elif self.backend == QueueBackend.REDIS:
                queue = RedisQueue(queue_name, self.backend_config)
            else:
                logger.error(f"Unsupported backend: {self.backend}")
                return False

            self.queues[queue_name] = queue
            self.queue_configs[queue_name] = config

            # Start processor for this queue
            self._start_queue_processor(queue_name)

            logger.info(f"Created queue '{queue_name}' with {self.backend.value} backend")
            return True

        except Exception as e:
            logger.error(f"Failed to create queue '{queue_name}': {e}")
            return False

    def _start_queue_processor(self, queue_name: str):
        """Start processor thread for a queue"""
        processor = ThreadPoolExecutor(
            max_workers=self.queue_configs[queue_name].get("max_workers", 4),
            thread_name_prefix=f"queue-{queue_name}"
        )
        self.processors[queue_name] = processor

        # Start processor thread
        thread = threading.Thread(target=self._process_queue_messages, args=(queue_name,))
        thread.daemon = True
        thread.start()
        self.processor_threads[queue_name] = thread

    def _process_queue_messages(self, queue_name: str):
        """Process messages from a specific queue"""
        queue = self.queues[queue_name]
        processor = self.processors[queue_name]

        while self.running:
            try:
                # Get message from queue
                message = queue.dequeue()
                if message is None:
                    time.sleep(0.1)  # Short sleep when queue is empty
                    continue

                # Process message asynchronously
                future = processor.submit(self._process_message, queue_name, message)

                # Handle completion
                future.add_done_callback(lambda f: self._handle_message_completion(f, queue_name, message))

            except Exception as e:
                logger.error(f"Error processing queue '{queue_name}': {e}")
                time.sleep(1)

    def _process_message(self, queue_name: str, message: Message) -> Any:
        """Process a single message"""
        start_time = time.time()

        try:
            # Check if message is expired
            if message.is_expired:
                logger.warning(f"Message {message.message_id} has expired")
                return None

            # Get message handler
            handler_key = message.message_type.value
            if message.topic:
                handler_key = f"{message.topic}:{handler_key}"

            handler = self.message_handlers.get(handler_key)
            if not handler:
                logger.warning(f"No handler for message type: {handler_key}")
                return None

            # Process message
            result = handler(message)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Update global metrics
            self.global_metrics["total_messages_processed"] += 1
            self.global_metrics["total_processing_time"] += processing_time_ms

            return result

        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            raise

    def _handle_message_completion(self, future, queue_name: str, message: Message):
        """Handle message processing completion"""
        try:
            result = future.result()
            processing_time_ms = (time.time() - message.created_at) * 1000

            # Mark message as completed
            self.queues[queue_name].complete(message.message_id, processing_time_ms)

        except Exception as e:
            logger.error(f"Message {message.message_id} processing failed: {e}")
            self.queues[queue_name].fail(message.message_id, str(e))

    def enqueue_message(self,
                       message_type: MessageType,
                       payload: Dict[str, Any],
                       sender_id: str,
                       recipient_id: Optional[str] = None,
                       topic: Optional[str] = None,
                       priority: MessagePriority = MessagePriority.NORMAL,
                       queue_name: str = "default",
                       expires_in_seconds: Optional[int] = None,
                       requires_ack: bool = True) -> str:

        """Enqueue a new message"""
        message_id = str(uuid.uuid4())

        message = Message(
            message_id=message_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            sender_id=sender_id,
            recipient_id=recipient_id,
            topic=topic,
            expires_at=time.time() + expires_in_seconds if expires_in_seconds else None,
            requires_ack=requires_ack
        )

        # Route to appropriate queue
        target_queue = self._route_message_to_queue(message, queue_name)
        if target_queue and target_queue.enqueue(message):
            logger.debug(f"Enqueued message {message_id} to queue {target_queue.queue_name}")
            return message_id
        else:
            logger.error(f"Failed to enqueue message {message_id}")
            return ""

    def _route_message_to_queue(self, message: Message, preferred_queue: str) -> Optional[Union[MemoryQueue, RedisQueue]]:
        """Route message to appropriate queue based on priority and type"""
        # High priority messages go to critical queue
        if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
            return self.queues.get("critical")

        # Batch messages go to batch queue
        if message.priority == MessagePriority.BATCH:
            return self.queues.get("batch")

        # Default to preferred queue
        return self.queues.get(preferred_queue, self.queues.get("default"))

    def register_handler(self, message_type: str, topic: Optional[str], handler: Callable):
        """Register message handler"""
        if topic:
            handler_key = f"{topic}:{message_type}"
        else:
            handler_key = message_type

        self.message_handlers[handler_key] = handler
        logger.info(f"Registered handler for {handler_key}")

    def get_queue_metrics(self, queue_name: str) -> Optional[QueueMetrics]:
        """Get metrics for a specific queue"""
        queue = self.queues.get(queue_name)
        if queue:
            return queue.get_metrics()
        return None

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global message queue metrics"""
        uptime = time.time() - self.global_metrics["start_time"]
        throughput = self.global_metrics["total_messages_processed"] / max(uptime, 1)
        avg_processing_time = (self.global_metrics["total_processing_time"] / max(self.global_metrics["total_messages_processed"], 1)) * 1000

        # Aggregate queue metrics
        queue_metrics = {}
        for queue_name, queue in self.queues.items():
            queue_metrics[queue_name] = queue.get_metrics().__dict__

        return {
            "global": {
                "total_messages_processed": self.global_metrics["total_messages_processed"],
                "total_processing_time_ms": self.global_metrics["total_processing_time"],
                "uptime_seconds": uptime,
                "throughput_per_sec": throughput,
                "avg_processing_time_ms": avg_processing_time,
                "active_queues": len(self.queues),
                "backend": self.backend.value
            },
            "queues": queue_metrics
        }

# Example usage
if __name__ == "__main__":
    # Create message queue manager
    manager = MessageQueueManager(backend=QueueBackend.MEMORY)

    try:
        # Define a message handler
        def handle_task_assignment(message: Message):
            print(f"Processing task: {message.payload}")
            time.sleep(0.1)  # Simulate processing
            return {"status": "completed", "result": "success"}

        # Register handler
        manager.register_handler("task_assignment", None, handle_task_assignment)

        # Enqueue a message
        message_id = manager.enqueue_message(
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={"task": "process_data", "data": "sample"},
            sender_id="system",
            priority=MessagePriority.HIGH
        )

        print(f"Enqueued message: {message_id}")

        # Wait for processing
        time.sleep(2)

        # Get metrics
        metrics = manager.get_global_metrics()
        print(f"Global metrics: {metrics}")

    finally:
        manager.stop()