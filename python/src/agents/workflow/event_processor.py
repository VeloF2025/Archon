"""
Event Processor for Workflow Automation
Handles event-driven workflow triggers and executions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict, deque
import hashlib
import uuid

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for workflow triggers"""
    SYSTEM = "system"
    USER = "user"
    API = "api"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    FILE = "file"
    DATABASE = "database"
    MESSAGE = "message"
    METRIC = "metric"
    ERROR = "error"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"


@dataclass
class EventPattern:
    """Pattern for event matching"""
    pattern_id: str
    name: str
    event_type: EventType
    conditions: Dict[str, Any]
    regex_patterns: Dict[str, str] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    correlation_window: Optional[timedelta] = None
    
    def matches(self, event: 'Event') -> bool:
        """Check if event matches this pattern"""
        if event.event_type != self.event_type:
            return False
            
        # Check required fields
        for field in self.required_fields:
            if field not in event.data:
                return False
                
        # Check conditions
        for key, expected_value in self.conditions.items():
            if key not in event.data:
                return False
            if isinstance(expected_value, dict) and "$operator" in expected_value:
                if not self._evaluate_operator(event.data[key], expected_value):
                    return False
            elif event.data[key] != expected_value:
                return False
                
        # Check regex patterns
        for key, pattern in self.regex_patterns.items():
            if key not in event.data:
                return False
            if not re.match(pattern, str(event.data[key])):
                return False
                
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in str(event.data):
                return False
                
        return True
    
    def _evaluate_operator(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate operator-based conditions"""
        operator = condition["$operator"]
        expected = condition.get("value")
        
        if operator == "$gt":
            return value > expected
        elif operator == "$gte":
            return value >= expected
        elif operator == "$lt":
            return value < expected
        elif operator == "$lte":
            return value <= expected
        elif operator == "$eq":
            return value == expected
        elif operator == "$ne":
            return value != expected
        elif operator == "$in":
            return value in expected
        elif operator == "$nin":
            return value not in expected
        elif operator == "$regex":
            return bool(re.match(expected, str(value)))
        elif operator == "$exists":
            return (value is not None) == expected
        
        return False


@dataclass
class Event:
    """Workflow event"""
    event_id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority = EventPriority.NORMAL
    status: EventStatus = EventStatus.PENDING
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[timedelta] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
            "metadata": self.metadata,
            "retry_count": self.retry_count
        }


@dataclass
class EventHandler:
    """Handler for processing events"""
    handler_id: str
    name: str
    pattern: EventPattern
    callback: Callable
    filter_func: Optional[Callable] = None
    transform_func: Optional[Callable] = None
    error_handler: Optional[Callable] = None
    enabled: bool = True
    async_execution: bool = True
    timeout: Optional[float] = 30.0
    
    async def handle(self, event: Event) -> Any:
        """Handle an event"""
        if not self.enabled:
            return None
            
        # Apply filter if provided
        if self.filter_func and not self.filter_func(event):
            return None
            
        # Transform event if needed
        if self.transform_func:
            event = self.transform_func(event)
            
        # Execute handler
        try:
            if self.async_execution:
                if self.timeout:
                    return await asyncio.wait_for(
                        self.callback(event),
                        timeout=self.timeout
                    )
                else:
                    return await self.callback(event)
            else:
                return self.callback(event)
        except asyncio.TimeoutError:
            logger.error(f"Handler {self.name} timed out processing event {event.event_id}")
            if self.error_handler:
                return await self.error_handler(event, "timeout")
            raise
        except Exception as e:
            logger.error(f"Handler {self.name} failed: {str(e)}")
            if self.error_handler:
                return await self.error_handler(event, e)
            raise


@dataclass
class EventCorrelation:
    """Event correlation for complex event processing"""
    correlation_id: str
    pattern: str
    events: List[Event] = field(default_factory=list)
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    completed: bool = False
    
    def add_event(self, event: Event) -> None:
        """Add event to correlation"""
        self.events.append(event)
        if not self.window_start:
            self.window_start = event.timestamp
        self.window_end = event.timestamp
    
    def is_complete(self, required_count: int = None) -> bool:
        """Check if correlation is complete"""
        if required_count:
            return len(self.events) >= required_count
        return self.completed


class EventProcessor:
    """Main event processor for workflow automation"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue: List[Tuple[int, Event]] = []
        self.handlers: Dict[str, EventHandler] = {}
        self.patterns: Dict[str, EventPattern] = {}
        self.correlations: Dict[str, EventCorrelation] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.metrics: Dict[str, int] = defaultdict(int)
        self.processing = False
        self.workers: List[asyncio.Task] = []
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.dead_letter_queue: deque = deque(maxlen=100)
        
    async def start(self, num_workers: int = 5) -> None:
        """Start event processing"""
        if self.processing:
            return
            
        self.processing = True
        logger.info(f"Starting event processor with {num_workers} workers")
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._process_events(i))
            self.workers.append(worker)
            
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_events())
        
    async def stop(self) -> None:
        """Stop event processing"""
        self.processing = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Event processor stopped")
        
    async def emit(self, event: Event) -> str:
        """Emit an event for processing"""
        if not self.processing:
            raise RuntimeError("Event processor not running")
            
        # Generate event ID if not provided
        if not event.event_id:
            event.event_id = str(uuid.uuid4())
            
        # Add to appropriate queue based on priority
        if event.priority == EventPriority.CRITICAL:
            # Process immediately
            await self._process_event(event)
        else:
            await self.event_queue.put(event)
            
        self.metrics["events_emitted"] += 1
        logger.debug(f"Emitted event {event.event_id} with priority {event.priority}")
        
        return event.event_id
        
    def register_handler(self, handler: EventHandler) -> None:
        """Register an event handler"""
        self.handlers[handler.handler_id] = handler
        self.patterns[handler.pattern.pattern_id] = handler.pattern
        
        # Update subscriptions
        event_type = handler.pattern.event_type.value
        self.subscriptions[event_type].add(handler.handler_id)
        
        logger.info(f"Registered handler {handler.name}")
        
    def unregister_handler(self, handler_id: str) -> None:
        """Unregister an event handler"""
        if handler_id in self.handlers:
            handler = self.handlers[handler_id]
            event_type = handler.pattern.event_type.value
            self.subscriptions[event_type].discard(handler_id)
            del self.handlers[handler_id]
            logger.info(f"Unregistered handler {handler_id}")
            
    async def _process_events(self, worker_id: int) -> None:
        """Worker to process events from queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.processing:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                await self._process_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    async def _process_event(self, event: Event) -> None:
        """Process a single event"""
        try:
            event.status = EventStatus.PROCESSING
            self.event_history.append(event)
            
            # Check for correlation
            if event.correlation_id:
                await self._handle_correlation(event)
                
            # Find matching handlers
            handlers = self._find_matching_handlers(event)
            
            if not handlers:
                logger.debug(f"No handlers found for event {event.event_id}")
                event.status = EventStatus.PROCESSED
                return
                
            # Execute handlers
            results = []
            for handler in handlers:
                try:
                    result = await handler.handle(event)
                    results.append(result)
                    self.metrics[f"handler_{handler.handler_id}_success"] += 1
                except Exception as e:
                    logger.error(f"Handler {handler.name} failed: {str(e)}")
                    self.metrics[f"handler_{handler.handler_id}_failure"] += 1
                    
                    # Retry if needed
                    if event.retry_count < event.max_retries:
                        event.retry_count += 1
                        event.status = EventStatus.RETRY
                        await self.emit(event)
                    else:
                        event.status = EventStatus.FAILED
                        self.dead_letter_queue.append(event)
                        
            event.status = EventStatus.PROCESSED
            self.metrics["events_processed"] += 1
            
        except Exception as e:
            logger.error(f"Failed to process event {event.event_id}: {str(e)}")
            event.status = EventStatus.FAILED
            self.metrics["events_failed"] += 1
            
    def _find_matching_handlers(self, event: Event) -> List[EventHandler]:
        """Find handlers that match the event"""
        matching_handlers = []
        
        # Get handlers subscribed to this event type
        event_type = event.event_type.value
        handler_ids = self.subscriptions.get(event_type, set())
        
        for handler_id in handler_ids:
            handler = self.handlers.get(handler_id)
            if handler and handler.pattern.matches(event):
                matching_handlers.append(handler)
                
        return matching_handlers
        
    async def _handle_correlation(self, event: Event) -> None:
        """Handle event correlation"""
        correlation_id = event.correlation_id
        
        if correlation_id not in self.correlations:
            self.correlations[correlation_id] = EventCorrelation(
                correlation_id=correlation_id,
                pattern=event.metadata.get("correlation_pattern", "")
            )
            
        correlation = self.correlations[correlation_id]
        correlation.add_event(event)
        
        # Check if correlation is complete
        required_count = event.metadata.get("correlation_required_count")
        if correlation.is_complete(required_count):
            # Emit correlation completion event
            completion_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM,
                source="event_processor",
                data={
                    "correlation_id": correlation_id,
                    "event_count": len(correlation.events),
                    "duration": (correlation.window_end - correlation.window_start).total_seconds()
                },
                timestamp=datetime.now(),
                priority=EventPriority.HIGH
            )
            await self.emit(completion_event)
            
    async def _cleanup_expired_events(self) -> None:
        """Clean up expired events and correlations"""
        while self.processing:
            try:
                now = datetime.now()
                
                # Clean up expired correlations
                expired_correlations = []
                for correlation_id, correlation in self.correlations.items():
                    if correlation.window_start:
                        age = now - correlation.window_start
                        if age > timedelta(hours=1):  # 1 hour timeout
                            expired_correlations.append(correlation_id)
                            
                for correlation_id in expired_correlations:
                    del self.correlations[correlation_id]
                    
                if expired_correlations:
                    logger.info(f"Cleaned up {len(expired_correlations)} expired correlations")
                    
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
                await asyncio.sleep(60)
                
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        return {
            "events_emitted": self.metrics["events_emitted"],
            "events_processed": self.metrics["events_processed"],
            "events_failed": self.metrics["events_failed"],
            "queue_size": self.event_queue.qsize(),
            "handlers_registered": len(self.handlers),
            "active_correlations": len(self.correlations),
            "dead_letter_queue_size": len(self.dead_letter_queue)
        }
        
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        events = list(self.event_history)[-limit:]
        return [event.to_dict() for event in events]
        
    async def replay_event(self, event_id: str) -> bool:
        """Replay a specific event"""
        # Find event in history
        for event in self.event_history:
            if event.event_id == event_id:
                # Reset event status and retry count
                event.status = EventStatus.PENDING
                event.retry_count = 0
                
                # Re-emit the event
                await self.emit(event)
                logger.info(f"Replayed event {event_id}")
                return True
                
        logger.warning(f"Event {event_id} not found in history")
        return False
        
    def create_event_pattern(self, name: str, event_type: EventType,
                           conditions: Dict[str, Any] = None,
                           regex_patterns: Dict[str, str] = None) -> EventPattern:
        """Create a new event pattern"""
        pattern = EventPattern(
            pattern_id=str(uuid.uuid4()),
            name=name,
            event_type=event_type,
            conditions=conditions or {},
            regex_patterns=regex_patterns or {}
        )
        self.patterns[pattern.pattern_id] = pattern
        return pattern
        
    async def emit_batch(self, events: List[Event]) -> List[str]:
        """Emit multiple events in batch"""
        event_ids = []
        for event in events:
            event_id = await self.emit(event)
            event_ids.append(event_id)
        return event_ids
        
    def enable_handler(self, handler_id: str) -> None:
        """Enable a handler"""
        if handler_id in self.handlers:
            self.handlers[handler_id].enabled = True
            
    def disable_handler(self, handler_id: str) -> None:
        """Disable a handler"""
        if handler_id in self.handlers:
            self.handlers[handler_id].enabled = False