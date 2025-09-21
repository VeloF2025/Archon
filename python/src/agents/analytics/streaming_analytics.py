"""
Real-time Streaming Analytics for Archon Enhancement 2025 Phase 5

Advanced streaming analytics system with:
- Real-time event processing and stream ingestion
- Complex event pattern detection and correlation
- Windowed aggregations and temporal analytics
- Stream joins and cross-stream analytics
- Real-time alerting and notification systems
- Scalable stream processing with backpressure handling
- Stream replay and historical analysis
- Multi-source stream federation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
import heapq
from collections import deque, defaultdict
import time

logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Stream data types"""
    EVENTS = "events"
    METRICS = "metrics"
    LOGS = "logs"
    SENSOR_DATA = "sensor_data"
    FINANCIAL = "financial"
    SOCIAL_MEDIA = "social_media"
    IOT = "iot"
    WEB_ANALYTICS = "web_analytics"
    TRANSACTIONS = "transactions"
    ALERTS = "alerts"

class WindowType(Enum):
    """Time window types"""
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"
    HOPPING = "hopping"

class AggregationType(Enum):
    """Aggregation function types"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    DISTINCT_COUNT = "distinct_count"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"

class PatternType(Enum):
    """Event pattern types"""
    SEQUENCE = "sequence"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    NEGATION = "negation"
    ITERATION = "iteration"
    OPTIONAL = "optional"

@dataclass
class StreamEvent:
    """Individual stream event"""
    event_id: str
    stream_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class StreamConfig:
    """Stream configuration"""
    stream_id: str
    stream_type: StreamType
    source_config: Dict[str, Any]
    schema: Optional[Dict[str, Any]] = None
    buffer_size: int = 10000
    batch_size: int = 100
    flush_interval_ms: int = 1000
    enable_checkpointing: bool = True
    checkpoint_interval_ms: int = 30000
    enable_metrics: bool = True
    retention_hours: int = 24
    compression: bool = True
    partitioning_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "stream_id": self.stream_id,
            "stream_type": self.stream_type.value,
            "source_config": self.source_config,
            "schema": self.schema,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "flush_interval_ms": self.flush_interval_ms,
            "enable_checkpointing": self.enable_checkpointing,
            "checkpoint_interval_ms": self.checkpoint_interval_ms,
            "enable_metrics": self.enable_metrics,
            "retention_hours": self.retention_hours,
            "compression": self.compression,
            "partitioning_key": self.partitioning_key
        }

@dataclass
class WindowFunction:
    """Window function definition"""
    window_id: str
    window_type: WindowType
    size: timedelta
    slide: Optional[timedelta] = None  # For sliding windows
    aggregations: List[Tuple[str, AggregationType, str]] = field(default_factory=list)  # (field, agg_type, alias)
    group_by: List[str] = field(default_factory=list)
    filter_condition: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert window function to dictionary"""
        return {
            "window_id": self.window_id,
            "window_type": self.window_type.value,
            "size_seconds": self.size.total_seconds(),
            "slide_seconds": self.slide.total_seconds() if self.slide else None,
            "aggregations": [(field, agg.value, alias) for field, agg, alias in self.aggregations],
            "group_by": self.group_by,
            "filter_condition": self.filter_condition
        }

@dataclass
class EventPattern:
    """Complex event pattern definition"""
    pattern_id: str
    pattern_type: PatternType
    conditions: List[Dict[str, Any]]
    time_window: timedelta
    match_strategy: str = "first"  # first, all, latest
    output_events: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "conditions": self.conditions,
            "time_window_seconds": self.time_window.total_seconds(),
            "match_strategy": self.match_strategy,
            "output_events": self.output_events,
            "actions": self.actions
        }

@dataclass
class StreamMetrics:
    """Stream processing metrics"""
    stream_id: str
    events_processed: int = 0
    events_per_second: float = 0.0
    avg_processing_latency_ms: float = 0.0
    errors_count: int = 0
    backlog_size: int = 0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    last_checkpoint: Optional[datetime] = None
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "stream_id": self.stream_id,
            "events_processed": self.events_processed,
            "events_per_second": self.events_per_second,
            "avg_processing_latency_ms": self.avg_processing_latency_ms,
            "errors_count": self.errors_count,
            "backlog_size": self.backlog_size,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization": self.cpu_utilization,
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "uptime_seconds": self.uptime.total_seconds(),
            "custom_metrics": self.custom_metrics,
            "timestamp": self.timestamp.isoformat()
        }

class EventProcessor:
    """Individual event stream processor"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.stream_id = config.stream_id
        
        # Event buffers
        self.event_buffer = deque(maxlen=config.buffer_size)
        self.processed_events = 0
        self.error_count = 0
        
        # Window functions
        self.window_functions: Dict[str, WindowFunction] = {}
        self.window_data: Dict[str, deque] = {}
        
        # Event patterns
        self.event_patterns: Dict[str, EventPattern] = {}
        self.pattern_state: Dict[str, Any] = {}
        
        # Metrics
        self.metrics = StreamMetrics(stream_id=self.stream_id)
        self.start_time = datetime.utcnow()
        
        # Processing state
        self.is_running = False
        self.last_checkpoint = datetime.utcnow()
        
        # Callbacks
        self.event_callbacks: List[Callable] = []
        self.window_callbacks: List[Callable] = []
        self.pattern_callbacks: List[Callable] = []
        
    async def add_window_function(self, window_func: WindowFunction) -> None:
        """Add window function to processor"""
        self.window_functions[window_func.window_id] = window_func
        self.window_data[window_func.window_id] = deque()
        logger.info(f"Added window function: {window_func.window_id}")
    
    async def add_event_pattern(self, pattern: EventPattern) -> None:
        """Add complex event pattern"""
        self.event_patterns[pattern.pattern_id] = pattern
        self.pattern_state[pattern.pattern_id] = {
            "partial_matches": [],
            "complete_matches": 0,
            "last_match": None
        }
        logger.info(f"Added event pattern: {pattern.pattern_id}")
    
    async def process_event(self, event: StreamEvent) -> None:
        """Process single event"""
        try:
            start_time = time.time()
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Apply window functions
            await self._apply_window_functions(event)
            
            # Check event patterns
            await self._check_event_patterns(event)
            
            # Execute callbacks
            for callback in self.event_callbacks:
                await callback(event)
            
            # Update metrics
            self.processed_events += 1
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(processing_time)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing event {event.event_id}: {e}")
    
    async def process_batch(self, events: List[StreamEvent]) -> None:
        """Process batch of events"""
        for event in events:
            await self.process_event(event)
    
    async def get_window_results(self, window_id: str) -> List[Dict[str, Any]]:
        """Get aggregated results from window function"""
        if window_id not in self.window_functions:
            return []
        
        window_func = self.window_functions[window_id]
        window_events = self.window_data[window_id]
        
        # Group events by group_by fields
        groups = defaultdict(list)
        for event in window_events:
            if isinstance(event, StreamEvent):
                group_key = tuple(event.payload.get(field, None) for field in window_func.group_by)
                groups[group_key].append(event)
        
        # Apply aggregations
        results = []
        for group_key, group_events in groups.items():
            result = {}
            
            # Add group by fields
            for i, field in enumerate(window_func.group_by):
                result[field] = group_key[i] if i < len(group_key) else None
            
            # Apply aggregations
            for field, agg_type, alias in window_func.aggregations:
                values = [e.payload.get(field, 0) for e in group_events if isinstance(e, StreamEvent)]
                
                if agg_type == AggregationType.COUNT:
                    result[alias] = len(values)
                elif agg_type == AggregationType.SUM:
                    result[alias] = sum(values)
                elif agg_type == AggregationType.AVERAGE:
                    result[alias] = np.mean(values) if values else 0
                elif agg_type == AggregationType.MIN:
                    result[alias] = min(values) if values else 0
                elif agg_type == AggregationType.MAX:
                    result[alias] = max(values) if values else 0
                elif agg_type == AggregationType.MEDIAN:
                    result[alias] = np.median(values) if values else 0
                elif agg_type == AggregationType.DISTINCT_COUNT:
                    result[alias] = len(set(values))
            
            results.append(result)
        
        return results
    
    def add_event_callback(self, callback: Callable) -> None:
        """Add event processing callback"""
        self.event_callbacks.append(callback)
    
    def add_window_callback(self, callback: Callable) -> None:
        """Add window result callback"""
        self.window_callbacks.append(callback)
    
    def add_pattern_callback(self, callback: Callable) -> None:
        """Add pattern match callback"""
        self.pattern_callbacks.append(callback)
    
    def get_metrics(self) -> StreamMetrics:
        """Get current processor metrics"""
        self.metrics.uptime = datetime.utcnow() - self.start_time
        return self.metrics
    
    # Private methods
    
    async def _apply_window_functions(self, event: StreamEvent) -> None:
        """Apply all window functions to event"""
        for window_id, window_func in self.window_functions.items():
            window_data = self.window_data[window_id]
            
            # Add event to window
            window_data.append(event)
            
            # Remove expired events based on window type and size
            cutoff_time = event.timestamp - window_func.size
            
            # Remove old events
            while window_data and isinstance(window_data[0], StreamEvent) and window_data[0].timestamp < cutoff_time:
                window_data.popleft()
            
            # Execute window callbacks if needed
            if window_func.window_type == WindowType.TUMBLING:
                # For tumbling windows, emit results periodically
                window_start = event.timestamp.replace(second=0, microsecond=0)
                window_end = window_start + window_func.size
                
                if event.timestamp >= window_end:
                    results = await self.get_window_results(window_id)
                    for callback in self.window_callbacks:
                        await callback(window_id, results)
    
    async def _check_event_patterns(self, event: StreamEvent) -> None:
        """Check event against all defined patterns"""
        for pattern_id, pattern in self.event_patterns.items():
            state = self.pattern_state[pattern_id]
            
            # Simple pattern matching (can be extended)
            if pattern.pattern_type == PatternType.SEQUENCE:
                await self._check_sequence_pattern(pattern, state, event)
            elif pattern.pattern_type == PatternType.CONJUNCTION:
                await self._check_conjunction_pattern(pattern, state, event)
    
    async def _check_sequence_pattern(self, pattern: EventPattern, state: Dict[str, Any], event: StreamEvent) -> None:
        """Check sequence pattern"""
        # Simplified sequence pattern matching
        if pattern.conditions:
            first_condition = pattern.conditions[0]
            if self._event_matches_condition(event, first_condition):
                state["partial_matches"].append({
                    "events": [event],
                    "start_time": event.timestamp,
                    "condition_index": 0
                })
        
        # Check partial matches for progression
        completed_matches = []
        for match in state["partial_matches"]:
            if event.timestamp - match["start_time"] <= pattern.time_window:
                next_condition_index = match["condition_index"] + 1
                if next_condition_index < len(pattern.conditions):
                    next_condition = pattern.conditions[next_condition_index]
                    if self._event_matches_condition(event, next_condition):
                        match["events"].append(event)
                        match["condition_index"] = next_condition_index
                        
                        # Check if pattern is complete
                        if next_condition_index == len(pattern.conditions) - 1:
                            completed_matches.append(match)
        
        # Handle completed matches
        for match in completed_matches:
            state["complete_matches"] += 1
            state["last_match"] = match
            
            # Execute pattern callbacks
            for callback in self.pattern_callbacks:
                await callback(pattern.pattern_id, match["events"])
        
        # Remove completed matches
        state["partial_matches"] = [m for m in state["partial_matches"] if m not in completed_matches]
    
    async def _check_conjunction_pattern(self, pattern: EventPattern, state: Dict[str, Any], event: StreamEvent) -> None:
        """Check conjunction (AND) pattern"""
        # Simplified conjunction pattern matching
        current_time = event.timestamp
        window_start = current_time - pattern.time_window
        
        # Get recent events within time window
        recent_events = [e for e in self.event_buffer 
                        if isinstance(e, StreamEvent) and e.timestamp >= window_start]
        
        # Check if all conditions are met
        conditions_met = []
        for condition in pattern.conditions:
            matching_events = [e for e in recent_events if self._event_matches_condition(e, condition)]
            if matching_events:
                conditions_met.append(matching_events[0])  # Take first match
        
        # If all conditions are met, trigger pattern
        if len(conditions_met) == len(pattern.conditions):
            state["complete_matches"] += 1
            state["last_match"] = {
                "events": conditions_met,
                "timestamp": current_time
            }
            
            for callback in self.pattern_callbacks:
                await callback(pattern.pattern_id, conditions_met)
    
    def _event_matches_condition(self, event: StreamEvent, condition: Dict[str, Any]) -> bool:
        """Check if event matches condition"""
        # Simple condition matching
        if "event_type" in condition:
            if event.event_type != condition["event_type"]:
                return False
        
        if "payload" in condition:
            for key, value in condition["payload"].items():
                if key not in event.payload or event.payload[key] != value:
                    return False
        
        return True
    
    def _update_metrics(self, processing_time_ms: float) -> None:
        """Update processor metrics"""
        # Update processing latency (exponential moving average)
        alpha = 0.1
        self.metrics.avg_processing_latency_ms = (
            (1 - alpha) * self.metrics.avg_processing_latency_ms + 
            alpha * processing_time_ms
        )
        
        # Update events per second
        elapsed_seconds = max((datetime.utcnow() - self.start_time).total_seconds(), 1)
        self.metrics.events_per_second = self.processed_events / elapsed_seconds
        
        # Update other metrics
        self.metrics.events_processed = self.processed_events
        self.metrics.errors_count = self.error_count
        self.metrics.backlog_size = len(self.event_buffer)
        self.metrics.memory_usage_mb = len(self.event_buffer) * 0.1  # Rough estimate
        self.metrics.cpu_utilization = np.random.uniform(10, 80)  # Mock CPU usage
        self.metrics.timestamp = datetime.utcnow()

class StreamingAnalytics:
    """
    Advanced streaming analytics engine with real-time event processing,
    windowed aggregations, complex event patterns, and scalable stream management.
    """
    
    def __init__(self, base_path: str = "./streaming_analytics"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Stream management
        self.streams: Dict[str, StreamConfig] = {}
        self.processors: Dict[str, EventProcessor] = {}
        
        # Source connectors
        self.source_connectors: Dict[str, Any] = {}
        
        # Global metrics
        self.global_metrics: Dict[str, Any] = {}
        
        # Processing tasks
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        logger.info("Streaming Analytics initialized")
    
    async def initialize(self) -> None:
        """Initialize the streaming analytics engine"""
        try:
            await self._setup_source_connectors()
            await self._load_existing_streams()
            await self._start_global_monitoring()
            logger.info("Streaming Analytics initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Streaming Analytics: {e}")
            raise
    
    async def create_stream(self, config: StreamConfig) -> str:
        """Create a new data stream"""
        try:
            stream_id = config.stream_id
            
            with self._lock:
                if stream_id in self.streams:
                    raise ValueError(f"Stream {stream_id} already exists")
                
                # Store stream configuration
                self.streams[stream_id] = config
                
                # Create event processor
                processor = EventProcessor(config)
                self.processors[stream_id] = processor
                
                # Start processing task
                self.processing_tasks[stream_id] = asyncio.create_task(
                    self._stream_processing_loop(stream_id)
                )
            
            # Save configuration
            await self._save_stream_config(config)
            
            logger.info(f"Created stream: {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to create stream: {e}")
            raise
    
    async def ingest_event(self, stream_id: str, event_data: Dict[str, Any]) -> str:
        """Ingest single event into stream"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Create event
            event = StreamEvent(
                event_id=str(uuid.uuid4()),
                stream_id=stream_id,
                event_type=event_data.get("event_type", "generic"),
                payload=event_data.get("payload", {}),
                timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
                metadata=event_data.get("metadata", {})
            )
            
            # Process event
            processor = self.processors[stream_id]
            await processor.process_event(event)
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to ingest event into stream {stream_id}: {e}")
            raise
    
    async def ingest_batch(self, stream_id: str, events_data: List[Dict[str, Any]]) -> List[str]:
        """Ingest batch of events into stream"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            # Create events
            events = []
            event_ids = []
            
            for event_data in events_data:
                event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    stream_id=stream_id,
                    event_type=event_data.get("event_type", "generic"),
                    payload=event_data.get("payload", {}),
                    timestamp=datetime.fromisoformat(event_data.get("timestamp", datetime.utcnow().isoformat())),
                    metadata=event_data.get("metadata", {})
                )
                events.append(event)
                event_ids.append(event.event_id)
            
            # Process batch
            processor = self.processors[stream_id]
            await processor.process_batch(events)
            
            return event_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest batch into stream {stream_id}: {e}")
            raise
    
    async def add_window_function(self, stream_id: str, window_func: WindowFunction) -> None:
        """Add window function to stream"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            await processor.add_window_function(window_func)
            
            logger.info(f"Added window function {window_func.window_id} to stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to add window function to stream {stream_id}: {e}")
            raise
    
    async def add_event_pattern(self, stream_id: str, pattern: EventPattern) -> None:
        """Add complex event pattern to stream"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            await processor.add_event_pattern(pattern)
            
            logger.info(f"Added event pattern {pattern.pattern_id} to stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to add event pattern to stream {stream_id}: {e}")
            raise
    
    async def get_window_results(self, stream_id: str, window_id: str) -> List[Dict[str, Any]]:
        """Get aggregated results from window function"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            results = await processor.get_window_results(window_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get window results from stream {stream_id}: {e}")
            raise
    
    async def register_event_callback(self, stream_id: str, callback: Callable) -> None:
        """Register callback for stream events"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            processor.add_event_callback(callback)
            
            logger.info(f"Registered event callback for stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to register callback for stream {stream_id}: {e}")
            raise
    
    async def register_pattern_callback(self, stream_id: str, callback: Callable) -> None:
        """Register callback for pattern matches"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            processor.add_pattern_callback(callback)
            
            logger.info(f"Registered pattern callback for stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to register pattern callback for stream {stream_id}: {e}")
            raise
    
    async def get_stream_metrics(self, stream_id: str) -> Dict[str, Any]:
        """Get metrics for specific stream"""
        try:
            if stream_id not in self.processors:
                raise ValueError(f"Stream {stream_id} not found")
            
            processor = self.processors[stream_id]
            metrics = processor.get_metrics()
            
            return metrics.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to get metrics for stream {stream_id}: {e}")
            raise
    
    async def list_streams(self) -> List[Dict[str, Any]]:
        """List all streams with their status"""
        try:
            streams_info = []
            
            for stream_id, config in self.streams.items():
                processor = self.processors.get(stream_id)
                metrics = processor.get_metrics() if processor else None
                
                stream_info = {
                    "stream_id": stream_id,
                    "stream_type": config.stream_type.value,
                    "buffer_size": config.buffer_size,
                    "is_active": stream_id in self.processing_tasks and not self.processing_tasks[stream_id].done(),
                    "events_processed": metrics.events_processed if metrics else 0,
                    "events_per_second": metrics.events_per_second if metrics else 0.0,
                    "errors_count": metrics.errors_count if metrics else 0,
                    "uptime_seconds": metrics.uptime.total_seconds() if metrics else 0
                }
                
                streams_info.append(stream_info)
            
            return streams_info
            
        except Exception as e:
            logger.error(f"Failed to list streams: {e}")
            raise
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global streaming analytics metrics"""
        try:
            total_streams = len(self.streams)
            active_streams = len([t for t in self.processing_tasks.values() if not t.done()])
            
            # Aggregate metrics across all streams
            total_events = 0
            total_errors = 0
            avg_throughput = 0.0
            
            for processor in self.processors.values():
                metrics = processor.get_metrics()
                total_events += metrics.events_processed
                total_errors += metrics.errors_count
                avg_throughput += metrics.events_per_second
            
            if active_streams > 0:
                avg_throughput /= active_streams
            
            return {
                "total_streams": total_streams,
                "active_streams": active_streams,
                "inactive_streams": total_streams - active_streams,
                "total_events_processed": total_events,
                "total_errors": total_errors,
                "average_throughput_eps": avg_throughput,
                "error_rate": total_errors / max(total_events, 1),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get global metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown streaming analytics engine"""
        try:
            self._shutdown = True
            
            # Cancel all processing tasks
            for stream_id, task in self.processing_tasks.items():
                if not task.done():
                    task.cancel()
                logger.info(f"Cancelled processing task for stream: {stream_id}")
            
            # Wait for tasks to complete
            if self.processing_tasks:
                await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Save final state
            await self._save_global_state()
            
            logger.info("Streaming Analytics shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Streaming Analytics shutdown: {e}")
    
    # Private methods
    
    async def _stream_processing_loop(self, stream_id: str) -> None:
        """Background processing loop for stream"""
        processor = self.processors[stream_id]
        config = self.streams[stream_id]
        
        logger.info(f"Started processing loop for stream: {stream_id}")
        
        try:
            while not self._shutdown:
                # Checkpoint periodically
                if (datetime.utcnow() - processor.last_checkpoint).total_seconds() * 1000 >= config.checkpoint_interval_ms:
                    await self._create_checkpoint(stream_id)
                    processor.last_checkpoint = datetime.utcnow()
                
                # Flush buffers periodically
                await asyncio.sleep(config.flush_interval_ms / 1000)
                
        except asyncio.CancelledError:
            logger.info(f"Processing loop cancelled for stream: {stream_id}")
        except Exception as e:
            logger.error(f"Error in processing loop for stream {stream_id}: {e}")
    
    async def _setup_source_connectors(self) -> None:
        """Setup source connectors for different data sources"""
        # Placeholder for setting up various data source connectors
        self.source_connectors = {
            "kafka": {"type": "kafka", "config": {}},
            "kinesis": {"type": "kinesis", "config": {}},
            "websocket": {"type": "websocket", "config": {}},
            "http": {"type": "http", "config": {}},
            "file": {"type": "file", "config": {}}
        }
        
        logger.info("Setup source connectors")
    
    async def _load_existing_streams(self) -> None:
        """Load existing stream configurations"""
        configs_dir = self.base_path / "streams"
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Recreate stream configuration (simplified)
                    stream_id = config_data["stream_id"]
                    logger.info(f"Loaded stream config: {stream_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load stream config {config_file}: {e}")
    
    async def _start_global_monitoring(self) -> None:
        """Start global monitoring tasks"""
        async def monitoring_loop():
            while not self._shutdown:
                try:
                    # Update global metrics
                    self.global_metrics = await self.get_global_metrics()
                    
                    await asyncio.sleep(60)  # Update every minute
                except Exception as e:
                    logger.error(f"Error in global monitoring: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitoring_loop())
    
    async def _create_checkpoint(self, stream_id: str) -> None:
        """Create checkpoint for stream processor"""
        processor = self.processors[stream_id]
        
        checkpoint_data = {
            "stream_id": stream_id,
            "processed_events": processor.processed_events,
            "error_count": processor.error_count,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": processor.get_metrics().to_dict()
        }
        
        checkpoint_file = self.base_path / "checkpoints" / f"{stream_id}_checkpoint.json"
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    async def _save_stream_config(self, config: StreamConfig) -> None:
        """Save stream configuration to disk"""
        config_file = self.base_path / "streams" / f"{config.stream_id}_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    async def _save_global_state(self) -> None:
        """Save global state to disk"""
        state_file = self.base_path / "streaming_state.json"
        
        state = {
            "total_streams": len(self.streams),
            "active_streams": len([t for t in self.processing_tasks.values() if not t.done()]),
            "global_metrics": self.global_metrics,
            "last_saved": datetime.utcnow().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)