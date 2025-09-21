"""
Real-Time Streaming Service - Kafka Integration
Bridges Kafka messaging with live data streams and Socket.IO for real-time updates
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import uuid

from ..config.logfire_config import get_logger
from .kafka_integration_service import get_kafka_service, EventData
from ..socketio_app import get_socketio_instance
from ...agents.analytics.streaming_analytics import (
    StreamingAnalytics, StreamConfig, StreamType, StreamEvent,
    WindowFunction, WindowType, AggregationType, EventPattern, PatternType
)

logger = get_logger(__name__)

@dataclass
class LiveStreamConfig:
    """Configuration for live streaming data"""
    stream_name: str
    kafka_topic: str
    stream_type: StreamType
    aggregation_window_seconds: int = 60
    enable_pattern_detection: bool = True
    socket_room: Optional[str] = None
    buffer_size: int = 1000
    batch_size: int = 50
    real_time_threshold_ms: int = 100

@dataclass
class StreamMetrics:
    """Real-time stream metrics"""
    stream_name: str
    events_per_second: float = 0.0
    total_events: int = 0
    avg_processing_latency_ms: float = 0.0
    error_count: int = 0
    last_event_timestamp: Optional[datetime] = None
    uptime_seconds: float = 0.0

class RealTimeStreamingService:
    """
    Real-time streaming service that connects Kafka messaging with live data flows,
    Socket.IO broadcasts, and analytics processing
    """
    
    def __init__(self):
        self.kafka_service = get_kafka_service()
        self.socketio = get_socketio_instance()
        self.streaming_analytics = StreamingAnalytics()
        
        # Active streams management
        self.live_streams: Dict[str, LiveStreamConfig] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self.stream_metrics: Dict[str, StreamMetrics] = {}
        
        # Real-time callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        self.pattern_callbacks: Dict[str, List[Callable]] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Real-Time Streaming Service initialized")
    
    async def initialize(self) -> None:
        """Initialize the streaming service"""
        try:
            # Initialize streaming analytics
            await self.streaming_analytics.initialize()
            
            # Set up standard event streams
            await self._setup_standard_streams()
            
            # Start monitoring
            await self._start_monitoring()
            
            self.is_running = True
            logger.info("✅ Real-Time Streaming Service fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Real-Time Streaming Service: {e}")
            raise
    
    async def create_live_stream(
        self, 
        stream_name: str, 
        kafka_topic: str, 
        stream_type: StreamType,
        **kwargs
    ) -> str:
        """Create a new live data stream"""
        try:
            # Create stream configuration
            config = LiveStreamConfig(
                stream_name=stream_name,
                kafka_topic=kafka_topic,
                stream_type=stream_type,
                **kwargs
            )
            
            # Store configuration
            self.live_streams[stream_name] = config
            self.stream_metrics[stream_name] = StreamMetrics(stream_name=stream_name)
            
            # Create streaming analytics stream
            analytics_config = StreamConfig(
                stream_id=stream_name,
                stream_type=stream_type,
                source_config={
                    "type": "kafka",
                    "topic": kafka_topic,
                    "real_time": True
                },
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                enable_metrics=True
            )
            
            await self.streaming_analytics.create_stream(analytics_config)
            
            # Set up aggregation window
            if config.aggregation_window_seconds > 0:
                window_func = WindowFunction(
                    window_id=f"{stream_name}_aggregation",
                    window_type=WindowType.TUMBLING,
                    size=timedelta(seconds=config.aggregation_window_seconds),
                    aggregations=[
                        ("value", AggregationType.COUNT, "event_count"),
                        ("value", AggregationType.AVERAGE, "avg_value"),
                        ("value", AggregationType.MAX, "max_value"),
                        ("value", AggregationType.MIN, "min_value")
                    ]
                )
                
                await self.streaming_analytics.add_window_function(stream_name, window_func)
            
            # Set up pattern detection if enabled
            if config.enable_pattern_detection:
                await self._setup_pattern_detection(stream_name, config)
            
            # Start stream processing task
            self.stream_tasks[stream_name] = asyncio.create_task(
                self._stream_processing_loop(stream_name, config)
            )
            
            # Register Kafka event handler
            await self._register_kafka_handler(stream_name, config)
            
            logger.info(f"✅ Created live stream: {stream_name} -> {kafka_topic}")
            return stream_name
            
        except Exception as e:
            logger.error(f"❌ Failed to create live stream {stream_name}: {e}")
            raise
    
    async def publish_to_stream(
        self, 
        stream_name: str, 
        data: Dict[str, Any],
        event_type: str = "data"
    ) -> str:
        """Publish data to a live stream"""
        try:
            if stream_name not in self.live_streams:
                raise ValueError(f"Stream {stream_name} not found")
            
            config = self.live_streams[stream_name]
            
            # Create event data
            event_data = {
                "event_id": str(uuid.uuid4()),
                "stream_name": stream_name,
                "event_type": event_type,
                "payload": data,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "source": "streaming_service",
                    "stream_type": config.stream_type.value
                }
            }
            
            # Publish to Kafka
            success = await self.kafka_service.publish_live_stream_data(
                stream_id=stream_name,
                data=event_data,
                stream_type=config.stream_type.value
            )
            
            if success:
                # Update metrics
                metrics = self.stream_metrics[stream_name]
                metrics.total_events += 1
                metrics.last_event_timestamp = datetime.utcnow()
                
                logger.debug(f"📤 Published to stream {stream_name}: {event_type}")
                return event_data["event_id"]
            else:
                raise RuntimeError("Failed to publish to Kafka")
                
        except Exception as e:
            logger.error(f"❌ Failed to publish to stream {stream_name}: {e}")
            # Update error metrics
            if stream_name in self.stream_metrics:
                self.stream_metrics[stream_name].error_count += 1
            raise
    
    async def subscribe_to_stream(
        self, 
        stream_name: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to real-time events from a stream"""
        try:
            if stream_name not in self.event_callbacks:
                self.event_callbacks[stream_name] = []
            
            self.event_callbacks[stream_name].append(callback)
            logger.info(f"📥 Added callback to stream {stream_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to stream {stream_name}: {e}")
            raise
    
    async def get_stream_metrics(self, stream_name: str) -> Dict[str, Any]:
        """Get real-time metrics for a stream"""
        try:
            if stream_name not in self.stream_metrics:
                return {"error": f"Stream {stream_name} not found"}
            
            metrics = self.stream_metrics[stream_name]
            analytics_metrics = await self.streaming_analytics.get_stream_metrics(stream_name)
            
            # Combine metrics
            combined_metrics = {
                "stream_name": stream_name,
                "real_time_metrics": {
                    "events_per_second": metrics.events_per_second,
                    "total_events": metrics.total_events,
                    "avg_processing_latency_ms": metrics.avg_processing_latency_ms,
                    "error_count": metrics.error_count,
                    "uptime_seconds": metrics.uptime_seconds,
                    "last_event": metrics.last_event_timestamp.isoformat() if metrics.last_event_timestamp else None
                },
                "analytics_metrics": analytics_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get metrics for stream {stream_name}: {e}")
            return {"error": str(e)}
    
    async def list_active_streams(self) -> List[Dict[str, Any]]:
        """List all active streams with their status"""
        try:
            streams = []
            
            for stream_name, config in self.live_streams.items():
                metrics = self.stream_metrics[stream_name]
                task = self.stream_tasks.get(stream_name)
                
                stream_info = {
                    "stream_name": stream_name,
                    "kafka_topic": config.kafka_topic,
                    "stream_type": config.stream_type.value,
                    "is_active": task is not None and not task.done(),
                    "total_events": metrics.total_events,
                    "events_per_second": metrics.events_per_second,
                    "error_count": metrics.error_count,
                    "uptime_seconds": metrics.uptime_seconds,
                    "socket_room": config.socket_room,
                    "aggregation_window": config.aggregation_window_seconds
                }
                
                streams.append(stream_info)
            
            return streams
            
        except Exception as e:
            logger.error(f"❌ Failed to list active streams: {e}")
            return []
    
    async def shutdown(self) -> None:
        """Shutdown the streaming service"""
        try:
            self.is_running = False
            
            # Cancel all stream tasks
            for stream_name, task in self.stream_tasks.items():
                if not task.done():
                    task.cancel()
                logger.info(f"Cancelled stream task: {stream_name}")
            
            # Wait for tasks to complete
            if self.stream_tasks:
                await asyncio.gather(*self.stream_tasks.values(), return_exceptions=True)
            
            # Cancel monitoring task
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
            
            # Shutdown streaming analytics
            await self.streaming_analytics.shutdown()
            
            logger.info("✅ Real-Time Streaming Service shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during streaming service shutdown: {e}")
    
    # Private methods
    
    async def _setup_standard_streams(self) -> None:
        """Set up standard system streams"""
        try:
            # System events stream
            await self.create_live_stream(
                stream_name="system_events",
                kafka_topic="archon.system.events",
                stream_type=StreamType.EVENTS,
                socket_room="system_events",
                aggregation_window_seconds=30
            )
            
            # Agent communication stream
            await self.create_live_stream(
                stream_name="agent_communications",
                kafka_topic="archon.agents.communication",
                stream_type=StreamType.EVENTS,
                socket_room="agent_comms",
                aggregation_window_seconds=60
            )
            
            # Analytics stream
            await self.create_live_stream(
                stream_name="analytics_data",
                kafka_topic="archon.analytics.metrics",
                stream_type=StreamType.METRICS,
                socket_room="analytics",
                aggregation_window_seconds=60
            )
            
            # Performance metrics stream
            await self.create_live_stream(
                stream_name="performance_metrics",
                kafka_topic="archon.system.performance",
                stream_type=StreamType.METRICS,
                socket_room="performance",
                aggregation_window_seconds=30
            )
            
            logger.info("✅ Standard streams configured")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup standard streams: {e}")
            raise
    
    async def _setup_pattern_detection(self, stream_name: str, config: LiveStreamConfig) -> None:
        """Set up event pattern detection for a stream"""
        try:
            # High-frequency events pattern
            high_freq_pattern = EventPattern(
                pattern_id=f"{stream_name}_high_frequency",
                pattern_type=PatternType.SEQUENCE,
                conditions=[
                    {"event_type": "data", "payload": {"frequency": "high"}},
                    {"event_type": "data", "payload": {"frequency": "high"}},
                    {"event_type": "data", "payload": {"frequency": "high"}}
                ],
                time_window=timedelta(seconds=10),
                actions=[{"type": "alert", "severity": "warning", "message": "High frequency events detected"}]
            )
            
            await self.streaming_analytics.add_event_pattern(stream_name, high_freq_pattern)
            
            # Error burst pattern
            error_burst_pattern = EventPattern(
                pattern_id=f"{stream_name}_error_burst",
                pattern_type=PatternType.CONJUNCTION,
                conditions=[
                    {"event_type": "error"},
                    {"event_type": "error"},
                    {"event_type": "error"}
                ],
                time_window=timedelta(seconds=30),
                actions=[{"type": "alert", "severity": "critical", "message": "Error burst detected"}]
            )
            
            await self.streaming_analytics.add_event_pattern(stream_name, error_burst_pattern)
            
            # Register pattern callbacks
            await self.streaming_analytics.register_pattern_callback(
                stream_name, 
                lambda pattern_id, events: asyncio.create_task(
                    self._handle_pattern_match(stream_name, pattern_id, events)
                )
            )
            
            logger.info(f"✅ Pattern detection configured for {stream_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup pattern detection for {stream_name}: {e}")
    
    async def _register_kafka_handler(self, stream_name: str, config: LiveStreamConfig) -> None:
        """Register Kafka event handler for the stream"""
        try:
            async def kafka_event_handler(event_data: EventData):
                """Handle incoming Kafka events"""
                try:
                    # Convert to StreamEvent
                    stream_event = StreamEvent(
                        event_id=event_data.event_id,
                        stream_id=stream_name,
                        event_type=event_data.event_type,
                        payload=event_data.data,
                        timestamp=event_data.timestamp,
                        metadata={"source": "kafka", "topic": config.kafka_topic}
                    )
                    
                    # Ingest into streaming analytics
                    await self.streaming_analytics.ingest_event(
                        stream_name,
                        {
                            "event_type": stream_event.event_type,
                            "payload": stream_event.payload,
                            "timestamp": stream_event.timestamp.isoformat(),
                            "metadata": stream_event.metadata
                        }
                    )
                    
                    # Execute callbacks
                    if stream_name in self.event_callbacks:
                        event_dict = stream_event.to_dict()
                        for callback in self.event_callbacks[stream_name]:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(event_dict)
                                else:
                                    callback(event_dict)
                            except Exception as cb_error:
                                logger.error(f"❌ Callback error for {stream_name}: {cb_error}")
                    
                    # Broadcast to Socket.IO if configured
                    if config.socket_room:
                        await self.socketio.emit(
                            "stream_data",
                            {
                                "stream_name": stream_name,
                                "event": stream_event.to_dict(),
                                "timestamp": datetime.utcnow().isoformat()
                            },
                            room=config.socket_room
                        )
                    
                    # Update metrics
                    metrics = self.stream_metrics[stream_name]
                    metrics.total_events += 1
                    metrics.last_event_timestamp = stream_event.timestamp
                    
                except Exception as e:
                    logger.error(f"❌ Error handling Kafka event for {stream_name}: {e}")
                    if stream_name in self.stream_metrics:
                        self.stream_metrics[stream_name].error_count += 1
            
            # Register handler with Kafka service
            self.kafka_service.register_event_handler(config.kafka_topic, kafka_event_handler)
            
            logger.info(f"✅ Kafka handler registered for {stream_name} -> {config.kafka_topic}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register Kafka handler for {stream_name}: {e}")
    
    async def _stream_processing_loop(self, stream_name: str, config: LiveStreamConfig) -> None:
        """Background processing loop for a stream"""
        logger.info(f"🔄 Started processing loop for stream: {stream_name}")
        
        try:
            start_time = datetime.utcnow()
            metrics = self.stream_metrics[stream_name]
            
            while self.is_running:
                try:
                    # Update uptime
                    metrics.uptime_seconds = (datetime.utcnow() - start_time).total_seconds()
                    
                    # Calculate events per second
                    if metrics.uptime_seconds > 0:
                        metrics.events_per_second = metrics.total_events / metrics.uptime_seconds
                    
                    # Get window results if aggregation is enabled
                    if config.aggregation_window_seconds > 0:
                        try:
                            window_results = await self.streaming_analytics.get_window_results(
                                stream_name, 
                                f"{stream_name}_aggregation"
                            )
                            
                            if window_results and config.socket_room:
                                await self.socketio.emit(
                                    "stream_aggregation",
                                    {
                                        "stream_name": stream_name,
                                        "window_results": window_results,
                                        "timestamp": datetime.utcnow().isoformat()
                                    },
                                    room=config.socket_room
                                )
                        except Exception as window_error:
                            logger.debug(f"Window results error for {stream_name}: {window_error}")
                    
                    # Sleep based on real-time threshold
                    await asyncio.sleep(config.real_time_threshold_ms / 1000)
                    
                except Exception as e:
                    logger.error(f"❌ Error in processing loop for {stream_name}: {e}")
                    metrics.error_count += 1
                    await asyncio.sleep(1)  # Prevent rapid error loops
            
        except asyncio.CancelledError:
            logger.info(f"🔄 Processing loop cancelled for stream: {stream_name}")
        except Exception as e:
            logger.error(f"❌ Fatal error in processing loop for {stream_name}: {e}")
    
    async def _handle_pattern_match(self, stream_name: str, pattern_id: str, events: List[StreamEvent]) -> None:
        """Handle pattern match detection"""
        try:
            pattern_data = {
                "stream_name": stream_name,
                "pattern_id": pattern_id,
                "matched_events": [event.to_dict() for event in events],
                "timestamp": datetime.utcnow().isoformat(),
                "event_count": len(events)
            }
            
            # Broadcast pattern match
            config = self.live_streams[stream_name]
            if config.socket_room:
                await self.socketio.emit(
                    "pattern_detected",
                    pattern_data,
                    room=config.socket_room
                )
            
            # Execute pattern callbacks
            if stream_name in self.pattern_callbacks:
                for callback in self.pattern_callbacks[stream_name]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(pattern_data)
                        else:
                            callback(pattern_data)
                    except Exception as cb_error:
                        logger.error(f"❌ Pattern callback error for {stream_name}: {cb_error}")
            
            # Publish pattern alert to Kafka
            await self.kafka_service.publish_system_alert(
                alert_type="pattern_detected",
                message=f"Pattern {pattern_id} detected in stream {stream_name}",
                severity="info",
                details=pattern_data
            )
            
            logger.info(f"🔍 Pattern detected in {stream_name}: {pattern_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to handle pattern match for {stream_name}: {e}")
    
    async def _start_monitoring(self) -> None:
        """Start monitoring task"""
        async def monitoring_loop():
            while self.is_running:
                try:
                    # Update global metrics
                    total_events = sum(metrics.total_events for metrics in self.stream_metrics.values())
                    total_errors = sum(metrics.error_count for metrics in self.stream_metrics.values())
                    avg_eps = sum(metrics.events_per_second for metrics in self.stream_metrics.values())
                    
                    if len(self.stream_metrics) > 0:
                        avg_eps /= len(self.stream_metrics)
                    
                    # Broadcast system metrics
                    system_metrics = {
                        "total_streams": len(self.live_streams),
                        "total_events": total_events,
                        "total_errors": total_errors,
                        "average_events_per_second": avg_eps,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await self.socketio.emit(
                        "system_metrics",
                        system_metrics,
                        room="system_metrics"
                    )
                    
                    # Publish to Kafka analytics
                    await self.kafka_service.publish_analytics_event(
                        metric_type="streaming_system_metrics",
                        value=system_metrics,
                        tags={"service": "real_time_streaming"}
                    )
                    
                    await asyncio.sleep(30)  # Update every 30 seconds
                    
                except Exception as e:
                    logger.error(f"❌ Error in monitoring loop: {e}")
                    await asyncio.sleep(30)
        
        self.monitoring_task = asyncio.create_task(monitoring_loop())

# Global instance
_streaming_service: Optional[RealTimeStreamingService] = None

def get_streaming_service() -> RealTimeStreamingService:
    """Get or create the streaming service instance"""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = RealTimeStreamingService()
    return _streaming_service

async def initialize_streaming_service() -> RealTimeStreamingService:
    """Initialize the streaming service"""
    service = get_streaming_service()
    await service.initialize()
    return service