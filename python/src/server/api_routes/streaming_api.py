"""
Real-Time Streaming API Routes
Provides endpoints for managing live data streams and real-time analytics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..config.logfire_config import api_logger
from ..services.real_time_streaming_service import get_streaming_service, LiveStreamConfig
from ...agents.analytics.streaming_analytics import StreamType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/streaming", tags=["streaming"])

@router.get("/status")
async def get_streaming_status():
    """Get real-time streaming service status"""
    try:
        streaming_service = get_streaming_service()
        
        if not streaming_service.is_running:
            return {
                "status": "inactive",
                "message": "Streaming service not running",
                "active_streams": 0
            }
        
        streams = await streaming_service.list_active_streams()
        
        return {
            "status": "active",
            "is_running": streaming_service.is_running,
            "active_streams": len(streams),
            "streams": streams,
            "kafka_connected": streaming_service.kafka_service.is_initialized,
            "analytics_initialized": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/streams/create")
async def create_stream(
    stream_name: str,
    kafka_topic: str,
    stream_type: str,
    aggregation_window_seconds: int = 60,
    enable_pattern_detection: bool = True,
    socket_room: Optional[str] = None,
    buffer_size: int = 1000,
    batch_size: int = 50
):
    """Create a new live data stream"""
    try:
        streaming_service = get_streaming_service()
        
        # Validate stream type
        try:
            stream_type_enum = StreamType(stream_type.lower())
        except ValueError:
            valid_types = [t.value for t in StreamType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stream_type. Must be one of: {valid_types}"
            )
        
        # Create the stream
        created_stream_name = await streaming_service.create_live_stream(
            stream_name=stream_name,
            kafka_topic=kafka_topic,
            stream_type=stream_type_enum,
            aggregation_window_seconds=aggregation_window_seconds,
            enable_pattern_detection=enable_pattern_detection,
            socket_room=socket_room,
            buffer_size=buffer_size,
            batch_size=batch_size
        )
        
        return {
            "status": "success",
            "stream_name": created_stream_name,
            "kafka_topic": kafka_topic,
            "stream_type": stream_type,
            "configuration": {
                "aggregation_window_seconds": aggregation_window_seconds,
                "enable_pattern_detection": enable_pattern_detection,
                "socket_room": socket_room,
                "buffer_size": buffer_size,
                "batch_size": batch_size
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating stream {stream_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/streams/{stream_name}/publish")
async def publish_to_stream(
    stream_name: str,
    data: Dict[str, Any],
    event_type: str = "data"
):
    """Publish data to a live stream"""
    try:
        streaming_service = get_streaming_service()
        
        event_id = await streaming_service.publish_to_stream(
            stream_name=stream_name,
            data=data,
            event_type=event_type
        )
        
        return {
            "status": "success",
            "event_id": event_id,
            "stream_name": stream_name,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error publishing to stream {stream_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/streams")
async def list_streams():
    """List all active streams"""
    try:
        streaming_service = get_streaming_service()
        streams = await streaming_service.list_active_streams()
        
        return {
            "status": "success",
            "streams": streams,
            "total_streams": len(streams),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/streams/{stream_name}/metrics")
async def get_stream_metrics(stream_name: str):
    """Get real-time metrics for a specific stream"""
    try:
        streaming_service = get_streaming_service()
        metrics = await streaming_service.get_stream_metrics(stream_name)
        
        if "error" in metrics:
            raise HTTPException(status_code=404, detail=metrics["error"])
        
        return {
            "status": "success",
            "data": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics for stream {stream_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/streams/{stream_name}/subscribe")
async def subscribe_to_stream_api(
    stream_name: str,
    callback_url: Optional[str] = None
):
    """Subscribe to stream events via API (webhook-based)"""
    try:
        # Note: This is a simplified API subscription
        # In a full implementation, you'd store webhook URLs and call them
        # For now, we'll just acknowledge the subscription
        
        streaming_service = get_streaming_service()
        
        # Check if stream exists
        metrics = await streaming_service.get_stream_metrics(stream_name)
        if "error" in metrics:
            raise HTTPException(status_code=404, detail=f"Stream {stream_name} not found")
        
        # For now, just return success
        # In production, you'd store the callback_url and set up webhook calls
        
        return {
            "status": "success",
            "stream_name": stream_name,
            "subscription_type": "webhook" if callback_url else "socket_io",
            "callback_url": callback_url,
            "message": "Use Socket.IO for real-time subscriptions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing to stream {stream_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analytics/create-window")
async def create_analytics_window(
    stream_name: str,
    window_id: str,
    window_type: str,
    size_seconds: int,
    slide_seconds: Optional[int] = None,
    aggregations: List[Dict[str, str]] = [],
    group_by: List[str] = []
):
    """Create an analytics window function for a stream"""
    try:
        from ...agents.analytics.streaming_analytics import WindowFunction, WindowType, AggregationType
        from datetime import timedelta
        
        streaming_service = get_streaming_service()
        
        # Validate window type
        try:
            window_type_enum = WindowType(window_type.lower())
        except ValueError:
            valid_types = [t.value for t in WindowType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid window_type. Must be one of: {valid_types}"
            )
        
        # Parse aggregations
        parsed_aggregations = []
        for agg in aggregations:
            try:
                field = agg["field"]
                agg_type = AggregationType(agg["type"].upper())
                alias = agg.get("alias", f"{agg['type']}_{field}")
                parsed_aggregations.append((field, agg_type, alias))
            except (KeyError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid aggregation specification: {agg}. Error: {e}"
                )
        
        # Create window function
        window_func = WindowFunction(
            window_id=window_id,
            window_type=window_type_enum,
            size=timedelta(seconds=size_seconds),
            slide=timedelta(seconds=slide_seconds) if slide_seconds else None,
            aggregations=parsed_aggregations,
            group_by=group_by
        )
        
        # Add to stream
        await streaming_service.streaming_analytics.add_window_function(stream_name, window_func)
        
        return {
            "status": "success",
            "window_id": window_id,
            "stream_name": stream_name,
            "window_type": window_type,
            "configuration": window_func.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating analytics window: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/{stream_name}/window/{window_id}/results")
async def get_window_results(stream_name: str, window_id: str):
    """Get aggregated results from a window function"""
    try:
        streaming_service = get_streaming_service()
        
        results = await streaming_service.streaming_analytics.get_window_results(
            stream_name, window_id
        )
        
        return {
            "status": "success",
            "stream_name": stream_name,
            "window_id": window_id,
            "results": results,
            "result_count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting window results for {stream_name}/{window_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/patterns/create")
async def create_event_pattern(
    stream_name: str,
    pattern_id: str,
    pattern_type: str,
    conditions: List[Dict[str, Any]],
    time_window_seconds: int,
    match_strategy: str = "first",
    actions: List[Dict[str, Any]] = []
):
    """Create a complex event pattern for detection"""
    try:
        from ...agents.analytics.streaming_analytics import EventPattern, PatternType
        from datetime import timedelta
        
        streaming_service = get_streaming_service()
        
        # Validate pattern type
        try:
            pattern_type_enum = PatternType(pattern_type.upper())
        except ValueError:
            valid_types = [t.value for t in PatternType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pattern_type. Must be one of: {valid_types}"
            )
        
        # Create event pattern
        pattern = EventPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type_enum,
            conditions=conditions,
            time_window=timedelta(seconds=time_window_seconds),
            match_strategy=match_strategy,
            actions=actions
        )
        
        # Add to stream
        await streaming_service.streaming_analytics.add_event_pattern(stream_name, pattern)
        
        return {
            "status": "success",
            "pattern_id": pattern_id,
            "stream_name": stream_name,
            "pattern_type": pattern_type,
            "configuration": pattern.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating event pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def streaming_health_check():
    """Check streaming service health"""
    try:
        streaming_service = get_streaming_service()
        
        health_data = {
            "service": "real_time_streaming",
            "is_running": streaming_service.is_running,
            "kafka_initialized": streaming_service.kafka_service.is_initialized,
            "analytics_initialized": True,
            "active_streams": len(streaming_service.live_streams),
            "monitoring_active": streaming_service.monitoring_task is not None and not streaming_service.monitoring_task.done(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get Kafka health
        if streaming_service.kafka_service.messaging_system:
            backend_status = streaming_service.kafka_service.messaging_system.get_backend_status()
            health_data["kafka_backends"] = {
                name: status["connected"] for name, status in backend_status.items()
            }
        
        return {
            "status": "success" if streaming_service.is_running else "warning",
            "data": health_data
        }
        
    except Exception as e:
        logger.error(f"Error checking streaming health: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/test/publish-sample")
async def publish_sample_data():
    """Publish sample data to test streams"""
    try:
        streaming_service = get_streaming_service()
        
        # Sample data for different stream types
        samples = [
            {
                "stream_name": "system_events",
                "data": {
                    "event_type": "system_startup",
                    "component": "streaming_service",
                    "status": "active",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "event_type": "system_event"
            },
            {
                "stream_name": "analytics_data",
                "data": {
                    "metric": "cpu_usage",
                    "value": 45.2,
                    "unit": "percent",
                    "host": "archon-server"
                },
                "event_type": "metric"
            },
            {
                "stream_name": "performance_metrics",
                "data": {
                    "endpoint": "/api/streaming/status",
                    "response_time_ms": 15,
                    "status_code": 200,
                    "method": "GET"
                },
                "event_type": "performance"
            }
        ]
        
        results = []
        for sample in samples:
            try:
                event_id = await streaming_service.publish_to_stream(
                    stream_name=sample["stream_name"],
                    data=sample["data"],
                    event_type=sample["event_type"]
                )
                results.append({
                    "stream_name": sample["stream_name"],
                    "event_id": event_id,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "stream_name": sample["stream_name"],
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "success",
            "message": "Sample data published to available streams",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error publishing sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))