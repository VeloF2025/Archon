#!/usr/bin/env python3
"""
Kafka API Routes
Provides endpoints for monitoring and interacting with Kafka messaging system
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..services.kafka_integration_service import get_kafka_service, EventData
from ..config.logfire_config import api_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kafka", tags=["kafka"])


@router.get("/status")
async def get_kafka_status():
    """Get Kafka integration service status"""
    try:
        kafka_service = get_kafka_service()
        status = kafka_service.get_status()
        
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        logger.error(f"Error getting Kafka status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics")
async def list_kafka_topics():
    """List active Kafka topic subscriptions"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.messaging_system:
            return {"status": "error", "message": "Kafka service not initialized"}
        
        subscriptions = kafka_service.messaging_system.list_subscriptions()
        
        return {
            "status": "success",
            "data": {
                "subscriptions": subscriptions,
                "total": len(subscriptions)
            }
        }
    except Exception as e:
        logger.error(f"Error listing Kafka topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish/system-event")
async def publish_system_event(
    event_type: str,
    data: Dict[str, Any],
    priority: str = "normal"
):
    """Publish a system event to Kafka"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        # Map priority string to enum
        from ..services.kafka_integration_service import MessagePriority
        priority_map = {
            "critical": MessagePriority.CRITICAL,
            "high": MessagePriority.HIGH,
            "normal": MessagePriority.NORMAL,
            "low": MessagePriority.LOW,
            "background": MessagePriority.BACKGROUND
        }
        
        priority_enum = priority_map.get(priority, MessagePriority.NORMAL)
        
        success = await kafka_service.publish_system_event(
            event_type=event_type,
            data=data,
            priority=priority_enum
        )
        
        if success:
            return {
                "status": "success",
                "message": f"System event '{event_type}' published successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish event")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing system event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish/agent-command")
async def publish_agent_command(
    agent_id: str,
    command: str,
    params: Optional[Dict[str, Any]] = None
):
    """Publish an agent command to Kafka"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        success = await kafka_service.publish_agent_command(
            agent_id=agent_id,
            command=command,
            params=params or {}
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Command '{command}' sent to agent '{agent_id}'"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish command")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing agent command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish/analytics-event")
async def publish_analytics_event(
    metric_type: str,
    value: Any,
    tags: Optional[Dict[str, str]] = None
):
    """Publish an analytics event to Kafka"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        success = await kafka_service.publish_analytics_event(
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Analytics event '{metric_type}' published"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish analytics event")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing analytics event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish/stream-data")
async def publish_stream_data(
    stream_id: str,
    data: Any,
    stream_type: str = "default"
):
    """Publish live streaming data to Kafka"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        success = await kafka_service.publish_live_stream_data(
            stream_id=stream_id,
            data=data,
            stream_type=stream_type
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Stream data published to '{stream_id}'"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish stream data")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing stream data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert")
async def publish_system_alert(
    alert_type: str,
    message: str,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None
):
    """Publish a system alert to Kafka"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        if severity not in ["info", "warning", "error", "critical"]:
            raise HTTPException(status_code=400, detail="Invalid severity level")
        
        success = await kafka_service.publish_system_alert(
            alert_type=alert_type,
            message=message,
            severity=severity,
            details=details or {}
        )
        
        if success:
            return {
                "status": "success",
                "message": f"System alert '{alert_type}' published"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish alert")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing system alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_kafka_metrics():
    """Get Kafka system metrics"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.messaging_system:
            return {"status": "error", "message": "Kafka service not initialized"}
        
        backend_status = kafka_service.messaging_system.get_backend_status()
        system_metrics = kafka_service.messaging_system.get_system_metrics()
        
        return {
            "status": "success",
            "data": {
                "backend_status": backend_status,
                "system_metrics": {
                    "messages_sent": system_metrics.messages_sent,
                    "messages_received": system_metrics.messages_received,
                    "messages_failed": system_metrics.messages_failed,
                    "messages_retried": system_metrics.messages_retried,
                    "connections_active": system_metrics.connections_active,
                    "connections_failed": system_metrics.connections_failed,
                    "average_latency": system_metrics.average_latency,
                    "throughput_per_second": system_metrics.throughput_per_second,
                    "last_updated": system_metrics.last_updated.isoformat() if system_metrics.last_updated else None
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting Kafka metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-publish")
async def test_kafka_publish():
    """Test Kafka publishing functionality"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        # Publish a test event
        test_data = {
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "message": "Kafka integration test"
        }
        
        success = await kafka_service.publish_system_event(
            event_type="kafka_test",
            data=test_data
        )
        
        if success:
            return {
                "status": "success",
                "message": "Test event published successfully",
                "data": test_data
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to publish test event")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing Kafka publish: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscribe-test")
async def test_kafka_subscription():
    """Test Kafka subscription functionality"""
    try:
        kafka_service = get_kafka_service()
        
        if not kafka_service.is_initialized:
            raise HTTPException(status_code=503, detail="Kafka service not initialized")
        
        # Track received messages for testing
        received_messages = []
        
        def test_handler(event_data: EventData):
            received_messages.append({
                "event_id": event_data.event_id,
                "event_type": event_data.event_type,
                "source": event_data.source,
                "timestamp": event_data.timestamp.isoformat(),
                "data": event_data.data
            })
        
        # Register test handler
        kafka_service.register_event_handler("kafka_test", test_handler)
        
        return {
            "status": "success",
            "message": "Test subscription handler registered for 'kafka_test' events"
        }
        
    except Exception as e:
        logger.error(f"Error setting up Kafka subscription test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def kafka_health_check():
    """Check Kafka service health"""
    try:
        kafka_service = get_kafka_service()
        
        health_data = {
            "service": "kafka_integration",
            "initialized": kafka_service.is_initialized,
            "messaging_system": kafka_service.messaging_system is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if kafka_service.messaging_system:
            backend_status = kafka_service.messaging_system.get_backend_status()
            health_data["backends"] = {
                name: status["connected"] for name, status in backend_status.items()
            }
        
        return {
            "status": "success" if kafka_service.is_initialized else "warning",
            "data": health_data
        }
        
    except Exception as e:
        logger.error(f"Error checking Kafka health: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }