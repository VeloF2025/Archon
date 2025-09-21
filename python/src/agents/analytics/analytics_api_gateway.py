"""
Analytics API Gateway
Advanced Analytics & Intelligence Platform - Archon Enhancement 2025 Phase 5

Unified API gateway for all analytics services with:
- RESTful API endpoints for all analytics components
- Real-time WebSocket streaming for live analytics
- Authentication and authorization management
- Rate limiting and request throttling
- API versioning and backward compatibility
- Request/response transformation and validation
- Analytics service orchestration and routing
- Comprehensive API documentation and OpenAPI specs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import time
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class RequestType(Enum):
    ANALYTICS = "analytics"
    STREAMING = "streaming"
    PREDICTION = "prediction"
    ANOMALY = "anomaly"
    GRAPH = "graph"
    TIME_SERIES = "time_series"


# Request/Response Models
class BaseRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    client_id: Optional[str] = None


class BaseResponse(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool
    message: str = ""
    execution_time_ms: float = 0.0


class ErrorResponse(BaseResponse):
    error_code: str
    error_details: Dict[str, Any] = Field(default_factory=dict)


# Analytics Service Requests
class StreamingAnalyticsRequest(BaseRequest):
    data: List[Dict[str, Any]]
    stream_config: Dict[str, Any]
    window_size: int = 100
    aggregations: List[str] = Field(default_factory=list)


class PredictiveAnalyticsRequest(BaseRequest):
    model_name: str
    input_data: List[Dict[str, Any]]
    include_confidence: bool = True
    include_explanations: bool = False


class AnomalyDetectionRequest(BaseRequest):
    data: List[Dict[str, Any]]
    detector_ids: Optional[List[str]] = None
    threshold_override: Optional[float] = None
    sensitivity: str = "medium"  # low, medium, high


class GraphAnalyticsRequest(BaseRequest):
    graph_data: Optional[Dict[str, Any]] = None
    graph_id: Optional[str] = None
    operation: str  # analyze, centrality, community, path
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TimeSeriesAnalyticsRequest(BaseRequest):
    series_data: List[Dict[str, Any]]
    series_id: Optional[str] = None
    operations: List[str] = Field(default_factory=lambda: ["decompose", "detect_anomalies"])
    frequency: str = "auto"


class BusinessIntelligenceRequest(BaseRequest):
    dashboard_id: Optional[str] = None
    query: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregation_level: str = "day"


# Response Models
class StreamingAnalyticsResponse(BaseResponse):
    stream_metrics: Dict[str, float]
    processed_events: int
    window_results: List[Dict[str, Any]]


class PredictiveAnalyticsResponse(BaseResponse):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    confidence_scores: Optional[List[float]] = None
    explanations: Optional[List[Dict[str, Any]]] = None


class AnomalyDetectionResponse(BaseResponse):
    anomalies_found: int
    anomalies: List[Dict[str, Any]]
    detection_summary: Dict[str, Any]
    alerts_triggered: int


class GraphAnalyticsResponse(BaseResponse):
    graph_metrics: Dict[str, Any]
    results: Dict[str, Any]
    visualization_data: Optional[Dict[str, Any]] = None


class TimeSeriesAnalyticsResponse(BaseResponse):
    series_metrics: Dict[str, Any]
    analysis_results: Dict[str, Any]
    forecasts: Optional[List[Dict[str, Any]]] = None


class BusinessIntelligenceResponse(BaseResponse):
    dashboard_data: Optional[Dict[str, Any]] = None
    query_results: Optional[List[Dict[str, Any]]] = None
    kpi_summary: Dict[str, Any]
    charts: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_size_minutes: int = 1


@dataclass
class APIKeyInfo:
    """API key information"""
    key_id: str
    client_name: str
    permissions: List[str]
    rate_limit: RateLimitConfig
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class RequestMetrics:
    """Request metrics tracking"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    timestamp: datetime
    client_id: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class ServiceHealth:
    """Service health information"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time_ms: float
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    dependencies: List[str] = field(default_factory=list)


class AuthenticationManager:
    """API authentication and authorization manager"""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKeyInfo] = {}
        self.blocked_keys: Set[str] = set()
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
    
    def create_api_key(self, client_name: str, permissions: List[str], 
                      rate_limit: Optional[RateLimitConfig] = None) -> str:
        """Create new API key"""
        key_id = f"ak_{uuid.uuid4().hex}"
        api_key = f"archon_analytics_{key_id}_{uuid.uuid4().hex[:16]}"
        
        key_info = APIKeyInfo(
            key_id=key_id,
            client_name=client_name,
            permissions=permissions,
            rate_limit=rate_limit or RateLimitConfig(),
            created_at=datetime.now()
        )
        
        self.api_keys[api_key] = key_info
        logger.info(f"Created API key for client: {client_name}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate API key and return key info"""
        if api_key in self.blocked_keys:
            return None
        
        key_info = self.api_keys.get(api_key)
        if not key_info or not key_info.is_active:
            return None
        
        # Check expiration
        if key_info.expires_at and datetime.now() > key_info.expires_at:
            return None
        
        return key_info
    
    def check_rate_limit(self, api_key: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return False
        
        now = datetime.now()
        rate_limit = key_info.rate_limit
        
        # Get request history for this key
        history = self.request_history[api_key]
        
        # Clean old requests
        cutoff_minute = now - timedelta(minutes=rate_limit.window_size_minutes)
        cutoff_hour = now - timedelta(hours=1)
        
        recent_requests = [req_time for req_time in history if req_time > cutoff_minute]
        hourly_requests = [req_time for req_time in history if req_time > cutoff_hour]
        
        # Check limits
        if len(recent_requests) >= rate_limit.requests_per_minute:
            return False
        
        if len(hourly_requests) >= rate_limit.requests_per_hour:
            return False
        
        # Check burst limit
        last_second_requests = [req_time for req_time in history if req_time > now - timedelta(seconds=1)]
        if len(last_second_requests) >= rate_limit.burst_limit:
            return False
        
        # Record this request
        history.append(now)
        
        # Keep only recent history
        self.request_history[api_key] = [req_time for req_time in history if req_time > cutoff_hour]
        
        return True
    
    def block_api_key(self, api_key: str, reason: str):
        """Block API key"""
        self.blocked_keys.add(api_key)
        logger.warning(f"Blocked API key: {api_key}, reason: {reason}")
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission"""
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return False
        
        return permission in key_info.permissions or "admin" in key_info.permissions


class MetricsCollector:
    """API metrics collection and monitoring"""
    
    def __init__(self):
        self.request_metrics: deque = deque(maxlen=10000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_requests': 0,
            'total_time': 0.0,
            'error_count': 0,
            'last_request': None
        })
        self.service_health: Dict[str, ServiceHealth] = {}
    
    def record_request(self, metrics: RequestMetrics):
        """Record request metrics"""
        self.request_metrics.append(metrics)
        
        # Update endpoint stats
        endpoint_key = f"{metrics.method}_{metrics.endpoint}"
        stats = self.endpoint_stats[endpoint_key]
        stats['total_requests'] += 1
        stats['total_time'] += metrics.response_time_ms
        stats['last_request'] = metrics.timestamp
        
        if metrics.status_code >= 400:
            stats['error_count'] += 1
            self.error_counts[metrics.error_type or 'unknown'] += 1
    
    def get_endpoint_metrics(self, endpoint: str, method: str = "GET") -> Dict[str, Any]:
        """Get metrics for specific endpoint"""
        endpoint_key = f"{method}_{endpoint}"
        stats = self.endpoint_stats[endpoint_key]
        
        if stats['total_requests'] == 0:
            return {'message': 'No requests recorded'}
        
        avg_response_time = stats['total_time'] / stats['total_requests']
        error_rate = stats['error_count'] / stats['total_requests']
        
        return {
            'endpoint': endpoint,
            'method': method,
            'total_requests': stats['total_requests'],
            'average_response_time_ms': avg_response_time,
            'error_rate': error_rate,
            'last_request': stats['last_request'].isoformat() if stats['last_request'] else None
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        if not self.request_metrics:
            return {'message': 'No metrics available'}
        
        recent_requests = [m for m in self.request_metrics 
                          if m.timestamp > datetime.now() - timedelta(minutes=5)]
        
        if not recent_requests:
            return {'message': 'No recent requests'}
        
        total_requests = len(recent_requests)
        avg_response_time = sum(m.response_time_ms for m in recent_requests) / total_requests
        error_count = sum(1 for m in recent_requests if m.status_code >= 400)
        error_rate = error_count / total_requests
        
        # Requests per minute
        requests_per_minute = len([m for m in recent_requests 
                                 if m.timestamp > datetime.now() - timedelta(minutes=1)])
        
        return {
            'requests_last_5_minutes': total_requests,
            'requests_per_minute': requests_per_minute,
            'average_response_time_ms': avg_response_time,
            'error_rate': error_rate,
            'top_errors': dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def update_service_health(self, service_name: str, status: ServiceStatus, 
                            response_time_ms: float, error_rate: float = 0.0):
        """Update service health status"""
        if service_name not in self.service_health:
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time_ms,
                error_rate=error_rate
            )
        else:
            health = self.service_health[service_name]
            health.status = status
            health.last_check = datetime.now()
            health.response_time_ms = response_time_ms
            health.error_rate = error_rate
    
    def get_service_health_report(self) -> Dict[str, Any]:
        """Get comprehensive service health report"""
        if not self.service_health:
            return {'message': 'No service health data available'}
        
        healthy_services = sum(1 for h in self.service_health.values() if h.status == ServiceStatus.HEALTHY)
        total_services = len(self.service_health)
        
        return {
            'overall_health': 'healthy' if healthy_services == total_services else 'degraded',
            'healthy_services': healthy_services,
            'total_services': total_services,
            'services': {
                name: {
                    'status': health.status.value,
                    'response_time_ms': health.response_time_ms,
                    'error_rate': health.error_rate,
                    'uptime_percentage': health.uptime_percentage,
                    'last_check': health.last_check.isoformat()
                }
                for name, health in self.service_health.items()
            }
        }


class WebSocketManager:
    """WebSocket connection manager for real-time analytics"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, client_id: str, metadata: Dict[str, Any] = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = metadata or {}
        
        logger.info(f"WebSocket client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            
            # Remove subscriptions
            for subscription_type in self.subscriptions:
                self.subscriptions[subscription_type].discard(client_id)
            
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    def subscribe(self, client_id: str, subscription_type: str):
        """Subscribe client to specific data stream"""
        if client_id in self.active_connections:
            self.subscriptions[subscription_type].add(client_id)
            logger.info(f"Client {client_id} subscribed to {subscription_type}")
    
    def unsubscribe(self, client_id: str, subscription_type: str):
        """Unsubscribe client from data stream"""
        self.subscriptions[subscription_type].discard(client_id)
        logger.info(f"Client {client_id} unsubscribed from {subscription_type}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_subscribers(self, message: Dict[str, Any], subscription_type: str):
        """Broadcast message to all subscribers of a type"""
        subscribers = self.subscriptions[subscription_type].copy()
        
        for client_id in subscribers:
            await self.send_personal_message(message, client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        subscription_stats = {
            sub_type: len(clients) for sub_type, clients in self.subscriptions.items()
        }
        
        return {
            'active_connections': len(self.active_connections),
            'subscriptions': subscription_stats,
            'total_subscriptions': sum(subscription_stats.values())
        }


class AnalyticsAPIGateway:
    """Main Analytics API Gateway orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="Archon Analytics API Gateway",
            description="Unified API gateway for advanced analytics services",
            version="1.0.0"
        )
        
        # Initialize managers
        self.auth_manager = AuthenticationManager()
        self.metrics_collector = MetricsCollector()
        self.websocket_manager = WebSocketManager()
        
        # Service references (would be injected in production)
        self.analytics_services: Dict[str, Any] = {}
        
        # Security
        self.security = HTTPBearer()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Analytics API Gateway initialized")
    
    async def initialize(self):
        """Initialize API Gateway"""
        try:
            await self._setup_middleware()
            await self._setup_routes()
            await self._setup_analytics_services()
            await self._start_background_tasks()
            
            logger.info("Analytics API Gateway fully initialized")
            
        except Exception as e:
            logger.error(f"API Gateway initialization failed: {e}")
            raise
    
    async def _setup_middleware(self):
        """Setup middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('allowed_origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Record metrics
            client_id = getattr(request.state, 'client_id', None)
            
            metrics = RequestMetrics(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=process_time,
                timestamp=datetime.now(),
                client_id=client_id,
                error_type=None if response.status_code < 400 else 'client_error' if response.status_code < 500 else 'server_error'
            )
            
            self.metrics_collector.record_request(metrics)
            
            # Add response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = getattr(request.state, 'request_id', str(uuid.uuid4()))
            
            return response
    
    async def _setup_routes(self):
        """Setup API routes"""
        
        # Authentication dependency
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            api_key = credentials.credentials
            key_info = self.auth_manager.validate_api_key(api_key)
            
            if not key_info:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            # Check rate limits
            if not self.auth_manager.check_rate_limit(api_key, ""):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return key_info
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics(user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "metrics"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return self.metrics_collector.get_system_metrics()
        
        # Service health endpoint
        @self.app.get("/health/services")
        async def get_service_health(user: APIKeyInfo = Depends(get_current_user)):
            return self.metrics_collector.get_service_health_report()
        
        # Streaming Analytics endpoints
        @self.app.post("/v1/analytics/streaming", response_model=StreamingAnalyticsResponse)
        async def streaming_analytics(request: StreamingAnalyticsRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "streaming"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                # Process streaming analytics request
                result = await self._process_streaming_analytics(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return StreamingAnalyticsResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Streaming analytics completed successfully",
                    execution_time_ms=execution_time,
                    stream_metrics=result.get('metrics', {}),
                    processed_events=result.get('processed_events', 0),
                    window_results=result.get('window_results', [])
                )
                
            except Exception as e:
                logger.error(f"Streaming analytics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Predictive Analytics endpoints
        @self.app.post("/v1/analytics/predictive", response_model=PredictiveAnalyticsResponse)
        async def predictive_analytics(request: PredictiveAnalyticsRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "prediction"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                result = await self._process_predictive_analytics(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return PredictiveAnalyticsResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Predictive analytics completed successfully",
                    execution_time_ms=execution_time,
                    predictions=result.get('predictions', []),
                    model_info=result.get('model_info', {}),
                    confidence_scores=result.get('confidence_scores') if request.include_confidence else None,
                    explanations=result.get('explanations') if request.include_explanations else None
                )
                
            except Exception as e:
                logger.error(f"Predictive analytics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Anomaly Detection endpoints
        @self.app.post("/v1/analytics/anomaly", response_model=AnomalyDetectionResponse)
        async def anomaly_detection(request: AnomalyDetectionRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "anomaly"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                result = await self._process_anomaly_detection(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return AnomalyDetectionResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Anomaly detection completed successfully",
                    execution_time_ms=execution_time,
                    anomalies_found=result.get('anomalies_found', 0),
                    anomalies=result.get('anomalies', []),
                    detection_summary=result.get('detection_summary', {}),
                    alerts_triggered=result.get('alerts_triggered', 0)
                )
                
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Graph Analytics endpoints
        @self.app.post("/v1/analytics/graph", response_model=GraphAnalyticsResponse)
        async def graph_analytics(request: GraphAnalyticsRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "graph"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                result = await self._process_graph_analytics(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return GraphAnalyticsResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Graph analytics completed successfully",
                    execution_time_ms=execution_time,
                    graph_metrics=result.get('graph_metrics', {}),
                    results=result.get('results', {}),
                    visualization_data=result.get('visualization_data')
                )
                
            except Exception as e:
                logger.error(f"Graph analytics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Time Series Analytics endpoints
        @self.app.post("/v1/analytics/timeseries", response_model=TimeSeriesAnalyticsResponse)
        async def timeseries_analytics(request: TimeSeriesAnalyticsRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "timeseries"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                result = await self._process_timeseries_analytics(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return TimeSeriesAnalyticsResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Time series analytics completed successfully",
                    execution_time_ms=execution_time,
                    series_metrics=result.get('series_metrics', {}),
                    analysis_results=result.get('analysis_results', {}),
                    forecasts=result.get('forecasts')
                )
                
            except Exception as e:
                logger.error(f"Time series analytics failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Business Intelligence endpoints
        @self.app.post("/v1/analytics/bi", response_model=BusinessIntelligenceResponse)
        async def business_intelligence(request: BusinessIntelligenceRequest, user: APIKeyInfo = Depends(get_current_user)):
            if not self.auth_manager.has_permission(user.key_id, "bi"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            start_time = time.time()
            
            try:
                result = await self._process_business_intelligence(request)
                
                execution_time = (time.time() - start_time) * 1000
                
                return BusinessIntelligenceResponse(
                    request_id=request.request_id,
                    success=True,
                    message="Business intelligence analytics completed successfully",
                    execution_time_ms=execution_time,
                    dashboard_data=result.get('dashboard_data'),
                    query_results=result.get('query_results'),
                    kpi_summary=result.get('kpi_summary', {}),
                    charts=result.get('charts', [])
                )
                
            except Exception as e:
                logger.error(f"Business intelligence failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # WebSocket endpoint for real-time analytics
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await self.websocket_manager.connect(websocket, client_id)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle WebSocket messages
                    await self._handle_websocket_message(client_id, message)
                    
            except WebSocketDisconnect:
                self.websocket_manager.disconnect(client_id)
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
                self.websocket_manager.disconnect(client_id)
    
    async def _setup_analytics_services(self):
        """Setup analytics service connections"""
        # In production, these would be actual service instances
        self.analytics_services = {
            'streaming': None,  # StreamingAnalytics instance
            'predictive': None,  # PredictiveAnalytics instance
            'anomaly': None,    # AnomalyDetectionFramework instance
            'graph': None,      # GraphAnalytics instance
            'timeseries': None, # TimeSeriesAnalytics instance
            'bi': None          # BusinessIntelligence instance
        }
        
        # Update service health
        for service_name in self.analytics_services:
            self.metrics_collector.update_service_health(
                service_name, ServiceStatus.HEALTHY, 50.0
            )
        
        logger.info("Analytics services configured")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.background_tasks.append(
            asyncio.create_task(self._health_check_loop())
        )
        self.background_tasks.append(
            asyncio.create_task(self._metrics_aggregation_loop())
        )
        self.background_tasks.append(
            asyncio.create_task(self._websocket_heartbeat_loop())
        )
        
        logger.info("Background tasks started")
    
    # Service processing methods (placeholder implementations)
    async def _process_streaming_analytics(self, request: StreamingAnalyticsRequest) -> Dict[str, Any]:
        """Process streaming analytics request"""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate processing
        
        return {
            'metrics': {
                'events_per_second': 1000,
                'avg_latency_ms': 50,
                'throughput_mb_per_sec': 10.5
            },
            'processed_events': len(request.data),
            'window_results': [
                {'window_start': '2023-01-01T00:00:00', 'count': 100, 'sum': 5000},
                {'window_start': '2023-01-01T00:01:00', 'count': 120, 'sum': 6000}
            ]
        }
    
    async def _process_predictive_analytics(self, request: PredictiveAnalyticsRequest) -> Dict[str, Any]:
        """Process predictive analytics request"""
        # Placeholder implementation
        await asyncio.sleep(0.2)
        
        predictions = []
        for i, data_point in enumerate(request.input_data):
            predictions.append({
                'index': i,
                'prediction': np.random.random(),
                'confidence': np.random.random() if request.include_confidence else None
            })
        
        return {
            'predictions': predictions,
            'model_info': {
                'model_name': request.model_name,
                'version': '1.0.0',
                'accuracy': 0.95
            },
            'confidence_scores': [p['confidence'] for p in predictions if p['confidence'] is not None],
            'explanations': [{'feature_importance': {'feature1': 0.8, 'feature2': 0.2}}] * len(predictions) if request.include_explanations else None
        }
    
    async def _process_anomaly_detection(self, request: AnomalyDetectionRequest) -> Dict[str, Any]:
        """Process anomaly detection request"""
        # Placeholder implementation
        await asyncio.sleep(0.15)
        
        # Simulate finding some anomalies
        num_anomalies = max(0, len(request.data) // 20)  # ~5% anomalies
        
        anomalies = []
        for i in range(num_anomalies):
            anomalies.append({
                'index': i * 20,
                'score': np.random.uniform(0.8, 1.0),
                'severity': np.random.choice(['medium', 'high', 'critical']),
                'features': ['feature1', 'feature2']
            })
        
        return {
            'anomalies_found': num_anomalies,
            'anomalies': anomalies,
            'detection_summary': {
                'total_points': len(request.data),
                'anomaly_rate': num_anomalies / len(request.data),
                'detection_method': 'ensemble'
            },
            'alerts_triggered': len([a for a in anomalies if a['severity'] in ['high', 'critical']])
        }
    
    async def _process_graph_analytics(self, request: GraphAnalyticsRequest) -> Dict[str, Any]:
        """Process graph analytics request"""
        # Placeholder implementation
        await asyncio.sleep(0.3)
        
        return {
            'graph_metrics': {
                'nodes': 1000,
                'edges': 5000,
                'density': 0.01,
                'clustering_coefficient': 0.3,
                'average_path_length': 3.5
            },
            'results': {
                'operation': request.operation,
                'communities': 5 if request.operation == 'community' else None,
                'centrality_scores': {'node1': 0.8, 'node2': 0.6} if request.operation == 'centrality' else None
            },
            'visualization_data': {
                'nodes': [{'id': 'node1', 'value': 10}, {'id': 'node2', 'value': 8}],
                'edges': [{'source': 'node1', 'target': 'node2', 'weight': 1}]
            }
        }
    
    async def _process_timeseries_analytics(self, request: TimeSeriesAnalyticsRequest) -> Dict[str, Any]:
        """Process time series analytics request"""
        # Placeholder implementation
        await asyncio.sleep(0.25)
        
        return {
            'series_metrics': {
                'length': len(request.series_data),
                'frequency': request.frequency,
                'stationarity': 'stationary',
                'seasonality': 'weekly'
            },
            'analysis_results': {
                'trend_strength': 0.7,
                'seasonal_strength': 0.5,
                'anomalies_detected': 3,
                'change_points': 2
            },
            'forecasts': [
                {'timestamp': '2023-12-01T00:00:00', 'forecast': 100.5, 'lower': 95.0, 'upper': 106.0},
                {'timestamp': '2023-12-02T00:00:00', 'forecast': 102.3, 'lower': 97.0, 'upper': 108.0}
            ] if 'forecast' in request.operations else None
        }
    
    async def _process_business_intelligence(self, request: BusinessIntelligenceRequest) -> Dict[str, Any]:
        """Process business intelligence request"""
        # Placeholder implementation
        await asyncio.sleep(0.2)
        
        return {
            'dashboard_data': {
                'dashboard_id': request.dashboard_id,
                'widgets': [
                    {'type': 'kpi', 'title': 'Revenue', 'value': 1000000, 'change': '+5%'},
                    {'type': 'chart', 'title': 'Sales Trend', 'data': [100, 120, 110, 130]}
                ]
            } if request.dashboard_id else None,
            'query_results': [
                {'date': '2023-01-01', 'revenue': 10000, 'orders': 100},
                {'date': '2023-01-02', 'revenue': 12000, 'orders': 120}
            ] if request.query else None,
            'kpi_summary': {
                'revenue': {'value': 1000000, 'trend': 'up', 'change_percent': 5.2},
                'customers': {'value': 5000, 'trend': 'up', 'change_percent': 2.1},
                'orders': {'value': 15000, 'trend': 'stable', 'change_percent': 0.5}
            },
            'charts': [
                {'type': 'line', 'title': 'Revenue Trend', 'data': [100, 120, 110, 130, 140]},
                {'type': 'pie', 'title': 'Revenue by Category', 'data': [{'name': 'A', 'value': 40}, {'name': 'B', 'value': 60}]}
            ]
        }
    
    async def _handle_websocket_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = message.get('type', 'unknown')
        
        if message_type == 'subscribe':
            subscription_type = message.get('subscription')
            if subscription_type:
                self.websocket_manager.subscribe(client_id, subscription_type)
                await self.websocket_manager.send_personal_message(
                    {'type': 'subscription_confirmed', 'subscription': subscription_type},
                    client_id
                )
        
        elif message_type == 'unsubscribe':
            subscription_type = message.get('subscription')
            if subscription_type:
                self.websocket_manager.unsubscribe(client_id, subscription_type)
                await self.websocket_manager.send_personal_message(
                    {'type': 'unsubscription_confirmed', 'subscription': subscription_type},
                    client_id
                )
        
        elif message_type == 'ping':
            await self.websocket_manager.send_personal_message(
                {'type': 'pong', 'timestamp': datetime.now().isoformat()},
                client_id
            )
    
    # Background task methods
    async def _health_check_loop(self):
        """Background health check for services"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for service_name in self.analytics_services:
                    # Simulate health check
                    response_time = np.random.uniform(50, 200)
                    is_healthy = np.random.random() > 0.05  # 95% uptime
                    
                    status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.DEGRADED
                    error_rate = 0.0 if is_healthy else np.random.uniform(0.1, 0.2)
                    
                    self.metrics_collector.update_service_health(
                        service_name, status, response_time, error_rate
                    )
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _metrics_aggregation_loop(self):
        """Background metrics aggregation"""
        while True:
            try:
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
                # Aggregate and store metrics
                system_metrics = self.metrics_collector.get_system_metrics()
                
                # Broadcast to subscribers
                await self.websocket_manager.broadcast_to_subscribers(
                    {'type': 'metrics_update', 'data': system_metrics},
                    'metrics'
                )
                
            except Exception as e:
                logger.error(f"Metrics aggregation loop error: {e}")
    
    async def _websocket_heartbeat_loop(self):
        """Background WebSocket heartbeat"""
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                heartbeat_message = {
                    'type': 'heartbeat',
                    'timestamp': datetime.now().isoformat(),
                    'connections': len(self.websocket_manager.active_connections)
                }
                
                # Send to all connections
                for client_id in list(self.websocket_manager.active_connections.keys()):
                    await self.websocket_manager.send_personal_message(heartbeat_message, client_id)
                
            except Exception as e:
                logger.error(f"WebSocket heartbeat loop error: {e}")
    
    def create_api_key(self, client_name: str, permissions: List[str]) -> str:
        """Create new API key for client"""
        return self.auth_manager.create_api_key(client_name, permissions)
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the API Gateway server"""
        await self.initialize()
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        try:
            await server.serve()
        finally:
            # Cleanup background tasks
            for task in self.background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'api_gateway': {
                'status': 'healthy',
                'version': '1.0.0',
                'uptime': 'N/A'  # Would track actual uptime
            },
            'authentication': {
                'total_api_keys': len(self.auth_manager.api_keys),
                'blocked_keys': len(self.auth_manager.blocked_keys)
            },
            'metrics': self.metrics_collector.get_system_metrics(),
            'service_health': self.metrics_collector.get_service_health_report(),
            'websockets': self.websocket_manager.get_connection_stats(),
            'analytics_services': {
                name: 'configured' if service is not None else 'not_configured'
                for name, service in self.analytics_services.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize API Gateway
        config = {
            'allowed_origins': ["http://localhost:3000", "https://analytics.company.com"],
            'rate_limiting_enabled': True,
            'websocket_enabled': True
        }
        
        gateway = AnalyticsAPIGateway(config)
        
        # Create sample API keys
        admin_key = gateway.create_api_key(
            "admin_client", 
            ["admin", "streaming", "prediction", "anomaly", "graph", "timeseries", "bi", "metrics"]
        )
        
        user_key = gateway.create_api_key(
            "user_client",
            ["streaming", "prediction", "bi"]
        )
        
        print(f"Admin API Key: {admin_key}")
        print(f"User API Key: {user_key}")
        
        # Get system status
        status = gateway.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
        # Start server (commented out for testing)
        # await gateway.start_server(host="localhost", port=8080)
        
        logger.info("Analytics API Gateway demonstration completed")
    
    # Run the example
    asyncio.run(main())