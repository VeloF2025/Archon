"""
Model Serving Infrastructure for Archon Enhancement 2025 Phase 4

Production-grade model serving system with:
- Multi-framework model serving (PyTorch, TensorFlow, ONNX, etc.)
- Auto-scaling and load balancing
- A/B testing and canary deployments
- Real-time and batch inference
- Model versioning and rollback capabilities
- Performance monitoring and health checks
- Edge deployment and optimization
- Stream processing integration
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import pickle
import hashlib
import time
import aiohttp
from contextlib import asynccontextmanager
import signal

logger = logging.getLogger(__name__)

class ServingFramework(Enum):
    """Supported serving frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"

class DeploymentStrategy(Enum):
    """Model deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"
    MULTI_ARM_BANDIT = "multi_arm_bandit"

class InferenceType(Enum):
    """Types of inference"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"

class ServingStatus(Enum):
    """Serving endpoint status"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SCALING = "scaling"
    UPDATING = "updating"
    FAILED = "failed"
    TERMINATED = "terminated"

class OptimizationType(Enum):
    """Model optimization types"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    TENSORRT_OPTIMIZATION = "tensorrt_optimization"
    ONNX_OPTIMIZATION = "onnx_optimization"
    DYNAMIC_BATCHING = "dynamic_batching"

@dataclass
class ModelArtifact:
    """Model artifact metadata"""
    model_id: str
    version: str
    framework: ServingFramework
    model_path: str
    config_path: Optional[str] = None
    preprocessing_path: Optional[str] = None
    postprocessing_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    size_mb: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "framework": self.framework.value,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "preprocessing_path": self.preprocessing_path,
            "postprocessing_path": self.postprocessing_path,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "size_mb": self.size_mb,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ServingConfig:
    """Model serving configuration"""
    endpoint_name: str
    model_artifact: ModelArtifact
    inference_type: InferenceType = InferenceType.REAL_TIME
    
    # Scaling configuration
    min_replicas: int = 1
    max_replicas: int = 10
    target_utilization: float = 70.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    
    # Resource requirements
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    
    # Performance settings
    batch_size: int = 1
    max_batch_delay_ms: int = 100
    timeout_seconds: int = 30
    enable_batching: bool = False
    enable_model_cache: bool = True
    
    # Health check settings
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Optimization settings
    optimizations: List[OptimizationType] = field(default_factory=list)
    enable_monitoring: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Traffic settings
    traffic_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "endpoint_name": self.endpoint_name,
            "model_artifact": self.model_artifact.to_dict(),
            "inference_type": self.inference_type.value,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_utilization": self.target_utilization,
            "scale_up_cooldown": self.scale_up_cooldown,
            "scale_down_cooldown": self.scale_down_cooldown,
            "cpu_request": self.cpu_request,
            "cpu_limit": self.cpu_limit,
            "memory_request": self.memory_request,
            "memory_limit": self.memory_limit,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "batch_size": self.batch_size,
            "max_batch_delay_ms": self.max_batch_delay_ms,
            "timeout_seconds": self.timeout_seconds,
            "enable_batching": self.enable_batching,
            "enable_model_cache": self.enable_model_cache,
            "health_check_path": self.health_check_path,
            "readiness_check_path": self.readiness_check_path,
            "health_check_interval": self.health_check_interval,
            "health_check_timeout": self.health_check_timeout,
            "optimizations": [opt.value for opt in self.optimizations],
            "enable_monitoring": self.enable_monitoring,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "traffic_percentage": self.traffic_percentage
        }

@dataclass
class InferenceRequest:
    """Inference request structure"""
    request_id: str
    endpoint_name: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "endpoint_name": self.endpoint_name,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "timeout": self.timeout,
            "priority": self.priority,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class InferenceResponse:
    """Inference response structure"""
    request_id: str
    endpoint_name: str
    outputs: Dict[str, Any]
    model_version: str
    latency_ms: float
    status: str = "success"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "request_id": self.request_id,
            "endpoint_name": self.endpoint_name,
            "outputs": self.outputs,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class InferenceEndpoint:
    """Inference endpoint configuration and state"""
    endpoint_name: str
    config: ServingConfig
    status: ServingStatus = ServingStatus.INITIALIZING
    current_replicas: int = 1
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    
    # Health tracking
    health_check_failures: int = 0
    last_health_check: Optional[datetime] = None
    uptime: timedelta = field(default_factory=lambda: timedelta(0))
    
    # Deployment info
    deployment_strategy: Optional[DeploymentStrategy] = None
    rollout_status: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "endpoint_name": self.endpoint_name,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_replicas": self.current_replicas,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "average_latency_ms": self.average_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_rps": self.throughput_rps,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "health_check_failures": self.health_check_failures,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "uptime_seconds": self.uptime.total_seconds(),
            "deployment_strategy": self.deployment_strategy.value if self.deployment_strategy else None,
            "rollout_status": self.rollout_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    endpoint_a: str
    endpoint_b: str
    traffic_split: float = 0.5  # Percentage to endpoint A
    duration_hours: int = 24
    success_metric: str = "accuracy"
    statistical_significance: float = 0.95
    minimum_sample_size: int = 1000
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_name": self.test_name,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "traffic_split": self.traffic_split,
            "duration_hours": self.duration_hours,
            "success_metric": self.success_metric,
            "statistical_significance": self.statistical_significance,
            "minimum_sample_size": self.minimum_sample_size,
            "created_at": self.created_at.isoformat()
        }

class ModelServingInfrastructure:
    """
    Production-grade model serving infrastructure with comprehensive
    deployment strategies, auto-scaling, monitoring, and optimization.
    """
    
    def __init__(self, base_path: str = "./model_serving", max_concurrent_requests: int = 1000):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_requests = max_concurrent_requests
        
        # Endpoint management
        self.endpoints: Dict[str, InferenceEndpoint] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Request handling
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=max_concurrent_requests)
        self.active_requests: Dict[str, InferenceRequest] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_test_results: Dict[str, Dict[str, Any]] = {}
        
        # Framework handlers
        self.framework_handlers = self._initialize_framework_handlers()
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(max_workers=10)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        
        # Monitoring
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.RLock()
        self._shutdown = False
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Model Serving Infrastructure initialized")
    
    async def initialize(self) -> None:
        """Initialize the serving infrastructure"""
        try:
            await self._load_existing_endpoints()
            await self._start_request_processor()
            await self._start_health_monitors()
            await self._start_metrics_collection()
            logger.info("Model Serving Infrastructure initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Model Serving Infrastructure: {e}")
            raise
    
    async def deploy_model(self, config: ServingConfig, strategy: DeploymentStrategy = DeploymentStrategy.ROLLING) -> str:
        """Deploy a model with specified configuration and strategy"""
        try:
            endpoint_name = config.endpoint_name
            
            # Validate configuration
            await self._validate_serving_config(config)
            
            # Create or update endpoint
            if endpoint_name in self.endpoints:
                # Update existing endpoint
                await self._update_endpoint(config, strategy)
            else:
                # Create new endpoint
                await self._create_endpoint(config)
            
            endpoint = self.endpoints[endpoint_name]
            endpoint.deployment_strategy = strategy
            
            # Load and optimize model
            await self._load_model(endpoint)
            
            # Start deployment process
            await self._execute_deployment_strategy(endpoint, strategy)
            
            # Start monitoring
            await self._start_endpoint_monitoring(endpoint_name)
            
            logger.info(f"Deployed model endpoint: {endpoint_name} using {strategy.value} strategy")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to deploy model {config.endpoint_name}: {e}")
            raise
    
    async def update_model(self, endpoint_name: str, new_artifact: ModelArtifact, strategy: DeploymentStrategy = DeploymentStrategy.CANARY) -> bool:
        """Update model with new version"""
        try:
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = self.endpoints[endpoint_name]
            old_artifact = endpoint.config.model_artifact
            
            # Create new configuration
            new_config = ServingConfig(
                endpoint_name=f"{endpoint_name}_v{new_artifact.version}",
                model_artifact=new_artifact,
                **{k: v for k, v in endpoint.config.to_dict().items() 
                   if k not in ["endpoint_name", "model_artifact"]}
            )
            
            # Deploy new version with strategy
            if strategy == DeploymentStrategy.CANARY:
                await self._canary_deployment(endpoint, new_config)
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                await self._blue_green_deployment(endpoint, new_config)
            elif strategy == DeploymentStrategy.ROLLING:
                await self._rolling_deployment(endpoint, new_config)
            
            logger.info(f"Updated model endpoint {endpoint_name} from {old_artifact.version} to {new_artifact.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model {endpoint_name}: {e}")
            return False
    
    async def create_ab_test(self, test_config: ABTestConfig) -> str:
        """Create A/B test between two endpoints"""
        try:
            # Validate endpoints exist
            if test_config.endpoint_a not in self.endpoints:
                raise ValueError(f"Endpoint A {test_config.endpoint_a} not found")
            if test_config.endpoint_b not in self.endpoints:
                raise ValueError(f"Endpoint B {test_config.endpoint_b} not found")
            
            # Store test configuration
            with self._lock:
                self.ab_tests[test_config.test_name] = test_config
                self.ab_test_results[test_config.test_name] = {
                    "endpoint_a_requests": 0,
                    "endpoint_b_requests": 0,
                    "endpoint_a_success": 0,
                    "endpoint_b_success": 0,
                    "endpoint_a_latency": [],
                    "endpoint_b_latency": [],
                    "start_time": datetime.utcnow().isoformat(),
                    "status": "running"
                }
            
            # Start A/B test monitoring
            await self._start_ab_test_monitoring(test_config.test_name)
            
            logger.info(f"Created A/B test: {test_config.test_name}")
            return test_config.test_name
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Make prediction using specified endpoint"""
        try:
            start_time = time.time()
            
            # Route request for A/B testing if applicable
            endpoint_name = await self._route_request(request)
            
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = self.endpoints[endpoint_name]
            
            # Check endpoint health
            if endpoint.status != ServingStatus.HEALTHY:
                raise ValueError(f"Endpoint {endpoint_name} is not healthy (status: {endpoint.status.value})")
            
            # Add to active requests
            with self._lock:
                self.active_requests[request.request_id] = request
            
            try:
                # Get framework handler
                framework = endpoint.config.model_artifact.framework
                handler = self.framework_handlers.get(framework)
                
                if not handler:
                    raise ValueError(f"No handler for framework: {framework.value}")
                
                # Execute inference
                outputs = await handler(endpoint, request.inputs, request.parameters)
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                
                # Update endpoint metrics
                await self._update_endpoint_metrics(endpoint_name, latency_ms, True)
                
                # Create response
                response = InferenceResponse(
                    request_id=request.request_id,
                    endpoint_name=endpoint_name,
                    outputs=outputs,
                    model_version=endpoint.config.model_artifact.version,
                    latency_ms=latency_ms
                )
                
                # Update A/B test results if applicable
                await self._update_ab_test_results(request, response)
                
                return response
                
            finally:
                # Remove from active requests
                with self._lock:
                    self.active_requests.pop(request.request_id, None)
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Update error metrics
            if request.endpoint_name in self.endpoints:
                await self._update_endpoint_metrics(request.endpoint_name, latency_ms, False)
            
            # Create error response
            response = InferenceResponse(
                request_id=request.request_id,
                endpoint_name=request.endpoint_name,
                outputs={},
                model_version="unknown",
                latency_ms=latency_ms,
                status="error",
                error_message=str(e)
            )
            
            logger.error(f"Prediction failed for request {request.request_id}: {e}")
            return response
    
    async def batch_predict(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Execute batch predictions"""
        try:
            # Group requests by endpoint
            endpoint_requests = {}
            for request in requests:
                endpoint_name = await self._route_request(request)
                if endpoint_name not in endpoint_requests:
                    endpoint_requests[endpoint_name] = []
                endpoint_requests[endpoint_name].append(request)
            
            # Execute predictions for each endpoint
            all_responses = []
            
            for endpoint_name, endpoint_requests_list in endpoint_requests.items():
                if endpoint_name not in self.endpoints:
                    # Create error responses for invalid endpoint
                    for request in endpoint_requests_list:
                        response = InferenceResponse(
                            request_id=request.request_id,
                            endpoint_name=request.endpoint_name,
                            outputs={},
                            model_version="unknown",
                            latency_ms=0.0,
                            status="error",
                            error_message=f"Endpoint {endpoint_name} not found"
                        )
                        all_responses.append(response)
                    continue
                
                endpoint = self.endpoints[endpoint_name]
                
                # Execute batch inference
                if endpoint.config.enable_batching:
                    batch_responses = await self._execute_batch_inference(endpoint, endpoint_requests_list)
                else:
                    # Execute individual predictions
                    batch_responses = await asyncio.gather(*[
                        self.predict(request) for request in endpoint_requests_list
                    ])
                
                all_responses.extend(batch_responses)
            
            return all_responses
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return error responses for all requests
            return [
                InferenceResponse(
                    request_id=request.request_id,
                    endpoint_name=request.endpoint_name,
                    outputs={},
                    model_version="unknown",
                    latency_ms=0.0,
                    status="error",
                    error_message=str(e)
                )
                for request in requests
            ]
    
    async def scale_endpoint(self, endpoint_name: str, target_replicas: int) -> bool:
        """Scale endpoint to target number of replicas"""
        try:
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = self.endpoints[endpoint_name]
            
            if target_replicas < endpoint.config.min_replicas:
                target_replicas = endpoint.config.min_replicas
            elif target_replicas > endpoint.config.max_replicas:
                target_replicas = endpoint.config.max_replicas
            
            if target_replicas == endpoint.current_replicas:
                return True
            
            endpoint.status = ServingStatus.SCALING
            
            # Simulate scaling operation
            await asyncio.sleep(2.0)  # Simulate scaling time
            
            endpoint.current_replicas = target_replicas
            endpoint.status = ServingStatus.HEALTHY
            endpoint.updated_at = datetime.utcnow()
            
            logger.info(f"Scaled endpoint {endpoint_name} to {target_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale endpoint {endpoint_name}: {e}")
            return False
    
    async def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get comprehensive endpoint status"""
        try:
            if endpoint_name not in self.endpoints:
                raise ValueError(f"Endpoint {endpoint_name} not found")
            
            endpoint = self.endpoints[endpoint_name]
            
            # Get recent metrics
            recent_metrics = self.metrics_history.get(endpoint_name, [])[-10:]  # Last 10 metrics
            
            # Calculate uptime
            uptime = datetime.utcnow() - endpoint.created_at
            
            status = {
                "endpoint_name": endpoint_name,
                "status": endpoint.status.value,
                "current_replicas": endpoint.current_replicas,
                "target_replicas": endpoint.current_replicas,  # Simplified
                "uptime_hours": uptime.total_seconds() / 3600,
                "model_info": {
                    "model_id": endpoint.config.model_artifact.model_id,
                    "version": endpoint.config.model_artifact.version,
                    "framework": endpoint.config.model_artifact.framework.value
                },
                "performance_metrics": {
                    "total_requests": endpoint.total_requests,
                    "successful_requests": endpoint.successful_requests,
                    "failed_requests": endpoint.failed_requests,
                    "success_rate": endpoint.get_success_rate(),
                    "error_rate": endpoint.get_error_rate(),
                    "average_latency_ms": endpoint.average_latency_ms,
                    "p95_latency_ms": endpoint.p95_latency_ms,
                    "p99_latency_ms": endpoint.p99_latency_ms,
                    "throughput_rps": endpoint.throughput_rps
                },
                "resource_utilization": {
                    "cpu_utilization": endpoint.cpu_utilization,
                    "memory_utilization": endpoint.memory_utilization,
                    "gpu_utilization": endpoint.gpu_utilization
                },
                "health_info": {
                    "health_check_failures": endpoint.health_check_failures,
                    "last_health_check": endpoint.last_health_check.isoformat() if endpoint.last_health_check else None
                },
                "deployment_info": {
                    "strategy": endpoint.deployment_strategy.value if endpoint.deployment_strategy else None,
                    "rollout_status": endpoint.rollout_status
                },
                "recent_metrics": recent_metrics,
                "created_at": endpoint.created_at.isoformat(),
                "updated_at": endpoint.updated_at.isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get endpoint status {endpoint_name}: {e}")
            raise
    
    async def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results and analysis"""
        try:
            if test_name not in self.ab_tests:
                raise ValueError(f"A/B test {test_name} not found")
            
            test_config = self.ab_tests[test_name]
            results = self.ab_test_results[test_name]
            
            # Calculate statistics
            total_requests_a = results["endpoint_a_requests"]
            total_requests_b = results["endpoint_b_requests"]
            success_rate_a = results["endpoint_a_success"] / max(total_requests_a, 1)
            success_rate_b = results["endpoint_b_success"] / max(total_requests_b, 1)
            
            avg_latency_a = np.mean(results["endpoint_a_latency"]) if results["endpoint_a_latency"] else 0
            avg_latency_b = np.mean(results["endpoint_b_latency"]) if results["endpoint_b_latency"] else 0
            
            # Simple statistical significance test (simplified)
            total_requests = total_requests_a + total_requests_b
            is_significant = total_requests >= test_config.minimum_sample_size
            
            # Determine winner
            winner = None
            if is_significant:
                if test_config.success_metric == "success_rate":
                    winner = "A" if success_rate_a > success_rate_b else "B"
                elif test_config.success_metric == "latency":
                    winner = "A" if avg_latency_a < avg_latency_b else "B"
            
            analysis = {
                "test_name": test_name,
                "test_config": test_config.to_dict(),
                "status": results["status"],
                "duration_hours": (datetime.utcnow() - datetime.fromisoformat(results["start_time"])).total_seconds() / 3600,
                "sample_sizes": {
                    "endpoint_a": total_requests_a,
                    "endpoint_b": total_requests_b,
                    "total": total_requests
                },
                "performance_metrics": {
                    "endpoint_a": {
                        "success_rate": success_rate_a,
                        "average_latency_ms": avg_latency_a,
                        "total_requests": total_requests_a
                    },
                    "endpoint_b": {
                        "success_rate": success_rate_b,
                        "average_latency_ms": avg_latency_b,
                        "total_requests": total_requests_b
                    }
                },
                "statistical_analysis": {
                    "is_significant": is_significant,
                    "winner": winner,
                    "improvement": abs(success_rate_a - success_rate_b) if winner else 0,
                    "confidence_level": test_config.statistical_significance
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to get A/B test results {test_name}: {e}")
            raise
    
    async def list_endpoints(self, status_filter: Optional[ServingStatus] = None) -> List[Dict[str, Any]]:
        """List all endpoints with optional status filtering"""
        try:
            endpoints_info = []
            
            for endpoint_name, endpoint in self.endpoints.items():
                if status_filter and endpoint.status != status_filter:
                    continue
                
                endpoint_info = {
                    "endpoint_name": endpoint_name,
                    "status": endpoint.status.value,
                    "model_id": endpoint.config.model_artifact.model_id,
                    "model_version": endpoint.config.model_artifact.version,
                    "framework": endpoint.config.model_artifact.framework.value,
                    "inference_type": endpoint.config.inference_type.value,
                    "current_replicas": endpoint.current_replicas,
                    "total_requests": endpoint.total_requests,
                    "success_rate": endpoint.get_success_rate(),
                    "average_latency_ms": endpoint.average_latency_ms,
                    "cpu_utilization": endpoint.cpu_utilization,
                    "created_at": endpoint.created_at.isoformat()
                }
                
                endpoints_info.append(endpoint_info)
            
            # Sort by creation time (newest first)
            endpoints_info.sort(key=lambda x: x["created_at"], reverse=True)
            
            return endpoints_info
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get serving infrastructure performance metrics"""
        try:
            total_endpoints = len(self.endpoints)
            healthy_endpoints = len([e for e in self.endpoints.values() if e.status == ServingStatus.HEALTHY])
            total_requests = sum(e.total_requests for e in self.endpoints.values())
            total_successful = sum(e.successful_requests for e in self.endpoints.values())
            
            # Calculate overall metrics
            overall_success_rate = total_successful / max(total_requests, 1)
            avg_latency = np.mean([e.average_latency_ms for e in self.endpoints.values() if e.average_latency_ms > 0])
            total_replicas = sum(e.current_replicas for e in self.endpoints.values())
            
            # Framework distribution
            framework_dist = {}
            for endpoint in self.endpoints.values():
                framework = endpoint.config.model_artifact.framework.value
                framework_dist[framework] = framework_dist.get(framework, 0) + 1
            
            # Inference type distribution
            inference_dist = {}
            for endpoint in self.endpoints.values():
                inf_type = endpoint.config.inference_type.value
                inference_dist[inf_type] = inference_dist.get(inf_type, 0) + 1
            
            return {
                "total_endpoints": total_endpoints,
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": total_endpoints - healthy_endpoints,
                "health_rate": healthy_endpoints / max(total_endpoints, 1),
                "total_replicas": total_replicas,
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "overall_success_rate": overall_success_rate,
                "overall_error_rate": 1.0 - overall_success_rate,
                "average_latency_ms": float(avg_latency) if not np.isnan(avg_latency) else 0.0,
                "active_ab_tests": len([t for t in self.ab_test_results.values() if t["status"] == "running"]),
                "framework_distribution": framework_dist,
                "inference_type_distribution": inference_dist,
                "active_requests": len(self.active_requests),
                "model_cache_size": len(self.model_cache),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get serving metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the serving infrastructure"""
        try:
            self._shutdown = True
            
            # Cancel monitoring tasks
            for task_name, task in self._monitoring_tasks.items():
                if not task.done():
                    task.cancel()
                logger.info(f"Cancelled monitoring task: {task_name}")
            
            # Shutdown all endpoints
            for endpoint_name in list(self.endpoints.keys()):
                await self._shutdown_endpoint(endpoint_name)
            
            # Clear model cache
            self.model_cache.clear()
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            # Save state
            await self._save_infrastructure_state()
            
            logger.info("Model Serving Infrastructure shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Model Serving Infrastructure shutdown: {e}")
    
    # Private helper methods
    
    def _initialize_framework_handlers(self) -> Dict[ServingFramework, Callable]:
        """Initialize framework-specific serving handlers"""
        return {
            ServingFramework.PYTORCH: self._handle_pytorch_inference,
            ServingFramework.TENSORFLOW: self._handle_tensorflow_inference,
            ServingFramework.ONNX: self._handle_onnx_inference,
            ServingFramework.SKLEARN: self._handle_sklearn_inference,
            ServingFramework.XGBOOST: self._handle_xgboost_inference,
            ServingFramework.LIGHTGBM: self._handle_lightgbm_inference,
            ServingFramework.CUSTOM: self._handle_custom_inference
        }
    
    async def _validate_serving_config(self, config: ServingConfig) -> None:
        """Validate serving configuration"""
        if config.min_replicas < 1:
            raise ValueError("Minimum replicas must be at least 1")
        
        if config.max_replicas < config.min_replicas:
            raise ValueError("Maximum replicas must be >= minimum replicas")
        
        if not (0 < config.target_utilization <= 100):
            raise ValueError("Target utilization must be between 0 and 100")
        
        if config.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
    
    async def _create_endpoint(self, config: ServingConfig) -> None:
        """Create new inference endpoint"""
        endpoint = InferenceEndpoint(
            endpoint_name=config.endpoint_name,
            config=config,
            current_replicas=config.min_replicas
        )
        
        with self._lock:
            self.endpoints[config.endpoint_name] = endpoint
            self.metrics_history[config.endpoint_name] = []
    
    async def _update_endpoint(self, config: ServingConfig, strategy: DeploymentStrategy) -> None:
        """Update existing endpoint"""
        endpoint = self.endpoints[config.endpoint_name]
        endpoint.config = config
        endpoint.status = ServingStatus.UPDATING
        endpoint.updated_at = datetime.utcnow()
    
    async def _load_model(self, endpoint: InferenceEndpoint) -> None:
        """Load and cache model for endpoint"""
        model_key = f"{endpoint.config.model_artifact.model_id}_{endpoint.config.model_artifact.version}"
        
        if model_key not in self.model_cache:
            # Simulate model loading
            await asyncio.sleep(1.0)  # Simulate loading time
            
            # Cache mock model
            self.model_cache[model_key] = {
                "model": "mock_model_object",
                "preprocessor": "mock_preprocessor",
                "postprocessor": "mock_postprocessor",
                "loaded_at": datetime.utcnow().isoformat()
            }
        
        logger.info(f"Loaded model for endpoint: {endpoint.endpoint_name}")
    
    async def _execute_deployment_strategy(self, endpoint: InferenceEndpoint, strategy: DeploymentStrategy) -> None:
        """Execute deployment strategy"""
        if strategy == DeploymentStrategy.ROLLING:
            await self._rolling_deployment(endpoint, endpoint.config)
        elif strategy == DeploymentStrategy.BLUE_GREEN:
            await self._blue_green_deployment(endpoint, endpoint.config)
        elif strategy == DeploymentStrategy.CANARY:
            await self._canary_deployment(endpoint, endpoint.config)
        
        endpoint.status = ServingStatus.HEALTHY
    
    async def _rolling_deployment(self, endpoint: InferenceEndpoint, config: ServingConfig) -> None:
        """Execute rolling deployment"""
        logger.info(f"Starting rolling deployment for {endpoint.endpoint_name}")
        
        # Simulate rolling deployment
        for i in range(endpoint.current_replicas):
            endpoint.rollout_status[f"replica_{i}"] = "updating"
            await asyncio.sleep(0.5)  # Simulate update time
            endpoint.rollout_status[f"replica_{i}"] = "healthy"
        
        logger.info(f"Completed rolling deployment for {endpoint.endpoint_name}")
    
    async def _canary_deployment(self, endpoint: InferenceEndpoint, config: ServingConfig) -> None:
        """Execute canary deployment"""
        logger.info(f"Starting canary deployment for {endpoint.endpoint_name}")
        
        # Phase 1: Deploy canary (10% traffic)
        endpoint.rollout_status["phase"] = "canary"
        endpoint.rollout_status["canary_traffic"] = 10
        await asyncio.sleep(1.0)
        
        # Phase 2: Increase to 50% traffic
        endpoint.rollout_status["canary_traffic"] = 50
        await asyncio.sleep(1.0)
        
        # Phase 3: Full rollout
        endpoint.rollout_status["canary_traffic"] = 100
        endpoint.rollout_status["phase"] = "complete"
        
        logger.info(f"Completed canary deployment for {endpoint.endpoint_name}")
    
    async def _blue_green_deployment(self, endpoint: InferenceEndpoint, config: ServingConfig) -> None:
        """Execute blue-green deployment"""
        logger.info(f"Starting blue-green deployment for {endpoint.endpoint_name}")
        
        # Phase 1: Deploy green environment
        endpoint.rollout_status["active_environment"] = "blue"
        endpoint.rollout_status["green_status"] = "deploying"
        await asyncio.sleep(1.5)
        
        # Phase 2: Switch traffic to green
        endpoint.rollout_status["green_status"] = "ready"
        endpoint.rollout_status["active_environment"] = "green"
        endpoint.rollout_status["blue_status"] = "standby"
        
        logger.info(f"Completed blue-green deployment for {endpoint.endpoint_name}")
    
    async def _route_request(self, request: InferenceRequest) -> str:
        """Route request based on A/B tests and load balancing"""
        endpoint_name = request.endpoint_name
        
        # Check for active A/B tests
        for test_name, test_config in self.ab_tests.items():
            if (test_config.endpoint_a == endpoint_name or 
                test_config.endpoint_b == endpoint_name):
                
                # Route based on traffic split
                import random
                if random.random() < test_config.traffic_split:
                    return test_config.endpoint_a
                else:
                    return test_config.endpoint_b
        
        return endpoint_name
    
    # Framework-specific inference handlers
    
    async def _handle_pytorch_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle PyTorch inference"""
        # Simulate PyTorch inference
        await asyncio.sleep(np.random.uniform(0.01, 0.1))  # Simulate inference time
        
        # Mock outputs based on inputs
        if "image" in inputs:
            outputs = {"predictions": [0.1, 0.3, 0.6], "confidence": 0.6}
        elif "text" in inputs:
            outputs = {"sentiment": "positive", "confidence": 0.85}
        else:
            outputs = {"result": np.random.rand(5).tolist()}
        
        return outputs
    
    async def _handle_tensorflow_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TensorFlow inference"""
        # Simulate TensorFlow inference
        await asyncio.sleep(np.random.uniform(0.02, 0.12))
        
        # Mock outputs
        if "features" in inputs:
            outputs = {"predictions": np.random.rand(3).tolist(), "probabilities": np.random.rand(3).tolist()}
        else:
            outputs = {"output": np.random.rand(10).tolist()}
        
        return outputs
    
    async def _handle_onnx_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ONNX inference"""
        # Simulate ONNX inference (typically faster)
        await asyncio.sleep(np.random.uniform(0.005, 0.05))
        
        outputs = {"predictions": np.random.rand(5).tolist()}
        return outputs
    
    async def _handle_sklearn_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scikit-learn inference"""
        # Simulate sklearn inference (very fast)
        await asyncio.sleep(np.random.uniform(0.001, 0.01))
        
        if "features" in inputs:
            features = inputs["features"]
            if isinstance(features, list) and len(features) > 0:
                # Simulate classification or regression
                if len(features) < 10:
                    outputs = {"prediction": int(np.random.randint(0, 3)), "probability": float(np.random.rand())}
                else:
                    outputs = {"prediction": float(np.random.rand())}
            else:
                outputs = {"prediction": 0, "probability": 0.5}
        else:
            outputs = {"prediction": int(np.random.randint(0, 2))}
        
        return outputs
    
    async def _handle_xgboost_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle XGBoost inference"""
        await asyncio.sleep(np.random.uniform(0.002, 0.02))
        
        outputs = {"prediction": float(np.random.rand()), "feature_importance": np.random.rand(5).tolist()}
        return outputs
    
    async def _handle_lightgbm_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LightGBM inference"""
        await asyncio.sleep(np.random.uniform(0.001, 0.015))
        
        outputs = {"prediction": float(np.random.rand()), "leaf_values": np.random.rand(3).tolist()}
        return outputs
    
    async def _handle_custom_inference(self, endpoint: InferenceEndpoint, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle custom inference"""
        await asyncio.sleep(np.random.uniform(0.01, 0.1))
        
        outputs = {"custom_output": "custom_result", "values": np.random.rand(3).tolist()}
        return outputs
    
    async def _execute_batch_inference(self, endpoint: InferenceEndpoint, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Execute batch inference for multiple requests"""
        start_time = time.time()
        
        # Combine all inputs
        batch_inputs = {}
        for i, request in enumerate(requests):
            for key, value in request.inputs.items():
                if key not in batch_inputs:
                    batch_inputs[key] = []
                batch_inputs[key].append(value)
        
        # Execute batch inference
        framework = endpoint.config.model_artifact.framework
        handler = self.framework_handlers.get(framework)
        batch_outputs = await handler(endpoint, batch_inputs, {})
        
        # Split outputs back to individual responses
        responses = []
        batch_latency = (time.time() - start_time) * 1000
        
        for i, request in enumerate(requests):
            # Extract individual outputs (simplified)
            individual_outputs = {}
            for key, values in batch_outputs.items():
                if isinstance(values, list) and len(values) > i:
                    individual_outputs[key] = values[i]
                else:
                    individual_outputs[key] = values
            
            response = InferenceResponse(
                request_id=request.request_id,
                endpoint_name=endpoint.endpoint_name,
                outputs=individual_outputs,
                model_version=endpoint.config.model_artifact.version,
                latency_ms=batch_latency / len(requests)  # Distribute latency
            )
            responses.append(response)
        
        return responses
    
    async def _update_endpoint_metrics(self, endpoint_name: str, latency_ms: float, success: bool) -> None:
        """Update endpoint performance metrics"""
        if endpoint_name not in self.endpoints:
            return
        
        endpoint = self.endpoints[endpoint_name]
        
        # Update counters
        endpoint.total_requests += 1
        if success:
            endpoint.successful_requests += 1
        else:
            endpoint.failed_requests += 1
        
        # Update latency metrics (running average)
        if endpoint.total_requests == 1:
            endpoint.average_latency_ms = latency_ms
            endpoint.p95_latency_ms = latency_ms
            endpoint.p99_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            endpoint.average_latency_ms = (1 - alpha) * endpoint.average_latency_ms + alpha * latency_ms
            endpoint.p95_latency_ms = max(endpoint.p95_latency_ms * 0.95, latency_ms)
            endpoint.p99_latency_ms = max(endpoint.p99_latency_ms * 0.99, latency_ms)
        
        # Update resource utilization (mock values)
        endpoint.cpu_utilization = np.random.uniform(30, 80)
        endpoint.memory_utilization = np.random.uniform(40, 85)
        if endpoint.config.gpu_count > 0:
            endpoint.gpu_utilization = np.random.uniform(60, 95)
        
        # Calculate throughput (requests per second)
        time_window = 60  # 1 minute window
        recent_requests = min(endpoint.total_requests, time_window)
        endpoint.throughput_rps = recent_requests / time_window
        
        # Store metrics in history
        metrics_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": endpoint.total_requests,
            "success_rate": endpoint.get_success_rate(),
            "average_latency_ms": endpoint.average_latency_ms,
            "throughput_rps": endpoint.throughput_rps,
            "cpu_utilization": endpoint.cpu_utilization,
            "memory_utilization": endpoint.memory_utilization,
            "gpu_utilization": endpoint.gpu_utilization
        }
        
        with self._lock:
            if endpoint_name not in self.metrics_history:
                self.metrics_history[endpoint_name] = []
            
            self.metrics_history[endpoint_name].append(metrics_entry)
            
            # Keep only recent metrics (last 1000 entries)
            if len(self.metrics_history[endpoint_name]) > 1000:
                self.metrics_history[endpoint_name] = self.metrics_history[endpoint_name][-1000:]
    
    async def _update_ab_test_results(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Update A/B test results"""
        for test_name, test_config in self.ab_tests.items():
            if response.endpoint_name in [test_config.endpoint_a, test_config.endpoint_b]:
                results = self.ab_test_results[test_name]
                
                if response.endpoint_name == test_config.endpoint_a:
                    results["endpoint_a_requests"] += 1
                    if response.status == "success":
                        results["endpoint_a_success"] += 1
                    results["endpoint_a_latency"].append(response.latency_ms)
                else:
                    results["endpoint_b_requests"] += 1
                    if response.status == "success":
                        results["endpoint_b_success"] += 1
                    results["endpoint_b_latency"].append(response.latency_ms)
    
    async def _start_request_processor(self) -> None:
        """Start background request processor"""
        async def request_processor():
            while not self._shutdown:
                try:
                    # Process queued requests
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in request processor: {e}")
        
        self._monitoring_tasks["request_processor"] = asyncio.create_task(request_processor())
    
    async def _start_health_monitors(self) -> None:
        """Start health monitoring for all endpoints"""
        async def health_monitor():
            while not self._shutdown:
                try:
                    for endpoint_name, endpoint in self.endpoints.items():
                        await self._check_endpoint_health(endpoint)
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Error in health monitor: {e}")
                    await asyncio.sleep(30)
        
        self._monitoring_tasks["health_monitor"] = asyncio.create_task(health_monitor())
    
    async def _start_metrics_collection(self) -> None:
        """Start metrics collection"""
        async def metrics_collector():
            while not self._shutdown:
                try:
                    # Collect and aggregate metrics
                    await asyncio.sleep(60)  # Collect every minute
                except Exception as e:
                    logger.error(f"Error in metrics collection: {e}")
                    await asyncio.sleep(60)
        
        self._monitoring_tasks["metrics_collector"] = asyncio.create_task(metrics_collector())
    
    async def _start_endpoint_monitoring(self, endpoint_name: str) -> None:
        """Start monitoring for specific endpoint"""
        # Auto-scaling monitoring
        async def autoscaler():
            while not self._shutdown and endpoint_name in self.endpoints:
                try:
                    endpoint = self.endpoints[endpoint_name]
                    
                    # Simple auto-scaling logic
                    if endpoint.cpu_utilization > endpoint.config.target_utilization:
                        if endpoint.current_replicas < endpoint.config.max_replicas:
                            await self.scale_endpoint(endpoint_name, endpoint.current_replicas + 1)
                    elif endpoint.cpu_utilization < endpoint.config.target_utilization * 0.5:
                        if endpoint.current_replicas > endpoint.config.min_replicas:
                            await self.scale_endpoint(endpoint_name, endpoint.current_replicas - 1)
                    
                    await asyncio.sleep(endpoint.config.scale_up_cooldown)
                except Exception as e:
                    logger.error(f"Error in autoscaler for {endpoint_name}: {e}")
                    await asyncio.sleep(300)
        
        self._monitoring_tasks[f"autoscaler_{endpoint_name}"] = asyncio.create_task(autoscaler())
    
    async def _start_ab_test_monitoring(self, test_name: str) -> None:
        """Start monitoring for A/B test"""
        async def ab_test_monitor():
            test_config = self.ab_tests[test_name]
            test_results = self.ab_test_results[test_name]
            
            # Monitor for test duration
            end_time = datetime.utcnow() + timedelta(hours=test_config.duration_hours)
            
            while datetime.utcnow() < end_time and not self._shutdown:
                await asyncio.sleep(3600)  # Check hourly
            
            # Mark test as completed
            test_results["status"] = "completed"
            test_results["end_time"] = datetime.utcnow().isoformat()
            
            logger.info(f"A/B test {test_name} completed")
        
        self._monitoring_tasks[f"ab_test_{test_name}"] = asyncio.create_task(ab_test_monitor())
    
    async def _check_endpoint_health(self, endpoint: InferenceEndpoint) -> None:
        """Check endpoint health"""
        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            
            # Mock health check logic
            health_score = np.random.uniform(0.7, 1.0)
            
            if health_score > 0.9:
                if endpoint.status != ServingStatus.HEALTHY:
                    endpoint.status = ServingStatus.HEALTHY
                    endpoint.health_check_failures = 0
            elif health_score > 0.7:
                endpoint.status = ServingStatus.DEGRADED
            else:
                endpoint.status = ServingStatus.UNHEALTHY
                endpoint.health_check_failures += 1
            
            endpoint.last_health_check = datetime.utcnow()
            
        except Exception as e:
            endpoint.health_check_failures += 1
            logger.error(f"Health check failed for {endpoint.endpoint_name}: {e}")
    
    async def _shutdown_endpoint(self, endpoint_name: str) -> None:
        """Shutdown specific endpoint"""
        if endpoint_name in self.endpoints:
            endpoint = self.endpoints[endpoint_name]
            endpoint.status = ServingStatus.TERMINATED
            
            # Cancel endpoint-specific monitoring
            task_key = f"autoscaler_{endpoint_name}"
            if task_key in self._monitoring_tasks:
                self._monitoring_tasks[task_key].cancel()
                del self._monitoring_tasks[task_key]
            
            logger.info(f"Shutdown endpoint: {endpoint_name}")
    
    async def _load_existing_endpoints(self) -> None:
        """Load existing endpoints from storage"""
        # Placeholder for loading endpoints from persistent storage
        logger.info("Loading existing endpoints from storage")
    
    async def _save_infrastructure_state(self) -> None:
        """Save infrastructure state to storage"""
        state_file = self.base_path / "serving_state.json"
        
        state = {
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": len([e for e in self.endpoints.values() if e.status == ServingStatus.HEALTHY]),
            "total_ab_tests": len(self.ab_tests),
            "active_ab_tests": len([t for t in self.ab_test_results.values() if t["status"] == "running"]),
            "model_cache_size": len(self.model_cache),
            "total_requests_served": sum(e.total_requests for e in self.endpoints.values()),
            "last_saved": datetime.utcnow().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)