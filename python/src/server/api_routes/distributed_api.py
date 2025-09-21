"""
Distributed Systems API Routes for Archon Enhancement 2025 Phase 3

Provides REST endpoints for all distributed system operations including:
- Cluster management with auto-scaling
- Load balancing with multiple algorithms
- Service discovery with health monitoring
- Distributed caching with consistency levels
- Message queuing with guaranteed delivery
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
from datetime import datetime

# Import all distributed system components
from ...agents.distributed.cluster_manager import (
    ClusterManager, ClusterConfig, NodeType, NodeSpec, 
    WorkloadType, ClusterStatus
)
from ...agents.distributed.load_balancer import (
    LoadBalancer, LoadBalancerConfig, BalancingAlgorithm,
    HealthCheckConfig, BackendServer, ServerHealth
)
from ...agents.distributed.service_discovery import (
    ServiceDiscovery, ServiceConfig, ServiceInstance,
    HealthStatus, ServiceWatcher
)
from ...agents.distributed.distributed_cache import (
    DistributedCache, CacheConfig, ConsistencyLevel,
    ReplicationStrategy, CacheItem, EvictionPolicy
)
from ...agents.distributed.message_queue import (
    DistributedMessageQueue, MessageQueueConfig, DeliveryMode,
    PriorityStrategy, Message, MessageStatus, Consumer
)

router = APIRouter(prefix="/api/distributed", tags=["distributed"])

# Global distributed system instances
cluster_manager: Optional[ClusterManager] = None
load_balancer: Optional[LoadBalancer] = None
service_discovery: Optional[ServiceDiscovery] = None
distributed_cache: Optional[DistributedCache] = None
message_queue: Optional[DistributedMessageQueue] = None

# Pydantic models for API requests/responses

# Cluster Management Models
class ClusterConfigRequest(BaseModel):
    name: str = Field(..., description="Cluster name")
    max_nodes: int = Field(default=10, ge=1, le=100)
    auto_scaling: bool = Field(default=True)
    scale_up_threshold: float = Field(default=0.8, ge=0.1, le=1.0)
    scale_down_threshold: float = Field(default=0.3, ge=0.1, le=1.0)
    health_check_interval: int = Field(default=30, ge=5, le=300)

class NodeSpecRequest(BaseModel):
    node_type: NodeType
    cpu_cores: int = Field(ge=1, le=64)
    memory_gb: int = Field(ge=1, le=256)
    disk_gb: int = Field(ge=10, le=2000)
    gpu_count: int = Field(default=0, ge=0, le=8)

class WorkloadRequest(BaseModel):
    workload_id: str
    workload_type: WorkloadType
    resource_requirements: NodeSpecRequest
    replicas: int = Field(default=1, ge=1, le=20)
    labels: Dict[str, str] = Field(default_factory=dict)

# Load Balancer Models
class LoadBalancerConfigRequest(BaseModel):
    algorithm: BalancingAlgorithm = Field(default=BalancingAlgorithm.ROUND_ROBIN)
    enable_circuit_breaker: bool = Field(default=True)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=50)
    circuit_timeout: int = Field(default=60, ge=10, le=300)
    enable_rate_limiting: bool = Field(default=True)
    requests_per_second: int = Field(default=100, ge=1, le=10000)

class BackendServerRequest(BaseModel):
    server_id: str
    host: str
    port: int = Field(ge=1, le=65535)
    weight: int = Field(default=1, ge=1, le=100)
    labels: Dict[str, str] = Field(default_factory=dict)

class HealthCheckConfigRequest(BaseModel):
    path: str = Field(default="/health")
    interval: int = Field(default=30, ge=5, le=300)
    timeout: int = Field(default=10, ge=1, le=60)
    healthy_threshold: int = Field(default=2, ge=1, le=10)
    unhealthy_threshold: int = Field(default=3, ge=1, le=10)

# Service Discovery Models
class ServiceConfigRequest(BaseModel):
    service_name: str
    service_version: str = Field(default="1.0.0")
    port: int = Field(ge=1, le=65535)
    health_check_path: str = Field(default="/health")
    labels: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)

class ServiceInstanceRequest(BaseModel):
    instance_id: str
    host: str
    port: int = Field(ge=1, le=65535)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Distributed Cache Models
class CacheConfigRequest(BaseModel):
    cluster_name: str
    consistency_level: ConsistencyLevel = Field(default=ConsistencyLevel.EVENTUAL)
    replication_strategy: ReplicationStrategy = Field(default=ReplicationStrategy.ASYNC)
    replication_factor: int = Field(default=3, ge=1, le=10)
    eviction_policy: EvictionPolicy = Field(default=EvictionPolicy.LRU)
    ttl_seconds: Optional[int] = Field(default=3600, ge=1)
    enable_compression: bool = Field(default=True)

class CacheItemRequest(BaseModel):
    key: str
    value: Any
    ttl_seconds: Optional[int] = None
    tags: List[str] = Field(default_factory=list)

class CacheQueryRequest(BaseModel):
    pattern: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=1000)

# Message Queue Models
class MessageQueueConfigRequest(BaseModel):
    cluster_name: str
    delivery_mode: DeliveryMode = Field(default=DeliveryMode.AT_LEAST_ONCE)
    priority_strategy: PriorityStrategy = Field(default=PriorityStrategy.FIFO)
    enable_dead_letter: bool = Field(default=True)
    max_retry_count: int = Field(default=3, ge=0, le=10)
    retry_delay: int = Field(default=5, ge=1, le=300)
    enable_persistence: bool = Field(default=True)

class MessageRequest(BaseModel):
    topic: str
    content: Any
    priority: int = Field(default=0, ge=0, le=10)
    delay_seconds: int = Field(default=0, ge=0, le=3600)
    ttl_seconds: Optional[int] = Field(default=None, ge=1)
    headers: Dict[str, str] = Field(default_factory=dict)

class ConsumerRequest(BaseModel):
    consumer_id: str
    topics: List[str]
    auto_acknowledge: bool = Field(default=True)
    max_messages_per_poll: int = Field(default=10, ge=1, le=100)
    poll_timeout: int = Field(default=5, ge=1, le=60)

# Response models
class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetricsResponse(BaseModel):
    component: str
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ============================================================================
# CLUSTER MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/cluster/initialize", response_model=StatusResponse)
async def initialize_cluster(config: ClusterConfigRequest):
    """Initialize a new cluster with specified configuration"""
    global cluster_manager
    
    try:
        cluster_config = ClusterConfig(
            cluster_name=config.name,
            max_nodes=config.max_nodes,
            auto_scaling_enabled=config.auto_scaling,
            scale_up_threshold=config.scale_up_threshold,
            scale_down_threshold=config.scale_down_threshold,
            health_check_interval=config.health_check_interval
        )
        
        cluster_manager = ClusterManager(cluster_config)
        await cluster_manager.initialize()
        
        return StatusResponse(
            status="success",
            message=f"Cluster '{config.name}' initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster/nodes/add", response_model=StatusResponse)
async def add_cluster_node(node_spec: NodeSpecRequest):
    """Add a new node to the cluster"""
    if not cluster_manager:
        raise HTTPException(status_code=400, detail="Cluster not initialized")
    
    try:
        spec = NodeSpec(
            node_type=node_spec.node_type,
            cpu_cores=node_spec.cpu_cores,
            memory_gb=node_spec.memory_gb,
            disk_gb=node_spec.disk_gb,
            gpu_count=node_spec.gpu_count
        )
        
        node_id = await cluster_manager.add_node(spec)
        
        return StatusResponse(
            status="success",
            message=f"Node {node_id} added to cluster"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cluster/nodes/{node_id}", response_model=StatusResponse)
async def remove_cluster_node(node_id: str = Path(...)):
    """Remove a node from the cluster"""
    if not cluster_manager:
        raise HTTPException(status_code=400, detail="Cluster not initialized")
    
    try:
        await cluster_manager.remove_node(node_id)
        
        return StatusResponse(
            status="success",
            message=f"Node {node_id} removed from cluster"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cluster/workloads/deploy", response_model=StatusResponse)
async def deploy_workload(workload: WorkloadRequest):
    """Deploy a workload to the cluster"""
    if not cluster_manager:
        raise HTTPException(status_code=400, detail="Cluster not initialized")
    
    try:
        resource_req = NodeSpec(
            node_type=workload.resource_requirements.node_type,
            cpu_cores=workload.resource_requirements.cpu_cores,
            memory_gb=workload.resource_requirements.memory_gb,
            disk_gb=workload.resource_requirements.disk_gb,
            gpu_count=workload.resource_requirements.gpu_count
        )
        
        await cluster_manager.deploy_workload(
            workload.workload_id,
            workload.workload_type,
            resource_req,
            workload.replicas,
            workload.labels
        )
        
        return StatusResponse(
            status="success",
            message=f"Workload {workload.workload_id} deployed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cluster/status", response_model=Dict[str, Any])
async def get_cluster_status():
    """Get current cluster status and metrics"""
    if not cluster_manager:
        raise HTTPException(status_code=400, detail="Cluster not initialized")
    
    try:
        status = await cluster_manager.get_cluster_status()
        metrics = await cluster_manager.get_metrics()
        
        return {
            "status": status,
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LOAD BALANCER ENDPOINTS
# ============================================================================

@router.post("/loadbalancer/initialize", response_model=StatusResponse)
async def initialize_load_balancer(config: LoadBalancerConfigRequest):
    """Initialize load balancer with specified configuration"""
    global load_balancer
    
    try:
        lb_config = LoadBalancerConfig(
            algorithm=config.algorithm,
            enable_circuit_breaker=config.enable_circuit_breaker,
            circuit_failure_threshold=config.circuit_failure_threshold,
            circuit_timeout=config.circuit_timeout,
            enable_rate_limiting=config.enable_rate_limiting,
            requests_per_second=config.requests_per_second
        )
        
        load_balancer = LoadBalancer(lb_config)
        await load_balancer.initialize()
        
        return StatusResponse(
            status="success",
            message="Load balancer initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/loadbalancer/backends/add", response_model=StatusResponse)
async def add_backend_server(server: BackendServerRequest):
    """Add a backend server to the load balancer"""
    if not load_balancer:
        raise HTTPException(status_code=400, detail="Load balancer not initialized")
    
    try:
        backend = BackendServer(
            server_id=server.server_id,
            host=server.host,
            port=server.port,
            weight=server.weight,
            labels=server.labels
        )
        
        await load_balancer.add_backend(backend)
        
        return StatusResponse(
            status="success",
            message=f"Backend server {server.server_id} added"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/loadbalancer/backends/{server_id}", response_model=StatusResponse)
async def remove_backend_server(server_id: str = Path(...)):
    """Remove a backend server from the load balancer"""
    if not load_balancer:
        raise HTTPException(status_code=400, detail="Load balancer not initialized")
    
    try:
        await load_balancer.remove_backend(server_id)
        
        return StatusResponse(
            status="success",
            message=f"Backend server {server_id} removed"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/loadbalancer/health-check/configure", response_model=StatusResponse)
async def configure_health_check(config: HealthCheckConfigRequest):
    """Configure health check settings"""
    if not load_balancer:
        raise HTTPException(status_code=400, detail="Load balancer not initialized")
    
    try:
        health_config = HealthCheckConfig(
            path=config.path,
            interval=config.interval,
            timeout=config.timeout,
            healthy_threshold=config.healthy_threshold,
            unhealthy_threshold=config.unhealthy_threshold
        )
        
        await load_balancer.configure_health_check(health_config)
        
        return StatusResponse(
            status="success",
            message="Health check configuration updated"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loadbalancer/backends", response_model=List[Dict[str, Any]])
async def list_backend_servers():
    """List all backend servers and their health status"""
    if not load_balancer:
        raise HTTPException(status_code=400, detail="Load balancer not initialized")
    
    try:
        backends = await load_balancer.list_backends()
        return [
            {
                "server_id": backend.server_id,
                "host": backend.host,
                "port": backend.port,
                "weight": backend.weight,
                "health": backend.health.value,
                "labels": backend.labels,
                "last_check": backend.last_health_check
            }
            for backend in backends
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/loadbalancer/metrics", response_model=MetricsResponse)
async def get_load_balancer_metrics():
    """Get load balancer performance metrics"""
    if not load_balancer:
        raise HTTPException(status_code=400, detail="Load balancer not initialized")
    
    try:
        metrics = await load_balancer.get_metrics()
        
        return MetricsResponse(
            component="load_balancer",
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SERVICE DISCOVERY ENDPOINTS
# ============================================================================

@router.post("/discovery/initialize", response_model=StatusResponse)
async def initialize_service_discovery():
    """Initialize service discovery system"""
    global service_discovery
    
    try:
        service_discovery = ServiceDiscovery()
        await service_discovery.initialize()
        
        return StatusResponse(
            status="success",
            message="Service discovery initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/discovery/services/register", response_model=StatusResponse)
async def register_service(service_config: ServiceConfigRequest, instance: ServiceInstanceRequest):
    """Register a new service instance"""
    if not service_discovery:
        raise HTTPException(status_code=400, detail="Service discovery not initialized")
    
    try:
        config = ServiceConfig(
            service_name=service_config.service_name,
            service_version=service_config.service_version,
            port=service_config.port,
            health_check_path=service_config.health_check_path,
            labels=service_config.labels,
            dependencies=service_config.dependencies
        )
        
        service_instance = ServiceInstance(
            instance_id=instance.instance_id,
            host=instance.host,
            port=instance.port,
            metadata=instance.metadata
        )
        
        await service_discovery.register_service(config, service_instance)
        
        return StatusResponse(
            status="success",
            message=f"Service {service_config.service_name} registered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/discovery/services/{service_name}/instances/{instance_id}", response_model=StatusResponse)
async def deregister_service_instance(
    service_name: str = Path(...),
    instance_id: str = Path(...)
):
    """Deregister a service instance"""
    if not service_discovery:
        raise HTTPException(status_code=400, detail="Service discovery not initialized")
    
    try:
        await service_discovery.deregister_service(service_name, instance_id)
        
        return StatusResponse(
            status="success",
            message=f"Service instance {instance_id} deregistered"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discovery/services/{service_name}/instances", response_model=List[Dict[str, Any]])
async def discover_service_instances(
    service_name: str = Path(...),
    healthy_only: bool = Query(default=True)
):
    """Discover instances of a specific service"""
    if not service_discovery:
        raise HTTPException(status_code=400, detail="Service discovery not initialized")
    
    try:
        instances = await service_discovery.discover_services(service_name, healthy_only)
        
        return [
            {
                "instance_id": instance.instance_id,
                "host": instance.host,
                "port": instance.port,
                "health": instance.health.value,
                "metadata": instance.metadata,
                "last_check": instance.last_health_check
            }
            for instance in instances
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discovery/services", response_model=List[Dict[str, Any]])
async def list_all_services():
    """List all registered services"""
    if not service_discovery:
        raise HTTPException(status_code=400, detail="Service discovery not initialized")
    
    try:
        services = await service_discovery.list_services()
        return services
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DISTRIBUTED CACHE ENDPOINTS
# ============================================================================

@router.post("/cache/initialize", response_model=StatusResponse)
async def initialize_distributed_cache(config: CacheConfigRequest):
    """Initialize distributed cache with specified configuration"""
    global distributed_cache
    
    try:
        cache_config = CacheConfig(
            cluster_name=config.cluster_name,
            consistency_level=config.consistency_level,
            replication_strategy=config.replication_strategy,
            replication_factor=config.replication_factor,
            eviction_policy=config.eviction_policy,
            default_ttl=config.ttl_seconds,
            enable_compression=config.enable_compression
        )
        
        distributed_cache = DistributedCache(cache_config)
        await distributed_cache.initialize()
        
        return StatusResponse(
            status="success",
            message=f"Distributed cache '{config.cluster_name}' initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/items", response_model=StatusResponse)
async def set_cache_item(item: CacheItemRequest):
    """Set a cache item with optional TTL and tags"""
    if not distributed_cache:
        raise HTTPException(status_code=400, detail="Distributed cache not initialized")
    
    try:
        cache_item = CacheItem(
            key=item.key,
            value=item.value,
            ttl=item.ttl_seconds,
            tags=item.tags
        )
        
        await distributed_cache.set(cache_item)
        
        return StatusResponse(
            status="success",
            message=f"Cache item '{item.key}' set successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/items/{key}", response_model=Dict[str, Any])
async def get_cache_item(key: str = Path(...)):
    """Get a cache item by key"""
    if not distributed_cache:
        raise HTTPException(status_code=400, detail="Distributed cache not initialized")
    
    try:
        item = await distributed_cache.get(key)
        
        if item is None:
            raise HTTPException(status_code=404, detail="Cache item not found")
        
        return {
            "key": item.key,
            "value": item.value,
            "ttl": item.ttl,
            "tags": item.tags,
            "created_at": item.created_at,
            "expires_at": item.expires_at
        }
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/items/{key}", response_model=StatusResponse)
async def delete_cache_item(key: str = Path(...)):
    """Delete a cache item by key"""
    if not distributed_cache:
        raise HTTPException(status_code=400, detail="Distributed cache not initialized")
    
    try:
        await distributed_cache.delete(key)
        
        return StatusResponse(
            status="success",
            message=f"Cache item '{key}' deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/query", response_model=List[Dict[str, Any]])
async def query_cache_items(query: CacheQueryRequest):
    """Query cache items by pattern or tags"""
    if not distributed_cache:
        raise HTTPException(status_code=400, detail="Distributed cache not initialized")
    
    try:
        items = await distributed_cache.query(query.pattern, query.tags, query.limit)
        
        return [
            {
                "key": item.key,
                "value": item.value,
                "tags": item.tags,
                "created_at": item.created_at,
                "expires_at": item.expires_at
            }
            for item in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/metrics", response_model=MetricsResponse)
async def get_cache_metrics():
    """Get distributed cache performance metrics"""
    if not distributed_cache:
        raise HTTPException(status_code=400, detail="Distributed cache not initialized")
    
    try:
        metrics = await distributed_cache.get_metrics()
        
        return MetricsResponse(
            component="distributed_cache",
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MESSAGE QUEUE ENDPOINTS
# ============================================================================

@router.post("/messagequeue/initialize", response_model=StatusResponse)
async def initialize_message_queue(config: MessageQueueConfigRequest):
    """Initialize distributed message queue with specified configuration"""
    global message_queue
    
    try:
        mq_config = MessageQueueConfig(
            cluster_name=config.cluster_name,
            delivery_mode=config.delivery_mode,
            priority_strategy=config.priority_strategy,
            enable_dead_letter_queue=config.enable_dead_letter,
            max_retry_count=config.max_retry_count,
            retry_delay_seconds=config.retry_delay,
            enable_persistence=config.enable_persistence
        )
        
        message_queue = DistributedMessageQueue(mq_config)
        await message_queue.initialize()
        
        return StatusResponse(
            status="success",
            message=f"Message queue '{config.cluster_name}' initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/messagequeue/messages", response_model=StatusResponse)
async def send_message(message: MessageRequest):
    """Send a message to a topic"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        msg = Message(
            topic=message.topic,
            content=message.content,
            priority=message.priority,
            headers=message.headers
        )
        
        message_id = await message_queue.send_message(
            msg, 
            delay_seconds=message.delay_seconds,
            ttl_seconds=message.ttl_seconds
        )
        
        return StatusResponse(
            status="success",
            message=f"Message sent with ID: {message_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/messagequeue/consumers", response_model=StatusResponse)
async def register_consumer(consumer: ConsumerRequest):
    """Register a message consumer"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        consumer_obj = Consumer(
            consumer_id=consumer.consumer_id,
            topics=consumer.topics,
            auto_acknowledge=consumer.auto_acknowledge,
            max_messages_per_poll=consumer.max_messages_per_poll,
            poll_timeout=consumer.poll_timeout
        )
        
        await message_queue.register_consumer(consumer_obj)
        
        return StatusResponse(
            status="success",
            message=f"Consumer '{consumer.consumer_id}' registered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/messagequeue/consumers/{consumer_id}", response_model=StatusResponse)
async def deregister_consumer(consumer_id: str = Path(...)):
    """Deregister a message consumer"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        await message_queue.deregister_consumer(consumer_id)
        
        return StatusResponse(
            status="success",
            message=f"Consumer '{consumer_id}' deregistered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messagequeue/messages/{consumer_id}", response_model=List[Dict[str, Any]])
async def poll_messages(
    consumer_id: str = Path(...),
    max_messages: int = Query(default=10, ge=1, le=100)
):
    """Poll messages for a specific consumer"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        messages = await message_queue.poll_messages(consumer_id, max_messages)
        
        return [
            {
                "message_id": msg.message_id,
                "topic": msg.topic,
                "content": msg.content,
                "priority": msg.priority,
                "headers": msg.headers,
                "timestamp": msg.timestamp,
                "status": msg.status.value
            }
            for msg in messages
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/messagequeue/messages/{message_id}/acknowledge", response_model=StatusResponse)
async def acknowledge_message(message_id: str = Path(...)):
    """Acknowledge message processing"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        await message_queue.acknowledge_message(message_id)
        
        return StatusResponse(
            status="success",
            message=f"Message '{message_id}' acknowledged successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messagequeue/topics/{topic}/metrics", response_model=Dict[str, Any])
async def get_topic_metrics(topic: str = Path(...)):
    """Get metrics for a specific topic"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        metrics = await message_queue.get_topic_metrics(topic)
        
        return {
            "topic": topic,
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messagequeue/metrics", response_model=MetricsResponse)
async def get_message_queue_metrics():
    """Get overall message queue performance metrics"""
    if not message_queue:
        raise HTTPException(status_code=400, detail="Message queue not initialized")
    
    try:
        metrics = await message_queue.get_metrics()
        
        return MetricsResponse(
            component="message_queue",
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# SYSTEM-WIDE ENDPOINTS
# ============================================================================

@router.get("/health", response_model=Dict[str, Any])
async def get_distributed_systems_health():
    """Get health status of all distributed system components"""
    try:
        health_status = {
            "cluster_manager": {
                "initialized": cluster_manager is not None,
                "status": "healthy" if cluster_manager else "not_initialized"
            },
            "load_balancer": {
                "initialized": load_balancer is not None,
                "status": "healthy" if load_balancer else "not_initialized"
            },
            "service_discovery": {
                "initialized": service_discovery is not None,
                "status": "healthy" if service_discovery else "not_initialized"
            },
            "distributed_cache": {
                "initialized": distributed_cache is not None,
                "status": "healthy" if distributed_cache else "not_initialized"
            },
            "message_queue": {
                "initialized": message_queue is not None,
                "status": "healthy" if message_queue else "not_initialized"
            }
        }
        
        # Get detailed health for initialized components
        if cluster_manager:
            health_status["cluster_manager"]["details"] = await cluster_manager.get_health()
        
        if load_balancer:
            health_status["load_balancer"]["details"] = await load_balancer.get_health()
        
        if service_discovery:
            health_status["service_discovery"]["details"] = await service_discovery.get_health()
        
        if distributed_cache:
            health_status["distributed_cache"]["details"] = await distributed_cache.get_health()
        
        if message_queue:
            health_status["message_queue"]["details"] = await message_queue.get_health()
        
        return {
            "overall_status": "healthy",
            "components": health_status,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_all_distributed_metrics():
    """Get comprehensive metrics from all distributed system components"""
    try:
        metrics = {
            "cluster_manager": {},
            "load_balancer": {},
            "service_discovery": {},
            "distributed_cache": {},
            "message_queue": {}
        }
        
        # Collect metrics from initialized components
        if cluster_manager:
            metrics["cluster_manager"] = await cluster_manager.get_metrics()
        
        if load_balancer:
            metrics["load_balancer"] = await load_balancer.get_metrics()
        
        if service_discovery:
            metrics["service_discovery"] = await service_discovery.get_metrics()
        
        if distributed_cache:
            metrics["distributed_cache"] = await distributed_cache.get_metrics()
        
        if message_queue:
            metrics["message_queue"] = await message_queue.get_metrics()
        
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown", response_model=StatusResponse)
async def shutdown_all_distributed_systems():
    """Gracefully shutdown all distributed system components"""
    global cluster_manager, load_balancer, service_discovery, distributed_cache, message_queue
    
    try:
        shutdown_results = []
        
        # Shutdown each component gracefully
        if cluster_manager:
            await cluster_manager.shutdown()
            cluster_manager = None
            shutdown_results.append("cluster_manager")
        
        if load_balancer:
            await load_balancer.shutdown()
            load_balancer = None
            shutdown_results.append("load_balancer")
        
        if service_discovery:
            await service_discovery.shutdown()
            service_discovery = None
            shutdown_results.append("service_discovery")
        
        if distributed_cache:
            await distributed_cache.shutdown()
            distributed_cache = None
            shutdown_results.append("distributed_cache")
        
        if message_queue:
            await message_queue.shutdown()
            message_queue = None
            shutdown_results.append("message_queue")
        
        return StatusResponse(
            status="success",
            message=f"Distributed systems shutdown: {', '.join(shutdown_results) if shutdown_results else 'none were running'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))