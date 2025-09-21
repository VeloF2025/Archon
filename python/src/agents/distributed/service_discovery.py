"""
Advanced Service Discovery
Dynamic service registration, discovery, and health monitoring for distributed systems
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import aiohttp
from collections import defaultdict, deque
import socket

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status in discovery registry"""
    REGISTERING = "registering"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    DEREGISTERED = "deregistered"


class ServiceProtocol(Enum):
    """Service communication protocols"""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    GRPC = "grpc"
    WEBSOCKET = "websocket"


class DiscoveryStrategy(Enum):
    """Service discovery strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    LOAD_BASED = "load_based"
    PRIORITY_BASED = "priority_based"


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    host: str
    port: int
    protocol: ServiceProtocol = ServiceProtocol.HTTP
    path: str = "/"
    weight: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        """Get full URL for endpoint"""
        if self.protocol in [ServiceProtocol.HTTP, ServiceProtocol.HTTPS]:
            return f"{self.protocol.value}://{self.host}:{self.port}{self.path}"
        else:
            return f"{self.host}:{self.port}"
    
    @property
    def address(self) -> str:
        """Get address string"""
        return f"{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol.value,
            "path": self.path,
            "weight": self.weight,
            "url": self.url,
            "address": self.address,
            "metadata": self.metadata
        }


@dataclass
class HealthCheck:
    """Health check configuration for service"""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    path: str = "/health"
    method: str = "GET"
    expected_status: List[int] = field(default_factory=lambda: [200])
    expected_body: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServiceInstance:
    """Registered service instance"""
    instance_id: str
    service_name: str
    version: str
    endpoints: List[ServiceEndpoint]
    status: ServiceStatus = ServiceStatus.REGISTERING
    health_check: Optional[HealthCheck] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime information
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    active_connections: int = 0
    
    @property
    def is_healthy(self) -> bool:
        """Check if service instance is healthy"""
        if self.status != ServiceStatus.HEALTHY:
            return False
        
        # Check heartbeat (must be within last 2 minutes)
        if self.last_heartbeat:
            return datetime.now() - self.last_heartbeat < timedelta(minutes=2)
        
        return False
    
    @property
    def primary_endpoint(self) -> Optional[ServiceEndpoint]:
        """Get primary endpoint (first HTTP/HTTPS endpoint)"""
        for endpoint in self.endpoints:
            if endpoint.protocol in [ServiceProtocol.HTTP, ServiceProtocol.HTTPS]:
                return endpoint
        return self.endpoints[0] if self.endpoints else None
    
    def get_endpoint_by_protocol(self, protocol: ServiceProtocol) -> Optional[ServiceEndpoint]:
        """Get endpoint by protocol"""
        for endpoint in self.endpoints:
            if endpoint.protocol == protocol:
                return endpoint
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "instance_id": self.instance_id,
            "service_name": self.service_name,
            "version": self.version,
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "status": self.status.value,
            "health_check": self.health_check.to_dict() if self.health_check else None,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "is_healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_requests": self.total_requests,
            "active_connections": self.active_connections
        }


@dataclass
class ServiceQuery:
    """Query parameters for service discovery"""
    service_name: str
    version: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    protocol: Optional[ServiceProtocol] = None
    healthy_only: bool = True
    max_instances: Optional[int] = None
    strategy: DiscoveryStrategy = DiscoveryStrategy.ROUND_ROBIN


@dataclass
class ServiceWatcher:
    """Service change watcher"""
    watcher_id: str
    service_name: str
    callback: Callable[[str, ServiceInstance, str], None]  # action, instance, event_type
    filters: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class ServiceRegistry:
    """Service registry for managing service instances"""
    
    def __init__(self):
        # Service instances indexed by service name
        self.services: Dict[str, Dict[str, ServiceInstance]] = defaultdict(dict)
        
        # Service watchers
        self.watchers: Dict[str, ServiceWatcher] = {}
        
        # Discovery state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Metrics
        self.registry_metrics = {
            "total_services": 0,
            "total_instances": 0,
            "healthy_instances": 0,
            "registrations": 0,
            "deregistrations": 0,
            "health_checks_performed": 0,
            "discovery_requests": 0
        }
    
    def register_service(self, instance: ServiceInstance) -> bool:
        """Register service instance"""
        try:
            service_name = instance.service_name
            instance_id = instance.instance_id
            
            # Add to registry
            self.services[service_name][instance_id] = instance
            
            # Update metrics
            self.registry_metrics["total_instances"] = sum(len(instances) 
                                                         for instances in self.services.values())
            self.registry_metrics["total_services"] = len(self.services)
            self.registry_metrics["registrations"] += 1
            
            # Notify watchers
            self._notify_watchers(service_name, instance, "registered")
            
            logger.info(f"Registered service instance {instance_id} for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service instance: {str(e)}")
            return False
    
    def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister service instance"""
        try:
            if service_name not in self.services:
                return False
            
            if instance_id not in self.services[service_name]:
                return False
            
            instance = self.services[service_name][instance_id]
            instance.status = ServiceStatus.DEREGISTERED
            
            # Remove from registry
            del self.services[service_name][instance_id]
            
            # Clean up empty service entries
            if not self.services[service_name]:
                del self.services[service_name]
            
            # Update metrics
            self.registry_metrics["total_instances"] = sum(len(instances) 
                                                         for instances in self.services.values())
            self.registry_metrics["total_services"] = len(self.services)
            self.registry_metrics["deregistrations"] += 1
            
            # Notify watchers
            self._notify_watchers(service_name, instance, "deregistered")
            
            logger.info(f"Deregistered service instance {instance_id} for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service instance: {str(e)}")
            return False
    
    def update_instance_status(self, service_name: str, instance_id: str, status: ServiceStatus) -> bool:
        """Update service instance status"""
        try:
            if service_name not in self.services:
                return False
            
            if instance_id not in self.services[service_name]:
                return False
            
            instance = self.services[service_name][instance_id]
            old_status = instance.status
            instance.status = status
            
            # Update metrics
            self._update_health_metrics()
            
            # Notify watchers if status changed
            if old_status != status:
                self._notify_watchers(service_name, instance, "status_changed")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update instance status: {str(e)}")
            return False
    
    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Update service instance heartbeat"""
        try:
            if service_name not in self.services:
                return False
            
            if instance_id not in self.services[service_name]:
                return False
            
            instance = self.services[service_name][instance_id]
            instance.last_heartbeat = datetime.now()
            
            # If instance was unhealthy, mark as healthy on successful heartbeat
            if instance.status == ServiceStatus.UNHEALTHY:
                instance.status = ServiceStatus.HEALTHY
                instance.consecutive_failures = 0
                self._notify_watchers(service_name, instance, "recovered")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat: {str(e)}")
            return False
    
    def discover_services(self, query: ServiceQuery) -> List[ServiceInstance]:
        """Discover services matching query"""
        try:
            self.registry_metrics["discovery_requests"] += 1
            
            service_name = query.service_name
            
            if service_name not in self.services:
                return []
            
            # Get all instances for the service
            instances = list(self.services[service_name].values())
            
            # Apply filters
            filtered_instances = self._apply_filters(instances, query)
            
            # Apply discovery strategy
            selected_instances = self._apply_discovery_strategy(filtered_instances, query)
            
            return selected_instances
            
        except Exception as e:
            logger.error(f"Failed to discover services: {str(e)}")
            return []
    
    def _apply_filters(self, instances: List[ServiceInstance], query: ServiceQuery) -> List[ServiceInstance]:
        """Apply query filters to instances"""
        filtered = instances
        
        # Health filter
        if query.healthy_only:
            filtered = [instance for instance in filtered if instance.is_healthy]
        
        # Version filter
        if query.version:
            filtered = [instance for instance in filtered if instance.version == query.version]
        
        # Tags filter
        if query.tags:
            filtered = [instance for instance in filtered 
                       if query.tags.issubset(instance.tags)]
        
        # Metadata filters
        for key, value in query.metadata_filters.items():
            filtered = [instance for instance in filtered 
                       if instance.metadata.get(key) == value]
        
        # Protocol filter
        if query.protocol:
            filtered = [instance for instance in filtered 
                       if instance.get_endpoint_by_protocol(query.protocol)]
        
        return filtered
    
    def _apply_discovery_strategy(self, instances: List[ServiceInstance], 
                                query: ServiceQuery) -> List[ServiceInstance]:
        """Apply discovery strategy to select instances"""
        if not instances:
            return []
        
        strategy = query.strategy
        max_instances = query.max_instances or len(instances)
        
        if strategy == DiscoveryStrategy.ROUND_ROBIN:
            # Round-robin selection
            service_name = query.service_name
            counter = self.round_robin_counters[service_name]
            selected = []
            
            for i in range(min(max_instances, len(instances))):
                idx = (counter + i) % len(instances)
                selected.append(instances[idx])
            
            self.round_robin_counters[service_name] = (counter + max_instances) % len(instances)
            return selected
        
        elif strategy == DiscoveryStrategy.RANDOM:
            import random
            return random.sample(instances, min(max_instances, len(instances)))
        
        elif strategy == DiscoveryStrategy.LEAST_CONNECTIONS:
            # Sort by active connections
            sorted_instances = sorted(instances, key=lambda x: x.active_connections)
            return sorted_instances[:max_instances]
        
        elif strategy == DiscoveryStrategy.LOAD_BASED:
            # Sort by total requests (inverse load balancing)
            sorted_instances = sorted(instances, key=lambda x: x.total_requests)
            return sorted_instances[:max_instances]
        
        elif strategy == DiscoveryStrategy.PRIORITY_BASED:
            # Sort by weight (higher weight = higher priority)
            def get_weight(instance):
                endpoint = instance.primary_endpoint
                return endpoint.weight if endpoint else 0
            
            sorted_instances = sorted(instances, key=get_weight, reverse=True)
            return sorted_instances[:max_instances]
        
        else:
            # Default: return first N instances
            return instances[:max_instances]
    
    def add_watcher(self, watcher: ServiceWatcher) -> None:
        """Add service watcher"""
        self.watchers[watcher.watcher_id] = watcher
        logger.info(f"Added service watcher {watcher.watcher_id} for {watcher.service_name}")
    
    def remove_watcher(self, watcher_id: str) -> bool:
        """Remove service watcher"""
        if watcher_id in self.watchers:
            watcher = self.watchers[watcher_id]
            del self.watchers[watcher_id]
            logger.info(f"Removed service watcher {watcher_id}")
            return True
        return False
    
    def _notify_watchers(self, service_name: str, instance: ServiceInstance, event_type: str) -> None:
        """Notify watchers of service changes"""
        for watcher in self.watchers.values():
            if (watcher.active and 
                watcher.service_name == service_name):
                try:
                    # Apply watcher filters
                    if self._watcher_matches(watcher, instance):
                        watcher.callback(event_type, instance, event_type)
                except Exception as e:
                    logger.error(f"Error notifying watcher {watcher.watcher_id}: {str(e)}")
    
    def _watcher_matches(self, watcher: ServiceWatcher, instance: ServiceInstance) -> bool:
        """Check if instance matches watcher filters"""
        for key, value in watcher.filters.items():
            if key == "version" and instance.version != value:
                return False
            elif key == "tags" and not set(value).issubset(instance.tags):
                return False
            elif key in instance.metadata and instance.metadata[key] != value:
                return False
        return True
    
    def _update_health_metrics(self) -> None:
        """Update health-related metrics"""
        healthy_count = 0
        for service_instances in self.services.values():
            for instance in service_instances.values():
                if instance.is_healthy:
                    healthy_count += 1
        
        self.registry_metrics["healthy_instances"] = healthy_count
    
    def get_service_names(self) -> List[str]:
        """Get list of all service names"""
        return list(self.services.keys())
    
    def get_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all instances for a service"""
        return list(self.services.get(service_name, {}).values())
    
    def get_instance(self, service_name: str, instance_id: str) -> Optional[ServiceInstance]:
        """Get specific service instance"""
        return self.services.get(service_name, {}).get(instance_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics"""
        self._update_health_metrics()
        return self.registry_metrics.copy()


class ServiceDiscovery:
    """Main service discovery system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.registry = ServiceRegistry()
        
        # Health checking
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_check_interval = self.config.get("health_check_interval", 30)
        
        # Cleanup tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = self.config.get("cleanup_interval", 300)  # 5 minutes
        
        # Service discovery cache
        self.discovery_cache: Dict[str, Tuple[List[ServiceInstance], datetime]] = {}
        self.cache_ttl = timedelta(seconds=self.config.get("cache_ttl", 30))
        
        self.running = False
    
    async def start(self) -> None:
        """Start service discovery system"""
        if self.running:
            return
        
        self.running = True
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Service discovery system started")
    
    async def stop(self) -> None:
        """Stop service discovery system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        self.health_check_tasks.clear()
        
        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service discovery system stopped")
    
    async def register_service(self, service_name: str, version: str, endpoints: List[Dict[str, Any]],
                             health_check: Optional[Dict[str, Any]] = None,
                             tags: Set[str] = None, metadata: Dict[str, Any] = None,
                             instance_id: str = None) -> str:
        """Register a service instance"""
        if not instance_id:
            instance_id = str(uuid.uuid4())
        
        # Convert endpoint dictionaries to ServiceEndpoint objects
        endpoint_objects = []
        for ep_data in endpoints:
            endpoint = ServiceEndpoint(
                host=ep_data["host"],
                port=ep_data["port"],
                protocol=ServiceProtocol(ep_data.get("protocol", "http")),
                path=ep_data.get("path", "/"),
                weight=ep_data.get("weight", 100),
                metadata=ep_data.get("metadata", {})
            )
            endpoint_objects.append(endpoint)
        
        # Create health check configuration
        health_check_obj = None
        if health_check:
            health_check_obj = HealthCheck(
                enabled=health_check.get("enabled", True),
                interval_seconds=health_check.get("interval_seconds", 30),
                timeout_seconds=health_check.get("timeout_seconds", 10),
                failure_threshold=health_check.get("failure_threshold", 3),
                success_threshold=health_check.get("success_threshold", 2),
                path=health_check.get("path", "/health"),
                method=health_check.get("method", "GET"),
                expected_status=health_check.get("expected_status", [200])
            )
        
        # Create service instance
        instance = ServiceInstance(
            instance_id=instance_id,
            service_name=service_name,
            version=version,
            endpoints=endpoint_objects,
            health_check=health_check_obj,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        # Register in registry
        success = self.registry.register_service(instance)
        
        if success:
            # Start health checking if enabled
            if health_check_obj and health_check_obj.enabled:
                task = asyncio.create_task(self._health_check_loop(instance))
                self.health_check_tasks[instance_id] = task
            
            # Mark as healthy initially
            instance.status = ServiceStatus.HEALTHY
            instance.last_heartbeat = datetime.now()
            
        return instance_id if success else None
    
    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        # Stop health checking
        if instance_id in self.health_check_tasks:
            self.health_check_tasks[instance_id].cancel()
            try:
                await self.health_check_tasks[instance_id]
            except asyncio.CancelledError:
                pass
            del self.health_check_tasks[instance_id]
        
        # Deregister from registry
        return self.registry.deregister_service(service_name, instance_id)
    
    async def discover_service(self, service_name: str, version: str = None,
                             tags: Set[str] = None, protocol: str = None,
                             strategy: str = "round_robin",
                             max_instances: int = None,
                             use_cache: bool = True) -> List[Dict[str, Any]]:
        """Discover service instances"""
        # Check cache first
        cache_key = f"{service_name}:{version}:{str(sorted(tags or []))}:{protocol}:{strategy}"
        
        if use_cache and cache_key in self.discovery_cache:
            cached_instances, cache_time = self.discovery_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return [instance.to_dict() for instance in cached_instances[:max_instances]]
        
        # Create query
        query = ServiceQuery(
            service_name=service_name,
            version=version,
            tags=tags or set(),
            protocol=ServiceProtocol(protocol) if protocol else None,
            strategy=DiscoveryStrategy(strategy),
            max_instances=max_instances,
            healthy_only=True
        )
        
        # Discover services
        instances = self.registry.discover_services(query)
        
        # Cache results
        if use_cache:
            self.discovery_cache[cache_key] = (instances, datetime.now())
        
        # Convert to dictionaries
        return [instance.to_dict() for instance in instances]
    
    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Send heartbeat for service instance"""
        return self.registry.heartbeat(service_name, instance_id)
    
    def watch_service(self, service_name: str, callback: Callable[[str, Dict[str, Any], str], None],
                     filters: Dict[str, Any] = None) -> str:
        """Watch for service changes"""
        watcher_id = str(uuid.uuid4())
        
        # Wrapper to convert ServiceInstance to dict
        def wrapped_callback(action: str, instance: ServiceInstance, event_type: str) -> None:
            callback(action, instance.to_dict(), event_type)
        
        watcher = ServiceWatcher(
            watcher_id=watcher_id,
            service_name=service_name,
            callback=wrapped_callback,
            filters=filters or {}
        )
        
        self.registry.add_watcher(watcher)
        return watcher_id
    
    def unwatch_service(self, watcher_id: str) -> bool:
        """Stop watching service changes"""
        return self.registry.remove_watcher(watcher_id)
    
    async def _health_check_loop(self, instance: ServiceInstance) -> None:
        """Health check loop for service instance"""
        health_check = instance.health_check
        if not health_check or not health_check.enabled:
            return
        
        while self.running:
            try:
                # Perform health check
                is_healthy = await self._perform_health_check(instance)
                instance.last_health_check = datetime.now()
                
                self.registry.registry_metrics["health_checks_performed"] += 1
                
                if is_healthy:
                    instance.consecutive_successes += 1
                    instance.consecutive_failures = 0
                    
                    # Mark as healthy if threshold met
                    if (instance.status != ServiceStatus.HEALTHY and
                        instance.consecutive_successes >= health_check.success_threshold):
                        self.registry.update_instance_status(
                            instance.service_name, instance.instance_id, ServiceStatus.HEALTHY
                        )
                else:
                    instance.consecutive_failures += 1
                    instance.consecutive_successes = 0
                    
                    # Mark as unhealthy if threshold met
                    if (instance.status == ServiceStatus.HEALTHY and
                        instance.consecutive_failures >= health_check.failure_threshold):
                        self.registry.update_instance_status(
                            instance.service_name, instance.instance_id, ServiceStatus.UNHEALTHY
                        )
                
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {instance.instance_id}: {str(e)}")
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _perform_health_check(self, instance: ServiceInstance) -> bool:
        """Perform actual health check on service instance"""
        health_check = instance.health_check
        if not health_check:
            return True
        
        endpoint = instance.primary_endpoint
        if not endpoint:
            return False
        
        try:
            url = f"{endpoint.url.rstrip('/')}{health_check.path}"
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=health_check.timeout_seconds)
            ) as session:
                
                async with session.request(
                    method=health_check.method,
                    url=url,
                    headers=health_check.headers
                ) as response:
                    
                    # Check status code
                    if response.status not in health_check.expected_status:
                        return False
                    
                    # Check response body if expected
                    if health_check.expected_body:
                        body = await response.text()
                        if health_check.expected_body not in body:
                            return False
                    
                    return True
                    
        except Exception as e:
            logger.debug(f"Health check failed for {instance.instance_id}: {str(e)}")
            return False
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired instances and cache entries"""
        while self.running:
            try:
                await self._cleanup_expired_instances()
                await self._cleanup_discovery_cache()
                await asyncio.sleep(self.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_instances(self) -> None:
        """Remove expired service instances"""
        now = datetime.now()
        expired_instances = []
        
        for service_name, instances in self.registry.services.items():
            for instance_id, instance in instances.items():
                # Remove instances that haven't sent heartbeat in 5 minutes
                if (instance.last_heartbeat and 
                    now - instance.last_heartbeat > timedelta(minutes=5)):
                    expired_instances.append((service_name, instance_id))
        
        # Remove expired instances
        for service_name, instance_id in expired_instances:
            await self.deregister_service(service_name, instance_id)
            logger.info(f"Removed expired instance {instance_id} for {service_name}")
    
    async def _cleanup_discovery_cache(self) -> None:
        """Clean up expired cache entries"""
        now = datetime.now()
        expired_keys = []
        
        for cache_key, (instances, cache_time) in self.discovery_cache.items():
            if now - cache_time > self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.discovery_cache[key]
    
    def get_service_names(self) -> List[str]:
        """Get list of all registered service names"""
        return self.registry.get_service_names()
    
    def get_service_instances(self, service_name: str) -> List[Dict[str, Any]]:
        """Get all instances for a service"""
        instances = self.registry.get_service_instances(service_name)
        return [instance.to_dict() for instance in instances]
    
    def get_instance(self, service_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get specific service instance"""
        instance = self.registry.get_instance(service_name, instance_id)
        return instance.to_dict() if instance else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service discovery status"""
        metrics = self.registry.get_metrics()
        
        return {
            "running": self.running,
            "total_services": metrics["total_services"],
            "total_instances": metrics["total_instances"],
            "healthy_instances": metrics["healthy_instances"],
            "unhealthy_instances": metrics["total_instances"] - metrics["healthy_instances"],
            "active_watchers": len(self.registry.watchers),
            "health_check_tasks": len(self.health_check_tasks),
            "cache_entries": len(self.discovery_cache),
            "metrics": metrics
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        return self.registry.get_metrics()