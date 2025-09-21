"""
Advanced Load Balancer
High-performance load balancing with multiple algorithms, health checks, and traffic management
"""

import asyncio
import logging
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import aiohttp
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class BalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    GEOGRAPHIC = "geographic"
    RESOURCE_BASED = "resource_based"


class BackendStatus(Enum):
    """Backend server status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    DISABLED = "disabled"
    WARMING_UP = "warming_up"


class TrafficDirection(Enum):
    """Traffic routing direction"""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


@dataclass
class HealthCheck:
    """Health check configuration"""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    path: str = "/health"
    method: str = "GET"
    expected_codes: List[int] = field(default_factory=lambda: [200])
    expected_body: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class BackendServer:
    """Backend server configuration"""
    server_id: str
    host: str
    port: int
    weight: int = 100
    max_connections: int = 1000
    status: BackendStatus = BackendStatus.HEALTHY
    health_check: HealthCheck = field(default_factory=HealthCheck)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime metrics
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_response_time: float = 0.0
    average_response_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_health_check: Optional[datetime] = None
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_available(self) -> bool:
        return (self.status == BackendStatus.HEALTHY and 
                self.current_connections < self.max_connections)
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)"""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def update_response_time(self, response_time: float) -> None:
        """Update response time metrics"""
        self.last_response_time = response_time
        
        # Calculate rolling average (simple exponential smoothing)
        alpha = 0.1
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (alpha * response_time + 
                                        (1 - alpha) * self.average_response_time)
    
    def record_request(self, success: bool, response_time: float = 0.0) -> None:
        """Record request statistics"""
        self.total_requests += 1
        
        if success:
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            if response_time > 0:
                self.update_response_time(response_time)
        else:
            self.failed_requests += 1
            self.consecutive_successes = 0
            self.consecutive_failures += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "server_id": self.server_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "max_connections": self.max_connections,
            "status": self.status.value,
            "current_connections": self.current_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate": self.error_rate,
            "last_response_time": self.last_response_time,
            "average_response_time": self.average_response_time,
            "load_factor": self.load_factor,
            "is_available": self.is_available,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "metadata": self.metadata
        }


@dataclass
class LoadBalancerRule:
    """Load balancer routing rule"""
    rule_id: str
    name: str
    priority: int
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    enabled: bool = True
    
    def matches_request(self, request_info: Dict[str, Any]) -> bool:
        """Check if rule matches the request"""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            if not self._evaluate_condition(condition, request_info):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], request_info: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type = condition.get("type")
        
        if condition_type == "path":
            pattern = condition.get("pattern", "")
            path = request_info.get("path", "")
            if condition.get("operator") == "starts_with":
                return path.startswith(pattern)
            elif condition.get("operator") == "equals":
                return path == pattern
            elif condition.get("operator") == "regex":
                import re
                return bool(re.match(pattern, path))
        
        elif condition_type == "header":
            header_name = condition.get("name", "").lower()
            header_value = condition.get("value", "")
            request_headers = request_info.get("headers", {})
            actual_value = request_headers.get(header_name, "")
            return actual_value == header_value
        
        elif condition_type == "source_ip":
            allowed_ips = condition.get("ips", [])
            client_ip = request_info.get("client_ip", "")
            return client_ip in allowed_ips
        
        return True


@dataclass
class TrafficMetrics:
    """Traffic and performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    requests_per_second: float = 0.0
    connections_per_second: float = 0.0
    
    def __post_init__(self):
        self.response_times: deque = deque(maxlen=1000)
        self.request_timestamps: deque = deque(maxlen=1000)
        
    def record_request(self, response_time: float, success: bool, 
                      bytes_sent: int = 0, bytes_received: int = 0) -> None:
        """Record request metrics"""
        now = time.time()
        
        self.total_requests += 1
        self.total_bytes_sent += bytes_sent
        self.total_bytes_received += bytes_received
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        # Record response time
        self.response_times.append(response_time)
        self.request_timestamps.append(now)
        
        # Update average response time
        if len(self.response_times) > 0:
            self.average_response_time = statistics.mean(self.response_times)
            
            # Calculate percentiles
            sorted_times = sorted(self.response_times)
            count = len(sorted_times)
            if count > 0:
                self.p95_response_time = sorted_times[int(count * 0.95)]
                self.p99_response_time = sorted_times[int(count * 0.99)]
        
        # Calculate requests per second (last minute)
        cutoff_time = now - 60
        recent_requests = [ts for ts in self.request_timestamps if ts > cutoff_time]
        self.requests_per_second = len(recent_requests) / 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": (self.failed_requests / max(1, self.total_requests)) * 100,
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "average_response_time": self.average_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "requests_per_second": self.requests_per_second
        }


class ConsistentHashRing:
    """Consistent hash ring for load balancing"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
    
    def add_server(self, server_id: str) -> None:
        """Add server to hash ring"""
        for i in range(self.virtual_nodes):
            key = self._hash(f"{server_id}:{i}")
            self.ring[key] = server_id
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_server(self, server_id: str) -> None:
        """Remove server from hash ring"""
        keys_to_remove = []
        for key, sid in self.ring.items():
            if sid == server_id:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_server(self, key: str) -> Optional[str]:
        """Get server for given key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find first server clockwise from hash
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to first server
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class LoadBalancer:
    """Advanced load balancer with multiple algorithms and health checking"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.config.get("name", "archon-lb")
        self.algorithm = BalancingAlgorithm(self.config.get("algorithm", "round_robin"))
        
        # Backend servers
        self.backends: Dict[str, BackendServer] = {}
        self.backend_pools: Dict[str, List[str]] = {"default": []}
        
        # Load balancing state
        self.round_robin_index = 0
        self.consistent_hash_ring = ConsistentHashRing()
        
        # Rules and routing
        self.rules: List[LoadBalancerRule] = []
        
        # Metrics and monitoring
        self.metrics = TrafficMetrics()
        self.backend_metrics: Dict[str, TrafficMetrics] = {}
        
        # Health checking
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Session persistence
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> backend_id
        self.session_timeout = timedelta(minutes=30)
        
        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    def add_backend(self, server_id: str, host: str, port: int, weight: int = 100,
                   pool: str = "default", health_check: HealthCheck = None) -> BackendServer:
        """Add backend server"""
        backend = BackendServer(
            server_id=server_id,
            host=host,
            port=port,
            weight=weight,
            health_check=health_check or HealthCheck()
        )
        
        self.backends[server_id] = backend
        self.backend_metrics[server_id] = TrafficMetrics()
        
        # Add to pool
        if pool not in self.backend_pools:
            self.backend_pools[pool] = []
        self.backend_pools[pool].append(server_id)
        
        # Add to consistent hash ring
        self.consistent_hash_ring.add_server(server_id)
        
        # Start health checking
        if backend.health_check.enabled:
            task = asyncio.create_task(self._health_check_backend(server_id))
            self.health_check_tasks[server_id] = task
        
        # Initialize circuit breaker
        self.circuit_breakers[server_id] = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure": None,
            "next_attempt": None
        }
        
        logger.info(f"Added backend server {server_id} ({host}:{port})")
        return backend
    
    def remove_backend(self, server_id: str) -> bool:
        """Remove backend server"""
        if server_id not in self.backends:
            return False
        
        # Cancel health check task
        if server_id in self.health_check_tasks:
            self.health_check_tasks[server_id].cancel()
            del self.health_check_tasks[server_id]
        
        # Remove from pools
        for pool_servers in self.backend_pools.values():
            if server_id in pool_servers:
                pool_servers.remove(server_id)
        
        # Remove from consistent hash ring
        self.consistent_hash_ring.remove_server(server_id)
        
        # Clean up
        del self.backends[server_id]
        del self.backend_metrics[server_id]
        del self.circuit_breakers[server_id]
        
        logger.info(f"Removed backend server {server_id}")
        return True
    
    async def route_request(self, request_info: Dict[str, Any], pool: str = "default") -> Optional[str]:
        """Route request to appropriate backend server"""
        # Apply routing rules first
        for rule in sorted(self.rules, key=lambda r: r.priority, reverse=True):
            if rule.matches_request(request_info):
                # Execute rule actions
                for action in rule.actions:
                    if action.get("type") == "route_to_pool":
                        pool = action.get("pool", pool)
                    elif action.get("type") == "set_algorithm":
                        # Temporarily override algorithm for this request
                        temp_algorithm = BalancingAlgorithm(action.get("algorithm"))
                        return await self._select_backend(request_info, pool, temp_algorithm)
        
        return await self._select_backend(request_info, pool)
    
    async def _select_backend(self, request_info: Dict[str, Any], pool: str = "default",
                            algorithm: BalancingAlgorithm = None) -> Optional[str]:
        """Select backend server using specified algorithm"""
        algorithm = algorithm or self.algorithm
        
        # Get available backends from pool
        pool_backends = self.backend_pools.get(pool, [])
        available_backends = [
            server_id for server_id in pool_backends
            if (server_id in self.backends and 
                self.backends[server_id].is_available and
                self._is_circuit_breaker_closed(server_id))
        ]
        
        if not available_backends:
            logger.warning(f"No available backends in pool {pool}")
            return None
        
        # Check for sticky session
        session_id = request_info.get("session_id")
        if session_id and session_id in self.sticky_sessions:
            sticky_backend = self.sticky_sessions[session_id]
            if sticky_backend in available_backends:
                return sticky_backend
        
        # Apply load balancing algorithm
        selected_backend = None
        
        if algorithm == BalancingAlgorithm.ROUND_ROBIN:
            selected_backend = self._round_robin_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.LEAST_CONNECTIONS:
            selected_backend = self._least_connections_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            selected_backend = self._weighted_round_robin_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.LEAST_RESPONSE_TIME:
            selected_backend = self._least_response_time_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.IP_HASH:
            client_ip = request_info.get("client_ip", "")
            selected_backend = self._ip_hash_select(available_backends, client_ip)
        
        elif algorithm == BalancingAlgorithm.CONSISTENT_HASH:
            hash_key = request_info.get("hash_key", request_info.get("client_ip", ""))
            selected_backend = self.consistent_hash_ring.get_server(hash_key)
            if selected_backend not in available_backends:
                selected_backend = self._round_robin_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.RANDOM:
            selected_backend = random.choice(available_backends)
        
        elif algorithm == BalancingAlgorithm.WEIGHTED_RANDOM:
            selected_backend = self._weighted_random_select(available_backends)
        
        elif algorithm == BalancingAlgorithm.RESOURCE_BASED:
            selected_backend = self._resource_based_select(available_backends)
        
        else:
            selected_backend = self._round_robin_select(available_backends)
        
        # Create sticky session if requested
        if session_id and selected_backend:
            self.sticky_sessions[session_id] = selected_backend
        
        return selected_backend
    
    def _round_robin_select(self, backends: List[str]) -> str:
        """Round-robin selection"""
        if not backends:
            return None
        
        selected = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_select(self, backends: List[str]) -> str:
        """Select backend with least connections"""
        if not backends:
            return None
        
        return min(backends, key=lambda bid: self.backends[bid].current_connections)
    
    def _weighted_round_robin_select(self, backends: List[str]) -> str:
        """Weighted round-robin selection"""
        if not backends:
            return None
        
        # Build weighted list
        weighted_list = []
        for backend_id in backends:
            weight = self.backends[backend_id].weight
            weighted_list.extend([backend_id] * weight)
        
        if not weighted_list:
            return backends[0]
        
        selected = weighted_list[self.round_robin_index % len(weighted_list)]
        self.round_robin_index += 1
        return selected
    
    def _least_response_time_select(self, backends: List[str]) -> str:
        """Select backend with lowest average response time"""
        if not backends:
            return None
        
        return min(backends, key=lambda bid: self.backends[bid].average_response_time or float('inf'))
    
    def _ip_hash_select(self, backends: List[str], client_ip: str) -> str:
        """Select backend based on client IP hash"""
        if not backends:
            return None
        
        if not client_ip:
            return self._round_robin_select(backends)
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        return backends[hash_value % len(backends)]
    
    def _weighted_random_select(self, backends: List[str]) -> str:
        """Weighted random selection"""
        if not backends:
            return None
        
        weights = [self.backends[bid].weight for bid in backends]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(backends)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return backends[i]
        
        return backends[-1]
    
    def _resource_based_select(self, backends: List[str]) -> str:
        """Select backend based on resource utilization"""
        if not backends:
            return None
        
        # Select backend with lowest load factor
        return min(backends, key=lambda bid: self.backends[bid].load_factor)
    
    async def handle_request(self, request_info: Dict[str, Any], pool: str = "default") -> Dict[str, Any]:
        """Handle incoming request with load balancing"""
        start_time = time.time()
        
        # Route request to backend
        backend_id = await self.route_request(request_info, pool)
        
        if not backend_id:
            return {
                "success": False,
                "error": "No available backends",
                "backend_id": None,
                "response_time": 0.0
            }
        
        backend = self.backends[backend_id]
        
        # Increment connection count
        backend.current_connections += 1
        
        try:
            # Simulate request forwarding (in production, would use aiohttp or similar)
            response_time = await self._forward_request(backend, request_info)
            
            # Record successful request
            backend.record_request(True, response_time)
            self.backend_metrics[backend_id].record_request(response_time, True)
            self.metrics.record_request(response_time, True)
            
            # Reset circuit breaker on success
            self.circuit_breakers[backend_id]["failure_count"] = 0
            if self.circuit_breakers[backend_id]["state"] == "half_open":
                self.circuit_breakers[backend_id]["state"] = "closed"
            
            return {
                "success": True,
                "backend_id": backend_id,
                "backend_url": backend.url,
                "response_time": response_time,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            # Record failed request
            backend.record_request(False)
            self.backend_metrics[backend_id].record_request(0.0, False)
            self.metrics.record_request(0.0, False)
            
            # Update circuit breaker
            self._update_circuit_breaker(backend_id, False)
            
            logger.error(f"Request to backend {backend_id} failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "backend_id": backend_id,
                "response_time": 0.0
            }
            
        finally:
            # Decrement connection count
            backend.current_connections = max(0, backend.current_connections - 1)
    
    async def _forward_request(self, backend: BackendServer, request_info: Dict[str, Any]) -> float:
        """Forward request to backend server"""
        start_time = time.time()
        
        try:
            # Simulate network delay and processing
            import random
            delay = random.uniform(0.01, 0.1)  # 10ms to 100ms
            await asyncio.sleep(delay)
            
            # Simulate occasional failures
            if random.random() < 0.05:  # 5% failure rate
                raise Exception("Simulated backend error")
            
            return time.time() - start_time
            
        except Exception:
            raise
    
    async def _health_check_backend(self, backend_id: str) -> None:
        """Perform health checks on backend server"""
        backend = self.backends.get(backend_id)
        if not backend:
            return
        
        health_check = backend.health_check
        
        while backend_id in self.backends:
            try:
                # Perform health check
                is_healthy = await self._perform_health_check(backend)
                backend.last_health_check = datetime.now()
                
                if is_healthy:
                    backend.consecutive_successes += 1
                    backend.consecutive_failures = 0
                    
                    # Mark as healthy if it meets threshold
                    if (backend.status != BackendStatus.HEALTHY and
                        backend.consecutive_successes >= health_check.healthy_threshold):
                        backend.status = BackendStatus.HEALTHY
                        logger.info(f"Backend {backend_id} is now healthy")
                else:
                    backend.consecutive_failures += 1
                    backend.consecutive_successes = 0
                    
                    # Mark as unhealthy if it meets threshold
                    if (backend.status == BackendStatus.HEALTHY and
                        backend.consecutive_failures >= health_check.unhealthy_threshold):
                        backend.status = BackendStatus.UNHEALTHY
                        logger.warning(f"Backend {backend_id} is now unhealthy")
                
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for backend {backend_id}: {str(e)}")
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _perform_health_check(self, backend: BackendServer) -> bool:
        """Perform actual health check"""
        try:
            health_check = backend.health_check
            url = f"{backend.url}{health_check.path}"
            
            # Simulate health check (in production, would use aiohttp)
            await asyncio.sleep(0.01)  # Simulate network call
            
            # Simulate occasional health check failures
            return random.random() > 0.1  # 90% success rate
            
        except Exception as e:
            logger.debug(f"Health check failed for {backend.server_id}: {str(e)}")
            return False
    
    def _is_circuit_breaker_closed(self, backend_id: str) -> bool:
        """Check if circuit breaker allows requests"""
        cb = self.circuit_breakers.get(backend_id, {})
        state = cb.get("state", "closed")
        
        if state == "closed":
            return True
        elif state == "open":
            # Check if it's time to try half-open
            next_attempt = cb.get("next_attempt")
            if next_attempt and datetime.now() > next_attempt:
                cb["state"] = "half_open"
                return True
            return False
        elif state == "half_open":
            return True
        
        return False
    
    def _update_circuit_breaker(self, backend_id: str, success: bool) -> None:
        """Update circuit breaker state"""
        cb = self.circuit_breakers[backend_id]
        
        if success:
            cb["failure_count"] = 0
            if cb["state"] == "half_open":
                cb["state"] = "closed"
        else:
            cb["failure_count"] += 1
            cb["last_failure"] = datetime.now()
            
            # Open circuit breaker if failure threshold reached
            failure_threshold = 5  # Configurable
            if cb["failure_count"] >= failure_threshold and cb["state"] == "closed":
                cb["state"] = "open"
                cb["next_attempt"] = datetime.now() + timedelta(seconds=60)  # 1 minute timeout
                logger.warning(f"Circuit breaker opened for backend {backend_id}")
    
    def add_rule(self, rule: LoadBalancerRule) -> None:
        """Add routing rule"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove routing rule"""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                logger.info(f"Removed routing rule: {rule.name}")
                return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status and metrics"""
        healthy_backends = len([b for b in self.backends.values() 
                              if b.status == BackendStatus.HEALTHY])
        
        return {
            "name": self.name,
            "algorithm": self.algorithm.value,
            "total_backends": len(self.backends),
            "healthy_backends": healthy_backends,
            "unhealthy_backends": len(self.backends) - healthy_backends,
            "total_rules": len(self.rules),
            "metrics": self.metrics.to_dict(),
            "backend_pools": {pool: len(servers) for pool, servers in self.backend_pools.items()}
        }
    
    def get_backend_status(self, backend_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of specific backend or all backends"""
        if backend_id:
            backend = self.backends.get(backend_id)
            if backend:
                return backend.to_dict()
            return None
        else:
            return [backend.to_dict() for backend in self.backends.values()]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        backend_metrics = {}
        for backend_id, metrics in self.backend_metrics.items():
            backend_metrics[backend_id] = metrics.to_dict()
        
        return {
            "overall": self.metrics.to_dict(),
            "backends": backend_metrics,
            "circuit_breakers": {
                backend_id: {
                    "state": cb["state"],
                    "failure_count": cb["failure_count"],
                    "last_failure": cb["last_failure"].isoformat() if cb["last_failure"] else None
                }
                for backend_id, cb in self.circuit_breakers.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        # Cancel all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        self.health_check_tasks.clear()
        logger.info("Load balancer cleanup completed")