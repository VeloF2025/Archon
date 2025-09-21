"""
High-Performance Connection Pool Management

Implements intelligent connection pooling for databases, APIs, and external services:
- Dynamic connection scaling
- Health monitoring and automatic recovery
- Load balancing across connections
- Connection lifecycle management
- Performance metrics and optimization

Features:
- Adaptive pool sizing based on load
- Connection recycling and reuse
- Automatic failover and recovery
- Real-time health checks
- Connection cost optimization
"""

import asyncio
import logging
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from contextlib import contextmanager, asynccontextmanager
import weakref
import queue
import random
from pathlib import Path

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    """Connection status states"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"
    HEALTH_CHECKING = "health_checking"

class PoolStrategy(Enum):
    """Connection pool strategies"""
    FIXED = "fixed"                    # Fixed number of connections
    DYNAMIC = "dynamic"                # Scale up/down based on load
    LAZY = "lazy"                      # Create connections on demand
    EAGER = "eager"                    # Pre-create all connections
    ADAPTIVE = "adaptive"              # Adaptive hybrid strategy

@dataclass
class ConnectionInfo:
    """Connection metadata and health information"""
    connection_id: str
    created_at: float
    last_used: float
    status: ConnectionStatus
    health_score: float = 1.0  # 0.0 to 1.0
    error_count: int = 0
    total_requests: int = 0
    total_response_time: float = 0.0
    connection_object: Any = None
    last_health_check: float = field(default_factory=time.time)
    retry_count: int = 0

    @property
    def avg_response_time(self) -> float:
        """Average response time for this connection"""
        return self.total_response_time / max(self.total_requests, 1)

    @property
    def age_seconds(self) -> float:
        """Connection age in seconds"""
        return time.time() - self.created_at

    def record_request(self, response_time: float):
        """Record a successful request"""
        self.total_requests += 1
        self.total_response_time += response_time
        self.last_used = time.time()
        self.health_score = min(1.0, self.health_score + 0.01)

    def record_error(self):
        """Record an error"""
        self.error_count += 1
        self.health_score = max(0.0, self.health_score - 0.1)

class ConnectionPoolConfig:
    """Configuration for connection pools"""

    def __init__(self,
                 min_connections: int = 5,
                 max_connections: int = 50,
                 max_idle_time: int = 300,
                 connection_timeout: int = 30,
                 health_check_interval: int = 60,
                 retry_attempts: int = 3,
                 strategy: PoolStrategy = PoolStrategy.ADAPTIVE):

        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        self.retry_attempts = retry_attempts
        self.strategy = strategy

        # Adaptive parameters
        self.scale_up_threshold = 0.8  # 80% utilization triggers scale up
        self.scale_down_threshold = 0.3  # 30% utilization triggers scale down
        self.scale_up_factor = 1.5  # Scale up by 50%
        self.scale_down_factor = 0.7  # Scale down by 30%

class ConnectionPool:
    """Generic connection pool with intelligent management"""

    def __init__(self,
                 connection_factory: Callable,
                 pool_config: ConnectionPoolConfig,
                 pool_name: str = "default"):

        self.connection_factory = connection_factory
        self.config = pool_config
        self.pool_name = pool_name

        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.available_connections = deque()
        self.active_connections = set()
        self.connection_lock = threading.RLock()

        # Metrics tracking
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_errors = 0

        # Background tasks
        self.health_check_thread = None
        self.maintenance_thread = None
        self.running = False

        # Event handling
        self.connection_event = threading.Event()
        self.error_callbacks = []

        # Start background tasks
        self.start()

    def start(self):
        """Start connection pool management"""
        logger.info(f"Starting connection pool '{self.pool_name}'")
        self.running = True

        # Start health check thread
        self.health_check_thread = threading.Thread(target=self._health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()

        # Start maintenance thread
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()

        # Initialize minimum connections
        if self.config.strategy in [PoolStrategy.EAGER, PoolStrategy.ADAPTIVE]:
            self._ensure_minimum_connections()

        logger.info(f"Connection pool '{self.pool_name}' started")

    def stop(self):
        """Stop connection pool management"""
        logger.info(f"Stopping connection pool '{self.pool_name}'")
        self.running = False

        # Wait for background threads
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)

        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)

        # Close all connections
        self._close_all_connections()

        logger.info(f"Connection pool '{self.pool_name}' stopped")

    def _ensure_minimum_connections(self):
        """Ensure minimum number of connections are available"""
        current_count = len(self.connections)
        needed = self.config.min_connections - current_count

        for _ in range(needed):
            self._create_connection()

    def _create_connection(self) -> Optional[str]:
        """Create a new connection"""
        try:
            connection = self.connection_factory()
            connection_id = f"{self.pool_name}_{self.total_created}_{int(time.time() * 1000)}"

            connection_info = ConnectionInfo(
                connection_id=connection_id,
                created_at=time.time(),
                last_used=time.time(),
                status=ConnectionStatus.IDLE,
                connection_object=connection
            )

            self.connections[connection_id] = connection_info
            self.available_connections.append(connection_id)
            self.total_created += 1

            logger.debug(f"Created new connection {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self.total_errors += 1
            self._trigger_error_callbacks("connection_creation_failed", str(e))
            return None

    def acquire(self, timeout: Optional[int] = None) -> Optional[Any]:
        """Acquire a connection from the pool"""
        start_time = time.time()
        timeout = timeout or self.config.connection_timeout

        with self.connection_lock:
            # Try to get available connection
            while self.running:
                # Check available connections
                if self.available_connections:
                    connection_id = self.available_connections.popleft()
                    connection_info = self.connections.get(connection_id)

                    if connection_info and connection_info.status == ConnectionStatus.IDLE:
                        connection_info.status = ConnectionStatus.ACTIVE
                        self.active_connections.add(connection_id)
                        self.total_acquired += 1

                        # Trigger adaptive scaling if needed
                        self._check_scaling_needs()

                        return connection_info.connection_object

                # Check if we can create a new connection
                if len(self.connections) < self.config.max_connections:
                    connection_id = self._create_connection()
                    if connection_id:
                        connection_info = self.connections[connection_id]
                        connection_info.status = ConnectionStatus.ACTIVE
                        self.active_connections.add(connection_id)
                        self.total_acquired += 1
                        return connection_info.connection_object

                # Wait for connection to become available
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.total_timeouts += 1
                    logger.warning(f"Connection acquisition timeout after {elapsed:.1f}s")
                    return None

                # Wait with timeout
                self.connection_event.wait(timeout - elapsed)
                self.connection_event.clear()

        return None

    def release(self, connection_object: Any):
        """Release a connection back to the pool"""
        with self.connection_lock:
            # Find the connection info
            connection_id = None
            for conn_id, conn_info in self.connections.items():
                if conn_info.connection_object is connection_object:
                    connection_id = conn_id
                    break

            if connection_id and connection_id in self.active_connections:
                connection_info = self.connections[connection_id]
                connection_info.status = ConnectionStatus.IDLE
                self.active_connections.remove(connection_id)
                self.available_connections.append(connection_id)
                self.total_released += 1

                # Signal waiting threads
                self.connection_event.set()

    def _health_check_loop(self):
        """Background health checking loop"""
        while self.running:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(30)

    def _perform_health_checks(self):
        """Perform health checks on all connections"""
        with self.connection_lock:
            current_time = time.time()
            connections_to_check = []

            for connection_id, connection_info in self.connections.items():
                # Check if health check is needed
                if (current_time - connection_info.last_health_check) > self.config.health_check_interval:
                    connections_to_check.append(connection_id)

            # Perform health checks in parallel
            if connections_to_check:
                logger.debug(f"Performing health checks on {len(connections_to_check)} connections")

                for connection_id in connections_to_check:
                    if connection_id in self.connections:
                        self._health_check_connection(connection_id)

    def _health_check_connection(self, connection_id: str):
        """Health check a specific connection"""
        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return

        connection_info.status = ConnectionStatus.HEALTH_CHECKING
        connection_info.last_health_check = time.time()

        try:
            # Perform health check (implementation-specific)
            if self._is_connection_healthy(connection_info.connection_object):
                connection_info.health_score = min(1.0, connection_info.health_score + 0.05)
                connection_info.status = ConnectionStatus.IDLE
            else:
                connection_info.health_score = max(0.0, connection_info.health_score - 0.2)
                connection_info.record_error()

                # Remove unhealthy connection
                if connection_info.health_score < 0.3:
                    logger.warning(f"Removing unhealthy connection {connection_id}")
                    self._destroy_connection(connection_id)

        except Exception as e:
            logger.error(f"Health check failed for connection {connection_id}: {e}")
            connection_info.record_error()
            connection_info.status = ConnectionStatus.ERROR

            # Remove failed connection
            self._destroy_connection(connection_id)

    def _is_connection_healthy(self, connection_object: Any) -> bool:
        """Check if connection is healthy (override in subclasses)"""
        # Basic health check - subclasses should implement specific logic
        try:
            # For HTTP connections, try a simple request
            if hasattr(connection_object, 'get'):
                response = connection_object.get('/health', timeout=5)
                return response.status_code == 200
            # For database connections, try a simple query
            elif hasattr(connection_object, 'execute'):
                cursor = connection_object.cursor()
                cursor.execute("SELECT 1")
                return True
            else:
                # Default check - assume healthy if no exception
                return True
        except:
            return False

    def _maintenance_loop(self):
        """Background maintenance loop"""
        while self.running:
            try:
                self._perform_maintenance()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                time.sleep(30)

    def _perform_maintenance(self):
        """Perform pool maintenance tasks"""
        with self.connection_lock:
            current_time = time.time()

            # Remove idle connections
            idle_connections = []
            for connection_id, connection_info in self.connections.items():
                if (connection_info.status == ConnectionStatus.IDLE and
                    (current_time - connection_info.last_used) > self.config.max_idle_time):
                    idle_connections.append(connection_id)

            for connection_id in idle_connections:
                if len(self.connections) > self.config.min_connections:
                    self._destroy_connection(connection_id)

            # Adaptive scaling
            if self.config.strategy == PoolStrategy.ADAPTIVE:
                self._adaptive_scaling()

    def _check_scaling_needs(self):
        """Check if pool needs scaling (called on acquire)"""
        if self.config.strategy == PoolStrategy.ADAPTIVE:
            utilization = len(self.active_connections) / max(len(self.connections), 1)

            if utilization > self.config.scale_up_threshold:
                self._scale_up_pool()

    def _adaptive_scaling(self):
        """Perform adaptive scaling based on current load"""
        current_connections = len(self.connections)
        active_connections = len(self.active_connections)

        if current_connections == 0:
            return

        utilization = active_connections / current_connections

        # Scale up if needed
        if utilization > self.config.scale_up_threshold and current_connections < self.config.max_connections:
            self._scale_up_pool()

        # Scale down if needed
        elif utilization < self.config.scale_down_threshold and current_connections > self.config.min_connections:
            self._scale_down_pool()

    def _scale_up_pool(self):
        """Scale up the connection pool"""
        current_count = len(self.connections)
        target_count = min(int(current_count * self.config.scale_up_factor), self.config.max_connections)
        needed = target_count - current_count

        for _ in range(needed):
            self._create_connection()

        if needed > 0:
            logger.info(f"Scaled up pool '{self.pool_name}' by {needed} connections")

    def _scale_down_pool(self):
        """Scale down the connection pool"""
        current_count = len(self.connections)
        target_count = max(int(current_count * self.config.scale_down_factor), self.config.min_connections)
        excess = current_count - target_count

        if excess > 0:
            # Remove idle connections first
            idle_connections = [
                conn_id for conn_id, conn_info in self.connections.items()
                if conn_info.status == ConnectionStatus.IDLE
            ]

            removed_count = 0
            for connection_id in idle_connections[:excess]:
                self._destroy_connection(connection_id)
                removed_count += 1

            logger.info(f"Scaled down pool '{self.pool_name}' by {removed_count} connections")

    def _destroy_connection(self, connection_id: str):
        """Destroy a specific connection"""
        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return

        try:
            # Close connection
            if hasattr(connection_info.connection_object, 'close'):
                connection_info.connection_object.close()
            elif hasattr(connection_info.connection_object, 'cleanup'):
                connection_info.connection_object.cleanup()

        except Exception as e:
            logger.warning(f"Error closing connection {connection_id}: {e}")

        # Remove from pool
        self.connections.pop(connection_id, None)
        self.active_connections.discard(connection_id)

        # Remove from available queue
        try:
            self.available_connections.remove(connection_id)
        except ValueError:
            pass

        self.total_destroyed += 1

    def _close_all_connections(self):
        """Close all connections in the pool"""
        with self.connection_lock:
            connection_ids = list(self.connections.keys())
            for connection_id in connection_ids:
                self._destroy_connection(connection_id)

    @contextmanager
    def connection(self, timeout: Optional[int] = None):
        """Context manager for acquiring and releasing connections"""
        connection = None
        try:
            connection = self.acquire(timeout)
            if connection is None:
                raise Exception("Failed to acquire connection from pool")
            yield connection
        finally:
            if connection is not None:
                self.release(connection)

    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)

    def _trigger_error_callbacks(self, error_type: str, message: str):
        """Trigger error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(self.pool_name, error_type, message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.connection_lock:
            current_connections = len(self.connections)
            active_connections = len(self.active_connections)
            available_connections = len(self.available_connections)

            utilization = (active_connections / max(current_connections, 1)) * 100

            # Calculate average health score
            health_scores = [conn.health_score for conn in self.connections.values()]
            avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0

            # Calculate average response time
            response_times = [conn.avg_response_time for conn in self.connections.values() if conn.total_requests > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            return {
                "pool_name": self.pool_name,
                "strategy": self.config.strategy.value,
                "total_connections": current_connections,
                "active_connections": active_connections,
                "available_connections": available_connections,
                "utilization_percent": utilization,
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "avg_health_score": avg_health_score,
                "avg_response_time_ms": avg_response_time * 1000,
                "total_created": self.total_created,
                "total_destroyed": self.total_destroyed,
                "total_acquired": self.total_acquired,
                "total_released": self.total_released,
                "total_timeouts": self.total_timeouts,
                "total_errors": self.total_errors
            }

class ConnectionPoolManager:
    """Manages multiple connection pools for different services"""

    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.pool_configs: Dict[str, ConnectionPoolConfig] = {}
        self.global_stats = {
            "total_pools": 0,
            "total_connections": 0,
            "total_acquired": 0,
            "total_errors": 0
        }

    def create_pool(self,
                    pool_name: str,
                    connection_factory: Callable,
                    config: Optional[ConnectionPoolConfig] = None) -> ConnectionPool:

        if config is None:
            config = ConnectionPoolConfig()

        pool = ConnectionPool(connection_factory, config, pool_name)
        self.pools[pool_name] = pool
        self.pool_configs[pool_name] = config

        logger.info(f"Created connection pool '{pool_name}'")
        return pool

    def get_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name"""
        return self.pools.get(pool_name)

    def remove_pool(self, pool_name: str):
        """Remove a connection pool"""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            pool.stop()
            del self.pools[pool_name]
            del self.pool_configs[pool_name]
            logger.info(f"Removed connection pool '{pool_name}'")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics for all pools"""
        total_connections = 0
        total_acquired = 0
        total_errors = 0

        pool_stats = {}
        for pool_name, pool in self.pools.items():
            stats = pool.get_stats()
            pool_stats[pool_name] = stats
            total_connections += stats["total_connections"]
            total_acquired += stats["total_acquired"]
            total_errors += stats["total_errors"]

        return {
            "total_pools": len(self.pools),
            "total_connections": total_connections,
            "total_acquired": total_acquired,
            "total_errors": total_errors,
            "pool_details": pool_stats
        }

    def shutdown(self):
        """Shutdown all connection pools"""
        logger.info("Shutting down all connection pools")
        for pool_name in list(self.pools.keys()):
            self.remove_pool(pool_name)
        logger.info("All connection pools shutdown")

# Example usage
if __name__ == "__main__":
    import requests

    def create_http_connection():
        """Create HTTP connection for testing"""
        session = requests.Session()
        return session

    # Create connection pool manager
    pool_manager = ConnectionPoolManager()

    # Create a pool for API connections
    api_config = ConnectionPoolConfig(
        min_connections=5,
        max_connections=20,
        strategy=PoolStrategy.ADAPTIVE
    )

    api_pool = pool_manager.create_pool("api_pool", create_http_connection, api_config)

    try:
        # Use the pool
        with api_pool.connection() as conn:
            response = conn.get("https://httpbin.org/get")
            print(f"Response status: {response.status_code}")

        # Get statistics
        stats = api_pool.get_stats()
        print(f"Pool stats: {stats}")

        # Get global statistics
        global_stats = pool_manager.get_global_stats()
        print(f"Global stats: {global_stats}")

    finally:
        pool_manager.shutdown()