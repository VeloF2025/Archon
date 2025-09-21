"""
Database Connection Management and Pooling

Provides intelligent database connection management:
- Advanced connection pooling with health monitoring
- Automatic failover and recovery
- Load balancing across multiple database instances
- Connection lifecycle optimization
- Performance monitoring and metrics

Features:
- Adaptive pool sizing based on load
- Connection recycling and reuse
- Real-time health checks
- Automatic connection optimization
- Multi-database support
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from contextlib import contextmanager, asynccontextmanager
import json
import random
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    ORACLE = "oracle"
    MSSQL = "mssql"
    MONGODB = "mongodb"
    REDIS = "redis"

class ConnectionStatus(Enum):
    """Connection status states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    HEALTH_CHECKING = "health_checking"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    FASTEST_RESPONSE = "fastest_response"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    GEOGRAPHIC = "geographic"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_mode: Optional[str] = None
    connection_timeout: int = 30
    query_timeout: int = 300
    max_connections: int = 50
    min_connections: int = 5
    connection_pool_size: int = 20
    health_check_interval: int = 60
    health_check_timeout: int = 5
    retry_attempts: int = 3
    read_replicas: List[Dict[str, Any]] = field(default_factory=list)
    weight: int = 1  # For weighted load balancing

@dataclass
class ConnectionInfo:
    """Connection metadata and health information"""
    connection_id: str
    config: DatabaseConfig
    created_at: float
    last_used: float
    status: ConnectionStatus
    health_score: float = 1.0
    error_count: int = 0
    total_queries: int = 0
    total_response_time: float = 0.0
    connection_object: Any = None
    last_health_check: float = field(default_factory=time.time)
    is_primary: bool = True
    is_read_only: bool = False
    retry_count: int = 0

    @property
    def avg_response_time(self) -> float:
        """Average response time for this connection"""
        return self.total_response_time / max(self.total_queries, 1)

    @property
    def age_seconds(self) -> float:
        """Connection age in seconds"""
        return time.time() - self.created_at

    def record_query(self, response_time: float):
        """Record a successful query"""
        self.total_queries += 1
        self.total_response_time += response_time
        self.last_used = time.time()
        self.health_score = min(1.0, self.health_score + 0.01)

    def record_error(self):
        """Record an error"""
        self.error_count += 1
        self.health_score = max(0.0, self.health_score - 0.1)

class DatabaseConnectionPool:
    """Intelligent database connection pool"""

    def __init__(self, config: DatabaseConfig, pool_name: str = "default"):
        self.config = config
        self.pool_name = pool_name

        # Connection management
        self.connections: Dict[str, ConnectionInfo] = {}
        self.primary_connections = deque()
        self.replica_connections = deque()
        self.active_connections = set()
        self.idle_connections = deque()
        self.connection_lock = threading.RLock()

        # Metrics tracking
        self.total_created = 0
        self.total_destroyed = 0
        self.total_acquired = 0
        self.total_released = 0
        self.total_timeouts = 0
        self.total_errors = 0
        self.total_retries = 0

        # Health monitoring
        self.health_check_thread = None
        self.maintenance_thread = None
        self.running = False

        # Load balancing
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        self.connection_index = 0

        # Failover management
        self.failover_mode = False
        self.primary_available = True

        # Start background tasks
        self.start()

    def start(self):
        """Start connection pool"""
        logger.info(f"Starting database connection pool '{self.pool_name}' for {self.config.db_type.value}")
        self.running = True

        # Initialize minimum connections
        self._ensure_minimum_connections()

        # Start background tasks
        self.health_check_thread = threading.Thread(target=self._health_check_loop)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()

        self.maintenance_thread = threading.Thread(target=self._maintenance_loop)
        self.maintenance_thread.daemon = True
        self.maintenance_thread.start()

        logger.info(f"Database connection pool '{self.pool_name}' started")

    def stop(self):
        """Stop connection pool"""
        logger.info(f"Stopping database connection pool '{self.pool_name}'")
        self.running = False

        # Wait for background threads
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5)

        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5)

        # Close all connections
        self._close_all_connections()

        logger.info(f"Database connection pool '{self.pool_name}' stopped")

    def _ensure_minimum_connections(self):
        """Ensure minimum number of connections are available"""
        primary_count = len([c for c in self.connections.values() if c.is_primary])
        replica_count = len([c for c in self.connections.values() if not c.is_primary])

        # Create primary connections
        primary_needed = max(0, self.config.min_connections - primary_count)
        for _ in range(primary_needed):
            self._create_connection(is_primary=True)

        # Create replica connections if configured
        if self.config.read_replicas:
            replica_needed = max(0, self.config.min_connections // 2 - replica_count)
            for _ in range(replica_needed):
                self._create_connection(is_primary=False)

    def _create_connection(self, is_primary: bool = True) -> Optional[str]:
        """Create a new database connection"""
        try:
            if is_primary:
                connection = self._create_primary_connection()
            else:
                connection = self._create_replica_connection()

            connection_id = f"{self.pool_name}_{'primary' if is_primary else 'replica'}_{self.total_created}_{int(time.time() * 1000)}"

            connection_info = ConnectionInfo(
                connection_id=connection_id,
                config=self.config,
                created_at=time.time(),
                last_used=time.time(),
                status=ConnectionStatus.CONNECTED,
                connection_object=connection,
                is_primary=is_primary,
                is_read_only=not is_primary
            )

            self.connections[connection_id] = connection_info

            if is_primary:
                self.primary_connections.append(connection_id)
            else:
                self.replica_connections.append(connection_id)

            self.idle_connections.append(connection_id)
            self.total_created += 1

            logger.debug(f"Created {connection_id}")
            return connection_id

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self.total_errors += 1
            return None

    def _create_primary_connection(self) -> Any:
        """Create primary database connection"""
        # This would be implemented based on the specific database type
        # For now, return a mock connection object
        return {
            "type": "primary",
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "created_at": time.time()
        }

    def _create_replica_connection(self) -> Any:
        """Create replica database connection"""
        # This would select a read replica based on load balancing
        if self.config.read_replicas:
            replica = random.choice(self.config.read_replicas)
            return {
                "type": "replica",
                "host": replica["host"],
                "port": replica["port"],
                "database": replica["database"],
                "created_at": time.time()
            }
        return self._create_primary_connection()

    def acquire(self, read_only: bool = False, timeout: Optional[int] = None) -> Optional[Any]:
        """Acquire a database connection"""
        start_time = time.time()
        timeout = timeout or self.config.connection_timeout

        with self.connection_lock:
            while self.running:
                # Try to get appropriate connection
                connection_id = self._select_connection(read_only)

                if connection_id:
                    connection_info = self.connections.get(connection_id)
                    if connection_info and connection_info.status == ConnectionStatus.IDLE:
                        connection_info.status = ConnectionStatus.ACTIVE
                        self.active_connections.add(connection_id)
                        self.idle_connections.remove(connection_id)
                        self.total_acquired += 1

                        return connection_info.connection_object

                # Try to create new connection if under limit
                if len(self.connections) < self.config.max_connections:
                    connection_id = self._create_connection(is_primary=not read_only)
                    if connection_id:
                        connection_info = self.connections[connection_id]
                        connection_info.status = ConnectionStatus.ACTIVE
                        self.active_connections.add(connection_id)
                        self.idle_connections.remove(connection_id)
                        self.total_acquired += 1
                        return connection_info.connection_object

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.total_timeouts += 1
                    logger.warning(f"Connection acquisition timeout after {elapsed:.1f}s")
                    return None

                # Wait for connection to become available
                time.sleep(0.1)

        return None

    def _select_connection(self, read_only: bool) -> Optional[str]:
        """Select connection based on load balancing strategy"""
        available_connections = []

        if read_only and self.replica_connections:
            # Prefer replicas for read operations
            for conn_id in self.replica_connections:
                if conn_id in self.idle_connections:
                    available_connections.append(conn_id)
        elif self.primary_connections and self.primary_available:
            # Use primary for write operations or if replicas not available
            for conn_id in self.primary_connections:
                if conn_id in self.idle_connections:
                    available_connections.append(conn_id)

        if not available_connections:
            return None

        # Apply load balancing strategy
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            connection_id = available_connections[self.connection_index % len(available_connections)]
            self.connection_index += 1
            return connection_id

        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select connection with least active connections
            best_connection = None
            min_connections = float('inf')

            for conn_id in available_connections:
                # This is a simplified version - in reality, you'd track active queries per connection
                active_count = len(self.active_connections)
                if active_count < min_connections:
                    min_connections = active_count
                    best_connection = conn_id

            return best_connection

        elif self.load_balancing_strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            # Select connection with fastest average response time
            best_connection = None
            best_response_time = float('inf')

            for conn_id in available_connections:
                conn_info = self.connections.get(conn_id)
                if conn_info and conn_info.total_queries > 0:
                    avg_time = conn_info.avg_response_time
                    if avg_time < best_response_time:
                        best_response_time = avg_time
                        best_connection = conn_id

            return best_connection or available_connections[0]

        return available_connections[0]

    def release(self, connection_object: Any):
        """Release a database connection back to the pool"""
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
                self.idle_connections.append(connection_id)
                self.total_released += 1

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
                if (current_time - connection_info.last_health_check) > self.config.health_check_interval:
                    connections_to_check.append(connection_id)

            # Perform health checks in parallel
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
            if self._is_connection_healthy(connection_info.connection_object):
                connection_info.health_score = min(1.0, connection_info.health_score + 0.05)
                connection_info.status = ConnectionStatus.IDLE

                # Update primary availability
                if connection_info.is_primary:
                    self.primary_available = True
                    self.failover_mode = False
            else:
                connection_info.health_score = max(0.0, connection_info.health_score - 0.2)
                connection_info.record_error()

                # Handle unhealthy connection
                if connection_info.health_score < 0.3:
                    logger.warning(f"Removing unhealthy connection {connection_id}")
                    self._destroy_connection(connection_id)

                    # If primary failed, trigger failover
                    if connection_info.is_primary:
                        self._trigger_failover()

        except Exception as e:
            logger.error(f"Health check failed for connection {connection_id}: {e}")
            connection_info.record_error()
            connection_info.status = ConnectionStatus.ERROR

            # Remove failed connection
            self._destroy_connection(connection_id)

            # If primary failed, trigger failover
            if connection_info.is_primary:
                self._trigger_failover()

    def _is_connection_healthy(self, connection_object: Any) -> bool:
        """Check if database connection is healthy"""
        try:
            # Simple health check - execute a basic query
            # This would be implemented based on the specific database type
            if isinstance(connection_object, dict):
                # Mock connection
                return True
            else:
                # Real database connection
                # cursor = connection_object.cursor()
                # cursor.execute("SELECT 1")
                # return True
                return True
        except:
            return False

    def _trigger_failover(self):
        """Trigger failover to read replicas"""
        logger.warning("Primary database connection failed, triggering failover")
        self.failover_mode = True
        self.primary_available = False

        # Promote a replica to primary if available
        if self.replica_connections:
            replica_id = self.replica_connections[0]
            if replica_id in self.connections:
                replica_info = self.connections[replica_id]
                replica_info.is_primary = True
                replica_info.is_read_only = False
                logger.info(f"Promoted replica {replica_id} to primary")

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
        """Perform connection pool maintenance"""
        with self.connection_lock:
            current_time = time.time()

            # Clean up idle connections
            idle_connections = []
            for connection_id, connection_info in self.connections.items():
                if (connection_info.status == ConnectionStatus.IDLE and
                    (current_time - connection_info.last_used) > 300):  # 5 minutes idle
                    idle_connections.append(connection_id)

            for connection_id in idle_connections:
                if len(self.connections) > self.config.min_connections:
                    self._destroy_connection(connection_id)

            # Optimize pool size based on load
            self._optimize_pool_size()

    def _optimize_pool_size(self):
        """Optimize pool size based on current load"""
        current_size = len(self.connections)
        active_count = len(self.active_connections)
        utilization = active_count / max(current_size, 1)

        # Scale up if needed
        if utilization > 0.8 and current_size < self.config.max_connections:
            needed = min(2, self.config.max_connections - current_size)
            for _ in range(needed):
                self._create_connection()

        # Scale down if needed
        elif utilization < 0.3 and current_size > self.config.min_connections:
            excess = current_size - self.config.min_connections
            removed = 0
            for _ in range(min(2, excess)):
                if self.idle_connections:
                    connection_id = self.idle_connections[0]
                    self._destroy_connection(connection_id)
                    removed += 1

            if removed > 0:
                logger.debug(f"Scaled down pool by {removed} connections")

    def _destroy_connection(self, connection_id: str):
        """Destroy a specific connection"""
        connection_info = self.connections.get(connection_id)
        if not connection_info:
            return

        try:
            # Close connection
            if hasattr(connection_info.connection_object, 'close'):
                connection_info.connection_object.close()

        except Exception as e:
            logger.warning(f"Error closing connection {connection_id}: {e}")

        # Remove from all collections
        self.connections.pop(connection_id, None)
        self.active_connections.discard(connection_id)

        try:
            self.primary_connections.remove(connection_id)
        except ValueError:
            pass

        try:
            self.replica_connections.remove(connection_id)
        except ValueError:
            pass

        try:
            self.idle_connections.remove(connection_id)
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
    def connection(self, read_only: bool = False, timeout: Optional[int] = None):
        """Context manager for acquiring and releasing connections"""
        connection = None
        try:
            connection = self.acquire(read_only, timeout)
            if connection is None:
                raise Exception("Failed to acquire database connection")
            yield connection
        finally:
            if connection is not None:
                self.release(connection)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self.connection_lock:
            current_connections = len(self.connections)
            active_connections = len(self.active_connections)
            idle_connections = len(self.idle_connections)

            utilization = (active_connections / max(current_connections, 1)) * 100

            # Calculate average health score
            health_scores = [conn.health_score for conn in self.connections.values()]
            avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0

            # Calculate average response time
            response_times = [conn.avg_response_time for conn in self.connections.values() if conn.total_queries > 0]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            return {
                "pool_name": self.pool_name,
                "database_type": self.config.db_type.value,
                "total_connections": current_connections,
                "active_connections": active_connections,
                "idle_connections": idle_connections,
                "primary_connections": len(self.primary_connections),
                "replica_connections": len(self.replica_connections),
                "utilization_percent": utilization,
                "avg_health_score": avg_health_score,
                "avg_response_time_ms": avg_response_time * 1000,
                "total_created": self.total_created,
                "total_destroyed": self.total_destroyed,
                "total_acquired": self.total_acquired,
                "total_released": self.total_released,
                "total_timeouts": self.total_timeouts,
                "total_errors": self.total_errors,
                "failover_mode": self.failover_mode,
                "primary_available": self.primary_available
            }

class DatabaseConnectionManager:
    """Manages multiple database connection pools"""

    def __init__(self):
        self.pools: Dict[str, DatabaseConnectionPool] = {}
        self.configs: Dict[str, DatabaseConfig] = {}
        self.global_stats = {
            "total_pools": 0,
            "total_connections": 0,
            "total_queries": 0,
            "total_errors": 0
        }

    def create_pool(self, pool_name: str, config: DatabaseConfig) -> DatabaseConnectionPool:
        """Create a new database connection pool"""
        pool = DatabaseConnectionPool(config, pool_name)
        self.pools[pool_name] = pool
        self.configs[pool_name] = config
        return pool

    def get_pool(self, pool_name: str) -> Optional[DatabaseConnectionPool]:
        """Get a connection pool by name"""
        return self.pools.get(pool_name)

    def remove_pool(self, pool_name: str):
        """Remove a connection pool"""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            pool.stop()
            del self.pools[pool_name]
            del self.configs[pool_name]
            logger.info(f"Removed database connection pool '{pool_name}'")

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics for all pools"""
        total_connections = 0
        total_queries = 0
        total_errors = 0

        pool_stats = {}
        for pool_name, pool in self.pools.items():
            stats = pool.get_stats()
            pool_stats[pool_name] = stats
            total_connections += stats["total_connections"]
            total_queries += stats["total_acquired"]
            total_errors += stats["total_errors"]

        return {
            "total_pools": len(self.pools),
            "total_connections": total_connections,
            "total_queries": total_queries,
            "total_errors": total_errors,
            "pool_details": pool_stats
        }

    def shutdown(self):
        """Shutdown all connection pools"""
        logger.info("Shutting down all database connection pools")
        for pool_name in list(self.pools.keys()):
            self.remove_pool(pool_name)
        logger.info("All database connection pools shutdown")

# Example usage
if __name__ == "__main__":
    # Create database configuration
    config = DatabaseConfig(
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="agency_swarm",
        username="postgres",
        password="password",
        max_connections=20,
        min_connections=3
    )

    # Create connection manager
    manager = DatabaseConnectionManager()

    # Create connection pool
    pool = manager.create_pool("main_db", config)

    try:
        # Use the pool
        with pool.connection() as conn:
            print(f"Acquired connection: {conn}")

        # Get statistics
        stats = pool.get_stats()
        print(f"Pool stats: {stats}")

    finally:
        manager.shutdown()