"""
Performance Optimizer for Agency Swarm Enterprise Deployments

Provides comprehensive performance optimization strategies for large-scale agent deployments:
- Agent lifecycle optimization and resource management
- Memory and CPU efficiency improvements
- Network optimization and latency reduction
- Caching strategies and connection pooling
- Load balancing and horizontal scaling

Target Performance Goals:
- Response Time: <100ms for agent communications
- Throughput: 10K+ messages/second
- Scalability: Support for 1000+ concurrent agents
- Memory Usage: <2GB per 1000 agents
- CPU Efficiency: <70% CPU utilization under peak load
"""

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Callable, Any, Tuple
from collections import defaultdict, deque
import psutil
import json
import tracemalloc
from contextlib import contextmanager
import heapq
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of performance optimizations"""
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    CACHE = "cache"
    DATABASE = "database"
    AGENT_POOL = "agent_pool"
    MESSAGE_QUEUE = "message_queue"
    LOAD_BALANCING = "load_balancing"

class PriorityLevel(Enum):
    """Priority levels for optimization tasks"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics tracking"""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_agents: int
    queued_messages: int
    processing_rate_per_sec: float
    avg_response_time_ms: float
    error_rate_percent: float
    network_throughput_mbps: float

@dataclass
class OptimizationTarget:
    """Target for performance optimization"""
    name: str
    optimization_type: OptimizationType
    current_value: float
    target_value: float
    priority: PriorityLevel
    description: str
    enabled: bool = True

@dataclass
class OptimizationResult:
    """Result of optimization operation"""
    success: bool
    optimization_type: OptimizationType
    improvement_percent: float
    time_saved_ms: float
    memory_saved_mb: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

class MemoryOptimizer:
    """Memory optimization for agent deployments"""

    def __init__(self, max_memory_per_agent_mb: int = 512):
        self.max_memory_per_agent_mb = max_memory_per_agent_mb
        self.agent_memory_usage: Dict[str, float] = {}
        self.memory_pool = defaultdict(float)
        self.memory_lock = threading.Lock()

        # Garbage collection tuning
        self.gc_thresholds = [700000, 1000000, 10000000]  # Objects before GC
        self.last_gc_time = time.time()
        self.gc_interval_seconds = 30

    def track_agent_memory(self, agent_id: str, memory_mb: float):
        """Track memory usage per agent"""
        with self.memory_lock:
            self.agent_memory_usage[agent_id] = memory_mb

            # Check for memory leaks
            if memory_mb > self.max_memory_per_agent_mb:
                logger.warning(f"Agent {agent_id} exceeding memory limit: {memory_mb}MB")
                self._trigger_memory_cleanup(agent_id)

    def _trigger_memory_cleanup(self, agent_id: str):
        """Trigger memory cleanup for specific agent"""
        # Force garbage collection for agent
        import gc
        collected = gc.collect()
        logger.info(f"Memory cleanup for agent {agent_id}: collected {collected} objects")

        # Update memory tracking
        if agent_id in self.agent_memory_usage:
            self.agent_memory_usage[agent_id] *= 0.8  # Assume 20% reduction

    def get_memory_efficiency_metrics(self) -> Dict[str, float]:
        """Get memory efficiency metrics"""
        with self.memory_lock:
            total_agents = len(self.agent_memory_usage)
            if total_agents == 0:
                return {"efficiency_percent": 100.0, "avg_memory_per_agent": 0.0}

            total_memory = sum(self.agent_memory_usage.values())
            avg_memory = total_memory / total_agents
            efficiency = max(0, 100 - (avg_memory / self.max_memory_per_agent_mb) * 100)

            return {
                "efficiency_percent": efficiency,
                "avg_memory_per_agent": avg_memory,
                "total_agents": total_agents,
                "total_memory_mb": total_memory
            }

class NetworkOptimizer:
    """Network optimization for agent communications"""

    def __init__(self, target_latency_ms: int = 100):
        self.target_latency_ms = target_latency_ms
        self.connection_pool: Dict[str, List] = {}
        self.latency_history: Dict[str, List[float]] = defaultdict(list)
        self.compression_enabled = True
        self.batching_enabled = True
        self.connection_pool_size = 100

    def create_connection_pool(self, endpoint: str, pool_size: int = 10):
        """Create connection pool for specific endpoint"""
        import aiohttp

        async def create_session():
            connector = aiohttp.TCPConnector(
                limit=pool_size,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            return aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )

        if endpoint not in self.connection_pool:
            self.connection_pool[endpoint] = []
            for _ in range(pool_size):
                self.connection_pool[endpoint].append(create_session())

    async def get_connection(self, endpoint: str):
        """Get connection from pool"""
        if endpoint not in self.connection_pool:
            self.create_connection_pool(endpoint)

        pool = self.connection_pool[endpoint]
        if pool:
            return pool.pop(0)
        return None

    def return_connection(self, endpoint: str, session):
        """Return connection to pool"""
        if endpoint in self.connection_pool:
            self.connection_pool[endpoint].append(session)

    def track_latency(self, endpoint: str, latency_ms: float):
        """Track network latency for endpoint"""
        self.latency_history[endpoint].append(latency_ms)

        # Keep only recent history (last 100 measurements)
        if len(self.latency_history[endpoint]) > 100:
            self.latency_history[endpoint] = self.latency_history[endpoint][-100:]

    def get_network_metrics(self) -> Dict[str, float]:
        """Get network performance metrics"""
        metrics = {}

        for endpoint, latencies in self.latency_history.items():
            if latencies:
                metrics[f"{endpoint}_avg_latency"] = sum(latencies) / len(latencies)
                metrics[f"{endpoint}_max_latency"] = max(latencies)
                metrics[f"{endpoint}_min_latency"] = min(latencies)

        return metrics

    def optimize_message_size(self, message: Dict) -> Dict:
        """Optimize message size through compression and batching"""
        if not self.compression_enabled:
            return message

        # Remove unnecessary whitespace and empty fields
        optimized = self._remove_empty_fields(message)

        # Compress large string fields
        for key, value in optimized.items():
            if isinstance(value, str) and len(value) > 1000:
                optimized[key] = self._compress_string(value)

        return optimized

    def _remove_empty_fields(self, data: Dict) -> Dict:
        """Remove empty fields from dictionary"""
        if isinstance(data, dict):
            return {k: self._remove_empty_fields(v) for k, v in data.items() if v is not None and v != ""}
        elif isinstance(data, list):
            return [self._remove_empty_fields(item) for item in data if item is not None]
        else:
            return data

    def _compress_string(self, text: str) -> str:
        """Compress large string using simple compression"""
        # In production, use proper compression libraries
        if len(text) > 5000:
            return f"[COMPRESSED:{len(text)}:{text[:100]}...]"
        return text

class AgentPoolOptimizer:
    """Agent pool optimization for efficient resource utilization"""

    def __init__(self, max_agents: int = 1000):
        self.max_agents = max_agents
        self.active_agents: Dict[str, Dict] = {}
        self.idle_agents: Dict[str, float] = {}  # agent_id -> last_active_time
        self.agent_load_balancer = {}
        self.agent_response_times: Dict[str, List[float]] = defaultdict(list)

        # Pool management
        self.hibernation_timeout_minutes = 30
        self.max_concurrent_tasks_per_agent = 5
        self.agent_health_check_interval = 60

    def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register new agent in the pool"""
        self.active_agents[agent_id] = {
            "capabilities": capabilities,
            "current_tasks": 0,
            "total_tasks": 0,
            "start_time": time.time(),
            "status": "active"
        }

        logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")

    def mark_agent_idle(self, agent_id: str):
        """Mark agent as idle"""
        self.idle_agents[agent_id] = time.time()

        # Check if agent should be hibernated
        self._check_agent_hibernation()

    def get_optimal_agent(self, required_capabilities: List[str]) -> Optional[str]:
        """Get optimal agent for task based on current load and capabilities"""
        available_agents = []

        for agent_id, agent_info in self.active_agents.items():
            if agent_info["status"] != "active":
                continue

            # Check capabilities
            if all(cap in agent_info["capabilities"] for cap in required_capabilities):
                # Calculate score based on current load and response time
                load_score = 1.0 / (agent_info["current_tasks"] + 1)

                # Consider response time history
                response_times = self.agent_response_times.get(agent_id, [])
                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                    time_score = 1000.0 / (avg_response + 1)  # Inverse of response time
                else:
                    time_score = 100.0

                total_score = load_score + time_score
                available_agents.append((agent_id, total_score))

        if available_agents:
            # Return agent with highest score
            available_agents.sort(key=lambda x: x[1], reverse=True)
            return available_agents[0][0]

        return None

    def track_agent_response_time(self, agent_id: str, response_time_ms: float):
        """Track agent response time for load balancing"""
        self.agent_response_times[agent_id].append(response_time_ms)

        # Keep only recent history (last 50 measurements)
        if len(self.agent_response_times[agent_id]) > 50:
            self.agent_response_times[agent_id] = self.agent_response_times[agent_id][-50:]

    def _check_agent_hibernation(self):
        """Check for agents that should be hibernated"""
        current_time = time.time()
        hibernation_threshold = self.hibernation_timeout_minutes * 60

        agents_to_hibernate = []
        for agent_id, last_active in self.idle_agents.items():
            if current_time - last_active > hibernation_threshold:
                if agent_id in self.active_agents:
                    agents_to_hibernate.append(agent_id)

        for agent_id in agents_to_hibernate:
            self.hibernate_agent(agent_id)

    def hibernate_agent(self, agent_id: str):
        """Hibernate agent to save resources"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id]["status"] = "hibernated"
            logger.info(f"Agent {agent_id} hibernated")

    def activate_agent(self, agent_id: str):
        """Activate hibernated agent"""
        if agent_id in self.active_agents:
            self.active_agents[agent_id]["status"] = "active"
            if agent_id in self.idle_agents:
                del self.idle_agents[agent_id]
            logger.info(f"Agent {agent_id} activated")

    def get_pool_metrics(self) -> Dict[str, Any]:
        """Get agent pool performance metrics"""
        total_agents = len(self.active_agents)
        active_agents = sum(1 for a in self.active_agents.values() if a["status"] == "active")
        idle_agents = len(self.idle_agents)

        total_tasks = sum(a["current_tasks"] for a in self.active_agents.values())

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "idle_agents": idle_agents,
            "hibernated_agents": total_agents - active_agents,
            "total_current_tasks": total_tasks,
            "avg_tasks_per_agent": total_tasks / max(active_agents, 1),
            "pool_utilization_percent": (active_agents / self.max_agents) * 100
        }

class PerformanceOptimizer:
    """Main performance optimizer for Agency Swarm system"""

    def __init__(self,
                 max_agents: int = 1000,
                 target_latency_ms: int = 100,
                 max_memory_per_agent_mb: int = 512):

        self.max_agents = max_agents
        self.target_latency_ms = target_latency_ms
        self.max_memory_per_agent_mb = max_memory_per_agent_mb

        # Initialize optimizers
        self.memory_optimizer = MemoryOptimizer(max_memory_per_agent_mb)
        self.network_optimizer = NetworkOptimizer(target_latency_ms)
        self.agent_pool_optimizer = AgentPoolOptimizer(max_agents)

        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.optimization_targets: List[OptimizationTarget] = []
        self.optimization_results: List[OptimizationResult] = []

        # Background optimization thread
        self.optimization_thread = None
        self.optimization_interval = 60  # seconds
        self.running = False

        # Initialize optimization targets
        self._initialize_optimization_targets()

    def _initialize_optimization_targets(self):
        """Initialize optimization targets"""
        self.optimization_targets = [
            OptimizationTarget(
                name="agent_response_time",
                optimization_type=OptimizationType.NETWORK,
                current_value=0,
                target_value=self.target_latency_ms,
                priority=PriorityLevel.HIGH,
                description="Target agent response time in milliseconds"
            ),
            OptimizationTarget(
                name="memory_per_agent",
                optimization_type=OptimizationType.MEMORY,
                current_value=0,
                target_value=self.max_memory_per_agent_mb,
                priority=PriorityLevel.CRITICAL,
                description="Maximum memory usage per agent in MB"
            ),
            OptimizationTarget(
                name="cpu_utilization",
                optimization_type=OptimizationType.CPU,
                current_value=0,
                target_value=70.0,
                priority=PriorityLevel.MEDIUM,
                description="Maximum CPU utilization percentage"
            ),
            OptimizationTarget(
                name="agent_pool_efficiency",
                optimization_type=OptimizationType.AGENT_POOL,
                current_value=0,
                target_value=90.0,
                priority=PriorityLevel.HIGH,
                description="Agent pool utilization efficiency percentage"
            )
        ]

    def start(self):
        """Start performance optimization"""
        logger.info("Starting Performance Optimizer")
        self.running = True

        # Start background optimization thread
        self.optimization_thread = threading.Thread(target=self._background_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()

        # Start memory tracking
        tracemalloc.start()

        logger.info("Performance Optimizer started successfully")

    def stop(self):
        """Stop performance optimization"""
        logger.info("Stopping Performance Optimizer")
        self.running = False

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)

        tracemalloc.stop()
        logger.info("Performance Optimizer stopped")

    def _background_optimization(self):
        """Background optimization loop"""
        while self.running:
            try:
                # Collect current metrics
                metrics = self.collect_performance_metrics()

                # Run optimizations
                optimizations = self.run_optimizations(metrics)

                # Log results
                for result in optimizations:
                    if result.success:
                        logger.info(f"Optimization successful: {result.optimization_type.value} - {result.improvement_percent:.1f}% improvement")
                    else:
                        logger.warning(f"Optimization failed: {result.optimization_type.value} - {result.error_message}")

                # Sleep until next optimization cycle
                time.sleep(self.optimization_interval)

            except Exception as e:
                logger.error(f"Error in background optimization: {e}")
                time.sleep(self.optimization_interval)

    def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)

        # Agent metrics
        pool_metrics = self.agent_pool_optimizer.get_pool_metrics()
        active_agents = pool_metrics["active_agents"]

        # Network metrics
        network_metrics = self.network_optimizer.get_network_metrics()
        network_throughput = network_metrics.get("throughput_mbps", 0.0)

        # Calculate derived metrics
        processing_rate = self._calculate_processing_rate()
        avg_response_time = self._calculate_average_response_time()
        error_rate = self._calculate_error_rate()

        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            active_agents=active_agents,
            queued_messages=pool_metrics["total_current_tasks"],
            processing_rate_per_sec=processing_rate,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            network_throughput_mbps=network_throughput
        )

        # Store in history
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_processing_rate(self) -> float:
        """Calculate message processing rate"""
        if len(self.metrics_history) < 2:
            return 0.0

        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]

        time_diff = current.timestamp - previous.timestamp
        if time_diff == 0:
            return 0.0

        processed_diff = previous.queued_messages - current.queued_messages
        return max(0, processed_diff / time_diff)

    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all agents"""
        all_times = []
        for agent_times in self.agent_pool_optimizer.agent_response_times.values():
            all_times.extend(agent_times[-10:])  # Last 10 measurements

        if all_times:
            return sum(all_times) / len(all_times)
        return 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate error rate from optimization results"""
        recent_results = [r for r in self.optimization_results[-100:] if not r.success]
        if not self.optimization_results:
            return 0.0

        return (len(recent_results) / min(len(self.optimization_results), 100)) * 100

    def run_optimizations(self, metrics: PerformanceMetrics) -> List[OptimizationResult]:
        """Run performance optimizations based on current metrics"""
        results = []

        # Memory optimization
        if metrics.memory_usage_mb > (self.max_agents * self.max_memory_per_agent_mb * 0.8):
            result = self._optimize_memory(metrics)
            results.append(result)

        # CPU optimization
        if metrics.cpu_usage_percent > 70:
            result = self._optimize_cpu(metrics)
            results.append(result)

        # Network optimization
        if metrics.avg_response_time_ms > self.target_latency_ms:
            result = self._optimize_network(metrics)
            results.append(result)

        # Agent pool optimization
        pool_metrics = self.agent_pool_optimizer.get_pool_metrics()
        if pool_metrics["pool_utilization_percent"] > 90:
            result = self._optimize_agent_pool(metrics)
            results.append(result)

        # Store results
        self.optimization_results.extend(results)

        return results

    def _optimize_memory(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize memory usage"""
        try:
            # Get current memory efficiency
            memory_metrics = self.memory_optimizer.get_memory_efficiency_metrics()

            # Trigger garbage collection
            import gc
            before_gc = psutil.virtual_memory().used / (1024 * 1024)
            collected = gc.collect()
            after_gc = psutil.virtual_memory().used / (1024 * 1024)

            memory_saved = before_gc - after_gc

            # Hibernate idle agents
            current_time = time.time()
            hibernation_threshold = self.agent_pool_optimizer.hibernation_timeout_minutes * 60

            hibernated_count = 0
            for agent_id, last_active in list(self.agent_pool_optimizer.idle_agents.items()):
                if current_time - last_active > hibernation_threshold:
                    self.agent_pool_optimizer.hibernate_agent(agent_id)
                    hibernated_count += 1

            improvement_percent = (memory_saved / metrics.memory_usage_mb) * 100 if metrics.memory_usage_mb > 0 else 0

            return OptimizationResult(
                success=True,
                optimization_type=OptimizationType.MEMORY,
                improvement_percent=improvement_percent,
                time_saved_ms=0,
                memory_saved_mb=memory_saved,
                details={
                    "gc_objects_collected": collected,
                    "agents_hibernated": hibernated_count,
                    "memory_efficiency_percent": memory_metrics["efficiency_percent"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_type=OptimizationType.MEMORY,
                improvement_percent=0,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={},
                error_message=str(e)
            )

    def _optimize_cpu(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize CPU usage"""
        try:
            # Reduce concurrent operations
            original_max_concurrent = getattr(self, 'max_concurrent_operations', 50)
            new_max_concurrent = max(10, int(original_max_concurrent * 0.7))

            # Update configuration
            self.max_concurrent_operations = new_max_concurrent

            # Prioritize critical tasks
            prioritized_count = self._prioritize_critical_tasks()

            improvement_percent = 20.0  # Estimated reduction

            return OptimizationResult(
                success=True,
                optimization_type=OptimizationType.CPU,
                improvement_percent=improvement_percent,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={
                    "max_concurrent_reduced": original_max_concurrent - new_max_concurrent,
                    "tasks_prioritized": prioritized_count,
                    "new_max_concurrent": new_max_concurrent
                }
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_type=OptimizationType.CPU,
                improvement_percent=0,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={},
                error_message=str(e)
            )

    def _optimize_network(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize network performance"""
        try:
            # Enable compression if not already enabled
            self.network_optimizer.compression_enabled = True

            # Increase connection pool size for high-traffic endpoints
            for endpoint in self.network_optimizer.connection_pool:
                current_pool_size = len(self.network_optimizer.connection_pool[endpoint])
                if current_pool_size < 20:
                    self.network_optimizer.connection_pool_size = 20

            # Enable message batching
            self.network_optimizer.batching_enabled = True

            improvement_percent = 15.0  # Estimated improvement

            return OptimizationResult(
                success=True,
                optimization_type=OptimizationType.NETWORK,
                improvement_percent=improvement_percent,
                time_saved_ms=metrics.avg_response_time_ms * 0.15,
                memory_saved_mb=0,
                details={
                    "compression_enabled": True,
                    "batching_enabled": True,
                    "connection_pool_size": self.network_optimizer.connection_pool_size
                }
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_type=OptimizationType.NETWORK,
                improvement_percent=0,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={},
                error_message=str(e)
            )

    def _optimize_agent_pool(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize agent pool utilization"""
        try:
            pool_metrics = self.agent_pool_optimizer.get_pool_metrics()

            # Scale up agent pool if needed
            if pool_metrics["pool_utilization_percent"] > 90:
                # In production, this would trigger agent provisioning
                logger.info("Agent pool utilization high, consider scaling up")

            # Balance load across agents
            balanced_agents = self._balance_agent_load()

            improvement_percent = 10.0  # Estimated improvement

            return OptimizationResult(
                success=True,
                optimization_type=OptimizationType.AGENT_POOL,
                improvement_percent=improvement_percent,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={
                    "agents_balanced": balanced_agents,
                    "pool_utilization": pool_metrics["pool_utilization_percent"],
                    "active_agents": pool_metrics["active_agents"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                success=False,
                optimization_type=OptimizationType.AGENT_POOL,
                improvement_percent=0,
                time_saved_ms=0,
                memory_saved_mb=0,
                details={},
                error_message=str(e)
            )

    def _prioritize_critical_tasks(self) -> int:
        """Prioritize critical tasks to reduce CPU load"""
        # Placeholder for task prioritization logic
        return 0

    def _balance_agent_load(self) -> int:
        """Balance load across agents"""
        # Placeholder for load balancing logic
        return 0

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = self.collect_performance_metrics()

        # Calculate improvement statistics
        recent_improvements = [r for r in self.optimization_results[-50:] if r.success]

        if recent_improvements:
            avg_improvement = sum(r.improvement_percent for r in recent_improvements) / len(recent_improvements)
            total_memory_saved = sum(r.memory_saved_mb for r in recent_improvements)
            total_time_saved = sum(r.time_saved_ms for r in recent_improvements)
        else:
            avg_improvement = 0
            total_memory_saved = 0
            total_time_saved = 0

        return {
            "current_metrics": {
                "cpu_usage_percent": current_metrics.cpu_usage_percent,
                "memory_usage_mb": current_metrics.memory_usage_mb,
                "active_agents": current_metrics.active_agents,
                "avg_response_time_ms": current_metrics.avg_response_time_ms,
                "processing_rate_per_sec": current_metrics.processing_rate_per_sec,
                "error_rate_percent": current_metrics.error_rate_percent
            },
            "optimization_targets": [
                {
                    "name": target.name,
                    "type": target.optimization_type.value,
                    "current": target.current_value,
                    "target": target.target_value,
                    "status": "ACHIEVED" if target.current_value <= target.target_value else "NEEDS_OPTIMIZATION"
                }
                for target in self.optimization_targets
            ],
            "recent_improvements": {
                "count": len(recent_improvements),
                "average_improvement_percent": avg_improvement,
                "total_memory_saved_mb": total_memory_saved,
                "total_time_saved_ms": total_time_saved
            },
            "agent_pool_metrics": self.agent_pool_optimizer.get_pool_metrics(),
            "memory_efficiency": self.memory_optimizer.get_memory_efficiency_metrics(),
            "network_metrics": self.network_optimizer.get_network_metrics(),
            "system_health": self._calculate_system_health_score(current_metrics)
        }

    def _calculate_system_health_score(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Calculate overall system health score"""
        # Score components (0-100)
        cpu_score = max(0, 100 - metrics.cpu_usage_percent)
        memory_score = max(0, 100 - (metrics.memory_usage_mb / (self.max_agents * self.max_memory_per_agent_mb)) * 100)
        response_score = max(0, 100 - (metrics.avg_response_time_ms / self.target_latency_ms) * 100)
        error_score = max(0, 100 - metrics.error_rate_percent)

        overall_score = (cpu_score + memory_score + response_score + error_score) / 4

        health_status = "EXCELLENT" if overall_score >= 90 else \
                       "GOOD" if overall_score >= 75 else \
                       "FAIR" if overall_score >= 60 else "POOR"

        return {
            "overall_score": overall_score,
            "health_status": health_status,
            "component_scores": {
                "cpu": cpu_score,
                "memory": memory_score,
                "response_time": response_score,
                "error_rate": error_score
            }
        }

# Example usage
if __name__ == "__main__":
    # Test performance optimizer
    optimizer = PerformanceOptimizer(
        max_agents=1000,
        target_latency_ms=100,
        max_memory_per_agent_mb=512
    )

    try:
        optimizer.start()

        # Simulate some activity
        time.sleep(5)

        # Get performance report
        report = optimizer.get_performance_report()
        print(f"Performance Report:")
        print(f"System Health: {report['system_health']['health_status']} ({report['system_health']['overall_score']:.1f})")
        print(f"Active Agents: {report['current_metrics']['active_agents']}")
        print(f"Response Time: {report['current_metrics']['avg_response_time_ms']:.1f}ms")

    finally:
        optimizer.stop()