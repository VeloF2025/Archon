"""
Metrics Collector
Comprehensive metrics collection system for all Archon components
"""

import asyncio
import psutil
import time
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
import threading
from collections import defaultdict, deque
import inspect
import functools

logger = logging.getLogger(__name__)


class MetricSource(Enum):
    """Sources of metrics"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    API = "api"
    AGENT = "agent"
    MODEL = "model"
    USER = "user"
    CUSTOM = "custom"


@dataclass
class CollectorConfig:
    """Configuration for metrics collector"""
    enabled: bool = True
    collection_interval: int = 60  # seconds
    batch_size: int = 100
    flush_interval: int = 10  # seconds
    enable_system_metrics: bool = True
    enable_application_metrics: bool = True
    enable_database_metrics: bool = True
    enable_api_metrics: bool = True
    enable_agent_metrics: bool = True
    custom_collectors: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    cpu_count: int
    memory_used: float
    memory_percent: float
    memory_available: float
    disk_used: float
    disk_percent: float
    disk_available: float
    network_sent: float
    network_received: float
    process_count: int
    thread_count: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApplicationMetrics:
    """Application-level metrics"""
    active_users: int
    active_sessions: int
    request_count: int
    error_count: int
    response_time_ms: float
    throughput: float
    queue_size: int
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DatabaseMetrics:
    """Database metrics"""
    connection_count: int
    active_queries: int
    query_time_ms: float
    transaction_count: int
    deadlock_count: int
    cache_hit_rate: float
    table_size_mb: float
    index_size_mb: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APIMetrics:
    """API endpoint metrics"""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMetrics:
    """AI agent metrics"""
    agent_name: str
    task_type: str
    execution_time_ms: float
    success: bool
    tokens_used: int
    cost: float
    confidence_score: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsBuffer:
    """Thread-safe metrics buffer"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, metric: Dict[str, Any]) -> None:
        """Add metric to buffer"""
        with self.lock:
            self.buffer.append(metric)
    
    def flush(self) -> List[Dict[str, Any]]:
        """Flush and return all metrics"""
        with self.lock:
            metrics = list(self.buffer)
            self.buffer.clear()
            return metrics
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)


class MetricsCollector:
    """Central metrics collection system"""
    
    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.buffer = MetricsBuffer()
        self.collectors: Dict[str, Callable] = {}
        self.decorators_enabled = True
        
        # Statistics
        self.stats = {
            "total_collected": 0,
            "total_flushed": 0,
            "collection_errors": 0,
            "last_collection": None,
            "last_flush": None
        }
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # System metrics baseline
        self._network_baseline = None
        self._cpu_baseline = None
        
        # Initialize collectors
        self._initialize_collectors()
        
        # Start background tasks
        if self.config.enabled:
            asyncio.create_task(self._start_background_tasks())
    
    def _initialize_collectors(self) -> None:
        """Initialize metric collectors"""
        if self.config.enable_system_metrics:
            self.collectors["system"] = self._collect_system_metrics
        
        if self.config.enable_application_metrics:
            self.collectors["application"] = self._collect_application_metrics
        
        if self.config.enable_database_metrics:
            self.collectors["database"] = self._collect_database_metrics
        
        # Add custom collectors
        for name, collector in self.config.custom_collectors.items():
            self.collectors[name] = collector
    
    async def _start_background_tasks(self) -> None:
        """Start background collection tasks"""
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def _collection_loop(self) -> None:
        """Background task for metric collection"""
        while not self._stop_event.is_set():
            try:
                await self._collect_all_metrics()
                self.stats["last_collection"] = datetime.now()
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self.stats["collection_errors"] += 1
                await asyncio.sleep(10)
    
    async def _flush_loop(self) -> None:
        """Background task for flushing metrics"""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.flush_interval)
                
                if self.buffer.size() >= self.config.batch_size:
                    await self.flush_metrics()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_all_metrics(self) -> None:
        """Collect metrics from all collectors"""
        for name, collector in self.collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector):
                    metrics = await collector()
                else:
                    metrics = await asyncio.get_event_loop().run_in_executor(None, collector)
                
                if metrics:
                    if isinstance(metrics, list):
                        for metric in metrics:
                            self._add_metric(metric, source=name)
                    else:
                        self._add_metric(metrics, source=name)
                        
            except Exception as e:
                logger.error(f"Error collecting metrics from {name}: {e}")
                self.stats["collection_errors"] += 1
    
    def _add_metric(self, metric: Union[Dict[str, Any], object], source: str = "unknown") -> None:
        """Add metric to buffer"""
        try:
            # Convert dataclass to dict if needed
            if hasattr(metric, "__dataclass_fields__"):
                metric_dict = {
                    field.name: getattr(metric, field.name)
                    for field in metric.__dataclass_fields__.values()
                }
            elif isinstance(metric, dict):
                metric_dict = metric
            else:
                metric_dict = {"value": metric}
            
            # Add metadata
            metric_dict["_source"] = source
            metric_dict["_collected_at"] = datetime.now().isoformat()
            
            # Convert datetime objects to strings
            for key, value in metric_dict.items():
                if isinstance(value, datetime):
                    metric_dict[key] = value.isoformat()
            
            self.buffer.add(metric_dict)
            self.stats["total_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error adding metric: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024 ** 3)  # GB
            memory_percent = memory.percent
            memory_available = memory.available / (1024 ** 3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024 ** 3)  # GB
            disk_percent = disk.percent
            disk_available = disk.free / (1024 ** 3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            if self._network_baseline is None:
                self._network_baseline = (network.bytes_sent, network.bytes_recv)
                network_sent = 0
                network_received = 0
            else:
                network_sent = (network.bytes_sent - self._network_baseline[0]) / (1024 ** 2)  # MB
                network_received = (network.bytes_recv - self._network_baseline[1]) / (1024 ** 2)  # MB
                self._network_baseline = (network.bytes_sent, network.bytes_recv)
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Thread count for current process
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                memory_used=memory_used,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_used=disk_used,
                disk_percent=disk_percent,
                disk_available=disk_available,
                network_sent=network_sent,
                network_received=network_received,
                process_count=process_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-level metrics"""
        # This would be implemented based on actual application state
        # For now, return mock data
        return ApplicationMetrics(
            active_users=0,
            active_sessions=0,
            request_count=0,
            error_count=0,
            response_time_ms=0,
            throughput=0,
            queue_size=0,
            cache_hit_rate=0
        )
    
    def _collect_database_metrics(self) -> DatabaseMetrics:
        """Collect database metrics"""
        # This would be implemented based on actual database connection
        # For now, return mock data
        return DatabaseMetrics(
            connection_count=0,
            active_queries=0,
            query_time_ms=0,
            transaction_count=0,
            deadlock_count=0,
            cache_hit_rate=0,
            table_size_mb=0,
            index_size_mb=0
        )
    
    def record_api_call(self, endpoint: str, method: str, status_code: int,
                       response_time_ms: float, request_size: int = 0,
                       response_size: int = 0, error: Optional[str] = None) -> None:
        """Record API call metrics"""
        if not self.config.enable_api_metrics:
            return
        
        metric = APIMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            error=error
        )
        
        self._add_metric(metric, source=MetricSource.API.value)
    
    def record_agent_execution(self, agent_name: str, task_type: str,
                              execution_time_ms: float, success: bool,
                              tokens_used: int = 0, cost: float = 0,
                              confidence_score: float = 0,
                              error: Optional[str] = None) -> None:
        """Record AI agent execution metrics"""
        if not self.config.enable_agent_metrics:
            return
        
        metric = AgentMetrics(
            agent_name=agent_name,
            task_type=task_type,
            execution_time_ms=execution_time_ms,
            success=success,
            tokens_used=tokens_used,
            cost=cost,
            confidence_score=confidence_score,
            error=error
        )
        
        self._add_metric(metric, source=MetricSource.AGENT.value)
    
    def record_custom_metric(self, name: str, value: Union[float, int, Dict[str, Any]],
                           tags: Dict[str, str] = None, source: str = "custom") -> None:
        """Record custom metric"""
        metric = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._add_metric(metric, source=source)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric"""
        self.record_custom_metric(
            name=f"counter.{name}",
            value=value,
            tags=tags,
            source=MetricSource.APPLICATION.value
        )
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a gauge metric"""
        self.record_custom_metric(
            name=f"gauge.{name}",
            value=value,
            tags=tags,
            source=MetricSource.APPLICATION.value
        )
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a histogram metric"""
        self.record_custom_metric(
            name=f"histogram.{name}",
            value=value,
            tags=tags,
            source=MetricSource.APPLICATION.value
        )
    
    def time_execution(self, name: str = None):
        """Decorator to measure execution time"""
        def decorator(func):
            metric_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.decorators_enabled:
                    return await func(*args, **kwargs)
                
                start_time = time.time()
                error = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    execution_time = (time.time() - start_time) * 1000
                    self.record_custom_metric(
                        name=f"execution_time.{metric_name}",
                        value=execution_time,
                        tags={"success": str(error is None)},
                        source=MetricSource.APPLICATION.value
                    )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.decorators_enabled:
                    return func(*args, **kwargs)
                
                start_time = time.time()
                error = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    execution_time = (time.time() - start_time) * 1000
                    self.record_custom_metric(
                        name=f"execution_time.{metric_name}",
                        value=execution_time,
                        tags={"success": str(error is None)},
                        source=MetricSource.APPLICATION.value
                    )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def count_calls(self, name: str = None):
        """Decorator to count function calls"""
        def decorator(func):
            metric_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.decorators_enabled:
                    return await func(*args, **kwargs)
                
                self.increment_counter(f"calls.{metric_name}")
                try:
                    result = await func(*args, **kwargs)
                    self.increment_counter(f"calls.{metric_name}.success")
                    return result
                except Exception as e:
                    self.increment_counter(f"calls.{metric_name}.error")
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.decorators_enabled:
                    return func(*args, **kwargs)
                
                self.increment_counter(f"calls.{metric_name}")
                try:
                    result = func(*args, **kwargs)
                    self.increment_counter(f"calls.{metric_name}.success")
                    return result
                except Exception as e:
                    self.increment_counter(f"calls.{metric_name}.error")
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def register_collector(self, name: str, collector: Callable) -> None:
        """Register a custom metrics collector"""
        self.collectors[name] = collector
        logger.info(f"Registered custom collector: {name}")
    
    def unregister_collector(self, name: str) -> bool:
        """Unregister a metrics collector"""
        if name in self.collectors:
            del self.collectors[name]
            logger.info(f"Unregistered collector: {name}")
            return True
        return False
    
    async def flush_metrics(self) -> List[Dict[str, Any]]:
        """Flush metrics buffer and return metrics"""
        metrics = self.buffer.flush()
        self.stats["total_flushed"] += len(metrics)
        self.stats["last_flush"] = datetime.now()
        return metrics
    
    def get_buffer_size(self) -> int:
        """Get current buffer size"""
        return self.buffer.size()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            **self.stats,
            "buffer_size": self.buffer.size(),
            "collectors": list(self.collectors.keys()),
            "config": {
                "enabled": self.config.enabled,
                "collection_interval": self.config.collection_interval,
                "batch_size": self.config.batch_size,
                "flush_interval": self.config.flush_interval
            }
        }
    
    def enable(self) -> None:
        """Enable metrics collection"""
        self.config.enabled = True
        if not self._collection_task or self._collection_task.done():
            asyncio.create_task(self._start_background_tasks())
    
    def disable(self) -> None:
        """Disable metrics collection"""
        self.config.enabled = False
        self._stop_event.set()
    
    async def shutdown(self) -> None:
        """Shutdown metrics collector"""
        logger.info("Shutting down metrics collector...")
        
        # Stop background tasks
        self._stop_event.set()
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_metrics()
        
        logger.info("Metrics collector shutdown complete")