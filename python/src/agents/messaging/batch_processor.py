"""
Batch Processing System for High-Throughput Operations

Implements efficient batch processing for Agency Swarm:
- Intelligent batching of similar operations
- Dynamic batch sizing based on load
- Parallel batch execution
- Batch optimization and scheduling
- Performance monitoring and tuning

Target Performance:
- 10x improvement in throughput for batch operations
- <50ms average batch processing time
- 1000+ operations per batch
- Automatic batch optimization
- Resource-efficient processing
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import heapq

logger = logging.getLogger(__name__)

class BatchType(Enum):
    """Types of batch operations"""
    AGENT_TASKS = "agent_tasks"
    DATABASE_UPDATES = "database_updates"
    MESSAGE_BROADCASTS = "message_broadcasts"
    KNOWLEDGE_UPDATES = "knowledge_updates"
    METRICS_COLLECTION = "metrics_collection"
    CACHE_UPDATES = "cache_updates"
    FILE_OPERATIONS = "file_operations"
    API_CALLS = "api_calls"

class BatchStrategy(Enum):
    """Batch processing strategies"""
    TIME_BASED = "time_based"          # Batch by time interval
    SIZE_BASED = "size_based"          # Batch by number of items
    HYBRID = "hybrid"                  # Hybrid time/size based
    PRIORITY_BASED = "priority_based"  # Batch by priority
    LOAD_BASED = "load_based"          # Batch by system load

class BatchPriority(Enum):
    """Batch processing priorities"""
    CRITICAL = 1    # Process immediately
    HIGH = 2        # Process soon
    NORMAL = 3      # Normal processing
    LOW = 4         # Process when resources available
    BACKGROUND = 5  # Background processing only

@dataclass
class BatchItem:
    """Individual item in a batch"""
    item_id: str
    operation: str
    data: Dict[str, Any]
    priority: BatchPriority
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if item has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get item age in seconds"""
        return time.time() - self.created_at

    def can_retry(self) -> bool:
        """Check if item can be retried"""
        return self.retry_count < self.max_retries

@dataclass
class Batch:
    """Collection of batch items"""
    batch_id: str
    batch_type: BatchType
    items: List[BatchItem]
    priority: BatchPriority
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_processing_time: float = 30.0  # seconds
    retry_count: int = 0
    max_retries: int = 2
    status: str = "pending"  # pending, scheduled, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def size(self) -> int:
        """Get batch size"""
        return len(self.items)

    @property
    def processing_time(self) -> Optional[float]:
        """Get actual processing time"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    @property
    def wait_time(self) -> float:
        """Get wait time before processing"""
        if self.scheduled_at and self.started_at:
            return self.started_at - self.scheduled_at
        elif self.created_at and self.started_at:
            return self.started_at - self.created_at
        return 0

    @property
    def total_time(self) -> float:
        """Get total time from creation to completion"""
        if self.completed_at:
            return self.completed_at - self.created_at
        elif self.started_at:
            return time.time() - self.created_at
        return 0

@dataclass
class BatchMetrics:
    """Batch processing metrics"""
    batch_type: BatchType
    total_batches: int
    total_items: int
    completed_batches: int
    failed_batches: int
    avg_batch_size: float
    avg_processing_time_ms: float
    avg_wait_time_ms: float
    throughput_items_per_sec: float
    success_rate_percent: float
    last_updated: float = field(default_factory=time.time)

class BatchProcessor:
    """Handles batch processing for a specific type"""

    def __init__(self,
                 batch_type: BatchType,
                 processing_function: Callable,
                 max_batch_size: int = 100,
                 max_wait_time: float = 5.0,
                 max_concurrent_batches: int = 4):

        self.batch_type = batch_type
        self.processing_function = processing_function
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches

        # Batch management
        self.pending_items = defaultdict(list)  # priority -> items
        self.processing_batches = {}  # batch_id -> Batch
        self.completed_batches = {}  # batch_id -> Batch
        self.failed_batches = {}  # batch_id -> Batch

        # Metrics
        self.total_items_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.total_wait_time = 0.0
        self.error_count = 0

        # Processing control
        self.running = False
        self.processor_thread = None
        self.batch_executor = ThreadPoolExecutor(max_workers=max_concurrent_batches)

        # Performance optimization
        self.dynamic_sizing_enabled = True
        self.load_monitoring_enabled = True
        self.current_system_load = 0.0

    def start(self):
        """Start batch processor"""
        logger.info(f"Starting batch processor for {self.batch_type.value}")
        self.running = True

        # Start processor thread
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

        logger.info(f"Batch processor for {self.batch_type.value} started")

    def stop(self):
        """Stop batch processor"""
        logger.info(f"Stopping batch processor for {self.batch_type.value}")
        self.running = False

        # Wait for processor thread
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=10)

        # Shutdown executor
        self.batch_executor.shutdown(wait=True)

        logger.info(f"Batch processor for {self.batch_type.value} stopped")

    def add_item(self,
                 operation: str,
                 data: Dict[str, Any],
                 priority: BatchPriority = BatchPriority.NORMAL,
                 expires_in_seconds: Optional[int] = None,
                 dependencies: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:

        """Add item to batch processing queue"""
        item_id = str(uuid.uuid4())

        item = BatchItem(
            item_id=item_id,
            operation=operation,
            data=data,
            priority=priority,
            expires_at=time.time() + expires_in_seconds if expires_in_seconds else None,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )

        # Add to pending queue
        self.pending_items[priority].append(item)

        logger.debug(f"Added item {item_id} to batch processor")
        return item_id

    def _processing_loop(self):
        """Main batch processing loop"""
        last_batch_time = time.time()

        while self.running:
            try:
                current_time = time.time()

                # Check if we should create a new batch
                should_create_batch = False

                # Time-based batching
                if current_time - last_batch_time >= self.max_wait_time:
                    should_create_batch = True

                # Size-based batching
                total_pending = sum(len(items) for items in self.pending_items.values())
                if total_pending >= self.max_batch_size:
                    should_create_batch = True

                # Priority-based batching (critical items)
                if self.pending_items[BatchPriority.CRITICAL]:
                    should_create_batch = True

                if should_create_batch and self._can_create_batch():
                    batch = self._create_batch()
                    if batch:
                        last_batch_time = current_time
                        self._schedule_batch(batch)

                # Update system load
                if self.load_monitoring_enabled:
                    self._update_system_load()

                # Optimize batch size based on load
                if self.dynamic_sizing_enabled:
                    self._optimize_batch_size()

                # Sleep for next cycle
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(1)

    def _can_create_batch(self) -> bool:
        """Check if we can create a new batch"""
        active_batches = len(self.processing_batches)
        return active_batches < self.max_concurrent_batches

    def _create_batch(self) -> Optional[Batch]:
        """Create a new batch from pending items"""
        if not any(self.pending_items.values()):
            return None

        # Collect items in priority order
        items = []
        for priority in [BatchPriority.CRITICAL, BatchPriority.HIGH, BatchPriority.NORMAL, BatchPriority.LOW, BatchPriority.BACKGROUND]:
            priority_items = self.pending_items[priority]

            # Filter out expired items
            valid_items = [item for item in priority_items if not item.is_expired]

            # Add items to batch (respect max batch size)
            remaining_space = self.max_batch_size - len(items)
            if remaining_space > 0:
                items.extend(valid_items[:remaining_space])
                self.pending_items[priority] = valid_items[remaining_space:]

            if len(items) >= self.max_batch_size:
                break

        if not items:
            return None

        # Determine batch priority (highest priority item)
        batch_priority = min(item.priority for item in items)

        batch = Batch(
            batch_id=str(uuid.uuid4()),
            batch_type=self.batch_type,
            items=items,
            priority=batch_priority,
            scheduled_at=time.time()
        )

        logger.debug(f"Created batch {batch.batch_id} with {len(items)} items")
        return batch

    def _schedule_batch(self, batch: Batch):
        """Schedule batch for processing"""
        batch.status = "scheduled"
        self.processing_batches[batch.batch_id] = batch

        # Submit batch for processing
        future = self.batch_executor.submit(self._process_batch, batch)
        future.add_done_callback(lambda f: self._handle_batch_completion(f, batch))

    def _process_batch(self, batch: Batch) -> Dict[str, Any]:
        """Process a batch of items"""
        batch.status = "processing"
        batch.started_at = time.time()

        try:
            # Prepare batch data
            batch_data = {
                "batch_id": batch.batch_id,
                "batch_type": batch.batch_type.value,
                "items": [
                    {
                        "item_id": item.item_id,
                        "operation": item.operation,
                        "data": item.data,
                        "metadata": item.metadata
                    }
                    for item in batch.items
                ],
                "priority": batch.priority.value,
                "created_at": batch.created_at
            }

            # Process batch
            result = self.processing_function(batch_data)

            # Update metrics
            batch.completed_at = time.time()
            batch.status = "completed"
            batch.result = result

            processing_time = batch.processing_time
            wait_time = batch.wait_time

            self.total_items_processed += len(batch.items)
            self.total_batches_processed += 1
            self.total_processing_time += processing_time
            self.total_wait_time += wait_time

            logger.info(f"Processed batch {batch.batch_id}: {len(batch.items)} items in {processing_time:.2f}s")

            return result

        except Exception as e:
            # Handle batch failure
            batch.completed_at = time.time()
            batch.status = "failed"
            batch.error = str(e)

            self.error_count += 1

            logger.error(f"Batch {batch.batch_id} failed: {e}")

            # Retry if possible
            if batch.retry_count < batch.max_retries:
                batch.retry_count += 1
                batch.status = "pending"
                self._reschedule_failed_batch(batch)
            else:
                # Move to failed batches
                self.failed_batches[batch.batch_id] = batch

            raise

    def _handle_batch_completion(self, future, batch: Batch):
        """Handle batch processing completion"""
        try:
            result = future.result()
            # Batch was completed successfully
            self.completed_batches[batch.batch_id] = batch
        except Exception as e:
            # Batch failed (already handled in _process_batch)
            pass

        # Remove from processing batches
        self.processing_batches.pop(batch.batch_id, None)

    def _reschedule_failed_batch(self, batch: Batch):
        """Reschedule failed batch with exponential backoff"""
        # Retry with smaller batch size
        if len(batch.items) > 1:
            # Split batch into smaller batches
            mid_point = len(batch.items) // 2
            first_half_items = batch.items[:mid_point]
            second_half_items = batch.items[mid_point:]

            # Create new batches
            for i, half_items in enumerate([first_half_items, second_half_items]):
                if half_items:
                    new_batch = Batch(
                        batch_id=str(uuid.uuid4()),
                        batch_type=batch.batch_type,
                        items=half_items,
                        priority=batch.priority,
                        created_at=time.time(),
                        max_retries=batch.max_retries - batch.retry_count,
                        retry_count=batch.retry_count
                    )
                    self._schedule_batch(new_batch)
        else:
            # Retry single item batch
            self._schedule_batch(batch)

    def _update_system_load(self):
        """Update system load metrics"""
        # Calculate current load based on active batches and queue size
        active_batches = len(self.processing_batches)
        pending_items = sum(len(items) for items in self.pending_items.values())

        # Normalize load (0.0 to 1.0)
        load_factor = (active_batches / self.max_concurrent_batches) * 0.7 + (pending_items / self.max_batch_size) * 0.3
        self.current_system_load = min(1.0, load_factor)

    def _optimize_batch_size(self):
        """Dynamically optimize batch size based on system load"""
        if self.current_system_load > 0.8:
            # High load - reduce batch size
            new_size = max(10, int(self.max_batch_size * 0.7))
            if new_size != self.max_batch_size:
                self.max_batch_size = new_size
                logger.info(f"Reduced batch size to {new_size} due to high load")
        elif self.current_system_load < 0.3:
            # Low load - increase batch size
            new_size = min(500, int(self.max_batch_size * 1.2))
            if new_size != self.max_batch_size:
                self.max_batch_size = new_size
                logger.info(f"Increased batch size to {new_size} due to low load")

    def get_metrics(self) -> BatchMetrics:
        """Get batch processing metrics"""
        total_batches = len(self.completed_batches) + len(self.failed_batches)
        total_items = sum(len(batch.items) for batch in self.completed_batches.values())

        if self.total_batches_processed > 0:
            avg_batch_size = self.total_items_processed / self.total_batches_processed
            avg_processing_time = (self.total_processing_time / self.total_batches_processed) * 1000
            avg_wait_time = (self.total_wait_time / self.total_batches_processed) * 1000
        else:
            avg_batch_size = 0
            avg_processing_time = 0
            avg_wait_time = 0

        # Calculate throughput
        uptime = time.time() - getattr(self, 'start_time', time.time())
        throughput = self.total_items_processed / max(uptime, 1)

        # Calculate success rate
        success_rate = (self.total_batches_processed / max(total_batches, 1)) * 100

        return BatchMetrics(
            batch_type=self.batch_type,
            total_batches=total_batches,
            total_items=total_items,
            completed_batches=len(self.completed_batches),
            failed_batches=len(self.failed_batches),
            avg_batch_size=avg_batch_size,
            avg_processing_time_ms=avg_processing_time,
            avg_wait_time_ms=avg_wait_time,
            throughput_items_per_sec=throughput,
            success_rate_percent=success_rate
        )

class BatchProcessingManager:
    """Manages multiple batch processors"""

    def __init__(self):
        self.processors: Dict[BatchType, BatchProcessor] = {}
        self.processor_configs: Dict[BatchType, Dict] = {}
        self.running = False

    def register_processor(self,
                          batch_type: BatchType,
                          processing_function: Callable,
                          config: Optional[Dict] = None) -> bool:

        """Register a batch processor"""
        try:
            if config is None:
                config = {}

            processor = BatchProcessor(
                batch_type=batch_type,
                processing_function=processing_function,
                max_batch_size=config.get("max_batch_size", 100),
                max_wait_time=config.get("max_wait_time", 5.0),
                max_concurrent_batches=config.get("max_concurrent_batches", 4)
            )

            self.processors[batch_type] = processor
            self.processor_configs[batch_type] = config

            logger.info(f"Registered batch processor for {batch_type.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to register batch processor for {batch_type.value}: {e}")
            return False

    def start(self):
        """Start all batch processors"""
        logger.info("Starting Batch Processing Manager")
        self.running = True

        for processor in self.processors.values():
            processor.start()

        logger.info("Batch Processing Manager started")

    def stop(self):
        """Stop all batch processors"""
        logger.info("Stopping Batch Processing Manager")
        self.running = False

        for processor in self.processors.values():
            processor.stop()

        logger.info("Batch Processing Manager stopped")

    def add_batch_item(self,
                       batch_type: BatchType,
                       operation: str,
                       data: Dict[str, Any],
                       priority: BatchPriority = BatchPriority.NORMAL,
                       **kwargs) -> str:

        """Add item to specific batch processor"""
        processor = self.processors.get(batch_type)
        if not processor:
            logger.error(f"No processor registered for batch type: {batch_type.value}")
            return ""

        return processor.add_item(operation, data, priority, **kwargs)

    def get_processor_metrics(self, batch_type: BatchType) -> Optional[BatchMetrics]:
        """Get metrics for specific processor"""
        processor = self.processors.get(batch_type)
        if processor:
            return processor.get_metrics()
        return None

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all processors"""
        metrics = {}
        for batch_type, processor in self.processors.items():
            metrics[batch_type.value] = processor.get_metrics().__dict__

        # Calculate global metrics
        total_items = sum(m["total_items"] for m in metrics.values())
        total_batches = sum(m["total_batches"] for m in metrics.values())
        total_failed = sum(m["failed_batches"] for m in metrics.values())

        return {
            "processors": metrics,
            "summary": {
                "total_processors": len(self.processors),
                "total_items_processed": total_items,
                "total_batches_processed": total_batches,
                "total_failed_batches": total_failed,
                "overall_success_rate": ((total_batches - total_failed) / max(total_batches, 1)) * 100
            }
        }

# Example usage and test functions
def process_agent_tasks_batch(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Example batch processing function for agent tasks"""
    items = batch_data["items"]
    results = []

    for item in items:
        # Simulate processing
        time.sleep(0.001)  # 1ms per item
        results.append({
            "item_id": item["item_id"],
            "status": "completed",
            "result": f"Processed {item['operation']}"
        })

    return {
        "batch_id": batch_data["batch_id"],
        "processed_items": len(results),
        "results": results
    }

def process_database_updates_batch(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Example batch processing function for database updates"""
    items = batch_data["items"]
    results = []

    for item in items:
        # Simulate database operation
        time.sleep(0.002)  # 2ms per item
        results.append({
            "item_id": item["item_id"],
            "status": "completed",
            "rows_affected": 1
        })

    return {
        "batch_id": batch_data["batch_id"],
        "processed_items": len(results),
        "total_rows_affected": len(results)
    }

if __name__ == "__main__":
    # Create batch processing manager
    manager = BatchProcessingManager()

    try:
        # Register processors
        manager.register_processor(
            BatchType.AGENT_TASKS,
            process_agent_tasks_batch,
            {"max_batch_size": 50, "max_wait_time": 2.0}
        )

        manager.register_processor(
            BatchType.DATABASE_UPDATES,
            process_database_updates_batch,
            {"max_batch_size": 100, "max_wait_time": 5.0}
        )

        # Start manager
        manager.start()

        # Add some items
        for i in range(150):
            manager.add_batch_item(
                BatchType.AGENT_TASKS,
                "execute_task",
                {"task_id": f"task_{i}", "command": "process"},
                priority=BatchPriority.NORMAL
            )

        for i in range(200):
            manager.add_batch_item(
                BatchType.DATABASE_UPDATES,
                "update_record",
                {"table": "agents", "id": f"agent_{i}", "field": "status", "value": "active"},
                priority=BatchPriority.HIGH
            )

        # Wait for processing
        time.sleep(10)

        # Get metrics
        metrics = manager.get_all_metrics()
        print(f"Batch processing metrics: {metrics}")

    finally:
        manager.stop()