"""
Advanced Distributed Cache
High-performance distributed caching with consistency, partitioning, and replication
"""

import asyncio
import logging
import time
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, OrderedDict
import threading
import zlib
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"


class ConsistencyLevel(Enum):
    """Cache consistency levels"""
    EVENTUAL = "eventual"  # Eventually consistent
    STRONG = "strong"     # Strong consistency
    WEAK = "weak"         # Weak consistency


class SerializationFormat(Enum):
    """Serialization formats"""
    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"


class CacheNodeStatus(Enum):
    """Cache node status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    FAILED = "failed"


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[timedelta] = None
    version: int = 1
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl:
            return datetime.now() - self.created_at > self.ttl
        return False
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self) -> None:
        """Update access time and count"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl.total_seconds() if self.ttl else None,
            "version": self.version,
            "size_bytes": self.size_bytes,
            "is_expired": self.is_expired,
            "age_seconds": self.age_seconds,
            "metadata": self.metadata
        }


@dataclass
class CacheNode:
    """Distributed cache node"""
    node_id: str
    host: str
    port: int
    weight: int = 100
    status: CacheNodeStatus = CacheNodeStatus.ACTIVE
    last_heartbeat: Optional[datetime] = None
    total_memory: int = 0
    used_memory: int = 0
    entries_count: int = 0
    hit_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_usage_percent(self) -> float:
        """Get memory usage percentage"""
        if self.total_memory > 0:
            return (self.used_memory / self.total_memory) * 100
        return 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy"""
        if self.status != CacheNodeStatus.ACTIVE:
            return False
        if self.last_heartbeat:
            return datetime.now() - self.last_heartbeat < timedelta(minutes=2)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "memory_usage_percent": self.memory_usage_percent,
            "total_memory": self.total_memory,
            "used_memory": self.used_memory,
            "entries_count": self.entries_count,
            "hit_rate": self.hit_rate,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None
        }


class ConsistentHashRing:
    """Consistent hashing for cache partitioning"""
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self.nodes: Set[str] = set()
    
    def add_node(self, node_id: str) -> None:
        """Add node to hash ring"""
        if node_id in self.nodes:
            return
        
        self.nodes.add(node_id)
        
        for i in range(self.virtual_nodes):
            key = self._hash(f"{node_id}:{i}")
            self.ring[key] = node_id
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from hash ring"""
        if node_id not in self.nodes:
            return
        
        self.nodes.discard(node_id)
        
        keys_to_remove = []
        for key, nid in self.ring.items():
            if nid == node_id:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for given key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find first node clockwise from hash
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # Wrap around to first node
        return self.ring[self.sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring or count <= 0:
            return []
        
        hash_key = self._hash(key)
        nodes = []
        seen_nodes = set()
        
        # Find starting position
        start_idx = 0
        for i, ring_key in enumerate(self.sorted_keys):
            if ring_key >= hash_key:
                start_idx = i
                break
        
        # Collect nodes in clockwise order
        for i in range(len(self.sorted_keys)):
            idx = (start_idx + i) % len(self.sorted_keys)
            node_id = self.ring[self.sorted_keys[idx]]
            
            if node_id not in seen_nodes:
                nodes.append(node_id)
                seen_nodes.add(node_id)
                
                if len(nodes) >= count:
                    break
        
        return nodes
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class LocalCache:
    """Local in-memory cache with eviction policies"""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict = OrderedDict()  # For LRU
        self.frequency_counter: Dict[str, int] = defaultdict(int)  # For LFU
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            entry = self.entries.get(key)
            
            if not entry:
                self.misses += 1
                return None
            
            if entry.is_expired:
                self._remove(key)
                self.misses += 1
                return None
            
            # Update access tracking
            entry.touch()
            self._update_access_tracking(key)
            
            self.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            # Check if we need to evict
            if key not in self.entries and len(self.entries) >= self.max_size:
                self._evict()
            
            # Calculate size
            size_bytes = len(pickle.dumps(value)) if value else 0
            
            # Create or update entry
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Update existing entry version
            if key in self.entries:
                entry.version = self.entries[key].version + 1
            
            self.entries[key] = entry
            self._update_access_tracking(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.lock:
            return self._remove(key)
    
    def _remove(self, key: str) -> bool:
        """Remove entry (internal)"""
        if key in self.entries:
            del self.entries[key]
            self.access_order.pop(key, None)
            self.frequency_counter.pop(key, None)
            return True
        return False
    
    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for eviction strategies"""
        if self.strategy == CacheStrategy.LRU:
            # Move to end for LRU
            self.access_order.pop(key, None)
            self.access_order[key] = datetime.now()
        elif self.strategy == CacheStrategy.LFU:
            # Increment frequency counter
            self.frequency_counter[key] += 1
    
    def _evict(self) -> None:
        """Evict entries based on strategy"""
        if not self.entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = next(iter(self.access_order))
            self._remove(oldest_key)
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            lfu_key = min(self.frequency_counter.keys(), 
                         key=lambda k: self.frequency_counter[k])
            self._remove(lfu_key)
            
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = [key for key, entry in self.entries.items() 
                          if entry.is_expired]
            for key in expired_keys:
                self._remove(key)
            
            # If no expired entries, use LRU
            if len(self.entries) >= self.max_size:
                self._evict_lru()
                
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest entry by creation time
            oldest_key = min(self.entries.keys(), 
                           key=lambda k: self.entries[k].created_at)
            self._remove(oldest_key)
            
        elif self.strategy == CacheStrategy.RANDOM:
            # Remove random entry
            import random
            random_key = random.choice(list(self.entries.keys()))
            self._remove(random_key)
        
        self.evictions += 1
    
    def _evict_lru(self) -> None:
        """Evict using LRU when other strategies fallback"""
        if self.access_order:
            oldest_key = next(iter(self.access_order))
            self._remove(oldest_key)
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        with self.lock:
            expired_keys = [key for key, entry in self.entries.items() 
                          if entry.is_expired]
            
            for key in expired_keys:
                self._remove(key)
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self.entries.values())
            
            return {
                "entries": len(self.entries),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "total_size_bytes": total_size,
                "strategy": self.strategy.value
            }


class DistributedCache:
    """Main distributed cache system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cluster_id = self.config.get("cluster_id", str(uuid.uuid4()))
        
        # Cache configuration
        self.replication_factor = self.config.get("replication_factor", 2)
        self.consistency_level = ConsistencyLevel(self.config.get("consistency_level", "eventual"))
        self.serialization_format = SerializationFormat(self.config.get("serialization", "pickle"))
        
        # Local cache
        local_config = self.config.get("local_cache", {})
        self.local_cache = LocalCache(
            max_size=local_config.get("max_size", 10000),
            strategy=CacheStrategy(local_config.get("strategy", "lru"))
        )
        
        # Distributed components
        self.nodes: Dict[str, CacheNode] = {}
        self.hash_ring = ConsistentHashRing(self.config.get("virtual_nodes", 150))
        
        # Redis integration (optional)
        self.redis_client: Optional[redis.Redis] = None
        if self.config.get("redis_url"):
            self._initialize_redis()
        
        # Synchronization
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30)
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "network_hits": 0,
            "network_misses": 0,
            "replication_syncs": 0,
            "consistency_conflicts": 0
        }
        
        self.running = False
    
    def _initialize_redis(self) -> None:
        """Initialize Redis client for distributed backend"""
        try:
            redis_url = self.config["redis_url"]
            self.redis_client = redis.from_url(redis_url)
            logger.info("Redis client initialized for distributed cache backend")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
    
    async def start(self) -> None:
        """Start distributed cache system"""
        if self.running:
            return
        
        self.running = True
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info(f"Distributed cache cluster {self.cluster_id} started")
    
    async def stop(self) -> None:
        """Stop distributed cache system"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop sync tasks
        for task in self.sync_tasks.values():
            task.cancel()
        
        if self.sync_tasks:
            await asyncio.gather(*self.sync_tasks.values(), return_exceptions=True)
        
        self.sync_tasks.clear()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Distributed cache stopped")
    
    def add_node(self, node_id: str, host: str, port: int, weight: int = 100) -> CacheNode:
        """Add cache node to cluster"""
        node = CacheNode(
            node_id=node_id,
            host=host,
            port=port,
            weight=weight,
            total_memory=self.config.get("node_memory", 1000000000)  # 1GB default
        )
        
        self.nodes[node_id] = node
        self.hash_ring.add_node(node_id)
        
        # Start sync task for new node
        if self.running:
            task = asyncio.create_task(self._sync_with_node(node_id))
            self.sync_tasks[node_id] = task
        
        logger.info(f"Added cache node {node_id} ({host}:{port})")
        return node
    
    def remove_node(self, node_id: str) -> bool:
        """Remove cache node from cluster"""
        if node_id not in self.nodes:
            return False
        
        # Stop sync task
        if node_id in self.sync_tasks:
            self.sync_tasks[node_id].cancel()
            del self.sync_tasks[node_id]
        
        # Remove from hash ring and nodes
        self.hash_ring.remove_node(node_id)
        del self.nodes[node_id]
        
        logger.info(f"Removed cache node {node_id}")
        return True
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from distributed cache"""
        self.metrics["total_requests"] += 1
        
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            self.metrics["cache_hits"] += 1
            return value
        
        # Try Redis if available
        if self.redis_client:
            try:
                redis_value = await self.redis_client.get(key)
                if redis_value:
                    value = self._deserialize(redis_value)
                    # Update local cache
                    self.local_cache.set(key, value)
                    self.metrics["cache_hits"] += 1
                    self.metrics["network_hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {str(e)}")
        
        # Try distributed nodes
        target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
        
        for node_id in target_nodes:
            node = self.nodes.get(node_id)
            if node and node.is_healthy:
                try:
                    value = await self._get_from_node(node, key)
                    if value is not None:
                        # Update local cache and Redis
                        self.local_cache.set(key, value)
                        if self.redis_client:
                            await self.redis_client.set(key, self._serialize(value))
                        
                        self.metrics["cache_hits"] += 1
                        self.metrics["network_hits"] += 1
                        return value
                except Exception as e:
                    logger.error(f"Failed to get from node {node_id}: {str(e)}")
                    continue
        
        self.metrics["cache_misses"] += 1
        self.metrics["network_misses"] += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in distributed cache"""
        self.metrics["total_requests"] += 1
        
        # Set in local cache
        self.local_cache.set(key, value, ttl)
        
        # Set in Redis if available
        if self.redis_client:
            try:
                serialized_value = self._serialize(value)
                if ttl:
                    await self.redis_client.setex(key, int(ttl.total_seconds()), serialized_value)
                else:
                    await self.redis_client.set(key, serialized_value)
            except Exception as e:
                logger.error(f"Redis set error: {str(e)}")
        
        # Set in distributed nodes based on consistency level
        target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
        
        if self.consistency_level == ConsistencyLevel.STRONG:
            # Wait for all replicas to confirm
            success_count = 0
            for node_id in target_nodes:
                node = self.nodes.get(node_id)
                if node and node.is_healthy:
                    try:
                        success = await self._set_on_node(node, key, value, ttl)
                        if success:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to set on node {node_id}: {str(e)}")
            
            return success_count >= (self.replication_factor // 2 + 1)  # Majority
            
        else:
            # Fire and forget for eventual consistency
            for node_id in target_nodes:
                node = self.nodes.get(node_id)
                if node and node.is_healthy:
                    asyncio.create_task(self._set_on_node_async(node, key, value, ttl))
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from distributed cache"""
        self.metrics["total_requests"] += 1
        
        # Delete from local cache
        self.local_cache.delete(key)
        
        # Delete from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {str(e)}")
        
        # Delete from distributed nodes
        target_nodes = self.hash_ring.get_nodes(key, self.replication_factor)
        success_count = 0
        
        for node_id in target_nodes:
            node = self.nodes.get(node_id)
            if node and node.is_healthy:
                try:
                    success = await self._delete_from_node(node, key)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete from node {node_id}: {str(e)}")
        
        return success_count > 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        value = await self.get(key)
        return value is not None
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        # Clear local cache
        self.local_cache.entries.clear()
        self.local_cache.access_order.clear()
        self.local_cache.frequency_counter.clear()
        
        # Clear Redis
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis clear error: {str(e)}")
        
        # Clear all nodes
        success_count = 0
        for node in self.nodes.values():
            if node.is_healthy:
                try:
                    success = await self._clear_node(node)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to clear node {node.node_id}: {str(e)}")
        
        return success_count > 0
    
    async def _get_from_node(self, node: CacheNode, key: str) -> Optional[Any]:
        """Get value from specific node"""
        # Simulate network call to cache node
        # In production, would use HTTP/gRPC client
        await asyncio.sleep(0.001)  # 1ms network latency
        
        # Simulate cache miss on remote node
        import random
        if random.random() > 0.7:  # 70% hit rate on remote nodes
            return None
        
        # Simulate returning cached value
        return f"cached_value_for_{key}"
    
    async def _set_on_node(self, node: CacheNode, key: str, value: Any, ttl: Optional[timedelta]) -> bool:
        """Set value on specific node"""
        # Simulate network call
        await asyncio.sleep(0.002)  # 2ms network latency
        
        # Update node stats
        node.entries_count += 1
        node.used_memory += len(str(value))
        
        return True
    
    async def _set_on_node_async(self, node: CacheNode, key: str, value: Any, ttl: Optional[timedelta]) -> None:
        """Set value on node asynchronously"""
        try:
            await self._set_on_node(node, key, value, ttl)
        except Exception as e:
            logger.error(f"Async set failed on node {node.node_id}: {str(e)}")
    
    async def _delete_from_node(self, node: CacheNode, key: str) -> bool:
        """Delete value from specific node"""
        # Simulate network call
        await asyncio.sleep(0.001)
        
        # Update node stats
        node.entries_count = max(0, node.entries_count - 1)
        
        return True
    
    async def _clear_node(self, node: CacheNode) -> bool:
        """Clear all entries from node"""
        # Simulate network call
        await asyncio.sleep(0.005)
        
        # Reset node stats
        node.entries_count = 0
        node.used_memory = 0
        
        return True
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.serialization_format == SerializationFormat.PICKLE:
            return pickle.dumps(value)
        elif self.serialization_format == SerializationFormat.JSON:
            return json.dumps(value).encode('utf-8')
        else:
            return pickle.dumps(value)  # Fallback
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.serialization_format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif self.serialization_format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)  # Fallback
    
    async def _sync_with_node(self, node_id: str) -> None:
        """Synchronization task for cache node"""
        while self.running and node_id in self.nodes:
            try:
                node = self.nodes[node_id]
                
                # Perform sync operations
                await self._perform_node_sync(node)
                
                await asyncio.sleep(60)  # Sync every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error with node {node_id}: {str(e)}")
                await asyncio.sleep(30)
    
    async def _perform_node_sync(self, node: CacheNode) -> None:
        """Perform synchronization with node"""
        # Update node stats
        node.last_heartbeat = datetime.now()
        
        # Simulate getting node stats
        node.hit_rate = 85.0  # Simulate 85% hit rate
        
        self.metrics["replication_syncs"] += 1
    
    async def _cleanup_loop(self) -> None:
        """Cleanup expired entries periodically"""
        while self.running:
            try:
                # Clean up local cache
                expired_count = self.local_cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired entries from local cache")
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to maintain cluster membership"""
        while self.running:
            try:
                # Update heartbeats for all nodes
                for node in self.nodes.values():
                    if node.status == CacheNodeStatus.ACTIVE:
                        node.last_heartbeat = datetime.now()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {str(e)}")
                await asyncio.sleep(30)
    
    def get_status(self) -> Dict[str, Any]:
        """Get cache cluster status"""
        healthy_nodes = len([n for n in self.nodes.values() if n.is_healthy])
        local_stats = self.local_cache.get_stats()
        
        total_requests = self.metrics["total_requests"]
        cache_hit_rate = 0.0
        if total_requests > 0:
            cache_hit_rate = (self.metrics["cache_hits"] / total_requests) * 100
        
        return {
            "cluster_id": self.cluster_id,
            "running": self.running,
            "replication_factor": self.replication_factor,
            "consistency_level": self.consistency_level.value,
            "total_nodes": len(self.nodes),
            "healthy_nodes": healthy_nodes,
            "local_cache": local_stats,
            "cache_hit_rate": cache_hit_rate,
            "redis_enabled": self.redis_client is not None
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed cache metrics"""
        local_stats = self.local_cache.get_stats()
        node_stats = [node.to_dict() for node in self.nodes.values()]
        
        return {
            "cluster_metrics": self.metrics.copy(),
            "local_cache": local_stats,
            "nodes": node_stats,
            "hash_ring_nodes": len(self.hash_ring.nodes)
        }
    
    def get_node_status(self, node_id: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get status of specific node or all nodes"""
        if node_id:
            node = self.nodes.get(node_id)
            return node.to_dict() if node else None
        else:
            return [node.to_dict() for node in self.nodes.values()]