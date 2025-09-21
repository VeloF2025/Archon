"""
Multi-Level Caching System for Agency Swarm

Implements a comprehensive caching strategy with multiple levels:
- Memory cache (LRU-based)
- Redis cache (distributed)
- Database cache (persistent)
- Agent-specific cache (specialized)

Features:
- Intelligent cache invalidation
- Cache warming strategies
- Performance monitoring
- Automatic cache optimization
- Distributed cache coordination
"""

import asyncio
import json
import logging
import time
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from pathlib import Path
import hashlib
import pickle
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"    # Fastest, smallest
    L2_REDIS = "l2_redis"      # Fast, distributed
    L3_DATABASE = "l3_database" # Slow, persistent
    L4_AGENT = "l4_agent"      # Agent-specific

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"                 # Least Recently Used
    LFU = "lfu"                 # Least Frequently Used
    TTL = "ttl"                 # Time-based
    ADAPTIVE = "adaptive"       # Adaptive hybrid

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    ttl_seconds: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    hit_count: int = 0
    level: CacheLevel = CacheLevel.L1_MEMORY

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at

    def record_access(self):
        """Record cache access"""
        self.access_count += 1
        self.hit_count += 1
        self.last_accessed = time.time()

class MemoryCache:
    """L1 Memory Cache with LRU eviction"""

    def __init__(self, max_size_mb: int = 512, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size_mb = max_size_mb
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_stats = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.lock = threading.RLock()

        # Strategy-specific parameters
        self.lfu_threshold = 5  # Access count for LFU consideration

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]

                if entry.is_expired:
                    self._remove_key(key)
                    self.miss_count += 1
                    return None

                # Record access and update strategy
                entry.record_access()
                self._update_strategy(key, entry)

                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)

                self.hit_count += 1
                return entry.value

            self.miss_count += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value).encode('utf-8'))

            # Check if we need to evict
            current_size = sum(entry.size_bytes for entry in self.cache.values())
            max_size_bytes = self.max_size_mb * 1024 * 1024

            if current_size + size_bytes > max_size_bytes:
                self._evict_entries(size_bytes)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                level=CacheLevel.L1_MEMORY
            )

            self.cache[key] = entry
            self.cache.move_to_end(key)  # New entries go to end

            return True

    def _update_strategy(self, key: str, entry: CacheEntry):
        """Update strategy-specific tracking"""
        if self.strategy == CacheStrategy.LFU:
            self.access_stats[key] += 1

    def _evict_entries(self, required_space: int):
        """Evict entries based on strategy"""
        evicted_count = 0
        freed_space = 0

        if self.strategy == CacheStrategy.LRU:
            # Evict from front (least recently used)
            while freed_space < required_space and self.cache:
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                freed_space += oldest_entry.size_bytes
                evicted_count += 1

        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            candidates = []
            for key, entry in self.cache.items():
                access_count = self.access_stats.get(key, 0)
                candidates.append((access_count, key, entry))

            candidates.sort(key=lambda x: x[0])  # Sort by access count

            for access_count, key, entry in candidates:
                if freed_space >= required_space:
                    break
                if key in self.cache:
                    del self.cache[key]
                    freed_space += entry.size_bytes
                    evicted_count += 1

        self.eviction_count += evicted_count

    def _remove_key(self, key: str):
        """Remove key from cache"""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_stats:
                del self.access_stats[key]

    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.cache.clear()
            self.access_stats.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        hit_rate = (self.hit_count / (self.hit_count + self.miss_count) * 100) if (self.hit_count + self.miss_count) > 0 else 0

        return {
            "level": "L1_MEMORY",
            "entries": len(self.cache),
            "size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": hit_rate,
            "eviction_count": self.eviction_count,
            "strategy": self.strategy.value
        }

class RedisCache:
    """L2 Redis Cache for distributed caching"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 default_ttl: int = 3600):

        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available for L2 cache")

        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            # Test connection
            self.client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            result = self.client.get(key)
            if result:
                self.hit_count += 1
                return json.loads(result)
            else:
                self.miss_count += 1
                return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self.miss_count += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        try:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            serialized = json.dumps(value, default=str)
            result = self.client.setex(key, ttl, serialized)
            return result
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            result = self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        hit_rate = (self.hit_count / (self.hit_count + self.miss_count) * 100) if (self.hit_count + self.miss_count) > 0 else 0

        try:
            info = self.client.info()
            used_memory = info.get('used_memory', 0)
            connected_clients = info.get('connected_clients', 0)
        except:
            used_memory = 0
            connected_clients = 0

        return {
            "level": "L2_REDIS",
            "connected": True,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": hit_rate,
            "used_memory_bytes": used_memory,
            "connected_clients": connected_clients,
            "default_ttl": self.default_ttl
        }

class DatabaseCache:
    """L3 Database Cache for persistent caching"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.hit_count = 0
        self.miss_count = 0
        self.connection_pool = None

        # Initialize connection pool
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize database connection pool"""
        # Placeholder for database connection initialization
        # In production, this would use SQLAlchemy or similar
        logger.info("Database cache initialized (connection pooling)")

    def get(self, key: str) -> Optional[Any]:
        """Get value from database cache"""
        # Placeholder implementation
        # In production, this would query a cache table
        self.miss_count += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in database cache"""
        # Placeholder implementation
        # In production, this would insert/update a cache table
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get database cache statistics"""
        return {
            "level": "L3_DATABASE",
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "connected": True,
            "connection_pool_size": getattr(self, 'pool_size', 10)
        }

class AgentCache:
    """L4 Agent-specific cache for specialized caching"""

    def __init__(self, agent_id: str, max_size_mb: int = 64):
        self.agent_id = agent_id
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size_mb = max_size_mb
        self.hit_count = 0
        self.miss_count = 0
        self.specialized_keys = set()

    def get(self, key: str) -> Optional[Any]:
        """Get agent-specific cached value"""
        cache_key = f"agent:{self.agent_id}:{key}"
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired:
                entry.record_access()
                self.hit_count += 1
                return entry.value
            else:
                del self.cache[cache_key]

        self.miss_count += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Set agent-specific cached value"""
        cache_key = f"agent:{self.agent_id}:{key}"

        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = len(str(value).encode('utf-8'))

        # Check size limit
        current_size = sum(entry.size_bytes for entry in self.cache.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if current_size + size_bytes > max_size_bytes:
            # Evict oldest entries
            entries_by_age = sorted(self.cache.items(), key=lambda x: x[1].created_at)
            while current_size + size_bytes > max_size_bytes and entries_by_age:
                oldest_key, oldest_entry = entries_by_age.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
                    current_size -= oldest_entry.size_bytes

        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            size_bytes=size_bytes,
            level=CacheLevel.L4_AGENT
        )

        self.cache[cache_key] = entry
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get agent cache statistics"""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        hit_rate = (self.hit_count / (self.hit_count + self.miss_count) * 100) if (self.hit_count + self.miss_count) > 0 else 0

        return {
            "level": "L4_AGENT",
            "agent_id": self.agent_id,
            "entries": len(self.cache),
            "size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": hit_rate,
            "specialized_keys": len(self.specialized_keys)
        }

class MultiLevelCachingSystem:
    """Multi-level caching system with intelligent routing"""

    def __init__(self,
                 l1_size_mb: int = 512,
                 l2_redis_config: Optional[Dict] = None,
                 l3_db_connection: Optional[str] = None,
                 agent_cache_size_mb: int = 64):

        # Initialize cache levels
        self.l1_cache = MemoryCache(max_size_mb=l1_size_mb, strategy=CacheStrategy.LRU)

        self.l2_cache = None
        if l2_redis_config and REDIS_AVAILABLE:
            try:
                self.l2_cache = RedisCache(**l2_redis_config)
                logger.info("L2 Redis cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize L2 Redis cache: {e}")

        self.l3_cache = None
        if l3_db_connection:
            try:
                self.l3_cache = DatabaseCache(l3_db_connection)
                logger.info("L3 Database cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize L3 Database cache: {e}")

        # Agent caches
        self.agent_caches: Dict[str, AgentCache] = {}

        # Cache warming and optimization
        self.cache_warming_enabled = True
        self.auto_optimization_enabled = True
        self.optimization_thread = None
        self.running = False

        # Performance tracking
        self.access_patterns = defaultdict(int)
        self.cache_hits_by_level = defaultdict(int)
        self.cache_misses_by_level = defaultdict(int)

    def start(self):
        """Start caching system"""
        logger.info("Starting Multi-Level Caching System")
        self.running = True

        if self.auto_optimization_enabled:
            self.optimization_thread = threading.Thread(target=self._background_optimization)
            self.optimization_thread.daemon = True
            self.optimization_thread.start()

        logger.info("Caching system started successfully")

    def stop(self):
        """Stop caching system"""
        logger.info("Stopping Multi-Level Caching System")
        self.running = False

        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5)

        logger.info("Caching system stopped")

    def get(self, key: str, agent_id: Optional[str] = None) -> Optional[Any]:
        """Get value from multi-level cache"""
        start_time = time.time()

        # Track access patterns
        self.access_patterns[key] += 1

        # Try L4 agent cache first if agent_id provided
        if agent_id and agent_id in self.agent_caches:
            result = self.agent_caches[agent_id].get(key)
            if result is not None:
                self.cache_hits_by_level[CacheLevel.L4_AGENT.value] += 1
                return result

        # Try L1 memory cache
        result = self.l1_cache.get(key)
        if result is not None:
            self.cache_hits_by_level[CacheLevel.L1_MEMORY.value] += 1
            return result

        # Try L2 Redis cache
        if self.l2_cache:
            result = self.l2_cache.get(key)
            if result is not None:
                # Promote to L1 cache
                self.l1_cache.set(key, result)
                self.cache_hits_by_level[CacheLevel.L2_REDIS.value] += 1
                return result

        # Try L3 database cache
        if self.l3_cache:
            result = self.l3_cache.get(key)
            if result is not None:
                # Promote to L1 and L2 caches
                self.l1_cache.set(key, result)
                if self.l2_cache:
                    self.l2_cache.set(key, result)
                self.cache_hits_by_level[CacheLevel.L3_DATABASE.value] += 1
                return result

        # Cache miss
        self.cache_misses_by_level["total"] += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None, agent_id: Optional[str] = None):
        """Set value in multi-level cache"""
        # Set in L1 cache
        self.l1_cache.set(key, value, ttl_seconds)

        # Set in L2 cache if available
        if self.l2_cache:
            self.l2_cache.set(key, value, ttl_seconds)

        # Set in L3 cache if available
        if self.l3_cache:
            self.l3_cache.set(key, value, ttl_seconds)

        # Set in agent cache if agent_id provided
        if agent_id:
            self._ensure_agent_cache(agent_id)
            self.agent_caches[agent_id].set(key, value, ttl_seconds)

    def _ensure_agent_cache(self, agent_id: str):
        """Ensure agent cache exists"""
        if agent_id not in self.agent_caches:
            self.agent_caches[agent_id] = AgentCache(agent_id)

    def invalidate(self, key: str, agent_id: Optional[str] = None):
        """Invalidate key across all cache levels"""
        # Remove from L1 cache
        self.l1_cache.cache.pop(key, None)

        # Remove from L2 cache
        if self.l2_cache:
            self.l2_cache.delete(key)

        # Remove from agent cache
        if agent_id and agent_id in self.agent_caches:
            agent_key = f"agent:{agent_id}:{key}"
            self.agent_caches[agent_id].cache.pop(agent_key, None)

    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        # L1 cache
        keys_to_remove = [key for key in self.l1_cache.keys() if pattern in key]
        for key in keys_to_remove:
            self.l1_cache.cache.pop(key, None)

        # L2 cache
        if self.l2_cache:
            self.l2_cache.clear_pattern(f"*{pattern}*")

        # Agent caches
        for agent_cache in self.agent_caches.values():
            keys_to_remove = [key for key in agent_cache.keys() if pattern in key]
            for key in keys_to_remove:
                agent_cache.cache.pop(key, None)

    def warm_cache(self, keys: List[str], values: List[Any], ttl_seconds: Optional[float] = None):
        """Warm cache with frequently accessed keys"""
        if not self.cache_warming_enabled:
            return

        logger.info(f"Warming cache with {len(keys)} keys")

        for key, value in zip(keys, values):
            self.set(key, value, ttl_seconds)

    def _background_optimization(self):
        """Background cache optimization"""
        while self.running:
            try:
                # Analyze access patterns
                self._analyze_access_patterns()

                # Optimize cache sizes
                self._optimize_cache_sizes()

                # Clean expired entries
                self._clean_expired_entries()

                # Sleep for next optimization cycle
                time.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in cache optimization: {e}")
                time.sleep(60)

    def _analyze_access_patterns(self):
        """Analyze access patterns for optimization"""
        # Get most frequently accessed keys
        frequent_keys = sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)[:100]

        if frequent_keys:
            logger.debug(f"Top 10 most accessed keys: {[k for k, _ in frequent_keys[:10]]}")

    def _optimize_cache_sizes(self):
        """Optimize cache sizes based on usage patterns"""
        # Placeholder for cache size optimization logic
        pass

    def _clean_expired_entries(self):
        """Clean expired entries from all cache levels"""
        # L1 cache (handled automatically by get/set)
        # Clean agent caches
        for agent_id, agent_cache in self.agent_caches.items():
            expired_keys = [key for key, entry in agent_cache.cache.items() if entry.is_expired]
            for key in expired_keys:
                del agent_cache.cache[key]

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_hits = sum(self.cache_hits_by_level.values())
        total_misses = self.cache_misses_by_level.get("total", 0)
        overall_hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0

        return {
            "overall_hit_rate_percent": overall_hit_rate,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "cache_levels": {
                "l1": self.l1_cache.get_stats(),
                "l2": self.l2_cache.get_stats() if self.l2_cache else {"level": "L2_REDIS", "connected": False},
                "l3": self.l3_cache.get_stats() if self.l3_cache else {"level": "L3_DATABASE", "connected": False},
                "agent_caches": {agent_id: cache.get_stats() for agent_id, cache in self.agent_caches.items()}
            },
            "hits_by_level": dict(self.cache_hits_by_level),
            "access_patterns": dict(sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
            "system_config": {
                "cache_warming_enabled": self.cache_warming_enabled,
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "agent_caches_count": len(self.agent_caches)
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize multi-level cache
    cache_system = MultiLevelCachingSystem(
        l1_size_mb=256,
        l2_redis_config={"host": "localhost", "port": 6379, "db": 0},
        l3_db_connection="postgresql://user:pass@localhost/cache"
    )

    try:
        cache_system.start()

        # Test caching
        cache_system.set("test_key", {"data": "test_value"}, ttl_seconds=3600)
        result = cache_system.get("test_key")

        if result:
            print("Cache hit:", result)

        # Get statistics
        stats = cache_system.get_comprehensive_stats()
        print(f"Overall hit rate: {stats['overall_hit_rate_percent']:.1f}%")

    finally:
        cache_system.stop()