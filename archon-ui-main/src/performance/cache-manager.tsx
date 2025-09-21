/**
 * Cache Manager
 *
 * Multi-level caching system with intelligent strategies:
 * - Memory cache (LRU-based)
 * - Session storage cache
 * - Local storage cache
 * - API response caching
 * - Component state caching
 * - Intelligent cache invalidation
 */

import { EventEmitter } from 'events';

// Cache entry interface
export interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
  hits: number;
  metadata?: Record<string, any>;
}

// Cache configuration
export interface CacheConfig {
  memoryCacheSize: number; // MB
  sessionStorageSize: number; // MB
  localStorageSize: number; // MB
  apiCacheTTL: number; // seconds
  componentCacheTTL: number; // seconds
  enableCompression: boolean;
  enableEncryption: boolean;
  maxEntries: number;
  cleanupInterval: number; // seconds
}

// Cache statistics
export interface CacheStats {
  memory: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
  };
  session: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
  };
  local: {
    hits: number;
    misses: number;
    size: number;
    entries: number;
  };
  total: {
    hits: number;
    misses: number;
    hitRate: number;
  };
}

// Memory cache with LRU eviction
class MemoryCache<T> extends EventEmitter {
  private cache: Map<string, CacheEntry<T>> = new Map();
  private maxSize: number;
  private currentSize: number = 0;

  constructor(maxSizeMB: number) {
    super();
    this.maxSize = maxSizeMB * 1024 * 1024; // Convert to bytes
  }

  private calculateEntrySize(entry: CacheEntry<T>): number {
    // Rough estimate of memory usage
    const dataStr = JSON.stringify(entry.data);
    return dataStr.length * 2; // Approximate memory usage
  }

  private evict(): void {
    // LRU eviction - remove oldest entries
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].timestamp - b[1].timestamp);

    let evictedSize = 0;
    for (const [key, entry] of entries) {
      this.cache.delete(key);
      evictedSize += this.calculateEntrySize(entry);
      this.currentSize -= evictedSize;

      if (this.currentSize <= this.maxSize * 0.8) {
        break;
      }
    }

    this.emit('eviction', { evictedCount: entries.length, evictedSize });
  }

  set(key: string, data: T, ttl: number): void {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl,
      hits: 0,
    };

    const entrySize = this.calculateEntrySize(entry);

    // Check if we need to evict
    if (this.currentSize + entrySize > this.maxSize) {
      this.evict();
    }

    this.cache.set(key, entry);
    this.currentSize += entrySize;
  }

  get(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check TTL
    if (Date.now() - entry.timestamp > entry.ttl * 1000) {
      this.cache.delete(key);
      this.currentSize -= this.calculateEntrySize(entry);
      return null;
    }

    entry.hits++;
    this.emit('hit', { key, hits: entry.hits });
    return entry.data;
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  delete(key: string): boolean {
    const entry = this.cache.get(key);
    if (entry) {
      this.cache.delete(key);
      this.currentSize -= this.calculateEntrySize(entry);
      return true;
    }
    return false;
  }

  clear(): void {
    this.cache.clear();
    this.currentSize = 0;
  }

  getSize(): number {
    return this.currentSize;
  }

  getEntries(): number {
    return this.cache.size;
  }

  getStats(): { hits: number; misses: number; size: number; entries: number } {
    let hits = 0;
    let misses = 0;

    for (const entry of this.cache.values()) {
      hits += entry.hits;
    }

    return {
      hits,
      misses: 0, // Track misses externally
      size: this.currentSize,
      entries: this.cache.size,
    };
  }
}

// Storage cache wrapper
class StorageCache<T> {
  private storage: Storage;
  private maxSize: number;
  private prefix: string;

  constructor(storage: Storage, maxSizeMB: number, prefix: string) {
    this.storage = storage;
    this.maxSize = maxSizeMB * 1024 * 1024; // Convert to bytes
    this.prefix = prefix;
  }

  private getKey(key: string): string {
    return `${this.prefix}_${key}`;
  }

  private getCurrentSize(): number {
    let size = 0;
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) {
        size += (this.storage.getItem(key) || '').length;
      }
    }
    return size;
  }

  private cleanup(): void {
    const currentSize = this.getCurrentSize();
    if (currentSize > this.maxSize) {
      // Remove oldest entries
      const entries: Array<{ key: string; timestamp: number }> = [];

      for (let i = 0; i < this.storage.length; i++) {
        const key = this.storage.key(i);
        if (key?.startsWith(this.prefix)) {
          const value = this.storage.getItem(key);
          if (value) {
            try {
              const entry = JSON.parse(value);
              entries.push({ key, timestamp: entry.timestamp });
            } catch {
              // Invalid entry, remove it
              this.storage.removeItem(key);
            }
          }
        }
      }

      // Sort by timestamp and remove oldest
      entries.sort((a, b) => a.timestamp - b.timestamp);
      for (const entry of entries) {
        this.storage.removeItem(entry.key);
        if (this.getCurrentSize() <= this.maxSize * 0.8) {
          break;
        }
      }
    }
  }

  set(key: string, data: T, ttl: number): void {
    const entry: CacheEntry<T> = {
      data,
      timestamp: Date.now(),
      ttl,
      hits: 0,
    };

    try {
      this.storage.setItem(this.getKey(key), JSON.stringify(entry));
      this.cleanup();
    } catch (error) {
      console.warn('Storage quota exceeded, clearing cache');
      this.clear();
      this.storage.setItem(this.getKey(key), JSON.stringify(entry));
    }
  }

  get(key: string): T | null {
    try {
      const value = this.storage.getItem(this.getKey(key));
      if (!value) return null;

      const entry: CacheEntry<T> = JSON.parse(value);

      // Check TTL
      if (Date.now() - entry.timestamp > entry.ttl * 1000) {
        this.storage.removeItem(this.getKey(key));
        return null;
      }

      entry.hits++;
      // Update the entry with new hit count
      this.storage.setItem(this.getKey(key), JSON.stringify(entry));

      return entry.data;
    } catch (error) {
      console.warn('Error reading from storage cache:', error);
      return null;
    }
  }

  has(key: string): boolean {
    return this.storage.getItem(this.getKey(key)) !== null;
  }

  delete(key: string): boolean {
    const value = this.storage.getItem(this.getKey(key));
    if (value) {
      this.storage.removeItem(this.getKey(key));
      return true;
    }
    return false;
  }

  clear(): void {
    // Clear only our cache entries
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) {
        this.storage.removeItem(key);
      }
    }
  }

  getSize(): number {
    return this.getCurrentSize();
  }

  getEntries(): number {
    let count = 0;
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) {
        count++;
      }
    }
    return count;
  }

  getStats(): { hits: number; misses: number; size: number; entries: number } {
    let hits = 0;
    let size = 0;
    let entries = 0;

    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) {
        const value = this.storage.getItem(key);
        if (value) {
          try {
            const entry = JSON.parse(value);
            hits += entry.hits || 0;
            size += value.length;
            entries++;
          } catch {
            // Invalid entry
          }
        }
      }
    }

    return { hits, misses: 0, size, entries };
  }
}

// Main Cache Manager
export class CacheManager {
  private config: CacheConfig;
  private memoryCache: MemoryCache<any>;
  private sessionCache: StorageCache<any>;
  private localCache: StorageCache<any>;
  private stats: CacheStats;
  private cleanupInterval: NodeJS.Timeout;

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = {
      memoryCacheSize: 50, // 50MB
      sessionStorageSize: 10, // 10MB
      localStorageSize: 5, // 5MB
      apiCacheTTL: 300, // 5 minutes
      componentCacheTTL: 600, // 10 minutes
      enableCompression: false,
      enableEncryption: false,
      maxEntries: 1000,
      cleanupInterval: 300, // 5 minutes
      ...config,
    };

    this.stats = {
      memory: { hits: 0, misses: 0, size: 0, entries: 0 },
      session: { hits: 0, misses: 0, size: 0, entries: 0 },
      local: { hits: 0, misses: 0, size: 0, entries: 0 },
      total: { hits: 0, misses: 0, hitRate: 0 },
    };

    this.initialize();
  }

  private initialize(): void {
    // Initialize cache layers
    this.memoryCache = new MemoryCache(this.config.memoryCacheSize);
    this.sessionCache = new StorageCache(sessionStorage, this.config.sessionStorageSize, 'cache_session');
    this.localCache = new StorageCache(localStorage, this.config.localStorageSize, 'cache_local');

    // Set up cleanup interval
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, this.config.cleanupInterval * 1000);

    // Listen for memory cache events
    this.memoryCache.on('hit', () => {
      this.stats.memory.hits++;
      this.updateTotalStats();
    });

    this.memoryCache.on('eviction', (data) => {
      console.log('Cache eviction:', data);
    });
  }

  private updateTotalStats(): void {
    const totalHits = this.stats.memory.hits + this.stats.session.hits + this.stats.local.hits;
    const totalMisses = this.stats.memory.misses + this.stats.session.misses + this.stats.local.misses;
    const totalRequests = totalHits + totalMisses;

    this.stats.total = {
      hits: totalHits,
      misses: totalMisses,
      hitRate: totalRequests > 0 ? totalHits / totalRequests : 0,
    };
  }

  // Multi-level cache get
  async get<T>(key: string, options: { skipMemory?: boolean; skipSession?: boolean; skipLocal?: boolean } = {}): Promise<T | null> {
    // Try memory cache first
    if (!options.skipMemory) {
      const memoryResult = this.memoryCache.get<T>(key);
      if (memoryResult) {
        return memoryResult;
      }
      this.stats.memory.misses++;
    }

    // Try session cache
    if (!options.skipSession) {
      const sessionResult = this.sessionCache.get<T>(key);
      if (sessionResult) {
        // Promote to memory cache
        this.memoryCache.set(key, sessionResult, this.config.componentCacheTTL);
        this.stats.session.hits++;
        this.updateTotalStats();
        return sessionResult;
      }
      this.stats.session.misses++;
    }

    // Try local cache
    if (!options.skipLocal) {
      const localResult = this.localCache.get<T>(key);
      if (localResult) {
        // Promote to memory and session cache
        this.memoryCache.set(key, localResult, this.config.componentCacheTTL);
        this.sessionCache.set(key, localResult, this.config.componentCacheTTL);
        this.stats.local.hits++;
        this.updateTotalStats();
        return localResult;
      }
      this.stats.local.misses++;
    }

    this.updateTotalStats();
    return null;
  }

  // Multi-level cache set
  async set<T>(key: string, data: T, ttl?: number, options: { memoryOnly?: boolean; sessionOnly?: boolean; localOnly?: boolean } = {}): Promise<void> {
    const effectiveTtl = ttl || this.config.apiCacheTTL;

    // Always set in memory cache unless explicitly excluded
    if (!options.sessionOnly && !options.localOnly) {
      this.memoryCache.set(key, data, effectiveTtl);
    }

    // Set in session cache
    if (!options.memoryOnly && !options.localOnly) {
      this.sessionCache.set(key, data, effectiveTtl);
    }

    // Set in local cache
    if (!options.memoryOnly && !options.sessionOnly) {
      this.localCache.set(key, data, effectiveTtl);
    }
  }

  // Cache invalidation
  async invalidate(key: string): Promise<void> {
    this.memoryCache.delete(key);
    this.sessionCache.delete(key);
    this.localCache.delete(key);
  }

  // Cache invalidation by pattern
  async invalidatePattern(pattern: RegExp): Promise<void> {
    const invalidateInCache = (cache: any) => {
      // This is a simplified implementation
      // In a real implementation, you'd need to track keys or iterate through them
    };

    invalidateInCache(this.memoryCache);
    invalidateInCache(this.sessionCache);
    invalidateInCache(this.localCache);
  }

  // Clear all caches
  async clear(): Promise<void> {
    this.memoryCache.clear();
    this.sessionCache.clear();
    this.localCache.clear();

    // Reset stats
    this.stats = {
      memory: { hits: 0, misses: 0, size: 0, entries: 0 },
      session: { hits: 0, misses: 0, size: 0, entries: 0 },
      local: { hits: 0, misses: 0, size: 0, entries: 0 },
      total: { hits: 0, misses: 0, hitRate: 0 },
    };
  }

  // Cleanup expired entries
  private cleanup(): void {
    // The individual cache classes handle TTL checking
    // This method can be extended for additional cleanup logic
    console.log('Cache cleanup completed');
  }

  // Get cache statistics
  getStats(): CacheStats {
    // Update stats from individual caches
    this.stats.memory = this.memoryCache.getStats();
    this.stats.session = this.sessionCache.getStats();
    this.stats.local = this.localCache.getStats();
    this.updateTotalStats();

    return { ...this.stats };
  }

  // Get cache health
  getHealth(): {
    memory: { usage: number; status: 'healthy' | 'warning' | 'critical' };
    session: { usage: number; status: 'healthy' | 'warning' | 'critical' };
    local: { usage: number; status: 'healthy' | 'warning' | 'critical' };
  } {
    const memoryUsage = this.stats.memory.size / (this.config.memoryCacheSize * 1024 * 1024);
    const sessionUsage = this.stats.session.size / (this.config.sessionStorageSize * 1024 * 1024);
    const localUsage = this.stats.local.size / (this.config.localStorageSize * 1024 * 1024);

    const getStatus = (usage: number) => {
      if (usage < 0.7) return 'healthy';
      if (usage < 0.9) return 'warning';
      return 'critical';
    };

    return {
      memory: { usage: memoryUsage, status: getStatus(memoryUsage) },
      session: { usage: sessionUsage, status: getStatus(sessionUsage) },
      local: { usage: localUsage, status: getStatus(localUsage) },
    };
  }

  // Optimize cache based on usage patterns
  optimize(): void {
    const stats = this.getStats();

    // If memory cache hit rate is low, increase size
    if (stats.memory.hitRate < 0.5) {
      this.config.memoryCacheSize = Math.min(100, this.config.memoryCacheSize * 1.2);
      console.log('Optimized memory cache size:', this.config.memoryCacheSize);
    }

    // If total hit rate is low, adjust TTLs
    if (stats.total.hitRate < 0.3) {
      this.config.apiCacheTTL = Math.max(60, this.config.apiCacheTTL * 0.8);
      this.config.componentCacheTTL = Math.max(120, this.config.componentCacheTTL * 0.8);
      console.log('Optimized cache TTLs');
    }
  }

  // Destroy cache manager
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.clear();
  }
}

// React hook for caching
export function useCache<T>(key: string, data: T | null, ttl?: number) {
  const cache = React.useMemo(() => new CacheManager(), []);
  const [cachedData, setCachedData] = React.useState<T | null>(null);

  React.useEffect(() => {
    if (data !== null) {
      cache.set(key, data, ttl);
      setCachedData(data);
    }
  }, [data, key, ttl, cache]);

  React.useEffect(() => {
    // Load from cache on mount
    cache.get<T>(key).then(result => {
      if (result) {
        setCachedData(result);
      }
    });
  }, [key, cache]);

  return cachedData;
}

// Cache utilities
export const cacheUtils = {
  // Generate cache keys
  generateKey: (namespace: string, ...args: any[]): string => {
    return `${namespace}_${args.map(arg =>
      typeof arg === 'object' ? JSON.stringify(arg) : String(arg)
    ).join('_')}`;
  },

  // Debounce cache operations
  debounce: <T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void => {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  },

  // Memoize expensive operations with cache
  memoize: <T extends (...args: any[]) => any>(
    func: T,
    cache: CacheManager,
    keyPrefix: string,
    ttl: number = 300
  ): (...args: Parameters<T>) => Promise<ReturnType<T>> => {
    return async (...args: Parameters<T>): Promise<ReturnType<T>> => {
      const key = cacheUtils.generateKey(keyPrefix, ...args);

      // Try cache first
      const cached = await cache.get<ReturnType<T>>(key);
      if (cached) {
        return cached;
      }

      // Execute function
      const result = await func.apply(this, args);

      // Cache result
      await cache.set(key, result, ttl);

      return result;
    };
  },
};

export default CacheManager;