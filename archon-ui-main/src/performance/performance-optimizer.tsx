/**
 * Frontend Performance Optimization System
 *
 * Enterprise-grade UI performance optimizations including:
 * - Virtualization and lazy loading for large datasets
 * - Multi-level caching strategies
 * - Resource optimization and bundle analysis
 * - Performance monitoring and metrics
 * - Image and content optimization
 *
 * Target: <1.5s page load, <200ms API responses, 60fps interactions
 */

import { PerformanceMonitor } from './performance-monitor';
import { VirtualizationManager } from './virtualization-manager';
import { CacheManager } from './cache-manager';
import { ResourceOptimizer } from './resource-optimizer';
import { ImageOptimizer } from './image-optimizer';
import { BundleAnalyzer } from './bundle-analyzer';

// Performance configuration
export interface PerformanceConfig {
  // Virtualization settings
  virtualization: {
    enabled: boolean;
    itemHeight: number;
    containerHeight: number;
    overscanCount: number;
    enableDynamicSizing: boolean;
  };

  // Caching settings
  caching: {
    memoryCacheSize: number; // MB
    sessionStorageSize: number; // MB
    localStorageSize: number; // MB
    apiCacheTTL: number; // seconds
    componentCacheTTL: number; // seconds
  };

  // Resource optimization
  resources: {
    enableLazyLoading: boolean;
    enablePreloading: boolean;
    enablePrefetching: boolean;
    maxConcurrentRequests: number;
    enableCompression: boolean;
  };

  // Monitoring
  monitoring: {
    enabled: boolean;
    sampleRate: number; // 0-1
    reportErrors: boolean;
    consoleMetrics: boolean;
    performanceThresholds: {
      firstContentfulPaint: number; // ms
      largestContentfulPaint: number; // ms
      cumulativeLayoutShift: number; // score
      firstInputDelay: number; // ms
      timeToInteractive: number; // ms
    };
  };
}

// Performance metrics
export interface PerformanceMetrics {
  // Loading metrics
  loading: {
    firstContentfulPaint: number;
    largestContentfulPaint: number;
    timeToInteractive: number;
    domContentLoaded: number;
    firstByte: number;
  };

  // Runtime metrics
  runtime: {
    frameRate: number;
    memoryUsage: number;
    cpuUsage: number;
    responseTime: number;
  };

  // Resource metrics
  resources: {
    bundleSize: number;
    requestCount: number;
    cacheHitRate: number;
    lazyLoadCount: number;
  };

  // User experience
  ux: {
    cumulativeLayoutShift: number;
    firstInputDelay: number;
    interactionTime: number;
  };
}

// Performance optimization strategies
export class PerformanceOptimizer {
  private config: PerformanceConfig;
  private monitor: PerformanceMonitor;
  private virtualization: VirtualizationManager;
  private cache: CacheManager;
  private resources: ResourceOptimizer;
  private images: ImageOptimizer;
  private bundle: BundleAnalyzer;

  constructor(config: Partial<PerformanceConfig> = {}) {
    // Default configuration
    this.config = {
      virtualization: {
        enabled: true,
        itemHeight: 50,
        containerHeight: 600,
        overscanCount: 5,
        enableDynamicSizing: true,
        ...config.virtualization,
      },
      caching: {
        memoryCacheSize: 50,
        sessionStorageSize: 10,
        localStorageSize: 5,
        apiCacheTTL: 300,
        componentCacheTTL: 600,
        ...config.caching,
      },
      resources: {
        enableLazyLoading: true,
        enablePreloading: true,
        enablePrefetching: true,
        maxConcurrentRequests: 6,
        enableCompression: true,
        ...config.resources,
      },
      monitoring: {
        enabled: true,
        sampleRate: 0.1,
        reportErrors: true,
        consoleMetrics: true,
        performanceThresholds: {
          firstContentfulPaint: 1500,
          largestContentfulPaint: 2500,
          cumulativeLayoutShift: 0.1,
          firstInputDelay: 100,
          timeToInteractive: 3000,
          ...config.monitoring?.performanceThresholds,
        },
        ...config.monitoring,
      },
    };

    // Initialize components
    this.monitor = new PerformanceMonitor(this.config.monitoring);
    this.virtualization = new VirtualizationManager(this.config.virtualization);
    this.cache = new CacheManager(this.config.caching);
    this.resources = new ResourceOptimizer(this.config.resources);
    this.images = new ImageOptimizer();
    this.bundle = new BundleAnalyzer();

    this.initialize();
  }

  private initialize() {
    // Initialize performance monitoring
    if (this.config.monitoring.enabled) {
      this.monitor.startMonitoring();
    }

    // Initialize caching
    this.cache.initialize();

    // Initialize resource optimization
    this.resources.initialize();

    // Set up performance observers
    this.setupPerformanceObservers();

    // Set up global error handling
    this.setupErrorHandling();
  }

  private setupPerformanceObservers() {
    // PerformanceObserver for Core Web Vitals
    if ('PerformanceObserver' in window) {
      try {
        // First Contentful Paint
        const fcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const fcp = entries[entries.length - 1];
          this.monitor.updateMetric('firstContentfulPaint', fcp.startTime);
        });
        fcpObserver.observe({ entryTypes: ['paint'] });

        // Largest Contentful Paint
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lcp = entries[entries.length - 1];
          this.monitor.updateMetric('largestContentfulPaint', lcp.startTime);
        });
        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

        // Layout Shift
        const clsObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            this.monitor.updateMetric('cumulativeLayoutShift', entry.value);
          }
        });
        clsObserver.observe({ entryTypes: ['layout-shift'] });

        // First Input Delay
        const fidObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const fid = entries[0];
          this.monitor.updateMetric('firstInputDelay', fid.processingStart - fid.startTime);
        });
        fidObserver.observe({ entryTypes: ['first-input'] });

      } catch (e) {
        console.warn('PerformanceObserver not fully supported');
      }
    }
  }

  private setupErrorHandling() {
    // Global error handler
    window.addEventListener('error', (event) => {
      this.monitor.recordError({
        message: event.message,
        filename: event.filename,
        line: event.lineno,
        column: event.colno,
        error: event.error,
        timestamp: Date.now(),
      });
    });

    // Unhandled promise rejection handler
    window.addEventListener('unhandledrejection', (event) => {
      this.monitor.recordError({
        message: 'Unhandled Promise Rejection',
        reason: event.reason,
        timestamp: Date.now(),
      });
    });
  }

  // Virtualization methods
  createVirtualList<T>(items: T[], renderItem: (item: T, index: number) => React.ReactNode) {
    if (!this.config.virtualization.enabled) {
      return items.map((item, index) => renderItem(item, index));
    }

    return this.virtualization.createVirtualList(items, renderItem);
  }

  createVirtualGrid<T>(items: T[], renderItem: (item: T, index: number) => React.ReactNode) {
    if (!this.config.virtualization.enabled) {
      return items.map((item, index) => renderItem(item, index));
    }

    return this.virtualization.createVirtualGrid(items, renderItem);
  }

  // Caching methods
  async cacheData<T>(key: string, data: T, ttl?: number): Promise<void> {
    return this.cache.set(key, data, ttl);
  }

  async getCachedData<T>(key: string): Promise<T | null> {
    return this.cache.get<T>(key);
  }

  async invalidateCache(key: string): Promise<void> {
    return this.cache.invalidate(key);
  }

  async clearCache(): Promise<void> {
    return this.cache.clear();
  }

  // Resource optimization methods
  async preloadResources(urls: string[]): Promise<void> {
    return this.resources.preload(urls);
  }

  async prefetchResources(urls: string[]): Promise<void> {
    return this.resources.prefetch(urls);
  }

  optimizeImages(): void {
    this.images.optimize();
  }

  // Performance monitoring methods
  startPerformanceMeasurement(label: string): void {
    this.monitor.startMeasurement(label);
  }

  endPerformanceMeasurement(label: string): number {
    return this.monitor.endMeasurement(label);
  }

  getCurrentMetrics(): PerformanceMetrics {
    return this.monitor.getMetrics();
  }

  // Bundle analysis
  analyzeBundle(): Promise<any> {
    return this.bundle.analyze();
  }

  getBundleRecommendations(): string[] {
    return this.bundle.getRecommendations();
  }

  // Performance optimization recommendations
  getOptimizationRecommendations(): string[] {
    const metrics = this.getCurrentMetrics();
    const recommendations: string[] = [];

    // Loading performance
    if (metrics.loading.firstContentfulPaint > this.config.monitoring.performanceThresholds.firstContentfulPaint) {
      recommendations.push('Optimize critical rendering path for faster first contentful paint');
    }

    if (metrics.loading.largestContentfulPaint > this.config.monitoring.performanceThresholds.largestContentfulPaint) {
      recommendations.push('Optimize images and largest contentful paint');
    }

    // Runtime performance
    if (metrics.runtime.frameRate < 55) {
      recommendations.push('Optimize heavy computations for better frame rate');
    }

    if (metrics.runtime.memoryUsage > 100) {
      recommendations.push('Check for memory leaks and optimize memory usage');
    }

    // UX metrics
    if (metrics.ux.cumulativeLayoutShift > this.config.monitoring.performanceThresholds.cumulativeLayoutShift) {
      recommendations.push('Reduce layout shifts by reserving space for dynamic content');
    }

    if (metrics.ux.firstInputDelay > this.config.monitoring.performanceThresholds.firstInputDelay) {
      recommendations.push('Reduce JavaScript execution time for better responsiveness');
    }

    // Resource usage
    if (metrics.resources.bundleSize > 500000) {
      recommendations.push('Consider code splitting to reduce initial bundle size');
    }

    if (metrics.resources.cacheHitRate < 0.7) {
      recommendations.push('Implement better caching strategies');
    }

    return recommendations;
  }

  // Auto-optimization
  async autoOptimize(): Promise<void> {
    const recommendations = this.getOptimizationRecommendations();

    for (const recommendation of recommendations) {
      try {
        await this.applyOptimization(recommendation);
      } catch (error) {
        console.warn(`Failed to apply optimization: ${recommendation}`, error);
      }
    }
  }

  private async applyOptimization(recommendation: string): Promise<void> {
    // Apply specific optimizations based on recommendations
    switch (recommendation) {
      case 'Optimize critical rendering path for faster first contentful paint':
        await this.resources.optimizeCriticalPath();
        break;
      case 'Optimize images and largest contentful paint':
        this.images.optimize();
        break;
      case 'Consider code splitting to reduce initial bundle size':
        await this.bundle.optimize();
        break;
      case 'Implement better caching strategies':
        await this.cache.optimize();
        break;
      default:
        console.warn(`Unknown optimization: ${recommendation}`);
    }
  }

  // Cleanup
  destroy(): void {
    this.monitor.stopMonitoring();
    this.cache.destroy();
    this.resources.destroy();
  }
}

// Export singleton instance
export const performanceOptimizer = new PerformanceOptimizer();

// React hook for performance optimization
export function usePerformanceOptimization(config?: Partial<PerformanceConfig>) {
  const optimizer = React.useMemo(() => new PerformanceOptimizer(config), [config]);

  React.useEffect(() => {
    return () => {
      optimizer.destroy();
    };
  }, [optimizer]);

  return optimizer;
}

// Performance utility functions
export const performanceUtils = {
  // Debounce for performance
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

  // Throttle for performance
  throttle: <T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): (...args: Parameters<T>) => void => {
    let inThrottle: boolean;
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  // Memoize expensive computations
  memoize: <T extends (...args: any[]) => any>(
    func: T
  ): (...args: Parameters<T>) => ReturnType<T> => {
    const cache = new Map();
    return (...args: Parameters<T>): ReturnType<T> => {
      const key = JSON.stringify(args);
      if (cache.has(key)) {
        return cache.get(key);
      }
      const result = func.apply(this, args);
      cache.set(key, result);
      return result;
    };
  },

  // Request animation frame wrapper
  raf: (callback: FrameRequestCallback) => {
    return requestAnimationFrame(callback);
  },

  // Idle callback wrapper
  idle: (callback: IdleRequestCallback, options?: IdleRequestOptions) => {
    return requestIdleCallback(callback, options);
  },
};

export default PerformanceOptimizer;