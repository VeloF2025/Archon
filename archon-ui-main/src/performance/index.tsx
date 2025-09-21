/**
 * Performance Optimization System
 *
 * Enterprise-grade performance optimization suite for the Archon UI
 *
 * Features:
 * - Multi-level caching (Memory, Session, Local)
 * - Advanced virtualization for large datasets
 * - Lazy loading with Intersection Observer
 * - Image optimization with modern formats
 * - Resource optimization and preloading
 * - Real-time performance monitoring
 * - Bundle analysis and optimization
 * - Core Web Vitals tracking
 */

// Core components
export { default as PerformanceOptimizer, usePerformanceOptimization, performanceUtils, performanceOptimizer } from './performance-optimizer';
export { default as VirtualizationManager, VirtualList, VirtualGrid } from './virtualization-manager';
export { default as CacheManager, useCache, cacheUtils } from './cache-manager';
export { default as LazyImage, LazyComponent, LazyList, LazyGrid, useIntersectionObserver, lazyUtils, preloadManager } from './lazy-loading';
export { default as PerformanceMonitor, usePerformanceMonitor } from './performance-monitor';
export { default as ResourceOptimizer, useResourceOptimizer, resourceUtils } from './resource-optimizer';
export { default as ImageOptimizer, OptimizedImage, OptimizedImageGallery, useImageOptimizer, imageUtils } from './image-optimizer';
export { default as BundleAnalyzer, useBundleAnalyzer, BundleAnalysisVisualization } from './bundle-analyzer';

// Types and interfaces
export type {
  PerformanceConfig,
  PerformanceMetrics,
  VirtualizationConfig,
  VirtualItem,
  CacheConfig,
  CacheEntry,
  CacheStats,
  LazyConfig,
  LoadingState,
  MonitorConfig,
  ResourceConfig,
  ResourceEntry,
  ResourceMetrics,
  ImageOptimizationConfig,
  ImageSource,
  OptimizedImageProps,
  BundleAnalysis,
  BundleChunk,
  BundleModule,
  BundleAsset,
  DuplicateModule,
  BundleMetrics,
  BundleAnalyzerConfig,
} from './performance-optimizer';

// Utility functions
export * from './performance-optimizer';

// React hooks
export { usePerformanceOptimization, useCache, usePerformanceMonitor, useResourceOptimizer, useImageOptimizer, useBundleAnalyzer, useIntersectionObserver };

// Default exports
export { performanceOptimizer as default };

// Version
export const PERFORMANCE_VERSION = '1.0.0';

// Performance optimization helper functions
export const performanceHelpers = {
  // Debounce function for performance
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

  // Throttle function for performance
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

  // Memoize expensive operations
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
    if ('requestIdleCallback' in window) {
      return requestIdleCallback(callback, options);
    } else {
      // Fallback for browsers without idle callback
      return setTimeout(() => {
        callback({
          didTimeout: false,
          timeRemaining: () => 50,
        });
      }, 1) as unknown as number;
    }
  },

  // Performance measurement wrapper
  measure: <T>(name: string, fn: () => T): T => {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    console.log(`⚡ ${name}: ${(end - start).toFixed(2)}ms`);
    return result;
  },

  // Batch DOM operations for performance
  batchDOM: (operations: () => void) => {
    if (typeof window !== 'undefined' && 'requestAnimationFrame' in window) {
      requestAnimationFrame(() => {
        operations();
      });
    } else {
      operations();
    }
  },

  // Virtual scroll helper
  calculateVisibleRange: (
    scrollTop: number,
    containerHeight: number,
    itemHeight: number,
    totalItems: number,
    overscan: number = 5
  ) => {
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const endIndex = Math.min(
      totalItems - 1,
      Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
    );
    return { startIndex, endIndex };
  },

  // Image optimization helper
  getOptimalImageUrl: (
    baseUrl: string,
    options: {
      width?: number;
      height?: number;
      quality?: number;
      format?: 'webp' | 'avif' | 'jpeg' | 'png';
    }
  ) => {
    const url = new URL(baseUrl, window.location.origin);
    const params = new URLSearchParams(url.search);

    if (options.width) params.set('w', options.width.toString());
    if (options.height) params.set('h', options.height.toString());
    if (options.quality) params.set('q', options.quality.toString());
    if (options.format) params.set('f', options.format);

    return `${url.origin}${url.pathname}?${params.toString()}`;
  },

  // Performance score calculator
  calculatePerformanceScore: (metrics: {
    fcp?: number;
    lcp?: number;
    cls?: number;
    fid?: number;
    tti?: number;
    bundleSize?: number;
  }) => {
    let score = 100;

    // FCP score (0-30 points)
    if (metrics.fcp) {
      if (metrics.fcp > 3000) score -= 30;
      else if (metrics.fcp > 1800) score -= 15;
    }

    // LCP score (0-25 points)
    if (metrics.lcp) {
      if (metrics.lcp > 4000) score -= 25;
      else if (metrics.lcp > 2500) score -= 12;
    }

    // CLS score (0-20 points)
    if (metrics.cls) {
      if (metrics.cls > 0.25) score -= 20;
      else if (metrics.cls > 0.1) score -= 10;
    }

    // FID score (0-15 points)
    if (metrics.fid) {
      if (metrics.fid > 300) score -= 15;
      else if (metrics.fid > 100) score -= 7;
    }

    // Bundle size score (0-10 points)
    if (metrics.bundleSize) {
      if (metrics.bundleSize > 1000000) score -= 10;
      else if (metrics.bundleSize > 500000) score -= 5;
    }

    return Math.max(0, score);
  },

  // Performance health check
  healthCheck: () => {
    const health = {
      performance: true,
      accessibility: true,
      bestPractices: true,
      seo: true,
      issues: [] as string[],
    };

    // Check for common performance issues
    if ('performance' in window) {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        const ttfb = navigation.responseStart - navigation.requestStart;
        if (ttfb > 600) {
          health.performance = false;
          health.issues.push('High Time to First Byte (TTFB)');
        }
      }
    }

    // Check for large images
    const images = document.querySelectorAll('img');
    images.forEach((img) => {
      if (img.naturalWidth > 2000 || img.naturalHeight > 2000) {
        health.performance = false;
        health.issues.push('Large images detected');
      }
    });

    // Check for render-blocking resources
    const renderBlockingResources = document.querySelectorAll('script[src], link[rel="stylesheet"][href]');
    if (renderBlockingResources.length > 5) {
      health.performance = false;
      health.issues.push('Too many render-blocking resources');
    }

    return health;
  },
};

// Performance optimization configurations
export const performanceConfigs = {
  // Aggressive optimization for production
  production: {
    virtualization: {
      enabled: true,
      itemHeight: 50,
      containerHeight: 600,
      overscanCount: 5,
      enableDynamicSizing: true,
    },
    caching: {
      memoryCacheSize: 100,
      sessionStorageSize: 20,
      localStorageSize: 10,
      apiCacheTTL: 600,
      componentCacheTTL: 1200,
    },
    resources: {
      enableLazyLoading: true,
      enablePreloading: true,
      enablePrefetching: true,
      maxConcurrentRequests: 6,
      enableCompression: true,
    },
    monitoring: {
      enabled: true,
      sampleRate: 0.1,
      reportErrors: true,
      consoleMetrics: false,
      performanceThresholds: {
        firstContentfulPaint: 1500,
        largestContentfulPaint: 2500,
        cumulativeLayoutShift: 0.1,
        firstInputDelay: 100,
        timeToInteractive: 3000,
      },
    },
  },

  // Balanced optimization for development
  development: {
    virtualization: {
      enabled: true,
      itemHeight: 50,
      containerHeight: 600,
      overscanCount: 3,
      enableDynamicSizing: false,
    },
    caching: {
      memoryCacheSize: 50,
      sessionStorageSize: 10,
      localStorageSize: 5,
      apiCacheTTL: 300,
      componentCacheTTL: 600,
    },
    resources: {
      enableLazyLoading: true,
      enablePreloading: false,
      enablePrefetching: false,
      maxConcurrentRequests: 4,
      enableCompression: false,
    },
    monitoring: {
      enabled: true,
      sampleRate: 1.0,
      reportErrors: true,
      consoleMetrics: true,
      performanceThresholds: {
        firstContentfulPaint: 3000,
        largestContentfulPaint: 4000,
        cumulativeLayoutShift: 0.25,
        firstInputDelay: 200,
        timeToInteractive: 5000,
      },
    },
  },

  // Lightweight optimization for testing
  testing: {
    virtualization: {
      enabled: false,
      itemHeight: 50,
      containerHeight: 600,
      overscanCount: 1,
      enableDynamicSizing: false,
    },
    caching: {
      memoryCacheSize: 10,
      sessionStorageSize: 5,
      localStorageSize: 2,
      apiCacheTTL: 60,
      componentCacheTTL: 120,
    },
    resources: {
      enableLazyLoading: false,
      enablePreloading: false,
      enablePrefetching: false,
      maxConcurrentRequests: 2,
      enableCompression: false,
    },
    monitoring: {
      enabled: false,
      sampleRate: 0,
      reportErrors: false,
      consoleMetrics: true,
      performanceThresholds: {
        firstContentfulPaint: 10000,
        largestContentfulPaint: 15000,
        cumulativeLayoutShift: 1.0,
        firstInputDelay: 1000,
        timeToInteractive: 20000,
      },
    },
  },
};

// Performance optimization presets
export const performancePresets = {
  // E-commerce preset
  ecommerce: {
    ...performanceConfigs.production,
    virtualization: {
      ...performanceConfigs.production.virtualization,
      itemHeight: 80,
      overscanCount: 8,
    },
    caching: {
      ...performanceConfigs.production.caching,
      apiCacheTTL: 1800, // 30 minutes for product data
      componentCacheTTL: 3600, // 1 hour for components
    },
  },

  // Dashboard preset
  dashboard: {
    ...performanceConfigs.production,
    virtualization: {
      ...performanceConfigs.production.virtualization,
      itemHeight: 60,
      overscanCount: 10,
      enableDynamicSizing: true,
    },
    resources: {
      ...performanceConfigs.production.resources,
      enablePrefetching: true,
      maxConcurrentRequests: 8,
    },
  },

  // Content preset
  content: {
    ...performanceConfigs.production,
    virtualization: {
      ...performanceConfigs.production.virtualization,
      itemHeight: 120,
      overscanCount: 5,
      enableDynamicSizing: true,
    },
    resources: {
      ...performanceConfigs.production.resources,
      enablePreloading: true,
    },
  },
};

// Export default
export default performanceOptimizer;