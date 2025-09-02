/**
 * Advanced Lazy Loading Component with Performance Optimizations
 * 
 * Features:
 * - Intersection Observer for viewport-based loading
 * - Preloading on hover/focus
 * - Error boundaries and retry logic
 * - Performance monitoring
 * - Memory management
 */

import React, { 
  Suspense, 
  useState, 
  useEffect, 
  useRef, 
  useCallback,
  ComponentType,
  LazyExoticComponent
} from 'react';
import { ErrorBoundary } from 'react-error-boundary';

export interface LazyLoadOptions {
  // Intersection Observer options
  rootMargin?: string;
  threshold?: number | number[];
  
  // Preloading options
  preloadOnHover?: boolean;
  preloadOnFocus?: boolean;
  preloadDelay?: number;
  
  // Error handling
  retryAttempts?: number;
  retryDelay?: number;
  
  // Performance
  enablePerformanceTracking?: boolean;
  chunkName?: string;
}

export interface LazyComponentState {
  isVisible: boolean;
  isLoading: boolean;
  isLoaded: boolean;
  hasError: boolean;
  retryCount: number;
  loadTime?: number;
}

// Default loading fallback
const DefaultFallback: React.FC<{ 
  chunkName?: string; 
  error?: Error;
  isRetrying?: boolean;
  onRetry?: () => void;
}> = ({ 
  chunkName, 
  error, 
  isRetrying = false, 
  onRetry 
}) => {
  if (error) {
    return (
      <div className="flex items-center justify-center p-8 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20 dark:border-red-800">
        <div className="text-center space-y-4">
          <div className="text-red-600 dark:text-red-400">
            <svg className="w-8 h-8 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="font-medium">Failed to load {chunkName || 'component'}</p>
            <p className="text-sm text-red-500 dark:text-red-300 mt-1">
              {error.message}
            </p>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              disabled={isRetrying}
              className="px-4 py-2 text-sm font-medium text-red-600 dark:text-red-400 border border-red-300 dark:border-red-600 rounded hover:bg-red-50 dark:hover:bg-red-900/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isRetrying ? 'Retrying...' : 'Retry'}
            </button>
          )}
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex items-center justify-center p-8">
      <div className="text-center space-y-4">
        {/* Loading spinner */}
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <div className="text-sm text-gray-600 dark:text-gray-400">
          Loading {chunkName || 'component'}...
        </div>
      </div>
    </div>
  );
};

// Performance tracker
class LazyLoadPerformanceTracker {
  private static instance: LazyLoadPerformanceTracker;
  private metrics: Map<string, {
    loadCount: number;
    totalLoadTime: number;
    averageLoadTime: number;
    errorCount: number;
    retryCount: number;
  }> = new Map();
  
  static getInstance() {
    if (!LazyLoadPerformanceTracker.instance) {
      LazyLoadPerformanceTracker.instance = new LazyLoadPerformanceTracker();
    }
    return LazyLoadPerformanceTracker.instance;
  }
  
  trackLoad(chunkName: string, loadTime: number, hadError = false, retries = 0) {
    const existing = this.metrics.get(chunkName) || {
      loadCount: 0,
      totalLoadTime: 0,
      averageLoadTime: 0,
      errorCount: 0,
      retryCount: 0
    };
    
    existing.loadCount += 1;
    existing.totalLoadTime += loadTime;
    existing.averageLoadTime = existing.totalLoadTime / existing.loadCount;
    if (hadError) existing.errorCount += 1;
    existing.retryCount += retries;
    
    this.metrics.set(chunkName, existing);
    
    // Log performance in development
    if (process.env.NODE_ENV === 'development') {
      console.log(`🚀 LazyLoad Performance [${chunkName}]:`, {
        loadTime: `${loadTime}ms`,
        avgLoadTime: `${existing.averageLoadTime.toFixed(2)}ms`,
        errorRate: `${((existing.errorCount / existing.loadCount) * 100).toFixed(1)}%`
      });
    }
  }
  
  getMetrics(chunkName?: string) {
    if (chunkName) {
      return this.metrics.get(chunkName);
    }
    return Object.fromEntries(this.metrics.entries());
  }
  
  reset() {
    this.metrics.clear();
  }
}

const performanceTracker = LazyLoadPerformanceTracker.getInstance();

// Main LazyLoad component
export function LazyLoad<T = {}>(
  factory: () => Promise<{ default: ComponentType<T> }>,
  options: LazyLoadOptions = {}
) {
  const {
    rootMargin = '50px',
    threshold = 0.1,
    preloadOnHover = true,
    preloadOnFocus = true,
    preloadDelay = 200,
    retryAttempts = 3,
    retryDelay = 1000,
    enablePerformanceTracking = true,
    chunkName = 'unknown-chunk'
  } = options;
  
  // Create the lazy component
  const LazyComponent = React.lazy(factory);
  
  return React.forwardRef<any, T>((props, ref) => {
    const [state, setState] = useState<LazyComponentState>({
      isVisible: false,
      isLoading: false,
      isLoaded: false,
      hasError: false,
      retryCount: 0
    });
    
    const containerRef = useRef<HTMLDivElement>(null);
    const preloadTimeoutRef = useRef<NodeJS.Timeout>();
    const loadStartTimeRef = useRef<number>();
    const lazyComponentRef = useRef<LazyExoticComponent<ComponentType<T>>>();
    
    // Initialize lazy component reference
    if (!lazyComponentRef.current) {
      lazyComponentRef.current = LazyComponent;
    }
    
    // Intersection Observer for viewport-based loading
    useEffect(() => {
      if (!containerRef.current) return;
      
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting && !state.isVisible) {
              setState(prev => ({ ...prev, isVisible: true }));
              loadStartTimeRef.current = performance.now();
            }
          });
        },
        { rootMargin, threshold }
      );
      
      observer.observe(containerRef.current);
      
      return () => observer.disconnect();
    }, [rootMargin, threshold, state.isVisible]);
    
    // Preload handlers
    const handlePreload = useCallback(() => {
      if (state.isLoaded || state.isLoading) return;
      
      preloadTimeoutRef.current = setTimeout(() => {
        setState(prev => ({ ...prev, isVisible: true }));
        loadStartTimeRef.current = performance.now();
      }, preloadDelay);
    }, [state.isLoaded, state.isLoading, preloadDelay]);
    
    const handlePreloadCancel = useCallback(() => {
      if (preloadTimeoutRef.current) {
        clearTimeout(preloadTimeoutRef.current);
      }
    }, []);
    
    // Retry logic
    const handleRetry = useCallback(() => {
      if (state.retryCount >= retryAttempts) return;
      
      setState(prev => ({
        ...prev,
        hasError: false,
        retryCount: prev.retryCount + 1,
        isVisible: true
      }));
      
      loadStartTimeRef.current = performance.now();
      
      // Add delay between retries
      setTimeout(() => {
        // Force re-render to trigger suspense
        setState(prev => ({ ...prev, isLoading: true }));
      }, retryDelay);
    }, [state.retryCount, retryAttempts, retryDelay]);
    
    // Performance tracking
    const trackLoadComplete = useCallback(() => {
      if (!enablePerformanceTracking || !loadStartTimeRef.current) return;
      
      const loadTime = performance.now() - loadStartTimeRef.current;
      performanceTracker.trackLoad(
        chunkName,
        loadTime,
        state.hasError,
        state.retryCount
      );
      
      setState(prev => ({ ...prev, isLoaded: true, loadTime }));
    }, [enablePerformanceTracking, chunkName, state.hasError, state.retryCount]);
    
    // Error boundary fallback
    const ErrorFallback = useCallback(({ error, resetErrorBoundary }: any) => {
      setState(prev => ({ ...prev, hasError: true }));
      
      return (
        <DefaultFallback
          chunkName={chunkName}
          error={error}
          isRetrying={state.retryCount > 0 && state.retryCount < retryAttempts}
          onRetry={() => {
            resetErrorBoundary();
            handleRetry();
          }}
        />
      );
    }, [chunkName, state.retryCount, retryAttempts, handleRetry]);
    
    // Loading fallback
    const LoadingFallback = useCallback(() => (
      <DefaultFallback chunkName={chunkName} />
    ), [chunkName]);
    
    // Cleanup
    useEffect(() => {
      return () => {
        if (preloadTimeoutRef.current) {
          clearTimeout(preloadTimeoutRef.current);
        }
      };
    }, []);
    
    return (
      <div
        ref={containerRef}
        onMouseEnter={preloadOnHover ? handlePreload : undefined}
        onMouseLeave={preloadOnHover ? handlePreloadCancel : undefined}
        onFocus={preloadOnFocus ? handlePreload : undefined}
        onBlur={preloadOnFocus ? handlePreloadCancel : undefined}
        className="lazy-load-container"
      >
        {state.isVisible ? (
          <ErrorBoundary
            FallbackComponent={ErrorFallback}
            onReset={() => {
              setState(prev => ({ ...prev, hasError: false }));
            }}
            resetKeys={[state.retryCount]}
          >
            <Suspense fallback={<LoadingFallback />}>
              <LazyComponent
                {...(props as T)}
                ref={ref}
                onLoad={trackLoadComplete}
              />
            </Suspense>
          </ErrorBoundary>
        ) : (
          <div className="h-64 flex items-center justify-center text-gray-500">
            <div className="text-center space-y-2">
              <div className="text-sm">Component ready to load</div>
              <div className="text-xs text-gray-400">
                Scroll into view or hover to load {chunkName}
              </div>
            </div>
          </div>
        )}
        
        {/* Development info */}
        {process.env.NODE_ENV === 'development' && state.loadTime && (
          <div className="absolute top-2 right-2 bg-black/80 text-white text-xs p-2 rounded font-mono">
            {chunkName}: {state.loadTime.toFixed(0)}ms
          </div>
        )}
      </div>
    );
  });
}

// Utility function for creating lazy-loaded route components
export function createLazyRoute<T = {}>(
  factory: () => Promise<{ default: ComponentType<T> }>,
  options?: LazyLoadOptions
) {
  return LazyLoad(factory, {
    preloadOnHover: false, // Don't preload routes on hover
    preloadOnFocus: false,
    ...options
  });
}

// Export performance utilities
export const lazyLoadUtils = {
  getPerformanceMetrics: (chunkName?: string) => 
    performanceTracker.getMetrics(chunkName),
  resetPerformanceMetrics: () => 
    performanceTracker.reset(),
  trackCustomMetric: (chunkName: string, loadTime: number, hadError?: boolean, retries?: number) =>
    performanceTracker.trackLoad(chunkName, loadTime, hadError, retries)
};

export default LazyLoad;