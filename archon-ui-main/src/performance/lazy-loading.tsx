/**
 * Lazy Loading Components
 *
 * Advanced lazy loading with:
 * - Intersection Observer API
 * - Preloading strategies
 * - Error boundaries
 * - Loading states
 * - Performance optimizations
 */

import React, { useState, useEffect, useRef, Suspense, lazy, ComponentType } from 'react';

// Lazy loading configuration
export interface LazyConfig {
  threshold?: number;
  rootMargin?: string;
  triggerOnce?: boolean;
  placeholder?: React.ReactNode;
  fallback?: React.ReactNode;
  errorComponent?: React.ReactNode;
  retryCount?: number;
  retryDelay?: number;
  preloadDistance?: number;
}

// Loading states
export type LoadingState = 'idle' | 'loading' | 'loaded' | 'error';

// Intersection Observer hook
export function useIntersectionObserver(
  elementRef: React.RefObject<Element>,
  config: IntersectionObserverInit = {}
): boolean {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(([entry]) => {
      setIsVisible(entry.isIntersecting);
    }, {
      threshold: 0.1,
      rootMargin: '50px',
      ...config,
    });

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [elementRef, config]);

  return isVisible;
}

// Lazy Image component
interface LazyImageProps extends React.ImgHTMLAttributes<HTMLImageElement> {
  src: string;
  placeholder?: string;
  threshold?: number;
  rootMargin?: string;
  onLoad?: () => void;
  onError?: () => void;
  className?: string;
  style?: React.CSSProperties;
}

export const LazyImage: React.FC<LazyImageProps> = ({
  src,
  placeholder,
  threshold = 0.1,
  rootMargin = '50px',
  onLoad,
  onError,
  className = '',
  style = {},
  ...props
}) => {
  const [imgSrc, setImgSrc] = useState(placeholder);
  const [loading, setLoading] = useState<LoadingState>('idle');
  const imgRef = useRef<HTMLImageElement>(null);
  const hasLoaded = useRef(false);

  const isVisible = useIntersectionObserver(imgRef, {
    threshold,
    rootMargin,
  });

  useEffect(() => {
    if (isVisible && !hasLoaded.current && src) {
      setLoading('loading');

      const img = new Image();
      img.onload = () => {
        setImgSrc(src);
        setLoading('loaded');
        hasLoaded.current = true;
        onLoad?.();
      };

      img.onerror = () => {
        setLoading('error');
        onError?.();
      };

      img.src = src;
    }
  }, [isVisible, src, onLoad, onError]);

  // Preload image when hovering near it
  const handleMouseEnter = () => {
    if (!hasLoaded.current && src && loading === 'idle') {
      const img = new Image();
      img.src = src;
    }
  };

  return (
    <img
      ref={imgRef}
      src={imgSrc}
      onMouseEnter={handleMouseEnter}
      className={`${className} ${loading === 'loading' ? 'opacity-50' : ''} transition-opacity duration-300`}
      style={{
        ...style,
        filter: loading === 'loading' ? 'blur(4px)' : 'none',
      }}
      {...props}
    />
  );
};

// Lazy Component wrapper
interface LazyComponentProps {
  component: ComponentType<any>;
  props?: any;
  config?: LazyConfig;
  preload?: boolean;
}

export const LazyComponent: React.FC<LazyComponentProps> = ({
  component: Component,
  props = {},
  config = {},
  preload = false,
}) => {
  const [LazyLoadedComponent, setLazyLoadedComponent] = useState<ComponentType<any> | null>(null);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [retryCount, setRetryCount] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const isVisible = useIntersectionObserver(containerRef, {
    threshold: config.threshold || 0.1,
    rootMargin: config.rootMargin || '50px',
  });

  const loadComponent = async () => {
    if (loadingState !== 'idle') return;

    setLoadingState('loading');

    try {
      // Dynamic import
      const module = await import(/* webpackChunkName: "lazy-component" */ 'react');
      const LazyComp = lazy(() => Promise.resolve({ default: Component }));
      setLazyLoadedComponent(() => LazyComp);
      setLoadingState('loaded');
    } catch (error) {
      console.error('Failed to load lazy component:', error);
      setLoadingState('error');

      // Retry logic
      if (retryCount < (config.retryCount || 3)) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          setLoadingState('idle');
        }, config.retryDelay || 1000);
      }
    }
  };

  useEffect(() => {
    if ((isVisible || preload) && loadingState === 'idle') {
      loadComponent();
    }
  }, [isVisible, preload, loadingState]);

  const renderPlaceholder = () => {
    return config.placeholder || (
      <div className="flex items-center justify-center p-8 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  };

  const renderError = () => {
    return config.errorComponent || (
      <div className="flex items-center justify-center p-8 bg-red-50 dark:bg-red-900/20 rounded-lg">
        <div className="text-center">
          <div className="text-red-500 text-4xl mb-4">⚠️</div>
          <p className="text-red-600 dark:text-red-400">Failed to load component</p>
          {retryCount < (config.retryCount || 3) && (
            <button
              onClick={loadComponent}
              className="mt-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div ref={containerRef} className="lazy-component-container">
      {loadingState === 'loading' && renderPlaceholder()}
      {loadingState === 'error' && renderError()}
      {LazyLoadedComponent && loadingState === 'loaded' && (
        <Suspense fallback={renderPlaceholder()}>
          <LazyLoadedComponent {...props} />
        </Suspense>
      )}
    </div>
  );
};

// Lazy Route component
interface LazyRouteProps {
  path: string;
  component: ComponentType<any>;
  exact?: boolean;
  preload?: boolean;
}

export const LazyRoute: React.FC<LazyRouteProps> = ({
  path,
  component: Component,
  exact = false,
  preload = false,
}) => {
  const LazyLoadedRoute = lazy(() => Promise.resolve({ default: Component }));

  return (
    <Route
      path={path}
      exact={exact}
      element={
        <Suspense fallback={<div>Loading route...</div>}>
          <LazyLoadedRoute />
        </Suspense>
      }
    />
  );
};

// Lazy List component
interface LazyListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  batchSize?: number;
  threshold?: number;
  placeholder?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

export const LazyList = React.memo(<T extends any>({
  items,
  renderItem,
  batchSize = 10,
  threshold = 0.1,
  placeholder,
  className = '',
  style = {},
}: LazyListProps<T>) => {
  const [visibleCount, setVisibleCount] = useState(batchSize);
  const listRef = useRef<HTMLDivElement>(null);
  const isLoading = useRef(false);

  const loadMore = () => {
    if (isLoading.current || visibleCount >= items.length) return;

    isLoading.current = true;
    setVisibleCount(prev => Math.min(prev + batchSize, items.length));

    // Simulate loading delay
    setTimeout(() => {
      isLoading.current = false;
    }, 100);
  };

  const isVisible = useIntersectionObserver(listRef, {
    threshold,
    rootMargin: '100px',
  });

  useEffect(() => {
    if (isVisible && visibleCount < items.length) {
      loadMore();
    }
  }, [isVisible, visibleCount, items.length]);

  const visibleItems = items.slice(0, visibleCount);

  return (
    <div ref={listRef} className={`lazy-list ${className}`} style={style}>
      {visibleItems.map((item, index) => (
        <div key={index} className="lazy-list-item">
          {renderItem(item, index)}
        </div>
      ))}

      {visibleCount < items.length && (
        <div className="lazy-list-loading">
          {placeholder || (
            <div className="flex justify-center p-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

// Lazy Grid component
interface LazyGridProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  columns?: number;
  batchSize?: number;
  threshold?: number;
  gap?: string;
  className?: string;
  style?: React.CSSProperties;
}

export const LazyGrid = React.memo(<T extends any>({
  items,
  renderItem,
  columns = 3,
  batchSize = 12,
  threshold = 0.1,
  gap = '1rem',
  className = '',
  style = {},
}: LazyGridProps<T>) => {
  const [visibleCount, setVisibleCount] = useState(batchSize);
  const gridRef = useRef<HTMLDivElement>(null);

  const loadMore = () => {
    if (visibleCount >= items.length) return;
    setVisibleCount(prev => Math.min(prev + batchSize, items.length));
  };

  const isVisible = useIntersectionObserver(gridRef, {
    threshold,
    rootMargin: '100px',
  });

  useEffect(() => {
    if (isVisible && visibleCount < items.length) {
      loadMore();
    }
  }, [isVisible, visibleCount, items.length]);

  const visibleItems = items.slice(0, visibleCount);

  return (
    <div
      ref={gridRef}
      className={`lazy-grid ${className}`}
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap,
        ...style,
      }}
    >
      {visibleItems.map((item, index) => (
        <div key={index} className="lazy-grid-item">
          {renderItem(item, index)}
        </div>
      ))}

      {visibleCount < items.length && (
        <div className="lazy-grid-loading col-span-full">
          <div className="flex justify-center p-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          </div>
        </div>
      )}
    </div>
  );
});

// Preload manager
class PreloadManager {
  private preloadedComponents = new Set<string>();
  private preloadedImages = new Set<string>();

  preloadComponent(componentPath: string): void {
    if (this.preloadedComponents.has(componentPath)) return;

    // Start preloading
    import(/* webpackPrefetch: true */ `@/components/${componentPath}`)
      .catch(error => {
        console.warn('Failed to preload component:', componentPath, error);
      });

    this.preloadedComponents.add(componentPath);
  }

  preloadImage(src: string): void {
    if (this.preloadedImages.has(src)) return;

    const img = new Image();
    img.src = src;
    this.preloadedImages.add(src);
  }

  preloadOnHover(element: HTMLElement, componentPath: string): void {
    element.addEventListener('mouseenter', () => {
      this.preloadComponent(componentPath);
    }, { once: true });
  }

  preloadOnFocus(element: HTMLElement, componentPath: string): void {
    element.addEventListener('focus', () => {
      this.preloadComponent(componentPath);
    }, { once: true });
  }
}

export const preloadManager = new PreloadManager();

// Lazy loading utilities
export const lazyUtils = {
  // Create lazy component with automatic code splitting
  createLazyComponent: <T extends ComponentType<any>>(
    importFn: () => Promise<{ default: T }>,
    fallback?: React.ReactNode
  ): React.FC<React.ComponentProps<T>> => {
    const LazyComponent = lazy(importFn);
    return (props) => (
      <Suspense fallback={fallback || <div>Loading...</div>}>
        <LazyComponent {...props} />
      </Suspense>
    );
  },

  // Preload multiple components
  preloadComponents: (componentPaths: string[]) => {
    componentPaths.forEach(path => {
      preloadManager.preloadComponent(path);
    });
  },

  // Preload multiple images
  preloadImages: (imageUrls: string[]) => {
    imageUrls.forEach(url => {
      preloadManager.preloadImage(url);
    });
  },

  // Create intersection observer with custom options
  createIntersectionObserver: (
    callback: IntersectionObserverCallback,
    options?: IntersectionObserverInit
  ) => {
    return new IntersectionObserver(callback, {
      threshold: [0, 0.1, 0.5, 1],
      rootMargin: '50px',
      ...options,
    });
  },
};

export default LazyImage;