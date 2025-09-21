/**
 * Resource Optimizer
 *
 * Advanced resource optimization with:
 * - Preloading and prefetching
 * - Critical CSS extraction
 * - Font optimization
 * - Resource prioritization
 * - Bundle analysis and optimization
 */

// Resource configuration
export interface ResourceConfig {
  enableLazyLoading: boolean;
  enablePreloading: boolean;
  enablePrefetching: boolean;
  maxConcurrentRequests: number;
  enableCompression: boolean;
  criticalResources: string[];
  preloadResources: string[];
  prefetchResources: string[];
  priorityResources: string[];
}

// Resource entry
export interface ResourceEntry {
  url: string;
  type: 'script' | 'style' | 'image' | 'font' | 'other';
  priority: 'high' | 'medium' | 'low';
  size?: number;
  loadTime?: number;
  cached: boolean;
  critical?: boolean;
}

// Resource metrics
export interface ResourceMetrics {
  totalSize: number;
  totalRequests: number;
  cacheHitRate: number;
  averageLoadTime: number;
  criticalResourcesLoaded: number;
  priorityScore: number;
}

// Resource Optimizer class
export class ResourceOptimizer {
  private config: ResourceConfig;
  private resources: Map<string, ResourceEntry> = new Map();
  private loadQueue: string[] = [];
  private loadingResources: Set<string> = new Set();
  private observer: IntersectionObserver | null = null;
  private metrics: ResourceMetrics;

  constructor(config: Partial<ResourceConfig> = {}) {
    this.config = {
      enableLazyLoading: true,
      enablePreloading: true,
      enablePrefetching: true,
      maxConcurrentRequests: 6,
      enableCompression: true,
      criticalResources: [],
      preloadResources: [],
      prefetchResources: [],
      priorityResources: [],
      ...config,
    };

    this.metrics = this.initializeMetrics();
    this.initialize();
  }

  private initializeMetrics(): ResourceMetrics {
    return {
      totalSize: 0,
      totalRequests: 0,
      cacheHitRate: 0,
      averageLoadTime: 0,
      criticalResourcesLoaded: 0,
      priorityScore: 0,
    };
  }

  private initialize(): void {
    this.setupResourceObserver();
    this.setupNetworkMonitoring();
    this.initializeCriticalResources();
    this.startOptimization();
  }

  // Setup Intersection Observer for lazy loading
  private setupResourceObserver(): void {
    if (!('IntersectionObserver' in window)) return;

    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const element = entry.target as HTMLElement;
            const resourceUrl = element.getAttribute('data-resource-url');
            if (resourceUrl) {
              this.loadResource(resourceUrl);
              this.observer?.unobserve(element);
            }
          }
        });
      },
      {
        rootMargin: '50px',
        threshold: 0.1,
      }
    );
  }

  // Setup network monitoring
  private setupNetworkMonitoring(): void {
    // Monitor resource timing
    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      this.processResourceEntries(entries);
    });

    try {
      observer.observe({ entryTypes: ['resource'] });
    } catch (error) {
      console.warn('Resource timing API not supported');
    }
  }

  // Process resource timing entries
  private processResourceEntries(entries: PerformanceResourceTiming[]): void {
    entries.forEach((entry) => {
      const url = entry.name;
      const resource: ResourceEntry = {
        url,
        type: this.getResourceType(entry),
        priority: this.getPriorityFromURL(url),
        size: (entry as any).transferSize || 0,
        loadTime: entry.duration,
        cached: (entry as any).transferSize === 0,
        critical: this.config.criticalResources.includes(url),
      };

      this.resources.set(url, resource);
      this.updateMetrics(resource);
    });
  }

  // Get resource type from entry
  private getResourceType(entry: PerformanceResourceTiming): 'script' | 'style' | 'image' | 'font' | 'other' {
    const url = entry.name.toLowerCase();
    if (url.endsWith('.js') || url.includes('.js?')) return 'script';
    if (url.endsWith('.css') || url.includes('.css?')) return 'style';
    if (url.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) return 'image';
    if (url.match(/\.(woff|woff2|ttf|otf|eot)$/)) return 'font';
    return 'other';
  }

  // Get priority from URL
  private getPriorityFromURL(url: string): 'high' | 'medium' | 'low' {
    if (this.config.priorityResources.includes(url)) return 'high';
    if (this.config.criticalResources.includes(url)) return 'high';
    if (url.includes('critical') || url.includes('important')) return 'high';
    return 'medium';
  }

  // Update metrics
  private updateMetrics(resource: ResourceEntry): void {
    this.metrics.totalRequests++;
    this.metrics.totalSize += resource.size || 0;
    this.metrics.averageLoadTime = (this.metrics.averageLoadTime * (this.metrics.totalRequests - 1) + (resource.loadTime || 0)) / this.metrics.totalRequests;

    if (resource.cached) {
      this.metrics.cacheHitRate = (this.metrics.cacheHitRate * (this.metrics.totalRequests - 1) + 1) / this.metrics.totalRequests;
    }

    if (resource.critical) {
      this.metrics.criticalResourcesLoaded++;
    }

    // Calculate priority score
    const priorityWeight = resource.priority === 'high' ? 3 : resource.priority === 'medium' ? 2 : 1;
    this.metrics.priorityScore += priorityWeight;
  }

  // Initialize critical resources
  private initializeCriticalResources(): void {
    this.config.criticalResources.forEach(url => {
      this.preloadResource(url);
    });
  }

  // Start optimization process
  private startOptimization(): void {
    // Preload critical resources
    this.config.preloadResources.forEach(url => {
      this.preloadResource(url);
    });

    // Prefetch resources
    if (this.config.enablePrefetching) {
      this.config.prefetchResources.forEach(url => {
        this.prefetchResource(url);
      });
    }

    // Optimize fonts
    this.optimizeFonts();

    // Optimize images
    this.optimizeImages();
  }

  // Preload resource
  preloadResource(url: string): void {
    if (!this.config.enablePreloading || this.loadingResources.has(url)) return;

    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = url;
    link.as = this.getResourceAsAttribute(url);

    link.onload = () => {
      this.loadingResources.delete(url);
    };

    link.onerror = () => {
      this.loadingResources.delete(url);
    };

    document.head.appendChild(link);
    this.loadingResources.add(url);
  }

  // Prefetch resource
  prefetchResource(url: string): void {
    if (!this.config.enablePrefetching || this.loadingResources.has(url)) return;

    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = url;

    document.head.appendChild(link);
    this.loadingResources.add(url);
  }

  // Get resource as attribute
  private getResourceAsAttribute(url: string): string {
    const urlLower = url.toLowerCase();
    if (urlLower.endsWith('.js')) return 'script';
    if (urlLower.endsWith('.css')) return 'style';
    if (urlLower.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) return 'image';
    if (urlLower.match(/\.(woff|woff2|ttf|otf|eot)$/)) return 'font';
    return 'fetch';
  }

  // Optimize fonts
  private optimizeFonts(): void {
    // Add font-display: swap for better performance
    const style = document.createElement('style');
    style.textContent = `
      @font-face {
        font-display: swap;
      }
    `;
    document.head.appendChild(style);

    // Preload critical fonts
    document.querySelectorAll('link[rel="stylesheet"]').forEach(link => {
      const href = link.getAttribute('href');
      if (href && href.includes('font')) {
        this.preloadResource(href);
      }
    });
  }

  // Optimize images
  private optimizeImages(): void {
    // Add lazy loading to images
    document.querySelectorAll('img').forEach(img => {
      if (!img.hasAttribute('loading')) {
        img.setAttribute('loading', 'lazy');
      }
    });
  }

  // Load resource with queuing
  async loadResource(url: string): Promise<void> {
    if (this.loadingResources.has(url)) return;

    // Check if already loaded
    if (this.resources.has(url)) {
      const resource = this.resources.get(url)!;
      if (resource.cached) return;
    }

    // Add to queue
    this.loadQueue.push(url);
    this.processQueue();
  }

  // Process load queue
  private async processQueue(): Promise<void> {
    while (this.loadQueue.length > 0 && this.loadingResources.size < this.config.maxConcurrentRequests) {
      const url = this.loadQueue.shift();
      if (!url) break;

      this.loadingResources.add(url);

      try {
        await this.fetchResource(url);
      } catch (error) {
        console.warn(`Failed to load resource: ${url}`, error);
      } finally {
        this.loadingResources.delete(url);
      }
    }
  }

  // Fetch resource
  private async fetchResource(url: string): Promise<void> {
    const startTime = performance.now();

    const response = await fetch(url, {
      priority: this.config.priorityResources.includes(url) ? 'high' : 'auto',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const loadTime = performance.now() - startTime;
    const resource: ResourceEntry = {
      url,
      type: this.getResourceTypeFromResponse(response),
      priority: this.getPriorityFromURL(url),
      size: parseInt(response.headers.get('content-length') || '0'),
      loadTime,
      cached: response.headers.get('x-from-cache') === 'true',
      critical: this.config.criticalResources.includes(url),
    };

    this.resources.set(url, resource);
    this.updateMetrics(resource);
  }

  // Get resource type from response
  private getResourceTypeFromResponse(response: Response): 'script' | 'style' | 'image' | 'font' | 'other' {
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('javascript')) return 'script';
    if (contentType.includes('css')) return 'style';
    if (contentType.includes('image')) return 'image';
    if (contentType.includes('font')) return 'font';
    return 'other';
  }

  // Optimize critical path
  async optimizeCriticalPath(): Promise<void> {
    // Extract critical CSS
    await this.extractCriticalCSS();

    // Inline critical CSS
    await this.inlineCriticalCSS();

    // Load critical resources
    await this.loadCriticalResources();
  }

  // Extract critical CSS
  private async extractCriticalCSS(): Promise<void> {
    // This is a placeholder for critical CSS extraction
    // In a real implementation, you would use a tool like critical CSS
    console.log('Extracting critical CSS...');
  }

  // Inline critical CSS
  private async inlineCriticalCSS(): Promise<void> {
    // This is a placeholder for critical CSS inlining
    console.log('Inlining critical CSS...');
  }

  // Load critical resources
  private async loadCriticalResources(): Promise<void> {
    const promises = this.config.criticalResources.map(url => this.loadResource(url));
    await Promise.all(promises);
  }

  // Get resource metrics
  getMetrics(): ResourceMetrics {
    return { ...this.metrics };
  }

  // Get resource entries
  getResourceEntries(): ResourceEntry[] {
    return Array.from(this.resources.values());
  }

  // Get optimization recommendations
  getRecommendations(): string[] {
    const recommendations: string[] = [];

    if (this.metrics.cacheHitRate < 0.7) {
      recommendations.push('Implement better caching strategies');
    }

    if (this.metrics.averageLoadTime > 1000) {
      recommendations.push('Optimize resource loading and consider CDN');
    }

    if (this.metrics.totalSize > 2000000) { // 2MB
      recommendations.push('Consider code splitting and lazy loading');
    }

    // Check for large resources
    const largeResources = Array.from(this.resources.values()).filter(r => (r.size || 0) > 500000);
    if (largeResources.length > 0) {
      recommendations.push('Optimize large resources');
    }

    return recommendations;
  }

  // Analyze bundle
  async analyzeBundle(): Promise<{
    bundleSize: number;
    chunkCount: number;
    largeAssets: string[];
    duplicates: string[];
  }> {
    // This is a placeholder for bundle analysis
    // In a real implementation, you would use webpack-bundle-analyzer
    return {
      bundleSize: 0,
      chunkCount: 0,
      largeAssets: [],
      duplicates: [],
    };
  }

  // Clear cache
  clearCache(): void {
    this.resources.clear();
    this.metrics = this.initializeMetrics();
  }

  // Destroy optimizer
  destroy(): void {
    if (this.observer) {
      this.observer.disconnect();
    }
    this.clearCache();
  }
}

// React hook for resource optimization
export function useResourceOptimizer(config?: Partial<ResourceConfig>) {
  const [metrics, setMetrics] = useState<ResourceMetrics | null>(null);
  const optimizer = React.useMemo(() => new ResourceOptimizer(config), [config]);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(optimizer.getMetrics());
    }, 5000);

    setMetrics(optimizer.getMetrics());

    return () => {
      clearInterval(interval);
      optimizer.destroy();
    };
  }, [optimizer]);

  return { metrics, optimizer };
}

// Resource utilities
export const resourceUtils = {
  // Generate resource priority
  generatePriority: (url: string, type: string): 'high' | 'medium' | 'low' => {
    if (url.includes('critical') || url.includes('important')) return 'high';
    if (type === 'script' && url.includes('main')) return 'high';
    if (type === 'style' && url.includes('main')) return 'high';
    return 'medium';
  },

  // Calculate resource score
  calculateResourceScore: (resource: ResourceEntry): number => {
    let score = 0;

    // Priority weight
    if (resource.priority === 'high') score += 30;
    else if (resource.priority === 'medium') score += 20;
    else score += 10;

    // Critical weight
    if (resource.critical) score += 25;

    // Cache penalty
    if (!resource.cached) score += 15;

    // Size penalty
    if (resource.size && resource.size > 500000) score -= 10;

    return Math.max(0, Math.min(100, score));
  },

  // Optimize resource URL
  optimizeUrl: (url: string, options: { format?: string; quality?: number } = {}): string => {
    if (!url.includes('?')) return url;

    const params = new URLSearchParams();
    if (options.format) params.set('format', options.format);
    if (options.quality) params.set('quality', options.quality.toString());

    return params.toString() ? `${url}&${params.toString()}` : url;
  },
};

export default ResourceOptimizer;