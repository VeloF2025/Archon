/**
 * Performance Monitor
 *
 * Real-time performance monitoring with:
 * - Core Web Vitals tracking
 * - Custom metrics collection
 * - Anomaly detection
 * - Performance reporting
 * - User experience metrics
 */

import { EventEmitter } from 'events';

// Performance metrics
export interface PerformanceMetrics {
  // Core Web Vitals
  coreWebVitals: {
    firstContentfulPaint: number;
    largestContentfulPaint: number;
    cumulativeLayoutShift: number;
    firstInputDelay: number;
    timeToInteractive: number;
  };

  // Network metrics
  network: {
    requestCount: number;
    totalSize: number;
    cacheHitRate: number;
    averageResponseTime: number;
  };

  // Runtime metrics
  runtime: {
    frameRate: number;
    memoryUsage: number;
    cpuUsage: number;
    responseTime: number;
  };

  // Custom metrics
  custom: {
    [key: string]: number;
  };

  // User interaction metrics
  interactions: {
    clickTime: number;
    scrollTime: number;
    inputTime: number;
    interactionCount: number;
  };
}

// Performance configuration
export interface MonitorConfig {
  enabled: boolean;
  sampleRate: number;
  reportErrors: boolean;
  consoleMetrics: boolean;
  performanceThresholds: {
    firstContentfulPaint: number;
    largestContentfulPaint: number;
    cumulativeLayoutShift: number;
    firstInputDelay: number;
    timeToInteractive: number;
    frameRate: number;
    memoryUsage: number;
    responseTime: number;
  };
  reportInterval: number;
  enableLongTasks: boolean;
  enableUserTiming: boolean;
}

// Error tracking
export interface PerformanceError {
  message: string;
  filename?: string;
  line?: number;
  column?: number;
  error?: Error;
  timestamp: number;
  type: 'js' | 'resource' | 'network' | 'custom';
}

// Performance Monitor class
export class PerformanceMonitor extends EventEmitter {
  private config: MonitorConfig;
  private metrics: PerformanceMetrics;
  private measurements: Map<string, number> = new Map();
  private observers: PerformanceObserver[] = [];
  private frameCount = 0;
  private lastFrameTime = 0;
  private reportInterval: NodeJS.Timeout;

  constructor(config: Partial<MonitorConfig> = {}) {
    super();

    this.config = {
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
        frameRate: 55,
        memoryUsage: 100,
        responseTime: 200,
      },
      reportInterval: 30000, // 30 seconds
      enableLongTasks: true,
      enableUserTiming: true,
      ...config,
    };

    this.metrics = this.initializeMetrics();
  }

  private initializeMetrics(): PerformanceMetrics {
    return {
      coreWebVitals: {
        firstContentfulPaint: 0,
        largestContentfulPaint: 0,
        cumulativeLayoutShift: 0,
        firstInputDelay: 0,
        timeToInteractive: 0,
      },
      network: {
        requestCount: 0,
        totalSize: 0,
        cacheHitRate: 0,
        averageResponseTime: 0,
      },
      runtime: {
        frameRate: 0,
        memoryUsage: 0,
        cpuUsage: 0,
        responseTime: 0,
      },
      custom: {},
      interactions: {
        clickTime: 0,
        scrollTime: 0,
        inputTime: 0,
        interactionCount: 0,
      },
    };
  }

  // Start monitoring
  startMonitoring(): void {
    if (!this.config.enabled) return;

    this.setupCoreWebVitals();
    this.setupNetworkMonitoring();
    this.setupRuntimeMonitoring();
    this.setupUserInteractionTracking();
    this.setupErrorTracking();

    if (this.config.enableLongTasks) {
      this.setupLongTaskMonitoring();
    }

    if (this.config.enableUserTiming) {
      this.setupUserTiming();
    }

    // Start reporting interval
    this.reportInterval = setInterval(() => {
      this.reportMetrics();
    }, this.config.reportInterval);

    // Start FPS monitoring
    this.startFPSMonitoring();

    this.emit('start');
  }

  // Stop monitoring
  stopMonitoring(): void {
    if (this.reportInterval) {
      clearInterval(this.reportInterval);
    }

    this.observers.forEach(observer => {
      observer.disconnect();
    });

    this.observers = [];
    this.emit('stop');
  }

  // Setup Core Web Vitals monitoring
  private setupCoreWebVitals(): void {
    if (!('PerformanceObserver' in window)) return;

    try {
      // First Contentful Paint
      const paintObserver = new PerformanceObserver((list) => {
        const entries = list.getEntriesByName('first-contentful-paint');
        if (entries.length > 0) {
          const fcp = entries[entries.length - 1];
          this.updateMetric('firstContentfulPaint', fcp.startTime);
          this.checkThreshold('firstContentfulPaint', fcp.startTime);
        }
      });
      paintObserver.observe({ entryTypes: ['paint'] });
      this.observers.push(paintObserver);

      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lcp = entries[entries.length - 1];
        this.updateMetric('largestContentfulPaint', lcp.startTime);
        this.checkThreshold('largestContentfulPaint', lcp.startTime);
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      this.observers.push(lcpObserver);

      // Cumulative Layout Shift
      const clsObserver = new PerformanceObserver((list) => {
        let cls = 0;
        for (const entry of list.getEntries()) {
          cls += entry.value;
        }
        this.updateMetric('cumulativeLayoutShift', cls);
        this.checkThreshold('cumulativeLayoutShift', cls);
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.push(clsObserver);

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const fid = entries[0];
        const delay = fid.processingStart - fid.startTime;
        this.updateMetric('firstInputDelay', delay);
        this.checkThreshold('firstInputDelay', delay);
      });
      fidObserver.observe({ entryTypes: ['first-input'] });
      this.observers.push(fidObserver);

      // Time to Interactive
      this.calculateTimeToInteractive();

    } catch (error) {
      console.warn('Failed to setup Core Web Vitals monitoring:', error);
    }
  }

  // Setup network monitoring
  private setupNetworkMonitoring(): void {
    // Resource timing
    const resourceObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      let totalSize = 0;
      let totalTime = 0;
      let cacheHits = 0;

      entries.forEach(entry => {
        totalSize += (entry as any).transferSize || 0;
        totalTime += entry.duration;
        if ((entry as any).transferSize === 0) {
          cacheHits++;
        }
      });

      this.metrics.network.requestCount = entries.length;
      this.metrics.network.totalSize = totalSize;
      this.metrics.network.cacheHitRate = entries.length > 0 ? cacheHits / entries.length : 0;
      this.metrics.network.averageResponseTime = entries.length > 0 ? totalTime / entries.length : 0;
    });

    try {
      resourceObserver.observe({ entryTypes: ['resource'] });
      this.observers.push(resourceObserver);
    } catch (error) {
      console.warn('Failed to setup resource timing:', error);
    }

    // Navigation timing
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        this.updateMetric('timeToInteractive', navigation.domInteractive - navigation.fetchStart);
      }
    });
  }

  // Setup runtime monitoring
  private setupRuntimeMonitoring(): void {
    // Memory usage
    if ('memory' in performance) {
      setInterval(() => {
        const memory = (performance as any).memory;
        this.metrics.runtime.memoryUsage = memory.usedJSHeapSize / (1024 * 1024); // MB
      }, 5000);
    }

    // CPU usage estimation
    this.estimateCPUUsage();
  }

  // Setup user interaction tracking
  private setupUserInteractionTracking(): void {
    let interactionStart = 0;

    const trackInteraction = (type: 'click' | 'scroll' | 'input') => {
      return (event: Event) => {
        interactionStart = performance.now();
        this.metrics.interactions.interactionCount++;

        event.addEventListener('mouseup' in event ? 'mouseup' : 'input', () => {
          const duration = performance.now() - interactionStart;
          switch (type) {
            case 'click':
              this.metrics.interactions.clickTime = duration;
              break;
            case 'scroll':
              this.metrics.interactions.scrollTime = duration;
              break;
            case 'input':
              this.metrics.interactions.inputTime = duration;
              break;
          }
        }, { once: true });
      };
    };

    document.addEventListener('mousedown', trackInteraction('click'));
    document.addEventListener('touchstart', trackInteraction('click'));
    document.addEventListener('wheel', trackInteraction('scroll'));
    document.addEventListener('keydown', trackInteraction('input'));
  }

  // Setup error tracking
  private setupErrorTracking(): void {
    window.addEventListener('error', (event) => {
      if (this.config.reportErrors) {
        this.recordError({
          message: event.message,
          filename: event.filename,
          line: event.lineno,
          column: event.colno,
          error: event.error,
          timestamp: Date.now(),
          type: 'js',
        });
      }
    });

    window.addEventListener('unhandledrejection', (event) => {
      if (this.config.reportErrors) {
        this.recordError({
          message: 'Unhandled Promise Rejection',
          error: event.reason as Error,
          timestamp: Date.now(),
          type: 'js',
        });
      }
    });
  }

  // Setup long task monitoring
  private setupLongTaskMonitoring(): void {
    try {
      const longTaskObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.duration > 50) { // Tasks longer than 50ms
            this.emit('longtask', {
              duration: entry.duration,
              startTime: entry.startTime,
              name: entry.name,
            });
          }
        }
      });

      longTaskObserver.observe({ entryTypes: ['longtask'] });
      this.observers.push(longTaskObserver);
    } catch (error) {
      console.warn('Long Task API not supported');
    }
  }

  // Setup user timing
  private setupUserTiming(): void {
    try {
      const userTimingObserver = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          this.metrics.custom[entry.name] = entry.duration;
        }
      });

      userTimingObserver.observe({ entryTypes: ['measure'] });
      this.observers.push(userTimingObserver);
    } catch (error) {
      console.warn('User timing API not supported');
    }
  }

  // FPS monitoring
  private startFPSMonitoring(): void {
    const measureFPS = () => {
      const now = performance.now();

      if (this.lastFrameTime) {
        const fps = 1000 / (now - this.lastFrameTime);
        this.metrics.runtime.frameRate = fps;
        this.frameCount++;

        if (this.frameCount % 60 === 0) {
          this.checkThreshold('frameRate', fps);
        }
      }

      this.lastFrameTime = now;
      requestAnimationFrame(measureFPS);
    };

    requestAnimationFrame(measureFPS);
  }

  // CPU usage estimation
  private estimateCPUUsage(): void {
    const measureCPU = () => {
      const start = performance.now();

      // Perform a simple calculation
      for (let i = 0; i < 1000000; i++) {
        Math.random() * Math.random();
      }

      const duration = performance.now() - start;
      this.metrics.runtime.cpuUsage = duration;

      setTimeout(measureCPU, 1000);
    };

    setTimeout(measureCPU, 1000);
  }

  // Calculate Time to Interactive
  private calculateTimeToInteractive(): void {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    if (navigation) {
      this.updateMetric('timeToInteractive', navigation.domInteractive - navigation.fetchStart);
    }
  }

  // Update metric
  updateMetric(name: string, value: number): void {
    const path = name.split('.');
    let current: any = this.metrics;

    for (let i = 0; i < path.length - 1; i++) {
      if (!current[path[i]]) {
        current[path[i]] = {};
      }
      current = current[path[i]];
    }

    current[path[path.length - 1]] = value;
    this.emit('metricUpdate', { name, value });
  }

  // Check performance threshold
  private checkThreshold(metricName: string, value: number): void {
    const threshold = this.config.performanceThresholds[metricName as keyof typeof this.config.performanceThresholds];
    if (threshold && value > threshold) {
      this.emit('thresholdExceeded', {
        metric: metricName,
        value,
        threshold,
        severity: this.getSeverity(value, threshold),
      });
    }
  }

  // Get severity level
  private getSeverity(value: number, threshold: number): 'warning' | 'critical' {
    return value > threshold * 1.5 ? 'critical' : 'warning';
  }

  // Record error
  recordError(error: PerformanceError): void {
    this.emit('error', error);

    if (this.config.consoleMetrics) {
      console.error('Performance Error:', error);
    }
  }

  // Start custom measurement
  startMeasurement(label: string): void {
    this.measurements.set(label, performance.now());
  }

  // End custom measurement
  endMeasurement(label: string): number {
    const start = this.measurements.get(label);
    if (!start) {
      console.warn(`No measurement found for label: ${label}`);
      return 0;
    }

    const duration = performance.now() - start;
    this.measurements.delete(label);
    this.updateMetric(`custom.${label}`, duration);

    return duration;
  }

  // Get current metrics
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  // Get performance score
  getPerformanceScore(): number {
    const scores = [
      this.getFCPScore(),
      this.getLCPScore(),
      this.getCLSScore(),
      this.getFIDScore(),
      this.getTTIScore(),
      this.getFPSScore(),
    ];

    return Math.round(scores.reduce((sum, score) => sum + score, 0) / scores.length);
  }

  // Individual score calculations
  private getFCPScore(): number {
    const fcp = this.metrics.coreWebVitals.firstContentfulPaint;
    if (fcp <= 1800) return 100;
    if (fcp <= 3000) return 50;
    return 0;
  }

  private getLCPScore(): number {
    const lcp = this.metrics.coreWebVitals.largestContentfulPaint;
    if (lcp <= 2500) return 100;
    if (lcp <= 4000) return 50;
    return 0;
  }

  private getCLSScore(): number {
    const cls = this.metrics.coreWebVitals.cumulativeLayoutShift;
    if (cls <= 0.1) return 100;
    if (cls <= 0.25) return 50;
    return 0;
  }

  private getFIDScore(): number {
    const fid = this.metrics.coreWebVitals.firstInputDelay;
    if (fid <= 100) return 100;
    if (fid <= 300) return 50;
    return 0;
  }

  private getTTIScore(): number {
    const tti = this.metrics.coreWebVitals.timeToInteractive;
    if (tti <= 3800) return 100;
    if (tti <= 7300) return 50;
    return 0;
  }

  private getFPSScore(): number {
    const fps = this.metrics.runtime.frameRate;
    if (fps >= 55) return 100;
    if (fps >= 30) return 50;
    return 0;
  }

  // Report metrics
  private reportMetrics(): void {
    const report = {
      timestamp: Date.now(),
      metrics: this.getMetrics(),
      score: this.getPerformanceScore(),
      userAgent: navigator.userAgent,
      url: window.location.href,
    };

    this.emit('report', report);

    if (this.config.consoleMetrics) {
      console.log('📊 Performance Report:', report);
    }
  }

  // Get recommendations
  getRecommendations(): string[] {
    const recommendations: string[] = [];
    const metrics = this.metrics;

    if (metrics.coreWebVitals.firstContentfulPaint > this.config.performanceThresholds.firstContentfulPaint) {
      recommendations.push('Optimize critical rendering path for faster FCP');
    }

    if (metrics.coreWebVitals.largestContentfulPaint > this.config.performanceThresholds.largestContentfulPaint) {
      recommendations.push('Optimize images and LCP elements');
    }

    if (metrics.coreWebVitals.cumulativeLayoutShift > this.config.performanceThresholds.cumulativeLayoutShift) {
      recommendations.push('Add size attributes to images and reserve space for dynamic content');
    }

    if (metrics.coreWebVitals.firstInputDelay > this.config.performanceThresholds.firstInputDelay) {
      recommendations.push('Reduce JavaScript execution time and break up long tasks');
    }

    if (metrics.runtime.frameRate < this.config.performanceThresholds.frameRate) {
      recommendations.push('Optimize animations and reduce main thread work');
    }

    if (metrics.runtime.memoryUsage > this.config.performanceThresholds.memoryUsage) {
      recommendations.push('Check for memory leaks and optimize memory usage');
    }

    return recommendations;
  }
}

// React hook for performance monitoring
export function usePerformanceMonitor(config?: Partial<MonitorConfig>) {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [score, setScore] = useState<number>(0);
  const monitor = React.useMemo(() => new PerformanceMonitor(config), [config]);

  React.useEffect(() => {
    monitor.startMonitoring();

    const handleMetricsUpdate = () => {
      setMetrics(monitor.getMetrics());
      setScore(monitor.getPerformanceScore());
    };

    monitor.on('metricUpdate', handleMetricsUpdate);
    handleMetricsUpdate();

    return () => {
      monitor.stopMonitoring();
      monitor.removeAllListeners();
    };
  }, [monitor]);

  return { metrics, score, monitor };
}

export default PerformanceMonitor;