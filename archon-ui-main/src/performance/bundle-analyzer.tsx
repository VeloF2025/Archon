/**
 * Bundle Analyzer
 *
 * Advanced bundle analysis with:
 * - Bundle size optimization
 * - Code splitting analysis
 * - Dependency graph visualization
 * - Performance recommendations
 * - Real-time monitoring
 */

// Bundle analysis result
export interface BundleAnalysis {
  totalSize: number;
  totalChunks: number;
  initialSize: number;
  cacheSize: number;
  chunks: BundleChunk[];
  modules: BundleModule[];
  assets: BundleAsset[];
  duplicates: DuplicateModule[];
  recommendations: string[];
  metrics: BundleMetrics;
}

// Bundle chunk
export interface BundleChunk {
  id: string;
  name: string;
  size: number;
  initial: boolean;
  entry: boolean;
  modules: string[];
  reasons: string[];
}

// Bundle module
export interface BundleModule {
  id: string;
  name: string;
  size: number;
  chunks: string[];
  reasons: string[];
  depth: number;
}

// Bundle asset
export interface BundleAsset {
  name: string;
  size: number;
  chunks: string[];
  emitted: boolean;
}

// Duplicate module
export interface DuplicateModule {
  name: string;
  occurrences: string[];
  totalSize: number;
}

// Bundle metrics
export interface BundleMetrics {
  timeToFirstByte: number;
  firstContentfulPaint: number;
  largestContentfulPaint: number;
  cumulativeLayoutShift: number;
  firstInputDelay: number;
  bundleLoadTime: number;
  cacheHitRate: number;
}

// Bundle analyzer configuration
export interface BundleAnalyzerConfig {
  enableRealTimeAnalysis: boolean;
  enableRecommendations: boolean;
  enableVisualization: boolean;
  maxSizeThreshold: number;
  maxChunkThreshold: number;
  duplicateThreshold: number;
  analysisInterval: number;
}

// Bundle Analyzer class
export class BundleAnalyzer {
  private config: BundleAnalyzerConfig;
  private analysis: BundleAnalysis | null = null;
  private observer: MutationObserver | null = null;
  private analysisInterval: NodeJS.Timeout | null = null;

  constructor(config: Partial<BundleAnalyzerConfig> = {}) {
    this.config = {
      enableRealTimeAnalysis: true,
      enableRecommendations: true,
      enableVisualization: true,
      maxSizeThreshold: 244000, // 244KB (gzip)
      maxChunkThreshold: 50000, // 50KB
      duplicateThreshold: 3, // 3 occurrences
      analysisInterval: 30000, // 30 seconds
      ...config,
    };

    this.initialize();
  }

  private initialize(): void {
    if (this.config.enableRealTimeAnalysis) {
      this.setupRealTimeAnalysis();
    }

    this.analyzeBundle();
  }

  // Setup real-time analysis
  private setupRealTimeAnalysis(): void {
    // Monitor script and link tags
    this.observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            const element = node as Element;
            if (element.tagName === 'SCRIPT' || element.tagName === 'LINK') {
              this.analyzeBundle();
            }
          }
        });
      });
    });

    this.observer.observe(document.head, {
      childList: true,
      subtree: true,
    });

    // Set up interval analysis
    this.analysisInterval = setInterval(() => {
      this.analyzeBundle();
    }, this.config.analysisInterval);
  }

  // Analyze bundle
  async analyzeBundle(): Promise<BundleAnalysis> {
    const analysis: BundleAnalysis = {
      totalSize: 0,
      totalChunks: 0,
      initialSize: 0,
      cacheSize: 0,
      chunks: [],
      modules: [],
      assets: [],
      duplicates: [],
      recommendations: [],
      metrics: this.calculateMetrics(),
    };

    // Get performance entries
    const resources = performance.getEntriesByType('resource');
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;

    // Analyze resources
    const chunks = this.analyzeChunks(resources);
    const modules = this.analyzeModules(resources);
    const assets = this.analyzeAssets(resources);
    const duplicates = this.findDuplicates(modules);

    analysis.chunks = chunks;
    analysis.modules = modules;
    analysis.assets = assets;
    analysis.duplicates = duplicates;

    // Calculate sizes
    analysis.totalSize = chunks.reduce((sum, chunk) => sum + chunk.size, 0);
    analysis.totalChunks = chunks.length;
    analysis.initialSize = chunks.filter(chunk => chunk.initial).reduce((sum, chunk) => sum + chunk.size, 0);
    analysis.cacheSize = chunks.filter(chunk => !chunk.initial).reduce((sum, chunk) => sum + chunk.size, 0);

    // Generate recommendations
    if (this.config.enableRecommendations) {
      analysis.recommendations = this.generateRecommendations(analysis);
    }

    this.analysis = analysis;
    return analysis;
  }

  // Analyze chunks
  private analyzeChunks(resources: PerformanceResourceTiming[]): BundleChunk[] {
    const chunks: BundleChunk[] = [];
    const processedUrls = new Set<string>();

    resources.forEach((resource) => {
      if (processedUrls.has(resource.name)) return;
      processedUrls.add(resource.name);

      const url = new URL(resource.name);
      const isJS = url.pathname.endsWith('.js');
      const isCSS = url.pathname.endsWith('.css');

      if (isJS || isCSS) {
        const chunk: BundleChunk = {
          id: url.pathname.split('/').pop() || 'unknown',
          name: url.pathname.split('/').pop() || 'unknown',
          size: (resource as any).transferSize || resource.duration,
          initial: this.isInitialChunk(url.pathname),
          entry: this.isEntryChunk(url.pathname),
          modules: [],
          reasons: [],
        };

        chunks.push(chunk);
      }
    });

    return chunks;
  }

  // Analyze modules
  private analyzeModules(resources: PerformanceResourceTiming[]): BundleModule[] {
    const modules: BundleModule[] = [];

    // This is a simplified implementation
    // In a real implementation, you would use webpack stats or similar
    resources.forEach((resource) => {
      const url = new URL(resource.name);
      const isJS = url.pathname.endsWith('.js');

      if (isJS) {
        const module: BundleModule = {
          id: url.pathname,
          name: url.pathname.split('/').pop() || 'unknown',
          size: (resource as any).transferSize || resource.duration,
          chunks: [url.pathname],
          reasons: [],
          depth: 0,
        };

        modules.push(module);
      }
    });

    return modules;
  }

  // Analyze assets
  private analyzeAssets(resources: PerformanceResourceTiming[]): BundleAsset[] {
    const assets: BundleAsset[] = [];

    resources.forEach((resource) => {
      const url = new URL(resource.name);
      const asset: BundleAsset = {
        name: url.pathname.split('/').pop() || 'unknown',
        size: (resource as any).transferSize || resource.duration,
        chunks: [url.pathname],
        emitted: true,
      };

      assets.push(asset);
    });

    return assets;
  }

  // Find duplicate modules
  private findDuplicates(modules: BundleModule[]): DuplicateModule[] {
    const duplicates: DuplicateModule[] = [];
    const moduleMap = new Map<string, BundleModule[]>();

    modules.forEach((module) => {
      const key = module.name;
      if (!moduleMap.has(key)) {
        moduleMap.set(key, []);
      }
      moduleMap.get(key)!.push(module);
    });

    moduleMap.forEach((occurrences, name) => {
      if (occurrences.length > this.config.duplicateThreshold) {
        duplicates.push({
          name,
          occurrences: occurrences.map(m => m.name),
          totalSize: occurrences.reduce((sum, m) => sum + m.size, 0),
        });
      }
    });

    return duplicates;
  }

  // Check if initial chunk
  private isInitialChunk(pathname: string): boolean {
    return pathname.includes('main') || pathname.includes('index') || pathname.includes('vendor');
  }

  // Check if entry chunk
  private isEntryChunk(pathname: string): boolean {
    return pathname.includes('main') || pathname.includes('index');
  }

  // Calculate metrics
  private calculateMetrics(): BundleMetrics {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');

    return {
      timeToFirstByte: navigation ? navigation.responseStart - navigation.requestStart : 0,
      firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
      largestContentfulPaint: 0, // Would need LCP observer
      cumulativeLayoutShift: 0, // Would need CLS observer
      firstInputDelay: 0, // Would need FID observer
      bundleLoadTime: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
      cacheHitRate: this.calculateCacheHitRate(),
    };
  }

  // Calculate cache hit rate
  private calculateCacheHitRate(): number {
    const resources = performance.getEntriesByType('resource');
    const cacheHits = resources.filter(r => (r as any).transferSize === 0).length;
    return resources.length > 0 ? cacheHits / resources.length : 0;
  }

  // Generate recommendations
  private generateRecommendations(analysis: BundleAnalysis): string[] {
    const recommendations: string[] = [];

    // Size-based recommendations
    if (analysis.totalSize > this.config.maxSizeThreshold) {
      recommendations.push('Bundle size exceeds recommended limit. Consider code splitting.');
    }

    if (analysis.initialSize > this.config.maxSizeThreshold * 0.6) {
      recommendations.push('Initial bundle is too large. Consider dynamic imports for non-critical code.');
    }

    // Chunk-based recommendations
    const largeChunks = analysis.chunks.filter(chunk => chunk.size > this.config.maxChunkThreshold);
    if (largeChunks.length > 0) {
      recommendations.push(`Found ${largeChunks.length} large chunks. Consider splitting them further.`);
    }

    // Duplicate recommendations
    if (analysis.duplicates.length > 0) {
      recommendations.push(`Found ${analysis.duplicates.length} duplicate modules. Consider deduplication.`);
    }

    // Cache recommendations
    if (analysis.metrics.cacheHitRate < 0.7) {
      recommendations.push('Low cache hit rate. Consider implementing better caching strategies.');
    }

    // Performance recommendations
    if (analysis.metrics.firstContentfulPaint > 1500) {
      recommendations.push('First Contentful Paint is slow. Consider optimizing critical rendering path.');
    }

    return recommendations;
  }

  // Get analysis result
  getAnalysis(): BundleAnalysis | null {
    return this.analysis;
  }

  // Get optimization score
  getOptimizationScore(): number {
    if (!this.analysis) return 0;

    let score = 100;

    // Size penalty
    if (this.analysis.totalSize > this.config.maxSizeThreshold) {
      score -= 20;
    }

    // Initial size penalty
    if (this.analysis.initialSize > this.config.maxSizeThreshold * 0.6) {
      score -= 15;
    }

    // Duplicate penalty
    if (this.analysis.duplicates.length > 0) {
      score -= 10;
    }

    // Cache rate penalty
    if (this.analysis.metrics.cacheHitRate < 0.7) {
      score -= 10;
    }

    return Math.max(0, score);
  }

  // Get bundle health
  getBundleHealth(): {
    status: 'excellent' | 'good' | 'fair' | 'poor';
    issues: string[];
    score: number;
  } {
    const score = this.getOptimizationScore();
    const issues: string[] = [];

    if (!this.analysis) {
      return { status: 'poor', issues: ['No analysis available'], score: 0 };
    }

    if (this.analysis.totalSize > this.config.maxSizeThreshold) {
      issues.push('Bundle size too large');
    }

    if (this.analysis.initialSize > this.config.maxSizeThreshold * 0.6) {
      issues.push('Initial bundle too large');
    }

    if (this.analysis.duplicates.length > 0) {
      issues.push('Duplicate modules found');
    }

    if (this.analysis.metrics.cacheHitRate < 0.7) {
      issues.push('Low cache hit rate');
    }

    let status: 'excellent' | 'good' | 'fair' | 'poor';
    if (score >= 90) status = 'excellent';
    else if (score >= 70) status = 'good';
    else if (score >= 50) status = 'fair';
    else status = 'poor';

    return { status, issues, score };
  }

  // Optimize bundle
  async optimize(): Promise<void> {
    if (!this.analysis) return;

    const recommendations = this.generateRecommendations(this.analysis);

    for (const recommendation of recommendations) {
      await this.applyOptimization(recommendation);
    }
  }

  // Apply optimization
  private async applyOptimization(recommendation: string): Promise<void> {
    // This is a placeholder for optimization logic
    // In a real implementation, you would:
    // - Trigger code splitting
    // - Remove duplicates
    // - Implement caching strategies
    console.log('Applying optimization:', recommendation);
  }

  // Export analysis
  exportAnalysis(format: 'json' | 'csv' = 'json'): string {
    if (!this.analysis) return '';

    if (format === 'json') {
      return JSON.stringify(this.analysis, null, 2);
    }

    if (format === 'csv') {
      const headers = ['Name', 'Size', 'Type', 'Initial'];
      const rows = this.analysis.chunks.map(chunk => [
        chunk.name,
        chunk.size.toString(),
        'chunk',
        chunk.initial.toString(),
      ]);

      return [headers, ...rows].map(row => row.join(',')).join('\n');
    }

    return '';
  }

  // Clear analysis
  clearAnalysis(): void {
    this.analysis = null;
  }

  // Destroy analyzer
  destroy(): void {
    if (this.observer) {
      this.observer.disconnect();
    }

    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
    }

    this.clearAnalysis();
  }
}

// React hook for bundle analysis
export function useBundleAnalyzer(config?: Partial<BundleAnalyzerConfig>) {
  const [analysis, setAnalysis] = useState<BundleAnalysis | null>(null);
  const [health, setHealth] = useState<{ status: string; issues: string[]; score: number } | null>(null);
  const analyzer = React.useMemo(() => new BundleAnalyzer(config), [config]);

  React.useEffect(() => {
    const interval = setInterval(async () => {
      const result = await analyzer.analyzeBundle();
      setAnalysis(result);
      setHealth(analyzer.getBundleHealth());
    }, 5000);

    // Initial analysis
    analyzer.analyzeBundle().then(result => {
      setAnalysis(result);
      setHealth(analyzer.getBundleHealth());
    });

    return () => {
      clearInterval(interval);
      analyzer.destroy();
    };
  }, [analyzer]);

  return { analysis, health, analyzer };
}

// Bundle analysis visualization component
interface BundleAnalysisVisualizationProps {
  analysis: BundleAnalysis;
  width?: number;
  height?: number;
  className?: string;
}

export const BundleAnalysisVisualization: React.FC<BundleAnalysisVisualizationProps> = ({
  analysis,
  width = 800,
  height = 400,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw bundle visualization
    drawBundleVisualization(ctx, analysis, width, height);
  }, [analysis, width, height]);

  return (
    <div className={`bundle-analysis-visualization ${className}`}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full h-full border rounded"
      />
    </div>
  );
};

// Draw bundle visualization
function drawBundleVisualization(
  ctx: CanvasRenderingContext2D,
  analysis: BundleAnalysis,
  width: number,
  height: number
): void {
  const colors = {
    initial: '#3b82f6',
    cache: '#10b981',
    duplicate: '#f59e0b',
    large: '#ef4444',
  };

  // Draw chunks
  let x = 0;
  const barHeight = height / 2;
  const totalSize = analysis.totalSize;

  analysis.chunks.forEach((chunk) => {
    const chunkWidth = (chunk.size / totalSize) * width;
    let color = colors.cache;

    if (chunk.initial) color = colors.initial;
    if (chunk.size > 50000) color = colors.large;

    ctx.fillStyle = color;
    ctx.fillRect(x, 0, chunkWidth, barHeight);

    // Draw chunk name
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.fillText(chunk.name, x + 5, barHeight / 2 + 4);

    x += chunkWidth;
  });

  // Draw duplicates
  x = 0;
  const duplicateHeight = height / 4;
  const duplicateY = barHeight + 20;

  analysis.duplicates.forEach((duplicate) => {
    const duplicateWidth = (duplicate.totalSize / totalSize) * width;

    ctx.fillStyle = colors.duplicate;
    ctx.fillRect(x, duplicateY, duplicateWidth, duplicateHeight);

    x += duplicateWidth;
  });

  // Draw metrics
  ctx.fillStyle = '#374151';
  ctx.font = '14px sans-serif';
  ctx.fillText(`Total Size: ${(analysis.totalSize / 1024).toFixed(2)} KB`, 10, height - 40);
  ctx.fillText(`Initial: ${(analysis.initialSize / 1024).toFixed(2)} KB`, 10, height - 20);
  ctx.fillText(`Cache: ${(analysis.cacheSize / 1024).toFixed(2)} KB`, 10, height - 5);
}

export default BundleAnalyzer;