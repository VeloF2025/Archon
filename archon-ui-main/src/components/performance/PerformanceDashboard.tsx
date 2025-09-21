/**
 * Performance Dashboard
 *
 * Real-time performance monitoring and optimization dashboard
 * Shows metrics, recommendations, and allows optimization control
 */

import React, { useState, useEffect } from 'react';
import {
  PerformanceOptimizer,
  usePerformanceMonitor,
  useBundleAnalyzer,
  performanceHelpers,
  performanceConfigs,
} from '@/performance';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/Card';
import {
  Button,
  Badge,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui';
import {
  Activity,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Settings,
  Download,
} from 'lucide-react';

interface PerformanceDashboardProps {
  config?: any;
  className?: string;
}

export const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  config = performanceConfigs.production,
  className = '',
}) => {
  const [optimizer] = useState(() => new PerformanceOptimizer(config));
  const { metrics, score } = usePerformanceMonitor();
  const { analysis, health } = useBundleAnalyzer();
  const [activeTab, setActiveTab] = useState('overview');
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);

  useEffect(() => {
    if (metrics) {
      setRecommendations(optimizer.getOptimizationRecommendations());
    }
  }, [metrics, optimizer]);

  const handleOptimize = async () => {
    setIsOptimizing(true);
    try {
      await optimizer.autoOptimize();
      setRecommendations(optimizer.getOptimizationRecommendations());
    } catch (error) {
      console.error('Optimization failed:', error);
    } finally {
      setIsOptimizing(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-green-600 dark:text-green-400';
    if (score >= 70) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 90) return 'Excellent';
    if (score >= 70) return 'Good';
    if (score >= 50) return 'Fair';
    return 'Poor';
  };

  const formatMetric = (value: number, unit: string) => {
    if (unit === 'ms') {
      return value < 1000 ? `${value.toFixed(0)}ms` : `${(value / 1000).toFixed(2)}s`;
    }
    if (unit === 'kb') {
      return `${(value / 1024).toFixed(1)}KB`;
    }
    if (unit === 'mb') {
      return `${(value / 1024 / 1024).toFixed(1)}MB`;
    }
    return value.toFixed(0);
  };

  return (
    <div className={`performance-dashboard ${className}`}>
      <div className="space-y-6">
        {/* Header */}
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <div>
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Activity className="w-6 h-6" />
                  Performance Dashboard
                </CardTitle>
                <p className="text-muted-foreground mt-1">
                  Real-time monitoring and optimization for Archon UI
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.location.reload()}
                >
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
                <Button
                  onClick={handleOptimize}
                  disabled={isOptimizing}
                  size="sm"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  {isOptimizing ? 'Optimizing...' : 'Optimize'}
                </Button>
              </div>
            </div>
          </CardHeader>
        </Card>

        {/* Performance Score */}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Performance Score</h3>
                <p className="text-muted-foreground">Overall system performance</p>
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${getScoreColor(score)}`}>
                  {score}
                </div>
                <div className="text-sm text-muted-foreground">
                  {getScoreLabel(score)}
                </div>
              </div>
            </div>
            <Progress value={score} className="mt-4 h-3" />
            <div className="flex justify-between text-sm text-muted-foreground mt-2">
              <span>0</span>
              <span>50</span>
              <span>100</span>
            </div>
          </CardContent>
        </Card>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
            <TabsTrigger value="bundle">Bundle</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Core Web Vitals */}
              {metrics && (
                <>
                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">FCP</span>
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                      <div className="text-2xl font-bold">
                        {formatMetric(metrics.coreWebVitals.firstContentfulPaint, 'ms')}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        First Contentful Paint
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">LCP</span>
                        {metrics.coreWebVitals.largestContentfulPaint > 2500 ? (
                          <AlertTriangle className="w-4 h-4 text-yellow-500" />
                        ) : (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        )}
                      </div>
                      <div className="text-2xl font-bold">
                        {formatMetric(metrics.coreWebVitals.largestContentfulPaint, 'ms')}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Largest Contentful Paint
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">CLS</span>
                        {metrics.coreWebVitals.cumulativeLayoutShift > 0.1 ? (
                          <AlertTriangle className="w-4 h-4 text-yellow-500" />
                        ) : (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        )}
                      </div>
                      <div className="text-2xl font-bold">
                        {metrics.coreWebVitals.cumulativeLayoutShift.toFixed(3)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Cumulative Layout Shift
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">FID</span>
                        {metrics.coreWebVitals.firstInputDelay > 100 ? (
                          <AlertTriangle className="w-4 h-4 text-yellow-500" />
                        ) : (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        )}
                      </div>
                      <div className="text-2xl font-bold">
                        {formatMetric(metrics.coreWebVitals.firstInputDelay, 'ms')}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        First Input Delay
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}
            </div>

            {/* Health Status */}
            {health && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    System Health
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <Badge variant={health.status === 'excellent' ? 'default' : 'secondary'}>
                      {health.status.toUpperCase()}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      Score: {health.score}/100
                    </span>
                  </div>
                  {health.issues.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-medium mb-2">Issues:</h4>
                      <ul className="text-sm text-muted-foreground space-y-1">
                        {health.issues.map((issue, index) => (
                          <li key={index} className="flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-yellow-500" />
                            {issue}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-4">
            {metrics && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Loading Metrics */}
                <Card>
                  <CardHeader>
                    <CardTitle>Loading Performance</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>First Contentful Paint</span>
                      <span className="font-medium">
                        {formatMetric(metrics.coreWebVitals.firstContentfulPaint, 'ms')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Largest Contentful Paint</span>
                      <span className="font-medium">
                        {formatMetric(metrics.coreWebVitals.largestContentfulPaint, 'ms')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Time to Interactive</span>
                      <span className="font-medium">
                        {formatMetric(metrics.coreWebVitals.timeToInteractive, 'ms')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>DOM Content Loaded</span>
                      <span className="font-medium">
                        {formatMetric(metrics.coreWebVitals.firstContentfulPaint, 'ms')}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Runtime Metrics */}
                <Card>
                  <CardHeader>
                    <CardTitle>Runtime Performance</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Frame Rate</span>
                      <span className="font-medium">
                        {metrics.runtime.frameRate.toFixed(1)} FPS
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Memory Usage</span>
                      <span className="font-medium">
                        {formatMetric(metrics.runtime.memoryUsage, 'mb')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>CPU Usage</span>
                      <span className="font-medium">
                        {metrics.runtime.cpuUsage.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Response Time</span>
                      <span className="font-medium">
                        {formatMetric(metrics.runtime.responseTime, 'ms')}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Resource Metrics */}
                <Card>
                  <CardHeader>
                    <CardTitle>Resource Usage</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Bundle Size</span>
                      <span className="font-medium">
                        {formatMetric(metrics.resources.bundleSize, 'kb')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Request Count</span>
                      <span className="font-medium">
                        {metrics.resources.requestCount}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Cache Hit Rate</span>
                      <span className="font-medium">
                        {(metrics.resources.cacheHitRate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Lazy Load Count</span>
                      <span className="font-medium">
                        {metrics.resources.lazyLoadCount}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* User Experience */}
                <Card>
                  <CardHeader>
                    <CardTitle>User Experience</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Cumulative Layout Shift</span>
                      <span className="font-medium">
                        {metrics.ux.cumulativeLayoutShift.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>First Input Delay</span>
                      <span className="font-medium">
                        {formatMetric(metrics.ux.firstInputDelay, 'ms')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Interaction Time</span>
                      <span className="font-medium">
                        {formatMetric(metrics.ux.interactionTime, 'ms')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Interaction Count</span>
                      <span className="font-medium">
                        {metrics.interactions.interactionCount}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Bundle Tab */}
          <TabsContent value="bundle" className="space-y-4">
            {analysis && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Bundle Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle>Bundle Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between">
                      <span>Total Size</span>
                      <span className="font-medium">
                        {formatMetric(analysis.totalSize, 'kb')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Initial Size</span>
                      <span className="font-medium">
                        {formatMetric(analysis.initialSize, 'kb')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Cache Size</span>
                      <span className="font-medium">
                        {formatMetric(analysis.cacheSize, 'kb')}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Total Chunks</span>
                      <span className="font-medium">
                        {analysis.totalChunks}
                      </span>
                    </div>
                  </CardContent>
                </Card>

                {/* Bundle Health */}
                <Card>
                  <CardHeader>
                    <CardTitle>Bundle Health</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span>Status</span>
                        <Badge variant={health?.status === 'excellent' ? 'default' : 'secondary'}>
                          {health?.status.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Score</span>
                        <span className="font-medium">{health?.score}/100</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Duplicates</span>
                        <span className="font-medium">
                          {analysis.duplicates.length} found
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Recommendations</span>
                        <span className="font-medium">
                          {analysis.recommendations.length}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Large Chunks */}
                <Card>
                  <CardHeader>
                    <CardTitle>Large Chunks</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {analysis.chunks
                        .filter(chunk => chunk.size > 50000)
                        .slice(0, 5)
                        .map((chunk, index) => (
                          <div key={index} className="flex justify-between text-sm">
                            <span className="truncate">{chunk.name}</span>
                            <span className="font-medium">
                              {formatMetric(chunk.size, 'kb')}
                            </span>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Duplicate Modules */}
                <Card>
                  <CardHeader>
                    <CardTitle>Duplicate Modules</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {analysis.duplicates.slice(0, 5).map((duplicate, index) => (
                        <div key={index} className="space-y-1">
                          <div className="text-sm font-medium">{duplicate.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {duplicate.occurrences.length} occurrences
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Recommendations Tab */}
          <TabsContent value="recommendations" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Optimization Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {recommendations.length > 0 ? (
                    <div className="space-y-3">
                      {recommendations.map((recommendation, index) => (
                        <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                          <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5" />
                          <div className="flex-1">
                            <p className="text-sm">{recommendation}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
                      <p className="text-lg font-medium">Great job!</p>
                      <p className="text-muted-foreground">
                        No optimization recommendations found
                      </p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default PerformanceDashboard;