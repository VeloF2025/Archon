/**
 * Optimized DeepConf Dashboard Page
 * 
 * Performance Optimizations:
 * - React Query caching for API responses
 * - Batched real-time updates
 * - Memoized components and calculations
 * - Lazy loading for heavy components
 * - Memory management and cleanup
 */

import React, { useState, useCallback, useMemo, Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { useOptimizedDeepConf, usePrefetchDeepConfData } from '../hooks/useOptimizedDeepConf';
import { 
  DEFAULT_DASHBOARD_CONFIG
} from '../components/deepconf';
import { 
  DashboardConfig,
  MetricType 
} from '../components/deepconf/types';

// Lazy load heavy components for code splitting
const SCWTDashboard = React.lazy(() => 
  import('../components/deepconf/SCWTDashboard').then(module => ({
    default: module.SCWTDashboard
  }))
);

// Loading fallback component
const DashboardSkeleton: React.FC = React.memo(() => (
  <div className="p-6 space-y-6 animate-pulse">
    <div className="h-8 bg-gray-300 rounded mb-4"></div>
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {[1, 2, 3, 4, 5, 6].map(i => (
        <div key={i} className="h-64 bg-gray-300 rounded"></div>
      ))}
    </div>
  </div>
));

// Error fallback component
const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = React.memo(({ 
  error, 
  resetErrorBoundary 
}) => (
  <div className="p-6">
    <div className="bg-red-50 border border-red-200 rounded-lg p-6">
      <h2 className="text-lg font-semibold text-red-800 mb-2">
        Dashboard Error
      </h2>
      <p className="text-red-600 mb-4">
        {error.message}
      </p>
      <button
        onClick={resetErrorBoundary}
        className="bg-red-100 hover:bg-red-200 text-red-800 px-4 py-2 rounded transition-colors"
      >
        Retry Dashboard
      </button>
    </div>
  </div>
));

// Connection status indicator (optimized)
const ConnectionStatus: React.FC<{ isConnected: boolean }> = React.memo(({ isConnected }) => (
  <div className="fixed bottom-4 right-4 z-50">
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium shadow-lg backdrop-blur-sm transition-colors ${
      isConnected 
        ? 'bg-green-100/90 text-green-800 border border-green-200 dark:bg-green-900/90 dark:text-green-200 dark:border-green-800'
        : 'bg-orange-100/90 text-orange-800 border border-orange-200 dark:bg-orange-900/90 dark:text-orange-200 dark:border-orange-800'
    }`}>
      <div className={`w-2 h-2 rounded-full transition-colors ${
        isConnected ? 'bg-green-500 animate-pulse' : 'bg-orange-500'
      }`} />
      {isConnected ? 'Live Updates Active' : 'Offline Mode'}
    </div>
  </div>
));

/**
 * Optimized DeepConf Dashboard Page Component
 */
export const OptimizedDeepConfPage: React.FC = () => {
  // Use optimized hook with caching
  const {
    data: dashboardData,
    isLoading,
    error,
    isError,
    refresh,
    isRefreshing,
    isConnected,
    performanceMetrics
  } = useOptimizedDeepConf();

  // State management (memoized for performance)
  const [config, setConfig] = useState<DashboardConfig>(() => DEFAULT_DASHBOARD_CONFIG);
  
  // Prefetch utility
  const prefetchData = usePrefetchDeepConfData();
  
  // Memoized event handlers to prevent unnecessary re-renders
  const handleMetricSelect = useCallback((metric: MetricType) => {
    console.log('DeepConf: Metric selected:', metric);
    // Future: Could trigger specific metric streams or updates
  }, []);
  
  const handleExportData = useCallback(() => {
    if (!dashboardData) return;
    
    try {
      const exportData = {
        timestamp: new Date().toISOString(),
        current_metrics: dashboardData.current,
        confidence: dashboardData.confidence,
        performance: dashboardData.performance,
        history: dashboardData.history,
        status: dashboardData.status,
        performance_meta: performanceMetrics
      };
      
      // Create and download JSON file
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `deepconf-metrics-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      console.log('DeepConf: Optimized data exported successfully');
    } catch (err) {
      console.error('DeepConf: Failed to export data:', err);
    }
  }, [dashboardData, performanceMetrics]);
  
  const handleRefresh = useCallback(async () => {
    try {
      // Prefetch data before refreshing for smoother UX
      prefetchData();
      await refresh();
      console.log('DeepConf: Dashboard data refreshed with optimizations');
    } catch (err) {
      console.error('DeepConf: Failed to refresh:', err);
    }
  }, [refresh, prefetchData]);
  
  // Memoized config change handler
  const handleConfigChange = useCallback((newConfig: Partial<DashboardConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);
  
  // Performance monitoring display (development only)
  const performanceDisplay = useMemo(() => {
    if (process.env.NODE_ENV !== 'development') return null;
    
    return (
      <div className="fixed top-4 right-4 z-50 bg-black/80 text-white text-xs p-2 rounded font-mono">
        <div>Cache Hit: {(performanceMetrics.cacheHitRate * 100).toFixed(1)}%</div>
        <div>Query Time: {performanceMetrics.queryTime || 0}ms</div>
        <div>Updates: {isConnected ? 'Real-time' : 'Cached'}</div>
      </div>
    );
  }, [performanceMetrics, isConnected]);
  
  // Render the page with optimizations
  return (
    <div className="min-h-screen bg-background">
      {/* Performance monitoring (dev only) */}
      {performanceDisplay}
      
      {/* Page Header - Optimized */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-b border-gray-200 dark:border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                DeepConf Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Real-time confidence scoring and SCWT metrics (Optimized)
              </p>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
              <span>Service Status:</span>
              <span className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                isConnected
                  ? 'bg-green-100 text-green-800 dark:bg-green-800/20 dark:text-green-300'
                  : 'bg-orange-100 text-orange-800 dark:bg-orange-800/20 dark:text-orange-300'
              }`}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
              {isRefreshing && (
                <span className="px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-800/20 dark:text-blue-300">
                  Refreshing...
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Dashboard Content with Error Boundary and Suspense */}
      <div className="flex-1">
        <ErrorBoundary
          FallbackComponent={ErrorFallback}
          onReset={() => {
            // Reset any local state and refetch
            handleRefresh();
          }}
          resetKeys={[dashboardData]} // Reset when data changes
        >
          <Suspense fallback={<DashboardSkeleton />}>
            <SCWTDashboard
              data={dashboardData}
              config={config}
              onMetricSelect={handleMetricSelect}
              onExportData={handleExportData}
              onRefresh={handleRefresh}
              isLoading={isLoading}
              error={error}
            />
          </Suspense>
        </ErrorBoundary>
      </div>
      
      {/* Connection Status Indicator */}
      <ConnectionStatus isConnected={isConnected} />
      
      {/* Performance Info Panel (Development) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-20 right-4 bg-gray-900 text-white p-3 rounded-lg text-xs font-mono max-w-xs">
          <div className="font-bold mb-2">Performance Metrics</div>
          <div>Cache Hit Rate: {(performanceMetrics.cacheHitRate * 100).toFixed(1)}%</div>
          <div>Last Update: {performanceMetrics.lastUpdateTime 
            ? new Date(performanceMetrics.lastUpdateTime).toLocaleTimeString()
            : 'Never'}</div>
          <div>Query Time: ~{performanceMetrics.queryTime || 0}ms</div>
          <div>Status: {dashboardData?.status || 'unknown'}</div>
          <div>History Points: {dashboardData?.history?.length || 0}</div>
        </div>
      )}
    </div>
  );
};

export default OptimizedDeepConfPage;