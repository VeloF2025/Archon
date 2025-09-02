/**
 * DeepConf Dashboard Page
 * Real-time confidence scoring and SCWT metrics visualization
 * CACHE BUSTING UPDATE: 2025-09-01-17:00 - Added emergency diagnostics
 * 
 * Integrates with Phase 5 DeepConf backend services
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { SCWTDashboard } from '../components/deepconf/SCWTDashboard';
import { 
  deepconfServiceFixed as deepconfService, 
  RealTimeData, 
  DeepConfError,
  createDeepConfServiceFixed as createDeepConfService 
} from '../services/deepconfService_fixed';
import { runDeepConfDiagnostic } from '../services/deepconfServiceTest';
import { 
  DEFAULT_DASHBOARD_CONFIG
} from '../components/deepconf';
import { 
  DashboardConfig,
  MetricType 
} from '../components/deepconf/types';

/**
 * DeepConf Dashboard Page Component
 */
export const DeepConfPage: React.FC = () => {
  // State management
  const [dashboardData, setDashboardData] = useState<RealTimeData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<DeepConfError | null>(null);
  const [config, setConfig] = useState<DashboardConfig>(DEFAULT_DASHBOARD_CONFIG);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  
  // Refs for cleanup
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isInitializedRef = useRef(false);

  /**
   * Initialize DeepConf service and load data
   * EMERGENCY FIX: Added comprehensive diagnostics and fallback mechanisms
   */
  const initializeDeepConf = useCallback(async () => {
    try {
      console.log('DeepConf Page: Starting initialization...');
      
      // EMERGENCY DIAGNOSTIC: Run comprehensive service check
      runDeepConfDiagnostic();
      
      setIsLoading(true);
      setError(null);

      // Test service creation - if primary fails, try creating fresh instance
      let serviceToUse = deepconfService;
      try {
        const connectionStatus = serviceToUse.getConnectionStatus();
        console.log('DeepConf Page: Primary service connection status:', connectionStatus);
      } catch (serviceError) {
        console.error('DeepConf Page: Primary service failed, creating fresh instance:', serviceError);
        serviceToUse = createDeepConfService();
      }

      // Check service health first
      const health = await serviceToUse.checkHealth();
      if (health.status === 'error') {
        console.warn('DeepConf: Service health check failed, using fallback data');
      } else {
        console.log('DeepConf: Service health check passed:', health);
      }

      // Load initial dashboard data
      const data = await deepconfService.getDashboardData();
      setDashboardData(data);
      setLastRefresh(new Date());

      // Setup real-time listeners
      setupRealTimeUpdates();

      console.log('DeepConf: Dashboard initialized successfully');
    } catch (err) {
      console.error('DeepConf: Failed to initialize:', err);
      setError(err instanceof DeepConfError ? err : new DeepConfError(`Failed to initialize: ${err}`));
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Setup real-time Socket.IO listeners for live data updates
   */
  const setupRealTimeUpdates = useCallback(() => {
    // Listen for confidence updates
    deepconfService.subscribe('confidence_update', (data) => {
      setDashboardData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          confidence: data.confidence,
          lastUpdate: new Date()
        };
      });
    });

    // Listen for SCWT metrics updates
    deepconfService.subscribe('scwt_metrics_update', (data) => {
      setDashboardData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          current: {
            structuralWeight: data.structural_weight || 0,
            contextWeight: data.context_weight || 0,
            temporalWeight: data.temporal_weight || 0,
            combinedScore: data.combined_score || 0,
            confidence: data.confidence,
            timestamp: new Date(data.timestamp),
            task_id: data.task_id,
            agent_id: data.agent_id,
          },
          history: prev.history ? [...prev.history, prev.current] : [prev.current],
          lastUpdate: new Date()
        };
      });
    });

    // Listen for task-specific confidence updates
    deepconfService.subscribe('task_confidence_update', (data) => {
      console.log('DeepConf: Task confidence update:', data);
      // Update dashboard with task-specific confidence data
      setDashboardData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          confidence: {
            ...prev.confidence,
            overall: data.confidence || prev.confidence.overall
          },
          lastUpdate: new Date()
        };
      });
    });

    // Listen for performance updates
    deepconfService.subscribe('performance_update', (data) => {
      console.log('DeepConf: Performance update:', data);
      setDashboardData(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          performance: data,
          lastUpdate: new Date()
        };
      });
    });

    console.log('DeepConf: Real-time listeners setup complete');
  }, []);

  /**
   * Manual refresh of dashboard data
   */
  const handleRefresh = useCallback(async () => {
    try {
      setError(null);
      const data = await deepconfService.getDashboardData();
      setDashboardData(data);
      setLastRefresh(new Date());
      console.log('DeepConf: Dashboard data refreshed');
    } catch (err) {
      console.error('DeepConf: Failed to refresh:', err);
      setError(err instanceof DeepConfError ? err : new DeepConfError(`Failed to refresh: ${err}`));
    }
  }, []);

  /**
   * Handle metric selection changes
   */
  const handleMetricSelect = useCallback((metric: MetricType) => {
    console.log('DeepConf: Metric selected:', metric);
    // Future: Could trigger specific metric streams or updates
  }, []);

  /**
   * Handle data export
   */
  const handleExportData = useCallback(() => {
    if (!dashboardData) return;

    try {
      const exportData = {
        timestamp: new Date().toISOString(),
        current_metrics: dashboardData.current,
        confidence: dashboardData.confidence,
        performance: dashboardData.performance,
        history: dashboardData.history,
        status: dashboardData.status
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

      console.log('DeepConf: Data exported successfully');
    } catch (err) {
      console.error('DeepConf: Failed to export data:', err);
    }
  }, [dashboardData]);

  /**
   * Setup automatic refresh interval
   */
  useEffect(() => {
    if (config.autoRefresh && dashboardData && !error) {
      refreshIntervalRef.current = setInterval(() => {
        handleRefresh();
      }, config.refreshInterval);

      return () => {
        if (refreshIntervalRef.current) {
          clearInterval(refreshIntervalRef.current);
        }
      };
    }
  }, [config.autoRefresh, config.refreshInterval, dashboardData, error, handleRefresh]);

  /**
   * Initialize component
   */
  useEffect(() => {
    if (!isInitializedRef.current) {
      isInitializedRef.current = true;
      initializeDeepConf();
    }

    // Cleanup on unmount
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [initializeDeepConf]);

  /**
   * Connection status indicator
   */
  const ConnectionStatus: React.FC = () => {
    const isConnected = deepconfService.getConnectionStatus();
    
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium shadow-lg backdrop-blur-sm ${
          isConnected 
            ? 'bg-green-100/90 text-green-800 border border-green-200 dark:bg-green-900/90 dark:text-green-200 dark:border-green-800'
            : 'bg-orange-100/90 text-orange-800 border border-orange-200 dark:bg-orange-900/90 dark:text-orange-200 dark:border-orange-800'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500 animate-pulse' : 'bg-orange-500'
          }`} />
          {isConnected ? 'Live Updates Active' : 'Offline Mode'}
        </div>
      </div>
    );
  };

  /**
   * Render the page
   */
  return (
    <div className="min-h-screen bg-background">
      {/* Page Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-b border-gray-200 dark:border-gray-800">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                DeepConf Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Real-time confidence scoring and SCWT metrics
              </p>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
              <span>Service Status:</span>
              <span className={`px-2 py-1 rounded text-xs font-medium ${
                deepconfService.getConnectionStatus()
                  ? 'bg-green-100 text-green-800 dark:bg-green-800/20 dark:text-green-300'
                  : 'bg-orange-100 text-orange-800 dark:bg-orange-800/20 dark:text-orange-300'
              }`}>
                {deepconfService.getConnectionStatus() ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Content */}
      <div className="flex-1">
        <SCWTDashboard
          data={dashboardData}
          config={config}
          onMetricSelect={handleMetricSelect}
          onExportData={handleExportData}
          onRefresh={handleRefresh}
          isLoading={isLoading}
          error={error}
        />
      </div>

      {/* Connection Status Indicator */}
      <ConnectionStatus />
    </div>
  );
};

export default DeepConfPage;