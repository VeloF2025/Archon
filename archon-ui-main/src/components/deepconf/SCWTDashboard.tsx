/**
 * SCWT Metrics Dashboard Component
 * Phase 7 DeepConf Integration
 * 
 * Main dashboard component for real-time SCWT metrics visualization
 * Includes performance monitoring, confidence tracking, and interactive debugging
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { Toggle } from '../ui/Toggle';
import { ThemeToggle } from '../ui/ThemeToggle';
import { 
  SCWTDashboardProps, 
  RealTimeData, 
  DashboardConfig, 
  TimeRange,
  TimeRangePresets,
  MetricType,
  SCWTMetrics,
  PerformanceMetrics,
  ConfidenceMetrics
} from './types';
import { ConfidenceVisualization } from './index';
import { PerformanceMetrics as PerformanceComponent } from './PerformanceMetrics';
import { DebugTools } from './DebugTools';
import { RealTimeMonitoring } from './RealTimeMonitoring';

// Icons (using Lucide React or similar)
const RefreshIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M3 2v6h6M3 8a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6 2l-3 3"/>
  </svg>
);

const SettingsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="3"/>
    <path d="m12 1 1.98 2.75L17.65 2.4l.7 3.45L21.9 7.2l-.97 3.41L24 12l-2.75 1.98L22.6 17.65l-3.45.7L17.8 21.9l-3.41-.97L12 24l-1.98-2.75L6.35 22.6l-.7-3.45L2.1 17.8l.97-3.41L0 12l2.75-1.98L1.4 6.35l3.45-.7L6.2 2.1l3.41.97z"/>
  </svg>
);

const ExportIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7,10 12,15 17,10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);

const AlertTriangleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <path d="M12 9v4M12 17h.01"/>
  </svg>
);

// Default configuration
const defaultConfig: DashboardConfig = {
  theme: 'auto',
  layout: {
    columns: 2,
    cardSpacing: 16,
    compactMode: false,
  },
  autoRefresh: true,
  refreshInterval: 5000, // 5 seconds
  dataRetention: {
    maxHistoryPoints: 1000,
    retentionPeriod: 24, // 24 hours
  },
  accessibility: {
    highContrast: false,
    reducedMotion: false,
    screenReaderOptimized: false,
  },
};

// 🟢 WORKING: Ultra-aggressive NaN protection for SCWTDashboard
const safeNumber = (value: any, defaultValue: number = 0): number => {
  if (value === null || value === undefined) return defaultValue;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    if (isNaN(parsed) || !isFinite(parsed)) return defaultValue;
    return parsed;
  }
  if (typeof value !== 'number') return defaultValue;
  if (isNaN(value) || !isFinite(value) || value === Infinity || value === -Infinity) {
    console.warn(`[SCWTDashboard] Invalid numeric value: ${value}, using default: ${defaultValue}`);
    return defaultValue;
  }
  return value;
};

const safePercentage = (value: any, defaultValue: number = 0.5): number => {
  const num = safeNumber(value, defaultValue);
  const clamped = Math.max(0, Math.min(1, num));
  if (isNaN(clamped) || !isFinite(clamped)) {
    console.warn(`[SCWTDashboard] Invalid percentage: ${clamped}, using default: ${defaultValue}`);
    return defaultValue;
  }
  return clamped;
};

// Safe display formatting for metrics
const safeMetricDisplay = (value: any, defaultValue: number = 0, decimals: number = 3): string => {
  const safeVal = safeNumber(value, defaultValue);
  try {
    return safeVal.toFixed(decimals);
  } catch (error) {
    console.error('[SCWTDashboard] Error formatting metric:', error);
    return defaultValue.toFixed(decimals);
  }
};

// Time range presets
const createTimeRangePresets = (): TimeRangePresets => {
  const now = new Date();
  return {
    last15Minutes: {
      start: new Date(now.getTime() - 15 * 60 * 1000),
      end: now,
      granularity: 'minute',
    },
    lastHour: {
      start: new Date(now.getTime() - 60 * 60 * 1000),
      end: now,
      granularity: 'minute',
    },
    last24Hours: {
      start: new Date(now.getTime() - 24 * 60 * 60 * 1000),
      end: now,
      granularity: 'hour',
    },
    lastWeek: {
      start: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
      end: now,
      granularity: 'day',
    },
    lastMonth: {
      start: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
      end: now,
      granularity: 'day',
    },
    custom: null,
  };
};

export const SCWTDashboard: React.FC<SCWTDashboardProps> = ({
  data,
  config = defaultConfig,
  onMetricSelect,
  onExportData,
  onRefresh,
  isLoading = false,
  error = null,
}) => {
  // State management
  const [dashboardConfig, setDashboardConfig] = useState<DashboardConfig>(config);
  const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>(
    createTimeRangePresets().last24Hours
  );
  const [selectedMetrics, setSelectedMetrics] = useState<MetricType[]>([
    'combinedScore',
    'structuralWeight',
    'contextWeight',
    'temporalWeight',
  ]);
  const [showSettings, setShowSettings] = useState(false);
  const [showDebugTools, setShowDebugTools] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Auto-refresh effect
  useEffect(() => {
    if (!dashboardConfig.autoRefresh) return;

    const interval = setInterval(() => {
      onRefresh?.();
      setLastRefresh(new Date());
    }, dashboardConfig.refreshInterval);

    return () => clearInterval(interval);
  }, [dashboardConfig.autoRefresh, dashboardConfig.refreshInterval, onRefresh]);

  // Memoized time range options
  const timeRangePresets = useMemo(() => createTimeRangePresets(), []);

  // Event handlers
  const handleRefresh = useCallback(() => {
    onRefresh?.();
    setLastRefresh(new Date());
  }, [onRefresh]);

  const handleMetricToggle = useCallback((metric: MetricType) => {
    setSelectedMetrics(prev => {
      const newMetrics = prev.includes(metric)
        ? prev.filter(m => m !== metric)
        : [...prev, metric];
      onMetricSelect?.(metric);
      return newMetrics;
    });
  }, [onMetricSelect]);

  const handleExport = useCallback(() => {
    onExportData?.();
  }, [onExportData]);

  const handleConfigChange = useCallback((newConfig: Partial<DashboardConfig>) => {
    setDashboardConfig(prev => ({ ...prev, ...newConfig }));
  }, []);

  // Render loading state
  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-300 rounded mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3, 4, 5, 6].map(i => (
              <div key={i} className="h-64 bg-gray-300 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="p-6">
        <Card accentColor="orange" className="border-red-200 dark:border-red-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600 dark:text-red-400">
              <AlertTriangleIcon />
              Dashboard Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-red-600 dark:text-red-300 mb-4">
              {error.message}
            </p>
            <Button onClick={handleRefresh} variant="outline" size="sm">
              <RefreshIcon />
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Main dashboard render
  return (
    <div 
      className="p-6 space-y-6 min-h-screen bg-background"
      role="main"
      aria-label="SCWT Metrics Dashboard"
    >
      {/* Dashboard Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">
            SCWT Metrics Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Real-time performance and confidence visualization
          </p>
          <p className="text-sm text-muted-foreground">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
        </div>

        {/* Dashboard Controls */}
        <div className="flex items-center gap-2">
          <Button
            onClick={handleRefresh}
            variant="outline"
            size="sm"
            className="gap-2"
            disabled={isLoading}
            aria-label="Refresh dashboard data"
          >
            <RefreshIcon />
            Refresh
          </Button>

          <Button
            onClick={handleExport}
            variant="outline"
            size="sm"
            className="gap-2"
            aria-label="Export dashboard data"
          >
            <ExportIcon />
            Export
          </Button>

          <Toggle
            pressed={showDebugTools}
            onPressedChange={setShowDebugTools}
            aria-label="Toggle debug tools"
          >
            Debug
          </Toggle>

          <Button
            onClick={() => setShowSettings(!showSettings)}
            variant="outline"
            size="sm"
            aria-label="Dashboard settings"
          >
            <SettingsIcon />
          </Button>

          <ThemeToggle />
        </div>
      </div>

      {/* Time Range Selector */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap items-center gap-4">
            <label className="font-medium text-sm">Time Range:</label>
            <div className="flex gap-2">
              {Object.entries(timeRangePresets).filter(([key]) => key !== 'custom').map(([key, range]) => (
                <Button
                  key={key}
                  onClick={() => setSelectedTimeRange(range)}
                  variant={selectedTimeRange === range ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </Button>
              ))}
            </div>
            
            {/* Metric Selection */}
            <div className="flex flex-wrap items-center gap-2 ml-auto">
              <label className="font-medium text-sm">Metrics:</label>
              {(['combinedScore', 'structuralWeight', 'contextWeight', 'temporalWeight'] as MetricType[]).map(metric => (
                <Toggle
                  key={metric}
                  pressed={selectedMetrics.includes(metric)}
                  onPressedChange={() => handleMetricToggle(metric)}
                  size="sm"
                  className="text-xs"
                >
                  {metric.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                </Toggle>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Settings Panel */}
      {showSettings && (
        <Card accentColor="blue">
          <CardHeader>
            <CardTitle>Dashboard Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Auto Refresh</label>
                <Toggle
                  pressed={dashboardConfig.autoRefresh}
                  onPressedChange={(pressed) => 
                    handleConfigChange({ autoRefresh: pressed })
                  }
                >
                  {dashboardConfig.autoRefresh ? 'Enabled' : 'Disabled'}
                </Toggle>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Compact Mode</label>
                <Toggle
                  pressed={dashboardConfig.layout.compactMode}
                  onPressedChange={(pressed) => 
                    handleConfigChange({ 
                      layout: { ...dashboardConfig.layout, compactMode: pressed }
                    })
                  }
                >
                  {dashboardConfig.layout.compactMode ? 'On' : 'Off'}
                </Toggle>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">High Contrast</label>
                <Toggle
                  pressed={dashboardConfig.accessibility.highContrast}
                  onPressedChange={(pressed) => 
                    handleConfigChange({ 
                      accessibility: { 
                        ...dashboardConfig.accessibility, 
                        highContrast: pressed 
                      }
                    })
                  }
                >
                  {dashboardConfig.accessibility.highContrast ? 'On' : 'Off'}
                </Toggle>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Dashboard Grid */}
      <div className={`grid gap-6 ${
        dashboardConfig.layout.compactMode 
          ? 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3' 
          : 'grid-cols-1 lg:grid-cols-2'
      }`}>
        
        {/* Real-time Monitoring */}
        <div className="lg:col-span-2">
          <RealTimeMonitoring
            data={data}
            timeRange={selectedTimeRange}
            selectedMetrics={selectedMetrics}
            config={dashboardConfig}
          />
        </div>

        {/* Confidence Visualization */}
        <div className={dashboardConfig.layout.compactMode ? '' : 'lg:col-span-1'}>
          <ConfidenceVisualization
            metrics={data?.confidence ? [data.confidence] : []}
            timeRange={selectedTimeRange}
            interactive={true}
            chartType="area"
          />
        </div>

        {/* Performance Metrics */}
        <div>
          <PerformanceComponent
            metrics={data?.performance ? [data.performance] : []}
            displayMetrics={['tokenEfficiency', 'cost', 'timing', 'quality']}
            updateFrequency={dashboardConfig.refreshInterval}
          />
        </div>

        {/* Current Metrics Summary */}
        <div>
          <Card accentColor="green">
            <CardHeader>
              <CardTitle>Current SCWT Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              {data?.current ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {safeMetricDisplay(data.current.combinedScore, 0, 3)}
                      </div>
                      <div className="text-sm text-muted-foreground">Combined Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                        {data.current.confidence 
                          ? safeMetricDisplay(data.current.confidence, 0.5, 3)
                          : 'N/A'}
                      </div>
                      <div className="text-sm text-muted-foreground">Confidence</div>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Structural Weight:</span>
                      <span className="text-sm font-medium">{safeMetricDisplay(data.current.structuralWeight, 0, 3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Context Weight:</span>
                      <span className="text-sm font-medium">{safeMetricDisplay(data.current.contextWeight, 0, 3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Temporal Weight:</span>
                      <span className="text-sm font-medium">{safeMetricDisplay(data.current.temporalWeight, 0, 3)}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted-foreground py-8">
                  No metrics data available
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* System Status */}
        <div>
          <Card accentColor={data?.status === 'active' ? 'green' : 'orange'}>
            <CardHeader>
              <CardTitle>System Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span>Status:</span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    data?.status === 'active' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-800/20 dark:text-green-300'
                      : 'bg-orange-100 text-orange-800 dark:bg-orange-800/20 dark:text-orange-300'
                  }`}>
                    {data?.status || 'Unknown'}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Data Points:</span>
                  <span className="font-medium">{data?.history?.length || 0}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span>Last Update:</span>
                  <span className="text-sm text-muted-foreground">
                    {data?.lastUpdate ? data.lastUpdate.toLocaleTimeString() : 'Never'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Debug Tools */}
      {showDebugTools && data?.current && data?.confidence && (
        <DebugTools
          metrics={data.current}
          confidence={data.confidence}
          debugMode={true}
          enableAnalysis={true}
          exportFormats={['json', 'csv', 'excel']}
        />
      )}
    </div>
  );
};

export default SCWTDashboard;