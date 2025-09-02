/**
 * Performance Metrics Component
 * Phase 7 DeepConf Integration
 * 
 * Token efficiency tracking, cost monitoring, and performance analysis
 * with real-time updates and alert thresholds
 */

import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Toggle } from '../ui/Toggle';
import { Progress } from '../ui/progress';
import { 
  PerformanceMetricsProps, 
  PerformanceMetrics as PerfMetrics,
  PerformanceThresholds 
} from './types';

// Icons
const DollarSignIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="1" x2="12" y2="23"/>
    <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
  </svg>
);

const ClockIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12,6 12,12 16,14"/>
  </svg>
);

const ZapIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="13,2 3,14 12,14 11,22 21,10 12,10 13,2"/>
  </svg>
);

const TargetIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/>
    <circle cx="12" cy="12" r="6"/>
    <circle cx="12" cy="12" r="2"/>
  </svg>
);

const TrendingUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22,7 13.5,15.5 8.5,10.5 2,17"/>
    <polyline points="16,7 22,7 22,13"/>
  </svg>
);

const TrendingDownIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22,17 13.5,8.5 8.5,13.5 2,7"/>
    <polyline points="16,17 22,17 22,11"/>
  </svg>
);

const AlertTriangleIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <path d="M12 9v4M12 17h.01"/>
  </svg>
);

// Default performance thresholds
const defaultThresholds: PerformanceThresholds = {
  tokenEfficiency: {
    excellent: 0.9,
    good: 0.75,
    acceptable: 0.6,
    poor: 0.4,
  },
  responseTime: {
    fast: 100,
    acceptable: 500,
    slow: 2000,
    timeout: 10000,
  },
  cost: {
    budget: 0.01,
    warning: 0.05,
    critical: 0.10,
  },
};

export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  metrics,
  displayMetrics = ['tokenEfficiency', 'cost', 'timing', 'quality'],
  baseline,
  thresholds = defaultThresholds,
  updateFrequency = 5000,
}) => {
  // State management
  const [selectedPeriod, setSelectedPeriod] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [showComparison, setShowComparison] = useState(!!baseline);
  const [showTrends, setShowTrends] = useState(true);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  
  // Current metrics (latest data point)
  const currentMetrics = metrics[metrics.length - 1];
  
  // Calculate performance statistics
  const stats = useMemo(() => {
    if (!metrics.length) return null;

    const getAverage = (key: keyof PerfMetrics, subKey: string) => {
      const values = metrics
        .map(m => (m[key] as any)?.[subKey])
        .filter(v => typeof v === 'number');
      return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    };

    const getTrend = (key: keyof PerfMetrics, subKey: string): 'up' | 'down' | 'stable' => {
      if (metrics.length < 2) return 'stable';
      const recent = metrics.slice(-5).map(m => (m[key] as any)?.[subKey]).filter(v => typeof v === 'number');
      const older = metrics.slice(-10, -5).map(m => (m[key] as any)?.[subKey]).filter(v => typeof v === 'number');
      
      if (!recent.length || !older.length) return 'stable';
      
      const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
      const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
      const change = (recentAvg - olderAvg) / olderAvg;
      
      if (Math.abs(change) < 0.05) return 'stable';
      return change > 0 ? 'up' : 'down';
    };

    return {
      tokenEfficiency: {
        avg: getAverage('tokenEfficiency', 'efficiencyScore'),
        trend: getTrend('tokenEfficiency', 'efficiencyScore'),
        compressionRatio: getAverage('tokenEfficiency', 'compressionRatio'),
      },
      cost: {
        avgTotal: getAverage('cost', 'totalCost'),
        avgPerQuery: getAverage('cost', 'costPerQuery'),
        totalSavings: getAverage('cost', 'costSavings'),
        trend: getTrend('cost', 'totalCost'),
      },
      timing: {
        avgProcessing: getAverage('timing', 'processingTime'),
        avgTotal: getAverage('timing', 'totalResponseTime'),
        throughput: getAverage('timing', 'throughput'),
        trend: getTrend('timing', 'totalResponseTime'),
      },
      quality: {
        avgAccuracy: getAverage('quality', 'accuracy'),
        avgF1: getAverage('quality', 'f1Score'),
        trend: getTrend('quality', 'accuracy'),
      },
    };
  }, [metrics]);

  // Get performance level and color
  const getPerformanceLevel = (value: number, thresholdType: 'tokenEfficiency' | 'responseTime' | 'cost') => {
    if (thresholdType === 'tokenEfficiency') {
      if (value >= thresholds.tokenEfficiency.excellent) return { level: 'excellent', color: 'text-green-600 dark:text-green-400' };
      if (value >= thresholds.tokenEfficiency.good) return { level: 'good', color: 'text-blue-600 dark:text-blue-400' };
      if (value >= thresholds.tokenEfficiency.acceptable) return { level: 'acceptable', color: 'text-yellow-600 dark:text-yellow-400' };
      return { level: 'poor', color: 'text-red-600 dark:text-red-400' };
    }
    
    if (thresholdType === 'responseTime') {
      if (value <= thresholds.responseTime.fast) return { level: 'fast', color: 'text-green-600 dark:text-green-400' };
      if (value <= thresholds.responseTime.acceptable) return { level: 'acceptable', color: 'text-yellow-600 dark:text-yellow-400' };
      if (value <= thresholds.responseTime.slow) return { level: 'slow', color: 'text-orange-600 dark:text-orange-400' };
      return { level: 'timeout', color: 'text-red-600 dark:text-red-400' };
    }
    
    // cost
    if (value <= thresholds.cost.budget) return { level: 'budget', color: 'text-green-600 dark:text-green-400' };
    if (value <= thresholds.cost.warning) return { level: 'warning', color: 'text-yellow-600 dark:text-yellow-400' };
    return { level: 'critical', color: 'text-red-600 dark:text-red-400' };
  };

  // Format currency
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 4,
    }).format(value);
  };

  // Format time
  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  // Format percentage
  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Render trend icon
  const renderTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    if (trend === 'up') return <TrendingUpIcon />;
    if (trend === 'down') return <TrendingDownIcon />;
    return <span className="w-4 h-4 flex items-center justify-center">—</span>;
  };

  // Render metric card
  const renderMetricCard = (title: string, icon: React.ReactNode, children: React.ReactNode, accentColor?: string) => (
    <Card accentColor={accentColor as any}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          {icon}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {children}
      </CardContent>
    </Card>
  );

  if (!currentMetrics) {
    return (
      <div className="space-y-4">
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            No performance data available
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Performance Overview */}
      <Card accentColor="purple">
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="flex items-center gap-2">
                <ZapIcon />
                Performance Metrics
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Real-time token efficiency and cost tracking
              </p>
            </div>
            
            <div className="flex items-center gap-2">
              <Toggle
                pressed={showTrends}
                onPressedChange={setShowTrends}
                size="sm"
              >
                Trends
              </Toggle>
              
              <Toggle
                pressed={showComparison}
                onPressedChange={setShowComparison}
                size="sm"
                disabled={!baseline}
              >
                Compare
              </Toggle>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4 gap-4">
        
        {/* Token Efficiency */}
        {displayMetrics.includes('tokenEfficiency') && (
          renderMetricCard(
            'Token Efficiency',
            <ZapIcon />,
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">
                  {formatPercentage(currentMetrics.tokenEfficiency.efficiencyScore)}
                </span>
                {showTrends && stats && (
                  <div className={`flex items-center gap-1 ${
                    stats.tokenEfficiency.trend === 'up' ? 'text-green-600 dark:text-green-400' :
                    stats.tokenEfficiency.trend === 'down' ? 'text-red-600 dark:text-red-400' :
                    'text-muted-foreground'
                  }`}>
                    {renderTrendIcon(stats.tokenEfficiency.trend)}
                  </div>
                )}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Compression:</span>
                  <span className="font-medium">
                    {currentMetrics.tokenEfficiency.compressionRatio.toFixed(2)}x
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Total Tokens:</span>
                  <span className="font-medium">
                    {currentMetrics.tokenEfficiency.totalTokens.toLocaleString()}
                  </span>
                </div>
              </div>
              
              <Progress 
                value={currentMetrics.tokenEfficiency.efficiencyScore * 100} 
                className="h-2"
              />
              
              {showComparison && baseline && (
                <div className="text-xs text-muted-foreground">
                  vs baseline: {
                    ((currentMetrics.tokenEfficiency.efficiencyScore - baseline.tokenEfficiency.efficiencyScore) * 100).toFixed(1)
                  }%
                </div>
              )}
            </div>,
            'green'
          )
        )}

        {/* Cost Metrics */}
        {displayMetrics.includes('cost') && (
          renderMetricCard(
            'Cost Tracking',
            <DollarSignIcon />,
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">
                  {formatCurrency(currentMetrics.cost.totalCost)}
                </span>
                {showTrends && stats && (
                  <div className={`flex items-center gap-1 ${
                    stats.cost.trend === 'down' ? 'text-green-600 dark:text-green-400' :
                    stats.cost.trend === 'up' ? 'text-red-600 dark:text-red-400' :
                    'text-muted-foreground'
                  }`}>
                    {renderTrendIcon(stats.cost.trend)}
                  </div>
                )}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Per Query:</span>
                  <span className="font-medium">
                    {formatCurrency(currentMetrics.cost.costPerQuery)}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Savings:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {formatCurrency(currentMetrics.cost.costSavings)}
                  </span>
                </div>
              </div>
              
              {getPerformanceLevel(currentMetrics.cost.totalCost, 'cost').level !== 'budget' && (
                <div className={`flex items-center gap-1 text-xs ${
                  getPerformanceLevel(currentMetrics.cost.totalCost, 'cost').color
                }`}>
                  <AlertTriangleIcon />
                  {getPerformanceLevel(currentMetrics.cost.totalCost, 'cost').level.toUpperCase()} COST
                </div>
              )}
            </div>,
            getPerformanceLevel(currentMetrics.cost.totalCost, 'cost').level === 'budget' ? 'green' : 'orange'
          )
        )}

        {/* Timing Metrics */}
        {displayMetrics.includes('timing') && (
          renderMetricCard(
            'Response Time',
            <ClockIcon />,
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">
                  {formatTime(currentMetrics.timing.totalResponseTime)}
                </span>
                {showTrends && stats && (
                  <div className={`flex items-center gap-1 ${
                    stats.timing.trend === 'down' ? 'text-green-600 dark:text-green-400' :
                    stats.timing.trend === 'up' ? 'text-red-600 dark:text-red-400' :
                    'text-muted-foreground'
                  }`}>
                    {renderTrendIcon(stats.timing.trend)}
                  </div>
                )}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing:</span>
                  <span className="font-medium">
                    {formatTime(currentMetrics.timing.processingTime)}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Throughput:</span>
                  <span className="font-medium">
                    {currentMetrics.timing.throughput.toFixed(1)} req/s
                  </span>
                </div>
              </div>
              
              <div className={`text-xs font-medium ${
                getPerformanceLevel(currentMetrics.timing.totalResponseTime, 'responseTime').color
              }`}>
                {getPerformanceLevel(currentMetrics.timing.totalResponseTime, 'responseTime').level.toUpperCase()}
              </div>
            </div>,
            getPerformanceLevel(currentMetrics.timing.totalResponseTime, 'responseTime').level === 'fast' ? 'green' : 'blue'
          )
        )}

        {/* Quality Metrics */}
        {displayMetrics.includes('quality') && (
          renderMetricCard(
            'Quality Score',
            <TargetIcon />,
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">
                  {formatPercentage(currentMetrics.quality.accuracy)}
                </span>
                {showTrends && stats && (
                  <div className={`flex items-center gap-1 ${
                    stats.quality.trend === 'up' ? 'text-green-600 dark:text-green-400' :
                    stats.quality.trend === 'down' ? 'text-red-600 dark:text-red-400' :
                    'text-muted-foreground'
                  }`}>
                    {renderTrendIcon(stats.quality.trend)}
                  </div>
                )}
              </div>
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Precision:</span>
                  <span className="font-medium">
                    {formatPercentage(currentMetrics.quality.precision)}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>F1 Score:</span>
                  <span className="font-medium">
                    {formatPercentage(currentMetrics.quality.f1Score)}
                  </span>
                </div>
              </div>
              
              <Progress 
                value={currentMetrics.quality.accuracy * 100} 
                className="h-2"
              />
            </div>,
            'cyan'
          )
        )}
      </div>

      {/* Historical Trends */}
      {showTrends && stats && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Historical Averages</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <div className="text-sm text-muted-foreground">Token Efficiency</div>
                <div className="text-xl font-semibold">
                  {formatPercentage(stats.tokenEfficiency.avg)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Avg compression: {stats.tokenEfficiency.compressionRatio.toFixed(2)}x
                </div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground">Cost Per Query</div>
                <div className="text-xl font-semibold">
                  {formatCurrency(stats.cost.avgPerQuery)}
                </div>
                <div className="text-xs text-green-600 dark:text-green-400">
                  Total saved: {formatCurrency(stats.cost.totalSavings)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground">Response Time</div>
                <div className="text-xl font-semibold">
                  {formatTime(stats.timing.avgTotal)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Processing: {formatTime(stats.timing.avgProcessing)}
                </div>
              </div>
              
              <div>
                <div className="text-sm text-muted-foreground">Quality</div>
                <div className="text-xl font-semibold">
                  {formatPercentage(stats.quality.avgAccuracy)}
                </div>
                <div className="text-xs text-muted-foreground">
                  F1: {formatPercentage(stats.quality.avgF1)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PerformanceMetrics;