/**
 * DeepConf Components - Phase 7 Integration
 * SCWT Metrics Dashboard and Confidence Visualization
 * 
 * Export all components and types for the DeepConf integration
 */

// Main Components
export { default as SCWTDashboard } from './SCWTDashboard';
export { default as ConfidenceVisualization } from './ConfidenceVisualizationSafe';
export { default as PerformanceMetrics } from './PerformanceMetrics';
export { default as DebugTools } from './DebugTools';
export { default as RealTimeMonitoring } from './RealTimeMonitoring';

// Specialized Chart Components
export { default as ConfidenceChart } from './ConfidenceChart';
export { default as UncertaintyBounds } from './UncertaintyBounds';

// Type Definitions
export type {
  // Core Types
  SCWTMetrics,
  ConfidenceMetrics,
  PerformanceMetrics,
  RealTimeData,
  
  // Configuration Types
  DashboardConfig,
  ConfidenceThresholds,
  PerformanceThresholds,
  WebSocketConfig,
  
  // Component Props
  SCWTDashboardProps,
  ConfidenceVisualizationProps,
  PerformanceMetricsProps,
  DebugToolsProps,
  
  // Chart Types
  ChartDataPoint,
  ConfidenceInterval,
  MultiDimensionalChartData,
  ChartType,
  
  // Analysis Types
  AnalysisResult,
  Anomaly,
  TrendAnalysis,
  DebugAction,
  
  // Utility Types
  MetricType,
  TimeRange,
  TimeRangePresets,
  ThemeMode,
  
  // WebSocket Types
  WebSocketMessage,
  
  // Export Types
  ExportData,
  
  // Error Types
  DeepConfError,
  
  // State Types
  DashboardState,
  VisualizationState,
  
  // Type Guards
  isValidSCWTMetrics,
  isValidPerformanceMetrics,
} from './types';

// Re-export commonly used UI components for convenience
export { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
export { Button } from '../ui/Button';
export { Toggle } from '../ui/Toggle';
export { Select } from '../ui/Select';
export { Progress } from '../ui/progress';
export { Badge } from '../ui/badge';

// Default configurations for easy setup
export const DEFAULT_DASHBOARD_CONFIG: DashboardConfig = {
  theme: 'auto',
  layout: {
    columns: 2,
    cardSpacing: 16,
    compactMode: false,
  },
  autoRefresh: true,
  refreshInterval: 5000,
  dataRetention: {
    maxHistoryPoints: 1000,
    retentionPeriod: 24,
  },
  accessibility: {
    highContrast: false,
    reducedMotion: false,
    screenReaderOptimized: false,
  },
};

export const DEFAULT_CONFIDENCE_THRESHOLDS: ConfidenceThresholds = {
  low: 0.3,
  medium: 0.6,
  high: 0.8,
  critical: 0.95,
};

export const DEFAULT_PERFORMANCE_THRESHOLDS: PerformanceThresholds = {
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

export const DEFAULT_WEBSOCKET_CONFIG: WebSocketConfig = {
  url: 'http://localhost:8181',  // Socket.IO endpoint instead of WebSocket
  reconnectAttempts: 5,
  reconnectDelay: 3000,
  heartbeatInterval: 30000,
  timeout: 10000,
};

// Utility functions for common operations
export const createTimeRange = (
  hours: number,
  granularity: 'minute' | 'hour' | 'day' = 'hour'
): TimeRange => {
  const now = new Date();
  return {
    start: new Date(now.getTime() - hours * 60 * 60 * 1000),
    end: now,
    granularity,
  };
};

export const formatConfidence = (confidence: number): string => {
  return `${(confidence * 100).toFixed(1)}%`;
};

export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 4,
  }).format(amount);
};

export const formatDuration = (milliseconds: number): string => {
  if (milliseconds < 1000) {
    return `${milliseconds.toFixed(0)}ms`;
  } else if (milliseconds < 60000) {
    return `${(milliseconds / 1000).toFixed(2)}s`;
  } else {
    return `${(milliseconds / 60000).toFixed(2)}m`;
  }
};

export const getConfidenceLevel = (
  confidence: number,
  thresholds: ConfidenceThresholds = DEFAULT_CONFIDENCE_THRESHOLDS
): 'critical' | 'high' | 'medium' | 'low' | 'very-low' => {
  if (confidence >= thresholds.critical) return 'critical';
  if (confidence >= thresholds.high) return 'high';
  if (confidence >= thresholds.medium) return 'medium';
  if (confidence >= thresholds.low) return 'low';
  return 'very-low';
};

export const getConfidenceColor = (
  confidence: number,
  thresholds: ConfidenceThresholds = DEFAULT_CONFIDENCE_THRESHOLDS
): string => {
  const level = getConfidenceLevel(confidence, thresholds);
  
  switch (level) {
    case 'critical': return 'text-green-600 dark:text-green-400';
    case 'high': return 'text-blue-600 dark:text-blue-400';
    case 'medium': return 'text-yellow-600 dark:text-yellow-400';
    case 'low': return 'text-orange-600 dark:text-orange-400';
    case 'very-low': return 'text-red-600 dark:text-red-400';
    default: return 'text-muted-foreground';
  }
};

// Mock data generators for development and testing
export const generateMockSCWTMetrics = (count: number = 50): SCWTMetrics[] => {
  const metrics: SCWTMetrics[] = [];
  
  for (let i = 0; i < count; i++) {
    const timestamp = new Date(Date.now() - (count - i - 1) * 60000);
    const structuralWeight = 0.3 + Math.random() * 0.4;
    const contextWeight = 0.2 + Math.random() * 0.5;
    const temporalWeight = 0.1 + Math.random() * 0.3;
    const combinedScore = (structuralWeight + contextWeight + temporalWeight) / 3;
    
    metrics.push({
      structuralWeight,
      contextWeight,
      temporalWeight,
      combinedScore,
      timestamp,
      confidence: 0.6 + Math.random() * 0.3,
    });
  }
  
  return metrics;
};

export const generateMockConfidenceMetrics = (count: number = 50): ConfidenceMetrics[] => {
  const metrics: ConfidenceMetrics[] = [];
  
  for (let i = 0; i < count; i++) {
    const overall = 0.5 + Math.random() * 0.4;
    const epistemic = Math.random() * 0.2;
    const aleatoric = Math.random() * 0.15;
    
    metrics.push({
      overall,
      bayesian: {
        lower: Math.max(0, overall - 0.1),
        upper: Math.min(1, overall + 0.1),
        mean: overall,
        variance: 0.02 + Math.random() * 0.03,
      },
      dimensions: {
        structural: overall + (Math.random() - 0.5) * 0.1,
        contextual: overall + (Math.random() - 0.5) * 0.1,
        temporal: overall + (Math.random() - 0.5) * 0.1,
        semantic: overall + (Math.random() - 0.5) * 0.1,
      },
      uncertainty: {
        epistemic,
        aleatoric,
        total: Math.sqrt(epistemic * epistemic + aleatoric * aleatoric),
      },
      trend: ['increasing', 'decreasing', 'stable', 'volatile'][Math.floor(Math.random() * 4)] as any,
    });
  }
  
  return metrics;
};

export const generateMockPerformanceMetrics = (count: number = 50): PerformanceMetrics[] => {
  const metrics: PerformanceMetrics[] = [];
  
  for (let i = 0; i < count; i++) {
    const inputTokens = 1000 + Math.random() * 2000;
    const outputTokens = 200 + Math.random() * 800;
    const totalTokens = inputTokens + outputTokens;
    
    metrics.push({
      tokenEfficiency: {
        inputTokens,
        outputTokens,
        totalTokens,
        compressionRatio: 2 + Math.random() * 3,
        efficiencyScore: 0.6 + Math.random() * 0.3,
      },
      cost: {
        inputCost: inputTokens * 0.00001,
        outputCost: outputTokens * 0.00003,
        totalCost: (inputTokens * 0.00001) + (outputTokens * 0.00003),
        costPerQuery: 0.005 + Math.random() * 0.015,
        costSavings: Math.random() * 0.01,
      },
      timing: {
        processingTime: 50 + Math.random() * 200,
        networkLatency: 10 + Math.random() * 50,
        totalResponseTime: 100 + Math.random() * 300,
        throughput: 5 + Math.random() * 15,
      },
      quality: {
        accuracy: 0.8 + Math.random() * 0.15,
        precision: 0.75 + Math.random() * 0.2,
        recall: 0.7 + Math.random() * 0.25,
        f1Score: 0.75 + Math.random() * 0.2,
      },
    });
  }
  
  return metrics;
};