/**
 * SCWT Metrics Dashboard - TypeScript Interfaces
 * Phase 7 DeepConf Integration
 * 
 * Comprehensive type definitions for confidence visualization and metrics tracking
 */

// Core SCWT Metrics Types
export interface SCWTMetrics {
  /** Structural complexity weight */
  structuralWeight: number;
  /** Context awareness weight */
  contextWeight: number;
  /** Temporal consistency weight */
  temporalWeight: number;
  /** Combined SCWT score */
  combinedScore: number;
  /** Timestamp when metrics were calculated */
  timestamp: Date;
  /** Confidence level in the metrics */
  confidence: number;
}

// Confidence Metrics
export interface ConfidenceMetrics {
  /** Overall confidence score (0-1) */
  overall: number;
  /** Bayesian confidence interval */
  bayesian: {
    lower: number;
    upper: number;
    mean: number;
    variance: number;
  };
  /** Multi-dimensional confidence scores */
  dimensions: {
    structural: number;
    contextual: number;
    temporal: number;
    semantic: number;
  };
  /** Uncertainty bounds */
  uncertainty: {
    epistemic: number; // Model uncertainty
    aleatoric: number; // Data uncertainty
    total: number;     // Combined uncertainty
  };
  /** Confidence trend over time */
  trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
}

// Performance Metrics
export interface PerformanceMetrics {
  /** Token efficiency metrics */
  tokenEfficiency: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    compressionRatio: number;
    efficiencyScore: number;
  };
  /** Cost tracking */
  cost: {
    inputCost: number;
    outputCost: number;
    totalCost: number;
    costPerQuery: number;
    costSavings: number;
  };
  /** Response time metrics */
  timing: {
    processingTime: number;
    networkLatency: number;
    totalResponseTime: number;
    throughput: number;
  };
  /** Quality metrics */
  quality: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

// Real-time Data Structures
export interface RealTimeData {
  /** Current metrics snapshot */
  current: SCWTMetrics;
  /** Historical data points */
  history: SCWTMetrics[];
  /** Performance metrics */
  performance: PerformanceMetrics;
  /** Confidence metrics */
  confidence: ConfidenceMetrics;
  /** System status */
  status: 'active' | 'inactive' | 'error' | 'maintenance';
  /** Last update timestamp */
  lastUpdate: Date;
}

// Chart Data Types
export interface ChartDataPoint {
  timestamp: Date;
  value: number;
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface MultiDimensionalChartData {
  structural: ChartDataPoint[];
  contextual: ChartDataPoint[];
  temporal: ChartDataPoint[];
  combined: ChartDataPoint[];
}

export interface ConfidenceInterval {
  timestamp: Date;
  mean: number;
  lower: number;
  upper: number;
  uncertainty: number;
}

// Component Props Types
export interface SCWTDashboardProps {
  /** Real-time data source */
  data?: RealTimeData;
  /** Dashboard configuration */
  config?: DashboardConfig;
  /** Event handlers */
  onMetricSelect?: (metric: keyof SCWTMetrics) => void;
  onExportData?: () => void;
  onRefresh?: () => void;
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: Error | null;
}

export interface ConfidenceVisualizationProps {
  /** Confidence metrics data */
  metrics: ConfidenceMetrics[];
  /** Chart type selection */
  chartType?: 'line' | 'area' | 'bar' | 'heatmap';
  /** Time range filter */
  timeRange?: TimeRange;
  /** Interactive features */
  interactive?: boolean;
  /** Confidence threshold indicators */
  thresholds?: ConfidenceThresholds;
  /** Event handlers */
  onConfidenceClick?: (point: ChartDataPoint) => void;
  onThresholdChange?: (thresholds: ConfidenceThresholds) => void;
}

export interface PerformanceMetricsProps {
  /** Performance data */
  metrics: PerformanceMetrics[];
  /** Metrics to display */
  displayMetrics?: (keyof PerformanceMetrics)[];
  /** Comparison baseline */
  baseline?: PerformanceMetrics;
  /** Alert thresholds */
  thresholds?: PerformanceThresholds;
  /** Update frequency */
  updateFrequency?: number;
}

export interface DebugToolsProps {
  /** Current SCWT metrics */
  metrics: SCWTMetrics;
  /** Confidence data */
  confidence: ConfidenceMetrics;
  /** Debug mode */
  debugMode?: boolean;
  /** Interactive analysis features */
  enableAnalysis?: boolean;
  /** Data export capabilities */
  exportFormats?: ('json' | 'csv' | 'excel')[];
  /** Event handlers */
  onDebugAction?: (action: DebugAction) => void;
}

// Configuration Types
export interface DashboardConfig {
  /** Theme settings */
  theme: 'light' | 'dark' | 'auto';
  /** Layout configuration */
  layout: {
    columns: number;
    cardSpacing: number;
    compactMode: boolean;
  };
  /** Refresh settings */
  autoRefresh: boolean;
  refreshInterval: number;
  /** Data retention */
  dataRetention: {
    maxHistoryPoints: number;
    retentionPeriod: number; // in hours
  };
  /** Accessibility settings */
  accessibility: {
    highContrast: boolean;
    reducedMotion: boolean;
    screenReaderOptimized: boolean;
  };
}

export interface ConfidenceThresholds {
  /** Low confidence threshold */
  low: number;
  /** Medium confidence threshold */
  medium: number;
  /** High confidence threshold */
  high: number;
  /** Critical confidence threshold */
  critical: number;
}

export interface PerformanceThresholds {
  /** Token efficiency thresholds */
  tokenEfficiency: {
    excellent: number;
    good: number;
    acceptable: number;
    poor: number;
  };
  /** Response time thresholds (ms) */
  responseTime: {
    fast: number;
    acceptable: number;
    slow: number;
    timeout: number;
  };
  /** Cost thresholds */
  cost: {
    budget: number;
    warning: number;
    critical: number;
  };
}

// Time Range Types
export interface TimeRange {
  start: Date;
  end: Date;
  granularity: 'minute' | 'hour' | 'day' | 'week';
}

export interface TimeRangePresets {
  last15Minutes: TimeRange;
  lastHour: TimeRange;
  last24Hours: TimeRange;
  lastWeek: TimeRange;
  lastMonth: TimeRange;
  custom: TimeRange | null;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'metrics' | 'performance' | 'confidence' | 'error' | 'status';
  data: any;
  timestamp: Date;
  id: string;
}

export interface WebSocketConfig {
  url: string;
  reconnectAttempts: number;
  reconnectDelay: number;
  heartbeatInterval: number;
  timeout: number;
}

// Debug and Analysis Types
export interface DebugAction {
  type: 'analyze' | 'export' | 'reset' | 'simulate' | 'validate';
  parameters?: Record<string, any>;
  timestamp: Date;
}

export interface AnalysisResult {
  insights: string[];
  recommendations: string[];
  anomalies: Anomaly[];
  trends: TrendAnalysis[];
  confidence: number;
}

export interface Anomaly {
  type: 'performance' | 'confidence' | 'cost' | 'quality';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: Date;
  metrics: Record<string, number>;
  recommendations: string[];
}

export interface TrendAnalysis {
  metric: string;
  direction: 'increasing' | 'decreasing' | 'stable';
  rate: number;
  confidence: number;
  prediction: {
    next24h: number;
    nextWeek: number;
    uncertainty: number;
  };
}

// Export Data Types
export interface ExportData {
  format: 'json' | 'csv' | 'excel' | 'pdf';
  timeRange: TimeRange;
  metrics: SCWTMetrics[];
  performance: PerformanceMetrics[];
  confidence: ConfidenceMetrics[];
  metadata: {
    exportedAt: Date;
    exportedBy: string;
    version: string;
  };
}

// Error Types
export interface DeepConfError extends Error {
  code: 'WEBSOCKET_ERROR' | 'DATA_VALIDATION' | 'API_ERROR' | 'EXPORT_ERROR';
  details?: Record<string, any>;
  timestamp: Date;
  recoverable: boolean;
}

// Component State Types
export interface DashboardState {
  data: RealTimeData | null;
  loading: boolean;
  error: DeepConfError | null;
  config: DashboardConfig;
  selectedTimeRange: TimeRange;
  selectedMetrics: (keyof SCWTMetrics)[];
}

export interface VisualizationState {
  chartType: 'line' | 'area' | 'bar' | 'heatmap';
  zoomLevel: number;
  selectedPoints: ChartDataPoint[];
  filters: Record<string, any>;
  annotations: ChartAnnotation[];
}

export interface ChartAnnotation {
  id: string;
  type: 'threshold' | 'event' | 'anomaly' | 'note';
  position: { x: number | Date; y: number };
  content: string;
  color: string;
  visible: boolean;
}

// Utility Types
export type MetricType = keyof SCWTMetrics;
export type ChartType = 'line' | 'area' | 'bar' | 'heatmap' | 'scatter';
export type ThemeMode = 'light' | 'dark' | 'auto';
export type RefreshMode = 'auto' | 'manual' | 'on-demand';

// Type Guards
export const isValidSCWTMetrics = (data: any): data is SCWTMetrics => {
  return (
    typeof data === 'object' &&
    typeof data.structuralWeight === 'number' &&
    typeof data.contextWeight === 'number' &&
    typeof data.temporalWeight === 'number' &&
    typeof data.combinedScore === 'number' &&
    data.timestamp instanceof Date
  );
};

export const isValidPerformanceMetrics = (data: any): data is PerformanceMetrics => {
  return (
    typeof data === 'object' &&
    typeof data.tokenEfficiency === 'object' &&
    typeof data.cost === 'object' &&
    typeof data.timing === 'object' &&
    typeof data.quality === 'object'
  );
};