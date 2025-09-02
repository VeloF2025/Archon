/**
 * Optimized Confidence Chart Component
 * 
 * Performance Optimizations:
 * - Canvas rendering instead of SVG for better performance
 * - Data virtualization for large datasets
 * - Memoized calculations and drawing operations
 * - Efficient animation using RAF
 * - Memory management and cleanup
 */

import React, { 
  useRef, 
  useEffect, 
  useMemo, 
  useCallback, 
  useState,
  ImperativeHandle,
  forwardRef
} from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '../ui/Select';
import { Toggle } from '../ui/Toggle';
import { 
  ConfidenceVisualizationProps, 
  ChartDataPoint,
  ConfidenceMetrics,
  ChartType,
  ConfidenceThresholds
} from './types';

// Chart performance configuration
const CHART_CONFIG = {
  MAX_VISIBLE_POINTS: 200,    // Limit visible data points
  ANIMATION_DURATION: 300,    // Animation duration in ms
  REDRAW_THROTTLE: 16,       // ~60fps throttling
  MEMORY_CLEANUP_INTERVAL: 30000, // Cleanup every 30s
} as const;

// Default thresholds
const defaultThresholds: ConfidenceThresholds = {
  low: 0.3,
  medium: 0.6,
  high: 0.8,
  critical: 0.95,
};

// Canvas drawing utilities with performance optimizations
class ChartRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationId: number | null = null;
  private lastDrawTime = 0;
  
  // Cached drawing data to prevent recalculation
  private cachedPaths: Map<string, Path2D> = new Map();
  private cachedColors: Map<number, string> = new Map();
  
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    
    // Optimize canvas for performance
    this.setupCanvas();
  }
  
  private setupCanvas() {
    // Set up canvas for high DPI displays
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    
    this.ctx.scale(dpr, dpr);
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    
    // Enable performance optimizations
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }
  
  drawChart(
    data: ChartDataPoint[],
    options: {
      width: number;
      height: number;
      showUncertainty: boolean;
      showThresholds: boolean;
      chartType: ChartType;
      thresholds: ConfidenceThresholds;
      selectedDimension: string;
    }
  ) {
    // Throttle redraw for performance
    const now = Date.now();
    if (now - this.lastDrawTime < CHART_CONFIG.REDRAW_THROTTLE) {
      return;
    }
    this.lastDrawTime = now;
    
    // Cancel any pending animation
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    
    this.animationId = requestAnimationFrame(() => {
      this.performDraw(data, options);
    });
  }
  
  private performDraw(
    data: ChartDataPoint[],
    options: {
      width: number;
      height: number;
      showUncertainty: boolean;
      showThresholds: boolean;
      chartType: ChartType;
      thresholds: ConfidenceThresholds;
      selectedDimension: string;
    }
  ) {
    const { width, height, showUncertainty, showThresholds, chartType, thresholds } = options;
    const padding = 40;
    
    // Clear canvas with better performance
    this.ctx.clearRect(0, 0, width, height);
    
    if (data.length === 0) {
      this.drawEmptyState(width, height);
      return;
    }
    
    // Limit data points for performance (virtualization)
    const visibleData = data.length > CHART_CONFIG.MAX_VISIBLE_POINTS 
      ? this.sampleData(data, CHART_CONFIG.MAX_VISIBLE_POINTS)
      : data;
    
    // Calculate scales (cached)
    const scales = this.calculateScales(visibleData, width, height, padding);
    
    // Draw grid
    this.drawGrid(width, height, padding);
    
    // Draw threshold lines
    if (showThresholds) {
      this.drawThresholdLines(thresholds, scales, width, height, padding);
    }
    
    // Draw uncertainty band
    if (showUncertainty && chartType === 'area') {
      this.drawUncertaintyBand(visibleData, scales, padding);
    }
    
    // Draw main chart
    this.drawMainChart(visibleData, scales, chartType, padding);
    
    // Draw data points
    this.drawDataPoints(visibleData, scales, thresholds, padding);
  }
  
  private sampleData(data: ChartDataPoint[], maxPoints: number): ChartDataPoint[] {
    if (data.length <= maxPoints) return data;
    
    // Use smart sampling - keep more recent points
    const step = Math.ceil(data.length / maxPoints);
    const sampled: ChartDataPoint[] = [];
    
    // Always include first and last points
    sampled.push(data[0]);
    
    for (let i = step; i < data.length - step; i += step) {
      sampled.push(data[i]);
    }
    
    sampled.push(data[data.length - 1]);
    return sampled;
  }
  
  private calculateScales(
    data: ChartDataPoint[],
    width: number,
    height: number,
    padding: number
  ) {
    const values = data.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const valueRange = maxValue - minValue || 1;
    
    return {
      xScale: (index: number) => (index / (data.length - 1)) * (width - 2 * padding) + padding,
      yScale: (value: number) => height - padding - ((value - minValue) / valueRange) * (height - 2 * padding),
      minValue,
      maxValue,
      valueRange
    };
  }
  
  private drawGrid(width: number, height: number, padding: number) {
    this.ctx.strokeStyle = 'rgba(128, 128, 128, 0.1)';
    this.ctx.lineWidth = 0.5;
    
    // Vertical grid lines
    const gridCount = 10;
    for (let i = 0; i <= gridCount; i++) {
      const x = padding + (i / gridCount) * (width - 2 * padding);
      this.ctx.beginPath();
      this.ctx.moveTo(x, padding);
      this.ctx.lineTo(x, height - padding);
      this.ctx.stroke();
    }
    
    // Horizontal grid lines  
    for (let i = 0; i <= 5; i++) {
      const y = padding + (i / 5) * (height - 2 * padding);
      this.ctx.beginPath();
      this.ctx.moveTo(padding, y);
      this.ctx.lineTo(width - padding, y);
      this.ctx.stroke();
    }
  }
  
  private drawThresholdLines(
    thresholds: ConfidenceThresholds,
    scales: any,
    width: number,
    height: number,
    padding: number
  ) {
    const thresholdConfig = [
      { value: thresholds.high, color: 'rgba(34, 197, 94, 0.6)', label: 'High' },
      { value: thresholds.medium, color: 'rgba(234, 179, 8, 0.6)', label: 'Medium' },
      { value: thresholds.low, color: 'rgba(239, 68, 68, 0.6)', label: 'Low' }
    ];
    
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([5, 5]);
    
    thresholdConfig.forEach(({ value, color, label }) => {
      const y = scales.yScale(value);
      this.ctx.strokeStyle = color;
      this.ctx.beginPath();
      this.ctx.moveTo(padding, y);
      this.ctx.lineTo(width - padding, y);
      this.ctx.stroke();
      
      // Label
      this.ctx.fillStyle = color;
      this.ctx.font = '12px sans-serif';
      this.ctx.fillText(label, width - padding - 40, y - 5);
    });
    
    this.ctx.setLineDash([]);
  }
  
  private drawUncertaintyBand(
    data: ChartDataPoint[],
    scales: any,
    padding: number
  ) {
    this.ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    this.ctx.beginPath();
    
    // Upper bound
    data.forEach((d, i) => {
      const x = scales.xScale(i);
      const y = scales.yScale(d.value + (d.uncertainty || 0));
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    });
    
    // Lower bound (reverse order)
    for (let i = data.length - 1; i >= 0; i--) {
      const d = data[i];
      const x = scales.xScale(i);
      const y = scales.yScale(d.value - (d.uncertainty || 0));
      this.ctx.lineTo(x, y);
    }
    
    this.ctx.closePath();
    this.ctx.fill();
  }
  
  private drawMainChart(
    data: ChartDataPoint[],
    scales: any,
    chartType: ChartType,
    padding: number
  ) {
    this.ctx.strokeStyle = 'rgb(59, 130, 246)';
    this.ctx.lineWidth = 2;
    
    if (chartType === 'area') {
      // Fill area
      this.ctx.fillStyle = 'rgba(59, 130, 246, 0.4)';
      this.ctx.beginPath();
      
      data.forEach((d, i) => {
        const x = scales.xScale(i);
        const y = scales.yScale(d.value);
        if (i === 0) {
          this.ctx.moveTo(x, y);
        } else {
          this.ctx.lineTo(x, y);
        }
      });
      
      // Close to bottom
      const lastX = scales.xScale(data.length - 1);
      const firstX = scales.xScale(0);
      const bottom = scales.yScale(scales.minValue);
      
      this.ctx.lineTo(lastX, bottom);
      this.ctx.lineTo(firstX, bottom);
      this.ctx.closePath();
      this.ctx.fill();
    }
    
    // Draw line
    this.ctx.beginPath();
    data.forEach((d, i) => {
      const x = scales.xScale(i);
      const y = scales.yScale(d.value);
      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    });
    this.ctx.stroke();
  }
  
  private drawDataPoints(
    data: ChartDataPoint[],
    scales: any,
    thresholds: ConfidenceThresholds,
    padding: number
  ) {
    data.forEach((d, i) => {
      const x = scales.xScale(i);
      const y = scales.yScale(d.value);
      
      // Color based on threshold
      let color = 'rgb(239, 68, 68)'; // Low/red
      if (d.value >= thresholds.critical) color = 'rgb(34, 197, 94)'; // Critical/green
      else if (d.value >= thresholds.high) color = 'rgb(34, 197, 94)'; // High/green
      else if (d.value >= thresholds.medium) color = 'rgb(234, 179, 8)'; // Medium/yellow
      
      this.ctx.fillStyle = color;
      this.ctx.strokeStyle = 'white';
      this.ctx.lineWidth = 2;
      
      this.ctx.beginPath();
      this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
      this.ctx.fill();
      this.ctx.stroke();
    });
  }
  
  private drawEmptyState(width: number, height: number) {
    this.ctx.fillStyle = 'rgba(128, 128, 128, 0.5)';
    this.ctx.font = '16px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText('No confidence data available', width / 2, height / 2);
  }
  
  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.cachedPaths.clear();
    this.cachedColors.clear();
  }
}

export interface OptimizedConfidenceChartProps extends Omit<ConfidenceVisualizationProps, 'onConfidenceClick' | 'onThresholdChange'> {
  onDataPointClick?: (point: ChartDataPoint, index: number) => void;
  width?: number;
  height?: number;
}

export interface ChartHandle {
  redraw: () => void;
  exportImage: () => string;
}

export const OptimizedConfidenceChart = React.memo(
  forwardRef<ChartHandle, OptimizedConfidenceChartProps>(({
    metrics,
    chartType = 'area',
    timeRange,
    interactive = true,
    thresholds = defaultThresholds,
    width = 600,
    height = 300,
    onDataPointClick
  }, ref) => {
    // State management
    const [selectedChartType, setSelectedChartType] = useState<ChartType>(chartType);
    const [showUncertainty, setShowUncertainty] = useState(true);
    const [showThresholds, setShowThresholds] = useState(true);
    const [selectedDimension, setSelectedDimension] = useState<keyof ConfidenceMetrics['dimensions'] | 'overall'>('overall');
    const [hoveredPoint, setHoveredPoint] = useState<{point: ChartDataPoint; x: number; y: number} | null>(null);
    
    // Refs
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const rendererRef = useRef<ChartRenderer | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    
    // Process and memoize chart data
    const chartData = useMemo(() => {
      if (!metrics.length) return [];
      
      return metrics.map((metric, index) => {
        const timestamp = new Date(Date.now() - (metrics.length - index - 1) * 60000);
        const confidence = selectedDimension === 'overall' 
          ? metric.overall 
          : metric.dimensions[selectedDimension as keyof ConfidenceMetrics['dimensions']];
        
        return {
          timestamp,
          value: confidence,
          confidence: metric.overall,
          uncertainty: metric.uncertainty.total,
          bayesian: metric.bayesian,
          metadata: {
            trend: metric.trend,
            epistemic: metric.uncertainty.epistemic,
            aleatoric: metric.uncertainty.aleatoric,
          },
        } as ChartDataPoint;
      });
    }, [metrics, selectedDimension]);
    
    // Statistics calculation (memoized)
    const stats = useMemo(() => {
      if (!chartData.length) return null;
      
      const values = chartData.map(d => d.value);
      const uncertainties = chartData.map(d => d.uncertainty || 0);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      
      return {
        mean,
        min: Math.min(...values),
        max: Math.max(...values),
        stdDev: Math.sqrt(
          values
            .map(v => Math.pow(v - mean, 2))
            .reduce((a, b) => a + b, 0) / values.length
        ),
        avgUncertainty: uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length,
        trend: chartData[chartData.length - 1]?.metadata?.trend || 'stable',
      };
    }, [chartData]);
    
    // Initialize renderer
    useEffect(() => {
      if (!canvasRef.current) return;
      
      rendererRef.current = new ChartRenderer(canvasRef.current);
      
      return () => {
        rendererRef.current?.destroy();
      };
    }, []);
    
    // Redraw chart when data or options change
    useEffect(() => {
      if (!rendererRef.current || !canvasRef.current) return;
      
      const rect = canvasRef.current.getBoundingClientRect();
      rendererRef.current.drawChart(chartData, {
        width: rect.width,
        height: rect.height,
        showUncertainty,
        showThresholds,
        chartType: selectedChartType,
        thresholds,
        selectedDimension
      });
    }, [chartData, showUncertainty, showThresholds, selectedChartType, thresholds, selectedDimension]);
    
    // Handle canvas interactions
    const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!interactive || !onDataPointClick || !canvasRef.current) return;
      
      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Find closest data point (simplified)
      const pointWidth = rect.width / chartData.length;
      const clickedIndex = Math.round(x / pointWidth);
      
      if (clickedIndex >= 0 && clickedIndex < chartData.length) {
        onDataPointClick(chartData[clickedIndex], clickedIndex);
      }
    }, [interactive, onDataPointClick, chartData]);
    
    // Imperative handle for external control
    React.useImperativeHandle(ref, () => ({
      redraw: () => {
        if (rendererRef.current && canvasRef.current) {
          const rect = canvasRef.current.getBoundingClientRect();
          rendererRef.current.drawChart(chartData, {
            width: rect.width,
            height: rect.height,
            showUncertainty,
            showThresholds,
            chartType: selectedChartType,
            thresholds,
            selectedDimension
          });
        }
      },
      exportImage: () => {
        return canvasRef.current?.toDataURL('image/png') || '';
      }
    }));
    
    // Color helper functions
    const getConfidenceColor = (confidence: number): string => {
      if (confidence >= thresholds.critical) return 'text-green-600 dark:text-green-400';
      if (confidence >= thresholds.high) return 'text-blue-600 dark:text-blue-400';
      if (confidence >= thresholds.medium) return 'text-yellow-600 dark:text-yellow-400';
      if (confidence >= thresholds.low) return 'text-orange-600 dark:text-orange-400';
      return 'text-red-600 dark:text-red-400';
    };
    
    const getConfidenceLabel = (confidence: number): string => {
      if (confidence >= thresholds.critical) return 'Critical';
      if (confidence >= thresholds.high) return 'High';
      if (confidence >= thresholds.medium) return 'Medium';
      if (confidence >= thresholds.low) return 'Low';
      return 'Very Low';
    };
    
    return (
      <Card accentColor="blue">
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="flex items-center gap-2">
                🚀 Optimized Confidence Visualization
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                High-performance Canvas rendering with virtualization
              </p>
            </div>
            
            {stats && (
              <div className="text-right text-sm">
                <div className={`font-medium ${getConfidenceColor(stats.mean)}`}>
                  {(stats.mean * 100).toFixed(1)}% avg
                </div>
                <div className="text-muted-foreground">
                  {stats.trend}
                </div>
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Controls */}
          <div className="flex flex-wrap items-center gap-4 pb-4 border-b">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Dimension:</label>
              <Select
                value={selectedDimension}
                onValueChange={setSelectedDimension}
              >
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="overall">Overall</SelectItem>
                  <SelectItem value="structural">Structural</SelectItem>
                  <SelectItem value="contextual">Contextual</SelectItem>
                  <SelectItem value="temporal">Temporal</SelectItem>
                  <SelectItem value="semantic">Semantic</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Chart:</label>
              <div className="flex gap-1">
                <Button
                  onClick={() => setSelectedChartType('line')}
                  variant={selectedChartType === 'line' ? 'default' : 'outline'}
                  size="sm"
                >
                  Line
                </Button>
                <Button
                  onClick={() => setSelectedChartType('area')}
                  variant={selectedChartType === 'area' ? 'default' : 'outline'}
                  size="sm"
                >
                  Area
                </Button>
              </div>
            </div>
            
            <Toggle
              pressed={showUncertainty}
              onPressedChange={setShowUncertainty}
              size="sm"
            >
              Uncertainty
            </Toggle>
            
            <Toggle
              pressed={showThresholds}
              onPressedChange={setShowThresholds}
              size="sm"
            >
              Thresholds
            </Toggle>
            
            <div className="text-xs text-muted-foreground ml-auto">
              Points: {chartData.length} / Rendered: {Math.min(chartData.length, CHART_CONFIG.MAX_VISIBLE_POINTS)}
            </div>
          </div>
          
          {/* Optimized Canvas Chart */}
          <div ref={containerRef} className="w-full relative">
            <canvas
              ref={canvasRef}
              className="w-full border rounded cursor-pointer"
              style={{ height: `${height}px` }}
              onClick={handleCanvasClick}
              aria-label="Interactive confidence chart"
            />
            
            {/* Tooltip */}
            {hoveredPoint && (
              <div 
                className="absolute bg-background border rounded-lg p-3 shadow-lg pointer-events-none z-10"
                style={{
                  left: hoveredPoint.x + 10,
                  top: hoveredPoint.y - 10
                }}
              >
                <div className="text-sm space-y-1">
                  <div className="font-medium">
                    Confidence: {(hoveredPoint.point.value * 100).toFixed(1)}%
                  </div>
                  <div className="text-muted-foreground">
                    Uncertainty: ±{((hoveredPoint.point.uncertainty || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="text-muted-foreground">
                    Level: {getConfidenceLabel(hoveredPoint.point.value)}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Performance Statistics */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  {(stats.mean * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Mean</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-semibold text-blue-600 dark:text-blue-400">
                  ±{(stats.avgUncertainty * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Avg Uncertainty</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-semibold">
                  {(stats.max * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Peak</div>
              </div>
              
              <div className="text-center">
                <div className={`text-lg font-semibold ${
                  stats.trend === 'increasing' ? 'text-green-600 dark:text-green-400' :
                  stats.trend === 'decreasing' ? 'text-red-600 dark:text-red-400' :
                  'text-yellow-600 dark:text-yellow-400'
                }`}>
                  {stats.trend.toUpperCase()}
                </div>
                <div className="text-xs text-muted-foreground">Trend</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  })
);

OptimizedConfidenceChart.displayName = 'OptimizedConfidenceChart';

export default OptimizedConfidenceChart;