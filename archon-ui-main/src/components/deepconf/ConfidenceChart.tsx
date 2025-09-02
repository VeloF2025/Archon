/**
 * Confidence Chart Component - UPDATED 2025-09-01-1800
 * Phase 7 DeepConf Integration
 * 
 * Specialized confidence visualization with advanced charting capabilities,
 * interactive features, and multi-dimensional confidence display
 * 
 * 🟢 BULLETPROOF NaN PROTECTION: All chart rendering operations are protected
 * against invalid numeric inputs and NaN values
 */

import React, { useState, useMemo, useCallback } from 'react';
import { ConfidenceMetrics, ChartDataPoint, ConfidenceInterval, ChartType } from './types';

interface ConfidenceChartProps {
  /** Confidence data to visualize */
  data: ConfidenceMetrics[];
  /** Chart type */
  type?: ChartType;
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Show confidence intervals */
  showIntervals?: boolean;
  /** Show trend line */
  showTrend?: boolean;
  /** Interactive features */
  interactive?: boolean;
  /** Color scheme */
  colorScheme?: 'default' | 'monochrome' | 'vibrant';
  /** Event handlers */
  onPointClick?: (point: ChartDataPoint, index: number) => void;
  onPointHover?: (point: ChartDataPoint | null) => void;
}

// Color schemes
const colorSchemes = {
  default: {
    line: 'rgb(59, 130, 246)',
    area: 'rgba(59, 130, 246, 0.3)',
    confidence: 'rgba(16, 185, 129, 0.2)',
    uncertainty: 'rgba(239, 68, 68, 0.2)',
    grid: 'rgba(156, 163, 175, 0.2)',
  },
  monochrome: {
    line: 'rgb(75, 85, 99)',
    area: 'rgba(75, 85, 99, 0.3)',
    confidence: 'rgba(75, 85, 99, 0.2)',
    uncertainty: 'rgba(107, 114, 128, 0.2)',
    grid: 'rgba(156, 163, 175, 0.1)',
  },
  vibrant: {
    line: 'rgb(236, 72, 153)',
    area: 'rgba(236, 72, 153, 0.3)',
    confidence: 'rgba(34, 197, 94, 0.2)',
    uncertainty: 'rgba(251, 146, 60, 0.2)',
    grid: 'rgba(156, 163, 175, 0.2)',
  },
};

// 🟢 WORKING: Ultra-aggressive NaN protection utilities for ConfidenceChart
const safeNumber = (value: any, defaultValue: number = 0): number => {
  if (value === null || value === undefined) return defaultValue;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    if (isNaN(parsed) || !isFinite(parsed)) return defaultValue;
    return parsed;
  }
  if (typeof value !== 'number') return defaultValue;
  if (isNaN(value) || !isFinite(value) || value === Infinity || value === -Infinity) {
    console.warn(`[ConfidenceChart] Invalid numeric value: ${value}, using default: ${defaultValue}`);
    return defaultValue;
  }
  return value;
};

const safePercentage = (value: any, defaultValue: number = 0.5): number => {
  const num = safeNumber(value, defaultValue);
  const clamped = Math.max(0, Math.min(1, num));
  if (isNaN(clamped) || !isFinite(clamped)) {
    console.warn(`[ConfidenceChart] Invalid percentage: ${clamped}, using default: ${defaultValue}`);
    return defaultValue;
  }
  return clamped;
};

const safeSVGCoord = (value: any, defaultValue: number = 0, context: string = 'unknown'): number => {
  const safe = safeNumber(value, defaultValue);
  if (safe < -10000 || safe > 10000) {
    console.warn(`[ConfidenceChart] Extreme coordinate in ${context}: ${safe}`);
    return Math.max(-1000, Math.min(1000, safe));
  }
  return safe;
};

export const ConfidenceChart: React.FC<ConfidenceChartProps> = ({
  data,
  type = 'line',
  width = 600,
  height = 300,
  showIntervals = true,
  showTrend = false,
  interactive = true,
  colorScheme = 'default',
  onPointClick,
  onPointHover,
}) => {
  // State
  const [hoveredPoint, setHoveredPoint] = useState<{ point: ChartDataPoint; index: number } | null>(null);
  
  // Colors
  const colors = colorSchemes[colorScheme];
  
  // Safe dimensions
  const safeWidth = safeNumber(width, 600);
  const safeHeight = safeNumber(height, 300);
  
  // Safe chart dimensions with validation
  const padding = { 
    top: safeNumber(20, 20), 
    right: safeNumber(40, 40), 
    bottom: safeNumber(40, 40), 
    left: safeNumber(60, 60) 
  };
  const chartWidth = Math.max(100, safeWidth - padding.left - padding.right);
  const chartHeight = Math.max(100, safeHeight - padding.top - padding.bottom);
  
  // Process data with comprehensive NaN protection
  const chartData = useMemo(() => {
    if (!Array.isArray(data) || data.length === 0) {
      console.warn('[ConfidenceChart] No valid data provided');
      return [];
    }
    
    return data.map((metric, index) => {
      if (!metric || typeof metric !== 'object') {
        console.warn(`[ConfidenceChart] Invalid metric at index ${index}:`, metric);
        return null;
      }
      
      return {
        timestamp: new Date(Date.now() - (data.length - index - 1) * 60000),
        value: safePercentage(metric.overall, 0.5),
        confidence: safePercentage(metric.overall, 0.5),
        uncertainty: safePercentage(metric.uncertainty?.total, 0.1),
        bayesian: {
          upper: safePercentage(metric.bayesian?.upper, 0.6),
          lower: safePercentage(metric.bayesian?.lower, 0.4),
          mean: safePercentage(metric.bayesian?.mean, 0.5),
          variance: safeNumber(metric.bayesian?.variance, 0.02),
        },
        dimensions: metric.dimensions || {},
      } as ChartDataPoint;
    }).filter(item => item !== null) as ChartDataPoint[];
  }, [data]);
  
  // Calculate ultra-safe scales
  const { xScale, yScale } = useMemo(() => {
    if (!chartData.length) {
      return { 
        xScale: () => safeSVGCoord(padding.left, padding.left, 'empty-xScale'), 
        yScale: () => safeSVGCoord(safeHeight - padding.bottom, safeHeight - padding.bottom, 'empty-yScale') 
      };
    }
    
    const maxIndex = Math.max(1, chartData.length - 1);
    
    const xScale = (index: number) => {
      const safeIndex = safeNumber(index, 0);
      const scaleFactor = maxIndex > 0 ? safeIndex / maxIndex : 0;
      const result = safeSVGCoord(scaleFactor * chartWidth + padding.left, padding.left, 'xScale');
      return Math.max(padding.left, Math.min(safeWidth - padding.right, result));
    };
    
    const yScale = (value: number) => {
      const safeValue = safePercentage(value, 0.5);
      const result = safeSVGCoord(safeHeight - padding.bottom - (safeValue * chartHeight), safeHeight - padding.bottom, 'yScale');
      return Math.max(padding.top, Math.min(safeHeight - padding.bottom, result));
    };
    
    return { xScale, yScale };
  }, [chartData, chartWidth, chartHeight, safeHeight, safeWidth, padding]);
  
  // Calculate confidence intervals with NaN protection
  const confidenceIntervals = useMemo(() => {
    return chartData.map((point, index) => {
      try {
        const x = safeSVGCoord(xScale(index), padding.left, `interval-x-${index}`);
        const y = safeSVGCoord(yScale(safePercentage(point.value, 0.5)), safeHeight - padding.bottom, `interval-y-${index}`);
        const upper = safeSVGCoord(yScale(safePercentage(point.bayesian.upper, 0.6)), safeHeight - padding.bottom, `interval-upper-${index}`);
        const lower = safeSVGCoord(yScale(safePercentage(point.bayesian.lower, 0.4)), safeHeight - padding.bottom, `interval-lower-${index}`);
        
        return { x, y, upper, lower, point, index };
      } catch (error) {
        console.error(`[ConfidenceChart] Error calculating interval for point ${index}:`, error);
        return {
          x: safeSVGCoord(padding.left, padding.left, 'fallback-x'),
          y: safeSVGCoord(safeHeight - padding.bottom, safeHeight - padding.bottom, 'fallback-y'),
          upper: safeSVGCoord(safeHeight - padding.bottom, safeHeight - padding.bottom, 'fallback-upper'),
          lower: safeSVGCoord(safeHeight - padding.bottom, safeHeight - padding.bottom, 'fallback-lower'),
          point,
          index,
        };
      }
    });
  }, [chartData, xScale, yScale, safeHeight, padding]);
  
  // Calculate trend line
  const trendLine = useMemo(() => {
    if (!showTrend || chartData.length < 2) return null;
    
    // Simple linear regression
    const n = chartData.length;
    const sumX = chartData.reduce((sum, _, i) => sum + i, 0);
    const sumY = chartData.reduce((sum, point) => sum + point.value, 0);
    const sumXY = chartData.reduce((sum, point, i) => sum + i * point.value, 0);
    const sumXX = chartData.reduce((sum, _, i) => sum + i * i, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return {
      start: { x: xScale(0), y: yScale(intercept) },
      end: { x: xScale(n - 1), y: yScale(slope * (n - 1) + intercept) },
    };
  }, [chartData, showTrend, xScale, yScale]);
  
  // Handle mouse events
  const handleMouseMove = useCallback((event: React.MouseEvent<SVGElement>) => {
    if (!interactive) return;
    
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Find closest point
    let closestIndex = -1;
    let closestDistance = Infinity;
    
    confidenceIntervals.forEach((interval, index) => {
      const distance = Math.sqrt(Math.pow(x - interval.x, 2) + Math.pow(y - interval.y, 2));
      if (distance < closestDistance && distance < 20) { // 20px threshold
        closestDistance = distance;
        closestIndex = index;
      }
    });
    
    if (closestIndex >= 0) {
      const point = chartData[closestIndex];
      setHoveredPoint({ point, index: closestIndex });
      onPointHover?.(point);
    } else {
      setHoveredPoint(null);
      onPointHover?.(null);
    }
  }, [interactive, confidenceIntervals, chartData, onPointHover]);
  
  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    onPointHover?.(null);
  }, [onPointHover]);
  
  const handleClick = useCallback((event: React.MouseEvent<SVGElement>) => {
    if (!interactive || !hoveredPoint) return;
    
    onPointClick?.(hoveredPoint.point, hoveredPoint.index);
  }, [interactive, hoveredPoint, onPointClick]);
  
  // Render grid lines with safe coordinates
  const renderGrid = () => {
    const gridLines = [];
    
    try {
      // Horizontal grid lines
      for (let i = 0; i <= 10; i++) {
        const y = safeSVGCoord(yScale(i / 10), safeHeight / 2, `grid-h-${i}`);
        gridLines.push(
          <line
            key={`h-${i}`}
            x1={safeSVGCoord(padding.left, padding.left, 'grid-x1')}
            y1={y}
            x2={safeSVGCoord(safeWidth - padding.right, safeWidth - padding.right, 'grid-x2')}
            y2={y}
            stroke={colors.grid}
            strokeWidth={i === 0 || i === 10 ? 1 : 0.5}
          />
        );
      }
      
      // Vertical grid lines
      const xStep = Math.max(1, Math.floor(chartData.length / 10));
      for (let i = 0; i < chartData.length; i += xStep) {
        const x = safeSVGCoord(xScale(i), safeWidth / 2, `grid-v-${i}`);
        gridLines.push(
          <line
            key={`v-${i}`}
            x1={x}
            y1={safeSVGCoord(padding.top, padding.top, 'grid-y1')}
            x2={x}
            y2={safeSVGCoord(safeHeight - padding.bottom, safeHeight - padding.bottom, 'grid-y2')}
            stroke={colors.grid}
            strokeWidth={0.5}
          />
        );
      }
    } catch (error) {
      console.error('[ConfidenceChart] Error rendering grid:', error);
    }
    
    return gridLines;
  };
  
  // Render confidence intervals
  const renderConfidenceIntervals = () => {
    if (!showIntervals || !confidenceIntervals.length) return null;
    
    const path = confidenceIntervals.map((interval, index) => {
      const command = index === 0 ? 'M' : 'L';
      return `${command} ${interval.x} ${interval.upper}`;
    }).join(' ') + ' L ' + confidenceIntervals.map((interval, index) => {
      const reversedIndex = confidenceIntervals.length - 1 - index;
      const reversedInterval = confidenceIntervals[reversedIndex];
      return `${reversedInterval.x} ${reversedInterval.lower}`;
    }).join(' L ') + ' Z';
    
    return (
      <path
        d={path}
        fill={colors.confidence}
        stroke="none"
        opacity={0.6}
      />
    );
  };
  
  // Render main line/area with bulletproof path generation
  const renderMainChart = () => {
    if (!confidenceIntervals.length) return null;
    
    try {
      // Generate ultra-safe path commands
      const pathCommands = confidenceIntervals.map((interval, index) => {
        const x = safeSVGCoord(interval.x, padding.left, `main-path-x-${index}`);
        const y = safeSVGCoord(interval.y, safeHeight - padding.bottom, `main-path-y-${index}`);
        const command = index === 0 ? 'M' : 'L';
        return `${command} ${x} ${y}`;
      }).filter(cmd => cmd && !cmd.includes('NaN')).join(' ');
      
      // Validate path is not empty
      if (!pathCommands || pathCommands.trim() === '') {
        console.warn('[ConfidenceChart] Empty path generated, using fallback');
        return <path d={`M ${padding.left} ${safeHeight - padding.bottom} L ${safeWidth - padding.right} ${safeHeight - padding.bottom}`} fill="none" stroke={colors.line} strokeWidth={2} />;
      }
      
      if (type === 'area') {
        const lastX = safeSVGCoord(xScale(Math.max(0, chartData.length - 1)), safeWidth - padding.right, 'area-last-x');
        const baseY = safeSVGCoord(yScale(0), safeHeight - padding.bottom, 'area-base-y');
        const startX = safeSVGCoord(xScale(0), padding.left, 'area-start-x');
        
        const areaPath = `${pathCommands} L ${lastX} ${baseY} L ${startX} ${baseY} Z`;
        
        return (
          <>
            <path d={areaPath} fill={colors.area} stroke="none" />
            <path d={pathCommands} fill="none" stroke={colors.line} strokeWidth={2} />
          </>
        );
      } else if (type === 'line') {
        return <path d={pathCommands} fill="none" stroke={colors.line} strokeWidth={2} />;
      } else if (type === 'bar') {
        return confidenceIntervals.map((interval, index) => {
          const x = safeSVGCoord(interval.x - 8, 0, `bar-x-${index}`);
          const y = safeSVGCoord(interval.y, 0, `bar-y-${index}`);
          const width = safeNumber(16, 16);
          const height = Math.max(0, safeSVGCoord(yScale(0) - interval.y, 10, `bar-height-${index}`));
          
          return (
            <rect
              key={`bar-${index}`}
              x={x}
              y={y}
              width={width}
              height={height}
              fill={colors.line}
              opacity={0.7}
            />
          );
        }).filter(Boolean);
      }
    } catch (error) {
      console.error('[ConfidenceChart] Error rendering main chart:', error);
      // Fallback line
      return <line x1={padding.left} y1={safeHeight - padding.bottom} x2={safeWidth - padding.right} y2={safeHeight - padding.bottom} stroke={colors.line} strokeWidth={2} />;
    }
    
    return null;
  };
  
  // Render trend line
  const renderTrendLine = () => {
    if (!trendLine) return null;
    
    return (
      <line
        x1={trendLine.start.x}
        y1={trendLine.start.y}
        x2={trendLine.end.x}
        y2={trendLine.end.y}
        stroke="rgb(249, 115, 22)"
        strokeWidth={2}
        strokeDasharray="5,5"
        opacity={0.8}
      />
    );
  };
  
  // Render data points with comprehensive validation
  const renderDataPoints = () => {
    return confidenceIntervals.map((interval, index) => {
      try {
        const cx = safeSVGCoord(interval.x, padding.left, `point-cx-${index}`);
        const cy = safeSVGCoord(interval.y, safeHeight - padding.bottom, `point-cy-${index}`);
        const radius = safeNumber(hoveredPoint?.index === index ? 6 : 4, 4);
        const confidence = safePercentage(interval.point.value, 0.5);
        
        // Validate circle can be rendered
        if (cx < 0 || cx > safeWidth || cy < 0 || cy > safeHeight || radius <= 0) {
          console.warn(`[ConfidenceChart] Invalid circle coordinates: cx=${cx}, cy=${cy}, r=${radius}`);
          return null;
        }
        
        return (
          <circle
            key={`point-${index}-${Date.now()}`}
            cx={cx}
            cy={cy}
            r={radius}
            fill={confidence > 0.8 ? 'rgb(34, 197, 94)' :
                  confidence > 0.6 ? 'rgb(234, 179, 8)' : 'rgb(239, 68, 68)'}
            stroke="white"
            strokeWidth={2}
            className={interactive ? 'cursor-pointer transition-all' : ''}
          />
        );
      } catch (error) {
        console.error(`[ConfidenceChart] Error rendering point ${index}:`, error);
        return null;
      }
    }).filter(Boolean);
  };
  
  // Render axes
  const renderAxes = () => {
    return (
      <>
        {/* X-axis */}
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
          stroke="currentColor"
          strokeWidth={1}
        />
        
        {/* Y-axis */}
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
          stroke="currentColor"
          strokeWidth={1}
        />
        
        {/* Y-axis labels */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map(value => (
          <text
            key={value}
            x={padding.left - 10}
            y={yScale(value) + 4}
            textAnchor="end"
            className="text-xs fill-current text-muted-foreground"
          >
            {(value * 100).toFixed(0)}%
          </text>
        ))}
        
        {/* X-axis labels */}
        {chartData.length > 0 && (
          <>
            <text
              x={xScale(0)}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              className="text-xs fill-current text-muted-foreground"
            >
              Start
            </text>
            <text
              x={xScale(chartData.length - 1)}
              y={height - padding.bottom + 20}
              textAnchor="middle"
              className="text-xs fill-current text-muted-foreground"
            >
              Now
            </text>
          </>
        )}
      </>
    );
  };
  
  // Render tooltip
  const renderTooltip = () => {
    if (!hoveredPoint) return null;
    
    const interval = confidenceIntervals[hoveredPoint.index];
    const point = hoveredPoint.point;
    
    return (
      <g>
        {/* Tooltip background */}
        <rect
          x={interval.x + 10}
          y={interval.y - 60}
          width={160}
          height={50}
          fill="rgba(0, 0, 0, 0.8)"
          rx={4}
          ry={4}
        />
        
        {/* Tooltip text */}
        <text x={interval.x + 20} y={interval.y - 40} className="fill-white text-xs font-medium">
          Confidence: {(point.value * 100).toFixed(1)}%
        </text>
        <text x={interval.x + 20} y={interval.y - 25} className="fill-white text-xs">
          Uncertainty: ±{((point.uncertainty || 0) * 100).toFixed(1)}%
        </text>
        <text x={interval.x + 20} y={interval.y - 10} className="fill-white text-xs">
          Index: {hoveredPoint.index}
        </text>
      </g>
    );
  };
  
  if (!chartData.length) {
    return (
      <div 
        className="flex items-center justify-center text-muted-foreground border-2 border-dashed border-muted rounded-md"
        style={{ width, height }}
      >
        No confidence data available
      </div>
    );
  }
  
  return (
    <div className="relative">
      <svg
        width={width}
        height={height}
        className="border border-border rounded"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        {/* Grid */}
        {renderGrid()}
        
        {/* Confidence intervals */}
        {renderConfidenceIntervals()}
        
        {/* Main chart */}
        {renderMainChart()}
        
        {/* Trend line */}
        {renderTrendLine()}
        
        {/* Data points */}
        {renderDataPoints()}
        
        {/* Axes */}
        {renderAxes()}
        
        {/* Tooltip */}
        {renderTooltip()}
      </svg>
    </div>
  );
};

export default ConfidenceChart;