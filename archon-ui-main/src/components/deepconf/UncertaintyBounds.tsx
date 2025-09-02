/**
 * Uncertainty Bounds Component
 * Phase 7 DeepConf Integration
 * 
 * Bayesian uncertainty visualization with epistemic and aleatoric
 * uncertainty decomposition and confidence interval display
 */

import React, { useState, useMemo, useCallback } from 'react';
import { ConfidenceMetrics, ConfidenceInterval, ChartDataPoint } from './types';

interface UncertaintyBoundsProps {
  /** Confidence metrics data */
  data: ConfidenceMetrics[];
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Show epistemic uncertainty separately */
  showEpistemic?: boolean;
  /** Show aleatoric uncertainty separately */
  showAleatoric?: boolean;
  /** Confidence level for intervals (e.g., 0.95 for 95% CI) */
  confidenceLevel?: number;
  /** Color scheme */
  colorScheme?: 'uncertainty' | 'probability' | 'medical';
  /** Interactive features */
  interactive?: boolean;
  /** Event handlers */
  onBoundsClick?: (point: ChartDataPoint, bounds: ConfidenceInterval) => void;
  onUncertaintyHover?: (uncertainty: { epistemic: number; aleatoric: number; total: number } | null) => void;
}

// Color schemes for different uncertainty visualization contexts
const colorSchemes = {
  uncertainty: {
    epistemic: 'rgba(59, 130, 246, 0.4)',     // Blue - model uncertainty
    aleatoric: 'rgba(239, 68, 68, 0.4)',      // Red - data uncertainty
    total: 'rgba(107, 114, 128, 0.3)',        // Gray - combined
    bounds: 'rgba(34, 197, 94, 0.2)',         // Green - confidence bounds
    mean: 'rgb(34, 197, 94)',                 // Green - mean line
  },
  probability: {
    epistemic: 'rgba(168, 85, 247, 0.4)',     // Purple - model uncertainty
    aleatoric: 'rgba(249, 115, 22, 0.4)',     // Orange - data uncertainty  
    total: 'rgba(156, 163, 175, 0.3)',        // Gray - combined
    bounds: 'rgba(59, 130, 246, 0.2)',        // Blue - confidence bounds
    mean: 'rgb(59, 130, 246)',                // Blue - mean line
  },
  medical: {
    epistemic: 'rgba(220, 38, 127, 0.4)',     // Pink - model uncertainty
    aleatoric: 'rgba(234, 179, 8, 0.4)',      // Yellow - data uncertainty
    total: 'rgba(75, 85, 99, 0.3)',           // Dark gray - combined
    bounds: 'rgba(16, 185, 129, 0.2)',        // Teal - confidence bounds
    mean: 'rgb(16, 185, 129)',                // Teal - mean line
  },
};

export const UncertaintyBounds: React.FC<UncertaintyBoundsProps> = ({
  data,
  width = 600,
  height = 400,
  showEpistemic = true,
  showAleatoric = true,
  confidenceLevel = 0.95,
  colorScheme = 'uncertainty',
  interactive = true,
  onBoundsClick,
  onUncertaintyHover,
}) => {
  // State
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);
  const [selectedUncertaintyType, setSelectedUncertaintyType] = useState<'total' | 'epistemic' | 'aleatoric'>('total');
  
  // Colors
  const colors = colorSchemes[colorScheme];
  
  // Chart dimensions
  const padding = { top: 30, right: 50, bottom: 50, left: 70 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  
  // Process data for uncertainty visualization
  const uncertaintyData = useMemo(() => {
    return data.map((metric, index) => {
      // Calculate confidence intervals based on Bayesian statistics
      const mean = metric.bayesian.mean;
      const variance = metric.bayesian.variance;
      const stdDev = Math.sqrt(variance);
      
      // Z-score for confidence level (approximation for normal distribution)
      const zScore = confidenceLevel === 0.95 ? 1.96 : 
                    confidenceLevel === 0.99 ? 2.58 : 
                    confidenceLevel === 0.90 ? 1.645 : 1.96;
      
      return {
        timestamp: metric.timestamp || new Date(), // Use real timestamp from metric data
        mean,
        variance,
        stdDev,
        epistemic: metric.uncertainty.epistemic,
        aleatoric: metric.uncertainty.aleatoric,
        total: metric.uncertainty.total,
        bayesian: {
          lower: Math.max(0, mean - zScore * stdDev),
          upper: Math.min(1, mean + zScore * stdDev),
          mean,
          variance,
        },
        // Enhanced bounds with uncertainty decomposition
        bounds: {
          epistemicLower: Math.max(0, mean - metric.uncertainty.epistemic),
          epistemicUpper: Math.min(1, mean + metric.uncertainty.epistemic),
          aleatoricLower: Math.max(0, mean - metric.uncertainty.aleatoric),
          aleatoricUpper: Math.min(1, mean + metric.uncertainty.aleatoric),
          totalLower: Math.max(0, mean - metric.uncertainty.total),
          totalUpper: Math.min(1, mean + metric.uncertainty.total),
        },
      };
    });
  }, [data, confidenceLevel]);
  
  // Calculate scales
  const { xScale, yScale } = useMemo(() => {
    if (!uncertaintyData.length) return { xScale: () => 0, yScale: () => 0 };
    
    const xScale = (index: number) => (index / (uncertaintyData.length - 1)) * chartWidth + padding.left;
    const yScale = (value: number) => height - padding.bottom - (value * chartHeight);
    
    return { xScale, yScale };
  }, [uncertaintyData, chartWidth, chartHeight, height, padding]);
  
  // Generate uncertainty bounds paths
  const uncertaintyPaths = useMemo(() => {
    if (!uncertaintyData.length) return { epistemic: '', aleatoric: '', total: '', bayesian: '' };
    
    const createBoundsPath = (getLower: (d: any) => number, getUpper: (d: any) => number) => {
      const upperPath = uncertaintyData.map((d, i) => {
        const x = xScale(i);
        const y = yScale(getUpper(d));
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      }).join(' ');
      
      const lowerPath = uncertaintyData.slice().reverse().map((d, i) => {
        const originalIndex = uncertaintyData.length - 1 - i;
        const x = xScale(originalIndex);
        const y = yScale(getLower(d));
        return `L ${x} ${y}`;
      }).join(' ');
      
      return upperPath + ' ' + lowerPath + ' Z';
    };
    
    return {
      epistemic: createBoundsPath(d => d.bounds.epistemicLower, d => d.bounds.epistemicUpper),
      aleatoric: createBoundsPath(d => d.bounds.aleatoricLower, d => d.bounds.aleatoricUpper),
      total: createBoundsPath(d => d.bounds.totalLower, d => d.bounds.totalUpper),
      bayesian: createBoundsPath(d => d.bayesian.lower, d => d.bayesian.upper),
    };
  }, [uncertaintyData, xScale, yScale]);
  
  // Mean line path
  const meanLinePath = useMemo(() => {
    if (!uncertaintyData.length) return '';
    
    return uncertaintyData.map((d, i) => {
      const x = xScale(i);
      const y = yScale(d.mean);
      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  }, [uncertaintyData, xScale, yScale]);
  
  // Handle mouse events
  const handleMouseMove = useCallback((event: React.MouseEvent<SVGElement>) => {
    if (!interactive) return;
    
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    
    // Find closest point
    let closestIndex = -1;
    let closestDistance = Infinity;
    
    uncertaintyData.forEach((_, index) => {
      const pointX = xScale(index);
      const distance = Math.abs(x - pointX);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestIndex = index;
      }
    });
    
    if (closestIndex >= 0 && closestDistance < 30) {
      setHoveredPoint(closestIndex);
      const point = uncertaintyData[closestIndex];
      onUncertaintyHover?.({
        epistemic: point.epistemic,
        aleatoric: point.aleatoric,
        total: point.total,
      });
    } else {
      setHoveredPoint(null);
      onUncertaintyHover?.(null);
    }
  }, [interactive, uncertaintyData, xScale, onUncertaintyHover]);
  
  const handleMouseLeave = useCallback(() => {
    setHoveredPoint(null);
    onUncertaintyHover?.(null);
  }, [onUncertaintyHover]);
  
  const handleClick = useCallback((event: React.MouseEvent<SVGElement>) => {
    if (!interactive || hoveredPoint === null) return;
    
    const point = uncertaintyData[hoveredPoint];
    const chartPoint: ChartDataPoint = {
      timestamp: point.timestamp,
      value: point.mean,
      confidence: point.mean,
      uncertainty: point.total,
      bayesian: point.bayesian,
    };
    
    const bounds: ConfidenceInterval = {
      timestamp: point.timestamp,
      mean: point.mean,
      lower: point.bayesian.lower,
      upper: point.bayesian.upper,
      uncertainty: point.total,
    };
    
    onBoundsClick?.(chartPoint, bounds);
  }, [interactive, hoveredPoint, uncertaintyData, onBoundsClick]);
  
  // Render grid
  const renderGrid = () => {
    const gridLines = [];
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = yScale(i / 10);
      gridLines.push(
        <line
          key={`h-${i}`}
          x1={padding.left}
          y1={y}
          x2={width - padding.right}
          y2={y}
          stroke="rgba(156, 163, 175, 0.2)"
          strokeWidth={i % 5 === 0 ? 1 : 0.5}
        />
      );
    }
    
    // Vertical grid lines
    const xStep = Math.max(1, Math.floor(uncertaintyData.length / 8));
    for (let i = 0; i < uncertaintyData.length; i += xStep) {
      const x = xScale(i);
      gridLines.push(
        <line
          key={`v-${i}`}
          x1={x}
          y1={padding.top}
          x2={x}
          y2={height - padding.bottom}
          stroke="rgba(156, 163, 175, 0.2)"
          strokeWidth={0.5}
        />
      );
    }
    
    return gridLines;
  };
  
  // Render uncertainty bounds
  const renderUncertaintyBounds = () => {
    return (
      <>
        {/* Bayesian confidence intervals (background) */}
        <path
          d={uncertaintyPaths.bayesian}
          fill={colors.bounds}
          stroke="none"
          opacity={0.3}
        />
        
        {/* Total uncertainty */}
        <path
          d={uncertaintyPaths.total}
          fill={colors.total}
          stroke="none"
          opacity={selectedUncertaintyType === 'total' ? 0.8 : 0.4}
        />
        
        {/* Epistemic uncertainty */}
        {showEpistemic && (
          <path
            d={uncertaintyPaths.epistemic}
            fill={colors.epistemic}
            stroke="none"
            opacity={selectedUncertaintyType === 'epistemic' ? 0.8 : 0.5}
          />
        )}
        
        {/* Aleatoric uncertainty */}
        {showAleatoric && (
          <path
            d={uncertaintyPaths.aleatoric}
            fill={colors.aleatoric}
            stroke="none"
            opacity={selectedUncertaintyType === 'aleatoric' ? 0.8 : 0.5}
          />
        )}
      </>
    );
  };
  
  // Render mean line
  const renderMeanLine = () => {
    return (
      <path
        d={meanLinePath}
        fill="none"
        stroke={colors.mean}
        strokeWidth={3}
        strokeLinecap="round"
      />
    );
  };
  
  // Render uncertainty indicators
  const renderUncertaintyIndicators = () => {
    return uncertaintyData.map((point, index) => {
      const x = xScale(index);
      const y = yScale(point.mean);
      const isHovered = hoveredPoint === index;
      
      return (
        <g key={index}>
          {/* Main point */}
          <circle
            cx={x}
            cy={y}
            r={isHovered ? 6 : 4}
            fill={colors.mean}
            stroke="white"
            strokeWidth={2}
          />
          
          {/* Uncertainty whiskers */}
          {isHovered && (
            <>
              {/* Total uncertainty whisker */}
              <line
                x1={x}
                y1={yScale(point.bounds.totalLower)}
                x2={x}
                y2={yScale(point.bounds.totalUpper)}
                stroke={colors.total.replace('0.3', '0.8')}
                strokeWidth={2}
              />
              
              {/* Uncertainty caps */}
              <line
                x1={x - 4}
                y1={yScale(point.bounds.totalLower)}
                x2={x + 4}
                y2={yScale(point.bounds.totalLower)}
                stroke={colors.total.replace('0.3', '0.8')}
                strokeWidth={2}
              />
              <line
                x1={x - 4}
                y1={yScale(point.bounds.totalUpper)}
                x2={x + 4}
                y2={yScale(point.bounds.totalUpper)}
                stroke={colors.total.replace('0.3', '0.8')}
                strokeWidth={2}
              />
            </>
          )}
        </g>
      );
    });
  };
  
  // Render axes and labels
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
        {[0, 0.25, 0.5, 0.75, 1.0].map(value => (
          <text
            key={value}
            x={padding.left - 15}
            y={yScale(value) + 4}
            textAnchor="end"
            className="text-xs fill-current text-muted-foreground"
          >
            {(value * 100).toFixed(0)}%
          </text>
        ))}
        
        {/* Axis titles */}
        <text
          x={width / 2}
          y={height - 15}
          textAnchor="middle"
          className="text-sm fill-current font-medium"
        >
          Time
        </text>
        
        <text
          x={20}
          y={height / 2}
          textAnchor="middle"
          transform={`rotate(-90 20 ${height / 2})`}
          className="text-sm fill-current font-medium"
        >
          Confidence
        </text>
      </>
    );
  };
  
  // Render tooltip
  const renderTooltip = () => {
    if (hoveredPoint === null) return null;
    
    const point = uncertaintyData[hoveredPoint];
    const x = xScale(hoveredPoint);
    const y = yScale(point.mean);
    
    return (
      <g>
        {/* Tooltip background */}
        <rect
          x={x + 15}
          y={y - 80}
          width={180}
          height={70}
          fill="rgba(0, 0, 0, 0.9)"
          rx={6}
          ry={6}
          stroke="rgba(255, 255, 255, 0.2)"
          strokeWidth={1}
        />
        
        {/* Tooltip content */}
        <text x={x + 25} y={y - 60} className="fill-white text-xs font-semibold">
          Confidence: {(point.mean * 100).toFixed(1)}%
        </text>
        <text x={x + 25} y={y - 45} className="fill-blue-300 text-xs">
          Epistemic: ±{(point.epistemic * 100).toFixed(1)}%
        </text>
        <text x={x + 25} y={y - 30} className="fill-red-300 text-xs">
          Aleatoric: ±{(point.aleatoric * 100).toFixed(1)}%
        </text>
        <text x={x + 25} y={y - 15} className="fill-gray-300 text-xs">
          Total: ±{(point.total * 100).toFixed(1)}%
        </text>
      </g>
    );
  };
  
  // Render legend
  const renderLegend = () => {
    const legendItems = [
      { label: 'Mean', color: colors.mean, type: 'line' },
      { label: `${(confidenceLevel * 100).toFixed(0)}% CI`, color: colors.bounds, type: 'area' },
      ...(showEpistemic ? [{ label: 'Epistemic', color: colors.epistemic, type: 'area' }] : []),
      ...(showAleatoric ? [{ label: 'Aleatoric', color: colors.aleatoric, type: 'area' }] : []),
      { label: 'Total', color: colors.total, type: 'area' },
    ];
    
    return (
      <g transform={`translate(${width - 150}, ${padding.top + 10})`}>
        {legendItems.map((item, index) => (
          <g key={item.label} transform={`translate(0, ${index * 20})`}>
            {item.type === 'line' ? (
              <line
                x1={0}
                y1={10}
                x2={15}
                y2={10}
                stroke={item.color}
                strokeWidth={3}
              />
            ) : (
              <rect
                x={0}
                y={5}
                width={15}
                height={10}
                fill={item.color}
              />
            )}
            <text
              x={20}
              y={14}
              className="text-xs fill-current"
            >
              {item.label}
            </text>
          </g>
        ))}
      </g>
    );
  };
  
  if (!uncertaintyData.length) {
    return (
      <div 
        className="flex items-center justify-center text-muted-foreground border-2 border-dashed border-muted rounded-md"
        style={{ width, height }}
      >
        No uncertainty data available
      </div>
    );
  }
  
  return (
    <div className="relative">
      {/* Uncertainty type selector */}
      <div className="mb-4 flex gap-2">
        {(['total', 'epistemic', 'aleatoric'] as const).map(type => (
          <button
            key={type}
            onClick={() => setSelectedUncertaintyType(type)}
            className={`px-3 py-1 text-xs rounded-md border transition-colors ${
              selectedUncertaintyType === type
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-background text-muted-foreground border-border hover:bg-muted'
            }`}
          >
            {type.charAt(0).toUpperCase() + type.slice(1)}
          </button>
        ))}
      </div>
      
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
        
        {/* Uncertainty bounds */}
        {renderUncertaintyBounds()}
        
        {/* Mean line */}
        {renderMeanLine()}
        
        {/* Uncertainty indicators */}
        {renderUncertaintyIndicators()}
        
        {/* Axes */}
        {renderAxes()}
        
        {/* Legend */}
        {renderLegend()}
        
        {/* Tooltip */}
        {renderTooltip()}
      </svg>
    </div>
  );
};

export default UncertaintyBounds;