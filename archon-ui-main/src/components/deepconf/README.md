# DeepConf Components - Phase 7 Integration

## SCWT Metrics Dashboard and Confidence Visualization UI Components

This package provides comprehensive UI components for visualizing SCWT (Structural, Contextual, Temporal Weight) metrics and confidence analysis in the Phase 7 DeepConf integration.

## Features

- **Real-time SCWT metrics visualization** with live WebSocket updates
- **Multi-dimensional confidence analysis** with Bayesian uncertainty bounds
- **Performance tracking** for token efficiency and cost monitoring
- **Interactive debugging tools** with confidence analysis capabilities
- **Accessibility compliant** (WCAG 2.1 AA) with keyboard navigation
- **Responsive design** supporting mobile, tablet, and desktop viewports
- **Dark/light theme support** with automatic detection
- **Comprehensive TypeScript types** for type safety
- **Production-ready components** with error handling and loading states

## Component Overview

### Core Components

1. **SCWTDashboard** - Main dashboard with real-time metrics overview
2. **ConfidenceVisualization** - Multi-dimensional confidence charts
3. **PerformanceMetrics** - Token efficiency and cost tracking
4. **DebugTools** - Interactive confidence analysis and debugging
5. **RealTimeMonitoring** - Live metrics with WebSocket connectivity

### Specialized Components

6. **ConfidenceChart** - Advanced confidence visualization with interaction
7. **UncertaintyBounds** - Bayesian uncertainty bounds with decomposition

## Quick Start

### Basic Usage

```tsx
import React from 'react';
import {
  SCWTDashboard,
  generateMockSCWTMetrics,
  generateMockConfidenceMetrics,
  generateMockPerformanceMetrics,
  DEFAULT_DASHBOARD_CONFIG,
  RealTimeData,
} from './components/deepconf';

const MyDashboard: React.FC = () => {
  // Mock data for development
  const mockMetrics = generateMockSCWTMetrics(100);
  const mockConfidence = generateMockConfidenceMetrics(100);
  const mockPerformance = generateMockPerformanceMetrics(100);
  
  const mockData: RealTimeData = {
    current: mockMetrics[mockMetrics.length - 1],
    history: mockMetrics,
    performance: mockPerformance[mockPerformance.length - 1],
    confidence: mockConfidence[mockConfidence.length - 1],
    status: 'active',
    lastUpdate: new Date(),
  };

  return (
    <SCWTDashboard
      data={mockData}
      config={DEFAULT_DASHBOARD_CONFIG}
      onMetricSelect={(metric) => console.log('Selected metric:', metric)}
      onExportData={() => console.log('Exporting data...')}
      onRefresh={() => console.log('Refreshing...')}
    />
  );
};
```

### Individual Components

```tsx
import React from 'react';
import {
  ConfidenceVisualization,
  PerformanceMetrics,
  DebugTools,
} from './components/deepconf';

const IndividualComponents: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Confidence Visualization */}
      <ConfidenceVisualization
        metrics={mockConfidenceMetrics}
        chartType="area"
        interactive={true}
        showThresholds={true}
      />
      
      {/* Performance Metrics */}
      <PerformanceMetrics
        metrics={mockPerformanceMetrics}
        displayMetrics={['tokenEfficiency', 'cost', 'timing']}
        showTrends={true}
      />
      
      {/* Debug Tools */}
      <DebugTools
        metrics={currentMetrics}
        confidence={currentConfidence}
        debugMode={true}
        enableAnalysis={true}
        exportFormats={['json', 'csv']}
      />
    </div>
  );
};
```

## Component API Reference

### SCWTDashboard

The main dashboard component that orchestrates all other components.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `RealTimeData` | - | Real-time data source |
| `config` | `DashboardConfig` | `DEFAULT_DASHBOARD_CONFIG` | Dashboard configuration |
| `onMetricSelect` | `(metric: MetricType) => void` | - | Metric selection handler |
| `onExportData` | `() => void` | - | Data export handler |
| `onRefresh` | `() => void` | - | Refresh handler |
| `isLoading` | `boolean` | `false` | Loading state |
| `error` | `Error \| null` | `null` | Error state |

#### Features

- **Auto-refresh** with configurable intervals
- **Time range selection** (15min, 1h, 24h, 1week, 1month)
- **Metric filtering** to show/hide specific SCWT components
- **Settings panel** with accessibility options
- **Responsive grid layout** that adapts to screen size
- **Real-time status indicators** and connection monitoring

### ConfidenceVisualization

Multi-dimensional confidence visualization with interactive charts.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `metrics` | `ConfidenceMetrics[]` | - | Confidence metrics data |
| `chartType` | `'line' \| 'area' \| 'bar' \| 'heatmap'` | `'area'` | Chart visualization type |
| `timeRange` | `TimeRange` | - | Time range filter |
| `interactive` | `boolean` | `true` | Enable interactive features |
| `thresholds` | `ConfidenceThresholds` | `DEFAULT_CONFIDENCE_THRESHOLDS` | Confidence level thresholds |

#### Features

- **Multi-dimensional views** (overall, structural, contextual, temporal, semantic)
- **Interactive threshold lines** with customizable confidence levels
- **Uncertainty band visualization** showing confidence intervals
- **Trend analysis** with directional indicators
- **Real-time confidence breakdown** with progress bars

### PerformanceMetrics

Token efficiency tracking and performance monitoring.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `metrics` | `PerformanceMetrics[]` | - | Performance data array |
| `displayMetrics` | `(keyof PerformanceMetrics)[]` | `['tokenEfficiency', 'cost', 'timing', 'quality']` | Metrics to display |
| `baseline` | `PerformanceMetrics` | - | Comparison baseline |
| `thresholds` | `PerformanceThresholds` | `DEFAULT_PERFORMANCE_THRESHOLDS` | Performance thresholds |
| `updateFrequency` | `number` | `5000` | Update frequency in ms |

#### Features

- **Token efficiency tracking** with compression ratios
- **Cost monitoring** with budget alerts and savings tracking
- **Response time analysis** with throughput metrics
- **Quality scores** (accuracy, precision, recall, F1)
- **Historical trends** with visual indicators
- **Performance level indicators** (excellent, good, acceptable, poor)

### DebugTools

Interactive debugging and analysis tools.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `metrics` | `SCWTMetrics` | - | Current SCWT metrics |
| `confidence` | `ConfidenceMetrics` | - | Confidence data |
| `debugMode` | `boolean` | `false` | Enable debug mode |
| `enableAnalysis` | `boolean` | `true` | Enable analysis features |
| `exportFormats` | `('json' \| 'csv' \| 'excel')[]` | `['json', 'csv']` | Available export formats |

#### Features

- **Interactive confidence analysis** with AI-powered insights
- **Anomaly detection** with severity levels and recommendations
- **Data export** in multiple formats (JSON, CSV, Excel)
- **Simulation tools** for stress testing and edge case analysis
- **Debug logs** with real-time monitoring
- **Performance profiling** with detailed metrics breakdown

### RealTimeMonitoring

Live metrics monitoring with WebSocket connectivity.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `data` | `RealTimeData` | - | Current data state |
| `timeRange` | `TimeRange` | - | Time range for display |
| `selectedMetrics` | `MetricType[]` | - | Metrics to monitor |
| `config` | `DashboardConfig` | - | Dashboard configuration |
| `webSocketUrl` | `string` | - | WebSocket endpoint URL |

#### Features

- **WebSocket connectivity** with automatic reconnection
- **Live streaming charts** with real-time updates
- **Connection status monitoring** with retry logic
- **Pause/resume functionality** for data collection
- **Streaming statistics** (update rate, data points, etc.)
- **Message history** with JSON inspection

## Advanced Usage

### Custom Themes and Styling

```tsx
import { SCWTDashboard, DashboardConfig } from './components/deepconf';

const customConfig: DashboardConfig = {
  theme: 'dark',
  layout: {
    columns: 3,
    cardSpacing: 20,
    compactMode: true,
  },
  accessibility: {
    highContrast: true,
    reducedMotion: false,
    screenReaderOptimized: true,
  },
  // ... other config options
};

<SCWTDashboard config={customConfig} {...otherProps} />
```

### WebSocket Integration

```tsx
import { RealTimeMonitoring, WebSocketConfig } from './components/deepconf';

const wsConfig: WebSocketConfig = {
  url: 'wss://your-websocket-endpoint.com/scwt-metrics',
  reconnectAttempts: 10,
  reconnectDelay: 5000,
  heartbeatInterval: 30000,
  timeout: 15000,
};

// The WebSocket will automatically:
// - Connect on component mount
// - Send subscription messages for selected metrics
// - Handle reconnection on connection loss
// - Parse incoming metric updates
// - Update charts in real-time
```

### Custom Analysis and Debugging

```tsx
import { DebugTools, DebugAction, AnalysisResult } from './components/deepconf';

const handleDebugAction = (action: DebugAction) => {
  switch (action.type) {
    case 'analyze':
      // Perform custom analysis
      performCustomAnalysis(action.parameters);
      break;
    case 'export':
      // Handle custom export logic
      exportToCustomFormat(action.parameters);
      break;
    case 'simulate':
      // Run custom simulations
      runCustomSimulation(action.parameters);
      break;
  }
};

<DebugTools
  onDebugAction={handleDebugAction}
  exportFormats={['json', 'csv', 'excel', 'pdf']}
  {...otherProps}
/>
```

## Accessibility Features

All components are designed with accessibility in mind:

### WCAG 2.1 AA Compliance

- **Color contrast ratios** meet minimum 4.5:1 for normal text, 3:1 for large text
- **Information is not conveyed by color alone** - uses icons, patterns, and text labels
- **Keyboard navigation** - all interactive elements accessible via keyboard
- **Screen reader support** - proper ARIA labels and semantic HTML
- **Focus management** - visible focus indicators and logical tab order

### Keyboard Navigation

- **Tab** - Navigate between interactive elements
- **Space/Enter** - Activate buttons and toggles
- **Arrow keys** - Navigate within charts and data points
- **Escape** - Close modals and tooltips

### Reduced Motion Support

```tsx
const config: DashboardConfig = {
  accessibility: {
    reducedMotion: true, // Disables animations for users who prefer reduced motion
  },
};
```

### High Contrast Mode

```tsx
const config: DashboardConfig = {
  accessibility: {
    highContrast: true, // Uses high contrast color scheme
  },
};
```

## Performance Considerations

### Data Optimization

- **Virtualized rendering** for large datasets (>1000 points)
- **Efficient re-rendering** using React.memo and useMemo
- **Data pagination** for historical metrics
- **Lazy loading** of non-critical components

### WebSocket Optimization

- **Connection pooling** for multiple metric subscriptions
- **Message batching** to reduce update frequency
- **Automatic throttling** during high-frequency updates
- **Memory management** with configurable data retention limits

### Bundle Size

The entire DeepConf component library is designed to be tree-shakeable:

```tsx
// Import only what you need
import { SCWTDashboard } from './components/deepconf/SCWTDashboard';
import { ConfidenceChart } from './components/deepconf/ConfidenceChart';

// Or import from the index for convenience
import { SCWTDashboard, ConfidenceChart } from './components/deepconf';
```

## Testing

### Component Testing

```tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SCWTDashboard, generateMockSCWTMetrics } from './components/deepconf';

describe('SCWTDashboard', () => {
  it('renders metrics correctly', async () => {
    const mockData = {
      current: generateMockSCWTMetrics(1)[0],
      // ... other mock data
    };
    
    render(<SCWTDashboard data={mockData} />);
    
    expect(screen.getByText('SCWT Metrics Dashboard')).toBeInTheDocument();
    expect(screen.getByText(/Combined Score/)).toBeInTheDocument();
  });
  
  it('handles metric selection', async () => {
    const onMetricSelect = jest.fn();
    
    render(
      <SCWTDashboard 
        data={mockData} 
        onMetricSelect={onMetricSelect}
      />
    );
    
    fireEvent.click(screen.getByText('Structural Weight'));
    expect(onMetricSelect).toHaveBeenCalledWith('structuralWeight');
  });
});
```

### Accessibility Testing

```tsx
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

describe('Accessibility', () => {
  it('should not have accessibility violations', async () => {
    const { container } = render(<SCWTDashboard {...props} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
```

## Troubleshooting

### Common Issues

1. **WebSocket connection fails**
   - Check WebSocket endpoint URL
   - Verify CORS settings on server
   - Check network connectivity

2. **Charts not rendering**
   - Ensure data arrays are not empty
   - Check for proper timestamp formats
   - Verify numeric data types

3. **Performance issues with large datasets**
   - Enable data pagination
   - Reduce update frequency
   - Use data filtering

4. **TypeScript errors**
   - Ensure all required props are provided
   - Check type compatibility for data structures
   - Import types from the main index file

### Debug Mode

Enable debug mode for additional logging:

```tsx
<DebugTools debugMode={true} />
```

This will:
- Show additional debug information
- Log all data transformations
- Display performance metrics
- Enable advanced debugging tools

## Contributing

When contributing to the DeepConf components:

1. **Follow accessibility guidelines** - ensure WCAG compliance
2. **Add comprehensive tests** - unit, integration, and accessibility
3. **Update TypeScript types** - maintain type safety
4. **Document new features** - update README and code comments
5. **Follow design system** - use existing UI component patterns

## License

This component library is part of the Archon UI system and follows the same licensing terms.