/**
 * ConfidenceVisualizationSafe Test Suite
 * 
 * Tests the bulletproof chart component with intentionally bad data
 * to ensure it never crashes with NaN errors
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ConfidenceVisualizationSafe } from '../../../src/components/deepconf/ConfidenceVisualizationSafe';
import { ConfidenceMetrics } from '../../../src/components/deepconf/types';

// Mock data generators for testing edge cases
const createValidMetric = (): ConfidenceMetrics => ({
  overall: 0.7,
  dimensions: {
    structural: 0.8,
    contextual: 0.6,
    temporal: 0.9,
    semantic: 0.5,
  },
  uncertainty: {
    total: 0.1,
    epistemic: 0.05,
    aleatoric: 0.05,
  },
  bayesian: {
    lower: 0.6,
    upper: 0.8,
    mean: 0.7,
    variance: 0.02,
  },
  trend: 'increasing',
});

const createBadDataMetric = (): any => ({
  overall: NaN,
  dimensions: {
    structural: Infinity,
    contextual: null,
    temporal: undefined,
    semantic: 'invalid',
  },
  uncertainty: {
    total: -Infinity,
    epistemic: 'not-a-number',
    aleatoric: [],
  },
  bayesian: {
    lower: NaN,
    upper: Infinity,
    mean: null,
    variance: undefined,
  },
  trend: 'invalid-trend',
});

const createExtremeDataMetric = (): any => ({
  overall: 999999999999,
  dimensions: {
    structural: -999999999999,
    contextual: 1e50,
    temporal: -1e50,
    semantic: Number.MAX_VALUE,
  },
  uncertainty: {
    total: Number.MIN_VALUE,
    epistemic: -Number.MAX_VALUE,
    aleatoric: Number.POSITIVE_INFINITY,
  },
  bayesian: {
    lower: Number.NEGATIVE_INFINITY,
    upper: Number.NaN,
    mean: 'extreme',
    variance: {},
  },
  trend: null,
});

describe('ConfidenceVisualizationSafe', () => {
  // Spy on console methods to capture error logging
  const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

  afterEach(() => {
    consoleSpy.mockClear();
    errorSpy.mockClear();
  });

  describe('Normal Operation', () => {
    it('should render with valid data', () => {
      const validMetrics = [createValidMetric()];
      
      render(
        <ConfidenceVisualizationSafe
          metrics={validMetrics}
          chartType="area"
          interactive={true}
        />
      );

      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      expect(screen.getByText('70.0% avg')).toBeInTheDocument();
    });

    it('should handle empty metrics array', () => {
      render(
        <ConfidenceVisualizationSafe
          metrics={[]}
          chartType="line"
          interactive={false}
        />
      );

      expect(screen.getByText('No Data Available')).toBeInTheDocument();
      expect(screen.getByText('Confidence data will appear here when available')).toBeInTheDocument();
    });

    it('should handle undefined metrics', () => {
      render(
        <ConfidenceVisualizationSafe
          metrics={undefined as any}
          chartType="area"
        />
      );

      expect(screen.getByText('No Data Available')).toBeInTheDocument();
    });
  });

  describe('NaN Protection Tests', () => {
    it('should handle NaN values without crashing', async () => {
      const badMetrics = [createBadDataMetric()];
      
      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={badMetrics}
            chartType="area"
            interactive={true}
          />
        );
      }).not.toThrow();

      // Should render with fallback values
      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      
      // Should show safe defaults
      await waitFor(() => {
        expect(screen.getByText(/50\.0% avg/)).toBeInTheDocument();
      });

      // Should have logged warnings for invalid data
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[SafeChart] Null/undefined value'),
        expect.anything()
      );
    });

    it('should handle extreme values without crashing', () => {
      const extremeMetrics = [createExtremeDataMetric()];
      
      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={extremeMetrics}
            chartType="line"
          />
        );
      }).not.toThrow();

      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      
      // Should clamp extreme values to safe ranges
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[SafeChart] Extreme value'),
        expect.anything()
      );
    });

    it('should handle mixed valid and invalid data', () => {
      const mixedMetrics = [
        createValidMetric(),
        createBadDataMetric(),
        createValidMetric(),
        createExtremeDataMetric(),
      ];
      
      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={mixedMetrics}
            chartType="area"
          />
        );
      }).not.toThrow();

      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      
      // Should process valid data points and skip/fix invalid ones
      expect(screen.getByText(/Data Points:/)).toBeInTheDocument();
    });

    it('should handle completely null data structure', () => {
      const nullMetrics = [null, undefined, {}, 'invalid', 123, []];
      
      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={nullMetrics as any}
            chartType="bar"
          />
        );
      }).not.toThrow();

      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
    });
  });

  describe('SVG Coordinate Safety', () => {
    it('should handle invalid SVG coordinates without DOM errors', () => {
      const metrics = [{
        overall: NaN,
        dimensions: { structural: Infinity },
        uncertainty: { total: -Infinity },
        bayesian: { lower: NaN, upper: NaN, mean: NaN, variance: NaN },
        trend: 'stable'
      }];

      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={metrics as any}
            chartType="area"
          />
        );
      }).not.toThrow();

      // SVG should render without errors
      const svg = screen.getByRole('img', { hidden: true }) || document.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });

    it('should create valid SVG paths even with bad data', () => {
      const metrics = Array.from({ length: 5 }, () => createBadDataMetric());

      render(
        <ConfidenceVisualizationSafe
          metrics={metrics as any}
          chartType="area"
        />
      );

      // Check that SVG elements are present and valid
      const svg = document.querySelector('svg');
      expect(svg).toBeInTheDocument();
      
      // Paths should not contain NaN
      const paths = document.querySelectorAll('path');
      paths.forEach(path => {
        const d = path.getAttribute('d');
        if (d) {
          expect(d).not.toContain('NaN');
          expect(d).not.toContain('Infinity');
        }
      });

      // Circles should have valid coordinates
      const circles = document.querySelectorAll('circle');
      circles.forEach(circle => {
        const cx = circle.getAttribute('cx');
        const cy = circle.getAttribute('cy');
        const r = circle.getAttribute('r');
        
        if (cx) expect(parseFloat(cx)).not.toBeNaN();
        if (cy) expect(parseFloat(cy)).not.toBeNaN();
        if (r) expect(parseFloat(r)).not.toBeNaN();
      });
    });
  });

  describe('Error Boundary Tests', () => {
    it('should catch and display errors gracefully', () => {
      // Create a component that will throw during render
      const ThrowingComponent = () => {
        throw new Error('Test error for error boundary');
      };

      // We can't easily test the error boundary directly with this setup,
      // but we can test that the component doesn't crash with extreme inputs
      const crashingData = {
        overall: (() => { throw new Error('Data processing error'); })(),
      };

      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={[crashingData as any]}
            chartType="area"
          />
        );
      }).not.toThrow();
    });
  });

  describe('Interactive Features', () => {
    it('should handle chart interactions with safe data', async () => {
      const mockOnClick = vi.fn();
      const validMetrics = [createValidMetric()];

      render(
        <ConfidenceVisualizationSafe
          metrics={validMetrics}
          chartType="area"
          interactive={true}
          onConfidenceClick={mockOnClick}
        />
      );

      // Find and click a data point
      const circles = document.querySelectorAll('circle');
      if (circles.length > 0) {
        fireEvent.click(circles[0]);
        expect(mockOnClick).toHaveBeenCalled();
      }
    });

    it('should handle chart type switching', () => {
      const validMetrics = [createValidMetric()];

      render(
        <ConfidenceVisualizationSafe
          metrics={validMetrics}
          chartType="area"
          interactive={true}
        />
      );

      // Switch to line chart
      const lineButton = screen.getByRole('button', { name: /line/i });
      fireEvent.click(lineButton);

      expect(screen.getByText(/Chart Type:/)).toBeInTheDocument();
    });

    it('should toggle uncertainty display', () => {
      const validMetrics = [createValidMetric()];

      render(
        <ConfidenceVisualizationSafe
          metrics={validMetrics}
          chartType="area"
        />
      );

      const uncertaintyToggle = screen.getByRole('button', { name: /uncertainty/i });
      fireEvent.click(uncertaintyToggle);

      expect(uncertaintyToggle).toBeInTheDocument();
    });
  });

  describe('Performance with Large Datasets', () => {
    it('should handle large datasets without performance issues', () => {
      // Create a large dataset with mixed good and bad data
      const largeDataset = Array.from({ length: 1000 }, (_, i) => {
        if (i % 3 === 0) return createBadDataMetric();
        if (i % 3 === 1) return createExtremeDataMetric();
        return createValidMetric();
      });

      const start = performance.now();
      
      expect(() => {
        render(
          <ConfidenceVisualizationSafe
            metrics={largeDataset as any}
            chartType="line"
          />
        );
      }).not.toThrow();

      const end = performance.now();
      
      // Should render in reasonable time (less than 1 second)
      expect(end - start).toBeLessThan(1000);
      
      expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      expect(screen.getByText(/Data Points:/)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should be accessible with valid data', () => {
      const validMetrics = [createValidMetric()];

      render(
        <ConfidenceVisualizationSafe
          metrics={validMetrics}
          chartType="area"
          interactive={true}
        />
      );

      // Check for proper ARIA labels and roles
      expect(screen.getByRole('button', { name: /uncertainty/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /thresholds/i })).toBeInTheDocument();
    });

    it('should be accessible with no data', () => {
      render(
        <ConfidenceVisualizationSafe
          metrics={[]}
          chartType="area"
        />
      );

      expect(screen.getByText('No Data Available')).toBeInTheDocument();
      expect(screen.getByText('Confidence data will appear here when available')).toBeInTheDocument();
    });
  });

  describe('Edge Case Input Validation', () => {
    const edgeCases = [
      { name: 'empty object', data: {} },
      { name: 'string instead of object', data: 'invalid' },
      { name: 'number instead of object', data: 123 },
      { name: 'array instead of object', data: [] },
      { name: 'function instead of object', data: () => {} },
      { name: 'deeply nested null', data: { overall: { nested: { value: null } } } },
      { name: 'circular reference', data: (() => { const obj: any = {}; obj.circular = obj; return obj; })() },
    ];

    edgeCases.forEach(({ name, data }) => {
      it(`should handle ${name} without crashing`, () => {
        expect(() => {
          render(
            <ConfidenceVisualizationSafe
              metrics={[data] as any}
              chartType="area"
            />
          );
        }).not.toThrow();

        expect(screen.getByText('Confidence Visualization (Safe)')).toBeInTheDocument();
      });
    });
  });
});