/**
 * Comprehensive tests for Advanced DeepConf Debug Tools Component - Phase 7 PRD Implementation
 * 
 * Tests all enhanced debugging functionality:
 * - Low confidence analysis with actionable insights
 * - Debug session management with state persistence
 * - Confidence factor tracing with detailed breakdowns
 * - Optimization suggestions with implementation guidance
 * - Multi-format data export for external analysis
 * - API integration and error handling
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DebugTools } from '../../../src/components/deepconf/DebugTools';

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock props
const mockProps = {
  metrics: {
    structuralWeight: 0.65,
    contextWeight: 0.70,
    temporalWeight: 0.60,
    combinedScore: 0.65
  },
  confidence: {
    overall: 0.25, // Low confidence for testing
    uncertainty: {
      epistemic: 0.3,
      aleatoric: 0.2,
      total: 0.5
    },
    trend: 'stable'
  },
  debugMode: true,
  enableAnalysis: true,
  exportFormats: ['json', 'csv'],
  onDebugAction: vi.fn()
};

describe('Advanced DeepConf Debug Tools', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('Component Rendering', () => {
    it('renders all debug tabs correctly', () => {
      render(<DebugTools {...mockProps} />);
      
      // Check all tabs are present
      expect(screen.getByText('analysis')).toBeInTheDocument();
      expect(screen.getByText('sessions')).toBeInTheDocument();
      expect(screen.getByText('trace')).toBeInTheDocument();
      expect(screen.getByText('optimize')).toBeInTheDocument();
      expect(screen.getByText('export')).toBeInTheDocument();
      expect(screen.getByText('logs')).toBeInTheDocument();
    });

    it('shows debug mode indicator when enabled', () => {
      render(<DebugTools {...mockProps} debugMode={true} />);
      
      expect(screen.getByText('Debug Mode Active')).toBeInTheDocument();
    });

    it('displays current metrics status correctly', () => {
      render(<DebugTools {...mockProps} />);
      
      // Check metrics display
      expect(screen.getByText(/structural weight/i)).toBeInTheDocument();
      expect(screen.getByText(/context weight/i)).toBeInTheDocument();
      expect(screen.getByText(/temporal weight/i)).toBeInTheDocument();
      expect(screen.getByText(/combined score/i)).toBeInTheDocument();
      expect(screen.getByText(/confidence/i)).toBeInTheDocument();
      
      // Check low confidence indication (25% should show as error)
      const confidenceElements = screen.getAllByText(/25\.0%/);
      expect(confidenceElements.length).toBeGreaterThan(0);
    });
  });

  describe('Low Confidence Analysis Tab', () => {
    it('renders analysis controls and inputs', () => {
      render(<DebugTools {...mockProps} />);
      
      // Should be on analysis tab by default
      expect(screen.getByText('Low Confidence Analysis')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter task ID')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter task description')).toBeInTheDocument();
      expect(screen.getByText('Analyze Low Confidence')).toBeInTheDocument();
    });

    it('performs low confidence analysis when button clicked', async () => {
      // Mock successful API response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          report: {
            report_id: 'test_report_001',
            issues: [
              {
                id: 'issue_001',
                severity: 'critical',
                category: 'confidence_factor',
                title: 'Critical Technical Complexity',
                description: 'Technical complexity scored very low',
                root_cause: 'Task complexity exceeds model capabilities',
                recommendations: ['Break task into smaller components', 'Add more context'],
                confidence_impact: -0.15
              }
            ],
            recommendations: ['Improve data quality', 'Add domain expertise'],
            confidence_projection: {
              current_confidence: 0.25,
              projected_confidence: 0.65,
              potential_improvement: 0.4
            }
          }
        })
      });

      render(<DebugTools {...mockProps} />);
      
      // Fill in task details
      const taskIdInput = screen.getByPlaceholderText('Enter task ID');
      const taskContentInput = screen.getByPlaceholderText('Enter task description');
      
      fireEvent.change(taskIdInput, { target: { value: 'test_task_001' } });
      fireEvent.change(taskContentInput, { target: { value: 'Test analysis task' } });
      
      // Click analyze button
      const analyzeButton = screen.getByText('Analyze Low Confidence');
      fireEvent.click(analyzeButton);
      
      // Should show loading state
      await waitFor(() => {
        expect(screen.getByText('Performing deep confidence analysis...')).toBeInTheDocument();
      });
      
      // Wait for analysis to complete
      await waitFor(() => {
        expect(screen.getByText('Confidence Projection')).toBeInTheDocument();
      });
      
      // Check that results are displayed
      expect(screen.getByText('25.0%')).toBeInTheDocument(); // Current confidence
      expect(screen.getByText('65.0%')).toBeInTheDocument(); // Projected confidence
      expect(screen.getByText('+40.0%')).toBeInTheDocument(); // Improvement
      
      // Check that issues are displayed
      expect(screen.getByText('Critical Technical Complexity')).toBeInTheDocument();
      expect(screen.getByText('CRITICAL')).toBeInTheDocument();
      
      // Check recommendations
      expect(screen.getByText(/Improve data quality/)).toBeInTheDocument();
      expect(screen.getByText(/Add domain expertise/)).toBeInTheDocument();
    });

    it('handles analysis API errors gracefully', async () => {
      // Mock API error
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      render(<DebugTools {...mockProps} />);
      
      const analyzeButton = screen.getByText('Analyze Low Confidence');
      fireEvent.click(analyzeButton);
      
      // Wait for error to be logged
      await waitFor(() => {
        // Check that error was logged (should appear in logs)
        const logsTab = screen.getByText('logs');
        fireEvent.click(logsTab);
        expect(screen.getByText(/ERROR.*analysis failed/i)).toBeInTheDocument();
      });
    });
  });

  describe('Debug Sessions Tab', () => {
    beforeEach(() => {
      // Mock health check API
      mockFetch.mockImplementation((url) => {
        if (url.includes('/health')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              status: 'healthy',
              debugger_initialized: true,
              engine_attached: true,
              active_sessions: 2,
              max_sessions: 10
            })
          });
        }
        if (url.includes('/sessions')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              success: true,
              sessions: [
                {
                  session_id: 'session_001',
                  task_id: 'task_001',
                  start_time: '2024-01-01T00:00:00',
                  is_active: true,
                  reports_count: 2,
                  duration_minutes: 15.5
                }
              ],
              metadata: {
                total_active: 1,
                returned_count: 1
              }
            })
          });
        }
        return Promise.reject(new Error('Unknown endpoint'));
      });
    });

    it('displays sessions tab correctly', async () => {
      render(<DebugTools {...mockProps} />);
      
      // Click sessions tab
      const sessionsTab = screen.getByText('sessions');
      fireEvent.click(sessionsTab);
      
      await waitFor(() => {
        expect(screen.getByText('Debug Sessions')).toBeInTheDocument();
        expect(screen.getByText('New Session')).toBeInTheDocument();
        expect(screen.getByText('Refresh')).toBeInTheDocument();
      });
      
      // Should show system health
      await waitFor(() => {
        expect(screen.getByText('System Health')).toBeInTheDocument();
        expect(screen.getByText('healthy')).toBeInTheDocument();
      });
    });

    it('creates new debug session when button clicked', async () => {
      // Mock session creation
      mockFetch.mockImplementation((url, options) => {
        if (url.includes('/sessions') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              success: true,
              session: {
                session_id: 'new_session_001',
                task_id: 'demo_task_001',
                start_time: '2024-01-01T01:00:00',
                is_active: true
              }
            })
          });
        }
        return mockFetch.mockImplementation(() => Promise.resolve({ ok: true, json: () => ({}) }));
      });

      render(<DebugTools {...mockProps} />);
      
      const sessionsTab = screen.getByText('sessions');
      fireEvent.click(sessionsTab);
      
      const newSessionButton = await screen.findByText('New Session');
      fireEvent.click(newSessionButton);
      
      // Should show active session
      await waitFor(() => {
        expect(screen.getByText('Active Session')).toBeInTheDocument();
        expect(screen.getByText('new_session_001')).toBeInTheDocument();
      });
    });
  });

  describe('Confidence Tracing Tab', () => {
    it('renders confidence tracing interface', () => {
      render(<DebugTools {...mockProps} />);
      
      const traceTab = screen.getByText('trace');
      fireEvent.click(traceTab);
      
      expect(screen.getByText('Confidence Factor Tracing')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter confidence ID or factor name to trace')).toBeInTheDocument();
      expect(screen.getByText('Trace Factors')).toBeInTheDocument();
    });

    it('performs confidence tracing when button clicked', async () => {
      // Mock tracing API response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          trace: {
            factor_name: 'technical_complexity',
            raw_score: 0.65,
            weighted_score: 0.1625,
            weight: 0.25,
            calculation_steps: [
              {
                step: 1,
                operation: 'complexity_assessment',
                calculation: 'base_score + domain_adjustment',
                result: 0.65,
                computation_time: 0.001
              }
            ],
            dependencies: ['domain_expertise', 'historical_performance'],
            confidence_contribution: 0.23
          }
        })
      });

      render(<DebugTools {...mockProps} />);
      
      const traceTab = screen.getByText('trace');
      fireEvent.click(traceTab);
      
      // Enter trace ID
      const traceInput = screen.getByPlaceholderText('Enter confidence ID or factor name to trace');
      fireEvent.change(traceInput, { target: { value: 'technical_complexity' } });
      
      // Click trace button
      const traceButton = screen.getByText('Trace Factors');
      fireEvent.click(traceButton);
      
      // Should show loading
      await waitFor(() => {
        expect(screen.getByText('Tracing confidence factor calculations...')).toBeInTheDocument();
      });
      
      // Wait for results
      await waitFor(() => {
        expect(screen.getByText('technical_complexity Analysis')).toBeInTheDocument();
      });
      
      // Check trace results
      expect(screen.getByText('0.650')).toBeInTheDocument(); // Raw score
      expect(screen.getByText('0.250')).toBeInTheDocument(); // Weight
      expect(screen.getByText('23.0%')).toBeInTheDocument(); // Contribution
      expect(screen.getByText('Calculation Steps')).toBeInTheDocument();
      expect(screen.getByText('complexity_assessment')).toBeInTheDocument();
    });
  });

  describe('Optimization Tab', () => {
    it('renders optimization interface', () => {
      render(<DebugTools {...mockProps} />);
      
      const optimizeTab = screen.getByText('optimize');
      fireEvent.click(optimizeTab);
      
      expect(screen.getByText('Optimization Strategies')).toBeInTheDocument();
      expect(screen.getByText('Generate Strategies')).toBeInTheDocument();
    });

    it('generates optimization strategies when button clicked', async () => {
      // Mock optimization API response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          optimization_suggestions: {
            strategies: [
              {
                strategy_id: 'cache_optimization_001',
                title: 'Improve Confidence Score Caching',
                description: 'Implement advanced caching strategies to reduce redundant calculations',
                implementation_complexity: 'medium',
                expected_improvement: {
                  latency_reduction: 0.4,
                  throughput_increase: 0.6
                },
                implementation_steps: [
                  'Implement LRU cache with intelligent invalidation',
                  'Add cache warming for frequently used patterns'
                ],
                risks: ['Memory usage increase', 'Cache invalidation complexity'],
                confidence: 0.85
              }
            ]
          }
        })
      });

      render(<DebugTools {...mockProps} />);
      
      const optimizeTab = screen.getByText('optimize');
      fireEvent.click(optimizeTab);
      
      const generateButton = screen.getByText('Generate Strategies');
      fireEvent.click(generateButton);
      
      // Wait for strategies to load
      await waitFor(() => {
        expect(screen.getByText('Improve Confidence Score Caching')).toBeInTheDocument();
      });
      
      // Check strategy details
      expect(screen.getByText('MEDIUM')).toBeInTheDocument(); // Complexity
      expect(screen.getByText('85% confidence')).toBeInTheDocument();
      expect(screen.getByText('Expected Improvements:')).toBeInTheDocument();
      expect(screen.getByText('+40%')).toBeInTheDocument(); // Latency reduction
      expect(screen.getByText('Implementation Steps:')).toBeInTheDocument();
      expect(screen.getByText(/Implement LRU cache/)).toBeInTheDocument();
    });

    it('shows empty state when no strategies generated', () => {
      render(<DebugTools {...mockProps} />);
      
      const optimizeTab = screen.getByText('optimize');
      fireEvent.click(optimizeTab);
      
      expect(screen.getByText('No optimization strategies generated yet.')).toBeInTheDocument();
      expect(screen.getByText(/Click "Generate Strategies"/)).toBeInTheDocument();
    });
  });

  describe('Export Tab', () => {
    it('renders export interface', () => {
      render(<DebugTools {...mockProps} />);
      
      const exportTab = screen.getByText('export');
      fireEvent.click(exportTab);
      
      expect(screen.getByText('Export Debug Data')).toBeInTheDocument();
      expect(screen.getByText('Format:')).toBeInTheDocument();
      expect(screen.getByText('Export Data')).toBeInTheDocument();
    });

    it('allows format selection', () => {
      render(<DebugTools {...mockProps} />);
      
      const exportTab = screen.getByText('export');
      fireEvent.click(exportTab);
      
      // Should have JSON selected by default
      const formatSelect = screen.getByRole('combobox');
      expect(formatSelect).toBeInTheDocument();
    });

    it('exports data when button clicked', async () => {
      // Mock URL.createObjectURL and related functions
      const mockCreateObjectURL = vi.fn(() => 'mock-url');
      const mockRevokeObjectURL = vi.fn();
      global.URL.createObjectURL = mockCreateObjectURL;
      global.URL.revokeObjectURL = mockRevokeObjectURL;
      
      // Mock DOM methods
      const mockClick = vi.fn();
      const mockAppendChild = vi.fn();
      const mockRemoveChild = vi.fn();
      const mockCreateElement = vi.fn(() => ({
        click: mockClick,
        href: '',
        download: ''
      }));
      
      Object.defineProperty(document, 'createElement', { value: mockCreateElement });
      Object.defineProperty(document.body, 'appendChild', { value: mockAppendChild });
      Object.defineProperty(document.body, 'removeChild', { value: mockRemoveChild });

      render(<DebugTools {...mockProps} />);
      
      const exportTab = screen.getByText('export');
      fireEvent.click(exportTab);
      
      const exportButton = screen.getByText('Export Data');
      fireEvent.click(exportButton);
      
      // Should create and trigger download
      await waitFor(() => {
        expect(mockCreateElement).toHaveBeenCalledWith('a');
        expect(mockClick).toHaveBeenCalled();
      });
    });
  });

  describe('Logs Tab', () => {
    it('renders logs interface', () => {
      render(<DebugTools {...mockProps} />);
      
      const logsTab = screen.getByText('logs');
      fireEvent.click(logsTab);
      
      expect(screen.getByText('Debug Logs')).toBeInTheDocument();
      expect(screen.getByText('Clear Logs')).toBeInTheDocument();
    });

    it('shows no logs message when empty', () => {
      render(<DebugTools {...mockProps} />);
      
      const logsTab = screen.getByText('logs');
      fireEvent.click(logsTab);
      
      expect(screen.getByText('No debug logs available')).toBeInTheDocument();
    });

    it('clears logs when clear button clicked', async () => {
      render(<DebugTools {...mockProps} />);
      
      // First, generate some logs by triggering an action
      const analyzeButton = screen.getByText('Analyze Low Confidence');
      fireEvent.click(analyzeButton);
      
      // Switch to logs tab
      const logsTab = screen.getByText('logs');
      fireEvent.click(logsTab);
      
      // Clear logs
      const clearButton = screen.getByText('Clear Logs');
      fireEvent.click(clearButton);
      
      // Should show no logs message
      expect(screen.getByText('No debug logs available')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles API errors gracefully', async () => {
      // Mock fetch to throw error
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      render(<DebugTools {...mockProps} />);
      
      // Trigger an action that calls API
      const analyzeButton = screen.getByText('Analyze Low Confidence');
      fireEvent.click(analyzeButton);
      
      // Should not crash the component
      await waitFor(() => {
        expect(screen.getByText('Debug Tools')).toBeInTheDocument();
      });
    });

    it('shows loading states during API calls', async () => {
      // Mock slow API response
      mockFetch.mockImplementation(() => 
        new Promise(resolve => 
          setTimeout(() => resolve({
            ok: true,
            json: async () => ({ success: true, report: { issues: [] } })
          }), 100)
        )
      );

      render(<DebugTools {...mockProps} />);
      
      const analyzeButton = screen.getByText('Analyze Low Confidence');
      fireEvent.click(analyzeButton);
      
      // Should show loading state
      expect(screen.getByText('Analyzing...')).toBeInTheDocument();
      
      // Wait for loading to finish
      await waitFor(() => {
        expect(screen.queryByText('Analyzing...')).not.toBeInTheDocument();
      }, { timeout: 2000 });
    });
  });

  describe('Integration with Debug Actions', () => {
    it('calls onDebugAction prop when actions are triggered', async () => {
      const mockOnDebugAction = vi.fn();
      
      render(<DebugTools {...mockProps} onDebugAction={mockOnDebugAction} />);
      
      // Trigger reset action
      const resetButton = screen.getByText('Reset');
      fireEvent.click(resetButton);
      
      expect(mockOnDebugAction).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'reset',
          timestamp: expect.any(Date)
        })
      );
    });
  });

  describe('Responsive Design', () => {
    it('adapts to smaller screens', () => {
      // Mock window dimensions
      Object.defineProperty(window, 'innerWidth', {
        value: 768,
        configurable: true
      });

      render(<DebugTools {...mockProps} />);
      
      // Component should render without issues on smaller screens
      expect(screen.getByText('Debug Tools')).toBeInTheDocument();
      
      // Tabs should still be accessible
      expect(screen.getByText('analysis')).toBeInTheDocument();
      expect(screen.getByText('sessions')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels and roles', () => {
      render(<DebugTools {...mockProps} />);
      
      // Check for buttons
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
      
      // Check for inputs
      const inputs = screen.getAllByRole('textbox');
      expect(inputs.length).toBeGreaterThan(0);
      
      // Each input should have associated label
      inputs.forEach(input => {
        const labels = screen.getAllByText(/task id|task content|confidence id/i);
        expect(labels.length).toBeGreaterThan(0);
      });
    });

    it('supports keyboard navigation', () => {
      render(<DebugTools {...mockProps} />);
      
      // Tab navigation should work
      const firstButton = screen.getByText('Analyze Low Confidence');
      firstButton.focus();
      expect(document.activeElement).toBe(firstButton);
      
      // Tab key should move focus
      fireEvent.keyDown(firstButton, { key: 'Tab' });
      // Note: Full keyboard navigation testing would require more sophisticated setup
    });
  });
});