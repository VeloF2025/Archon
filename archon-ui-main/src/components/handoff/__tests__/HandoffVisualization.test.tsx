/**
 * Tests for Handoff Visualization Component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HandoffVisualization } from '../HandoffVisualization';
import { handoffService } from '../../../services/handoffService';
import { HandoffStrategy, HandoffStatus } from '../../../types/handoffTypes';

// Mock the handoff service
jest.mock('../../../services/handoffService');

const mockHandoffService = handoffService as jest.Mocked<typeof handoffService>;

describe('HandoffVisualization', () => {
  const defaultProps = {
    projectId: 'test-project',
    refreshInterval: 1000,
    maxHistoryItems: 10,
    showMetrics: true,
    showControls: true,
  };

  const mockActiveHandoffs = [
    {
      handoff_id: 'handoff-1',
      source_agent: 'agent-1',
      target_agent: 'agent-2',
      status: HandoffStatus.IN_PROGRESS,
      progress: 65,
      strategy: HandoffStrategy.SEQUENTIAL,
      start_time: new Date(),
      confidence_score: 0.85,
      task_description: 'Code review for authentication module'
    },
    {
      handoff_id: 'handoff-2',
      source_agent: 'agent-3',
      target_agent: 'agent-1',
      status: HandoffStatus.PENDING,
      progress: 0,
      strategy: HandoffStrategy.COLLABORATIVE,
      start_time: new Date(),
      confidence_score: 0.72,
      task_description: 'Security audit of payment system'
    }
  ];

  const mockHandoffHistory = [
    {
      handoff_id: 'handoff-3',
      source_agent: 'agent-2',
      target_agent: 'agent-3',
      status: HandoffStatus.COMPLETED,
      strategy: HandoffStrategy.SEQUENTIAL,
      duration: 1500,
      success: true,
      timestamp: new Date(Date.now() - 3600000), // 1 hour ago
      task_summary: 'Database optimization completed'
    },
    {
      handoff_id: 'handoff-4',
      source_agent: 'agent-1',
      target_agent: 'agent-2',
      status: HandoffStatus.FAILED,
      strategy: HandoffStrategy.PARALLEL,
      duration: 3000,
      success: false,
      timestamp: new Date(Date.now() - 7200000), // 2 hours ago
      task_summary: 'API integration attempt failed'
    }
  ];

  const mockAgentStates = [
    {
      agent_id: 'agent-1',
      agent_name: 'Code Implementer',
      agent_type: 'CODE_IMPLEMENTER',
      current_status: 'available' as const,
      handoff_stats: {
        initiated_today: 3,
        received_today: 2,
        success_rate: 0.85,
        avg_response_time: 1200
      },
      capabilities: ['python_programming', 'web_development'],
      load_factor: 0.3
    },
    {
      agent_id: 'agent-2',
      agent_name: 'Security Auditor',
      agent_type: 'SECURITY_AUDITOR',
      current_status: 'busy' as const,
      handoff_stats: {
        initiated_today: 1,
        received_today: 4,
        success_rate: 0.92,
        avg_response_time: 1800
      },
      capabilities: ['security_analysis', 'vulnerability_assessment'],
      load_factor: 0.7
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock service methods
    mockHandoffService.getActiveHandoffs.mockResolvedValue(mockActiveHandoffs);
    mockHandoffService.getHandoffHistory.mockResolvedValue(mockHandoffHistory);
    mockHandoffService.getHandoffVisualization.mockResolvedValue({
      active_handoffs: mockActiveHandoffs,
      handoff_history: mockHandoffHistory,
      agent_states: mockAgentStates,
      performance_metrics: {
        total_handoffs_today: 5,
        success_rate_today: 0.8,
        avg_handoff_time: 1800,
        most_used_strategy: HandoffStrategy.SEQUENTIAL,
        most_active_agent: 'agent-1',
        confidence_trend: 0.1,
        performance_score: 0.85
      },
      last_updated: new Date()
    });
  });

  it('renders loading state initially', () => {
    render(<HandoffVisualization {...defaultProps} />);

    expect(screen.getByText('Loading handoff visualization...')).toBeInTheDocument();
  });

  it('renders handoff visualization after data loads', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.queryByText('Loading handoff visualization...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Agent Handoff Visualization')).toBeInTheDocument();
    expect(screen.getByText('Handoffs Today')).toBeInTheDocument();
    expect(screen.getByText('Success Rate')).toBeInTheDocument();
  });

  it('displays metrics correctly', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument(); // Total handoffs today
    });

    expect(screen.getByText('80.0%')).toBeInTheDocument(); // Success rate
  });

  it('displays active handoffs', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Code review for authentication module')).toBeInTheDocument();
    });

    expect(screen.getByText('Security audit of payment system')).toBeInTheDocument();
    expect(screen.getByText('65%')).toBeInTheDocument(); // Progress
  });

  it('displays agent states', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Code Implementer')).toBeInTheDocument();
    });

    expect(screen.getByText('Security Auditor')).toBeInTheDocument();
    expect(screen.getByText('30%')).toBeInTheDocument(); // Load factor
  });

  it('displays handoff history', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Database optimization completed')).toBeInTheDocument();
    });

    expect(screen.getByText('API integration attempt failed')).toBeInTheDocument();
  });

  it('handles handoff selection', async () => {
    const onHandoffSelect = jest.fn();
    render(<HandoffVisualization {...defaultProps} onHandoffSelect={onHandoffSelect} />);

    await waitFor(() => {
      const handoffCard = screen.getByText('Code review for authentication module').closest('div');
      if (handoffCard) {
        fireEvent.click(handoffCard);
      }
    });

    expect(onHandoffSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        handoff_id: 'handoff-1',
        task_description: 'Code review for authentication module'
      })
    );
  });

  it('handles agent selection', async () => {
    const onAgentSelect = jest.fn();
    render(<HandoffVisualization {...defaultProps} onAgentSelect={onAgentSelect} />);

    await waitFor(() => {
      const agentCard = screen.getByText('Code Implementer').closest('div');
      if (agentCard) {
        fireEvent.click(agentCard);
      }
    });

    expect(onAgentSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        agent_id: 'agent-1',
        agent_name: 'Code Implementer'
      })
    );
  });

  it('handles refresh button click', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      const refreshButton = screen.getByText('Refresh Now');
      fireEvent.click(refreshButton);
    });

    expect(mockHandoffService.getActiveHandoffs).toHaveBeenCalledTimes(2);
    expect(mockHandoffService.getHandoffHistory).toHaveBeenCalledTimes(2);
    expect(mockHandoffService.getHandoffVisualization).toHaveBeenCalledTimes(2);
  });

  it('toggles auto refresh', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      const autoRefreshButton = screen.getByText('Auto Refresh: ON');
      fireEvent.click(autoRefreshButton);
    });

    expect(screen.getByText('Auto Refresh: OFF')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Auto Refresh: OFF'));
    expect(screen.getByText('Auto Refresh: ON')).toBeInTheDocument();
  });

  it('displays empty state when no active handoffs', async () => {
    mockHandoffService.getActiveHandoffs.mockResolvedValue([]);
    mockHandoffService.getHandoffHistory.mockResolvedValue([]);
    mockHandoffService.getHandoffVisualization.mockResolvedValue({
      active_handoffs: [],
      handoff_history: [],
      agent_states: [],
      performance_metrics: {
        total_handoffs_today: 0,
        success_rate_today: 0,
        avg_handoff_time: 0,
        most_used_strategy: HandoffStrategy.SEQUENTIAL,
        most_active_agent: '',
        confidence_trend: 0,
        performance_score: 0
      },
      last_updated: new Date()
    });

    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('No active handoffs')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    mockHandoffService.getActiveHandoffs.mockRejectedValue(new Error('API Error'));

    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Error Loading Handoff Data')).toBeInTheDocument();
    });

    expect(screen.getByText('Failed to load handoff data')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('retries loading data on retry button click', async () => {
    mockHandoffService.getActiveHandoffs.mockRejectedValueOnce(new Error('API Error'));

    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Retry')).toBeInTheDocument();
      fireEvent.click(screen.getByText('Retry'));
    });

    expect(mockHandoffService.getActiveHandoffs).toHaveBeenCalledTimes(2);
  });

  it('formats duration correctly', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('1.5s')).toBeInTheDocument(); // 1500ms formatted
    });
  });

  it('displays correct strategy labels', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Sequential')).toBeInTheDocument();
      expect(screen.getByText('Collaborative')).toBeInTheDocument();
    });
  });

  it('updates last refresh time', async () => {
    render(<HandoffVisualization {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
    });
  });

  it('respects maxHistoryItems prop', async () => {
    const longHistory = Array(20).fill(null).map((_, i) => ({
      ...mockHandoffHistory[0],
      handoff_id: `handoff-${i}`,
      timestamp: new Date(Date.now() - i * 3600000)
    }));

    mockHandoffService.getHandoffHistory.mockResolvedValue(longHistory);

    render(<HandoffVisualization {...defaultProps} maxHistoryItems={5} />);

    await waitFor(() => {
      expect(screen.getByText(/Last 5 handoffs/)).toBeInTheDocument();
    });
  });
});