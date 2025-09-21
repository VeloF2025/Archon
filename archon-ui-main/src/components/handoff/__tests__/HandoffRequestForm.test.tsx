/**
 * Tests for Handoff Request Form Component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HandoffRequestForm } from '../HandoffRequestForm';
import { handoffService } from '../../../services/handoffService';
import { HandoffStrategy, HandoffTrigger } from '../../../types/handoffTypes';

// Mock the handoff service
jest.mock('../../../services/handoffService');

const mockHandoffService = handoffService as jest.Mocked<typeof handoffService>;

describe('HandoffRequestForm', () => {
  const defaultProps = {
    projectId: 'test-project',
    sourceAgentId: 'agent-1',
    onHandoffRequest: jest.fn(),
    availableAgents: [
      {
        agent_id: 'agent-2',
        agent_name: 'Security Auditor',
        agent_type: 'SECURITY_AUDITOR',
        current_status: 'available' as const,
        handoff_stats: {
          initiated_today: 2,
          received_today: 3,
          success_rate: 0.9,
          avg_response_time: 1500
        },
        capabilities: ['security_analysis', 'vulnerability_assessment'],
        load_factor: 0.2
      },
      {
        agent_id: 'agent-3',
        agent_name: 'System Architect',
        agent_type: 'SYSTEM_ARCHITECT',
        current_status: 'busy' as const,
        handoff_stats: {
          initiated_today: 1,
          received_today: 1,
          success_rate: 0.85,
          avg_response_time: 2000
        },
        capabilities: ['system_design', 'architecture_planning'],
        load_factor: 0.8
      }
    ],
    predefinedTasks: [
      {
        id: 'security-audit',
        title: 'Security Audit',
        description: 'Conduct comprehensive security analysis and vulnerability assessment',
        required_capabilities: ['security_analysis', 'vulnerability_assessment'],
        recommended_strategy: HandoffStrategy.SEQUENTIAL,
        estimated_complexity: 4
      },
      {
        id: 'code-review',
        title: 'Code Review',
        description: 'Review code quality and provide improvement suggestions',
        required_capabilities: ['code_analysis', 'quality_assessment'],
        recommended_strategy: HandoffStrategy.COLLABORATIVE,
        estimated_complexity: 3
      }
    ]
  };

  const mockRecommendations = [
    {
      agent_id: 'agent-2',
      agent_name: 'Security Auditor',
      match_score: 0.92,
      expertise_score: 0.95,
      performance_score: 0.90,
      availability_score: 0.80,
      load_balance_score: 0.85,
      reasoning: 'Perfect match for security-related tasks',
      estimated_response_time: 1200
    },
    {
      agent_id: 'agent-3',
      agent_name: 'System Architect',
      match_score: 0.75,
      expertise_score: 0.80,
      performance_score: 0.85,
      availability_score: 0.20,
      load_balance_score: 0.60,
      reasoning: 'Good capability match but currently busy',
      estimated_response_time: 3000
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockHandoffService.getHandoffRecommendations.mockResolvedValue({
      task_id: 'task-1',
      task_description: 'Test task',
      current_agent_id: 'agent-1',
      recommended: true,
      recommended_agents: mockRecommendations,
      required_capabilities: ['security_analysis'],
      reasoning: 'Security analysis required',
      confidence_score: 0.88,
      generated_at: new Date()
    });

    mockHandoffService.requestHandoff.mockResolvedValue({
      handoff_id: 'new-handoff-1',
      status: 'completed' as any,
      source_agent_id: 'agent-1',
      target_agent_id: 'agent-2',
      execution_time: 1500,
      metrics: {}
    });
  });

  it('renders form with all fields', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    expect(screen.getByText('Create Handoff Request')).toBeInTheDocument();
    expect(screen.getByLabelText('Target Agent *')).toBeInTheDocument();
    expect(screen.getByLabelText('Task Description *')).toBeInTheDocument();
    expect(screen.getByLabelText('Message for Target Agent *')).toBeInTheDocument();
    expect(screen.getByLabelText('Handoff Strategy *')).toBeInTheDocument();
    expect(screen.getByText('Priority')).toBeInTheDocument();
  });

  it('displays predefined tasks', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    expect(screen.getByText('Predefined Tasks')).toBeInTheDocument();
    expect(screen.getByText('Security Audit')).toBeInTheDocument();
    expect(screen.getByText('Code Review')).toBeInTheDocument();
  });

  it('filters out source agent from target options', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    const select = screen.getByLabelText('Target Agent *');
    const options = select.querySelectorAll('option');

    // Should have "Select..." + 2 agents (source agent filtered out)
    expect(options.length).toBe(3);
    expect(options[1].value).toBe('agent-2');
    expect(options[2].value).toBe('agent-3');
  });

  it('filters out offline agents from target options', () => {
    const propsWithOfflineAgent = {
      ...defaultProps,
      availableAgents: [
        ...defaultProps.availableAgents,
        {
          agent_id: 'agent-4',
          agent_name: 'Offline Agent',
          agent_type: 'OFFLINE_AGENT',
          current_status: 'offline' as const,
          handoff_stats: { initiated_today: 0, received_today: 0, success_rate: 0, avg_response_time: 0 },
          capabilities: [],
          load_factor: 0
        }
      ]
    };

    render(<HandoffRequestForm {...propsWithOfflineAgent} />);

    const select = screen.getByLabelText('Target Agent *');
    const options = select.querySelectorAll('option');

    // Should not include offline agent
    expect(Array.from(options).map(opt => opt.value)).not.toContain('agent-4');
  });

  it('handles predefined task selection', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    const select = screen.getByText('Select a predefined task...');
    fireEvent.change(select, { target: { value: 'security-audit' } });

    expect(screen.getByDisplayValue('Conduct comprehensive security analysis and vulnerability assessment')).toBeInTheDocument();
  });

  it('validates required fields on submit', async () => {
    render(<HandoffRequestForm {...defaultProps} />);

    const submitButton = screen.getByText('Create Handoff Request');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Please select a target agent')).toBeInTheDocument();
      expect(screen.getByText('Please provide a task description')).toBeInTheDocument();
      expect(screen.getByText('Please provide a message for the handoff')).toBeInTheDocument();
    });
  });

  it('submits form with valid data', async () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill form
    fireEvent.change(screen.getByLabelText('Target Agent *'), { target: { value: 'agent-2' } });
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test security audit' } });
    fireEvent.change(screen.getByLabelText('Message for Target Agent *'), { target: { value: 'Please help with security audit' } });

    const submitButton = screen.getByText('Create Handoff Request');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockHandoffService.requestHandoff).toHaveBeenCalledWith(
        expect.objectContaining({
          source_agent_id: 'agent-1',
          target_agent_id: 'agent-2',
          task_description: 'Test security audit',
          message: 'Please help with security audit',
          strategy: HandoffStrategy.SEQUENTIAL,
          trigger: HandoffTrigger.MANUAL_REQUEST,
          confidence_score: 0.8,
          priority: 3
        })
      );
    });

    expect(defaultProps.onHandoffRequest).toHaveBeenCalled();
  });

  it('gets recommendations when button clicked', async () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill required fields
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Security analysis needed' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    fireEvent.click(recommendationsButton);

    await waitFor(() => {
      expect(mockHandoffService.getHandoffRecommendations).toHaveBeenCalledWith(
        'Security analysis needed',
        'agent-1'
      );
    });
  });

  it('displays AI recommendations when received', async () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill required fields
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Security analysis needed' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    fireEvent.click(recommendationsButton);

    await waitFor(() => {
      expect(screen.getByText('AI Recommendations:')).toBeInTheDocument();
      expect(screen.getByText('Security Auditor (Match: 92%)')).toBeInTheDocument();
      expect(screen.getByText('System Architect (Match: 75%)')).toBeInTheDocument();
    });
  });

  it('auto-selects best recommendation', async () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill required fields
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Security analysis needed' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    fireEvent.click(recommendationsButton);

    await waitFor(() => {
      const targetSelect = screen.getByLabelText('Target Agent *') as HTMLSelectElement;
      expect(targetSelect.value).toBe('agent-2'); // Best match should be selected
    });
  });

  it('disables recommendations button when requirements not met', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    expect(recommendationsButton).toBeDisabled();

    // Fill task description
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });

    // Should still be disabled without source agent
    expect(recommendationsButton).toBeDisabled();
  });

  it('enables recommendations button when requirements met', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill task description (source agent is provided via props)
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    expect(recommendationsButton).not.toBeDisabled();
  });

  it('shows loading state during recommendation fetch', async () => {
    mockHandoffService.getHandoffRecommendations.mockImplementationOnce(
      () => new Promise(resolve => setTimeout(resolve, 1000))
    );

    render(<HandoffRequestForm {...defaultProps} />);

    // Fill required fields
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    fireEvent.click(recommendationsButton);

    expect(screen.getByText('Getting Recommendations...')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.queryByText('Getting Recommendations...')).not.toBeInTheDocument();
    }, 1100);
  });

  it('handles recommendation fetch error', async () => {
    mockHandoffService.getHandoffRecommendations.mockRejectedValue(new Error('API Error'));

    render(<HandoffRequestForm {...defaultProps} />);

    // Fill required fields
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });

    const recommendationsButton = screen.getByText('Get AI Recommendations');
    fireEvent.click(recommendationsButton);

    await waitFor(() => {
      expect(screen.getByText('Failed to get recommendations')).toBeInTheDocument();
    });
  });

  it('clears form when clear button clicked', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    // Fill some fields
    fireEvent.change(screen.getByLabelText('Target Agent *'), { target: { value: 'agent-2' } });
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });
    fireEvent.change(screen.getByLabelText('Message for Target Agent *'), { target: { value: 'Test message' } });

    const clearButton = screen.getByText('Clear');
    fireEvent.click(clearButton);

    expect(screen.getByDisplayValue('')).toBeInTheDocument(); // Task description should be cleared
    expect(screen.getByDisplayValue('')).toBeInTheDocument(); // Message should be cleared
  });

  it('disables submit button while loading', async () => {
    mockHandoffService.requestHandoff.mockImplementationOnce(
      () => new Promise(resolve => setTimeout(resolve, 1000))
    );

    render(<HandoffRequestForm {...defaultProps} />);

    // Fill form
    fireEvent.change(screen.getByLabelText('Target Agent *'), { target: { value: 'agent-2' } });
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });
    fireEvent.change(screen.getByLabelText('Message for Target Agent *'), { target: { value: 'Test message' } });

    const submitButton = screen.getByText('Create Handoff Request');
    fireEvent.click(submitButton);

    expect(screen.getByText('Creating Handoff...')).toBeInTheDocument();
    expect(submitButton).toBeDisabled();

    await waitFor(() => {
      expect(screen.queryByText('Creating Handoff...')).not.toBeInTheDocument();
    }, 1100);
  });

  it('shows all strategy options', () => {
    render(<HandoffRequestForm {...defaultProps} />);

    const strategySelect = screen.getByLabelText('Handoff Strategy *');
    const options = strategySelect.querySelectorAll('option');

    expect(options.length).toBe(6); // 5 strategies + empty option
    expect(Array.from(options).map(opt => opt.text)).toContain('Sequential - Agents work in sequence, one after another');
    expect(Array.from(options).map(opt => opt.text)).toContain('Collaborative - Agents work together on the same task');
  });

  it('handles submit error', async () => {
    mockHandoffService.requestHandoff.mockRejectedValue(new Error('Submission failed'));

    render(<HandoffRequestForm {...defaultProps} />);

    // Fill form
    fireEvent.change(screen.getByLabelText('Target Agent *'), { target: { value: 'agent-2' } });
    fireEvent.change(screen.getByLabelText('Task Description *'), { target: { value: 'Test task' } });
    fireEvent.change(screen.getByLabelText('Message for Target Agent *'), { target: { value: 'Test message' } });

    const submitButton = screen.getByText('Create Handoff Request');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Failed to submit handoff request')).toBeInTheDocument();
    });
  });

  it('works without predefined tasks', () => {
    const propsWithoutTasks = {
      ...defaultProps,
      predefinedTasks: []
    };

    render(<HandoffRequestForm {...propsWithoutTasks} />);

    expect(screen.queryByText('Predefined Tasks')).not.toBeInTheDocument();
    expect(screen.getByText('Select a predefined task...')).toBeInTheDocument();
  });

  it('works without source agent', () => {
    const propsWithoutSource = {
      ...defaultProps,
      sourceAgentId: undefined
    };

    render(<HandoffRequestForm {...propsWithoutSource} />);

    // Should still render normally
    expect(screen.getByText('Create Handoff Request')).toBeInTheDocument();
  });
});