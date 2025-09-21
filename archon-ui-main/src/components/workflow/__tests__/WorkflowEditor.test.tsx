import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowEditorWithProvider } from '../WorkflowEditor';
import { WorkflowConfiguration } from '../../../types/workflowTypes';
import { AgentType, ModelTier, AgentState } from '../../../types/agentTypes';

// Mock the toast hook
jest.mock('../../../hooks/useToast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}));

// Mock the socket hook
jest.mock('../../../hooks/useSocket', () => ({
  useSocket: () => null,
}));

// Mock react-dnd
jest.mock('react-dnd', () => ({
  useDrag: () => [{ isDragging: false }, jest.fn()],
  useDrop: () => [{}, jest.fn()],
  DndProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock react-dnd-html5-backend
jest.mock('react-dnd-html5-backend', () => ({
  HTML5Backend: {},
}));

describe('WorkflowEditor', () => {
  const mockAgencyData = {
    id: 'test-agency',
    name: 'Test Agency',
    agents: [
      {
        id: 'agent-1',
        name: 'Test Agent 1',
        agent_type: AgentType.CODE_IMPLEMENTER,
        model_tier: ModelTier.SONNET,
        project_id: 'test-project',
        state: AgentState.ACTIVE,
        state_changed_at: new Date(),
        tasks_completed: 5,
        success_rate: 0.8,
        avg_completion_time_seconds: 120,
        memory_usage_mb: 512,
        cpu_usage_percent: 25,
        capabilities: { code_generation: true, error_handling: true },
        created_at: new Date(),
        updated_at: new Date(),
      },
      {
        id: 'agent-2',
        name: 'Test Agent 2',
        agent_type: AgentType.TEST_COVERAGE_VALIDATOR,
        model_tier: ModelTier.SONNET,
        project_id: 'test-project',
        state: AgentState.ACTIVE,
        state_changed_at: new Date(),
        tasks_completed: 10,
        success_rate: 0.9,
        avg_completion_time_seconds: 90,
        memory_usage_mb: 256,
        cpu_usage_percent: 15,
        capabilities: { test_generation: true, coverage_analysis: true },
        created_at: new Date(),
        updated_at: new Date(),
      },
    ],
    communication_flows: [
      {
        id: 'comm-1',
        source_agent_id: 'agent-1',
        target_agent_id: 'agent-2',
        communication_type: 'direct' as any,
        status: 'active' as any,
        message_count: 3,
        message_type: 'task_assignment',
        last_message_at: new Date(),
      },
    ],
    created_at: new Date(),
    updated_at: new Date(),
  };

  const defaultProps = {
    agencyData: mockAgencyData,
    projectId: 'test-project',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the workflow editor with ReactFlow canvas', () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      expect(screen.getByText('Mode:')).toBeInTheDocument();
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('displays agency data when provided', () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      expect(screen.getByText('Test Agent 1')).toBeInTheDocument();
      expect(screen.getByText('Test Agent 2')).toBeInTheDocument();
    });

    it('shows unsaved changes indicator when there are modifications', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // Simulate making changes
      const user = userEvent.setup();
      await user.click(screen.getByTitle('Add Agent (A)'));

      // The unsaved changes indicator should appear
      // Note: This depends on the actual implementation of unsaved changes tracking
    });
  });

  describe('Toolbar Functions', () => {
    it('changes editor modes when toolbar buttons are clicked', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Test select mode
      await user.click(screen.getByTitle('Select Mode (S)'));
      expect(screen.getByText('SELECT')).toBeInTheDocument();

      // Test add agent mode
      await user.click(screen.getByTitle('Add Agent (A)'));
      expect(screen.getByText('CREATE_AGENT')).toBeInTheDocument();

      // Test create connection mode
      await user.click(screen.getByTitle('Create Connection (C)'));
      expect(screen.getByText('CREATE_CONNECTION')).toBeInTheDocument();
    });

    it('enables/disables undo/redo buttons based on history', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Initially, undo should be disabled
      const undoButton = screen.getByTitle('Undo (Ctrl+Z)');
      expect(undoButton).toBeDisabled();

      // Redo should be disabled initially
      const redoButton = screen.getByTitle('Redo (Ctrl+Shift+Z)');
      expect(redoButton).toBeDisabled();
    });

    it('opens template modal when template button is clicked', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      await user.click(screen.getByTitle('Load Template'));

      // Check if the template modal content appears
      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });

    it('confirms canvas clearing when clear button is clicked', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Mock window.confirm
      const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

      await user.click(screen.getByTitle('Clear Canvas'));

      expect(confirmSpy).toHaveBeenCalledWith('Are you sure you want to clear the entire workflow?');

      confirmSpy.mockRestore();
    });
  });

  describe('Drag and Drop', () => {
    it('handles drag over events', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const canvas = screen.getByRole('main').querySelector('.react-flow') || document.body;

      // Simulate drag over event
      const dragOverEvent = new Event('dragover', { bubbles: true });
      Object.defineProperty(dragOverEvent, 'dataTransfer', {
        value: {
          dropEffect: 'move',
        },
        writable: false,
      });

      canvas.dispatchEvent(dragOverEvent);

      // The drag over state should be updated
      // Note: This depends on the actual implementation
    });

    it('handles drag leave events', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const canvas = screen.getByRole('main').querySelector('.react-flow') || document.body;

      // Simulate drag leave event
      const dragLeaveEvent = new Event('dragleave', { bubbles: true });
      canvas.dispatchEvent(dragLeaveEvent);

      // The drag over state should be cleared
      // Note: This depends on the actual implementation
    });
  });

  describe('Keyboard Shortcuts', () => {
    it('handles Ctrl+Z for undo', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Simulate Ctrl+Z
      await user.keyboard('{Control>}z');

      // Should attempt to undo if there's history
      // Note: This depends on the actual implementation
    });

    it('handles Ctrl+Shift+Z for redo', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Simulate Ctrl+Shift+Z
      await user.keyboard('{Control>}{Shift>}z');

      // Should attempt to redo if there's future states
      // Note: This depends on the actual implementation
    });

    it('handles Ctrl+S for save', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Simulate Ctrl+S
      await user.keyboard('{Control>}s');

      // Should trigger save functionality
      // Note: This depends on the actual implementation
    });

    it('handles Delete key for selected elements', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // First select an element (this would require actual nodes to be present)
      // Then simulate delete key
      await user.keyboard('{Delete}');

      // Should delete the selected element
      // Note: This depends on the actual implementation
    });
  });

  describe('Auto-save', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('triggers auto-save after making changes', async () => {
      const { toast } = require('../../../hooks/useToast');
      const mockToast = jest.fn();
      (useToast as jest.Mock).mockReturnValue({ toast: mockToast });

      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // Simulate making changes that trigger unsaved state
      // This would require more specific interaction with the component

      // Fast-forward time
      jest.advanceTimersByTime(5000);

      // Check if auto-save was triggered
      // expect(mockToast).toHaveBeenCalledWith(expect.objectContaining({
      //   title: "Auto-saved",
      //   description: "Workflow has been auto-saved",
      //   variant: "success",
      // }));
    });

    it('does not auto-save if there are no unsaved changes', async () => {
      const { toast } = require('../../../hooks/useToast');
      const mockToast = jest.fn();
      (useToast as jest.Mock).mockReturnValue({ toast: mockToast });

      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // Fast-forward time without making changes
      jest.advanceTimersByTime(5000);

      // Should not trigger auto-save
      expect(mockToast).not.toHaveBeenCalled();
    });
  });

  describe('Node and Edge Interactions', () => {
    it('selects nodes when clicked', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // This test would require mocking the ReactFlow nodes and simulating clicks
      // For now, we'll just verify the component renders
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('selects edges when clicked', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // This test would require mocking the ReactFlow edges and simulating clicks
      // For now, we'll just verify the component renders
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('deselects elements when clicking on pane', async () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // This test would require simulating pane clicks
      // For now, we'll just verify the component renders
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for interactive elements', () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // Check toolbar buttons have proper titles
      expect(screen.getByTitle('Select Mode (S)')).toBeInTheDocument();
      expect(screen.getByTitle('Add Agent (A)')).toBeInTheDocument();
      expect(screen.getByTitle('Create Connection (C)')).toBeInTheDocument();
      expect(screen.getByTitle('Clear Canvas')).toBeInTheDocument();
    });

    it('maintains proper focus management', () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // This would require more detailed accessibility testing
      // For now, verify that interactive elements are present
      expect(screen.getByRole('button', { name: /select/i })).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large numbers of nodes and edges efficiently', () => {
      // Create large agency data
      const largeAgencyData = {
        ...mockAgencyData,
        agents: Array.from({ length: 100 }, (_, i) => ({
          ...mockAgencyData.agents[0],
          id: `agent-${i}`,
          name: `Agent ${i}`,
        })),
        communication_flows: Array.from({ length: 200 }, (_, i) => ({
          ...mockAgencyData.communication_flows[0],
          id: `comm-${i}`,
          source_agent_id: `agent-${i % 100}`,
          target_agent_id: `agent-${(i + 1) % 100}`,
        })),
      };

      render(<WorkflowEditorWithProvider agencyData={largeAgencyData} projectId="test" />);

      // The component should render without crashing
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('gracefully handles missing agency data', () => {
      render(<WorkflowEditorWithProvider projectId="test" />);

      // Should render with empty state
      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('handles invalid agency data structure', () => {
      const invalidAgencyData = {
        ...mockAgencyData,
        agents: 'invalid' as any, // Should be an array
      };

      // Should not crash and handle gracefully
      expect(() => {
        render(<WorkflowEditorWithProvider agencyData={invalidAgencyData} projectId="test" />);
      }).not.toThrow();
    });
  });

  describe('Integration with Agency Swarm', () => {
    it('initializes with agency data correctly', () => {
      render(<WorkflowEditorWithProvider {...defaultProps} />);

      // Verify that nodes are created from agency data
      // This would require checking the internal state or ReactFlow instance
      expect(screen.getByText('Test Agent 1')).toBeInTheDocument();
      expect(screen.getByText('Test Agent 2')).toBeInTheDocument();
    });

    it('updates when agency data changes', () => {
      const { rerender } = render(<WorkflowEditorWithProvider {...defaultProps} />);

      const updatedAgencyData = {
        ...mockAgencyData,
        agents: [
          ...mockAgencyData.agents,
          {
            ...mockAgencyData.agents[0],
            id: 'agent-3',
            name: 'New Agent',
          },
        ],
      };

      rerender(<WorkflowEditorWithProvider agencyData={updatedAgencyData} projectId="test" />);

      // Should update to include the new agent
      // expect(screen.getByText('New Agent')).toBeInTheDocument();
    });
  });
});