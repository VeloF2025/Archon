import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PropertyEditorWithProvider } from '../PropertyEditor';
import { AgentType, ModelTier, AgentState } from '../../../types/agentTypes';
import { ConnectionType, MessageType } from '../../../types/workflowTypes';

describe('PropertyEditor', () => {
  const mockAgentNode = {
    id: 'agent-1',
    type: 'agent',
    position: { x: 100, y: 100 },
    data: {
      agentId: 'agent-1',
      name: 'Test Agent',
      type: AgentType.CODE_IMPLEMENTER,
      tier: ModelTier.SONNET,
      state: AgentState.ACTIVE,
      capabilities: { code_generation: true, error_handling: true },
      config: {
        max_retries: 3,
        timeout_seconds: 30,
        priority: 'medium'
      }
    }
  };

  const mockConnectionEdge = {
    id: 'edge-1',
    source: 'agent-1',
    target: 'agent-2',
    type: ConnectionType.DIRECT,
    data: {
      connectionType: ConnectionType.DIRECT,
      messageType: MessageType.TASK_ASSIGNMENT,
      config: {
        retry_strategy: 'exponential_backoff',
        timeout_seconds: 30,
        priority: 'medium'
      }
    }
  };

  const defaultProps = {
    selectedElement: mockAgentNode,
    onUpdate: jest.fn(),
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders property editor with header for agent node', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      expect(screen.getByText('Property Editor')).toBeInTheDocument();
      expect(screen.getByText('Test Agent')).toBeInTheDocument();
      expect(screen.getByText('Agent Properties')).toBeInTheDocument();
    });

    it('renders property editor for connection edge', () => {
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      expect(screen.getByText('Property Editor')).toBeInTheDocument();
      expect(screen.getByText('Connection Properties')).toBeInTheDocument();
    });

    it('renders tab navigation', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      expect(screen.getByText('Basic')).toBeInTheDocument();
      expect(screen.getByText('Advanced')).toBeInTheDocument();
      expect(screen.getByText('Capabilities')).toBeInTheDocument();
    });

    it('shows close button', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      expect(screen.getByText('×')).toBeInTheDocument();
    });
  });

  describe('Tab Navigation', () => {
    it('switches between tabs when clicked', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Initially on Basic tab
      expect(screen.getByText('Name:')).toBeInTheDocument();

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));
      expect(screen.getByText('Max Retries:')).toBeInTheDocument();

      // Switch to Capabilities tab
      await user.click(screen.getByText('Capabilities'));
      expect(screen.getByText('Capabilities:')).toBeInTheDocument();
    });

    it('maintains tab state', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      // Re-render to test state persistence
      rerender(<PropertyEditorWithProvider {...defaultProps} />);

      // Should still be on Advanced tab
      expect(screen.getByText('Max Retries:')).toBeInTheDocument();
    });
  });

  describe('Agent Property Editing', () => {
    it('allows editing agent name', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const nameInput = screen.getByDisplayValue('Test Agent');
      await user.clear(nameInput);
      await user.type(nameInput, 'Updated Agent');

      expect(nameInput).toHaveValue('Updated Agent');
    });

    it('allows changing agent type', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const typeSelect = screen.getByDisplayValue('CODE_IMPLEMENTER');
      await user.selectOptions(typeSelect, 'SECURITY_AUDITOR');

      expect(typeSelect).toHaveValue('SECURITY_AUDITOR');
    });

    it('allows changing model tier', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const tierSelect = screen.getByDisplayValue('SONNET');
      await user.selectOptions(tierSelect, 'OPUS');

      expect(tierSelect).toHaveValue('OPUS');
    });

    it('allows changing agent state', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const stateSelect = screen.getByDisplayValue('ACTIVE');
      await user.selectOptions(stateSelect, 'PAUSED');

      expect(stateSelect).toHaveValue('PAUSED');
    });

    it('calls onUpdate when properties change', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const nameInput = screen.getByDisplayValue('Test Agent');
      await user.clear(nameInput);
      await user.type(nameInput, 'Updated Agent');

      expect(defaultProps.onUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({
            name: 'Updated Agent'
          })
        })
      );
    });
  });

  describe('Advanced Configuration', () => {
    it('renders advanced agent configuration', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      expect(screen.getByText('Max Retries:')).toBeInTheDocument();
      expect(screen.getByText('Timeout (seconds):')).toBeInTheDocument();
      expect(screen.getByText('Priority:')).toBeInTheDocument();
    });

    it('allows changing max retries', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      const retriesInput = screen.getByDisplayValue('3');
      await user.clear(retriesInput);
      await user.type(retriesInput, '5');

      expect(retriesInput).toHaveValue(5);
    });

    it('allows changing timeout', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      const timeoutInput = screen.getByDisplayValue('30');
      await user.clear(timeoutInput);
      await user.type(timeoutInput, '60');

      expect(timeoutInput).toHaveValue(60);
    });

    it('allows changing priority', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      await user.click(screen.getByText('High'));
      expect(screen.getByText('High').parentElement).toHaveClass('bg-blue-500');
    });
  });

  describe('Capabilities Configuration', () => {
    it('renders capabilities checkboxes', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Capabilities tab
      await user.click(screen.getByText('Capabilities'));

      expect(screen.getByText('code_generation')).toBeInTheDocument();
      expect(screen.getByText('error_handling')).toBeInTheDocument();
    });

    it('allows toggling capabilities', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Capabilities tab
      await user.click(screen.getByText('Capabilities'));

      const codeGenCheckbox = screen.getByLabelText('code_generation');
      await user.click(codeGenCheckbox);

      expect(codeGenCheckbox).not.toBeChecked();
    });

    it('shows capability descriptions', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Capabilities tab
      await user.click(screen.getByText('Capabilities'));

      expect(screen.getByText(/Ability to write production-ready code/)).toBeInTheDocument();
    });
  });

  describe('Connection Property Editing', () => {
    it('renders connection properties', () => {
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      expect(screen.getByText('Connection Type:')).toBeInTheDocument();
      expect(screen.getByText('Message Type:')).toBeInTheDocument();
    });

    it('allows changing connection type', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      const typeSelect = screen.getByDisplayValue('DIRECT');
      await user.selectOptions(typeSelect, 'BROADCAST');

      expect(typeSelect).toHaveValue('BROADCAST');
    });

    it('allows changing message type', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      const messageTypeSelect = screen.getByDisplayValue('TASK_ASSIGNMENT');
      await user.selectOptions(messageTypeSelect, 'STATUS_UPDATE');

      expect(messageTypeSelect).toHaveValue('STATUS_UPDATE');
    });

    it('renders connection configuration', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      expect(screen.getByText('Retry Strategy:')).toBeInTheDocument();
      expect(screen.getByText('Priority:')).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('validates required fields', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const nameInput = screen.getByDisplayValue('Test Agent');
      await user.clear(nameInput);

      // Should show validation error
      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });

    it('validates numeric inputs', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Switch to Advanced tab
      await user.click(screen.getByText('Advanced'));

      const retriesInput = screen.getByDisplayValue('3');
      await user.clear(retriesInput);
      await user.type(retriesInput, '-1');

      // Should show validation error
      expect(screen.getByText('Max retries must be positive')).toBeInTheDocument();
    });

    it('prevents invalid values', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const nameInput = screen.getByDisplayValue('Test Agent');
      await user.clear(nameInput);
      await user.type(nameInput, '');

      // Should not allow empty name
      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });
  });

  describe('Apply and Cancel Actions', () => {
    it('calls onUpdate when Apply is clicked', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Apply'));

      expect(defaultProps.onUpdate).toHaveBeenCalledWith(mockAgentNode);
    });

    it('calls onClose when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Cancel'));

      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('calls onClose when close button is clicked', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      await user.click(screen.getByText('×'));

      expect(defaultProps.onClose).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for form elements', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const nameInput = screen.getByLabelText('Name:');
      expect(nameInput).toBeInTheDocument();
      expect(nameInput).toHaveAttribute('id');

      const typeSelect = screen.getByLabelText('Type:');
      expect(typeSelect).toBeInTheDocument();
    });

    it('supports keyboard navigation', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const inputs = screen.getAllByRole('textbox');
      const selects = screen.getAllByRole('combobox');

      inputs.forEach(input => {
        expect(input).toBeInTheDocument();
      });

      selects.forEach(select => {
        expect(select).toBeInTheDocument();
      });
    });

    it('provides proper tab order', () => {
      render(<PropertyEditorWithProvider {...defaultProps} />);

      const focusableElements = screen.getAllByRole('textbox', 'combobox', 'button');
      focusableElements.forEach(element => {
        expect(element).toBeInTheDocument();
      });
    });
  });

  describe('State Management', () => {
    it('maintains internal state for form changes', async () => {
      const user = userEvent.setup();
      render(<PropertyEditorWithProvider {...defaultProps} />);

      // Make changes
      const nameInput = screen.getByDisplayValue('Test Agent');
      await user.clear(nameInput);
      await user.type(nameInput, 'Updated Agent');

      // Change should be reflected
      expect(nameInput).toHaveValue('Updated Agent');

      // Should not call onUpdate until Apply is clicked
      expect(defaultProps.onUpdate).not.toHaveBeenCalled();
    });

    it('resets form when selected element changes', () => {
      const { rerender } = render(<PropertyEditorWithProvider {...defaultProps} />);

      // Change form values
      const nameInput = screen.getByDisplayValue('Test Agent');
      fireEvent.change(nameInput, { target: { value: 'Modified Agent' } });

      // Change selected element
      rerender(<PropertyEditorWithProvider {...defaultProps} selectedElement={mockConnectionEdge} />);

      // Should show connection properties instead
      expect(screen.getByText('Connection Properties')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles missing selected element gracefully', () => {
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={null} />);

      expect(screen.getByText('No element selected')).toBeInTheDocument();
    });

    it('handles invalid selected element type gracefully', () => {
      const invalidElement = { id: 'invalid', type: 'unknown' };
      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={invalidElement as any} />);

      expect(screen.getByText('Unknown element type')).toBeInTheDocument();
    });

    it('handles onUpdate errors gracefully', () => {
      const mockOnUpdate = jest.fn().mockImplementation(() => {
        throw new Error('Update failed');
      });

      render(<PropertyEditorWithProvider {...defaultProps} onUpdate={mockOnUpdate} />);

      fireEvent.click(screen.getByText('Apply'));

      // Should not crash the component
      expect(screen.getByText('Property Editor')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('renders efficiently with rapid property changes', () => {
      const { rerender } = render(<PropertyEditorWithProvider {...defaultProps} />);

      // Simulate rapid changes
      for (let i = 0; i < 10; i++) {
        rerender(<PropertyEditorWithProvider {...defaultProps} selectedElement={{
          ...mockAgentNode,
          data: { ...mockAgentNode.data, name: `Agent ${i}` }
        }} />);
      }

      expect(screen.getByText('Property Editor')).toBeInTheDocument();
    });

    it('handles complex nested configuration objects', () => {
      const complexAgent = {
        ...mockAgentNode,
        data: {
          ...mockAgentNode.data,
          config: {
            ...mockAgentNode.data.config,
            custom_settings: {
              nested_level_1: {
                nested_level_2: {
                  value: 'deeply nested'
                }
              }
            }
          }
        }
      };

      render(<PropertyEditorWithProvider {...defaultProps} selectedElement={complexAgent} />);

      expect(screen.getByText('Property Editor')).toBeInTheDocument();
    });
  });
});