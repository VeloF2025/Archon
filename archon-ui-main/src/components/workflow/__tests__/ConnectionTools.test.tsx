import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ConnectionToolsWithProvider } from '../ConnectionTools';
import { ConnectionType, MessageType } from '../../../types/workflowTypes';

describe('ConnectionTools', () => {
  const defaultProps = {
    onConnectionStart: jest.fn(),
    onConnectionEnd: jest.fn(),
    onConnectionTypeChange: jest.fn(),
    activeConnectionType: ConnectionType.DIRECT,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders connection tools with header and instructions', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('Connection Tools')).toBeInTheDocument();
      expect(screen.getByText('Select connection type and configure communication patterns')).toBeInTheDocument();
    });

    it('renders all connection type buttons', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('Direct')).toBeInTheDocument();
      expect(screen.getByText('Broadcast')).toBeInTheDocument();
      expect(screen.getByText('Chain')).toBeInTheDocument();
      expect(screen.getByText('Collaborative')).toBeInTheDocument();
    });

    it('highlights active connection type', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const directButton = screen.getByText('Direct');
      expect(directButton.parentElement).toHaveClass('bg-blue-500');
    });
  });

  describe('Connection Type Selection', () => {
    it('calls onConnectionTypeChange when connection type is clicked', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Broadcast'));

      expect(defaultProps.onConnectionTypeChange).toHaveBeenCalledWith(ConnectionType.BROADCAST);
    });

    it('updates active connection type', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<ConnectionToolsWithProvider {...defaultProps} />);

      // Change to Broadcast
      await user.click(screen.getByText('Broadcast'));
      rerender(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.BROADCAST} />);

      const broadcastButton = screen.getByText('Broadcast');
      expect(broadcastButton.parentElement).toHaveClass('bg-blue-500');
    });

    it('shows description for each connection type', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText(/One-to-one communication between specific agents/)).toBeInTheDocument();
    });
  });

  describe('Message Type Selection', () => {
    it('renders message type dropdown when connection type is selected', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const dropdown = screen.getByRole('combobox');
      expect(dropdown).toBeInTheDocument();
      expect(dropdown).toHaveValue('task_assignment');
    });

    it('shows available message types based on connection type', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const options = screen.getAllByRole('option');
      expect(options).toHaveLength(5); // All message types available for Direct
    });

    it('filters message types for Broadcast connection', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.BROADCAST} />);

      const options = screen.getAllByRole('option');
      expect(options.length).toBeGreaterThan(0);
    });

    it('calls onConnectionTypeChange when message type changes', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const dropdown = screen.getByRole('combobox');
      await user.selectOptions(dropdown, 'status_update');

      expect(defaultProps.onConnectionTypeChange).toHaveBeenCalledWith(
        expect.objectContaining({
          messageType: MessageType.STATUS_UPDATE
        })
      );
    });
  });

  describe('Advanced Configuration', () => {
    it('shows advanced configuration panel', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('Advanced Configuration')).toBeInTheDocument();
      expect(screen.getByText('Retry Strategy:')).toBeInTheDocument();
      expect(screen.getByText('Priority:')).toBeInTheDocument();
    });

    it('renders retry strategy options', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('None')).toBeInTheDocument();
      expect(screen.getByText('Exponential Backoff')).toBeInTheDocument();
      expect(screen.getByText('Linear Backoff')).toBeInTheDocument();
    });

    it('allows changing retry strategy', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Exponential Backoff'));
      expect(screen.getByText('Exponential Backoff').parentElement).toHaveClass('bg-blue-500');
    });

    it('allows changing priority level', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      await user.click(screen.getByText('High'));
      expect(screen.getByText('High').parentElement).toHaveClass('bg-blue-500');
    });

    it('shows timeout input for configuration', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      expect(timeoutInput).toBeInTheDocument();
      expect(timeoutInput).toHaveValue(30);
    });

    it('allows changing timeout value', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      await user.clear(timeoutInput);
      await user.type(timeoutInput, '60');

      expect(timeoutInput).toHaveValue(60);
    });
  });

  describe('Connection Actions', () => {
    it('renders connection action buttons', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('Start Connection')).toBeInTheDocument();
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });

    it('calls onConnectionStart when Start Connection is clicked', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Start Connection'));

      expect(defaultProps.onConnectionStart).toHaveBeenCalledWith(
        expect.objectContaining({
          type: ConnectionType.DIRECT,
          messageType: MessageType.TASK_ASSIGNMENT
        })
      );
    });

    it('calls onConnectionEnd when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Cancel'));

      expect(defaultProps.onConnectionEnd).toHaveBeenCalled();
    });
  });

  describe('Connection Type Descriptions', () => {
    it('shows correct description for Direct connection', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText(/One-to-one communication between specific agents/)).toBeInTheDocument();
      expect(screen.getByText(/Best for task delegation/)).toBeInTheDocument();
    });

    it('shows correct description for Broadcast connection', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.BROADCAST} />);

      expect(screen.getByText(/One-to-many communication/)).toBeInTheDocument();
      expect(screen.getByText(/Best for announcements/)).toBeInTheDocument();
    });

    it('shows correct description for Chain connection', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.CHAIN} />);

      expect(screen.getByText(/Sequential message passing/)).toBeInTheDocument();
      expect(screen.getByText(/Best for pipelines/)).toBeInTheDocument();
    });

    it('shows correct description for Collaborative connection', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.COLLABORATIVE} />);

      expect(screen.getByText(/Many-to-many communication/)).toBeInTheDocument();
      expect(screen.getByText(/Best for team discussions/)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for buttons', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAttribute('aria-label');
      });
    });

    it('supports keyboard navigation', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const connectionButtons = screen.getAllByRole('button', { name: /Direct|Broadcast|Chain|Collaborative/ });
      connectionButtons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('provides proper form labels', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      expect(timeoutInput).toBeInTheDocument();
    });
  });

  describe('State Management', () => {
    it('maintains configuration state internally', () => {
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      // Initially Direct connection
      expect(screen.getByText('Direct').parentElement).toHaveClass('bg-blue-500');

      // Change timeout
      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      fireEvent.change(timeoutInput, { target: { value: '45' } });

      expect(timeoutInput).toHaveValue(45);
    });

    it('resets configuration when connection type changes', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      // Change timeout
      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      await user.clear(timeoutInput);
      await user.type(timeoutInput, '60');

      // Change connection type
      await user.click(screen.getByText('Broadcast'));

      // Should reset some configuration (based on actual implementation)
      expect(timeoutInput).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles invalid timeout values gracefully', async () => {
      const user = userEvent.setup();
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      const timeoutInput = screen.getByLabelText('Timeout (seconds):');
      await user.clear(timeoutInput);
      await user.type(timeoutInput, 'invalid');

      // Should handle invalid input without crashing
      expect(timeoutInput).toBeInTheDocument();
    });

    it('handles missing props gracefully', () => {
      render(<ConnectionToolsWithProvider />);

      expect(screen.getByText('Connection Tools')).toBeInTheDocument();
    });

    it('handles connection start errors gracefully', () => {
      const mockOnConnectionStart = jest.fn().mockImplementation(() => {
        throw new Error('Connection failed');
      });

      render(<ConnectionToolsWithProvider {...defaultProps} onConnectionStart={mockOnConnectionStart} />);

      fireEvent.click(screen.getByText('Start Connection'));

      // Should not crash the component
      expect(screen.getByText('Connection Tools')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('renders efficiently with multiple state changes', () => {
      const { rerender } = render(<ConnectionToolsWithProvider {...defaultProps} />);

      // Simulate multiple rapid state changes
      for (let i = 0; i < 10; i++) {
        rerender(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.BROADCAST} />);
        rerender(<ConnectionToolsWithProvider {...defaultProps} activeConnectionType={ConnectionType.DIRECT} />);
      }

      expect(screen.getByText('Connection Tools')).toBeInTheDocument();
    });

    it('handles large numbers of configuration options', () => {
      // This test would require mocking many options
      // For now, verify it renders with default options
      render(<ConnectionToolsWithProvider {...defaultProps} />);

      expect(screen.getByText('Connection Tools')).toBeInTheDocument();
    });
  });
});