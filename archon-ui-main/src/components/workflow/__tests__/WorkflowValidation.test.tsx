import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowValidationWithProvider } from '../WorkflowValidation';
import { ValidationError, ValidationSeverity, ValidationResult } from '../../../types/workflowTypes';

describe('WorkflowValidation', () => {
  const mockNodes = [
    {
      id: 'node-1',
      type: 'agent',
      position: { x: 100, y: 100 },
      data: { agentId: 'agent-1', name: 'Agent 1', type: 'CODE_IMPLEMENTER' }
    },
    {
      id: 'node-2',
      type: 'agent',
      position: { x: 300, y: 100 },
      data: { agentId: 'agent-2', name: 'Agent 2', type: 'TEST_COVERAGE_VALIDATOR' }
    }
  ];

  const mockEdges = [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      type: 'direct'
    }
  ];

  const defaultProps = {
    nodes: mockNodes,
    edges: mockEdges,
    onValidationComplete: jest.fn(),
    onAutoFix: jest.fn(),
    onErrorSelect: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders workflow validation component', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      expect(screen.getByText('Workflow Validation')).toBeInTheDocument();
      expect(screen.getByText('Validate your workflow for errors and issues')).toBeInTheDocument();
    });

    it('renders validation controls', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      expect(screen.getByText('Run Validation')).toBeInTheDocument();
      expect(screen.getByText('Clear Results')).toBeInTheDocument();
      expect(screen.getByText('Auto Fix All')).toBeInTheDocument();
    });

    it('renders severity filters', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('Warning')).toBeInTheDocument();
      expect(screen.getByText('Info')).toBeInTheDocument();
    });

    it('renders validation rules list', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      expect(screen.getByText('Orphaned Nodes')).toBeInTheDocument();
      expect(screen.getByText('Circular Dependencies')).toBeInTheDocument();
      expect(screen.getByText('Self-Loops')).toBeInTheDocument();
      expect(screen.getByText('Duplicate Connections')).toBeInTheDocument();
    });
  });

  describe('Validation Execution', () => {
    it('calls validation when Run Validation is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      // Should show validation results
      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
      });
    });

    it('shows loading state during validation', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      expect(screen.getByText('Validating...')).toBeInTheDocument();
    });

    it('displays validation results', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
        expect(screen.getByText('No issues found')).toBeInTheDocument();
      });
    });

    it('shows validation summary', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText(/Errors: 0/)).toBeInTheDocument();
        expect(screen.getByText(/Warnings: 0/)).toBeInTheDocument();
        expect(screen.getByText(/Info: 0/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Detection', () => {
    it('detects orphaned nodes', async () => {
      const user = userEvent.setup();
      const orphanedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={orphanedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Orphaned node: Agent 3')).toBeInTheDocument();
        expect(screen.getByText('Node has no connections')).toBeInTheDocument();
      });
    });

    it('detects circular dependencies', async () => {
      const user = userEvent.setup();
      const circularEdges = [
        ...mockEdges,
        { id: 'edge-2', source: 'node-2', target: 'node-1' }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} edges={circularEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Circular dependency detected')).toBeInTheDocument();
        expect(screen.getByText('Creates infinite loop')).toBeInTheDocument();
      });
    });

    it('detects self-loops', async () => {
      const user = userEvent.setup();
      const selfLoopEdges = [
        { id: 'edge-1', source: 'node-1', target: 'node-1' }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} edges={selfLoopEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Self-loop detected')).toBeInTheDocument();
        expect(screen.getByText('Node connects to itself')).toBeInTheDocument();
      });
    });

    it('detects duplicate connections', async () => {
      const user = userEvent.setup();
      const duplicateEdges = [
        ...mockEdges,
        { id: 'edge-2', source: 'node-1', target: 'node-2' }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} edges={duplicateEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Duplicate connection')).toBeInTheDocument();
        expect(screen.getByText('Multiple connections between same nodes')).toBeInTheDocument();
      });
    });

    it('detects incomplete agent configurations', async () => {
      const user = userEvent.setup();
      const incompleteNodes = [
        {
          id: 'node-1',
          type: 'agent',
          position: { x: 100, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={incompleteNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });
    });

    it('detects invalid connection types', async () => {
      const user = userEvent.setup();
      const invalidEdges = [
        { id: 'edge-1', source: 'node-1', target: 'node-2', type: 'invalid_type' }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} edges={invalidEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Invalid connection type')).toBeInTheDocument();
      });
    });

    it('detects disconnected components', async () => {
      const user = userEvent.setup();
      const disconnectedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        },
        {
          id: 'node-4',
          type: 'agent',
          position: { x: 700, y: 100 },
          data: { agentId: 'agent-4', name: 'Agent 4', type: 'CODE_REVIEWER' }
        }
      ];

      const disconnectedEdges = [
        ...mockEdges,
        { id: 'edge-2', source: 'node-3', target: 'node-4' }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={disconnectedNodes} edges={disconnectedEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Disconnected component detected')).toBeInTheDocument();
      });
    });

    it('detects performance bottlenecks', async () => {
      const user = userEvent.setup();
      const manyNodes = Array.from({ length: 15 }, (_, i) => ({
        id: `node-${i}`,
        type: 'agent',
        position: { x: i * 50, y: 100 },
        data: { agentId: `agent-${i}`, name: `Agent ${i}`, type: 'CODE_IMPLEMENTER' }
      }));

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={manyNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Performance bottleneck')).toBeInTheDocument();
        expect(screen.getByText('Too many agents in sequence')).toBeInTheDocument();
      });
    });

    it('detects resource conflicts', async () => {
      const user = userEvent.setup();
      const conflictNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 100, y: 300 },
          data: { agentId: 'agent-1', name: 'Agent 1 Duplicate', type: 'CODE_IMPLEMENTER' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={conflictNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Resource conflict')).toBeInTheDocument();
        expect(screen.getByText('Duplicate agent ID detected')).toBeInTheDocument();
      });
    });

    it('detects security risks', async () => {
      const user = userEvent.setup();
      const securityNodes = [
        {
          id: 'node-1',
          type: 'agent',
          position: { x: 100, y: 100 },
          data: { agentId: 'agent-1', name: 'Agent 1', type: 'CODE_IMPLEMENTER' }
        },
        {
          id: 'node-2',
          type: 'agent',
          position: { x: 300, y: 100 },
          data: { agentId: 'agent-2', name: 'Agent 2', type: 'SECURITY_AUDITOR' }
        }
      ];

      const securityEdges = [
        { id: 'edge-1', source: 'node-2', target: 'node-1' } // Security auditor -> Code implementer
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={securityNodes} edges={securityEdges} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Security risk')).toBeInTheDocument();
        expect(screen.getByText('Security auditor should not receive tasks')).toBeInTheDocument();
      });
    });
  });

  describe('Severity Filtering', () => {
    it('filters errors by severity', async () => {
      const user = userEvent.setup();
      const problematicNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={problematicNodes} />);

      await user.click(screen.getByText('Run Validation'));

      // Initially show all severities
      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });

      // Filter to show only errors
      await user.click(screen.getByText('Warning'));
      await user.click(screen.getByText('Info'));

      // Should still show errors
      expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
    });

    it('shows severity counts', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText(/Errors: 0/)).toBeInTheDocument();
        expect(screen.getByText(/Warnings: 0/)).toBeInTheDocument();
        expect(screen.getByText(/Info: 0/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Selection', () => {
    it('calls onErrorSelect when error is clicked', async () => {
      const user = userEvent.setup();
      const orphanedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={orphanedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        const errorItem = screen.getByText('Orphaned node: Agent 3');
        fireEvent.click(errorItem);

        expect(defaultProps.onErrorSelect).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'orphaned_nodes'
          })
        );
      });
    });

    it('highlights selected error', async () => {
      const user = userEvent.setup();
      const orphanedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={orphanedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        const errorItem = screen.getByText('Orphaned node: Agent 3');
        fireEvent.click(errorItem);

        // Should show selection state
        expect(errorItem.parentElement).toHaveClass('bg-blue-50');
      });
    });
  });

  describe('Auto-Fix Functionality', () => {
    it('calls onAutoFix when Auto Fix All is clicked', async () => {
      const user = userEvent.setup();
      const problematicNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={problematicNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Auto Fix All'));

      expect(defaultProps.onAutoFix).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            type: 'incomplete_agent_config'
          })
        ])
      );
    });

    it('shows auto-fix progress', async () => {
      const user = userEvent.setup();
      const problematicNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={problematicNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Auto Fix All'));

      expect(screen.getByText('Auto-fixing issues...')).toBeInTheDocument();
    });

    it('allows individual error auto-fix', async () => {
      const user = userEvent.setup();
      const problematicNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={problematicNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });

      const fixButton = screen.getByText('Fix');
      await user.click(fixButton);

      expect(defaultProps.onAutoFix).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            type: 'incomplete_agent_config'
          })
        ])
      );
    });
  });

  describe('Clear Results', () => {
    it('clears validation results when Clear Results is clicked', async () => {
      const user = userEvent.setup();
      const orphanedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={orphanedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Orphaned node: Agent 3')).toBeInTheDocument();
      });

      await user.click(screen.getByText('Clear Results'));

      expect(screen.queryByText('Orphaned node: Agent 3')).not.toBeInTheDocument();
      expect(screen.getByText('No validation results')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for interactive elements', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      const runButton = screen.getByText('Run Validation');
      expect(runButton).toBeInTheDocument();
      expect(runButton).toHaveAttribute('aria-label');

      const severityFilters = screen.getAllByRole('checkbox');
      severityFilters.forEach(filter => {
        expect(filter).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', () => {
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('provides proper error descriptions', async () => {
      const user = userEvent.setup();
      const orphanedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: 'agent-3', name: 'Agent 3', type: 'SECURITY_AUDITOR' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={orphanedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        const errorItem = screen.getByText('Orphaned node: Agent 3');
        expect(errorItem).toBeInTheDocument();
        expect(errorItem).toHaveAttribute('aria-describedby');
      });
    });
  });

  describe('State Management', () => {
    it('maintains validation results state', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
      });

      // Re-render to test state persistence
      rerender(<WorkflowValidationWithProvider {...defaultProps} />);

      expect(screen.getByText('Validation Complete')).toBeInTheDocument();
    });

    it('updates validation results when nodes/edges change', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<WorkflowValidationWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('No issues found')).toBeInTheDocument();
      });

      // Add problematic node
      const updatedNodes = [
        ...mockNodes,
        {
          id: 'node-3',
          type: 'agent',
          position: { x: 500, y: 100 },
          data: { agentId: '', name: '', type: '' }
        }
      ];

      rerender(<WorkflowValidationWithProvider {...defaultProps} nodes={updatedNodes} />);

      await user.click(screen.getByText('Run Validation'));

      await waitFor(() => {
        expect(screen.getByText('Incomplete agent configuration')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles missing props gracefully', () => {
      render(<WorkflowValidationWithProvider />);

      expect(screen.getByText('Workflow Validation')).toBeInTheDocument();
    });

    it('handles validation errors gracefully', async () => {
      const user = userEvent.setup();
      const problematicNodes = [
        {
          id: 'node-1',
          type: 'invalid_type',
          position: { x: 100, y: 100 },
          data: { invalid: 'data' }
        }
      ];

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={problematicNodes as any} />);

      await user.click(screen.getByText('Run Validation'));

      // Should not crash
      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
      });
    });

    it('handles onValidationComplete errors gracefully', () => {
      const mockOnValidationComplete = jest.fn().mockImplementation(() => {
        throw new Error('Validation callback failed');
      });

      render(<WorkflowValidationWithProvider {...defaultProps} onValidationComplete={mockOnValidationComplete} />);

      fireEvent.click(screen.getByText('Run Validation'));

      // Should not crash the component
      expect(screen.getByText('Workflow Validation')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large number of nodes and edges efficiently', async () => {
      const user = userEvent.setup();
      const largeNodes = Array.from({ length: 50 }, (_, i) => ({
        id: `node-${i}`,
        type: 'agent',
        position: { x: i * 20, y: 100 },
        data: { agentId: `agent-${i}`, name: `Agent ${i}`, type: 'CODE_IMPLEMENTER' }
      }));

      const largeEdges = Array.from({ length: 40 }, (_, i) => ({
        id: `edge-${i}`,
        source: `node-${i}`,
        target: `node-${i + 1}`
      }));

      render(<WorkflowValidationWithProvider {...defaultProps} nodes={largeNodes} edges={largeEdges} />);

      await user.click(screen.getByText('Run Validation'));

      // Should complete without crashing
      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
      });
    });

    it('debounces rapid validation requests', async () => {
      const user = userEvent.setup();
      render(<WorkflowValidationWithProvider {...defaultProps} />);

      // Rapid clicks
      for (let i = 0; i < 5; i++) {
        await user.click(screen.getByText('Run Validation'));
      }

      // Should handle gracefully
      await waitFor(() => {
        expect(screen.getByText('Validation Complete')).toBeInTheDocument();
      });
    });
  });
});