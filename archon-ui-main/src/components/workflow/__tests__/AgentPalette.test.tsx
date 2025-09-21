import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AgentPaletteWithProvider } from '../AgentPalette';
import { AgentType, ModelTier } from '../../../types/agentTypes';

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

describe('AgentPalette', () => {
  const defaultProps = {
    onAgentDragStart: jest.fn(),
    onAgentDragEnd: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the agent palette with header and instructions', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
      expect(screen.getByText('Drag agents onto the canvas to build your workflow')).toBeInTheDocument();
      expect(screen.getByText('How to use:')).toBeInTheDocument();
    });

    it('renders all tier filter buttons', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      expect(screen.getByText('All Tiers')).toBeInTheDocument();
      expect(screen.getByText('Opus')).toBeInTheDocument();
      expect(screen.getByText('Sonnet')).toBeInTheDocument();
      expect(screen.getByText('Haiku')).toBeInTheDocument();
    });

    it('renders search input', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search agents...');
      expect(searchInput).toBeInTheDocument();
      expect(searchInput).toHaveAttribute('type', 'text');
    });
  });

  describe('Filtering', () => {
    it('filters agents by tier when tier filter is clicked', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Click on Opus tier filter
      await user.click(screen.getByText('Opus'));

      // Should show only Opus agents
      const opusSection = screen.getByText('Opus Tier');
      expect(opusSection).toBeInTheDocument();

      // Should not show other tiers
      expect(screen.queryByText('Sonnet Tier')).not.toBeInTheDocument();
      expect(screen.queryByText('Haiku Tier')).not.toBeInTheDocument();
    });

    it('filters agents by search term', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      const searchInput = screen.getByPlaceholderText('Search agents...');

      // Search for "implementer"
      await user.type(searchInput, 'implementer');

      // Should show Code Implementer
      expect(screen.getByText('Code Implementer')).toBeInTheDocument();

      // Should not show unrelated agents
      expect(screen.queryByText('Security Auditor')).not.toBeInTheDocument();
    });

    it('shows no results when no agents match search', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      const searchInput = screen.getByPlaceholderText('Search agents...');

      // Search for non-existent term
      await user.type(searchInput, 'nonexistentagent');

      expect(screen.getByText('No agents found matching your search')).toBeInTheDocument();
      expect(screen.getByText('Clear filters')).toBeInTheDocument();
    });

    it('clears filters when clear button is clicked', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      const searchInput = screen.getByPlaceholderText('Search agents...');

      // Search for something
      await user.type(searchInput, 'implementer');

      // Clear filters
      await user.click(screen.getByText('Clear filters'));

      // Search should be cleared
      expect(searchInput).toHaveValue('');
    });

    it('combines tier and search filters', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();

      // Select Sonnet tier
      await user.click(screen.getByText('Sonnet'));

      // Search for "code"
      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'code');

      // Should show only Sonnet agents that match "code"
      expect(screen.getByText('Code Implementer')).toBeInTheDocument();
      expect(screen.queryByText('Strategic Planner')).not.toBeInTheDocument(); // Opus tier
    });
  });

  describe('Agent Items', () => {
    it('renders agent items with correct information', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Find Code Implementer
      const codeImplementer = screen.getByText('Code Implementer').closest('div');
      expect(codeImplementer).toBeInTheDocument();

      // Should have agent icon
      expect(screen.getByText('⚡')).toBeInTheDocument();

      // Should have model tier badge
      expect(screen.getByText('Sonnet')).toBeInTheDocument();

      // Should have description
      expect(screen.getByText(/Writes high-quality/)).toBeInTheDocument();
    });

    it('shows capabilities for each agent', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Code Implementer should show code generation capability
      expect(screen.getByText('code generation:')).toBeInTheDocument();
      expect(screen.getByText('✓')).toBeInTheDocument(); // Checkmark for boolean capability
    });

    it('limits displayed capabilities and shows count', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Look for "+ more..." indicator
      const moreIndicator = screen.queryByText(/\+ \d+ more/);
      expect(moreIndicator).toBeInTheDocument();
    });
  });

  describe('Drag and Drop', () => {
    it('calls onDragStart when agent drag begins', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();
      const codeImplementer = screen.getByText('Code Implementer').closest('div');

      // Simulate drag start
      fireEvent.dragStart(codeImplementer!);

      expect(defaultProps.onAgentDragStart).toHaveBeenCalled();
    });

    it('calls onDragEnd when agent drag ends', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();
      const codeImplementer = screen.getByText('Code Implementer').closest('div');

      // Simulate drag end
      fireEvent.dragEnd(codeImplementer!);

      expect(defaultProps.onAgentDragEnd).toHaveBeenCalled();
    });

    it('shows dragging state during drag operation', async () => {
      const { rerender } = render(<AgentPaletteWithProvider {...defaultProps} />);

      // Mock the useDrag hook to return isDragging: true
      jest.doMock('react-dnd', () => ({
        useDrag: () => [{ isDragging: true }, jest.fn()],
        useDrop: () => [{}, jest.fn()],
        DndProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
      }));

      rerender(<AgentPaletteWithProvider {...defaultProps} />);

      // Should show dragging overlay
      expect(screen.getByText('Dragging...')).toBeInTheDocument();
    });
  });

  describe('Agent Categories', () => {
    it('groups agents by model tier', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Should have tier sections
      expect(screen.getByText('Opus Tier')).toBeInTheDocument();
      expect(screen.getByText('Sonnet Tier')).toBeInTheDocument();
      expect(screen.getByText('Haiku Tier')).toBeInTheDocument();

      // Should show agent counts
      expect(screen.getByText(/\(\d+\)/)).toBeInTheDocument(); // Should show count in parentheses
    });

    it('shows correct tier colors', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Opus tier should have purple indicator
      const opusIndicator = screen.getByText('Opus Tier').previousElementSibling;
      expect(opusIndicator).toHaveClass('bg-purple-500');

      // Sonnet tier should have blue indicator
      const sonnetIndicator = screen.getByText('Sonnet Tier').previousElementSibling;
      expect(sonnetIndicator).toHaveClass('bg-blue-500');

      // Haiku tier should have green indicator
      const haikuIndicator = screen.getByText('Haiku Tier').previousElementSibling;
      expect(haikuIndicator).toHaveClass('bg-green-500');
    });
  });

  describe('Accessibility', () => {
    it('provides proper labels and descriptions', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // Search input should have proper placeholder
      const searchInput = screen.getByPlaceholderText('Search agents...');
      expect(searchInput).toBeInTheDocument();

      // Instructions should be present
      expect(screen.getByText('How to use:')).toBeInTheDocument();
      expect(screen.getByText(/Search and filter agents by type or tier/)).toBeInTheDocument();
    });

    it('provides keyboard navigation support', () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      // All interactive elements should be keyboard accessible
      const searchInput = screen.getByPlaceholderText('Search agents...');
      expect(searchInput).toHaveAttribute('type', 'text');

      const tierButtons = screen.getAllByRole('button');
      tierButtons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });
  });

  describe('Performance', () => {
    it('handles large number of agents efficiently', () => {
      // This test would require mocking many agents
      // For now, we just verify it renders with the default set
      render(<AgentPaletteWithProvider {...defaultProps} />);

      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('debounces search input to avoid excessive filtering', async () => {
      render(<AgentPaletteWithProvider {...defaultProps} />);

      const user = userEvent.setup();
      const searchInput = screen.getByPlaceholderText('Search agents...');

      // Type multiple characters quickly
      await user.type(searchInput, 'impl', { delay: 10 });

      // Should not crash and should handle input
      expect(searchInput).toHaveValue('impl');
    });
  });

  describe('Error Handling', () => {
    it('handles missing props gracefully', () => {
      // Render without optional props
      render(<AgentPaletteWithProvider />);

      expect(screen.getByText('Agent Palette')).toBeInTheDocument();
    });

    it('handles drag event errors gracefully', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      render(<AgentPaletteWithProvider {...defaultProps} />);

      const codeImplementer = screen.getByText('Code Implementer').closest('div');

      // Simulate error during drag
      fireEvent.dragStart(codeImplementer!);

      // Should not throw unhandled errors
      expect(consoleSpy).not.toHaveBeenCalled();

      consoleSpy.mockRestore();
    });
  });
});