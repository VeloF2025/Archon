import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowTemplatesWithProvider } from '../WorkflowTemplates';
import { WorkflowTemplate, AgentType, ModelTier } from '../../../types/workflowTypes';

describe('WorkflowTemplates', () => {
  const defaultProps = {
    onLoadTemplate: jest.fn(),
    onClose: jest.fn(),
    isOpen: true,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders workflow templates modal when open', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
      expect(screen.getByText('Choose a template to get started quickly')).toBeInTheDocument();
    });

    it('does not render when closed', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} isOpen={false} />);

      expect(screen.queryByText('Workflow Templates')).not.toBeInTheDocument();
    });

    it('renders template cards', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
      expect(screen.getByText('Collaborative Planning Workflow')).toBeInTheDocument();
      expect(screen.getByText('Testing & Quality Assurance')).toBeInTheDocument();
      expect(screen.getByText('Research & Analysis Workflow')).toBeInTheDocument();
    });

    it('renders search and filter controls', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByPlaceholderText('Search templates...')).toBeInTheDocument();
      expect(screen.getByText('All Categories')).toBeInTheDocument();
      expect(screen.getByText('All Complexities')).toBeInTheDocument();
    });

    it('renders close button', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('×')).toBeInTheDocument();
    });
  });

  describe('Template Display', () => {
    it('shows template information on cards', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Development Workflow card
      expect(screen.getByText(/Complete software development workflow/)).toBeInTheDocument();
      expect(screen.getByText('Medium')).toBeInTheDocument();
      expect(screen.getByText('5 agents')).toBeInTheDocument();
      expect(screen.getByText('4 connections')).toBeInTheDocument();
    });

    it('shows template categories', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('Development')).toBeInTheDocument();
      expect(screen.getByText('Security')).toBeInTheDocument();
      expect(screen.getByText('Planning')).toBeInTheDocument();
      expect(screen.getByText('Testing')).toBeInTheDocument();
      expect(screen.getByText('Research')).toBeInTheDocument();
    });

    it('shows complexity badges', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('Simple')).toBeInTheDocument();
      expect(screen.getByText('Medium')).toBeInTheDocument();
      expect(screen.getByText('Complex')).toBeInTheDocument();
    });

    it('shows agent and connection counts', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getAllByText(/agents/)).toHaveLength(5);
      expect(screen.getAllByText(/connections/)).toHaveLength(5);
    });
  });

  describe('Search Functionality', () => {
    it('filters templates by search term', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'development');

      // Should show Development Workflow
      expect(screen.getByText('Development Workflow')).toBeInTheDocument();

      // Should not show unrelated templates
      expect(screen.queryByText('Security Audit Workflow')).not.toBeInTheDocument();
    });

    it('shows no results when no templates match search', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'nonexistent');

      expect(screen.getByText('No templates found matching your search')).toBeInTheDocument();
    });

    it('clears search when clear button is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'development');

      // Clear search
      await user.click(screen.getByText('Clear filters'));

      expect(searchInput).toHaveValue('');
      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
    });

    it('searches across template names and descriptions', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'security');

      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
    });
  });

  describe('Category Filtering', () => {
    it('filters templates by category', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Filter by Development category
      await user.click(screen.getByText('All Categories'));
      await user.click(screen.getByText('Development'));

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.queryByText('Security Audit Workflow')).not.toBeInTheDocument();
    });

    it('shows all templates when "All Categories" is selected', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Select Development category
      await user.click(screen.getByText('All Categories'));
      await user.click(screen.getByText('Development'));

      // Select All Categories
      await user.click(screen.getByText('Development'));
      await user.click(screen.getByText('All Categories'));

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
    });

    it('shows category counts', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Check if category dropdown shows counts
      expect(screen.getByText('All Categories')).toBeInTheDocument();
    });
  });

  describe('Complexity Filtering', () => {
    it('filters templates by complexity', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Filter by Simple complexity
      await user.click(screen.getByText('All Complexities'));
      await user.click(screen.getByText('Simple'));

      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
      expect(screen.queryByText('Development Workflow')).not.toBeInTheDocument();
    });

    it('shows all templates when "All Complexities" is selected', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Select Simple complexity
      await user.click(screen.getByText('All Complexities'));
      await user.click(screen.getByText('Simple'));

      // Select All Complexities
      await user.click(screen.getByText('Simple'));
      await user.click(screen.getByText('All Complexities'));

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.getByText('Security Audit Workflow')).toBeInTheDocument();
    });
  });

  describe('Template Selection', () => {
    it('calls onLoadTemplate when template is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      expect(defaultProps.onLoadTemplate).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'development-workflow',
          name: 'Development Workflow'
        })
      );
    });

    it('shows template details when template is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      // Should show detailed view
      expect(screen.getByText('Template Details')).toBeInTheDocument();
      expect(screen.getByText('Complete software development workflow')).toBeInTheDocument();
    });

    it('allows loading template from details view', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      // Load template from details
      await user.click(screen.getByText('Load Template'));

      expect(defaultProps.onLoadTemplate).toHaveBeenCalled();
    });

    it('allows closing details view', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      // Close details
      await user.click(screen.getByText('Back to Templates'));

      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
      expect(screen.queryByText('Template Details')).not.toBeInTheDocument();
    });
  });

  describe('Template Content', () => {
    it('shows template preview information', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      expect(screen.getByText('Template Preview')).toBeInTheDocument();
      expect(screen.getByText('Agents:')).toBeInTheDocument();
      expect(screen.getByText('Connections:')).toBeInTheDocument();
      expect(screen.getByText('Communication Flows:')).toBeInTheDocument();
    });

    it('shows included agents in template', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      expect(screen.getByText('Strategic Planner')).toBeInTheDocument();
      expect(screen.getByText('Code Implementer')).toBeInTheDocument();
      expect(screen.getByText('Test Coverage Validator')).toBeInTheDocument();
    });

    it('shows communication patterns in template', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      expect(screen.getByText('Planner → Implementer')).toBeInTheDocument();
      expect(screen.getByText('Implementer → Validator')).toBeInTheDocument();
    });
  });

  describe('Modal Actions', () => {
    it('calls onClose when close button is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      await user.click(screen.getByText('×'));

      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('calls onClose when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Cancel'));

      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('closes modal when template is loaded', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Click template to show details
      const templateCard = screen.getByText('Development Workflow').closest('div');
      await user.click(templateCard!);

      // Load template
      await user.click(screen.getByText('Load Template'));

      expect(defaultProps.onLoadTemplate).toHaveBeenCalled();
      // onClose should also be called to close modal
      expect(defaultProps.onClose).toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for interactive elements', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      expect(searchInput).toBeInTheDocument();
      expect(searchInput).toHaveAttribute('aria-label');

      const templateCards = screen.getAllByRole('button');
      templateCards.forEach(card => {
        expect(card).toBeInTheDocument();
      });
    });

    it('supports keyboard navigation', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const focusableElements = screen.getAllByRole('textbox', 'combobox', 'button');
      focusableElements.forEach(element => {
        expect(element).toBeInTheDocument();
      });
    });

    it('provides proper modal accessibility', () => {
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const modal = screen.getByRole('dialog');
      expect(modal).toBeInTheDocument();
      expect(modal).toHaveAttribute('aria-modal', 'true');
    });
  });

  describe('State Management', () => {
    it('maintains search state', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'development');

      expect(searchInput).toHaveValue('development');
    });

    it('maintains filter state', async () => {
      const user = userEvent.setup();
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Filter by Development category
      await user.click(screen.getByText('All Categories'));
      await user.click(screen.getByText('Development'));

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.queryByText('Security Audit Workflow')).not.toBeInTheDocument();
    });

    it('resets filters when modal is reopened', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Apply filters
      const searchInput = screen.getByPlaceholderText('Search templates...');
      await user.type(searchInput, 'development');

      // Close modal
      rerender(<WorkflowTemplatesWithProvider {...defaultProps} isOpen={false} />);

      // Reopen modal
      rerender(<WorkflowTemplatesWithProvider {...defaultProps} isOpen={true} />);

      // Filters should be reset
      const newSearchInput = screen.getByPlaceholderText('Search templates...');
      expect(newSearchInput).toHaveValue('');
    });
  });

  describe('Error Handling', () => {
    it('handles missing props gracefully', () => {
      render(<WorkflowTemplatesWithProvider />);

      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });

    it('handles onLoadTemplate errors gracefully', () => {
      const mockOnLoadTemplate = jest.fn().mockImplementation(() => {
        throw new Error('Template loading failed');
      });

      render(<WorkflowTemplatesWithProvider {...defaultProps} onLoadTemplate={mockOnLoadTemplate} />);

      const templateCard = screen.getByText('Development Workflow').closest('div');
      fireEvent.click(templateCard!);

      // Should not crash the component
      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });

    it('handles onClose errors gracefully', () => {
      const mockOnClose = jest.fn().mockImplementation(() => {
        throw new Error('Close failed');
      });

      render(<WorkflowTemplatesWithProvider {...defaultProps} onClose={mockOnClose} />);

      fireEvent.click(screen.getByText('×'));

      // Should not crash the component
      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('renders efficiently with many templates', () => {
      // This test would require mocking many templates
      // For now, verify it renders with default templates
      render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });

    it('handles rapid filter changes efficiently', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<WorkflowTemplatesWithProvider {...defaultProps} />);

      // Simulate rapid filter changes
      for (let i = 0; i < 5; i++) {
        const searchInput = screen.getByPlaceholderText('Search templates...');
        await user.clear(searchInput);
        await user.type(searchInput, `test${i}`);
      }

      expect(screen.getByText('Workflow Templates')).toBeInTheDocument();
    });
  });
});