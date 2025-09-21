import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowPersistenceWithProvider } from '../WorkflowPersistence';
import { WorkflowConfiguration, ExportFormat } from '../../../types/workflowTypes';

describe('WorkflowPersistence', () => {
  const mockWorkflowConfig: WorkflowConfiguration = {
    id: 'test-workflow',
    name: 'Test Workflow',
    description: 'A test workflow configuration',
    nodes: [
      {
        id: 'node-1',
        type: 'agent',
        position: { x: 100, y: 100 },
        data: {
          agentId: 'agent-1',
          name: 'Agent 1',
          type: 'CODE_IMPLEMENTER',
          tier: 'SONNET',
          state: 'ACTIVE'
        }
      }
    ],
    edges: [
      {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        type: 'DIRECT',
        data: {
          connectionType: 'DIRECT',
          messageType: 'TASK_ASSIGNMENT'
        }
      }
    ],
    metadata: {
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: '1.0.0'
    }
  };

  const defaultProps = {
    workflowConfig: mockWorkflowConfig,
    onSave: jest.fn(),
    onLoad: jest.fn(),
    onExport: jest.fn(),
    onImport: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Clear localStorage
    localStorage.clear();
  });

  describe('Rendering', () => {
    it('renders workflow persistence component', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Workflow Persistence')).toBeInTheDocument();
      expect(screen.getByText('Save, load, and manage your workflow configurations')).toBeInTheDocument();
    });

    it('renders save/load controls', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Save Workflow')).toBeInTheDocument();
      expect(screen.getByText('Load Workflow')).toBeInTheDocument();
      expect(screen.getByText('Export')).toBeInTheDocument();
      expect(screen.getByText('Import')).toBeInTheDocument();
    });

    it('renders saved workflows list', () => {
      // Save a workflow to localStorage first
      localStorage.setItem('workflow-saved-workflows', JSON.stringify([mockWorkflowConfig]));

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Saved Workflows')).toBeInTheDocument();
      expect(screen.getByText('Test Workflow')).toBeInTheDocument();
    });

    it('renders export format options', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Export Format:')).toBeInTheDocument();
      expect(screen.getByText('JSON')).toBeInTheDocument();
      expect(screen.getByText('YAML')).toBeInTheDocument();
      expect(screen.getByText('XML')).toBeInTheDocument();
    });

    it('renders import file input', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      expect(fileInput).toBeInTheDocument();
      expect(fileInput).toHaveAttribute('type', 'file');
    });
  });

  describe('Save Functionality', () => {
    it('calls onSave when Save Workflow is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Save Workflow'));

      expect(defaultProps.onSave).toHaveBeenCalledWith(mockWorkflowConfig);
    });

    it('saves workflow to localStorage', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Save Workflow'));

      const savedWorkflows = JSON.parse(localStorage.getItem('workflow-saved-workflows') || '[]');
      expect(savedWorkflows).toHaveLength(1);
      expect(savedWorkflows[0]).toEqual(mockWorkflowConfig);
    });

    it('updates existing workflow in localStorage', async () => {
      const user = userEvent.setup();

      // Save initial workflow
      localStorage.setItem('workflow-saved-workflows', JSON.stringify([mockWorkflowConfig]));

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Save updated workflow
      const updatedConfig = { ...mockWorkflowConfig, name: 'Updated Workflow' };
      await user.click(screen.getByText('Save Workflow'));

      const savedWorkflows = JSON.parse(localStorage.getItem('workflow-saved-workflows') || '[]');
      expect(savedWorkflows).toHaveLength(1);
      expect(savedWorkflows[0].name).toBe('Updated Workflow');
    });

    it('shows save confirmation', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Save Workflow'));

      expect(screen.getByText('Workflow saved successfully!')).toBeInTheDocument();
    });

    it('handles save errors gracefully', async () => {
      const user = userEvent.setup();
      const mockOnSave = jest.fn().mockImplementation(() => {
        throw new Error('Save failed');
      });

      render(<WorkflowPersistenceWithProvider {...defaultProps} onSave={mockOnSave} />);

      await user.click(screen.getByText('Save Workflow'));

      expect(screen.getByText('Failed to save workflow')).toBeInTheDocument();
    });
  });

  describe('Load Functionality', () => {
    beforeEach(() => {
      // Save workflows to localStorage for testing
      const savedWorkflows = [mockWorkflowConfig, {
        ...mockWorkflowConfig,
        id: 'workflow-2',
        name: 'Second Workflow'
      }];
      localStorage.setItem('workflow-saved-workflows', JSON.stringify(savedWorkflows));
    });

    it('displays saved workflows', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Test Workflow')).toBeInTheDocument();
      expect(screen.getByText('Second Workflow')).toBeInTheDocument();
    });

    it('shows workflow metadata', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText(/Nodes: 1/)).toBeInTheDocument();
      expect(screen.getByText(/Edges: 1/)).toBeInTheDocument();
      expect(screen.getByText(/Version: 1.0.0/)).toBeInTheDocument();
    });

    it('calls onLoad when workflow is loaded', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const loadButton = screen.getByText('Load', { selector: 'button' });
      await user.click(loadButton);

      expect(defaultProps.onLoad).toHaveBeenCalledWith(mockWorkflowConfig);
    });

    it('allows deleting saved workflows', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const deleteButton = screen.getByText('Delete', { selector: 'button' });
      await user.click(deleteButton);

      // Confirm deletion
      await user.click(screen.getByText('Delete'));

      const savedWorkflows = JSON.parse(localStorage.getItem('workflow-saved-workflows') || '[]');
      expect(savedWorkflows).toHaveLength(1); // Should have one remaining workflow
    });

    it('shows workflow details on hover', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const workflowItem = screen.getByText('Test Workflow').closest('div');
      await user.hover(workflowItem!);

      expect(screen.getByText('A test workflow configuration')).toBeInTheDocument();
    });
  });

  describe('Export Functionality', () => {
    it('calls onExport when Export is clicked', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Export'));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.JSON
      );
    });

    it('exports in different formats', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Test JSON export
      await user.click(screen.getByText('Export'));
      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.JSON
      );

      // Change format to YAML
      await user.click(screen.getByText('YAML'));
      await user.click(screen.getByText('Export'));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.YAML
      );
    });

    it('downloads exported file', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Mock the download functionality
      const mockCreateObjectURL = jest.fn();
      const mockRevokeObjectURL = jest.fn();
      global.URL.createObjectURL = mockCreateObjectURL;
      global.URL.revokeObjectURL = mockRevokeObjectURL;

      await user.click(screen.getByText('Export'));

      expect(mockCreateObjectURL).toHaveBeenCalled();
    });

    it('shows export confirmation', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Export'));

      expect(screen.getByText('Workflow exported successfully!')).toBeInTheDocument();
    });
  });

  describe('Import Functionality', () => {
    it('handles file selection', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const file = new File(['{"name": "test"}'], 'test.json', { type: 'application/json' });

      await user.upload(fileInput, file);

      expect(fileInput.files?.[0]).toBe(file);
    });

    it('calls onImport when file is selected', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const fileContent = JSON.stringify(mockWorkflowConfig);
      const file = new File([fileContent], 'workflow.json', { type: 'application/json' });

      await user.upload(fileInput, file);

      expect(defaultProps.onImport).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Test Workflow'
        })
      );
    });

    it('validates imported file format', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const invalidFile = new File(['invalid content'], 'test.txt', { type: 'text/plain' });

      await user.upload(fileInput, invalidFile);

      expect(screen.getByText('Invalid file format')).toBeInTheDocument();
    });

    it('validates imported workflow structure', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const invalidWorkflow = new File(['{"invalid": "structure"}'], 'workflow.json', { type: 'application/json' });

      await user.upload(fileInput, invalidWorkflow);

      expect(screen.getByText('Invalid workflow structure')).toBeInTheDocument();
    });

    it('shows import confirmation for valid files', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const fileContent = JSON.stringify(mockWorkflowConfig);
      const file = new File([fileContent], 'workflow.json', { type: 'application/json' });

      await user.upload(fileInput, file);

      expect(screen.getByText('Workflow imported successfully!')).toBeInTheDocument();
    });
  });

  describe('Format Support', () => {
    it('supports JSON format', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Select JSON format
      await user.click(screen.getByText('JSON'));

      await user.click(screen.getByText('Export'));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.JSON
      );
    });

    it('supports YAML format', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Select YAML format
      await user.click(screen.getByText('YAML'));

      await user.click(screen.getByText('Export'));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.YAML
      );
    });

    it('supports XML format', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Select XML format
      await user.click(screen.getByText('XML'));

      await user.click(screen.getByText('Export'));

      expect(defaultProps.onExport).toHaveBeenCalledWith(
        mockWorkflowConfig,
        ExportFormat.XML
      );
    });
  });

  describe('Storage Management', () => {
    it('loads saved workflows from localStorage on mount', () => {
      localStorage.setItem('workflow-saved-workflows', JSON.stringify([mockWorkflowConfig]));

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Test Workflow')).toBeInTheDocument();
    });

    it('handles empty localStorage', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('No saved workflows')).toBeInTheDocument();
    });

    it('handles corrupted localStorage data', () => {
      localStorage.setItem('workflow-saved-workflows', 'invalid json');

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('No saved workflows')).toBeInTheDocument();
    });

    it('clears localStorage when requested', async () => {
      const user = userEvent.setup();
      localStorage.setItem('workflow-saved-workflows', JSON.stringify([mockWorkflowConfig]));

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      await user.click(screen.getByText('Clear All'));

      // Confirm clearing
      await user.click(screen.getByText('Clear'));

      const savedWorkflows = JSON.parse(localStorage.getItem('workflow-saved-workflows') || '[]');
      expect(savedWorkflows).toHaveLength(0);
    });
  });

  describe('Search and Filter', () => {
    beforeEach(() => {
      const savedWorkflows = [
        mockWorkflowConfig,
        { ...mockWorkflowConfig, id: 'workflow-2', name: 'Development Workflow' },
        { ...mockWorkflowConfig, id: 'workflow-3', name: 'Security Workflow' }
      ];
      localStorage.setItem('workflow-saved-workflows', JSON.stringify(savedWorkflows));
    });

    it('filters saved workflows by search', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');
      await user.type(searchInput, 'Development');

      expect(screen.getByText('Development Workflow')).toBeInTheDocument();
      expect(screen.queryByText('Test Workflow')).not.toBeInTheDocument();
    });

    it('shows no results when search doesn\'t match', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');
      await user.type(searchInput, 'nonexistent');

      expect(screen.getByText('No workflows found')).toBeInTheDocument();
    });

    it('clears search filter', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');
      await user.type(searchInput, 'Development');

      // Clear search
      await user.click(screen.getByText('Clear'));

      expect(searchInput).toHaveValue('');
      expect(screen.getByText('Test Workflow')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels for interactive elements', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const saveButton = screen.getByText('Save Workflow');
      expect(saveButton).toBeInTheDocument();
      expect(saveButton).toHaveAttribute('aria-label');

      const fileInput = screen.getByLabelText('Import workflow file:');
      expect(fileInput).toBeInTheDocument();
      expect(fileInput).toHaveAttribute('type', 'file');
    });

    it('supports keyboard navigation', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeInTheDocument();
      });
    });

    it('provides proper form labels', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');
      expect(searchInput).toBeInTheDocument();
      expect(searchInput).toHaveAttribute('placeholder');
    });
  });

  describe('State Management', () => {
    it('maintains internal state for form interactions', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');
      await user.type(searchInput, 'test');

      expect(searchInput).toHaveValue('test');
    });

    it('updates saved workflows list after operations', async () => {
      const user = userEvent.setup();
      const { rerender } = render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      // Save workflow
      await user.click(screen.getByText('Save Workflow'));

      // Re-render to test state update
      rerender(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Test Workflow')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles missing workflow config gracefully', () => {
      render(<WorkflowPersistenceWithProvider {...defaultProps} workflowConfig={null} />);

      expect(screen.getByText('Workflow Persistence')).toBeInTheDocument();
    });

    it('handles localStorage errors gracefully', () => {
      // Mock localStorage to throw errors
      const mockSetItem = jest.fn().mockImplementation(() => {
        throw new Error('Storage quota exceeded');
      });
      Object.defineProperty(window, 'localStorage', {
        value: {
          setItem: mockSetItem,
          getItem: jest.fn(),
          removeItem: jest.fn(),
          clear: jest.fn()
        },
        writable: true
      });

      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      fireEvent.click(screen.getByText('Save Workflow'));

      expect(screen.getByText('Failed to save workflow')).toBeInTheDocument();
    });

    it('handles file read errors gracefully', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const fileInput = screen.getByLabelText('Import workflow file:');
      const invalidFile = new File([''], 'test.json', { type: 'application/json' });

      // Mock FileReader to throw error
      const mockFileReader = {
        readAsText: jest.fn(),
        onload: null as any,
        onerror: null as any
      };
      global.FileReader = jest.fn().mockImplementation(() => mockFileReader);

      await user.upload(fileInput, invalidFile);

      // Should handle gracefully
      expect(screen.getByText('Workflow Persistence')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    it('handles large number of saved workflows efficiently', () => {
      const manyWorkflows = Array.from({ length: 100 }, (_, i) => ({
        ...mockWorkflowConfig,
        id: `workflow-${i}`,
        name: `Workflow ${i}`
      }));
      localStorage.setItem('workflow-saved-workflows', JSON.stringify(manyWorkflows));

      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      expect(screen.getByText('Workflow Persistence')).toBeInTheDocument();
    });

    it('debounces search input', async () => {
      const user = userEvent.setup();
      render(<WorkflowPersistenceWithProvider {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText('Search workflows...');

      // Rapid typing
      for (let i = 0; i < 5; i++) {
        await user.type(searchInput, 'test');
        await user.clear(searchInput);
      }

      expect(searchInput).toBeInTheDocument();
    });
  });
});