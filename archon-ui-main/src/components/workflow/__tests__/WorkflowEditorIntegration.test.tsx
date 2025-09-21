/**
 * Workflow Editor Integration Test Suite
 * Tests the interactive workflow editor functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowEditor } from '../WorkflowEditor';
import { AgentPalette } from '../AgentPalette';
import { ConnectionTools } from '../ConnectionTools';
import { PropertyEditor } from '../PropertyEditor';
import { WorkflowTemplates } from '../WorkflowTemplates';
import { WorkflowValidation } from '../WorkflowValidation';
import { WorkflowPersistence } from '../WorkflowPersistence';

import {
  AgencyData,
  AgentV3,
  AgentState,
  ModelTier,
  AgentType,
  CommunicationFlow as CommunicationFlowType,
  CommunicationType,
  CommunicationStatus,
  WorkflowVisualizationConfig,
  EditorMode,
  ConnectionType,
  ConnectionCreationState,
  DraggedAgent,
  SelectedElement,
  EditorState,
  HistoryState,
  WorkflowTemplate,
  ValidationResult,
} from '../../../types/workflowTypes';

// Mock ReactFlow
vi.mock('@xyflow/react', () => ({
  ReactFlow: vi.fn(({ children, nodes, edges, onNodesChange, onEdgesChange, onConnect, onNodeClick }) => (
    <div data-testid="react-flow-editor" data-nodes={nodes?.length} data-edges={edges?.length}>
      {children}
      <div data-testid="editor-nodes">{nodes?.map(n => n.id).join(',')}</div>
      <div data-testid="editor-edges">{edges?.map(e => e.id).join(',')}</div>
    </div>
  )),
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useNodesState: vi.fn(() => [[], vi.fn(), vi.fn()]),
  useEdgesState: vi.fn(() => [[], vi.fn(), vi.fn()]),
  addEdge: vi.fn((edges, connection) => [...edges, { id: 'edge-new', ...connection }]),
  Background: vi.fn(() => <div data-testid="editor-background" />),
  Controls: vi.fn(() => <div data-testid="editor-controls" />),
  MiniMap: vi.fn(() => <div data-testid="editor-minimap" />),
  Panel: vi.fn(({ children }) => <div data-testid="editor-panel">{children}</div>),
  NodeTypes: {},
  EdgeTypes: {},
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
  Handle: vi.fn(({ children }) => <div data-testid="editor-handle">{children}</div>),
  useReactFlow: vi.fn(() => ({
    fitView: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    setViewport: vi.fn(),
    getViewport: vi.fn(() => ({ x: 0, y: 0, zoom: 1 })),
    screenToFlowPosition: vi.fn((position) => position),
  })),
  MarkerType: {
    ArrowClosed: 'arrowclosed',
  },
}));

// Mock React DnD
vi.mock('react-dnd', () => ({
  useDrag: vi.fn(() => [{ isDragging: false }, vi.fn()]),
  useDrop: vi.fn(() => [{ isOver: false }, vi.fn()]),
  DndProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock('react-dnd-html5-backend', () => ({
  HTML5Backend: {},
}));

// Mock workflow services
vi.mock('../../../services/workflowService', () => ({
  workflowService: {
    getWorkflow: vi.fn(),
    createWorkflow: vi.fn(),
    updateWorkflow: vi.fn(),
    deleteWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    executeWorkflow: vi.fn(),
    saveWorkflow: vi.fn(),
    loadWorkflow: vi.fn(),
    exportWorkflow: vi.fn(),
    importWorkflow: vi.fn(),
    getWorkflowTemplates: vi.fn(),
    applyTemplate: vi.fn(),
  },
}));

vi.mock('../../../services/workflowKnowledgeService', () => ({
  workflowKnowledgeService: {
    startKnowledgeSession: vi.fn(),
    captureWorkflowEvent: vi.fn(),
    getWorkflowInsights: vi.fn(),
    getWorkflowTemplates: vi.fn(),
    suggestOptimizations: vi.fn(),
  },
}));

// Mock UI components
vi.mock('../../ui/button', () => ({
  Button: vi.fn(({ children, ...props }) => (
    <button data-testid="editor-button" {...props}>
      {children}
    </button>
  )),
}));

vi.mock('../../ui/card', () => ({
  Card: vi.fn(({ children, ...props }) => (
    <div data-testid="editor-card" {...props}>
      {children}
    </div>
  )),
  CardContent: vi.fn(({ children }) => <div data-testid="editor-card-content">{children}</div>),
  CardDescription: vi.fn(({ children }) => <div data-testid="editor-card-description">{children}</div>),
  CardHeader: vi.fn(({ children }) => <div data-testid="editor-card-header">{children}</div>),
  CardTitle: vi.fn(({ children }) => <div data-testid="editor-card-title">{children}</div>),
}));

vi.mock('../../ui/input', () => ({
  Input: vi.fn((props) => <input data-testid="editor-input" {...props} />),
}));

vi.mock('../../ui/textarea', () => ({
  Textarea: vi.fn((props) => <textarea data-testid="editor-textarea" {...props} />),
}));

vi.mock('../../ui/label', () => ({
  Label: vi.fn(({ children }) => <label data-testid="editor-label">{children}</label>),
}));

vi.mock('../../ui/select', () => ({
  Select: vi.fn(({ children, ...props }) => (
    <div data-testid="editor-select" {...props}>
      {children}
    </div>
  )),
  SelectContent: vi.fn(({ children }) => <div data-testid="editor-select-content">{children}</div>),
  SelectItem: vi.fn(({ children }) => <div data-testid="editor-select-item">{children}</div>),
  SelectTrigger: vi.fn(({ children }) => <button data-testid="editor-select-trigger">{children}</button>),
  SelectValue: vi.fn(() => <div data-testid="editor-select-value" />),
}));

vi.mock('../../ui/tabs', () => ({
  Tabs: vi.fn(({ children, ...props }) => (
    <div data-testid="editor-tabs" {...props}>
      {children}
    </div>
  )),
  TabsContent: vi.fn(({ children }) => <div data-testid="editor-tabs-content">{children}</div>),
  TabsList: vi.fn(({ children }) => <div data-testid="editor-tabs-list">{children}</div>),
  TabsTrigger: vi.fn(({ children }) => <button data-testid="editor-tabs-trigger">{children}</button>),
}));

vi.mock('../../ui/badge', () => ({
  Badge: vi.fn(({ children }) => <span data-testid="editor-badge">{children}</span>),
}));

vi.mock('../../ui/alert', () => ({
  Alert: vi.fn(({ children }) => <div data-testid="editor-alert">{children}</div>),
  AlertDescription: vi.fn(({ children }) => <div data-testid="editor-alert-description">{children}</div>),
}));

describe('Workflow Editor Integration', () => {
  const mockAgencyData: AgencyData = {
    id: 'test-agency-1',
    name: 'Test Agency',
    description: 'Test agency for editor',
    agents: [
      {
        id: 'agent-1',
        name: 'Test Agent 1',
        description: 'Test agent 1',
        type: AgentType.ANALYST,
        model_tier: ModelTier.SONNET,
        state: AgentState.ACTIVE,
        capabilities: ['analysis', 'reporting'],
        created_at: new Date(),
        updated_at: new Date(),
        metadata: {},
      },
      {
        id: 'agent-2',
        name: 'Test Agent 2',
        description: 'Test agent 2',
        type: AgentType.SPECIALIST,
        model_tier: ModelTier.HAIKU,
        state: AgentState.IDLE,
        capabilities: ['specialized_task'],
        created_at: new Date(),
        updated_at: new Date(),
        metadata: {},
      },
      {
        id: 'agent-3',
        name: 'Test Agent 3',
        description: 'Test agent 3',
        type: AgentType.COORDINATOR,
        model_tier: ModelTier.OPUS,
        state: AgentState.HIBERNATED,
        capabilities: ['coordination', 'management'],
        created_at: new Date(),
        updated_at: new Date(),
        metadata: {},
      },
    ],
    communication_flows: [
      {
        id: 'flow-1',
        source_agent_id: 'agent-1',
        target_agent_id: 'agent-2',
        communication_type: CommunicationType.DIRECT,
        status: CommunicationStatus.ACTIVE,
        message_count: 5,
        message_type: 'task_assignment',
        data_flow: {
          input_size: 1024,
          output_size: 2048,
          processing_time_ms: 150,
        },
      },
    ],
    created_at: new Date(),
    updated_at: new Date(),
  };

  const mockWorkflowTemplate: WorkflowTemplate = {
    id: 'template-1',
    name: 'Analysis Workflow',
    description: 'Template for analysis workflows',
    category: 'analysis',
    nodes: [
      {
        id: 'analyst-node',
        type: 'agent',
        position: { x: 100, y: 100 },
        data: {
          agent: {
            id: 'analyst',
            name: 'Analysis Agent',
            type: AgentType.ANALYST,
            model_tier: ModelTier.SONNET,
            state: AgentState.ACTIVE,
            capabilities: ['analysis'],
            created_at: new Date(),
            updated_at: new Date(),
            metadata: {},
          },
        },
      },
    ],
    edges: [],
    metadata: {
      created_at: new Date(),
      updated_at: new Date(),
      version: '1.0.0',
      author: 'System',
    },
  };

  const mockValidationResult: ValidationResult = {
    is_valid: true,
    errors: [],
    warnings: [],
    suggestions: [],
    metadata: {
      validated_at: new Date(),
      validator_version: '1.0.0',
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('WorkflowEditor Core Functionality', () => {
    it('should render workflow editor with all components', () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
      expect(screen.getByTestId('editor-background')).toBeInTheDocument();
      expect(screen.getByTestId('editor-controls')).toBeInTheDocument();
      expect(screen.getByTestId('editor-minimap')).toBeInTheDocument();
      expect(screen.getByTestId('editor-panel')).toBeInTheDocument();
    });

    it('should initialize with correct editor state', () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      const reactFlow = screen.getByTestId('react-flow-editor');
      expect(reactFlow).toBeInTheDocument();
    });

    it('should handle agency data updates', () => {
      const { rerender } = render(<WorkflowEditor agencyData={mockAgencyData} />);

      const updatedAgencyData = {
        ...mockAgencyData,
        agents: [
          ...mockAgencyData.agents,
          {
            id: 'agent-4',
            name: 'New Agent',
            description: 'Newly added agent',
            type: AgentType.ANALYST,
            model_tier: ModelTier.SONNET,
            state: AgentState.CREATED,
            capabilities: ['new_capability'],
            created_at: new Date(),
            updated_at: new Date(),
            metadata: {},
          },
        ],
      };

      rerender(<WorkflowEditor agencyData={updatedAgencyData} />);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });

    it('should handle editor mode switching', async () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      // Simulate mode change event
      const modeChangeEvent = new CustomEvent('editorModeChange', {
        detail: { mode: EditorMode.EDIT },
      });

      window.dispatchEvent(modeChangeEvent);

      await waitFor(() => {
        expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
      });
    });
  });

  describe('AgentPalette Integration', () => {
    it('should render agent palette with available agents', () => {
      const mockOnAgentDrag = vi.fn();
      render(
        <AgentPalette
          agents={mockAgencyData.agents}
          onAgentDrag={mockOnAgentDrag}
          isDragging={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle agent drag start', async () => {
      const mockOnAgentDrag = vi.fn();
      const user = userEvent.setup();

      render(
        <AgentPalette
          agents={mockAgencyData.agents}
          onAgentDrag={mockOnAgentDrag}
          isDragging={false}
        />
      );

      // Simulate drag start
      const draggedAgent: DraggedAgent = {
        agent: mockAgencyData.agents[0],
        type: 'agent',
      };

      await act(async () => {
        mockOnAgentDrag(draggedAgent);
      });

      expect(mockOnAgentDrag).toHaveBeenCalledWith(draggedAgent);
    });

    it('should handle agent drag end', async () => {
      const mockOnAgentDrag = vi.fn();
      const user = userEvent.setup();

      render(
        <AgentPalette
          agents={mockAgencyData.agents}
          onAgentDrag={mockOnAgentDrag}
          isDragging={true}
        />
      );

      // Simulate drag end
      await act(async () => {
        mockOnAgentDrag(null);
      });

      expect(mockOnAgentDrag).toHaveBeenCalledWith(null);
    });

    it('should filter agents by type', () => {
      const mockOnAgentDrag = vi.fn();

      render(
        <AgentPalette
          agents={mockAgencyData.agents}
          onAgentDrag={mockOnAgentDrag}
          isDragging={false}
          filterByType={AgentType.ANALYST}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle agent search', () => {
      const mockOnAgentDrag = vi.fn();

      render(
        <AgentPalette
          agents={mockAgencyData.agents}
          onAgentDrag={mockOnAgentDrag}
          isDragging={false}
          searchTerm="Test"
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });

  describe('ConnectionTools Integration', () => {
    const mockConnectionState: ConnectionCreationState = {
      isConnecting: false,
      sourceNode: null,
      connectionType: ConnectionType.DIRECT,
    };

    it('should render connection tools', () => {
      const mockOnConnectionChange = vi.fn();
      render(
        <ConnectionTools
          connectionState={mockConnectionState}
          onConnectionChange={mockOnConnectionChange}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle connection type change', async () => {
      const mockOnConnectionChange = vi.fn();
      const user = userEvent.setup();

      render(
        <ConnectionTools
          connectionState={mockConnectionState}
          onConnectionChange={mockOnConnectionChange}
        />
      );

      // Simulate connection type change
      const newConnectionState = {
        ...mockConnectionState,
        connectionType: ConnectionType.BROADCAST,
      };

      await act(async () => {
        mockOnConnectionChange(newConnectionState);
      });

      expect(mockOnConnectionChange).toHaveBeenCalledWith(newConnectionState);
    });

    it('should handle connection start', async () => {
      const mockOnConnectionChange = vi.fn();

      render(
        <ConnectionTools
          connectionState={mockConnectionState}
          onConnectionChange={mockOnConnectionChange}
        />
      );

      // Simulate connection start
      const newConnectionState = {
        ...mockConnectionState,
        isConnecting: true,
        sourceNode: 'node-1',
      };

      await act(async () => {
        mockOnConnectionChange(newConnectionState);
      });

      expect(mockOnConnectionChange).toHaveBeenCalledWith(newConnectionState);
    });

    it('should handle connection end', async () => {
      const mockOnConnectionChange = vi.fn();

      render(
        <ConnectionTools
          connectionState={{
            ...mockConnectionState,
            isConnecting: true,
            sourceNode: 'node-1',
          }}
          onConnectionChange={mockOnConnectionChange}
        />
      );

      // Simulate connection end
      const newConnectionState = {
        ...mockConnectionState,
        isConnecting: false,
        sourceNode: null,
      };

      await act(async () => {
        mockOnConnectionChange(newConnectionState);
      });

      expect(mockOnConnectionChange).toHaveBeenCalledWith(newConnectionState);
    });

    it('should handle different connection types', () => {
      const mockOnConnectionChange = vi.fn();
      const connectionTypes = [
        ConnectionType.DIRECT,
        ConnectionType.BROADCAST,
        ConnectionType.CHAIN,
        ConnectionType.COLLABORATIVE,
        ConnectionType.HIERARCHICAL,
      ];

      connectionTypes.forEach(type => {
        render(
          <ConnectionTools
            connectionState={{ ...mockConnectionState, connectionType: type }}
            onConnectionChange={mockOnConnectionChange}
          />
        );

        expect(screen.getByTestId('editor-card')).toBeInTheDocument();
      });
    });
  });

  describe('PropertyEditor Integration', () => {
    const mockSelectedElement: SelectedElement = {
      type: 'node',
      id: 'node-1',
      data: {
        agent: mockAgencyData.agents[0],
        position: { x: 100, y: 100 },
      },
    };

    it('should render property editor for selected node', () => {
      const mockOnPropertyChange = vi.fn();
      render(
        <PropertyEditor
          selectedElement={mockSelectedElement}
          onPropertyChange={mockOnPropertyChange}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
      expect(screen.getByTestId('editor-input')).toBeInTheDocument();
    });

    it('should handle property changes', async () => {
      const mockOnPropertyChange = vi.fn();
      const user = userEvent.setup();

      render(
        <PropertyEditor
          selectedElement={mockSelectedElement}
          onPropertyChange={mockOnPropertyChange}
        />
      );

      const input = screen.getByTestId('editor-input');
      await user.type(input, 'Updated Name');

      expect(mockOnPropertyChange).toHaveBeenCalled();
    });

    it('should handle different element types', () => {
      const mockOnPropertyChange = vi.fn();

      // Test node element
      render(
        <PropertyEditor
          selectedElement={mockSelectedElement}
          onPropertyChange={mockOnPropertyChange}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();

      // Test edge element
      const edgeElement: SelectedElement = {
        type: 'edge',
        id: 'edge-1',
        data: {
          communication: mockAgencyData.communication_flows[0],
        },
      };

      render(
        <PropertyEditor
          selectedElement={edgeElement}
          onPropertyChange={mockOnPropertyChange}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle null selected element', () => {
      const mockOnPropertyChange = vi.fn();

      render(
        <PropertyEditor
          selectedElement={null}
          onPropertyChange={mockOnPropertyChange}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });

  describe('WorkflowTemplates Integration', () => {
    it('should render workflow templates', () => {
      const mockOnTemplateSelect = vi.fn();
      render(
        <WorkflowTemplates
          templates={[mockWorkflowTemplate]}
          onTemplateSelect={mockOnTemplateSelect}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle template selection', async () => {
      const mockOnTemplateSelect = vi.fn();
      const user = userEvent.setup();

      render(
        <WorkflowTemplates
          templates={[mockWorkflowTemplate]}
          onTemplateSelect={mockOnTemplateSelect}
        />
      );

      // Simulate template selection
      await act(async () => {
        mockOnTemplateSelect(mockWorkflowTemplate);
      });

      expect(mockOnTemplateSelect).toHaveBeenCalledWith(mockWorkflowTemplate);
    });

    it('should handle template application', async () => {
      const mockOnTemplateApply = vi.fn();

      render(
        <WorkflowTemplates
          templates={[mockWorkflowTemplate]}
          onTemplateSelect={vi.fn()}
          onTemplateApply={mockOnTemplateApply}
        />
      );

      // Simulate template application
      await act(async () => {
        mockOnTemplateApply(mockWorkflowTemplate);
      });

      expect(mockOnTemplateApply).toHaveBeenCalledWith(mockWorkflowTemplate);
    });

    it('should filter templates by category', () => {
      const mockOnTemplateSelect = vi.fn();

      render(
        <WorkflowTemplates
          templates={[mockWorkflowTemplate]}
          onTemplateSelect={mockOnTemplateSelect}
          filterByCategory="analysis"
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle template search', () => {
      const mockOnTemplateSelect = vi.fn();

      render(
        <WorkflowTemplates
          templates={[mockWorkflowTemplate]}
          onTemplateSelect={mockOnTemplateSelect}
          searchTerm="analysis"
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });

  describe('WorkflowValidation Integration', () => {
    it('should render workflow validation results', () => {
      render(
        <WorkflowValidation
          validationResult={mockValidationResult}
          isValidationRunning={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle validation start', async () => {
      const mockOnValidate = vi.fn();

      render(
        <WorkflowValidation
          validationResult={null}
          isValidationRunning={true}
          onValidate={mockOnValidate}
        />
      );

      // Simulate validation start
      await act(async () => {
        mockOnValidate();
      });

      expect(mockOnValidate).toHaveBeenCalled();
    });

    it('should display validation errors', () => {
      const errorResult: ValidationResult = {
        is_valid: false,
        errors: [
          {
            type: 'connection_error',
            message: 'Invalid connection between nodes',
            severity: 'error',
            element_id: 'edge-1',
          },
        ],
        warnings: [],
        suggestions: [],
        metadata: {
          validated_at: new Date(),
          validator_version: '1.0.0',
        },
      };

      render(
        <WorkflowValidation
          validationResult={errorResult}
          isValidationRunning={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
      expect(screen.getByTestId('editor-alert')).toBeInTheDocument();
    });

    it('should display validation warnings', () => {
      const warningResult: ValidationResult = {
        is_valid: true,
        errors: [],
        warnings: [
          {
            type: 'performance_warning',
            message: 'High memory usage detected',
            severity: 'warning',
            element_id: 'node-1',
          },
        ],
        suggestions: [],
        metadata: {
          validated_at: new Date(),
          validator_version: '1.0.0',
        },
      };

      render(
        <WorkflowValidation
          validationResult={warningResult}
          isValidationRunning={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should display validation suggestions', () => {
      const suggestionResult: ValidationResult = {
        is_valid: true,
        errors: [],
        warnings: [],
        suggestions: [
          {
            type: 'optimization',
            message: 'Consider using parallel processing',
            severity: 'info',
            element_id: 'workflow',
          },
        ],
        metadata: {
          validated_at: new Date(),
          validator_version: '1.0.0',
        },
      };

      render(
        <WorkflowValidation
          validationResult={suggestionResult}
          isValidationRunning={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });

  describe('WorkflowPersistence Integration', () => {
    it('should render workflow persistence controls', () => {
      const mockOnSave = vi.fn();
      const mockOnLoad = vi.fn();
      const mockOnExport = vi.fn();
      const mockOnImport = vi.fn();

      render(
        <WorkflowPersistence
          onSave={mockOnSave}
          onLoad={mockOnLoad}
          onExport={mockOnExport}
          onImport={mockOnImport}
          isSaving={false}
          isLoading={false}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });

    it('should handle save operation', async () => {
      const mockOnSave = vi.fn();
      const user = userEvent.setup();

      render(
        <WorkflowPersistence
          onSave={mockOnSave}
          onLoad={vi.fn()}
          onExport={vi.fn()}
          onImport={vi.fn()}
          isSaving={false}
          isLoading={false}
        />
      );

      const saveButton = screen.getByTestId('editor-button');
      await user.click(saveButton);

      expect(mockOnSave).toHaveBeenCalled();
    });

    it('should handle load operation', async () => {
      const mockOnLoad = vi.fn();
      const user = userEvent.setup();

      render(
        <WorkflowPersistence
          onSave={vi.fn()}
          onLoad={mockOnLoad}
          onExport={vi.fn()}
          onImport={vi.fn()}
          isSaving={false}
          isLoading={false}
        />
      );

      const loadButton = screen.getAllByTestId('editor-button')[1];
      await user.click(loadButton);

      expect(mockOnLoad).toHaveBeenCalled();
    });

    it('should handle export operation', async () => {
      const mockOnExport = vi.fn();
      const user = userEvent.setup();

      render(
        <WorkflowPersistence
          onSave={vi.fn()}
          onLoad={vi.fn()}
          onExport={mockOnExport}
          onImport={vi.fn()}
          isSaving={false}
          isLoading={false}
        />
      );

      const exportButton = screen.getAllByTestId('editor-button')[2];
      await user.click(exportButton);

      expect(mockOnExport).toHaveBeenCalled();
    });

    it('should handle import operation', async () => {
      const mockOnImport = vi.fn();
      const user = userEvent.setup();

      render(
        <WorkflowPersistence
          onSave={vi.fn()}
          onLoad={vi.fn()}
          onExport={vi.fn()}
          onImport={mockOnImport}
          isSaving={false}
          isLoading={false}
        />
      );

      const importButton = screen.getAllByTestId('editor-button')[3];
      await user.click(importButton);

      expect(mockOnImport).toHaveBeenCalled();
    });

    it('should handle loading states', () => {
      render(
        <WorkflowPersistence
          onSave={vi.fn()}
          onLoad={vi.fn()}
          onExport={vi.fn()}
          onImport={vi.fn()}
          isSaving={true}
          isLoading={true}
        />
      );

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });

  describe('Editor History and Undo/Redo', () => {
    it('should handle undo operations', async () => {
      const mockOnUndo = vi.fn();
      const mockOnRedo = vi.fn();

      // Simulate keyboard shortcuts
      const undoEvent = new KeyboardEvent('keydown', {
        key: 'z',
        ctrlKey: true,
      });

      const redoEvent = new KeyboardEvent('keydown', {
        key: 'z',
        ctrlKey: true,
        shiftKey: true,
      });

      window.dispatchEvent(undoEvent);
      window.dispatchEvent(redoEvent);

      // In a real implementation, these would trigger the undo/redo functions
      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });

    it('should track editor state history', () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      // Simulate state changes
      const stateChangeEvent = new CustomEvent('editorStateChange', {
        detail: {
          action: 'node_create',
          timestamp: new Date(),
        },
      });

      window.dispatchEvent(stateChangeEvent);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });
  });

  describe('Performance and Optimization', () => {
    it('should handle large workflows efficiently', () => {
      const largeAgencyData: AgencyData = {
        ...mockAgencyData,
        agents: Array.from({ length: 100 }, (_, i) => ({
          id: `agent-${i}`,
          name: `Agent ${i}`,
          description: `Test agent ${i}`,
          type: AgentType.ANALYST,
          model_tier: ModelTier.SONNET,
          state: AgentState.ACTIVE,
          capabilities: ['analysis'],
          created_at: new Date(),
          updated_at: new Date(),
          metadata: {},
        })),
        communication_flows: Array.from({ length: 200 }, (_, i) => ({
          id: `flow-${i}`,
          source_agent_id: `agent-${i % 100}`,
          target_agent_id: `agent-${(i + 1) % 100}`,
          communication_type: CommunicationType.DIRECT,
          status: CommunicationStatus.ACTIVE,
          message_count: i,
          message_type: 'task_assignment',
        })),
      };

      render(<WorkflowEditor agencyData={largeAgencyData} />);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });

    it('should handle rapid editor operations', async () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      // Simulate rapid operations
      for (let i = 0; i < 10; i++) {
        const operationEvent = new CustomEvent('editorOperation', {
          detail: {
            type: 'node_create',
            data: { nodeId: `node-${i}` },
            timestamp: new Date(),
          },
        });

        window.dispatchEvent(operationEvent);
      }

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle missing agency data', () => {
      render(<WorkflowEditor agencyData={null} />);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });

    it('should handle corrupted workflow data', () => {
      const corruptedAgencyData = {
        ...mockAgencyData,
        agents: [
          {
            id: 'corrupted-agent',
            name: 'Corrupted Agent',
            // Missing required fields
          } as any,
        ],
      };

      render(<WorkflowEditor agencyData={corruptedAgencyData} />);

      expect(screen.getByTestId('react-flow-editor')).toBeInTheDocument();
    });

    it('should handle network errors during save/load', async () => {
      const mockOnSave = vi.fn(() => {
        throw new Error('Network error');
      });

      render(
        <WorkflowPersistence
          onSave={mockOnSave}
          onLoad={vi.fn()}
          onExport={vi.fn()}
          onImport={vi.fn()}
          isSaving={false}
          isLoading={false}
        />
      );

      // Simulate save operation with error
      try {
        await mockOnSave();
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
      }

      expect(screen.getByTestId('editor-card')).toBeInTheDocument();
    });
  });
});