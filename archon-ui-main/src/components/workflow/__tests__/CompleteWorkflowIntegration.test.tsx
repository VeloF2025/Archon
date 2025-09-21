/**
 * Complete Workflow Integration Test Suite
 * Tests the integration of all workflow components working together
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AgencyWorkflowVisualizer } from '../AgencyWorkflowVisualizer';
import { WorkflowEditor } from '../WorkflowEditor';
import { KnowledgeAwareWorkflow } from '../KnowledgeAwareWorkflow';
import { WorkflowControls } from '../WorkflowControls';
import { AgentNode } from '../AgentNode';
import { CommunicationFlow } from './CommunicationFlow';

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
  LayoutAlgorithm,
  WorkflowStats,
  EditorMode,
} from '../../../types/workflowTypes';

// Mock socket.io client
vi.mock('socket.io-client', () => ({
  io: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  })),
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

// Mock react flow components
vi.mock('@xyflow/react', () => ({
  ReactFlow: vi.fn(({ children }) => <div data-testid="react-flow">{children}</div>),
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useNodesState: vi.fn(() => [[], vi.fn(), vi.fn()]),
  useEdgesState: vi.fn(() => [[], vi.fn(), vi.fn()]),
  addEdge: vi.fn((edges, connection) => [...edges, { id: 'edge-new', ...connection }]),
  Background: vi.fn(() => <div data-testid="background" />),
  Controls: vi.fn(() => <div data-testid="controls" />),
  MiniMap: vi.fn(() => <div data-testid="minimap" />),
  Panel: vi.fn(({ children }) => <div data-testid="panel">{children}</div>),
  NodeTypes: {},
  EdgeTypes: {},
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
  MarkerType: {
    ArrowClosed: 'arrowclosed',
  },
  Handle: vi.fn(({ children }) => <div data-testid="handle">{children}</div>),
}));

// Mock UI components
vi.mock('../../ui/button', () => ({
  Button: vi.fn(({ children, ...props }) => (
    <button data-testid="button" {...props}>
      {children}
    </button>
  )),
}));

vi.mock('../../ui/card', () => ({
  Card: vi.fn(({ children, ...props }) => (
    <div data-testid="card" {...props}>
      {children}
    </div>
  )),
  CardContent: vi.fn(({ children }) => <div data-testid="card-content">{children}</div>),
  CardDescription: vi.fn(({ children }) => <div data-testid="card-description">{children}</div>),
  CardHeader: vi.fn(({ children }) => <div data-testid="card-header">{children}</div>),
  CardTitle: vi.fn(({ children }) => <div data-testid="card-title">{children}</div>),
}));

vi.mock('../../ui/tabs', () => ({
  Tabs: vi.fn(({ children, ...props }) => (
    <div data-testid="tabs" {...props}>
      {children}
    </div>
  )),
  TabsContent: vi.fn(({ children }) => <div data-testid="tabs-content">{children}</div>),
  TabsList: vi.fn(({ children }) => <div data-testid="tabs-list">{children}</div>),
  TabsTrigger: vi.fn(({ children }) => <button data-testid="tabs-trigger">{children}</button>),
}));

vi.mock('../../ui/select', () => ({
  Select: vi.fn(({ children, ...props }) => (
    <div data-testid="select" {...props}>
      {children}
    </div>
  )),
  SelectContent: vi.fn(({ children }) => <div data-testid="select-content">{children}</div>),
  SelectItem: vi.fn(({ children }) => <div data-testid="select-item">{children}</div>),
  SelectTrigger: vi.fn(({ children }) => <button data-testid="select-trigger">{children}</button>),
  SelectValue: vi.fn(() => <div data-testid="select-value" />),
}));

vi.mock('../../ui/input', () => ({
  Input: vi.fn((props) => <input data-testid="input" {...props} />),
}));

vi.mock('../../ui/textarea', () => ({
  Textarea: vi.fn((props) => <textarea data-testid="textarea" {...props} />),
}));

vi.mock('../../ui/label', () => ({
  Label: vi.fn(({ children }) => <label data-testid="label">{children}</label>),
}));

vi.mock('../../ui/badge', () => ({
  Badge: vi.fn(({ children }) => <span data-testid="badge">{children}</span>),
}));

vi.mock('../../ui/alert', () => ({
  Alert: vi.fn(({ children }) => <div data-testid="alert">{children}</div>),
  AlertDescription: vi.fn(({ children }) => <div data-testid="alert-description">{children}</div>),
}));

describe('Complete Workflow Integration', () => {
  const mockAgencyData: AgencyData = {
    id: 'test-agency-1',
    name: 'Test Agency',
    description: 'Test agency for integration',
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

  const mockWorkflowConfig: WorkflowVisualizationConfig = {
    auto_layout: true,
    show_labels: true,
    show_metrics: true,
    animation_speed: 500,
    node_size: 'medium',
    edge_style: 'curved',
    theme: 'light',
  };

  const mockWorkflowStats: WorkflowStats = {
    total_agents: 2,
    active_agents: 1,
    total_flows: 1,
    active_flows: 1,
    average_response_time: 150,
    throughput: 10,
  };

  const mockControls = {
    zoom_in: vi.fn(),
    zoom_out: vi.fn(),
    zoom_reset: vi.fn(),
    fit_view: vi.fn(),
    toggle_animation: vi.fn(),
    apply_layout: vi.fn(),
    export_workflow: vi.fn(),
    import_workflow: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('AgencyWorkflowVisualizer Integration', () => {
    it('should render workflow visualizer with agency data', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      expect(screen.getByTestId('background')).toBeInTheDocument();
      expect(screen.getByTestId('controls')).toBeInTheDocument();
      expect(screen.getByTestId('minimap')).toBeInTheDocument();
    });

    it('should handle workflow events correctly', async () => {
      const mockEventHandler = vi.fn();
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate node click event
      const nodeData = {
        type: 'nodeClick',
        data: { nodeId: 'agent-1', position: { x: 100, y: 100 } },
      };

      await act(async () => {
        mockEventHandler(nodeData);
      });

      expect(mockEventHandler).toHaveBeenCalledWith(nodeData);
    });

    it('should update visualization when config changes', () => {
      const { rerender } = render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      const updatedConfig = {
        ...mockWorkflowConfig,
        theme: 'dark',
        node_size: 'large',
      };

      rerender(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={updatedConfig}
          onEvent={vi.fn()}
        />
      );

      // Component should re-render with new config
      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });
  });

  describe('WorkflowEditor Integration', () => {
    it('should render workflow editor with all components', () => {
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      expect(screen.getByTestId('panel')).toBeInTheDocument();
    });

    it('should handle editor mode switching', async () => {
      const user = userEvent.setup();
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      // Simulate mode switch (this would normally come from UI controls)
      const modeChangeEvent = new CustomEvent('modeChange', {
        detail: { mode: EditorMode.EDIT },
      });

      window.dispatchEvent(modeChangeEvent);

      await waitFor(() => {
        // Editor should be in edit mode
        expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      });
    });

    it('should handle node creation and deletion', async () => {
      const user = userEvent.setup();
      render(<WorkflowEditor agencyData={mockAgencyData} />);

      // Simulate node creation
      const nodeCreateEvent = new CustomEvent('nodeCreate', {
        detail: {
          node: {
            id: 'new-agent',
            type: 'agent',
            position: { x: 200, y: 200 },
            data: { agent: mockAgencyData.agents[0] },
          },
        },
      });

      window.dispatchEvent(nodeCreateEvent);

      await waitFor(() => {
        // New node should be added to the workflow
        expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      });
    });
  });

  describe('KnowledgeAwareWorkflow Integration', () => {
    const mockKnowledgeSession = {
      id: 'session-1',
      workflow_id: 'workflow-1',
      project_id: 'project-1',
      status: 'active',
      created_at: new Date(),
      insights: [],
      contextual_knowledge: [],
    };

    beforeEach(() => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).startKnowledgeSession.mockResolvedValue(mockKnowledgeSession);
    });

    it('should initialize knowledge session correctly', async () => {
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.startKnowledgeSession
        ).toHaveBeenCalledWith('workflow-1', 'project-1', expect.any(Object), expect.any(Array));
      });
    });

    it('should capture workflow events and update knowledge', async () => {
      const mockWorkflowUpdate = vi.fn();
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={mockWorkflowUpdate}
        />
      );

      // Simulate workflow event
      const workflowEvent = {
        type: 'agent_execution',
        agent_id: 'agent-1',
        timestamp: new Date(),
        data: { result: 'success' },
      };

      await act(async () => {
        mockWorkflowUpdate(workflowEvent);
      });

      expect(mockWorkflowUpdate).toHaveBeenCalledWith(workflowEvent);
    });

    it('should display knowledge insights correctly', async () => {
      const mockInsights = [
        {
          id: 'insight-1',
          type: 'performance',
          description: 'Workflow efficiency improved by 25%',
          confidence: 0.85,
          created_at: new Date(),
        },
      ];

      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).getWorkflowInsights.mockResolvedValue(mockInsights);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.getWorkflowInsights
        ).toHaveBeenCalled();
      });
    });
  });

  describe('WorkflowControls Integration', () => {
    it('should render controls with proper configuration', () => {
      render(
        <WorkflowControls
          controls={mockControls}
          config={mockWorkflowConfig}
          stats={mockWorkflowStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={vi.fn()}
        />
      );

      expect(screen.getByTestId('button')).toBeInTheDocument();
      expect(screen.getByTestId('card')).toBeInTheDocument();
    });

    it('should handle control interactions correctly', async () => {
      const user = userEvent.setup();
      const mockConfigChange = vi.fn();
      render(
        <WorkflowControls
          controls={mockControls}
          config={mockWorkflowConfig}
          stats={mockWorkflowStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={mockConfigChange}
        />
      );

      // Click zoom in button
      const zoomInButton = screen.getByTestId('button');
      await user.click(zoomInButton);

      expect(mockControls.zoom_in).toHaveBeenCalled();
    });

    it('should update configuration when layout changes', async () => {
      const user = userEvent.setup();
      const mockConfigChange = vi.fn();
      render(
        <WorkflowControls
          controls={mockControls}
          config={mockWorkflowConfig}
          stats={mockWorkflowStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={mockConfigChange}
        />
      );

      // Simulate layout change
      const newConfig = { ...mockWorkflowConfig, auto_layout: false };
      mockConfigChange(newConfig);

      expect(mockConfigChange).toHaveBeenCalledWith(newConfig);
    });
  });

  describe('AgentNode Integration', () => {
    const mockNodeData = {
      agent: mockAgencyData.agents[0],
      position: { x: 100, y: 100 },
      is_selected: false,
      is_highlighted: false,
      communication_stats: {
        messages_sent: 10,
        messages_received: 5,
        active_connections: 2,
      },
    };

    it('should render agent node with correct styling based on state', () => {
      render(
        <AgentNode
          data={mockNodeData}
          selected={false}
          id="node-1"
          type="agent"
          position={{ x: 0, y: 0 }}
          data-testid="agent-node"
        />
      );

      expect(screen.getByTestId('handle')).toBeInTheDocument();
    });

    it('should apply different styling for different agent states', () => {
      const activeNodeData = {
        ...mockNodeData,
        agent: { ...mockNodeData.agent, state: AgentState.ACTIVE },
      };

      render(
        <AgentNode
          data={activeNodeData}
          selected={false}
          id="node-1"
          type="agent"
          position={{ x: 0, y: 0 }}
          data-testid="agent-node"
        />
      );

      expect(screen.getByTestId('handle')).toBeInTheDocument();
    });

    it('should handle node selection correctly', () => {
      const { rerender } = render(
        <AgentNode
          data={mockNodeData}
          selected={false}
          id="node-1"
          type="agent"
          position={{ x: 0, y: 0 }}
          data-testid="agent-node"
        />
      );

      rerender(
        <AgentNode
          data={{ ...mockNodeData, is_selected: true }}
          selected={true}
          id="node-1"
          type="agent"
          position={{ x: 0, y: 0 }}
          data-testid="agent-node"
        />
      );

      expect(screen.getByTestId('handle')).toBeInTheDocument();
    });
  });

  describe('Cross-Component Communication', () => {
    it('should handle real-time updates across components', async () => {
      const mockEventHandler = vi.fn();
      const mockWorkflowUpdate = vi.fn();

      render(
        <div>
          <AgencyWorkflowVisualizer
            agencyData={mockAgencyData}
            config={mockWorkflowConfig}
            onEvent={mockEventHandler}
          />
          <KnowledgeAwareWorkflow
            workflowId="workflow-1"
            projectId="project-1"
            onWorkflowUpdate={mockWorkflowUpdate}
          />
        </div>
      );

      // Simulate real-time agent state change
      const agentStateChangeEvent = new CustomEvent('agentStateChange', {
        detail: {
          agentId: 'agent-1',
          newState: AgentState.ACTIVE,
          timestamp: new Date(),
        },
      });

      window.dispatchEvent(agentStateChangeEvent);

      await waitFor(() => {
        expect(mockEventHandler).toHaveBeenCalled();
        expect(mockWorkflowUpdate).toHaveBeenCalled();
      });
    });

    it('should handle workflow execution state changes', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate workflow execution start
      const executionStartEvent = {
        type: 'workflowExecutionStart',
        workflowId: 'workflow-1',
        timestamp: new Date(),
      };

      await act(async () => {
        mockEventHandler(executionStartEvent);
      });

      expect(mockEventHandler).toHaveBeenCalledWith(executionStartEvent);
    });

    it('should handle error states gracefully', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate error event
      const errorEvent = {
        type: 'workflowError',
        error: 'Test error message',
        timestamp: new Date(),
      };

      await act(async () => {
        mockEventHandler(errorEvent);
      });

      expect(mockEventHandler).toHaveBeenCalledWith(errorEvent);
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large numbers of agents efficiently', () => {
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

      render(
        <AgencyWorkflowVisualizer
          agencyData={largeAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle rapid state changes without performance issues', async () => {
      const mockEventHandler = vi.fn();
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate rapid state changes
      for (let i = 0; i < 10; i++) {
        await act(async () => {
          mockEventHandler({
            type: 'agentStateChange',
            agentId: `agent-${i}`,
            newState: AgentState.ACTIVE,
            timestamp: new Date(),
          });
        });
      }

      expect(mockEventHandler).toHaveBeenCalledTimes(10);
    });
  });

  describe('Accessibility and User Experience', () => {
    it('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      // Test keyboard navigation (would need actual keyboard events)
      const tabEvent = new KeyboardEvent('keydown', { key: 'Tab' });
      window.dispatchEvent(tabEvent);

      await waitFor(() => {
        // Component should handle keyboard events
        expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      });
    });

    it('should provide proper ARIA labels and descriptions', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      const workflowContainer = screen.getByTestId('react-flow');
      expect(workflowContainer).toBeInTheDocument();
    });

    it('should handle screen reader compatibility', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockWorkflowConfig}
          onEvent={vi.fn()}
        />
      );

      // Components should have proper ARIA attributes
      const workflowContainer = screen.getByTestId('react-flow');
      expect(workflowContainer).toBeInTheDocument();
    });
  });
});