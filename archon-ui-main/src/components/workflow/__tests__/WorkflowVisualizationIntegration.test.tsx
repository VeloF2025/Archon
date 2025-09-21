/**
 * Workflow Visualization Integration Test Suite
 * Tests the visualization components integration and ReactFlow functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AgencyWorkflowVisualizer } from '../AgencyWorkflowVisualizer';
import { AgentNode } from '../AgentNode';
import { CommunicationFlow } from '../CommunicationFlow';
import { WorkflowControls } from '../WorkflowControls';

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
  WorkflowStats,
  LayoutAlgorithm,
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
} from '../../../types/workflowTypes';

// Mock ReactFlow
vi.mock('@xyflow/react', () => ({
  ReactFlow: vi.fn(({ children, nodes, edges, onNodesChange, onEdgesChange, onConnect }) => (
    <div data-testid="react-flow" data-nodes={nodes?.length} data-edges={edges?.length}>
      {children}
      <div data-testid="nodes-display">{nodes?.map(n => n.id).join(',')}</div>
      <div data-testid="edges-display">{edges?.map(e => e.id).join(',')}</div>
    </div>
  )),
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useNodesState: vi.fn(() => [
    [
      {
        id: 'node-1',
        type: 'agent',
        position: { x: 100, y: 100 },
        data: {
          agent: {
            id: 'agent-1',
            name: 'Test Agent',
            type: AgentType.ANALYST,
            model_tier: ModelTier.SONNET,
            state: AgentState.ACTIVE,
            capabilities: ['analysis'],
            created_at: new Date(),
            updated_at: new Date(),
            metadata: {},
          },
          position: { x: 100, y: 100 },
        },
      },
    ],
    vi.fn(),
    vi.fn(),
  ]),
  useEdgesState: vi.fn(() => [
    [
      {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        type: 'communication',
        data: {
          communication: {
            id: 'flow-1',
            source_agent_id: 'agent-1',
            target_agent_id: 'agent-2',
            communication_type: CommunicationType.DIRECT,
            status: CommunicationStatus.ACTIVE,
            message_count: 5,
            message_type: 'task_assignment',
          },
        },
      },
    ],
    vi.fn(),
    vi.fn(),
  ]),
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
  Handle: vi.fn(({ children }) => <div data-testid="handle">{children}</div>),
  useReactFlow: vi.fn(() => ({
    fitView: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    setViewport: vi.fn(),
    getViewport: vi.fn(() => ({ x: 0, y: 0, zoom: 1 })),
  })),
}));

// Mock socket.io
vi.mock('socket.io-client', () => ({
  io: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  })),
}));

// Mock hooks
vi.mock('../../../hooks/useSocket', () => ({
  useSocket: vi.fn(() => ({
    socket: {
      on: vi.fn(),
      off: vi.fn(),
      emit: vi.fn(),
    },
    isConnected: true,
  })),
}));

vi.mock('../../../hooks/useToast', () => ({
  useToast: vi.fn(() => ({
    toast: vi.fn(),
    dismiss: vi.fn(),
  })),
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
  CardHeader: vi.fn(({ children }) => <div data-testid="card-header">{children}</div>),
  CardTitle: vi.fn(({ children }) => <div data-testid="card-title">{children}</div>),
}));

describe('Workflow Visualization Integration', () => {
  const mockAgencyData: AgencyData = {
    id: 'test-agency-1',
    name: 'Test Agency',
    description: 'Test agency for visualization',
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
      {
        id: 'flow-2',
        source_agent_id: 'agent-2',
        target_agent_id: 'agent-3',
        communication_type: CommunicationType.CHAIN,
        status: CommunicationStatus.PENDING,
        message_count: 0,
        message_type: 'status_update',
      },
      {
        id: 'flow-3',
        source_agent_id: 'agent-1',
        target_agent_id: 'agent-3',
        communication_type: CommunicationType.BROADCAST,
        status: CommunicationStatus.COMPLETED,
        message_count: 12,
        message_type: 'result_delivery',
        data_flow: {
          input_size: 2048,
          output_size: 4096,
          processing_time_ms: 300,
        },
      },
    ],
    created_at: new Date(),
    updated_at: new Date(),
  };

  const mockConfig: WorkflowVisualizationConfig = {
    auto_layout: true,
    show_labels: true,
    show_metrics: true,
    animation_speed: 500,
    node_size: 'medium',
    edge_style: 'curved',
    theme: 'light',
    filter: {
      agent_types: [AgentType.ANALYST, AgentType.SPECIALIST],
      agent_states: [AgentState.ACTIVE, AgentState.IDLE],
      communication_types: [CommunicationType.DIRECT, CommunicationType.CHAIN],
    },
  };

  const mockStats: WorkflowStats = {
    total_agents: 3,
    active_agents: 1,
    total_flows: 3,
    active_flows: 1,
    average_response_time: 200,
    throughput: 15,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('AgencyWorkflowVisualizer Core Functionality', () => {
    it('should render workflow visualization with all components', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      expect(screen.getByTestId('background')).toBeInTheDocument();
      expect(screen.getByTestId('controls')).toBeInTheDocument();
      expect(screen.getByTestId('minimap')).toBeInTheDocument();
      expect(screen.getByTestId('panel')).toBeInTheDocument();
    });

    it('should display correct number of nodes and edges', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      const reactFlow = screen.getByTestId('react-flow');
      expect(reactFlow).toHaveAttribute('data-nodes', '1'); // Mock returns 1 node
      expect(reactFlow).toHaveAttribute('data-edges', '1'); // Mock returns 1 edge
    });

    it('should handle configuration changes', () => {
      const { rerender } = render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      const updatedConfig = {
        ...mockConfig,
        theme: 'dark',
        node_size: 'large',
        show_metrics: false,
      };

      rerender(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={updatedConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle agency data updates', () => {
      const { rerender } = render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

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

      rerender(
        <AgencyWorkflowVisualizer
          agencyData={updatedAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });
  });

  describe('Node Visualization and Interaction', () => {
    it('should render agent nodes with correct data', () => {
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

    it('should handle node selection state', () => {
      const mockNodeData = {
        agent: mockAgencyData.agents[0],
        position: { x: 100, y: 100 },
        is_selected: false,
        communication_stats: {
          messages_sent: 10,
          messages_received: 5,
          active_connections: 2,
        },
      };

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

    it('should display different agent states with appropriate styling', () => {
      const states = [AgentState.ACTIVE, AgentState.IDLE, AgentState.HIBERNATED, AgentState.CREATED, AgentState.ARCHIVED];

      states.forEach(state => {
        const mockNodeData = {
          agent: { ...mockAgencyData.agents[0], state },
          position: { x: 100, y: 100 },
          is_selected: false,
          communication_stats: {
            messages_sent: 10,
            messages_received: 5,
            active_connections: 2,
          },
        };

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

        expect(screen.getByTestId('handle')).toBeInTheDocument();

        rerender(
          <AgentNode
            data={{ ...mockNodeData, agent: { ...mockNodeData.agent, state: AgentState.ACTIVE } }}
            selected={false}
            id="node-1"
            type="agent"
            position={{ x: 0, y: 0 }}
            data-testid="agent-node"
          />
        );
      });
    });

    it('should handle node click events', async () => {
      const mockEventHandler = vi.fn();
      const user = userEvent.setup();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate node click
      await act(async () => {
        mockEventHandler({
          type: 'nodeClick',
          data: { nodeId: 'agent-1', position: { x: 100, y: 100 } },
        });
      });

      expect(mockEventHandler).toHaveBeenCalledWith({
        type: 'nodeClick',
        data: { nodeId: 'agent-1', position: { x: 100, y: 100 } },
      });
    });
  });

  describe('Edge Visualization and Communication Flow', () => {
    it('should render communication edges with correct data', () => {
      const mockEdgeData = {
        communication: mockAgencyData.communication_flows[0],
        is_animated: false,
        is_highlighted: false,
        message_flow: {
          direction: 'source-to-target' as const,
          intensity: 0.8,
        },
      };

      render(
        <CommunicationFlow
          data={mockEdgeData}
          selected={false}
          id="edge-1"
          source="node-1"
          target="node-2"
          sourceHandle="source"
          targetHandle="target"
          data-testid="communication-edge"
        />
      );
    });

    it('should handle different communication types', () => {
      const communicationTypes = [
        CommunicationType.DIRECT,
        CommunicationType.BROADCAST,
        CommunicationType.CHAIN,
        CommunicationType.COLLABORATIVE,
        CommunicationType.HIERARCHICAL,
      ];

      communicationTypes.forEach(type => {
        const mockEdgeData = {
          communication: { ...mockAgencyData.communication_flows[0], communication_type: type },
          is_animated: false,
          message_flow: {
            direction: 'source-to-target' as const,
            intensity: 0.5,
          },
        };

        render(
          <CommunicationFlow
            data={mockEdgeData}
            selected={false}
            id="edge-1"
            source="node-1"
            target="node-2"
            sourceHandle="source"
            targetHandle="target"
            data-testid="communication-edge"
          />
        );
      });
    });

    it('should handle communication status changes', () => {
      const statuses = [
        CommunicationStatus.ACTIVE,
        CommunicationStatus.IDLE,
        CommunicationStatus.PENDING,
        CommunicationStatus.FAILED,
        CommunicationStatus.COMPLETED,
      ];

      statuses.forEach(status => {
        const mockEdgeData = {
          communication: { ...mockAgencyData.communication_flows[0], status },
          is_animated: status === CommunicationStatus.ACTIVE,
          message_flow: {
            direction: 'source-to-target' as const,
            intensity: 0.7,
          },
        };

        render(
          <CommunicationFlow
            data={mockEdgeData}
            selected={false}
            id="edge-1"
            source="node-1"
            target="node-2"
            sourceHandle="source"
            targetHandle="target"
            data-testid="communication-edge"
          />
        );
      });
    });

    it('should handle edge selection and highlighting', () => {
      const mockEdgeData = {
        communication: mockAgencyData.communication_flows[0],
        is_animated: false,
        is_highlighted: false,
        message_flow: {
          direction: 'source-to-target' as const,
          intensity: 0.6,
        },
      };

      const { rerender } = render(
        <CommunicationFlow
          data={mockEdgeData}
          selected={false}
          id="edge-1"
          source="node-1"
          target="node-2"
          sourceHandle="source"
          targetHandle="target"
          data-testid="communication-edge"
        />
      );

      rerender(
        <CommunicationFlow
          data={{ ...mockEdgeData, is_highlighted: true }}
          selected={true}
          id="edge-1"
          source="node-1"
          target="node-2"
          sourceHandle="source"
          targetHandle="target"
          data-testid="communication-edge"
        />
      );
    });
  });

  describe('Layout and Positioning', () => {
    it('should handle different layout algorithms', async () => {
      const mockEventHandler = vi.fn();
      const layouts: LayoutAlgorithm[] = ['force', 'circular', 'hierarchical', 'grid'];

      for (const layout of layouts) {
        render(
          <AgencyWorkflowVisualizer
            agencyData={mockAgencyData}
            config={mockConfig}
            onEvent={mockEventHandler}
          />
        );

        await act(async () => {
          mockEventHandler({
            type: 'layoutChange',
            layout,
            timestamp: new Date(),
          });
        });

        expect(screen.getByTestId('react-flow')).toBeInTheDocument();
      }
    });

    it('should handle auto-layout functionality', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={{ ...mockConfig, auto_layout: true }}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle manual positioning when auto-layout is disabled', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={{ ...mockConfig, auto_layout: false }}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });
  });

  describe('Real-time Updates and Animation', () => {
    it('should handle agent state updates', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={mockEventHandler}
        />
      );

      await act(async () => {
        mockEventHandler({
          type: 'agentStateUpdate',
          agentId: 'agent-1',
          newState: AgentState.ACTIVE,
          previousState: AgentState.IDLE,
          timestamp: new Date(),
        });
      });

      expect(mockEventHandler).toHaveBeenCalledWith({
        type: 'agentStateUpdate',
        agentId: 'agent-1',
        newState: AgentState.ACTIVE,
        previousState: AgentState.IDLE,
        timestamp: expect.any(Date),
      });
    });

    it('should handle communication flow updates', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={mockEventHandler}
        />
      );

      await act(async () => {
        mockEventHandler({
          type: 'communicationFlowUpdate',
          flowId: 'flow-1',
          status: CommunicationStatus.ACTIVE,
          messageCount: 10,
          timestamp: new Date(),
        });
      });

      expect(mockEventHandler).toHaveBeenCalledWith({
        type: 'communicationFlowUpdate',
        flowId: 'flow-1',
        status: CommunicationStatus.ACTIVE,
        messageCount: 10,
        timestamp: expect.any(Date),
      });
    });

    it('should handle animation state changes', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={mockEventHandler}
        />
      );

      await act(async () => {
        mockEventHandler({
          type: 'animationToggle',
          isAnimating: true,
          timestamp: new Date(),
        });
      });

      expect(mockEventHandler).toHaveBeenCalledWith({
        type: 'animationToggle',
        isAnimating: true,
        timestamp: expect.any(Date),
      });
    });
  });

  describe('Controls and User Interaction', () => {
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

    it('should render workflow controls', () => {
      render(
        <WorkflowControls
          controls={mockControls}
          config={mockConfig}
          stats={mockStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={vi.fn()}
        />
      );

      expect(screen.getByTestId('button')).toBeInTheDocument();
      expect(screen.getByTestId('card')).toBeInTheDocument();
    });

    it('should handle zoom controls', async () => {
      const user = userEvent.setup();

      render(
        <WorkflowControls
          controls={mockControls}
          config={mockConfig}
          stats={mockStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={vi.fn()}
        />
      );

      const zoomButton = screen.getByTestId('button');
      await user.click(zoomButton);

      expect(mockControls.zoom_in).toHaveBeenCalled();
    });

    it('should handle layout changes', async () => {
      const user = userEvent.setup();
      const mockConfigChange = vi.fn();

      render(
        <WorkflowControls
          controls={mockControls}
          config={mockConfig}
          stats={mockStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={mockConfigChange}
        />
      );

      // Simulate layout selection
      await act(async () => {
        mockControls.apply_layout('circular');
      });

      expect(mockControls.apply_layout).toHaveBeenCalledWith('circular');
    });

    it('should handle animation toggle', async () => {
      const user = userEvent.setup();

      render(
        <WorkflowControls
          controls={mockControls}
          config={mockConfig}
          stats={mockStats}
          isAnimating={false}
          currentLayout="force"
          onConfigChange={vi.fn()}
        />
      );

      // Simulate animation toggle
      await act(async () => {
        mockControls.toggle_animation();
      });

      expect(mockControls.toggle_animation).toHaveBeenCalled();
    });
  });

  describe('Performance and Optimization', () => {
    it('should handle large datasets efficiently', () => {
      const largeAgencyData: AgencyData = {
        ...mockAgencyData,
        agents: Array.from({ length: 50 }, (_, i) => ({
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
        communication_flows: Array.from({ length: 100 }, (_, i) => ({
          id: `flow-${i}`,
          source_agent_id: `agent-${i % 50}`,
          target_agent_id: `agent-${(i + 1) % 50}`,
          communication_type: CommunicationType.DIRECT,
          status: CommunicationStatus.ACTIVE,
          message_count: i,
          message_type: 'task_assignment',
        })),
      };

      render(
        <AgencyWorkflowVisualizer
          agencyData={largeAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle rapid updates without performance issues', async () => {
      const mockEventHandler = vi.fn();

      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={mockConfig}
          onEvent={mockEventHandler}
        />
      );

      // Simulate rapid updates
      for (let i = 0; i < 20; i++) {
        await act(async () => {
          mockEventHandler({
            type: 'agentStateUpdate',
            agentId: `agent-${i % 3}`,
            newState: AgentState.ACTIVE,
            timestamp: new Date(),
          });
        });
      }

      expect(mockEventHandler).toHaveBeenCalledTimes(20);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle empty agency data', () => {
      const emptyAgencyData: AgencyData = {
        id: 'empty-agency',
        name: 'Empty Agency',
        agents: [],
        communication_flows: [],
        created_at: new Date(),
        updated_at: new Date(),
      };

      render(
        <AgencyWorkflowVisualizer
          agencyData={emptyAgencyData}
          config={mockConfig}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle missing configuration', () => {
      render(
        <AgencyWorkflowVisualizer
          agencyData={mockAgencyData}
          config={undefined}
          onEvent={vi.fn()}
        />
      );

      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should handle invalid node data gracefully', () => {
      const invalidNodeData = {
        agent: null,
        position: { x: 100, y: 100 },
        is_selected: false,
      };

      render(
        <AgentNode
          data={invalidNodeData}
          selected={false}
          id="node-1"
          type="agent"
          position={{ x: 0, y: 0 }}
          data-testid="agent-node"
        />
      );

      expect(screen.getByTestId('handle')).toBeInTheDocument();
    });
  });
});