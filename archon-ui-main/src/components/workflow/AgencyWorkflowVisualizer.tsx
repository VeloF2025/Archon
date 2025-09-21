import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Background,
  Controls,
  MiniMap,
  Panel,
  useReactFlow,
  ReactFlowProvider,
  Position,
  Handle,
  NodeTypes,
  EdgeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { AgentNode } from './AgentNode';
import { CommunicationFlowEdge } from './CommunicationFlow';
import { WorkflowControls } from './WorkflowControls';

import {
  AgencyData,
  AgentNodeData,
  CommunicationEdgeData,
  WorkflowVisualizationConfig,
  WorkflowStats,
  LayoutAlgorithm,
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
  WorkflowEventHandlers,
} from '../../types/workflowTypes';
import { AgentState, ModelTier, AgentType } from '../../types/agentTypes';
import { useSocket } from '../../hooks/useSocket';
import { useToast } from '../../hooks/useToast';

interface AgencyWorkflowVisualizerProps {
  agencyData: AgencyData;
  config?: Partial<WorkflowVisualizationConfig>;
  onEvent?: (event: any) => void;
  className?: string;
}

// Default configuration
const DEFAULT_CONFIG: WorkflowVisualizationConfig = {
  auto_layout: true,
  show_labels: true,
  show_metrics: true,
  animation_speed: 1000,
  node_size: 'medium',
  edge_style: 'curved',
  theme: 'dark',
};

export const AgencyWorkflowVisualizer: React.FC<AgencyWorkflowVisualizerProps> = ({
  agencyData,
  config = {},
  onEvent,
  className = '',
}) => {
  const { toast } = useToast();
  const socket = useSocket();
  const reactFlowInstance = useReactFlow();

  // Merge config with defaults
  const [visualizationConfig, setVisualizationConfig] = useState<WorkflowVisualizationConfig>({
    ...DEFAULT_CONFIG,
    ...config,
  });

  // State management
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(true);
  const [currentLayout, setCurrentLayout] = useState<LayoutAlgorithm>('force');
  const [workflowStats, setWorkflowStats] = useState<WorkflowStats | null>(null);

  // Custom node and edge types
  const nodeTypes: NodeTypes = useMemo(() => ({
    agent: AgentNode,
  }), []);

  const edgeTypes: EdgeTypes = useMemo(() => ({
    communication: CommunicationFlowEdge,
  }), []);

  // Calculate workflow statistics
  const calculateWorkflowStats = useCallback((data: AgencyData): WorkflowStats => {
    const activeAgents = data.agents.filter(agent => agent.state === AgentState.ACTIVE);
    const activeCommunications = data.communication_flows.filter(
      flow => flow.status === 'active'
    );

    const agentMessageCounts = new Map<string, number>();
    data.communication_flows.forEach(flow => {
      agentMessageCounts.set(
        flow.source_agent_id,
        (agentMessageCounts.get(flow.source_agent_id) || 0) + flow.message_count
      );
      agentMessageCounts.set(
        flow.target_agent_id,
        (agentMessageCounts.get(flow.target_agent_id) || 0) + flow.message_count
      );
    });

    let busiestAgent: { agent_id: string; message_count: number } | undefined;
    let maxMessages = 0;
    agentMessageCounts.forEach((count, agentId) => {
      if (count > maxMessages) {
        maxMessages = count;
        busiestAgent = { agent_id: agentId, message_count: count };
      }
    });

    const commTypeDistribution = data.communication_flows.reduce((acc, flow) => {
      acc[flow.communication_type] = (acc[flow.communication_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const agentTypeDistribution = data.agents.reduce((acc, agent) => {
      acc[agent.agent_type] = (acc[agent.agent_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      total_agents: data.agents.length,
      active_agents: activeAgents.length,
      total_communications: data.communication_flows.length,
      active_communications: activeCommunications.length,
      avg_messages_per_connection: data.communication_flows.length > 0
        ? data.communication_flows.reduce((sum, flow) => sum + flow.message_count, 0) / data.communication_flows.length
        : 0,
      busiest_agent,
      communication_type_distribution: commTypeDistribution as any,
      agent_type_distribution: agentTypeDistribution as any,
    };
  }, []);

  // Convert agency data to ReactFlow format
  const convertToReactFlowFormat = useCallback((data: AgencyData) => {
    const nodes: ExtendedAgentNode[] = data.agents.map((agent, index) => {
      const angle = (index / data.agents.length) * 2 * Math.PI;
      const radius = 200;
      const position = {
        x: 400 + Math.cos(angle) * radius,
        y: 300 + Math.sin(angle) * radius,
      };

      return {
        id: agent.id,
        type: 'agent',
        position,
        data: {
          agent,
          position,
          is_selected: selectedNode === agent.id,
          communication_stats: {
            messages_sent: data.communication_flows.filter(f => f.source_agent_id === agent.id).length,
            messages_received: data.communication_flows.filter(f => f.target_agent_id === agent.id).length,
            active_connections: data.communication_flows.filter(
              f => (f.source_agent_id === agent.id || f.target_agent_id === agent.id) &&
                   f.status === 'active'
            ).length,
          },
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };
    });

    const edges: ExtendedCommunicationEdge[] = data.communication_flows.map((flow) => ({
      id: flow.id,
      type: 'communication',
      source: flow.source_agent_id,
      target: flow.target_agent_id,
      data: {
        communication: flow,
        is_animated: isAnimating && flow.status === 'active',
        is_highlighted: false,
        message_flow: {
          direction: 'bidirectional',
          intensity: Math.min(flow.message_count / 10, 1),
        },
      },
      animated: isAnimating && flow.status === 'active',
      style: {
        strokeWidth: Math.max(1, Math.min(flow.message_count / 5, 5)),
        stroke: flow.status === 'active' ? '#10b981' : '#6b7280',
      },
    }));

    return { nodes, edges };
  }, [selectedNode, isAnimating]);

  // Initialize workflow
  useEffect(() => {
    const { nodes: initialNodes, edges: initialEdges } = convertToReactFlowFormat(agencyData);
    setNodes(initialNodes);
    setEdges(initialEdges);
    setWorkflowStats(calculateWorkflowStats(agencyData));
  }, [agencyData, convertToReactFlowFormat, calculateWorkflowStats]);

  // Socket event handlers for real-time updates
  useEffect(() => {
    if (!socket) return;

    const handleAgentUpdate = (data: any) => {
      setNodes(prevNodes =>
        prevNodes.map(node =>
          node.id === data.agentId
            ? {
                ...node,
                data: {
                  ...node.data,
                  agent: { ...node.data.agent, ...data.updates },
                },
              }
            : node
        )
      );
    };

    const handleCommunicationUpdate = (data: any) => {
      setEdges(prevEdges =>
        prevEdges.map(edge =>
          edge.id === data.communicationId
            ? {
                ...edge,
                data: {
                  ...edge.data,
                  communication: { ...edge.data.communication, ...data.updates },
                },
              }
            : edge
        )
      );
    };

    socket.addMessageHandler('workflow_agent_update', handleAgentUpdate);
    socket.addMessageHandler('workflow_communication_update', handleCommunicationUpdate);

    return () => {
      socket.removeMessageHandler('workflow_agent_update', handleAgentUpdate);
      socket.removeMessageHandler('workflow_communication_update', handleCommunicationUpdate);
    };
  }, [socket]);

  // ReactFlow event handlers
  const onConnect = useCallback(
    (params: Edge | Connection) => setEdges((eds) => addEdge(params as Edge, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
    setNodes(prevNodes =>
      prevNodes.map(n => ({
        ...n,
        data: {
          ...n.data,
          is_selected: n.id === node.id,
        },
      }))
    );

    if (onEvent) {
      onEvent({
        type: 'node_selected',
        data: { node_id: node.id, timestamp: new Date() },
      });
    }
  }, [setNodes, onEvent]);

  const onEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    setEdges(prevEdges =>
      prevEdges.map(e => ({
        ...e,
        data: {
          ...e.data,
          is_highlighted: e.id === edge.id,
        },
      }))
    );
  }, [setEdges]);

  // Layout algorithms
  const applyLayout = useCallback((layoutType: LayoutAlgorithm) => {
    setCurrentLayout(layoutType);

    const layoutNodes = [...nodes];
    const centerX = 400;
    const centerY = 300;

    switch (layoutType) {
      case 'circular':
        layoutNodes.forEach((node, index) => {
          const angle = (index / layoutNodes.length) * 2 * Math.PI;
          const radius = 200;
          node.position = {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
          };
        });
        break;

      case 'hierarchical':
        const tiers = {
          [ModelTier.OPUS]: 0,
          [ModelTier.SONNET]: 1,
          [ModelTier.HAIKU]: 2,
        };

        const tierGroups: Record<number, Node[]> = {};
        layoutNodes.forEach(node => {
          const tier = tiers[node.data.agent.model_tier] || 1;
          if (!tierGroups[tier]) tierGroups[tier] = [];
          tierGroups[tier].push(node);
        });

        Object.entries(tierGroups).forEach(([tier, tierNodes]) => {
          const y = 100 + parseInt(tier) * 150;
          tierNodes.forEach((node, index) => {
            const x = 100 + (index * 600) / Math.max(1, tierNodes.length - 1);
            node.position = { x, y };
          });
        });
        break;

      case 'grid':
        const cols = Math.ceil(Math.sqrt(layoutNodes.length));
        layoutNodes.forEach((node, index) => {
          const row = Math.floor(index / cols);
          const col = index % cols;
          node.position = {
            x: 100 + col * 150,
            y: 100 + row * 150,
          };
        });
        break;

      case 'force':
      default:
        // Use force-directed layout (simplified)
        layoutNodes.forEach((node, index) => {
          const angle = (index / layoutNodes.length) * 2 * Math.PI;
          const radius = 150 + Math.random() * 100;
          node.position = {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
          };
        });
        break;
    }

    setNodes(layoutNodes);
    setCurrentLayout(layoutType);
  }, [nodes, setNodes]);

  // Control panel functions
  const fitToScreen = useCallback(() => {
    reactFlowInstance.fitView();
  }, [reactFlowInstance]);

  const zoomIn = useCallback(() => {
    reactFlowInstance.zoomIn();
  }, [reactFlowInstance]);

  const zoomOut = useCallback(() => {
    reactFlowInstance.zoomOut();
  }, [reactFlowInstance]);

  const centerView = useCallback(() => {
    reactFlowInstance.setViewport({ x: 0, y: 0, zoom: 1 });
  }, [reactFlowInstance]);

  const toggleAnimation = useCallback(() => {
    setIsAnimating(prev => {
      const newValue = !prev;
      setEdges(prevEdges =>
        prevEdges.map(edge => ({
          ...edge,
          animated: newValue && edge.data.communication.status === 'active',
          data: {
            ...edge.data,
            is_animated: newValue && edge.data.communication.status === 'active',
          },
        }))
      );
      return newValue;
    });
  }, [setEdges]);

  const refreshData = useCallback(() => {
    const { nodes: updatedNodes, edges: updatedEdges } = convertToReactFlowFormat(agencyData);
    setNodes(updatedNodes);
    setEdges(updatedEdges);
    setWorkflowStats(calculateWorkflowStats(agencyData));

    toast({
      title: "Workflow Refreshed",
      description: "Agent positions and communication flows have been updated",
      variant: "success",
    });
  }, [agencyData, convertToReactFlowFormat, calculateWorkflowStats, setNodes, setEdges, toast]);

  const exportLayout = useCallback(() => {
    const flowData = reactFlowInstance.toObject();
    const dataStr = JSON.stringify(flowData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `workflow-${agencyData.name}-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: "Layout Exported",
      description: "Workflow layout has been exported as JSON",
      variant: "success",
    });
  }, [reactFlowInstance, agencyData.name]);

  // Control panel configuration
  const controls = {
    zoom_in: zoomIn,
    zoom_out: zoomOut,
    fit_to_screen: fitToScreen,
    center_view: centerView,
    toggle_animation: toggleAnimation,
    refresh_data: refreshData,
    export_layout: exportLayout,
    apply_layout: applyLayout,
    set_filter: (filter: any) => {
      setVisualizationConfig(prev => ({ ...prev, filter }));
    },
  };

  return (
    <div className={`relative w-full h-full bg-gray-900 rounded-lg overflow-hidden ${className}`}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onEdgeClick={onEdgeClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        attributionPosition="top-right"
        className="bg-gray-900"
      >
        <Background color="#374151" gap={20} />
        <Controls className="bg-gray-800/90 border-gray-700" />
        <MiniMap
          className="bg-gray-800/90 border-gray-700"
          nodeColor={(node) => {
            const agentNode = node as Node<AgentNodeData>;
            switch (agentNode.data.agent.state) {
              case AgentState.ACTIVE: return '#10b981';
              case AgentState.IDLE: return '#f59e0b';
              case AgentState.HIBERNATED: return '#6b7280';
              default: return '#ef4444';
            }
          }}
          maskColor="rgba(0, 0, 0, 0.8)"
        />

        {/* Control Panel */}
        <Panel position="top-left" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-4">
          <WorkflowControls
            controls={controls}
            config={visualizationConfig}
            stats={workflowStats}
            isAnimating={isAnimating}
            currentLayout={currentLayout}
            onConfigChange={setVisualizationConfig}
          />
        </Panel>

        {/* Info Panel */}
        {selectedNode && workflowStats && (
          <Panel position="top-right" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-4 max-w-sm">
            <div className="text-white">
              <h3 className="text-lg font-semibold mb-2">Agent Details</h3>
              {(() => {
                const selectedAgentData = nodes.find(n => n.id === selectedNode)?.data;
                if (!selectedAgentData) return null;

                const agent = selectedAgentData.agent;
                return (
                  <div className="space-y-2 text-sm">
                    <div><strong>Name:</strong> {agent.name}</div>
                    <div><strong>Type:</strong> {agent.agent_type}</div>
                    <div><strong>Tier:</strong> {agent.model_tier}</div>
                    <div><strong>State:</strong>
                      <span className={`ml-2 px-2 py-1 rounded text-xs ${
                        agent.state === AgentState.ACTIVE ? 'bg-green-500/20 text-green-400' :
                        agent.state === AgentState.IDLE ? 'bg-yellow-500/20 text-yellow-400' :
                        agent.state === AgentState.HIBERNATED ? 'bg-gray-500/20 text-gray-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {agent.state}
                      </span>
                    </div>
                    <div><strong>Tasks Completed:</strong> {agent.tasks_completed}</div>
                    <div><strong>Success Rate:</strong> {(agent.success_rate * 100).toFixed(1)}%</div>
                    <div><strong>Active Connections:</strong> {selectedAgentData.communication_stats?.active_connections || 0}</div>
                  </div>
                );
              })()}
            </div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
};

// Wrapper component with ReactFlowProvider
export const AgencyWorkflowVisualizerWithProvider: React.FC<AgencyWorkflowVisualizerProps> = (props) => (
  <ReactFlowProvider>
    <AgencyWorkflowVisualizer {...props} />
  </ReactFlowProvider>
);