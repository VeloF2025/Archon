import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { AgentNode } from './AgentNode';
import { CommunicationFlowEdge } from './CommunicationFlow';
import { AgentPalette } from './AgentPalette';
import { ConnectionTools } from './ConnectionTools';
import { PropertyEditor } from './PropertyEditor';
import { WorkflowTemplates } from './WorkflowTemplates';
import { WorkflowValidation } from './WorkflowValidation';
import { WorkflowPersistence } from './WorkflowPersistence';

import {
  AgencyData,
  AgentNodeData,
  CommunicationEdgeData,
  WorkflowVisualizationConfig,
  WorkflowStats,
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
  AgentState,
  ModelTier,
  AgentType,
  EditorMode,
  ConnectionType,
  ConnectionCreationState,
  DraggedAgent,
  SelectedElement,
  EditorState,
  HistoryState,
  WorkflowConfiguration,
  ValidationError,
} from '../../types/workflowTypes';

import { useSocket } from '../../hooks/useSocket';
import { useToast } from '../../hooks/useToast';

interface WorkflowEditorProps {
  agencyData?: AgencyData;
  config?: Partial<WorkflowVisualizationConfig>;
  onEvent?: (event: any) => void;
  className?: string;
  projectId?: string;
}

// Default configuration
const DEFAULT_CONFIG: WorkflowVisualizationConfig = {
  auto_layout: false,
  show_labels: true,
  show_metrics: true,
  animation_speed: 1000,
  node_size: 'medium',
  edge_style: 'curved',
  theme: 'dark',
};

export const WorkflowEditor: React.FC<WorkflowEditorProps> = ({
  agencyData,
  config = {},
  onEvent,
  className = '',
  projectId,
}) => {
  const { toast } = useToast();
  const socket = useSocket();
  const reactFlowInstance = useReactFlow();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // Merge config with defaults
  const [visualizationConfig, setVisualizationConfig] = useState<WorkflowVisualizationConfig>({
    ...DEFAULT_CONFIG,
    ...config,
  });

  // ReactFlow state
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Editor state
  const [editorState, setEditorState] = useState<EditorState>({
    mode: EditorMode.SELECT,
    selectedElement: { type: 'none' },
    connectionState: {
      isCreating: false,
      connectionType: ConnectionType.DIRECT,
    },
    isDragging: false,
    validationErrors: [],
    unsavedChanges: false,
    zoom: 1,
    pan: { x: 0, y: 0 },
    isGridEnabled: true,
    snapToGrid: true,
    gridSize: 20,
  });

  // History management
  const [history, setHistory] = useState<HistoryState>({
    past: [],
    present: {
      id: '',
      name: 'New Workflow',
      version: '1.0.0',
      created_at: new Date(),
      updated_at: new Date(),
      nodes: [],
      edges: [],
      metadata: {
        author: 'user',
        project_id: projectId,
        tags: [],
        is_template: false,
        execution_count: 0,
      },
    },
    future: [],
    currentIndex: -1,
  });

  // UI state
  const [showTemplates, setShowTemplates] = useState(false);
  const [showValidation, setShowValidation] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  // Custom node and edge types
  const nodeTypes: NodeTypes = useMemo(() => ({
    agent: AgentNode,
  }), []);

  const edgeTypes: EdgeTypes = useMemo(() => ({
    communication: CommunicationFlowEdge,
  }), []);

  // Save current state to history
  const saveToHistory = useCallback(() => {
    const currentConfig: WorkflowConfiguration = {
      ...history.present,
      nodes: nodes.map(n => ({ ...n })),
      edges: edges.map(e => ({ ...e })),
      updated_at: new Date(),
    };

    setHistory(prev => ({
      past: [...prev.past.slice(-49), prev.present], // Keep last 50 states
      present: currentConfig,
      future: [],
      currentIndex: prev.currentIndex + 1,
    }));

    setEditorState(prev => ({ ...prev, unsavedChanges: false }));
  }, [nodes, edges, history.present, projectId]);

  // Undo functionality
  const undo = useCallback(() => {
    if (history.currentIndex > 0) {
      const previousState = history.past[history.currentIndex - 1];
      setHistory(prev => ({
        past: prev.past.slice(0, prev.currentIndex - 1),
        present: previousState,
        future: [prev.present, ...prev.future],
        currentIndex: prev.currentIndex - 1,
      }));
      setNodes(previousState.nodes);
      setEdges(previousState.edges);
    }
  }, [history, setNodes, setEdges]);

  // Redo functionality
  const redo = useCallback(() => {
    if (history.future.length > 0) {
      const nextState = history.future[0];
      setHistory(prev => ({
        past: [...prev.past, prev.present],
        present: nextState,
        future: prev.future.slice(1),
        currentIndex: prev.currentIndex + 1,
      }));
      setNodes(nextState.nodes);
      setEdges(nextState.edges);
    }
  }, [history, setNodes, setEdges]);

  // Handle node drop
  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setIsDraggingOver(false);

      if (!reactFlowWrapper.current || !event.dataTransfer) return;

      const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const draggedAgentData = event.dataTransfer.getData('application/reactflow');
      if (!draggedAgentData) return;

      const draggedAgent: DraggedAgent = JSON.parse(draggedAgentData);

      // Create new agent node
      const newNode: ExtendedAgentNode = {
        id: `agent-${Date.now()}`,
        type: 'agent',
        position,
        data: {
          agent: {
            ...draggedAgent.default_config,
            id: `agent-${Date.now()}`,
            name: draggedAgent.agent_name,
            agent_type: draggedAgent.agent_type,
            model_tier: draggedAgent.model_tier,
            state: AgentState.CREATED,
            project_id: projectId || '',
            state_changed_at: new Date(),
            tasks_completed: 0,
            success_rate: 0,
            avg_completion_time_seconds: 0,
            memory_usage_mb: 0,
            cpu_usage_percent: 0,
            capabilities: draggedAgent.capabilities,
            created_at: new Date(),
            updated_at: new Date(),
          },
          position,
          is_selected: false,
          communication_stats: {
            messages_sent: 0,
            messages_received: 0,
            active_connections: 0,
          },
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
      };

      setNodes((nds) => nds.concat(newNode));
      setEditorState(prev => ({ ...prev, unsavedChanges: true }));

      toast({
        title: "Agent Added",
        description: `${draggedAgent.agent_name} has been added to the workflow`,
        variant: "success",
      });
    },
    [reactFlowInstance, projectId, setNodes, toast]
  );

  // Handle drag over
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    setIsDraggingOver(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDraggingOver(false);
  }, []);

  // ReactFlow event handlers
  const onConnect = useCallback(
    (params: Edge | Connection) => {
      const newEdge: ExtendedCommunicationEdge = {
        ...params,
        id: `edge-${Date.now()}`,
        type: 'communication',
        data: {
          communication: {
            id: `comm-${Date.now()}`,
            source_agent_id: params.source!,
            target_agent_id: params.target!,
            communication_type: CommunicationType.DIRECT,
            status: 'active' as any,
            message_count: 0,
            message_type: 'data_transfer',
          },
          is_animated: true,
          is_highlighted: false,
          message_flow: {
            direction: 'source-to-target',
            intensity: 0.5,
          },
        },
        animated: true,
        style: {
          strokeWidth: 2,
          stroke: '#10b981',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#10b981',
        },
      };

      setEdges((eds) => addEdge(newEdge as Edge, eds));
      setEditorState(prev => ({ ...prev, unsavedChanges: true }));
    },
    [setEdges]
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setNodes(prevNodes =>
      prevNodes.map(n => ({
        ...n,
        data: {
          ...n.data,
          is_selected: n.id === node.id,
        },
      }))
    );

    const agentNode = node as Node<AgentNodeData>;
    setEditorState(prev => ({
      ...prev,
      mode: EditorMode.SELECT,
      selectedElement: {
        type: 'agent',
        data: {
          id: agentNode.id,
          name: agentNode.data.agent.name,
          agent_type: agentNode.data.agent.agent_type,
          model_tier: agentNode.data.agent.model_tier,
          state: agentNode.data.agent.state,
          capabilities: agentNode.data.agent.capabilities,
          position: agentNode.position,
        },
      },
    }));

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

    const commEdge = edge as Edge<CommunicationEdgeData>;
    setEditorState(prev => ({
      ...prev,
      mode: EditorMode.SELECT,
      selectedElement: {
        type: 'connection',
        data: {
          id: commEdge.id,
          source_agent_id: commEdge.source!,
          target_agent_id: commEdge.target!,
          communication_type: commEdge.data.communication.communication_type,
          message_type: commEdge.data.communication.message_type,
          data_flow: commEdge.data.communication.data_flow,
          metadata: commEdge.data.communication.metadata,
        },
      },
    }));
  }, [setEdges]);

  const onPaneClick = useCallback(() => {
    setEditorState(prev => ({
      ...prev,
      mode: EditorMode.SELECT,
      selectedElement: { type: 'none' },
    }));
    setNodes(prevNodes =>
      prevNodes.map(n => ({
        ...n,
        data: {
          ...n.data,
          is_selected: false,
        },
      }))
    );
    setEdges(prevEdges =>
      prevEdges.map(e => ({
        ...e,
        data: {
          ...e.data,
          is_highlighted: false,
        },
      }))
    );
  }, [setNodes, setEdges]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
          case 'z':
            event.preventDefault();
            if (event.shiftKey) {
              redo();
            } else {
              undo();
            }
            break;
          case 's':
            event.preventDefault();
            saveToHistory();
            break;
          case 'Delete':
          case 'Backspace':
            event.preventDefault();
            if (editorState.selectedElement.type !== 'none') {
              // Handle deletion
              if (editorState.selectedElement.type === 'agent') {
                setNodes(prev => prev.filter(n => n.id !== editorState.selectedElement.data.id));
                setEdges(prev => prev.filter(e =>
                  e.source !== editorState.selectedElement.data.id &&
                  e.target !== editorState.selectedElement.data.id
                ));
              } else if (editorState.selectedElement.type === 'connection') {
                setEdges(prev => prev.filter(e => e.id !== editorState.selectedElement.data.id));
              }
              setEditorState(prev => ({ ...prev, selectedElement: { type: 'none' }, unsavedChanges: true }));
            }
            break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [editorState.selectedElement, undo, redo, saveToHistory, setNodes, setEdges]);

  // Auto-save functionality
  useEffect(() => {
    if (editorState.unsavedChanges) {
      const timer = setTimeout(() => {
        saveToHistory();
        toast({
          title: "Auto-saved",
          description: "Workflow has been auto-saved",
          variant: "success",
        });
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [editorState.unsavedChanges, saveToHistory, toast]);

  // Initialize from agency data if provided
  useEffect(() => {
    if (agencyData) {
      const existingNodes: ExtendedAgentNode[] = agencyData.agents.map((agent, index) => {
        const angle = (index / agencyData.agents.length) * 2 * Math.PI;
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
            is_selected: false,
            communication_stats: {
              messages_sent: agencyData.communication_flows.filter(f => f.source_agent_id === agent.id).length,
              messages_received: agencyData.communication_flows.filter(f => f.target_agent_id === agent.id).length,
              active_connections: agencyData.communication_flows.filter(
                f => (f.source_agent_id === agent.id || f.target_agent_id === agent.id) &&
                     f.status === 'active'
              ).length,
            },
          },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
        };
      });

      const existingEdges: ExtendedCommunicationEdge[] = agencyData.communication_flows.map((flow) => ({
        id: flow.id,
        type: 'communication',
        source: flow.source_agent_id,
        target: flow.target_agent_id,
        data: {
          communication: flow,
          is_animated: flow.status === 'active',
          is_highlighted: false,
          message_flow: {
            direction: 'bidirectional',
            intensity: Math.min(flow.message_count / 10, 1),
          },
        },
        animated: flow.status === 'active',
        style: {
          strokeWidth: Math.max(1, Math.min(flow.message_count / 5, 5)),
          stroke: flow.status === 'active' ? '#10b981' : '#6b7280',
        },
      }));

      setNodes(existingNodes);
      setEdges(existingEdges);

      // Update history
      setHistory(prev => ({
        ...prev,
        present: {
          ...prev.present,
          nodes: existingNodes,
          edges: existingEdges,
          name: agencyData.name || 'Imported Workflow',
        },
      }));
    }
  }, [agencyData, setNodes, setEdges]);

  // Tool panel functions
  const changeMode = useCallback((mode: EditorMode) => {
    setEditorState(prev => ({ ...prev, mode }));
  }, []);

  const clearCanvas = useCallback(() => {
    if (window.confirm('Are you sure you want to clear the entire workflow?')) {
      setNodes([]);
      setEdges([]);
      setEditorState(prev => ({ ...prev, unsavedChanges: true }));
      toast({
        title: "Canvas Cleared",
        description: "All agents and connections have been removed",
        variant: "info",
      });
    }
  }, [setNodes, setEdges, toast]);

  const applyTemplate = useCallback((templateId: string) => {
    // This will be implemented by the WorkflowTemplates component
    setShowTemplates(false);
    toast({
      title: "Template Applied",
      description: "Workflow template has been loaded",
      variant: "success",
    });
  }, [toast]);

  const exportWorkflow = useCallback(() => {
    const flowData = reactFlowInstance.toObject();
    const config: WorkflowConfiguration = {
      ...history.present,
      nodes: nodes.map(n => ({ ...n })),
      edges: edges.map(e => ({ ...e })),
    };

    const dataStr = JSON.stringify(config, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `workflow-${history.present.name}-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: "Workflow Exported",
      description: "Workflow has been exported as JSON",
      variant: "success",
    });
  }, [reactFlowInstance, history.present, nodes, edges]);

  return (
    <div className={`relative w-full h-full bg-gray-900 rounded-lg overflow-hidden ${className}`}>
      <div ref={reactFlowWrapper} className="w-full h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          attributionPosition="top-right"
          className="bg-gray-900"
          snapToGrid={editorState.snapToGrid}
          snapGrid={[editorState.gridSize, editorState.gridSize]}
        >
          <Background
            color={editorState.isGridEnabled ? "#374151" : "transparent"}
            gap={editorState.gridSize}
          />
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

          {/* Mode Indicator */}
          <Panel position="top-left" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-3">
            <div className="flex items-center space-x-2">
              <div className="text-white text-sm font-medium">Mode:</div>
              <div className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs">
                {editorState.mode.replace('_', ' ').toUpperCase()}
              </div>
              {editorState.unsavedChanges && (
                <div className="px-2 py-1 bg-yellow-500/20 text-yellow-300 rounded text-xs">
                  UNSAVED
                </div>
              )}
            </div>
          </Panel>

          {/* Tools Panel */}
          <Panel position="top-center" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-2">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => changeMode(EditorMode.SELECT)}
                className={`p-2 rounded transition-colors ${
                  editorState.mode === EditorMode.SELECT
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
                title="Select Mode (S)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2z" />
                </svg>
              </button>

              <button
                onClick={() => changeMode(EditorMode.CREATE_AGENT)}
                className={`p-2 rounded transition-colors ${
                  editorState.mode === EditorMode.CREATE_AGENT
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
                title="Add Agent (A)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </button>

              <button
                onClick={() => changeMode(EditorMode.CREATE_CONNECTION)}
                className={`p-2 rounded transition-colors ${
                  editorState.mode === EditorMode.CREATE_CONNECTION
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
                title="Create Connection (C)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
              </button>

              <div className="w-px h-6 bg-gray-600"></div>

              <button
                onClick={undo}
                disabled={history.currentIndex <= 0}
                className={`p-2 rounded transition-colors ${
                  history.currentIndex <= 0
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
                title="Undo (Ctrl+Z)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                </svg>
              </button>

              <button
                onClick={redo}
                disabled={history.future.length === 0}
                className={`p-2 rounded transition-colors ${
                  history.future.length === 0
                    ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
                title="Redo (Ctrl+Shift+Z)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10h-10a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
                </svg>
              </button>

              <div className="w-px h-6 bg-gray-600"></div>

              <button
                onClick={() => setShowTemplates(true)}
                className="p-2 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                title="Load Template"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
              </button>

              <button
                onClick={exportWorkflow}
                className="p-2 bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
                title="Export Workflow"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </button>

              <button
                onClick={clearCanvas}
                className="p-2 bg-red-700 text-red-300 rounded hover:bg-red-600 transition-colors"
                title="Clear Canvas"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          </Panel>

          {/* Agent Palette */}
          <Panel position="left" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-4 w-64">
            <AgentPalette
              onAgentDragStart={() => setIsDraggingOver(true)}
              onAgentDragEnd={() => setIsDraggingOver(false)}
            />
          </Panel>

          {/* Property Editor */}
          {editorState.selectedElement.type !== 'none' && (
            <Panel position="right" className="bg-gray-800/90 backdrop-blur-md border border-gray-700 rounded-lg p-4 w-80">
              <PropertyEditor
                selectedElement={editorState.selectedElement}
                onChange={(element) => {
                  if (element.type === 'agent') {
                    setNodes(prevNodes =>
                      prevNodes.map(n =>
                        n.id === element.data.id
                          ? {
                              ...n,
                              data: {
                                ...n.data,
                                agent: { ...n.data.agent, ...element.data },
                              },
                            }
                          : n
                      )
                    );
                  } else if (element.type === 'connection') {
                    setEdges(prevEdges =>
                      prevEdges.map(e =>
                        e.id === element.data.id
                          ? {
                              ...e,
                              data: {
                                ...e.data,
                                communication: { ...e.data.communication, ...element.data },
                              },
                            }
                          : e
                      )
                    );
                  }
                  setEditorState(prev => ({ ...prev, unsavedChanges: true }));
                }}
              />
            </Panel>
          )}

          {/* Drop zone indicator */}
          {isDraggingOver && (
            <Panel position="top-right" className="bg-blue-500/90 backdrop-blur-md border border-blue-400 rounded-lg p-3">
              <div className="text-white text-sm font-medium">
                Drop agent to add to workflow
              </div>
            </Panel>
          )}
        </ReactFlow>
      </div>

      {/* Modals */}
      {showTemplates && (
        <WorkflowTemplates
          onClose={() => setShowTemplates(false)}
          onTemplateSelect={applyTemplate}
          projectId={projectId}
        />
      )}

      {showValidation && (
        <WorkflowValidation
          nodes={nodes}
          edges={edges}
          onClose={() => setShowValidation(false)}
        />
      )}
    </div>
  );
};

// Wrapper component with ReactFlowProvider
export const WorkflowEditorWithProvider: React.FC<WorkflowEditorProps> = (props) => (
  <ReactFlowProvider>
    <WorkflowEditor {...props} />
  </ReactFlowProvider>
);