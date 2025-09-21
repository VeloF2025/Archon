/**
 * TypeScript definitions for ReactFlow workflow visualization
 * These types integrate with the existing Archon agent management system
 */

import { AgentV3, AgentState, ModelTier, AgentType } from './agentTypes';
import { Node, Edge, Position, XYPosition } from '@xyflow/react';

// Communication types between agents
export enum CommunicationType {
  DIRECT = 'direct',
  BROADCAST = 'broadcast',
  CHAIN = 'chain',
  COLLABORATIVE = 'collaborative',
  HIERARCHICAL = 'hierarchical'
}

export enum CommunicationStatus {
  ACTIVE = 'active',
  IDLE = 'idle',
  PENDING = 'pending',
  FAILED = 'failed',
  COMPLETED = 'completed'
}

// Communication flow data
export interface CommunicationFlow {
  id: string;
  source_agent_id: string;
  target_agent_id: string;
  communication_type: CommunicationType;
  status: CommunicationStatus;
  message_count: number;
  last_message_at?: Date;
  message_type: string; // 'task_assignment', 'status_update', 'data_request', 'result_delivery', etc.
  data_flow?: {
    input_size: number;
    output_size: number;
    processing_time_ms: number;
  };
  metadata?: Record<string, any>;
}

// Agency/Workflow data structure
export interface AgencyData {
  id: string;
  name: string;
  description?: string;
  agents: AgentV3[];
  communication_flows: CommunicationFlow[];
  workflow_rules?: {
    routing_rules: Record<string, any>;
    collaboration_patterns: Record<string, any>;
    escalation_paths: string[];
  };
  created_at: Date;
  updated_at: Date;
}

// ReactFlow node data for agents
export interface AgentNodeData {
  agent: AgentV3;
  position: XYPosition;
  is_selected?: boolean;
  is_highlighted?: boolean;
  communication_stats?: {
    messages_sent: number;
    messages_received: number;
    active_connections: number;
  };
}

// ReactFlow edge data for communications
export interface CommunicationEdgeData {
  communication: CommunicationFlow;
  is_animated?: boolean;
  is_highlighted?: boolean;
  message_flow?: {
    direction: 'source-to-target' | 'target-to-source' | 'bidirectional';
    intensity: number; // 0.0 to 1.0
  };
}

// Workflow visualization configuration
export interface WorkflowVisualizationConfig {
  auto_layout: boolean;
  show_labels: boolean;
  show_metrics: boolean;
  animation_speed: number; // milliseconds
  node_size: 'small' | 'medium' | 'large';
  edge_style: 'curved' | 'straight' | 'stepped';
  theme: 'light' | 'dark';
  filter?: {
    agent_types?: AgentType[];
    agent_states?: AgentState[];
    communication_types?: CommunicationType[];
  };
}

// Real-time workflow events
export interface WorkflowEvent {
  event_id: string;
  event_type: 'agent_created' | 'agent_updated' | 'agent_removed' |
              'communication_started' | 'communication_updated' | 'communication_ended' |
              'workflow_created' | 'workflow_updated';
  timestamp: Date;
  data: Record<string, any>;
}

// Workflow statistics
export interface WorkflowStats {
  total_agents: number;
  active_agents: number;
  total_communications: number;
  active_communications: number;
  avg_messages_per_connection: number;
  busiest_agent?: {
    agent_id: string;
    message_count: number;
  };
  communication_type_distribution: Record<CommunicationType, number>;
  agent_type_distribution: Record<AgentType, number>;
}

// Control panel actions
export interface WorkflowControls {
  zoom_in: () => void;
  zoom_out: () => void;
  fit_to_screen: () => void;
  center_view: () => void;
  toggle_animation: () => void;
  refresh_data: () => void;
  export_layout: () => void;
  apply_layout: (layoutType: 'hierarchical' | 'circular' | 'force' | 'grid') => void;
  set_filter: (filter: WorkflowVisualizationConfig['filter']) => void;
}

// ReactFlow custom node types
export type AgentNodeType = 'agent' | 'orchestrator' | 'coordinator' | 'worker';

// ReactFlow custom edge types
export type CommunicationEdgeType = 'communication' | 'hierarchical' | 'collaborative';

// Layout algorithms
export type LayoutAlgorithm = 'hierarchical' | 'circular' | 'force' | 'grid' | 'organic';

// Layout result
export interface LayoutResult {
  nodes: Node<AgentNodeData>[];
  edges: Edge<CommunicationEdgeData>[];
  stats: {
    total_nodes: number;
    total_edges: number;
    layout_time_ms: number;
    crossings_reduced?: number;
  };
}

// Message animation data
export interface MessageAnimation {
  id: string;
  edge_id: string;
  progress: number; // 0.0 to 1.0
  message_type: string;
  timestamp: Date;
  data?: Record<string, any>;
}

// Workflow interaction events
export interface WorkflowInteraction {
  type: 'node_selected' | 'node_clicked' | 'edge_clicked' | 'canvas_clicked' | 'node_dragged';
  data: {
    node_id?: string;
    edge_id?: string;
    position?: XYPosition;
    timestamp: Date;
  };
}

// Export configuration
export interface ExportConfig {
  format: 'png' | 'svg' | 'json';
  include_data: boolean;
  include_stats: boolean;
  resolution?: number;
}

// Performance metrics for the visualization
export interface VisualizationMetrics {
  fps: number;
  node_count: number;
  edge_count: number;
  render_time_ms: number;
  memory_usage_mb: number;
  animation_frame_count: number;
}

// Agent position in workflow (for layout calculations)
export interface AgentPosition {
  agent_id: string;
  position: XYPosition;
  layer?: number; // For hierarchical layouts
  group?: string; // For grouping agents
  rank?: number; // For ordering within layers
}

// Communication pattern types for visualization
export interface CommunicationPattern {
  id: string;
  name: string;
  description: string;
  pattern_type: 'star' | 'mesh' | 'ring' | 'tree' | 'line' | 'custom';
  participant_ids: string[];
  communication_rules: Record<string, any>;
}

// ReactFlow node extensions
export interface ExtendedAgentNode extends Node<AgentNodeData> {
  type: AgentNodeType;
  position: XYPosition;
  data: AgentNodeData;
  style?: React.CSSProperties;
  className?: string;
  sourcePosition?: Position;
  targetPosition?: Position;
}

// ReactFlow edge extensions
export interface ExtendedCommunicationEdge extends Edge<CommunicationEdgeData> {
  type: CommunicationEdgeType;
  data: CommunicationEdgeData;
  style?: React.CSSProperties;
  className?: string;
  animated?: boolean;
  markerEnd?: {
    type: string;
    color?: string;
  };
}

// Workflow session state
export interface WorkflowSession {
  id: string;
  agency_id: string;
  config: WorkflowVisualizationConfig;
  current_layout: LayoutAlgorithm;
  is_animating: boolean;
  selected_node_ids: string[];
  highlighted_edge_ids: string[];
  message_animations: MessageAnimation[];
  stats: WorkflowStats;
  last_updated: Date;
}

// Workflow Editor Types
// ===================

// Drag and drop types
export interface DraggedAgent {
  agent_type: AgentType;
  model_tier: ModelTier;
  agent_name: string;
  description: string;
  capabilities: Record<string, any>;
  default_config: AgentNodeData['agent'];
}

// Editor modes
export enum EditorMode {
  SELECT = 'select',
  CREATE_AGENT = 'create_agent',
  CREATE_CONNECTION = 'create_connection',
  PAN = 'pan',
  DELETE = 'delete'
}

// Connection creation types
export enum ConnectionType {
  DIRECT = 'direct',
  BROADCAST = 'broadcast',
  CHAIN = 'chain',
  COLLABORATIVE = 'collaborative'
}

export interface ConnectionCreationState {
  isCreating: boolean;
  sourceNodeId?: string;
  targetNodeId?: string;
  connectionType: ConnectionType;
  tempPosition?: { x: number; y: number };
}

// Template types
export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: 'development' | 'testing' | 'deployment' | 'collaboration' | 'analysis';
  agents: Array<{
    agent_type: AgentType;
    model_tier: ModelTier;
    name: string;
    capabilities: Record<string, any>;
    position: { x: number; y: number };
  }>;
  connections: Array<{
    source_index: number;
    target_index: number;
    communication_type: CommunicationType;
    message_type: string;
  }>;
  metadata: {
    created_by: string;
    created_at: Date;
    usage_count: number;
    rating: number;
    tags: string[];
  };
}

// Property editor types
export interface AgentProperties {
  id: string;
  name: string;
  agent_type: AgentType;
  model_tier: ModelTier;
  state: AgentState;
  capabilities: Record<string, any>;
  rules_profile_id?: string;
  position: { x: number; y: number };
}

export interface ConnectionProperties {
  id: string;
  source_agent_id: string;
  target_agent_id: string;
  communication_type: CommunicationType;
  message_type: string;
  data_flow?: {
    input_size: number;
    output_size: number;
    processing_time_ms: number;
  };
  metadata?: Record<string, any>;
}

export type SelectedElement =
  | { type: 'agent'; data: AgentProperties }
  | { type: 'connection'; data: ConnectionProperties }
  | { type: 'none' };

// Validation types
export interface ValidationError {
  id: string;
  type: 'error' | 'warning';
  element_type: 'agent' | 'connection' | 'workflow';
  element_id?: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  auto_fixable?: boolean;
  fix_suggestion?: string;
}

export interface ValidationResult {
  is_valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  score: number; // 0.0 to 1.0
  can_execute: boolean;
}

// Persistence types
export interface WorkflowConfiguration {
  id: string;
  name: string;
  description?: string;
  version: string;
  created_at: Date;
  updated_at: Date;
  nodes: ExtendedAgentNode[];
  edges: ExtendedCommunicationEdge[];
  metadata: {
    author: string;
    project_id?: string;
    tags: string[];
    is_template: boolean;
    execution_count: number;
    last_executed?: Date;
  };
}

export interface WorkflowExportOptions {
  format: 'json' | 'yaml' | 'xml';
  include_metadata: boolean;
  include_execution_history: boolean;
  redact_sensitive_data: boolean;
  compatibility_version?: string;
}

// Editor state
export interface EditorState {
  mode: EditorMode;
  selectedElement: SelectedElement;
  connectionState: ConnectionCreationState;
  isDragging: boolean;
  draggedAgent?: DraggedAgent;
  validationErrors: ValidationError[];
  unsavedChanges: boolean;
  zoom: number;
  pan: { x: number; y: number };
  isGridEnabled: boolean;
  snapToGrid: boolean;
  gridSize: number;
}

// History management (undo/redo)
export interface HistoryState {
  past: WorkflowConfiguration[];
  present: WorkflowConfiguration;
  future: WorkflowConfiguration[];
  currentIndex: number;
}

// Keyboard shortcuts
export interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  action: string;
  description: string;
}

// Performance metrics
export interface EditorPerformanceMetrics {
  fps: number;
  node_count: number;
  edge_count: number;
  render_time_ms: number;
  memory_usage_mb: number;
  interaction_response_time_ms: number;
}

// Settings
export interface WorkflowEditorSettings {
  auto_save: boolean;
  auto_save_interval: number; // seconds
  show_grid: boolean;
  snap_to_grid: boolean;
  grid_size: number;
  animation_enabled: boolean;
  keyboard_shortcuts: KeyboardShortcut[];
  validation_mode: 'realtime' | 'on_demand' | 'disabled';
  performance_mode: 'quality' | 'balanced' | 'performance';
}

// Execution preview
export interface ExecutionPreview {
  id: string;
  workflow_configuration: WorkflowConfiguration;
  preview_agents: string[]; // agent IDs to preview
  simulation_speed: number; // 0.1 to 10.0
  is_running: boolean;
  current_step: number;
  total_steps: number;
  execution_log: Array<{
    timestamp: Date;
    agent_id: string;
    action: string;
    result: 'success' | 'failed' | 'pending';
    duration_ms: number;
    message?: string;
  }>;
}

// Import/Export validation
export interface ImportValidationResult {
  is_valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  compatibility_issues: string[];
  estimated_migration_time?: number;
}

// Utility types
export type NodeId = string;
export type EdgeId = string;

// Event handlers for workflow interactions
export interface WorkflowEventHandlers {
  onNodeClick?: (node: ExtendedAgentNode, event: React.MouseEvent) => void;
  onNodeSelect?: (nodes: ExtendedAgentNode[]) => void;
  onEdgeClick?: (edge: ExtendedCommunicationEdge, event: React.MouseEvent) => void;
  onCanvasClick?: (event: React.MouseEvent) => void;
  onNodeDrag?: (node: ExtendedAgentNode, event: React.MouseEvent) => void;
  onNodeDragStop?: (node: ExtendedAgentNode, event: React.MouseEvent) => void;
  onPaneScroll?: (event: React.WheelEvent) => void;
  onPaneClick?: (event: React.MouseEvent) => void;
}

// Filter options
export interface WorkflowFilterOptions {
  agentStates?: AgentState[];
  agentTypes?: AgentType[];
  modelTiers?: ModelTier[];
  communicationTypes?: CommunicationType[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  searchTerm?: string;
  activityThreshold?: number; // minimum activity level
}

// Search and filter results
export interface FilterResult {
  nodes: ExtendedAgentNode[];
  edges: ExtendedCommunicationEdge[];
  total_filtered: number;
  search_term?: string;
  applied_filters: WorkflowFilterOptions;
}