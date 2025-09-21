/**
 * TypeScript definitions for Agent Handoff System
 *
 * These types correspond to the handoff models in the backend Python service.
 */

// Core handoff enums
export enum HandoffStrategy {
  SEQUENTIAL = 'sequential',
  COLLABORATIVE = 'collaborative',
  CONDITIONAL = 'conditional',
  PARALLEL = 'parallel',
  DELEGATION = 'delegation'
}

export enum HandoffTrigger {
  CAPABILITY_MISMATCH = 'capability_mismatch',
  PERFORMANCE_THRESHOLD = 'performance_threshold',
  CONTEXT_AWARE = 'context_aware',
  MANUAL_REQUEST = 'manual_request',
  LEARNING_RECOMMENDATION = 'learning_recommendation',
  ERROR_RECOVERY = 'error_recovery'
}

export enum HandoffStatus {
  PENDING = 'pending',
  INITIATED = 'initiated',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

// Capability system enums
export enum ExpertiseLevel {
  BEGINNER = 'beginner',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  EXPERT = 'expert'
}

export enum CapabilityCategory {
  CODING = 'coding',
  ANALYSIS = 'analysis',
  DESIGN = 'design',
  COMMUNICATION = 'communication',
  MANAGEMENT = 'management',
  RESEARCH = 'research',
  SECURITY = 'security',
  PERFORMANCE = 'performance',
  TESTING = 'testing',
  INTEGRATION = 'integration'
}

// Core capability definitions
export interface AgentCapability {
  capability_type: string;
  expertise_level: ExpertiseLevel;
  confidence_score: number; // 0.0 to 1.0
  performance_metrics: CapabilityPerformanceMetrics;
  last_used?: Date;
  validation_status: 'validated' | 'pending' | 'needs_improvement';
  capability_details: Record<string, any>;
}

export interface CapabilityPerformanceMetrics {
  tasks_completed: number;
  success_rate: number; // 0.0 to 1.0
  avg_completion_time: number;
  error_rate: number;
  user_satisfaction: number; // 0.0 to 1.0
}

// Handoff request and result models
export interface HandoffRequest {
  id: string;
  source_agent_id: string;
  target_agent_id: string;
  message: string;
  task_description: string;
  strategy: HandoffStrategy;
  trigger: HandoffTrigger;
  context_package_id?: string;
  confidence_score: number; // 0.0 to 1.0
  priority: number; // 1-5
  metadata: Record<string, any>;
  created_at: Date;
  initiated_at?: Date;
  completed_at?: Date;
}

export interface HandoffResult {
  handoff_id: string;
  status: HandoffStatus;
  source_agent_id: string;
  target_agent_id: string;
  response_content?: string;
  execution_time: number; // milliseconds
  error_message?: string;
  metrics: HandoffMetrics;
  context_package_id?: string;
  completed_at: Date;
}

export interface HandoffMetrics {
  context_transfer_time: number;
  agent_response_time: number;
  total_handoff_time: number;
  context_preservation_score: number; // 0.0 to 1.0
  success_indicators: Record<string, any>;
}

// Context preservation models
export interface ContextPackage {
  id: string;
  agent_id: string;
  task_id?: string;
  context_data: Record<string, any>;
  conversation_history: ContextMessage[];
  shared_knowledge: SharedKnowledge[];
  compression_ratio: number; // 0.0 to 1.0
  size_bytes: number;
  created_at: Date;
  expires_at?: Date;
  is_valid: boolean;
}

export interface ContextMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

// Analytics and learning models
export interface HandoffAnalytics {
  total_handoffs: number;
  success_rate: number; // 0.0 to 1.0
  strategy_performance: Record<HandoffStrategy, StrategyPerformance>;
  agent_performance: Record<string, AgentHandoffPerformance>;
  learning_insights: LearningInsights;
  time_range_hours: number;
  generated_at: Date;
}

export interface StrategyPerformance {
  usage_count: number;
  success_rate: number;
  avg_execution_time: number;
  avg_confidence_score: number;
  best_for_scenarios: string[];
}

export interface AgentHandoffPerformance {
  agent_id: string;
  handoffs_initiated: number;
  handoffs_received: number;
  success_rate_initiated: number;
  success_rate_received: number;
  avg_response_time: number;
  preferred_strategies: HandoffStrategy[];
  expertise_areas: string[];
}

export interface LearningInsights {
  improved_patterns: ImprovedPattern[];
  optimization_opportunities: OptimizationOpportunity[];
  capability_gaps: CapabilityGap[];
  confidence_improvements: ConfidenceImprovement[];
}

export interface ImprovedPattern {
  pattern_id: string;
  description: string;
  success_rate_improvement: number;
  confidence_score: number;
  occurrence_count: number;
}

export interface OptimizationOpportunity {
  opportunity_id: string;
  description: string;
  potential_impact: number; // 0.0 to 1.0
  implementation_complexity: number; // 1-5
  estimated_benefit: string;
}

export interface CapabilityGap {
  capability_type: string;
  gap_severity: number; // 0.0 to 1.0
  affected_agents: string[];
  recommended_training: string[];
  priority: number; // 1-5
}

export interface ConfidenceImprovement {
  factor: string;
  improvement_factor: number; // percentage
  current_confidence: number;
  target_confidence: number;
  timeframe: string;
}

// Handoff recommendation models
export interface HandoffRecommendation {
  task_id: string;
  task_description: string;
  current_agent_id: string;
  recommended: boolean;
  recommended_agents: AgentRecommendation[];
  strategy_recommendation?: StrategyRecommendation;
  required_capabilities: string[];
  reasoning: string;
  confidence_score: number;
  generated_at: Date;
}

export interface AgentRecommendation {
  agent_id: string;
  agent_name: string;
  match_score: number; // 0.0 to 1.0
  expertise_score: number;
  performance_score: number;
  availability_score: number;
  load_balance_score: number;
  reasoning: string;
  estimated_response_time: number;
}

export interface StrategyRecommendation {
  strategy: HandoffStrategy;
  confidence_score: number;
  reasoning: string;
  expected_improvement: number;
  risk_factors: string[];
}

// Real-time visualization models
export interface HandoffVisualization {
  active_handoffs: ActiveHandoff[];
  handoff_history: HandoffHistoryEntry[];
  agent_states: AgentHandoffState[];
  performance_metrics: VisualizationMetrics;
  last_updated: Date;
}

export interface ActiveHandoff {
  handoff_id: string;
  source_agent: string;
  target_agent: string;
  status: HandoffStatus;
  progress: number; // 0-100
  strategy: HandoffStrategy;
  start_time: Date;
  estimated_completion?: Date;
  confidence_score: number;
  task_description: string;
}

export interface HandoffHistoryEntry {
  handoff_id: string;
  source_agent: string;
  target_agent: string;
  status: HandoffStatus;
  strategy: HandoffStrategy;
  duration: number;
  success: boolean;
  timestamp: Date;
  task_summary: string;
}

export interface AgentHandoffState {
  agent_id: string;
  agent_name: string;
  agent_type: string;
  current_status: 'available' | 'busy' | 'handing_off' | 'receiving' | 'offline';
  current_handoff_id?: string;
  handoff_stats: {
    initiated_today: number;
    received_today: number;
    success_rate: number;
    avg_response_time: number;
  };
  capabilities: string[];
  load_factor: number; // 0.0 to 1.0
}

export interface VisualizationMetrics {
  total_handoffs_today: number;
  success_rate_today: number;
  avg_handoff_time: number;
  most_used_strategy: HandoffStrategy;
  most_active_agent: string;
  confidence_trend: number; // -1 to 1, negative = declining
  performance_score: number; // 0.0 to 1.0
}

// UI Component props
export interface HandoffVisualizationProps {
  projectId?: string;
  refreshInterval?: number;
  maxHistoryItems?: number;
  showMetrics?: boolean;
  showControls?: boolean;
  onHandoffSelect?: (handoff: ActiveHandoff) => void;
  onAgentSelect?: (agent: AgentHandoffState) => void;
}

export interface HandoffRequestFormProps {
  projectId?: string;
  sourceAgentId?: string;
  onHandoffRequest: (request: HandoffRequest) => void;
  availableAgents: AgentHandoffState[];
  predefinedTasks?: PredefinedTask[];
}

export interface PredefinedTask {
  id: string;
  title: string;
  description: string;
  required_capabilities: string[];
  recommended_strategy: HandoffStrategy;
  estimated_complexity: number; // 1-5
}

export interface HandoffAnalyticsViewProps {
  projectId?: string;
  timeRange: number; // hours
  showAdvancedMetrics?: boolean;
  onExportData?: (format: 'csv' | 'json') => void;
}

export interface CapabilityHeatmapProps {
  agents: AgentHandoffState[];
  capabilities: string[];
  projectId?: string;
  showLabels?: boolean;
  onCapabilityClick?: (capability: string, agent: string) => void;
}

// Service types
export interface HandoffService {
  getActiveHandoffs(projectId?: string): Promise<ActiveHandoff[]>;
  getHandoffHistory(projectId?: string, limit?: number): Promise<HandoffHistoryEntry[]>;
  getHandoffAnalytics(projectId?: string, timeRange?: number): Promise<HandoffAnalytics>;
  requestHandoff(request: Omit<HandoffRequest, 'id' | 'created_at'>): Promise<HandoffResult>;
  getHandoffRecommendations(taskDescription: string, currentAgentId: string): Promise<HandoffRecommendation>;
  getAgentCapabilities(agentId: string): Promise<AgentCapability[]>;
  cancelHandoff(handoffId: string): Promise<boolean>;
}

// Event types for real-time updates
export interface HandoffEvent {
  event_type: 'handoff_initiated' | 'handoff_progress' | 'handoff_completed' | 'handoff_failed';
  handoff_id: string;
  data: Record<string, any>;
  timestamp: Date;
  project_id?: string;
}

// Chart data types for visualizations
export interface ChartDataPoint {
  timestamp: Date;
  value: number;
  label?: string;
}

export interface HandoffTrendData {
  period: string; // 'hour' | 'day' | 'week'
  data: ChartDataPoint[];
  success_rate: ChartDataPoint[];
  execution_time: ChartDataPoint[];
}

export interface AgentCapabilityMatrix {
  agent_id: string;
  capabilities: Record<string, number>; // capability -> score (0.0 to 1.0)
  total_capability_score: number;
  expertise_summary: Record<string, ExpertiseLevel>;
}

// Utility types
export interface HandoffFilterOptions {
  status?: HandoffStatus[];
  strategy?: HandoffStrategy[];
  agentIds?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  confidenceRange?: {
    min: number;
    max: number;
  };
}

export interface HandoffSortOptions {
  field: 'timestamp' | 'duration' | 'confidence' | 'success';
  direction: 'asc' | 'desc';
}

// Export constants
export const HANDOFF_STRATEGY_LABELS: Record<HandoffStrategy, string> = {
  [HandoffStrategy.SEQUENTIAL]: 'Sequential',
  [HandoffStrategy.COLLABORATIVE]: 'Collaborative',
  [HandoffStrategy.CONDITIONAL]: 'Conditional',
  [HandoffStrategy.PARALLEL]: 'Parallel',
  [HandoffStrategy.DELEGATION]: 'Delegation'
};

export const HANDOFF_STRATEGY_DESCRIPTIONS: Record<HandoffStrategy, string> = {
  [HandoffStrategy.SEQUENTIAL]: 'Agents work in sequence, one after another',
  [HandoffStrategy.COLLABORATIVE]: 'Agents work together on the same task',
  [HandoffStrategy.CONDITIONAL]: 'Handoff based on specific conditions',
  [HandoffStrategy.PARALLEL]: 'Multiple agents work concurrently',
  [HandoffStrategy.DELEGATION]: 'Complete task transfer to another agent'
};

export const EXPERTISE_LEVEL_COLORS: Record<ExpertiseLevel, string> = {
  [ExpertiseLevel.BEGINNER]: '#94a3b8', // slate
  [ExpertiseLevel.INTERMEDIATE]: '#3b82f6', // blue
  [ExpertiseLevel.ADVANCED]: '#10b981', // emerald
  [ExpertiseLevel.EXPERT]: '#f59e0b' // amber
};

export const HANDOFF_STATUS_COLORS: Record<HandoffStatus, string> = {
  [HandoffStatus.PENDING]: '#64748b', // slate
  [HandoffStatus.INITIATED]: '#3b82f6', // blue
  [HandoffStatus.IN_PROGRESS]: '#f59e0b', // amber
  [HandoffStatus.COMPLETED]: '#10b981', // emerald
  [HandoffStatus.FAILED]: '#ef4444', // red
  [HandoffStatus.CANCELLED]: '#6b7280' // gray
};

// Type guards
export const isValidHandoffStrategy = (strategy: string): strategy is HandoffStrategy => {
  return Object.values(HandoffStrategy).includes(strategy as HandoffStrategy);
};

export const isValidHandoffStatus = (status: string): status is HandoffStatus => {
  return Object.values(HandoffStatus).includes(status as HandoffStatus);
};

export const isValidExpertiseLevel = (level: string): level is ExpertiseLevel => {
  return Object.values(ExpertiseLevel).includes(level as ExpertiseLevel);
};