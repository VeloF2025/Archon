/**
 * TypeScript definitions for Archon 3.0 Intelligence-Tiered Agent Management
 * 
 * These types correspond to the Pydantic models in the backend Python service.
 */

// Core enums
export enum AgentState {
  CREATED = 'CREATED',
  ACTIVE = 'ACTIVE', 
  IDLE = 'IDLE',
  HIBERNATED = 'HIBERNATED',
  ARCHIVED = 'ARCHIVED'
}

export enum ModelTier {
  OPUS = 'OPUS',
  SONNET = 'SONNET',
  HAIKU = 'HAIKU'
}

export enum AgentType {
  CODE_IMPLEMENTER = 'CODE_IMPLEMENTER',
  SYSTEM_ARCHITECT = 'SYSTEM_ARCHITECT',
  CODE_QUALITY_REVIEWER = 'CODE_QUALITY_REVIEWER',
  TEST_COVERAGE_VALIDATOR = 'TEST_COVERAGE_VALIDATOR',
  SECURITY_AUDITOR = 'SECURITY_AUDITOR',
  PERFORMANCE_OPTIMIZER = 'PERFORMANCE_OPTIMIZER',
  DEPLOYMENT_AUTOMATION = 'DEPLOYMENT_AUTOMATION',
  ANTIHALLUCINATION_VALIDATOR = 'ANTIHALLUCINATION_VALIDATOR',
  UI_UX_OPTIMIZER = 'UI_UX_OPTIMIZER',
  DATABASE_ARCHITECT = 'DATABASE_ARCHITECT',
  DOCUMENTATION_GENERATOR = 'DOCUMENTATION_GENERATOR',
  CODE_REFACTORING_OPTIMIZER = 'CODE_REFACTORING_OPTIMIZER',
  STRATEGIC_PLANNER = 'STRATEGIC_PLANNER',
  API_DESIGN_ARCHITECT = 'API_DESIGN_ARCHITECT',
  GENERAL_PURPOSE = 'GENERAL_PURPOSE'
}

// Core Agent model
export interface AgentV3 {
  id: string;
  name: string;
  agent_type: AgentType;
  model_tier: ModelTier;
  project_id: string;
  state: AgentState;
  state_changed_at: Date;
  tasks_completed: number;
  success_rate: number; // 0.0 to 1.0
  avg_completion_time_seconds: number;
  last_active_at?: Date;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  capabilities: Record<string, any>;
  rules_profile_id?: string;
  created_at: Date;
  updated_at: Date;
}

// Agent state history
export interface AgentStateHistory {
  id: string;
  agent_id: string;
  from_state?: AgentState;
  to_state: AgentState;
  reason?: string;
  metadata: Record<string, any>;
  changed_at: Date;
  changed_by: string;
}

// Agent pool management
export interface AgentPool {
  id: string;
  project_id: string;
  opus_limit: number;
  sonnet_limit: number;
  haiku_limit: number;
  opus_active: number;
  sonnet_active: number;
  haiku_active: number;
  auto_scaling_enabled: boolean;
  hibernation_timeout_minutes: number;
  created_at: Date;
  updated_at: Date;
}

// Task complexity assessment
export interface TaskComplexity {
  id: string;
  task_id: string;
  technical_complexity: number; // 0.0 to 1.0
  domain_expertise_required: number;
  code_volume_complexity: number;
  integration_complexity: number;
  overall_complexity?: number; // computed
  recommended_tier: ModelTier;
  assigned_tier: ModelTier;
  tier_justification?: string;
  assessed_by: string;
  assessed_at: Date;
}

// Intelligence routing rules
export interface RoutingRule {
  id: string;
  rule_name: string;
  rule_description?: string;
  opus_threshold: number; // 0.0 to 1.0
  sonnet_threshold: number;
  haiku_threshold: number;
  agent_type_preferences: Record<string, string>;
  project_tier_override: Record<string, string>;
  is_active: boolean;
  priority_order: number;
  created_at: Date;
  updated_at: Date;
}

// Knowledge management
export interface AgentKnowledge {
  id: string;
  agent_id: string;
  knowledge_type: string; // 'pattern', 'solution', 'error', 'context'
  title: string;
  content: string;
  confidence: number; // 0.0 to 1.0
  success_count: number;
  failure_count: number;
  last_used_at?: Date;
  context_tags: string[];
  project_id: string;
  task_context?: string;
  storage_layer: string; // 'temporary', 'working', 'long_term'
  embedding?: number[];
  created_at: Date;
  updated_at: Date;
}

export interface SharedKnowledge {
  id: string;
  knowledge_pattern: string;
  solution_approach: string;
  contributing_agents: string[];
  validation_count: number;
  success_rate: number; // 0.0 to 1.0
  applicable_agent_types: AgentType[];
  applicable_contexts: string[];
  embedding?: number[];
  is_verified: boolean;
  verification_threshold: number;
  created_at: Date;
  updated_at: Date;
}

// Cost tracking and optimization
export interface CostTracking {
  id: string;
  agent_id: string;
  project_id: string;
  task_id?: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens?: number; // computed
  input_cost: number; // in USD
  output_cost: number;
  total_cost?: number; // computed
  model_tier: ModelTier;
  model_name?: string;
  task_duration_seconds?: number;
  success: boolean;
  recorded_at: Date;
}

export interface BudgetConstraint {
  id: string;
  project_id: string;
  monthly_budget?: number; // in USD
  daily_budget?: number;
  per_task_budget?: number;
  warning_threshold: number; // percentage
  critical_threshold: number;
  current_monthly_spend: number;
  current_daily_spend: number;
  spend_reset_date: Date;
  auto_downgrade_enabled: boolean;
  emergency_stop_enabled: boolean;
  created_at: Date;
  updated_at: Date;
}

// Real-time collaboration
export interface SharedContext {
  id: string;
  task_id: string;
  project_id: string;
  context_name: string;
  discoveries: Record<string, any>[];
  blockers: Record<string, any>[];
  patterns: Record<string, any>[];
  participants: string[]; // agent IDs
  is_active: boolean;
  last_updated_by?: string;
  created_at: Date;
  updated_at: Date;
}

export interface BroadcastMessage {
  id: string;
  message_id: string;
  topic: string;
  content: Record<string, any>;
  message_type: string; // 'discovery', 'blocker', 'pattern', 'update'
  priority: number; // 1=low, 2=medium, 3=high, 4=critical
  sender_id?: string;
  target_agents: string[];
  target_topics: string[];
  delivered_count: number;
  acknowledgment_count: number;
  sent_at: Date;
  expires_at?: Date;
}

export interface TopicSubscription {
  id: string;
  agent_id: string;
  topic: string;
  priority_filter: number;
  content_filters: Record<string, any>;
  is_active: boolean;
  subscription_type: string;
  callback_endpoint?: string;
  callback_timeout_seconds: number;
  created_at: Date;
  updated_at: Date;
}

// Global rules integration
export interface RulesProfile {
  id: string;
  profile_name: string;
  agent_type?: AgentType;
  model_tier?: ModelTier;
  global_rules: string[];
  project_rules: string[];
  manifest_rules: string[];
  quality_gates: string[];
  security_rules: string[];
  performance_rules: string[];
  coding_standards: string[];
  rule_count: number;
  last_parsed_at?: Date;
  source_file_hashes: Record<string, string>;
  is_active: boolean;
  validation_status: string;
  created_at: Date;
  updated_at: Date;
}

export interface RuleViolation {
  id: string;
  agent_id: string;
  task_id?: string;
  rules_profile_id?: string;
  rule_name: string;
  rule_category?: string;
  violation_type: string; // 'WARNING', 'ERROR', 'CRITICAL'
  description: string;
  status: string; // 'open', 'resolved', 'acknowledged', 'suppressed'
  resolved_at?: Date;
  resolved_by?: string;
  resolution_notes?: string;
  detected_at: Date;
}

// API Request/Response models
export interface CreateAgentRequest {
  name: string;
  agent_type: AgentType;
  model_tier: ModelTier;
  project_id?: string;
  capabilities?: Record<string, any>;
}

export interface UpdateAgentRequest {
  name?: string;
  state?: AgentState;
  model_tier?: ModelTier;
  capabilities?: Record<string, any>;
}

// Analytics and dashboard models
export interface AgentPerformanceMetrics {
  agent_id: string;
  tasks_completed: number;
  success_rate: number;
  avg_completion_time_seconds: number;
  cost_last_30_days: number;
  knowledge_items_count: number;
  activity_level: string; // 'RECENT', 'TODAY', 'WEEK', 'INACTIVE'
}

export interface ProjectIntelligenceOverview {
  project_id: string;
  project_name: string;
  total_agents: number;
  active_agents: number;
  opus_agents: number;
  sonnet_agents: number;
  haiku_agents: number;
  avg_success_rate: number;
  total_tasks_completed: number;
  monthly_cost: number;
  monthly_budget: number;
  budget_utilization_percent: number;
  active_shared_contexts: number;
  recent_broadcasts: number;
}

export interface CostOptimizationRecommendation {
  agent_id: string;
  agent_type: AgentType;
  current_tier: ModelTier;
  total_cost: number;
  success_rate: number;
  avg_cost_per_task: number;
  recommendation: string; // 'CONSIDER_SONNET', 'CONSIDER_OPUS', 'CONSIDER_HAIKU', 'OPTIMAL'
  potential_monthly_savings: number;
}

// Utility types
export interface TierPricing {
  input: number; // per 1M tokens
  output: number; // per 1M tokens
}

export interface BudgetStatus {
  status: string;
  within_limits: boolean;
  daily_budget?: number;
  monthly_budget?: number;
  daily_spend: number;
  monthly_spend: number;
  warning_threshold: number;
  critical_threshold: number;
  alerts: Array<{
    type: 'warning' | 'critical';
    message: string;
  }>;
}

// Component prop types
export interface AgentManagementContextType {
  agents: AgentV3[];
  loading: boolean;
  error: string | null;
  createAgent: (data: CreateAgentRequest) => Promise<AgentV3>;
  updateAgentState: (agentId: string, state: AgentState) => Promise<void>;
  hibernateIdleAgents: () => Promise<number>;
  getPerformanceMetrics: (agentId: string) => Promise<AgentPerformanceMetrics>;
  getProjectOverview: () => Promise<ProjectIntelligenceOverview>;
  getCostRecommendations: () => Promise<CostOptimizationRecommendation[]>;
}

// Complexity calculation utilities
export interface ComplexityFactors {
  technical: number; // 0.0 to 1.0
  domain_expertise: number;
  code_volume: number;
  integration: number;
}

export interface ComplexityResult {
  overall_score: number;
  recommended_tier: ModelTier;
  justification: string;
}

// Knowledge search types
export interface KnowledgeSearchQuery {
  query: string;
  agent_id?: string;
  knowledge_type?: string;
  storage_layer?: string;
  min_confidence?: number;
  limit?: number;
}

export interface KnowledgeSearchResult {
  knowledge: AgentKnowledge;
  similarity_score: number;
}

// Collaboration types
export interface CollaborationEvent {
  event_type: 'discovery' | 'blocker' | 'pattern' | 'message';
  agent_id: string;
  agent_name: string;
  content: Record<string, any>;
  timestamp: Date;
  context_id?: string;
}

// Agent spawn/creation types
export interface AgentSpawnConfig {
  agent_type: AgentType;
  model_tier?: ModelTier;
  capabilities?: Record<string, any>;
  auto_activate?: boolean;
  task_context?: string;
}

export interface AgentSpawnResult {
  agent: AgentV3;
  spawn_time_ms: number;
  initial_state: AgentState;
  assigned_resources: {
    memory_allocated_mb: number;
    cpu_allocated_percent: number;
  };
}

// Export utility constants
export const TIER_PRICING: Record<ModelTier, TierPricing> = {
  [ModelTier.OPUS]: { input: 15.00, output: 75.00 },
  [ModelTier.SONNET]: { input: 3.00, output: 15.00 },
  [ModelTier.HAIKU]: { input: 0.25, output: 1.25 }
};

export const AGENT_TYPE_ICONS: Record<AgentType, string> = {
  [AgentType.CODE_IMPLEMENTER]: '⚡',
  [AgentType.SYSTEM_ARCHITECT]: '🏗️',
  [AgentType.CODE_QUALITY_REVIEWER]: '🔍',
  [AgentType.TEST_COVERAGE_VALIDATOR]: '🧪',
  [AgentType.SECURITY_AUDITOR]: '🛡️',
  [AgentType.PERFORMANCE_OPTIMIZER]: '🚀',
  [AgentType.DEPLOYMENT_AUTOMATION]: '🚢',
  [AgentType.ANTIHALLUCINATION_VALIDATOR]: '🎯',
  [AgentType.UI_UX_OPTIMIZER]: '🎨',
  [AgentType.DATABASE_ARCHITECT]: '🗄️',
  [AgentType.DOCUMENTATION_GENERATOR]: '📝',
  [AgentType.CODE_REFACTORING_OPTIMIZER]: '🔧',
  [AgentType.STRATEGIC_PLANNER]: '📋',
  [AgentType.API_DESIGN_ARCHITECT]: '🌐',
  [AgentType.GENERAL_PURPOSE]: '🤖'
};

export const DEFAULT_COMPLEXITY_THRESHOLDS = {
  opus: 0.75,   // Only truly complex tasks
  sonnet: 0.15, // Default for most tasks (Sonnet-first)
  haiku: 0.0    // Only most basic tasks
};

export const DEFAULT_AGENT_POOL_LIMITS = {
  opus: 2,      // Limited for cost control
  sonnet: 10,   // Primary workhorse tier
  haiku: 50     // High volume basic tasks
};

// Type guards
export const isValidAgentState = (state: string): state is AgentState => {
  return Object.values(AgentState).includes(state as AgentState);
};

export const isValidModelTier = (tier: string): tier is ModelTier => {
  return Object.values(ModelTier).includes(tier as ModelTier);
};

export const isValidAgentType = (type: string): type is AgentType => {
  return Object.values(AgentType).includes(type as AgentType);
};