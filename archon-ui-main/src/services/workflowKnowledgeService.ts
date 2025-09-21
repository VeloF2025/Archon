/**
 * Workflow Knowledge Integration Service
 *
 * Connects workflow execution with knowledge management to enable
 * automatic knowledge capture, contextual search, and pattern-based optimization.
 */

// Types for workflow knowledge integration
export interface WorkflowKnowledgeSession {
  session_id: string;
  workflow_id: string;
  project_id: string;
  capture_config: KnowledgeCaptureConfig;
  context_tags: string[];
  started_at: string;
  status: 'active' | 'completed' | 'error';
}

export interface KnowledgeCaptureConfig {
  auto_capture: boolean;
  capture_insights: boolean;
  capture_patterns: boolean;
  capture_errors: boolean;
  capture_successes: boolean;
  real_time_analysis: boolean;
  embedding_generation: boolean;
}

export interface WorkflowInsight {
  insight_id: string;
  session_id: string;
  insight_type: InsightType;
  insight_data: Record<string, any>;
  step_id?: string;
  execution_id?: string;
  importance_score: number;
  tags: string[];
  captured_at: string;
}

export type InsightType =
  | 'performance_optimization'
  | 'error_pattern'
  | 'success_pattern'
  | 'best_practice'
  | 'bottleneck_identified'
  | 'efficiency_gain'
  | 'cost_optimization'
  | 'quality_improvement'
  | 'risk_mitigation';

export interface ContextualKnowledge {
  knowledge_id: string;
  content: string;
  source: string;
  relevance_score: number;
  knowledge_type: string;
  metadata: Record<string, any>;
  created_at: string;
}

export interface WorkflowTemplate {
  template_id: string;
  name: string;
  description: string;
  category: string;
  flow_data: Record<string, any>;
  use_cases: string[];
  best_practices: string[];
  common_patterns: string[];
  tags: string[];
  metadata: TemplateMetadata;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

export interface TemplateMetadata {
  category: string;
  estimated_duration?: number;
  complexity_score: number;
  version: string;
  created_by: string;
  usage_count: number;
  success_rate: number;
  average_duration: number;
  rating: number;
}

export interface PerformanceMetrics {
  execution_id: string;
  step_metrics: Record<string, any>;
  overall_metrics: Record<string, any>;
  resource_usage: Record<string, any>;
  bottlenecks_identified: string[];
  optimization_opportunities: string[];
  captured_at: string;
}

export interface PerformanceInsight {
  insight_id: string;
  insight_type: string;
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  actionable: boolean;
  recommendation?: string;
  expected_improvement?: string;
  metadata: Record<string, any>;
}

export interface TemplateRecommendation {
  template_id: string;
  name: string;
  description: string;
  match_score: number;
  reasons: string[];
  category: string;
  complexity_score: number;
  estimated_duration?: number;
}

// API Configuration
import { API_BASE_URL } from '../config/api';

// Helper function for API requests
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || `HTTP ${response.status}`);
  }

  return response.json();
}

class WorkflowKnowledgeService {
  /**
   * Start a knowledge capture session for workflow execution
   */
  async startKnowledgeSession(
    workflowId: string,
    projectId: string,
    captureConfig: Partial<KnowledgeCaptureConfig> = {},
    contextTags: string[] = []
  ): Promise<WorkflowKnowledgeSession> {
    const defaultConfig: KnowledgeCaptureConfig = {
      auto_capture: true,
      capture_insights: true,
      capture_patterns: true,
      capture_errors: true,
      capture_successes: true,
      real_time_analysis: true,
      embedding_generation: true
    };

    const config = { ...defaultConfig, ...captureConfig };

    return apiRequest<WorkflowKnowledgeSession>('/api/workflow-knowledge/start-session', {
      method: 'POST',
      body: JSON.stringify({
        workflow_id: workflowId,
        project_id: projectId,
        capture_config: config,
        context_tags: contextTags
      })
    });
  }

  /**
   * Capture an insight during workflow execution
   */
  async captureInsight(
    sessionId: string,
    insightType: InsightType,
    insightData: Record<string, any>,
    options: {
      stepId?: string;
      executionId?: string;
      importanceScore?: number;
      tags?: string[];
    } = {}
  ): Promise<WorkflowInsight> {
    return apiRequest<WorkflowInsight>('/api/workflow-knowledge/capture-insight', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        insight_type: insightType,
        insight_data: insightData,
        step_id: options.stepId,
        execution_id: options.executionId,
        importance_score: options.importanceScore || 0.5,
        tags: options.tags || []
      })
    });
  }

  /**
   * Retrieve contextual knowledge for workflow execution
   */
  async getContextualKnowledge(
    sessionId: string,
    query: string,
    contextType: 'execution_context' | 'step_context' | 'project_context' | 'global_context' = 'execution_context',
    options: {
      maxResults?: number;
      similarityThreshold?: number;
    } = {}
  ): Promise<ContextualKnowledge[]> {
    return apiRequest<ContextualKnowledge[]>(`/api/workflow-knowledge/contextual/${sessionId}`, {
      method: 'POST',
      body: JSON.stringify({
        query,
        context_type: contextType,
        max_results: options.maxResults || 10,
        similarity_threshold: options.similarityThreshold || 0.7
      })
    });
  }

  /**
   * End a workflow knowledge session
   */
  async endKnowledgeSession(
    sessionId: string,
    options: {
      generateSummary?: boolean;
      extractPatterns?: boolean;
    } = {}
  ): Promise<{
    session_id: string;
    session_summary?: Record<string, any>;
    extracted_patterns?: any[];
    total_insights_captured: number;
    ended_at: string;
  }> {
    return apiRequest('/api/workflow-knowledge/end-session', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        generate_summary: options.generateSummary ?? true,
        extract_patterns: options.extractPatterns ?? true
      })
    });
  }

  /**
   * Store a workflow as a reusable template
   */
  async storeWorkflowTemplate(
    workflowId: string,
    templateName: string,
    templateDescription: string,
    options: {
      useCases?: string[];
      bestPractices?: string[];
      tags?: string[];
      isPublic?: boolean;
    } = {}
  ): Promise<{ template_id: string; message: string }> {
    return apiRequest('/api/workflow-knowledge/store-template', {
      method: 'POST',
      body: JSON.stringify({
        workflow_id: workflowId,
        template_name: templateName,
        template_description: templateDescription,
        use_cases: options.useCases || [],
        best_practices: options.bestPractices || [],
        tags: options.tags || [],
        is_public: options.isPublic || false
      })
    });
  }

  /**
   * Search workflow templates
   */
  async searchWorkflowTemplates(
    query: string,
    options: {
      projectId?: string;
      tags?: string[];
      limit?: number;
    } = {}
  ): Promise<WorkflowTemplate[]> {
    const params = new URLSearchParams();
    params.append('query', query);
    if (options.projectId) params.append('project_id', options.projectId);
    if (options.tags?.length) params.append('tags', options.tags.join(','));
    if (options.limit) params.append('limit', options.limit.toString());

    return apiRequest<WorkflowTemplate[]>(`/api/workflow-knowledge/templates/search?${params}`);
  }

  /**
   * Get template recommendations for a project
   */
  async getTemplateRecommendations(
    projectId: string,
    options: {
      workflowType?: string;
      complexityPreference?: 'low' | 'medium' | 'high';
      maxRecommendations?: number;
    } = {}
  ): Promise<TemplateRecommendation[]> {
    const params = new URLSearchParams();
    params.append('project_id', projectId);
    if (options.workflowType) params.append('workflow_type', options.workflowType);
    if (options.complexityPreference) params.append('complexity_preference', options.complexityPreference);
    if (options.maxRecommendations) params.append('max_recommendations', options.maxRecommendations.toString());

    return apiRequest<TemplateRecommendation[]>(`/api/workflow-knowledge/templates/recommendations?${params}`);
  }

  /**
   * Apply a template to create a new workflow
   */
  async applyTemplate(
    templateId: string,
    projectId: string,
    options: {
      workflowName?: string;
      workflowDescription?: string;
      customParameters?: Record<string, any>;
      overrideTags?: string[];
    } = {}
  ): Promise<{ workflow_id: string; workflow_name: string; message: string }> {
    return apiRequest('/api/workflow-knowledge/templates/apply', {
      method: 'POST',
      body: JSON.stringify({
        template_id: templateId,
        project_id: projectId,
        workflow_name: options.workflowName,
        workflow_description: options.workflowDescription,
        custom_parameters: options.customParameters || {},
        override_tags: options.overrideTags
      })
    });
  }

  /**
   * Capture performance metrics
   */
  async capturePerformanceMetrics(
    executionId: string,
    metrics: {
      stepMetrics: Record<string, any>;
      overallMetrics: Record<string, any>;
      resourceUsage?: Record<string, any>;
      bottlenecksIdentified?: string[];
      optimizationOpportunities?: string[];
    }
  ): Promise<{ metrics_id: string; message: string }> {
    return apiRequest('/api/workflow-knowledge/performance/capture', {
      method: 'POST',
      body: JSON.stringify({
        execution_id: executionId,
        step_metrics: metrics.stepMetrics,
        overall_metrics: metrics.overallMetrics,
        resource_usage: metrics.resourceUsage || {},
        bottlenecks_identified: metrics.bottlenecksIdentified || [],
        optimization_opportunities: metrics.optimizationOpportunities || []
      })
    });
  }

  /**
   * Generate performance insights
   */
  async generatePerformanceInsights(
    workflowId: string,
    options: {
      insightTypes?: string[];
      timePeriod?: { start: string; end: string };
      minConfidence?: number;
    } = {}
  ): Promise<PerformanceInsight[]> {
    return apiRequest<PerformanceInsight[]>('/api/workflow-knowledge/performance/insights', {
      method: 'POST',
      body: JSON.stringify({
        workflow_id: workflowId,
        insight_types: options.insightTypes || ['efficiency', 'cost', 'reliability'],
        time_period: options.timePeriod,
        min_confidence: options.minConfidence || 0.7
      })
    });
  }

  /**
   * Get workflow execution context with relevant knowledge
   */
  async getExecutionContext(
    executionId: string,
    options: {
      includeKnowledge?: boolean;
      maxContextItems?: number;
    } = {}
  ): Promise<{
    execution_context: Record<string, any>;
    relevant_knowledge?: ContextualKnowledge[];
  }> {
    const params = new URLSearchParams();
    params.append('execution_id', executionId);
    if (options.includeKnowledge !== undefined) params.append('include_knowledge', options.includeKnowledge.toString());
    if (options.maxContextItems) params.append('max_context_items', options.maxContextItems.toString());

    return apiRequest(`/api/workflow-knowledge/execution-context?${params}`);
  }

  /**
   * Track performance improvements
   */
  async trackPerformanceImprovements(
    workflowId: string,
    baselinePeriod: { start: string; end: string },
    improvementPeriod: { start: string; end: string },
    metricsToTrack: string[] = ['duration', 'cost', 'success_rate']
  ): Promise<{
    improvements: Record<string, any>;
    comparison_summary: string;
  }> {
    return apiRequest('/api/workflow-knowledge/performance/improvements', {
      method: 'POST',
      body: JSON.stringify({
        workflow_id: workflowId,
        baseline_period: baselinePeriod,
        improvement_period: improvementPeriod,
        metrics_to_track: metricsToTrack
      })
    });
  }
}

// Export singleton instance
export const workflowKnowledgeService = new WorkflowKnowledgeService();