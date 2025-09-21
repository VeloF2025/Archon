/**
 * Handoff Service for Agent Handoff System
 *
 * Frontend service for communicating with the Agent Handoff API
 */

import {
  HandoffRequest,
  HandoffResult,
  HandoffAnalytics,
  HandoffRecommendation,
  ActiveHandoff,
  HandoffHistoryEntry,
  AgentCapability,
  HandoffStrategy,
  HandoffStatus,
  HandoffService,
  HandoffVisualization
} from '../types/handoffTypes';
import { API_BASE_URL } from '../config/api';

class HandoffServiceImpl implements HandoffService {
  private baseURL: string;

  constructor() {
    this.baseURL = `${API_BASE_URL}/handoff`;
  }

  // =====================================================
  // ACTIVE HANDOFF MANAGEMENT
  // =====================================================

  async getActiveHandoffs(projectId?: string): Promise<ActiveHandoff[]> {
    let url = `${this.baseURL}/active`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch active handoffs: ${response.statusText}`);
      }

      const data = await response.json();
      return data.active_handoffs.map(this.transformActiveHandoffFromAPI);
    } catch (error) {
      console.error('Error fetching active handoffs:', error);
      return [];
    }
  }

  async getHandoffHistory(projectId?: string, limit: number = 50): Promise<HandoffHistoryEntry[]> {
    let url = `${this.baseURL}/history?limit=${limit}`;
    if (projectId) {
      url += `&project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch handoff history: ${response.statusText}`);
      }

      const data = await response.json();
      return data.history.map(this.transformHandoffHistoryFromAPI);
    } catch (error) {
      console.error('Error fetching handoff history:', error);
      return [];
    }
  }

  // =====================================================
  // HANDOFF EXECUTION
  // =====================================================

  async requestHandoff(request: Omit<HandoffRequest, 'id' | 'created_at'>): Promise<HandoffResult> {
    const response = await fetch(`${this.baseURL}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to execute handoff: ${errorData.detail || response.statusText}`);
    }

    const data = await response.json();
    return this.transformHandoffResultFromAPI(data);
  }

  async cancelHandoff(handoffId: string): Promise<boolean> {
    const response = await fetch(`${this.baseURL}/${handoffId}/cancel`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel handoff: ${response.statusText}`);
    }

    const data = await response.json();
    return data.cancelled;
  }

  // =====================================================
  // RECOMMENDATIONS AND ANALYTICS
  // =====================================================

  async getHandoffRecommendations(
    taskDescription: string,
    currentAgentId: string
  ): Promise<HandoffRecommendation> {
    const response = await fetch(`${this.baseURL}/recommendations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        task_description: taskDescription,
        current_agent_id: currentAgentId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to get handoff recommendations: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformHandoffRecommendationFromAPI(data);
  }

  async getHandoffAnalytics(projectId?: string, timeRange: number = 24): Promise<HandoffAnalytics> {
    let url = `${this.baseURL}/analytics?time_range_hours=${timeRange}`;
    if (projectId) {
      url += `&project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch handoff analytics: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformHandoffAnalyticsFromAPI(data);
    } catch (error) {
      console.error('Error fetching handoff analytics:', error);
      return {
        total_handoffs: 0,
        success_rate: 0,
        strategy_performance: {},
        agent_performance: {},
        learning_insights: {
          improved_patterns: [],
          optimization_opportunities: [],
          capability_gaps: [],
          confidence_improvements: []
        },
        time_range_hours: timeRange,
        generated_at: new Date()
      };
    }
  }

  // =====================================================
  // AGENT CAPABILITIES
  // =====================================================

  async getAgentCapabilities(agentId: string): Promise<AgentCapability[]> {
    try {
      const response = await fetch(`${this.baseURL}/agents/${agentId}/capabilities`);
      if (!response.ok) {
        throw new Error(`Failed to fetch agent capabilities: ${response.statusText}`);
      }

      const data = await response.json();
      return data.capabilities.map(this.transformAgentCapabilityFromAPI);
    } catch (error) {
      console.error('Error fetching agent capabilities:', error);
      return [];
    }
  }

  async getAllAgentCapabilities(projectId?: string): Promise<Record<string, AgentCapability[]>> {
    let url = `${this.baseURL}/agents/capabilities`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch all agent capabilities: ${response.statusText}`);
      }

      const data = await response.json();
      const result: Record<string, AgentCapability[]> = {};

      for (const [agentId, capabilities] of Object.entries(data.agent_capabilities)) {
        result[agentId] = (capabilities as any[]).map(this.transformAgentCapabilityFromAPI);
      }

      return result;
    } catch (error) {
      console.error('Error fetching all agent capabilities:', error);
      return {};
    }
  }

  // =====================================================
  // REAL-TIME VISUALIZATION
  // =====================================================

  async getHandoffVisualization(projectId?: string): Promise<HandoffVisualization> {
    let url = `${this.baseURL}/visualization`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch handoff visualization: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformHandoffVisualizationFromAPI(data);
    } catch (error) {
      console.error('Error fetching handoff visualization:', error);
      return {
        active_handoffs: [],
        handoff_history: [],
        agent_states: [],
        performance_metrics: {
          total_handoffs_today: 0,
          success_rate_today: 0,
          avg_handoff_time: 0,
          most_used_strategy: HandoffStrategy.SEQUENTIAL,
          most_active_agent: '',
          confidence_trend: 0,
          performance_score: 0
        },
        last_updated: new Date()
      };
    }
  }

  // =====================================================
  // STRATEGY MANAGEMENT
  // =====================================================

  async getAvailableStrategies(): Promise<Record<HandoffStrategy, {
    description: string;
    best_for: string[];
    complexity: number;
  }>> {
    try {
      const response = await fetch(`${this.baseURL}/strategies`);
      if (!response.ok) {
        throw new Error(`Failed to fetch available strategies: ${response.statusText}`);
      }

      const data = await response.json();
      return data.strategies;
    } catch (error) {
      console.error('Error fetching available strategies:', error);
      return {};
    }
  }

  async getOptimalStrategy(
    taskComplexity: number,
    agentCount: number,
    timeConstraint?: number
  ): Promise<{
    strategy: HandoffStrategy;
    confidence_score: number;
    reasoning: string;
  }> {
    const payload: any = {
      task_complexity: taskComplexity,
      agent_count: agentCount,
    };

    if (timeConstraint) {
      payload.time_constraint_minutes = timeConstraint;
    }

    const response = await fetch(`${this.baseOFF}/optimal-strategy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Failed to get optimal strategy: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      strategy: data.strategy,
      confidence_score: data.confidence_score,
      reasoning: data.reasoning
    };
  }

  // =====================================================
  // LEARNING AND OPTIMIZATION
  // =====================================================

  async runLearningCycle(): Promise<{
    success: boolean;
    message: string;
    updated_insights: any;
    timestamp: string;
  }> {
    const response = await fetch(`${this.baseURL}/learning/run-cycle`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to run learning cycle: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      success: data.success,
      message: data.message,
      updated_insights: data.updated_insights,
      timestamp: data.timestamp
    };
  }

  async cleanupExpiredContexts(): Promise<{
    success: boolean;
    packages_before: number;
    packages_after: number;
    cleaned_count: number;
    timestamp: string;
  }> {
    const response = await fetch(`${this.baseURL}/cleanup-contexts`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to cleanup expired contexts: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      success: data.success,
      packages_before: data.packages_before,
      packages_after: data.packages_after,
      cleaned_count: data.cleaned_count,
      timestamp: data.timestamp
    };
  }

  // =====================================================
  // UTILITY METHODS
  // =====================================================

  private transformActiveHandoffFromAPI(apiHandoff: any): ActiveHandoff {
    return {
      handoff_id: apiHandoff.handoff_id,
      source_agent: apiHandoff.source_agent,
      target_agent: apiHandoff.target_agent,
      status: apiHandoff.status as HandoffStatus,
      progress: apiHandoff.progress || 0,
      strategy: apiHandoff.strategy as HandoffStrategy,
      start_time: new Date(apiHandoff.start_time),
      estimated_completion: apiHandoff.estimated_completion ? new Date(apiHandoff.estimated_completion) : undefined,
      confidence_score: apiHandoff.confidence_score || 0,
      task_description: apiHandoff.task_description || ''
    };
  }

  private transformHandoffHistoryFromAPI(apiHistory: any): HandoffHistoryEntry {
    return {
      handoff_id: apiHistory.handoff_id,
      source_agent: apiHistory.source_agent,
      target_agent: apiHistory.target_agent,
      status: apiHistory.status as HandoffStatus,
      strategy: apiHistory.strategy as HandoffStrategy,
      duration: apiHistory.duration || 0,
      success: apiHistory.success || false,
      timestamp: new Date(apiHistory.timestamp),
      task_summary: apiHistory.task_summary || ''
    };
  }

  private transformHandoffResultFromAPI(apiResult: any): HandoffResult {
    return {
      handoff_id: apiResult.handoff_id,
      status: apiResult.status as HandoffStatus,
      source_agent_id: apiResult.source_agent_id,
      target_agent_id: apiResult.target_agent_id,
      response_content: apiResult.response_content,
      execution_time: apiResult.execution_time || 0,
      error_message: apiResult.error_message,
      metrics: apiResult.metrics || {},
      context_package_id: apiResult.context_package_id,
      completed_at: new Date(apiResult.completed_at)
    };
  }

  private transformHandoffRecommendationFromAPI(apiRecommendation: any): HandoffRecommendation {
    return {
      task_id: apiRecommendation.task_id || '',
      task_description: apiRecommendation.task_description,
      current_agent_id: apiRecommendation.current_agent_id,
      recommended: apiRecommendation.recommended || false,
      recommended_agents: (apiRecommendation.recommended_agents || []).map((agent: any) => ({
        agent_id: agent.agent_id,
        agent_name: agent.agent_name,
        match_score: agent.match_score || 0,
        expertise_score: agent.expertise_score || 0,
        performance_score: agent.performance_score || 0,
        availability_score: agent.availability_score || 0,
        load_balance_score: agent.load_balance_score || 0,
        reasoning: agent.reasoning || '',
        estimated_response_time: agent.estimated_response_time || 0
      })),
      strategy_recommendation: apiRecommendation.strategy_recommendation ? {
        strategy: apiRecommendation.strategy_recommendation.strategy as HandoffStrategy,
        confidence_score: apiRecommendation.strategy_recommendation.confidence_score || 0,
        reasoning: apiRecommendation.strategy_recommendation.reasoning || '',
        expected_improvement: apiRecommendation.strategy_recommendation.expected_improvement || 0,
        risk_factors: apiRecommendation.strategy_recommendation.risk_factors || []
      } : undefined,
      required_capabilities: apiRecommendation.required_capabilities || [],
      reasoning: apiRecommendation.reasoning || '',
      confidence_score: apiRecommendation.confidence_score || 0,
      generated_at: new Date(apiRecommendation.generated_at)
    };
  }

  private transformHandoffAnalyticsFromAPI(apiAnalytics: any): HandoffAnalytics {
    return {
      total_handoffs: apiAnalytics.total_handoffs || 0,
      success_rate: apiAnalytics.success_rate || 0,
      strategy_performance: apiAnalytics.strategy_performance || {},
      agent_performance: apiAnalytics.agent_performance || {},
      learning_insights: apiAnalytics.learning_insights || {
        improved_patterns: [],
        optimization_opportunities: [],
        capability_gaps: [],
        confidence_improvements: []
      },
      time_range_hours: apiAnalytics.time_range_hours || 24,
      generated_at: new Date(apiAnalytics.generated_at)
    };
  }

  private transformAgentCapabilityFromAPI(apiCapability: any): AgentCapability {
    return {
      capability_type: apiCapability.capability_type,
      expertise_level: apiCapability.expertise_level,
      confidence_score: apiCapability.confidence_score || 0,
      performance_metrics: apiCapability.performance_metrics || {
        tasks_completed: 0,
        success_rate: 0,
        avg_completion_time: 0,
        error_rate: 0,
        user_satisfaction: 0
      },
      last_used: apiCapability.last_used ? new Date(apiCapability.last_used) : undefined,
      validation_status: apiCapability.validation_status || 'pending',
      capability_details: apiCapability.capability_details || {}
    };
  }

  private transformHandoffVisualizationFromAPI(apiVisualization: any): HandoffVisualization {
    return {
      active_handoffs: (apiVisualization.active_handoffs || []).map(this.transformActiveHandoffFromAPI),
      handoff_history: (apiVisualization.handoff_history || []).map(this.transformHandoffHistoryFromAPI),
      agent_states: (apiVisualization.agent_states || []).map((state: any) => ({
        agent_id: state.agent_id,
        agent_name: state.agent_name,
        agent_type: state.agent_type,
        current_status: state.current_status,
        current_handoff_id: state.current_handoff_id,
        handoff_stats: state.handoff_stats || {
          initiated_today: 0,
          received_today: 0,
          success_rate: 0,
          avg_response_time: 0
        },
        capabilities: state.capabilities || [],
        load_factor: state.load_factor || 0
      })),
      performance_metrics: apiVisualization.performance_metrics || {
        total_handoffs_today: 0,
        success_rate_today: 0,
        avg_handoff_time: 0,
        most_used_strategy: HandoffStrategy.SEQUENTIAL,
        most_active_agent: '',
        confidence_trend: 0,
        performance_score: 0
      },
      last_updated: new Date(apiVisualization.last_updated)
    };
  }
}

export const handoffService = new HandoffServiceImpl();
export default handoffService;