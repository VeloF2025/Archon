/**
 * Agent Management Service
 * 
 * Frontend service for communicating with the Archon 3.0 Intelligence-Tiered Agent Management API
 */

import { 
  AgentV3, 
  AgentState, 
  ModelTier, 
  AgentType,
  CreateAgentRequest,
  UpdateAgentRequest,
  AgentPerformanceMetrics,
  ProjectIntelligenceOverview,
  CostOptimizationRecommendation,
  TaskComplexity,
  ComplexityFactors,
  AgentKnowledge,
  SharedContext,
  BroadcastMessage,
  BudgetStatus,
  AgentPool
} from '../types/agentTypes';
import { API_BASE_URL, getApiBasePath } from '../config/api';

class AgentManagementService {
  private baseURL: string;

  constructor() {
    this.baseURL = `${API_BASE_URL}/agent-management`;
  }

  // =====================================================
  // AGENT LIFECYCLE MANAGEMENT
  // =====================================================

  async getAgents(projectId?: string): Promise<AgentV3[]> {
    let url = `${this.baseURL}/agents`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch agents: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Handle case where backend response doesn't have expected structure
      if (!data || !Array.isArray(data.agents)) {
        console.warn('Invalid agents response structure, returning empty array');
        return [];
      }
      
      return data.agents.map(this.transformAgentFromAPI);
    } catch (error) {
      console.error('Error fetching agents:', error);
      // Return empty array to allow UI to handle gracefully
      return [];
    }
  }

  async getAgentById(agentId: string): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data.agent);
  }

  async createAgent(agentData: CreateAgentRequest): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(agentData),
    });

    if (!response.ok) {
      throw new Error(`Failed to create agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data.agent);
  }

  async updateAgentState(agentId: string, newState: AgentState, reason?: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/state`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        state: newState,
        reason: reason || 'User action'
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent state: ${response.statusText}`);
    }
  }

  async updateAgent(agentId: string, updates: UpdateAgentRequest): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data.agent);
  }

  async deleteAgent(agentId: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete agent: ${response.statusText}`);
    }
  }

  async hibernateIdleAgents(projectId?: string, idleTimeoutMinutes: number = 30): Promise<number> {
    let url = `${this.baseURL}/agents/hibernate-idle?idle_timeout_minutes=${idleTimeoutMinutes}`;
    if (projectId) {
      url += `&project_id=${encodeURIComponent(projectId)}`;
    }

    const response = await fetch(url, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to hibernate idle agents: ${response.statusText}`);
    }

    const data = await response.json();
    return data.hibernated_count;
  }

  async getAgentStateHistory(agentId: string): Promise<any[]> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/history`);
    if (!response.ok) {
      throw new Error(`Failed to fetch agent history: ${response.statusText}`);
    }

    const data = await response.json();
    return data.history;
  }

  // =====================================================
  // INTELLIGENCE TIER ROUTING
  // =====================================================

  async assessTaskComplexity(
    taskId: string, 
    factors: ComplexityFactors
  ): Promise<TaskComplexity> {
    const response = await fetch(`${this.baseURL}/complexity/assess`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        task_id: taskId,
        technical_complexity: factors.technical,
        domain_expertise_required: factors.domain_expertise,
        code_volume_complexity: factors.code_volume,
        integration_complexity: factors.integration
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to assess task complexity: ${response.statusText}`);
    }

    const data = await response.json();
    return data.complexity;
  }

  async getOptimalAgentForTask(
    projectId: string,
    taskComplexity: TaskComplexity,
    agentType: AgentType
  ): Promise<AgentV3 | null> {
    const response = await fetch(`${this.baseURL}/routing/optimal-agent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        project_id: projectId,
        task_complexity: taskComplexity,
        agent_type: agentType
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to find optimal agent: ${response.statusText}`);
    }

    const data = await response.json();
    return data.agent ? this.transformAgentFromAPI(data.agent) : null;
  }

  // =====================================================
  // KNOWLEDGE MANAGEMENT
  // =====================================================

  async getAgentKnowledge(agentId: string, limit: number = 20): Promise<AgentKnowledge[]> {
    const response = await fetch(
      `${this.baseURL}/agents/${agentId}/knowledge?limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch agent knowledge: ${response.statusText}`);
    }

    const data = await response.json();
    return data.knowledge;
  }

  async searchAgentKnowledge(
    agentId: string,
    query: string,
    limit: number = 10
  ): Promise<AgentKnowledge[]> {
    const response = await fetch(`${this.baseURL}/knowledge/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        agent_id: agentId,
        query: query,
        limit: limit
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to search knowledge: ${response.statusText}`);
    }

    const data = await response.json();
    return data.results;
  }

  async storeAgentKnowledge(knowledge: Partial<AgentKnowledge>): Promise<AgentKnowledge> {
    const response = await fetch(`${this.baseURL}/knowledge`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(knowledge),
    });

    if (!response.ok) {
      throw new Error(`Failed to store knowledge: ${response.statusText}`);
    }

    const data = await response.json();
    return data.knowledge;
  }

  async updateKnowledgeConfidence(
    knowledgeId: string,
    success: boolean
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/knowledge/${knowledgeId}/confidence`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ success }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update knowledge confidence: ${response.statusText}`);
    }
  }

  // =====================================================
  // COST TRACKING AND OPTIMIZATION
  // =====================================================

  async trackAgentCost(costData: {
    agent_id: string;
    project_id: string;
    task_id?: string;
    input_tokens: number;
    output_tokens: number;
    model_tier: ModelTier;
    task_duration_seconds?: number;
    success: boolean;
  }): Promise<void> {
    const response = await fetch(`${this.baseURL}/costs/track`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(costData),
    });

    if (!response.ok) {
      throw new Error(`Failed to track cost: ${response.statusText}`);
    }
  }

  async getBudgetStatus(projectId: string): Promise<BudgetStatus> {
    const response = await fetch(`${this.baseURL}/costs/budget/${projectId}/status`);
    if (!response.ok) {
      throw new Error(`Failed to get budget status: ${response.statusText}`);
    }

    const data = await response.json();
    return data.budget_status;
  }

  async getCostOptimizationRecommendations(
    projectId?: string
  ): Promise<CostOptimizationRecommendation[]> {
    let url = `${this.baseURL}/costs/recommendations`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to get cost recommendations: ${response.statusText}`);
    }

    const data = await response.json();
    return data.recommendations;
  }

  async updateBudgetConstraints(
    projectId: string,
    constraints: Partial<{
      monthly_budget: number;
      daily_budget: number;
      per_task_budget: number;
      warning_threshold: number;
      critical_threshold: number;
      auto_downgrade_enabled: boolean;
      emergency_stop_enabled: boolean;
    }>
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/costs/budget/${projectId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(constraints),
    });

    if (!response.ok) {
      throw new Error(`Failed to update budget constraints: ${response.statusText}`);
    }
  }

  // =====================================================
  // REAL-TIME COLLABORATION
  // =====================================================

  async createSharedContext(
    taskId: string,
    projectId: string,
    contextName: string
  ): Promise<SharedContext> {
    const response = await fetch(`${this.baseURL}/collaboration/contexts`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        task_id: taskId,
        project_id: projectId,
        context_name: contextName
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create shared context: ${response.statusText}`);
    }

    const data = await response.json();
    return data.context;
  }

  async getSharedContexts(projectId: string): Promise<SharedContext[]> {
    const response = await fetch(`${this.baseURL}/collaboration/contexts?project_id=${projectId}`);
    if (!response.ok) {
      throw new Error(`Failed to get shared contexts: ${response.statusText}`);
    }

    const data = await response.json();
    return data.contexts;
  }

  async broadcastMessage(message: {
    message_id: string;
    topic: string;
    content: Record<string, any>;
    message_type: string;
    priority: number;
    sender_id?: string;
    target_agents?: string[];
  }): Promise<void> {
    const response = await fetch(`${this.baseURL}/collaboration/broadcast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(message),
    });

    if (!response.ok) {
      throw new Error(`Failed to broadcast message: ${response.statusText}`);
    }
  }

  async subscribeAgentToTopic(
    agentId: string,
    topic: string,
    priorityFilter: number = 1
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/collaboration/subscribe`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        agent_id: agentId,
        topic: topic,
        priority_filter: priorityFilter
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to subscribe to topic: ${response.statusText}`);
    }
  }

  // =====================================================
  // ANALYTICS AND MONITORING
  // =====================================================

  async getAgentPerformanceMetrics(agentId?: string): Promise<AgentPerformanceMetrics[]> {
    let url = `${this.baseURL}/analytics/performance`;
    if (agentId) {
      url += `?agent_id=${encodeURIComponent(agentId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to get performance metrics: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Handle case where backend response doesn't have expected structure
      if (!data || !Array.isArray(data.metrics)) {
        console.warn('Invalid metrics response structure, returning empty array');
        return [];
      }
      
      return data.metrics;
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      return [];
    }
  }

  async getProjectIntelligenceOverview(projectId?: string): Promise<ProjectIntelligenceOverview> {
    let url = `${this.baseURL}/analytics/project-overview`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to get project overview: ${response.statusText}`);
    }

    const data = await response.json();
    return data.overview;
  }

  async getAgentPools(projectId?: string): Promise<AgentPool[]> {
    let url = `${this.baseURL}/pools`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to get agent pools: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Handle case where backend response doesn't have expected structure
      if (!data || !Array.isArray(data.pools)) {
        console.warn('Invalid pools response structure, returning empty array');
        return [];
      }
      
      return data.pools;
    } catch (error) {
      console.error('Error fetching agent pools:', error);
      return [];
    }
  }

  async updateAgentPool(
    poolId: string,
    updates: Partial<{
      opus_limit: number;
      sonnet_limit: number;
      haiku_limit: number;
      auto_scaling_enabled: boolean;
      hibernation_timeout_minutes: number;
    }>
  ): Promise<AgentPool> {
    const response = await fetch(`${this.baseURL}/pools/${poolId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent pool: ${response.statusText}`);
    }

    const data = await response.json();
    return data.pool;
  }

  // =====================================================
  // UTILITY METHODS
  // =====================================================

  private transformAgentFromAPI(apiAgent: any): AgentV3 {
    return {
      ...apiAgent,
      id: apiAgent.id,
      state: apiAgent.state as AgentState,
      model_tier: apiAgent.model_tier as ModelTier,
      agent_type: apiAgent.agent_type as AgentType,
      created_at: new Date(apiAgent.created_at),
      updated_at: new Date(apiAgent.updated_at),
      state_changed_at: new Date(apiAgent.state_changed_at),
      last_active_at: apiAgent.last_active_at ? new Date(apiAgent.last_active_at) : undefined,
    };
  }

  // Tier pricing constants
  static readonly TIER_PRICING = {
    [ModelTier.OPUS]: { input: 15.00, output: 75.00 },
    [ModelTier.SONNET]: { input: 3.00, output: 15.00 },
    [ModelTier.HAIKU]: { input: 0.25, output: 1.25 }
  };

  static calculateCost(
    tier: ModelTier,
    inputTokens: number,
    outputTokens: number
  ): { inputCost: number; outputCost: number; totalCost: number } {
    const pricing = this.TIER_PRICING[tier];
    const inputCost = (inputTokens / 1_000_000) * pricing.input;
    const outputCost = (outputTokens / 1_000_000) * pricing.output;
    
    return {
      inputCost,
      outputCost,
      totalCost: inputCost + outputCost
    };
  }

  static calculateComplexityScore(factors: ComplexityFactors): number {
    return (
      factors.technical +
      factors.domain_expertise +
      factors.code_volume +
      factors.integration
    ) / 4.0;
  }

  static recommendTier(complexityScore: number): ModelTier {
    if (complexityScore >= 0.75) {
      return ModelTier.OPUS;
    } else if (complexityScore >= 0.15) { // Sonnet-first preference
      return ModelTier.SONNET;
    } else {
      return ModelTier.HAIKU;
    }
  }
}

export const agentManagementService = new AgentManagementService();
export default agentManagementService;