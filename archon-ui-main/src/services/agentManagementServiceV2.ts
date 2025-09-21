/**
 * Agent Management Service V2 - Fixed Version
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

class AgentManagementServiceV2 {
  private baseURL: string;

  constructor() {
    // Use relative URL for API calls
    this.baseURL = '/api/agent-management';
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

  async createAgent(request: CreateAgentRequest): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Failed to create agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  async updateAgent(agentId: string, updates: UpdateAgentRequest): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(updates),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  async deleteAgent(agentId: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete agent: ${response.statusText}`);
    }
  }

  async startAgent(agentId: string): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/start`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to start agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  async stopAgent(agentId: string): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/stop`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to stop agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  async pauseAgent(agentId: string): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/pause`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to pause agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  async resumeAgent(agentId: string): Promise<AgentV3> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/resume`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to resume agent: ${response.statusText}`);
    }

    const data = await response.json();
    return this.transformAgentFromAPI(data);
  }

  // =====================================================
  // ANALYTICS & METRICS
  // =====================================================

  async getAgentPerformanceMetrics(agentId?: string, projectId?: string): Promise<AgentPerformanceMetrics[]> {
    let url = `${this.baseURL}/analytics/performance`;
    const params = new URLSearchParams();
    
    if (agentId) {
      params.append('agent_id', agentId);
    }
    if (projectId) {
      params.append('project_id', projectId);
    }
    
    if (params.toString()) {
      url += `?${params.toString()}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to fetch performance metrics: ${response.statusText}`);
        return [];
      }

      const data = await response.json();
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      return [];
    }
  }

  async getProjectOverview(projectId?: string): Promise<ProjectIntelligenceOverview | null> {
    let url = `${this.baseURL}/analytics/project-overview`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to fetch project overview: ${response.statusText}`);
        return null;
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching project overview:', error);
      return null;
    }
  }

  async getCostRecommendations(projectId?: string): Promise<CostOptimizationRecommendation[]> {
    let url = `${this.baseURL}/costs/recommendations`;
    if (projectId) {
      url += `?project_id=${encodeURIComponent(projectId)}`;
    }

    try {
      const response = await fetch(url);
      if (!response.ok) {
        console.warn(`Failed to fetch cost recommendations: ${response.statusText}`);
        return [];
      }

      const data = await response.json();
      return Array.isArray(data) ? data : [];
    } catch (error) {
      console.error('Error fetching cost recommendations:', error);
      return [];
    }
  }

  // Alias method for backward compatibility with AgentManagementPage
  async getCostOptimizationRecommendations(projectId?: string): Promise<CostOptimizationRecommendation[]> {
    return this.getCostRecommendations(projectId);
  }

  // Alias method for backward compatibility with AgentManagementPage  
  async getProjectIntelligenceOverview(projectId?: string): Promise<ProjectIntelligenceOverview | null> {
    return this.getProjectOverview(projectId);
  }

  // State management methods
  async updateAgentState(agentId: string, newState: AgentState): Promise<void> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/state`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ state: newState }),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent state: ${response.statusText}`);
    }
  }

  async hibernateIdleAgents(): Promise<number> {
    const response = await fetch(`${this.baseURL}/agents/hibernate-idle`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to hibernate idle agents: ${response.statusText}`);
    }

    const data = await response.json();
    return data.hibernated_count || 0;
  }

  // =====================================================
  // TASK MANAGEMENT
  // =====================================================

  async assignTask(agentId: string, task: {
    description: string;
    complexity: TaskComplexity;
    priority: number;
    dependencies?: string[];
    metadata?: Record<string, any>;
  }): Promise<string> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/tasks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(task),
    });

    if (!response.ok) {
      throw new Error(`Failed to assign task: ${response.statusText}`);
    }

    const data = await response.json();
    return data.task_id;
  }

  async analyzeTaskComplexity(task: string): Promise<ComplexityFactors> {
    const response = await fetch(`${this.baseURL}/tasks/analyze-complexity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ task }),
    });

    if (!response.ok) {
      throw new Error(`Failed to analyze task complexity: ${response.statusText}`);
    }

    return await response.json();
  }

  // =====================================================
  // KNOWLEDGE MANAGEMENT
  // =====================================================

  async updateAgentKnowledge(agentId: string, knowledge: Partial<AgentKnowledge>): Promise<void> {
    const response = await fetch(`${this.baseURL}/agents/${agentId}/knowledge`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(knowledge),
    });

    if (!response.ok) {
      throw new Error(`Failed to update agent knowledge: ${response.statusText}`);
    }
  }

  async shareContext(context: SharedContext): Promise<void> {
    const response = await fetch(`${this.baseURL}/context/share`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(context),
    });

    if (!response.ok) {
      throw new Error(`Failed to share context: ${response.statusText}`);
    }
  }

  async broadcastMessage(message: BroadcastMessage): Promise<void> {
    const response = await fetch(`${this.baseURL}/broadcast`, {
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

  // =====================================================
  // POOL MANAGEMENT
  // =====================================================

  async getAgentPools(): Promise<AgentPool[]> {
    const response = await fetch(`${this.baseURL}/pools`);

    if (!response.ok) {
      throw new Error(`Failed to fetch agent pools: ${response.statusText}`);
    }

    return await response.json();
  }

  async createAgentPool(pool: Omit<AgentPool, 'id' | 'agents' | 'created_at' | 'updated_at'>): Promise<AgentPool> {
    const response = await fetch(`${this.baseURL}/pools`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(pool),
    });

    if (!response.ok) {
      throw new Error(`Failed to create agent pool: ${response.statusText}`);
    }

    return await response.json();
  }

  // =====================================================
  // HELPER METHODS
  // =====================================================

  private transformAgentFromAPI(data: any): AgentV3 {
    // Transform snake_case API response to camelCase
    return {
      id: data.id,
      name: data.name,
      type: data.type,
      model: data.model,
      tier: data.tier,
      state: data.state,
      capabilities: data.capabilities || [],
      specialization: data.specialization,
      projectId: data.project_id,
      metadata: data.metadata || {},
      performanceScore: data.performance_score,
      tasksCompleted: data.tasks_completed || 0,
      successRate: data.success_rate || 0,
      avgResponseTime: data.avg_response_time || 0,
      lastActive: data.last_active ? new Date(data.last_active) : undefined,
      knowledge: data.knowledge || {
        domain: [],
        experience: [],
        learned_patterns: [],
        success_patterns: [],
        failure_patterns: []
      },
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
    };
  }
}

export const agentManagementServiceV2 = new AgentManagementServiceV2();
export default AgentManagementServiceV2;