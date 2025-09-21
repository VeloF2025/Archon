import { AgencyData, CommunicationFlow, CommunicationType, CommunicationStatus } from '../types/workflowTypes';
import { AgentV3, AgentState } from '../types/agentTypes';

/**
 * Service for managing workflow visualization data
 * This service provides methods to fetch and transform agent data into workflow format
 */

export class WorkflowService {
  private static instance: WorkflowService;
  private baseUrl: string;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8181';
  }

  public static getInstance(): WorkflowService {
    if (!WorkflowService.instance) {
      WorkflowService.instance = new WorkflowService();
    }
    return WorkflowService.instance;
  }

  /**
   * Convert agent data to agency workflow format
   */
  public convertAgentsToAgencyData(agents: AgentV3[], projectId?: string): AgencyData {
    // Generate communication flows based on agent interactions
    const communicationFlows = this.generateCommunicationFlows(agents);

    return {
      id: projectId || 'default-agency',
      name: 'Agent Collaboration Workflow',
      description: 'Real-time visualization of agent communication and collaboration patterns',
      agents,
      communication_flows: communicationFlows,
      workflow_rules: {
        routing_rules: {
          hierarchical: true,
          load_balancing: true,
          failure_handling: true,
        },
        collaboration_patterns: {
          peer_to_peer: true,
          broadcast: true,
          request_response: true,
        },
        escalation_paths: ['supervisor', 'manager', 'director'],
      },
      created_at: new Date(),
      updated_at: new Date(),
    };
  }

  /**
   * Generate communication flows between agents based on their types and states
   */
  private generateCommunicationFlows(agents: AgentV3[]): CommunicationFlow[] {
    const flows: CommunicationFlow[] = [];
    const activeAgents = agents.filter(agent => agent.state === AgentState.ACTIVE);

    // Generate flows based on agent types and typical communication patterns
    activeAgents.forEach((sourceAgent, index) => {
      activeAgents.forEach((targetAgent, targetIndex) => {
        if (index >= targetIndex) return; // Avoid duplicate flows

        const flow = this.createCommunicationFlow(sourceAgent, targetAgent);
        if (flow) {
          flows.push(flow);
        }
      });
    });

    // Add some broadcast communications for orchestrators
    const orchestrators = agents.filter(agent =>
      agent.agent_type.includes('ARCHITECT') ||
      agent.agent_type.includes('PLANNER') ||
      agent.agent_type.includes('COORDINATOR')
    );

    orchestrators.forEach(orchestrator => {
      activeAgents.forEach(targetAgent => {
        if (orchestrator.id !== targetAgent.id) {
          flows.push({
            id: `broadcast-${orchestrator.id}-${targetAgent.id}`,
            source_agent_id: orchestrator.id,
            target_agent_id: targetAgent.id,
            communication_type: CommunicationType.BROADCAST,
            status: Math.random() > 0.3 ? CommunicationStatus.ACTIVE : CommunicationStatus.IDLE,
            message_count: Math.floor(Math.random() * 20) + 5,
            last_message_at: new Date(Date.now() - Math.random() * 3600000), // Within last hour
            message_type: 'status_update',
            data_flow: {
              input_size: Math.floor(Math.random() * 1000) + 100,
              output_size: Math.floor(Math.random() * 500) + 50,
              processing_time_ms: Math.floor(Math.random() * 5000) + 100,
            },
            metadata: {
              priority: 'medium',
              encryption: 'enabled',
            },
          });
        }
      });
    });

    return flows;
  }

  /**
   * Create a communication flow between two agents
   */
  private createCommunicationFlow(sourceAgent: AgentV3, targetAgent: AgentV3): CommunicationFlow | null {
    // Determine communication type based on agent types
    let communicationType: CommunicationType;
    let shouldCreate = false;

    // Define typical communication patterns
    if (sourceAgent.agent_type.includes('ARCHITECT') &&
        (targetAgent.agent_type.includes('IMPLEMENTER') || targetAgent.agent_type.includes('DEVELOPER'))) {
      communicationType = CommunicationType.HIERARCHICAL;
      shouldCreate = true;
    } else if (sourceAgent.agent_type.includes('TESTER') &&
               targetAgent.agent_type.includes('IMPLEMENTER')) {
      communicationType = CommunicationType.COLLABORATIVE;
      shouldCreate = true;
    } else if (sourceAgent.agent_type === targetAgent.agent_type) {
      communicationType = CommunicationType.COLLABORATIVE;
      shouldCreate = Math.random() > 0.5; // 50% chance for same-type agents
    } else if (sourceAgent.model_tier === targetAgent.model_tier) {
      communicationType = CommunicationType.DIRECT;
      shouldCreate = Math.random() > 0.3; // 70% chance for same-tier agents
    } else {
      communicationType = CommunicationType.DIRECT;
      shouldCreate = Math.random() > 0.6; // 40% chance for cross-tier
    }

    if (!shouldCreate) return null;

    // Determine status based on agent states
    let status: CommunicationStatus;
    if (sourceAgent.state === AgentState.ACTIVE && targetAgent.state === AgentState.ACTIVE) {
      status = Math.random() > 0.2 ? CommunicationStatus.ACTIVE : CommunicationStatus.IDLE;
    } else if (sourceAgent.state === AgentState.IDLE || targetAgent.state === AgentState.IDLE) {
      status = CommunicationStatus.IDLE;
    } else {
      status = Math.random() > 0.8 ? CommunicationStatus.PENDING : CommunicationStatus.IDLE;
    }

    return {
      id: `flow-${sourceAgent.id}-${targetAgent.id}`,
      source_agent_id: sourceAgent.id,
      target_agent_id: targetAgent.id,
      communication_type: communicationType,
      status,
      message_count: Math.floor(Math.random() * 50) + 1,
      last_message_at: new Date(Date.now() - Math.random() * 7200000), // Within last 2 hours
      message_type: this.getMessageType(communicationType),
      data_flow: {
        input_size: Math.floor(Math.random() * 2000) + 100,
        output_size: Math.floor(Math.random() * 1000) + 50,
        processing_time_ms: Math.floor(Math.random() * 10000) + 100,
      },
      metadata: {
        priority: this.getPriority(communicationType),
        encryption: 'enabled',
        compression: 'adaptive',
      },
    };
  }

  /**
   * Get message type based on communication type
   */
  private getMessageType(communicationType: CommunicationType): string {
    switch (communicationType) {
      case CommunicationType.HIERARCHICAL:
        return 'task_assignment';
      case CommunicationType.COLLABORATIVE:
        return 'data_request';
      case CommunicationType.BROADCAST:
        return 'status_update';
      case CommunicationType.CHAIN:
        return 'workflow_step';
      case CommunicationType.DIRECT:
      default:
        return 'message';
    }
  }

  /**
   * Get priority based on communication type
   */
  private getPriority(communicationType: CommunicationType): string {
    switch (communicationType) {
      case CommunicationType.HIERARCHICAL:
        return 'high';
      case CommunicationType.COLLABORATIVE:
        return 'medium';
      case CommunicationType.BROADCAST:
        return 'low';
      case CommunicationType.CHAIN:
        return 'medium';
      case CommunicationType.DIRECT:
      default:
        return 'normal';
    }
  }

  /**
   * Simulate real-time communication updates
   */
  public simulateCommunicationUpdate(agencyData: AgencyData): AgencyData {
    const updatedFlows = agencyData.communication_flows.map(flow => {
      // Randomly update some flows
      if (Math.random() > 0.8) {
        return {
          ...flow,
          message_count: flow.message_count + Math.floor(Math.random() * 3),
          last_message_at: new Date(),
          status: Math.random() > 0.1 ? CommunicationStatus.ACTIVE : flow.status,
        };
      }
      return flow;
    });

    return {
      ...agencyData,
      communication_flows: updatedFlows,
      updated_at: new Date(),
    };
  }

  /**
   * Get workflow statistics for analytics
   */
  public getWorkflowStatistics(agencyData: AgencyData) {
    const activeCommunications = agencyData.communication_flows.filter(
      flow => flow.status === CommunicationStatus.ACTIVE
    );

    const totalMessages = agencyData.communication_flows.reduce(
      (sum, flow) => sum + flow.message_count, 0
    );

    const communicationTypeDistribution = agencyData.communication_flows.reduce((acc, flow) => {
      acc[flow.communication_type] = (acc[flow.communication_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const busiestAgent = this.getBusiestAgent(agencyData);

    return {
      total_agents: agencyData.agents.length,
      active_agents: agencyData.agents.filter(a => a.state === AgentState.ACTIVE).length,
      total_communications: agencyData.communication_flows.length,
      active_communications: activeCommunications.length,
      total_messages,
      average_messages_per_connection: totalMessages / agencyData.communication_flows.length || 0,
      communication_type_distribution: communicationTypeDistribution,
      busiest_agent: busiestAgent,
    };
  }

  /**
   * Find the busiest agent in the workflow
   */
  private getBusiestAgent(agencyData: AgencyData) {
    const agentMessageCounts = new Map<string, number>();

    agencyData.communication_flows.forEach(flow => {
      const sourceCount = agentMessageCounts.get(flow.source_agent_id) || 0;
      const targetCount = agentMessageCounts.get(flow.target_agent_id) || 0;

      agentMessageCounts.set(flow.source_agent_id, sourceCount + flow.message_count);
      agentMessageCounts.set(flow.target_agent_id, targetCount + flow.message_count);
    });

    let busiestAgentId = '';
    let maxMessages = 0;

    agentMessageCounts.forEach((count, agentId) => {
      if (count > maxMessages) {
        maxMessages = count;
        busiestAgentId = agentId;
      }
    });

    const busiestAgent = agencyData.agents.find(a => a.id === busiestAgentId);
    return busiestAgent
      ? { agent_id: busiestAgent.id, agent_name: busiestAgent.name, message_count: maxMessages }
      : null;
  }

  /**
   * Export workflow data as JSON
   */
  public exportWorkflowData(agencyData: AgencyData): string {
    return JSON.stringify({
      workflow: agencyData,
      statistics: this.getWorkflowStatistics(agencyData),
      exported_at: new Date().toISOString(),
      version: '1.0',
    }, null, 2);
  }

  /**
   * Import workflow data from JSON
   */
  public importWorkflowData(jsonData: string): AgencyData {
    try {
      const data = JSON.parse(jsonData);
      if (data.workflow) {
        return data.workflow as AgencyData;
      }
      throw new Error('Invalid workflow data format');
    } catch (error) {
      throw new Error(`Failed to import workflow data: ${error}`);
    }
  }
}

// Export singleton instance
export const workflowService = WorkflowService.getInstance();