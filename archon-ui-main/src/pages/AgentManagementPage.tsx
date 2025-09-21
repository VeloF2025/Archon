import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { useSocket } from '../hooks/useSocket';
import { useToast } from '../hooks/useToast';
import { agentManagementServiceV2 as agentManagementService } from '../services/agentManagementServiceV2';
import { 
  AgentV3, 
  AgentState, 
  ModelTier, 
  AgentType,
  AgentPerformanceMetrics,
  ProjectIntelligenceOverview,
  CostOptimizationRecommendation
} from '../types/agentTypes';
import { AgentCard } from '../components/agents/AgentCard';
import { AgentCreationModal } from '../components/agents/AgentCreationModal';
import { AgentDetailsModal } from '../components/agents/AgentDetailsModal';
import { AgentPoolView } from '../components/agents/AgentPoolView';
import { IntelligenceOverview } from '../components/agents/IntelligenceOverview';
import { CostDashboard } from '../components/agents/CostDashboard';
import { RealTimeCollaboration } from '../components/agents/RealTimeCollaboration';
import { KnowledgeManagement } from '../components/agents/KnowledgeManagement';
import { AgencyWorkflowVisualizerWithProvider } from '../components/workflow/AgencyWorkflowVisualizer';
import { workflowService } from '../services/workflowService';
import { AgencyData } from '../types/workflowTypes';

export const AgentManagementPage: React.FC = () => {
  // State management
  const [agents, setAgents] = useState<AgentV3[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentV3 | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [stateFilter, setStateFilter] = useState<AgentState | 'all'>('all');
  const [tierFilter, setTierFilter] = useState<ModelTier | 'all'>('all');
  
  // Analytics state
  const [performanceMetrics, setPerformanceMetrics] = useState<AgentPerformanceMetrics[]>([]);
  const [projectOverview, setProjectOverview] = useState<ProjectIntelligenceOverview | null>(null);
  const [costRecommendations, setCostRecommendations] = useState<CostOptimizationRecommendation[]>([]);

  // Workflow state
  const [agencyData, setAgencyData] = useState<AgencyData | null>(null);
  const [showWorkflowView, setShowWorkflowView] = useState(false);

  const { toast } = useToast();
  const socket = useSocket();

  // Load agents and analytics data
  useEffect(() => {
    loadAgentsData();
  }, []);

  // Generate workflow data when agents are loaded
  useEffect(() => {
    if (agents.length > 0) {
      const workflowData = workflowService.convertAgentsToAgencyData(agents);
      setAgencyData(workflowData);
    }
  }, [agents]);

  // Socket subscriptions for real-time updates
  useEffect(() => {
    if (!socket) return;

    const handleAgentUpdate = (data: { agentId: string; state: AgentState; metrics?: any }) => {
      setAgents(prev => prev.map(agent => 
        agent.id === data.agentId 
          ? { ...agent, state: data.state, ...data.metrics }
          : agent
      ));
    };

    const handleNewAgent = (agent: AgentV3) => {
      setAgents(prev => [...prev, agent]);
      toast({
        title: "New Agent Created",
        description: `${agent.name} (${agent.agent_type}) is now available`,
        variant: "success"
      });
    };

    const handleAgentRemoved = (agentId: string) => {
      setAgents(prev => prev.filter(agent => agent.id !== agentId));
    };

    // Create stable handler references for socket events
    const stateChangedHandler = (message: any) => handleAgentUpdate(message.data);
    const agentCreatedHandler = (message: any) => handleNewAgent(message.data);
    const agentRemovedHandler = (message: any) => handleAgentRemoved(message.data);
    const performanceUpdateHandler = (message: any) => handleAgentUpdate(message.data);

    // Workflow event handlers
    const handleWorkflowUpdate = (message: any) => {
      if (agencyData) {
        const updatedData = workflowService.simulateCommunicationUpdate(agencyData);
        setAgencyData(updatedData);
      }
    };

    // Subscribe to agent lifecycle events
    socket.addMessageHandler('agent_state_changed', stateChangedHandler);
    socket.addMessageHandler('agent_created', agentCreatedHandler);
    socket.addMessageHandler('agent_archived', agentRemovedHandler);
    socket.addMessageHandler('agent_performance_update', performanceUpdateHandler);
    socket.addMessageHandler('workflow_update', handleWorkflowUpdate);
    
    return () => {
      socket.removeMessageHandler('agent_state_changed', stateChangedHandler);
      socket.removeMessageHandler('agent_created', agentCreatedHandler);
      socket.removeMessageHandler('agent_archived', agentRemovedHandler);
      socket.removeMessageHandler('agent_performance_update', performanceUpdateHandler);
      socket.removeMessageHandler('workflow_update', handleWorkflowUpdate);
    };
  }, [socket, toast]);

  const loadAgentsData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load agents data (required)
      const agentsData = await agentManagementService.getAgents();
      setAgents(agentsData);

      // Load optional data with individual error handling
      try {
        const metricsData = await agentManagementService.getAgentPerformanceMetrics();
        setPerformanceMetrics(metricsData);
      } catch (err) {
        console.warn('Performance metrics not available:', err);
        setPerformanceMetrics([]);
      }

      try {
        const overviewData = await agentManagementService.getProjectIntelligenceOverview();
        setProjectOverview(overviewData);
      } catch (err) {
        console.warn('Project overview not available:', err);
        setProjectOverview(null);
      }

      try {
        const recommendationsData = await agentManagementService.getCostOptimizationRecommendations();
        setCostRecommendations(recommendationsData);
      } catch (err) {
        console.warn('Cost recommendations not available:', err);
        setCostRecommendations([]);
      }

    } catch (err) {
      console.error('Failed to load agents data:', err);
      setError('Failed to load agent management data');
      toast({
        title: "Error",
        description: "Failed to load agent management data",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAgent = async (agentData: {
    name: string;
    agent_type: AgentType;
    model_tier: ModelTier;
    capabilities: Record<string, any>;
  }) => {
    try {
      const newAgent = await agentManagementService.createAgent(agentData);
      setAgents(prev => [...prev, newAgent]);
      setShowCreateModal(false);
      
      toast({
        title: "Agent Created",
        description: `${newAgent.name} has been created successfully`,
        variant: "success"
      });
    } catch (err) {
      console.error('Failed to create agent:', err);
      toast({
        title: "Error",
        description: "Failed to create agent",
        variant: "destructive"
      });
    }
  };

  const handleUpdateAgentState = async (agentId: string, newState: AgentState) => {
    try {
      await agentManagementService.updateAgentState(agentId, newState);
      
      setAgents(prev => prev.map(agent => 
        agent.id === agentId 
          ? { ...agent, state: newState, state_changed_at: new Date() }
          : agent
      ));

      toast({
        title: "State Updated",
        description: `Agent state changed to ${newState}`,
        variant: "success"
      });
    } catch (err) {
      console.error('Failed to update agent state:', err);
      toast({
        title: "Error",
        description: "Failed to update agent state",
        variant: "destructive"
      });
    }
  };

  const handleHibernateIdleAgents = async () => {
    try {
      const hibernatedCount = await agentManagementService.hibernateIdleAgents();
      
      if (hibernatedCount > 0) {
        toast({
          title: "Agents Hibernated",
          description: `${hibernatedCount} idle agents have been hibernated to save resources`,
          variant: "success"
        });
        
        // Reload data to reflect changes
        loadAgentsData();
      } else {
        toast({
          title: "No Changes",
          description: "No idle agents found to hibernate",
          variant: "default"
        });
      }
    } catch (err) {
      console.error('Failed to hibernate agents:', err);
      toast({
        title: "Error",
        description: "Failed to hibernate idle agents",
        variant: "destructive"
      });
    }
  };

  // Filter agents based on search and filters
  const filteredAgents = agents.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.agent_type.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesState = stateFilter === 'all' || agent.state === stateFilter;
    const matchesTier = tierFilter === 'all' || agent.model_tier === tierFilter;
    
    return matchesSearch && matchesState && matchesTier;
  });

  // Aggregate statistics
  const stats = {
    total: agents.length,
    active: agents.filter(a => a.state === AgentState.ACTIVE).length,
    idle: agents.filter(a => a.state === AgentState.IDLE).length,
    hibernated: agents.filter(a => a.state === AgentState.HIBERNATED).length,
    opus: agents.filter(a => a.model_tier === ModelTier.OPUS).length,
    sonnet: agents.filter(a => a.model_tier === ModelTier.SONNET).length,
    haiku: agents.filter(a => a.model_tier === ModelTier.HAIKU).length,
    totalTasksCompleted: agents.reduce((sum, a) => sum + a.tasks_completed, 0),
    avgSuccessRate: agents.length > 0 
      ? agents.reduce((sum, a) => sum + parseFloat(a.success_rate.toString()), 0) / agents.length 
      : 0
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Card className="p-6">
          <div className="text-center">
            <div className="text-red-500 mb-2">⚠️</div>
            <h3 className="text-lg font-semibold mb-2">Error Loading Agent Management</h3>
            <p className="text-gray-600 mb-4">{error}</p>
            <Button onClick={loadAgentsData}>Retry</Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Agent Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Intelligence-Tiered Adaptive Agent Management System
          </p>
        </div>
        <div className="flex space-x-3">
          <Button
            onClick={handleHibernateIdleAgents}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <span>💤</span>
            <span>Hibernate Idle</span>
          </Button>
          <Button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center space-x-2"
          >
            <span>➕</span>
            <span>Create Agent</span>
          </Button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{stats.total}</div>
            <div className="text-sm text-gray-600">Total Agents</div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{stats.active}</div>
            <div className="text-sm text-gray-600">Active</div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">{stats.idle}</div>
            <div className="text-sm text-gray-600">Idle</div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{stats.hibernated}</div>
            <div className="text-sm text-gray-600">Hibernated</div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold">{stats.totalTasksCompleted}</div>
            <div className="text-sm text-gray-600">Tasks Done</div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <div className="text-2xl font-bold">{Math.round(stats.avgSuccessRate * 100)}%</div>
            <div className="text-sm text-gray-600">Success Rate</div>
          </div>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="agents" className="w-full">
        <TabsList className="grid w-full grid-cols-7">
          <TabsTrigger value="agents">🤖 Agents</TabsTrigger>
          <TabsTrigger value="workflow">🌐 Workflow</TabsTrigger>
          <TabsTrigger value="pools">🏊 Pools</TabsTrigger>
          <TabsTrigger value="intelligence">🧠 Intelligence</TabsTrigger>
          <TabsTrigger value="costs">💰 Costs</TabsTrigger>
          <TabsTrigger value="collaboration">🤝 Collaboration</TabsTrigger>
          <TabsTrigger value="knowledge">📚 Knowledge</TabsTrigger>
        </TabsList>

        {/* Agents Tab */}
        <TabsContent value="agents" className="space-y-4">
          {/* Filters */}
          <Card className="p-4">
            <div className="flex flex-wrap gap-4 items-center">
              <div className="flex-1 min-w-[200px]">
                <Input
                  placeholder="Search agents by name or type..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full"
                />
              </div>
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">State:</label>
                <select
                  value={stateFilter}
                  onChange={(e) => setStateFilter(e.target.value as AgentState | 'all')}
                  className="border rounded px-2 py-1"
                >
                  <option value="all">All States</option>
                  <option value={AgentState.CREATED}>Created</option>
                  <option value={AgentState.ACTIVE}>Active</option>
                  <option value={AgentState.IDLE}>Idle</option>
                  <option value={AgentState.HIBERNATED}>Hibernated</option>
                  <option value={AgentState.ARCHIVED}>Archived</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium">Tier:</label>
                <select
                  value={tierFilter}
                  onChange={(e) => setTierFilter(e.target.value as ModelTier | 'all')}
                  className="border rounded px-2 py-1"
                >
                  <option value="all">All Tiers</option>
                  <option value={ModelTier.OPUS}>Opus</option>
                  <option value={ModelTier.SONNET}>Sonnet</option>
                  <option value={ModelTier.HAIKU}>Haiku</option>
                </select>
              </div>
            </div>
          </Card>

          {/* Agent Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredAgents.map((agent) => (
              <AgentCard
                key={agent.id}
                agent={agent}
                onStateChange={(newState) => handleUpdateAgentState(agent.id, newState)}
                onClick={() => {
                  setSelectedAgent(agent);
                  setShowDetailsModal(true);
                }}
              />
            ))}
          </div>

          {filteredAgents.length === 0 && (
            <Card className="p-8">
              <div className="text-center text-gray-500">
                <div className="text-4xl mb-4">🔍</div>
                <h3 className="text-lg font-medium mb-2">No Agents Found</h3>
                <p>Try adjusting your search criteria or create a new agent.</p>
              </div>
            </Card>
          )}
        </TabsContent>

        {/* Workflow Visualization Tab */}
        <TabsContent value="workflow">
          <Card className="p-6">
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    Agent Workflow Visualization
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400">
                    Real-time visualization of agent communication flows and collaboration patterns
                  </p>
                </div>
                <Button
                  onClick={() => {
                    if (agencyData) {
                      const updatedData = workflowService.simulateCommunicationUpdate(agencyData);
                      setAgencyData(updatedData);
                      toast({
                        title: "Workflow Updated",
                        description: "Communication flows have been refreshed",
                        variant: "success"
                      });
                    }
                  }}
                  variant="outline"
                  accentColor="blue"
                >
                  <span className="mr-2">🔄</span>
                  Refresh Flows
                </Button>
              </div>

              {/* Workflow Visualizer */}
              {agencyData ? (
                <div className="h-[600px] border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                  <AgencyWorkflowVisualizerWithProvider
                    agencyData={agencyData}
                    config={{
                      auto_layout: true,
                      show_labels: true,
                      show_metrics: true,
                      animation_speed: 1000,
                      node_size: 'medium',
                      edge_style: 'curved',
                      theme: 'dark',
                    }}
                    onEvent={(event) => {
                      console.log('Workflow event:', event);
                    }}
                  />
                </div>
              ) : (
                <Card className="p-12">
                  <div className="text-center text-gray-500">
                    <div className="text-4xl mb-4">🌐</div>
                    <h3 className="text-lg font-medium mb-2">No Workflow Data</h3>
                    <p className="mb-4">Create some agents first to visualize their workflow patterns.</p>
                    <Button onClick={() => setShowCreateModal(true)}>
                      Create First Agent
                    </Button>
                  </div>
                </Card>
              )}

              {/* Workflow Statistics */}
              {agencyData && (
                <Card className="p-4">
                  <h3 className="text-lg font-semibold mb-3 text-gray-900 dark:text-white">
                    Workflow Statistics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {agencyData.agents.length}
                      </div>
                      <div className="text-sm text-gray-600">Total Agents</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {agencyData.agents.filter(a => a.state === 'ACTIVE').length}
                      </div>
                      <div className="text-sm text-gray-600">Active Agents</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {agencyData.communication_flows.length}
                      </div>
                      <div className="text-sm text-gray-600">Communication Flows</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {agencyData.communication_flows.filter(f => f.status === 'active').length}
                      </div>
                      <div className="text-sm text-gray-600">Active Flows</div>
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </Card>
        </TabsContent>

        {/* Agent Pools Tab */}
        <TabsContent value="pools">
          <AgentPoolView agents={agents} />
        </TabsContent>

        {/* Intelligence Overview Tab */}
        <TabsContent value="intelligence">
          <IntelligenceOverview 
            overview={projectOverview}
            performanceMetrics={performanceMetrics}
          />
        </TabsContent>

        {/* Cost Management Tab */}
        <TabsContent value="costs">
          <CostDashboard 
            recommendations={costRecommendations}
            overview={projectOverview}
          />
        </TabsContent>

        {/* Real-Time Collaboration Tab */}
        <TabsContent value="collaboration">
          <RealTimeCollaboration agents={agents} />
        </TabsContent>

        {/* Knowledge Management Tab */}
        <TabsContent value="knowledge">
          <KnowledgeManagement agents={agents} />
        </TabsContent>
      </Tabs>

      {/* Modals */}
      <AgentCreationModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreateAgent={handleCreateAgent}
      />

      <AgentDetailsModal
        isOpen={showDetailsModal}
        agent={selectedAgent}
        onClose={() => setShowDetailsModal(false)}
        onStateChange={handleUpdateAgentState}
      />
    </div>
  );
};