import React, { useState, useEffect } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { useSocket } from '../hooks/useSocket';
import { useToast } from '../hooks/useToast';
import { agentManagementService } from '../services/agentManagementService';
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
import { Knowledge管理 } from '../components/agents/KnowledgeManagement';

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

  const { toast } = useToast();
  const socket = useSocket();

  // Load agents and analytics data
  useEffect(() => {
    loadAgentsData();
  }, []);

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

    // Subscribe to agent lifecycle events
    socket.on('agent_state_changed', handleAgentUpdate);
    socket.on('agent_created', handleNewAgent);
    socket.on('agent_archived', handleAgentRemoved);
    socket.on('agent_performance_update', handleAgentUpdate);
    
    return () => {
      socket.off('agent_state_changed', handleAgentUpdate);
      socket.off('agent_created', handleNewAgent);
      socket.off('agent_archived', handleAgentRemoved);
      socket.off('agent_performance_update', handleAgentUpdate);
    };
  }, [socket, toast]);

  const loadAgentsData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [
        agentsData,
        metricsData,
        overviewData,
        recommendationsData
      ] = await Promise.all([
        agentManagementService.getAgents(),
        agentManagementService.getAgentPerformanceMetrics(),
        agentManagementService.getProjectIntelligenceOverview(),
        agentManagementService.getCostOptimizationRecommendations()
      ]);

      setAgents(agentsData);
      setPerformanceMetrics(metricsData);
      setProjectOverview(overviewData);
      setCostRecommendations(recommendationsData);

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
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="agents">🤖 Agents</TabsTrigger>
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