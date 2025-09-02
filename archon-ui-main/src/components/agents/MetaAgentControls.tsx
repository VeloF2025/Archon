import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';

interface MetaAgentControlsProps {
  className?: string;
}

interface ManagedAgent {
  agent_id: string;
  agent_role: string;
  instance_name: string;
  state: string;
  tasks_completed: number;
  tasks_failed: number;
  spawn_time: number;
  last_activity: number;
  specialization?: any;
}

interface OrchestrationStatus {
  is_running: boolean;
  total_managed_agents: number;
  agents_by_state: Record<string, number>;
  agents_by_role: Record<string, number>;
  performance_metrics: Record<string, any>;
  recent_decisions: any[];
  max_agents: number;
  auto_scale: boolean;
}

export const MetaAgentControls: React.FC<MetaAgentControlsProps> = ({ className = '' }) => {
  const [orchestrationStatus, setOrchestrationStatus] = useState<OrchestrationStatus | null>(null);
  const [managedAgents, setManagedAgents] = useState<ManagedAgent[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  // Fetch orchestration status
  const fetchOrchestrationStatus = async () => {
    try {
      // Mock API call - in real implementation would call meta-agent service
      const mockStatus: OrchestrationStatus = {
        is_running: true,
        total_managed_agents: 8,
        agents_by_state: {
          idle: 5,
          active: 2,
          busy: 1,
          error: 0,
          spawning: 0,
          terminated: 0
        },
        agents_by_role: {
          python_backend_coder: 2,
          typescript_frontend_agent: 2,
          security_auditor: 1,
          test_generator: 1,
          performance_optimizer: 1,
          documentation_writer: 1
        },
        performance_metrics: {
          total_tasks_completed: 147,
          total_tasks_failed: 8,
          efficiency_score: 0.91,
          system_uptime: 7200
        },
        recent_decisions: [
          { decision: 'spawn_agent', timestamp: Date.now() - 300000, details: { role: 'security_auditor' } },
          { decision: 'scale_up', timestamp: Date.now() - 600000, details: { roles: ['test_generator'] } },
          { decision: 'optimize_workflow', timestamp: Date.now() - 900000, details: {} }
        ],
        max_agents: 100,
        auto_scale: true
      };

      // Mock managed agents
      const mockAgents: ManagedAgent[] = [
        {
          agent_id: 'agent_001',
          agent_role: 'python_backend_coder',
          instance_name: 'python_backend_coder_1735505523',
          state: 'active',
          tasks_completed: 23,
          tasks_failed: 1,
          spawn_time: Date.now() - 3600000,
          last_activity: Date.now() - 300000,
          specialization: { focus: 'security', additional_skills: ['vulnerability_scanning'] }
        },
        {
          agent_id: 'agent_002',
          agent_role: 'typescript_frontend_agent',
          instance_name: 'typescript_frontend_agent_1735505645',
          state: 'idle',
          tasks_completed: 18,
          tasks_failed: 0,
          spawn_time: Date.now() - 3200000,
          last_activity: Date.now() - 150000
        },
        {
          agent_id: 'agent_003',
          agent_role: 'security_auditor',
          instance_name: 'security_auditor_1735505789',
          state: 'busy',
          tasks_completed: 31,
          tasks_failed: 2,
          spawn_time: Date.now() - 2800000,
          last_activity: Date.now() - 30000
        }
      ];

      setOrchestrationStatus(mockStatus);
      setManagedAgents(mockAgents);
      setError(null);
    } catch (err) {
      setError('Failed to fetch orchestration status');
      console.error('Error fetching orchestration status:', err);
    }
  };

  // Meta-agent control actions
  const executeMetaAction = async (action: string, params?: any) => {
    setIsLoading(true);
    try {
      console.log(`Executing meta-agent action: ${action}`, params);
      
      // Mock API calls - in real implementation would call meta-agent endpoints
      switch (action) {
        case 'start_orchestration':
          console.log('Starting meta-agent orchestration system');
          break;
        case 'stop_orchestration':
          console.log('Stopping meta-agent orchestration system');
          break;
        case 'force_decision_cycle':
          console.log('Forcing immediate decision cycle');
          break;
        case 'spawn_agent':
          console.log(`Spawning new agent with role: ${params?.role}`);
          break;
        case 'terminate_agent':
          console.log(`Terminating agent: ${params?.agent_id}`);
          break;
        case 'toggle_auto_scale':
          console.log('Toggling auto-scaling');
          break;
        case 'emergency_stop_all':
          console.log('Emergency stop - terminating all agents');
          break;
        default:
          console.log(`Unknown meta-agent action: ${action}`);
      }
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Refresh status
      await fetchOrchestrationStatus();
      
    } catch (err) {
      setError(`Failed to execute action: ${action}`);
      console.error('Meta-agent action failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh status
  useEffect(() => {
    fetchOrchestrationStatus();
    const interval = setInterval(fetchOrchestrationStatus, 15000); // Every 15 seconds
    return () => clearInterval(interval);
  }, []);

  const getStateColor = (state: string) => {
    switch (state.toLowerCase()) {
      case 'active': return 'bg-green-500';
      case 'busy': return 'bg-blue-500';
      case 'idle': return 'bg-gray-400';
      case 'error': return 'bg-red-500';
      case 'spawning': return 'bg-yellow-500';
      case 'terminated': return 'bg-gray-600';
      default: return 'bg-gray-400';
    }
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  };

  const formatTimeAgo = (timestamp: number) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  if (!orchestrationStatus) {
    return (
      <Card className={`p-4 ${className}`}>
        <div className="flex items-center justify-center h-32">
          <div className="text-gray-500">Loading meta-agent controls...</div>
        </div>
      </Card>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Meta-Agent System Overview */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Meta-Agent Orchestration
          </h3>
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${orchestrationStatus.is_running ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium">
              {orchestrationStatus.is_running ? 'Running' : 'Stopped'}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{orchestrationStatus.total_managed_agents}</div>
            <div className="text-xs text-gray-500">Managed Agents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{orchestrationStatus.performance_metrics.total_tasks_completed}</div>
            <div className="text-xs text-gray-500">Tasks Completed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {Math.round((orchestrationStatus.performance_metrics.efficiency_score || 0) * 100)}%
            </div>
            <div className="text-xs text-gray-500">Efficiency</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">
              {formatUptime(orchestrationStatus.performance_metrics.system_uptime || 0)}
            </div>
            <div className="text-xs text-gray-500">Uptime</div>
          </div>
        </div>

        {/* Control Actions */}
        <div className="flex flex-wrap gap-2 mb-4">
          <Button
            size="sm"
            onClick={() => executeMetaAction('force_decision_cycle')}
            disabled={isLoading || !orchestrationStatus.is_running}
            className="bg-blue-600 hover:bg-blue-700"
          >
            🧠 Force Decision Cycle
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => executeMetaAction('toggle_auto_scale')}
            disabled={isLoading}
          >
            📊 Toggle Auto-Scale
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => executeMetaAction(orchestrationStatus.is_running ? 'stop_orchestration' : 'start_orchestration')}
            disabled={isLoading}
          >
            {orchestrationStatus.is_running ? '⏸️ Stop' : '▶️ Start'} Orchestration
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => executeMetaAction('emergency_stop_all')}
            disabled={isLoading}
          >
            🛑 Emergency Stop All
          </Button>
        </div>

        {/* Agent States Overview */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(orchestrationStatus.agents_by_state).map(([state, count]) => (
            <Badge key={state} variant="secondary" className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${getStateColor(state)}`}></div>
              <span>{state}: {count}</span>
            </Badge>
          ))}
        </div>
      </Card>

      {/* Dynamic Agent Spawning */}
      <Card className="p-4">
        <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white">
          Dynamic Agent Spawning
        </h4>
        
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-3">
          {Object.entries(orchestrationStatus.agents_by_role).map(([role, count]) => (
            <Button
              key={role}
              size="sm"
              variant="outline"
              onClick={() => executeMetaAction('spawn_agent', { role })}
              disabled={isLoading}
              className="text-xs flex justify-between"
            >
              <span>{role.replace(/_/g, ' ')}</span>
              <Badge variant="secondary" className="ml-1">
                {count}
              </Badge>
            </Button>
          ))}
        </div>

        <div className="text-xs text-gray-500 mb-2">
          Agents: {orchestrationStatus.total_managed_agents}/{orchestrationStatus.max_agents} 
          {orchestrationStatus.auto_scale && ' (Auto-scaling enabled)'}
        </div>
      </Card>

      {/* Managed Agents List */}
      <Card className="p-4">
        <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white">
          Managed Agent Instances
        </h4>
        
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {managedAgents.map((agent) => (
            <div
              key={agent.agent_id}
              className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                selectedAgent === agent.agent_id ? 'bg-blue-50 border-blue-300' : 'bg-gray-50 border-gray-200'
              }`}
              onClick={() => setSelectedAgent(selectedAgent === agent.agent_id ? null : agent.agent_id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${getStateColor(agent.state)}`}></div>
                  <span className="font-medium text-sm">{agent.agent_role.replace(/_/g, ' ')}</span>
                  {agent.specialization && (
                    <Badge variant="outline" className="text-xs">
                      {agent.specialization.focus}
                    </Badge>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-500">
                    ✅ {agent.tasks_completed} / ❌ {agent.tasks_failed}
                  </span>
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      executeMetaAction('terminate_agent', { agent_id: agent.agent_id });
                    }}
                    disabled={isLoading}
                    className="h-6 w-6 p-0"
                  >
                    ×
                  </Button>
                </div>
              </div>
              
              {selectedAgent === agent.agent_id && (
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-600 space-y-1">
                  <div><strong>Instance:</strong> {agent.instance_name}</div>
                  <div><strong>State:</strong> {agent.state}</div>
                  <div><strong>Last Activity:</strong> {formatTimeAgo(agent.last_activity)}</div>
                  <div><strong>Success Rate:</strong> {
                    agent.tasks_completed + agent.tasks_failed > 0 
                      ? Math.round((agent.tasks_completed / (agent.tasks_completed + agent.tasks_failed)) * 100)
                      : 0
                  }%</div>
                  {agent.specialization && (
                    <div><strong>Specialization:</strong> {JSON.stringify(agent.specialization, null, 2)}</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      {/* Recent Decisions */}
      <Card className="p-4">
        <h4 className="text-md font-semibold mb-3 text-gray-900 dark:text-white">
          Recent Meta-Agent Decisions
        </h4>
        
        <div className="space-y-2">
          {orchestrationStatus.recent_decisions.slice(0, 5).map((decision, index) => (
            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">{decision.decision.replace(/_/g, ' ')}</span>
                {decision.details && Object.keys(decision.details).length > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {Object.entries(decision.details)[0][1]}
                  </Badge>
                )}
              </div>
              <span className="text-xs text-gray-500">
                {formatTimeAgo(decision.timestamp)}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {error}
        </div>
      )}
    </div>
  );
};