import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AgentQuickActions } from './AgentQuickActions';
import { MetaAgentControls } from './MetaAgentControls';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  Users, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Pause, 
  Play, 
  RotateCcw,
  CPU,
  Memory,
  Zap
} from 'lucide-react';

interface AgentInstance {
  agent_id: string;
  role: string;
  name: string;
  state: 'spawning' | 'idle' | 'active' | 'busy' | 'error' | 'terminated';
  spawn_time: number;
  last_activity: number;
  current_task_id?: string;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  completed_tasks: number;
  failed_tasks: number;
  error_message?: string;
}

interface AgentPoolStatus {
  total_agents: number;
  max_total_agents: number;
  agent_states: Record<string, number>;
  role_distribution: Record<string, {
    count: number;
    idle: number;
    busy: number;
    error: number;
  }>;
  resource_usage: {
    total_memory_mb: number;
    average_cpu_percent: number;
  };
  task_statistics: {
    total_completed: number;
    total_failed: number;
    currently_busy: number;
  };
}

interface TriggerEvent {
  event_id: string;
  event_type: string;
  file_path: string;
  timestamp: number;
  triggered_agents: string[];
}

interface SystemStatus {
  orchestrator: {
    active_executions: number;
    total_executions: number;
    last_execution?: string;
  };
  agent_pool: AgentPoolStatus;
  basic_triggers: {
    events_processed: number;
    agents_triggered: number;
    triggers_blocked_by_cooldown: number;
    last_activity?: number;
  };
  advanced_rules: {
    total_rules: number;
    enabled_rules: number;
    active_executions: number;
    pending_batches: Record<string, number>;
  };
}

const AgentDashboard: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [recentEvents, setRecentEvents] = useState<TriggerEvent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Fetch system status
  const fetchSystemStatus = useCallback(async () => {
    try {
      // In real implementation, these would be actual API calls
      // For now, we'll simulate the data structure
      const mockStatus: SystemStatus = {
        orchestrator: {
          active_executions: 2,
          total_executions: 147,
          last_execution: "exec_1234567890"
        },
        agent_pool: {
          total_agents: 15,
          max_total_agents: 20,
          agent_states: {
            idle: 8,
            active: 2,
            busy: 4,
            error: 1,
            spawning: 0,
            terminated: 0
          },
          role_distribution: {
            python_backend_coder: { count: 3, idle: 1, busy: 2, error: 0 },
            typescript_frontend_agent: { count: 2, idle: 1, busy: 1, error: 0 },
            security_auditor: { count: 2, idle: 2, busy: 0, error: 0 },
            test_generator: { count: 3, idle: 2, busy: 1, error: 0 },
            documentation_writer: { count: 2, idle: 1, busy: 0, error: 1 },
            api_integrator: { count: 1, idle: 1, busy: 0, error: 0 },
            devops_engineer: { count: 1, idle: 0, busy: 0, error: 0 },
            ui_ux_designer: { count: 1, idle: 0, busy: 0, error: 0 }
          },
          resource_usage: {
            total_memory_mb: 2847.5,
            average_cpu_percent: 12.3
          },
          task_statistics: {
            total_completed: 89,
            total_failed: 3,
            currently_busy: 4
          }
        },
        basic_triggers: {
          events_processed: 234,
          agents_triggered: 67,
          triggers_blocked_by_cooldown: 12,
          last_activity: Date.now() / 1000
        },
        advanced_rules: {
          total_rules: 8,
          enabled_rules: 7,
          active_executions: 2,
          pending_batches: {
            "security_critical_files": 2,
            "auto_test_generation": 4
          }
        }
      };
      
      setSystemStatus(mockStatus);
      setError(null);
    } catch (err) {
      setError('Failed to fetch system status');
      console.error('Error fetching system status:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Fetch recent events
  const fetchRecentEvents = useCallback(async () => {
    try {
      // Mock recent events data
      const mockEvents: TriggerEvent[] = [
        {
          event_id: "evt_001",
          event_type: "file_modified",
          file_path: "src/services/auth.py",
          timestamp: Date.now() / 1000 - 30,
          triggered_agents: ["security_auditor", "test_generator"]
        },
        {
          event_id: "evt_002", 
          event_type: "file_created",
          file_path: "components/LoginForm.tsx",
          timestamp: Date.now() / 1000 - 120,
          triggered_agents: ["typescript_frontend_agent", "ui_ux_designer"]
        },
        {
          event_id: "evt_003",
          event_type: "file_modified", 
          file_path: "requirements.txt",
          timestamp: Date.now() / 1000 - 180,
          triggered_agents: ["security_auditor", "python_backend_coder"]
        }
      ];
      
      setRecentEvents(mockEvents);
    } catch (err) {
      console.error('Error fetching recent events:', err);
    }
  }, []);
  
  // Auto-refresh data
  useEffect(() => {
    fetchSystemStatus();
    fetchRecentEvents();
    
    const interval = setInterval(() => {
      fetchSystemStatus();
      fetchRecentEvents();
    }, 5000); // Refresh every 5 seconds
    
    return () => clearInterval(interval);
  }, [fetchSystemStatus, fetchRecentEvents]);
  
  const getStateIcon = (state: string) => {
    switch (state) {
      case 'idle': return <Pause className="w-4 h-4" />;
      case 'active': case 'busy': return <Play className="w-4 h-4" />;
      case 'error': return <XCircle className="w-4 h-4" />;
      case 'spawning': return <RotateCcw className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };
  
  const getStateBadgeVariant = (state: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (state) {
      case 'busy': case 'active': return 'default';
      case 'idle': return 'secondary';
      case 'error': return 'destructive';
      default: return 'outline';
    }
  };
  
  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    const diffMinutes = Math.floor(diffSeconds / 60);
    
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    return date.toLocaleTimeString();
  };
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertTriangle className="w-8 h-8 text-red-500 mx-auto mb-2" />
          <p className="text-red-600">{error}</p>
          <Button onClick={fetchSystemStatus} className="mt-2">
            Retry
          </Button>
        </div>
      </div>
    );
  }
  
  if (!systemStatus) return null;
  
  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Agent Dashboard</h1>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="flex items-center space-x-1">
            <Activity className="w-3 h-3" />
            <span>Live</span>
          </Badge>
          <Button onClick={fetchSystemStatus} size="sm">
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>
      </div>
      
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {systemStatus.agent_pool.total_agents}/{systemStatus.agent_pool.max_total_agents}
            </div>
            <p className="text-xs text-muted-foreground">
              {((systemStatus.agent_pool.total_agents / systemStatus.agent_pool.max_total_agents) * 100).toFixed(0)}% capacity
            </p>
            <Progress 
              value={(systemStatus.agent_pool.total_agents / systemStatus.agent_pool.max_total_agents) * 100} 
              className="mt-2"
            />
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Tasks</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{systemStatus.agent_pool.task_statistics.currently_busy}</div>
            <p className="text-xs text-muted-foreground">
              {systemStatus.orchestrator.active_executions} orchestrated
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <Memory className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(systemStatus.agent_pool.resource_usage.total_memory_mb / 1024).toFixed(1)}GB
            </div>
            <p className="text-xs text-muted-foreground">
              Avg {systemStatus.agent_pool.resource_usage.average_cpu_percent.toFixed(1)}% CPU
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(
                (systemStatus.agent_pool.task_statistics.total_completed / 
                (systemStatus.agent_pool.task_statistics.total_completed + systemStatus.agent_pool.task_statistics.total_failed)) * 100
              ).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {systemStatus.agent_pool.task_statistics.total_completed} completed, {systemStatus.agent_pool.task_statistics.total_failed} failed
            </p>
          </CardContent>
        </Card>
      </div>
      
      {/* Quick Actions Panel */}
      <AgentQuickActions className="max-w-2xl mx-auto" />
      
      <Tabs defaultValue="agents" className="space-y-4">
        <TabsList>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="meta-agent">Meta-Agent</TabsTrigger>
          <TabsTrigger value="triggers">Triggers</TabsTrigger>
          <TabsTrigger value="events">Recent Events</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>
        
        <TabsContent value="agents" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Agent States Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Agent States</CardTitle>
                <CardDescription>Current status distribution</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(systemStatus.agent_pool.agent_states).map(([state, count]) => (
                    count > 0 && (
                      <div key={state} className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getStateIcon(state)}
                          <span className="capitalize">{state}</span>
                        </div>
                        <Badge variant={getStateBadgeVariant(state)}>{count}</Badge>
                      </div>
                    )
                  ))}
                </div>
              </CardContent>
            </Card>
            
            {/* Role Distribution */}
            <Card>
              <CardHeader>
                <CardTitle>Role Distribution</CardTitle>
                <CardDescription>Agents by specialized role</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {Object.entries(systemStatus.agent_pool.role_distribution).map(([role, stats]) => (
                    <div key={role} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          {role.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {stats.count} total
                        </span>
                      </div>
                      <div className="flex space-x-1">
                        {stats.idle > 0 && (
                          <Badge variant="secondary" className="text-xs px-1 py-0">
                            {stats.idle} idle
                          </Badge>
                        )}
                        {stats.busy > 0 && (
                          <Badge variant="default" className="text-xs px-1 py-0">
                            {stats.busy} busy
                          </Badge>
                        )}
                        {stats.error > 0 && (
                          <Badge variant="destructive" className="text-xs px-1 py-0">
                            {stats.error} error
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="meta-agent" className="space-y-4">
          <MetaAgentControls />
        </TabsContent>
        
        <TabsContent value="triggers" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Trigger Statistics</CardTitle>
                <CardDescription>File monitoring and agent triggering</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Events Processed</span>
                    <Badge variant="outline">{systemStatus.basic_triggers.events_processed}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Agents Triggered</span>
                    <Badge variant="outline">{systemStatus.basic_triggers.agents_triggered}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Blocked by Cooldown</span>
                    <Badge variant="secondary">{systemStatus.basic_triggers.triggers_blocked_by_cooldown}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Last Activity</span>
                    <Badge variant="outline">
                      {systemStatus.basic_triggers.last_activity 
                        ? formatTimestamp(systemStatus.basic_triggers.last_activity)
                        : 'None'
                      }
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Advanced Rules</CardTitle>
                <CardDescription>Sophisticated triggering patterns</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Rules</span>
                    <Badge variant="outline">{systemStatus.advanced_rules.total_rules}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Enabled Rules</span>
                    <Badge variant="default">{systemStatus.advanced_rules.enabled_rules}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Executions</span>
                    <Badge variant="default">{systemStatus.advanced_rules.active_executions}</Badge>
                  </div>
                  <div className="space-y-2">
                    <span className="text-sm font-medium">Pending Batches:</span>
                    {Object.entries(systemStatus.advanced_rules.pending_batches).map(([rule, count]) => (
                      count > 0 && (
                        <div key={rule} className="flex justify-between text-sm">
                          <span className="text-muted-foreground">{rule}</span>
                          <Badge variant="secondary" className="text-xs">{count}</Badge>
                        </div>
                      )
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="events" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Trigger Events</CardTitle>
              <CardDescription>Latest file system events and agent triggers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentEvents.map((event) => (
                  <div key={event.event_id} className="flex items-center space-x-4 p-3 border rounded-lg">
                    <div className="flex-shrink-0">
                      {getStateIcon(event.event_type)}
                    </div>
                    <div className="flex-grow">
                      <div className="font-medium">{event.file_path}</div>
                      <div className="text-sm text-muted-foreground">
                        {event.event_type.replace('_', ' ')} • {formatTimestamp(event.timestamp)}
                      </div>
                    </div>
                    <div className="flex-shrink-0 space-x-1">
                      {event.triggered_agents.map((agent) => (
                        <Badge key={agent} variant="outline" className="text-xs">
                          {agent.replace(/_/g, ' ').slice(0, 10)}...
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Resource Utilization</CardTitle>
                <CardDescription>System resource usage by agents</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm">Memory Usage</span>
                      <span className="text-sm">{(systemStatus.agent_pool.resource_usage.total_memory_mb / 1024).toFixed(1)}GB</span>
                    </div>
                    <Progress value={Math.min((systemStatus.agent_pool.resource_usage.total_memory_mb / 4096) * 100, 100)} />
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-sm">CPU Usage</span>
                      <span className="text-sm">{systemStatus.agent_pool.resource_usage.average_cpu_percent.toFixed(1)}%</span>
                    </div>
                    <Progress value={systemStatus.agent_pool.resource_usage.average_cpu_percent} />
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Task Performance</CardTitle>
                <CardDescription>Task execution metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600">
                      {systemStatus.agent_pool.task_statistics.total_completed}
                    </div>
                    <p className="text-sm text-muted-foreground">Tasks Completed</p>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Failed Tasks</span>
                    <span className="text-red-600">{systemStatus.agent_pool.task_statistics.total_failed}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Currently Running</span>
                    <span className="text-blue-600">{systemStatus.agent_pool.task_statistics.currently_busy}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AgentDashboard;