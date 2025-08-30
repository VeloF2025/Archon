import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Play, 
  Pause, 
  StopCircle, 
  RotateCcw, 
  Plus, 
  Settings, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap
} from 'lucide-react';

interface AgentControlPanelProps {
  systemStatus: any;
  onAgentAction: (action: string, agentId?: string, params?: any) => Promise<void>;
}

const AgentControlPanel: React.FC<AgentControlPanelProps> = ({ 
  systemStatus, 
  onAgentAction 
}) => {
  const [selectedRole, setSelectedRole] = useState<string>('');
  const [spawnCount, setSpawnCount] = useState<number>(1);
  const [isLoading, setIsLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  
  const agentRoles = [
    'python_backend_coder',
    'typescript_frontend_agent', 
    'security_auditor',
    'test_generator',
    'documentation_writer',
    'api_integrator',
    'database_designer',
    'ui_ux_designer',
    'devops_engineer',
    'performance_optimizer',
    'code_reviewer',
    'refactoring_specialist',
    'configuration_manager',
    'deployment_coordinator',
    'monitoring_agent',
    'quality_assurance',
    'technical_writer',
    'integration_tester',
    'hrm_reasoning_agent',
    'data_analyst',
    'system_architect'
  ];
  
  const handleAction = async (action: string, params?: any) => {
    try {
      setIsLoading(action);
      await onAgentAction(action, undefined, params);
      setMessage({ type: 'success', text: `${action} completed successfully` });
    } catch (error) {
      setMessage({ type: 'error', text: `${action} failed: ${error}` });
    } finally {
      setIsLoading(null);
      setTimeout(() => setMessage(null), 3000);
    }
  };
  
  const handleSpawnAgent = async () => {
    if (!selectedRole) return;
    
    try {
      setIsLoading('spawn');
      for (let i = 0; i < spawnCount; i++) {
        await onAgentAction('spawn', selectedRole);
      }
      setMessage({ 
        type: 'success', 
        text: `Spawned ${spawnCount} ${selectedRole} agent(s)` 
      });
      setSelectedRole('');
      setSpawnCount(1);
    } catch (error) {
      setMessage({ type: 'error', text: `Failed to spawn agents: ${error}` });
    } finally {
      setIsLoading(null);
      setTimeout(() => setMessage(null), 3000);
    }
  };
  
  const formatRoleName = (role: string) => {
    return role
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };
  
  return (
    <div className="space-y-6">
      {message && (
        <Alert className={message.type === 'success' ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}
      
      {/* Global Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="w-5 h-5" />
            <span>System Controls</span>
          </CardTitle>
          <CardDescription>
            Global agent system management and monitoring controls
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Button
              onClick={() => handleAction('start_monitoring')}
              disabled={isLoading === 'start_monitoring'}
              className="flex items-center space-x-2"
            >
              <Play className="w-4 h-4" />
              <span>Start All</span>
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleAction('pause_system')}
              disabled={isLoading === 'pause_system'}
              className="flex items-center space-x-2"
            >
              <Pause className="w-4 h-4" />
              <span>Pause</span>
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleAction('restart_system')}
              disabled={isLoading === 'restart_system'}
              className="flex items-center space-x-2"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Restart</span>
            </Button>
            
            <Button
              variant="destructive"
              onClick={() => handleAction('emergency_stop')}
              disabled={isLoading === 'emergency_stop'}
              className="flex items-center space-x-2"
            >
              <StopCircle className="w-4 h-4" />
              <span>Stop All</span>
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {/* Agent Spawning */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Plus className="w-5 h-5" />
            <span>Spawn New Agents</span>
          </CardTitle>
          <CardDescription>
            Create new agent instances for specific roles
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Agent Role</label>
                <Select value={selectedRole} onValueChange={setSelectedRole}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select agent role" />
                  </SelectTrigger>
                  <SelectContent>
                    {agentRoles.map((role) => (
                      <SelectItem key={role} value={role}>
                        {formatRoleName(role)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Count</label>
                <Input
                  type="number"
                  min="1"
                  max="5"
                  value={spawnCount}
                  onChange={(e) => setSpawnCount(parseInt(e.target.value) || 1)}
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Action</label>
                <Button
                  onClick={handleSpawnAgent}
                  disabled={!selectedRole || isLoading === 'spawn'}
                  className="w-full flex items-center space-x-2"
                >
                  <Plus className="w-4 h-4" />
                  <span>Spawn Agent(s)</span>
                </Button>
              </div>
            </div>
            
            {/* Quick Spawn Buttons */}
            <div className="border-t pt-4">
              <div className="text-sm font-medium mb-2">Quick Spawn Popular Roles:</div>
              <div className="flex flex-wrap gap-2">
                {['python_backend_coder', 'typescript_frontend_agent', 'security_auditor', 'test_generator'].map((role) => (
                  <Button
                    key={role}
                    variant="outline"
                    size="sm"
                    onClick={() => handleAction('spawn', { role })}
                    disabled={isLoading !== null}
                    className="text-xs"
                  >
                    <Plus className="w-3 h-3 mr-1" />
                    {formatRoleName(role)}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Resource Management */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="w-5 h-5" />
            <span>Resource Management</span>
          </CardTitle>
          <CardDescription>
            Monitor and manage system resources and scaling
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {systemStatus?.agent_pool?.total_agents || 0}
              </div>
              <div className="text-sm text-muted-foreground">Active Agents</div>
            </div>
            
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {systemStatus?.agent_pool?.resource_usage?.total_memory_mb 
                  ? (systemStatus.agent_pool.resource_usage.total_memory_mb / 1024).toFixed(1)
                  : '0'
                }GB
              </div>
              <div className="text-sm text-muted-foreground">Memory Used</div>
            </div>
            
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {systemStatus?.agent_pool?.resource_usage?.average_cpu_percent?.toFixed(1) || '0'}%
              </div>
              <div className="text-sm text-muted-foreground">Avg CPU</div>
            </div>
            
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {systemStatus?.agent_pool?.task_statistics?.currently_busy || 0}
              </div>
              <div className="text-sm text-muted-foreground">Busy Agents</div>
            </div>
          </div>
          
          <div className="mt-4 flex space-x-2">
            <Button
              variant="outline"
              onClick={() => handleAction('auto_scale')}
              disabled={isLoading === 'auto_scale'}
              className="flex items-center space-x-2"
            >
              <Zap className="w-4 h-4" />
              <span>Auto Scale</span>
            </Button>
            
            <Button
              variant="outline"
              onClick={() => handleAction('cleanup_idle')}
              disabled={isLoading === 'cleanup_idle'}
              className="flex items-center space-x-2"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Cleanup Idle</span>
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {/* Agent Role Status */}
      <Card>
        <CardHeader>
          <CardTitle>Role Status Overview</CardTitle>
          <CardDescription>Current status of each agent role</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {systemStatus?.agent_pool?.role_distribution && 
              Object.entries(systemStatus.agent_pool.role_distribution).map(([role, stats]: [string, any]) => (
                <div key={role} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex-grow">
                    <div className="font-medium">{formatRoleName(role)}</div>
                    <div className="text-sm text-muted-foreground">
                      {stats.count} total • {stats.idle} idle • {stats.busy} busy
                      {stats.error > 0 && ` • ${stats.error} error`}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {stats.busy > 0 && (
                      <Badge variant="default" className="flex items-center space-x-1">
                        <Play className="w-3 h-3" />
                        <span>{stats.busy}</span>
                      </Badge>
                    )}
                    {stats.idle > 0 && (
                      <Badge variant="secondary" className="flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>{stats.idle}</span>
                      </Badge>
                    )}
                    {stats.error > 0 && (
                      <Badge variant="destructive" className="flex items-center space-x-1">
                        <AlertTriangle className="w-3 h-3" />
                        <span>{stats.error}</span>
                      </Badge>
                    )}
                    
                    <Dialog>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm">
                          <Settings className="w-3 h-3" />
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>{formatRoleName(role)} Controls</DialogTitle>
                          <DialogDescription>
                            Manage agents for the {formatRoleName(role)} role
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4">
                          <div className="grid grid-cols-2 gap-2">
                            <Button
                              onClick={() => handleAction('spawn', { role })}
                              disabled={isLoading !== null}
                              className="flex items-center space-x-2"
                            >
                              <Plus className="w-4 h-4" />
                              <span>Spawn One</span>
                            </Button>
                            
                            <Button
                              variant="outline"
                              onClick={() => handleAction('restart_role', { role })}
                              disabled={isLoading !== null}
                              className="flex items-center space-x-2"
                            >
                              <RotateCcw className="w-4 h-4" />
                              <span>Restart All</span>
                            </Button>
                          </div>
                          
                          {stats.error > 0 && (
                            <Button
                              variant="destructive"
                              onClick={() => handleAction('terminate_error_agents', { role })}
                              disabled={isLoading !== null}
                              className="w-full flex items-center space-x-2"
                            >
                              <StopCircle className="w-4 h-4" />
                              <span>Terminate Error Agents</span>
                            </Button>
                          )}
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>
                </div>
              ))
            }
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AgentControlPanel;