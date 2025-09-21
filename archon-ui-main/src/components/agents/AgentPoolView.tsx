import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Search, Filter, Grid, List, Plus, Play, Pause, Settings } from 'lucide-react';
import { Agent } from './AgentDetailsModal';

interface AgentPoolViewProps {
  agents: Agent[];
  onCreateAgent: () => void;
  onViewAgent: (agent: Agent) => void;
  onToggleAgent: (agentId: string) => void;
  onEditAgent: (agent: Agent) => void;
}

const FILTER_OPTIONS = [
  { value: 'all', label: 'All Agents' },
  { value: 'active', label: 'Active' },
  { value: 'inactive', label: 'Inactive' },
  { value: 'running', label: 'Running' },
  { value: 'error', label: 'Error' },
];

const AGENT_TYPES = [
  { value: 'all', label: 'All Types' },
  { value: 'code-implementer', label: 'Code Implementer' },
  { value: 'system-architect', label: 'System Architect' },
  { value: 'security-auditor', label: 'Security Auditor' },
  { value: 'performance-optimizer', label: 'Performance Optimizer' },
];

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'running':
      return <Play className="w-4 h-4 text-green-500" />;
    case 'active':
      return <div className="w-2 h-2 rounded-full bg-green-500" />;
    case 'inactive':
      return <Pause className="w-4 h-4 text-gray-400" />;
    case 'error':
      return <div className="w-2 h-2 rounded-full bg-red-500" />;
    default:
      return <div className="w-2 h-2 rounded-full bg-gray-400" />;
  }
};

const getStatusColor = (status: string) => {
  switch (status) {
    case 'active':
      return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
    case 'running':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
    case 'inactive':
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
    case 'error':
      return 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400';
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900/20 dark:text-gray-400';
  }
};

export const AgentPoolView: React.FC<AgentPoolViewProps> = ({
  agents,
  onCreateAgent,
  onViewAgent,
  onToggleAgent,
  onEditAgent,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Filter agents
  const filteredAgents = agents.filter(agent => {
    const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         agent.description?.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || agent.status === statusFilter;
    const matchesType = typeFilter === 'all' || agent.type === typeFilter;
    
    return matchesSearch && matchesStatus && matchesType;
  });

  // Agent statistics
  const stats = {
    total: agents.length,
    active: agents.filter(a => a.status === 'active').length,
    running: agents.filter(a => a.status === 'running').length,
    inactive: agents.filter(a => a.status === 'inactive').length,
    error: agents.filter(a => a.status === 'error').length,
  };

  const AgentCard: React.FC<{ agent: Agent }> = ({ agent }) => (
    <Card className="hover:shadow-md transition-shadow cursor-pointer">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon(agent.status)}
            <CardTitle className="text-lg">{agent.name}</CardTitle>
          </div>
          <Badge className={getStatusColor(agent.status)}>
            {agent.status}
          </Badge>
        </div>
        <CardDescription>
          {agent.type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {agent.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-2">
            {agent.description}
          </p>
        )}
        
        {agent.metrics && (
          <div className="grid grid-cols-2 gap-2 mb-4 text-xs">
            <div>
              <span className="text-gray-500">Tasks:</span>{' '}
              <span className="font-medium">{agent.metrics.tasks_completed}</span>
            </div>
            <div>
              <span className="text-gray-500">Success:</span>{' '}
              <span className="font-medium">{Math.round(agent.metrics.success_rate * 100)}%</span>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onViewAgent(agent)}
            className="flex-1"
          >
            View Details
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onToggleAgent(agent.id)}
          >
            {agent.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onEditAgent(agent)}
          >
            <Settings className="w-4 h-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const AgentListItem: React.FC<{ agent: Agent }> = ({ agent }) => (
    <div className="flex items-center justify-between p-4 border-b hover:bg-gray-50 dark:hover:bg-gray-800">
      <div className="flex items-center gap-4 flex-1">
        {getStatusIcon(agent.status)}
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h3 className="font-medium">{agent.name}</h3>
            <Badge className={getStatusColor(agent.status)}>
              {agent.status}
            </Badge>
          </div>
          <p className="text-sm text-gray-500">
            {agent.type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </p>
          {agent.description && (
            <p className="text-sm text-gray-400 line-clamp-1 mt-1">{agent.description}</p>
          )}
        </div>
        {agent.metrics && (
          <div className="hidden md:flex gap-6 text-sm">
            <div>
              <span className="text-gray-500">Tasks:</span>{' '}
              <span className="font-medium">{agent.metrics.tasks_completed}</span>
            </div>
            <div>
              <span className="text-gray-500">Success:</span>{' '}
              <span className="font-medium">{Math.round(agent.metrics.success_rate * 100)}%</span>
            </div>
          </div>
        )}
      </div>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={() => onViewAgent(agent)}>
          View
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onToggleAgent(agent.id)}
        >
          {agent.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Agent Pool</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Manage your AI agents and monitor their performance
          </p>
        </div>
        <Button onClick={onCreateAgent}>
          <Plus className="w-4 h-4 mr-2" />
          Create Agent
        </Button>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold">{stats.total}</div>
            <div className="text-sm text-gray-500">Total Agents</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-green-600">{stats.active}</div>
            <div className="text-sm text-gray-500">Active</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-blue-600">{stats.running}</div>
            <div className="text-sm text-gray-500">Running</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-gray-600">{stats.inactive}</div>
            <div className="text-sm text-gray-500">Inactive</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-red-600">{stats.error}</div>
            <div className="text-sm text-gray-500">Error</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Search */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search agents..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
        <Select value={statusFilter} onValueChange={setStatusFilter}>
          {FILTER_OPTIONS.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </Select>
        <Select value={typeFilter} onValueChange={setTypeFilter}>
          {AGENT_TYPES.map(type => (
            <option key={type.value} value={type.value}>
              {type.label}
            </option>
          ))}
        </Select>
        <div className="flex gap-1">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('grid')}
          >
            <Grid className="w-4 h-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setViewMode('list')}
          >
            <List className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Agents Display */}
      <div className="space-y-4">
        {filteredAgents.length === 0 ? (
          <Card>
            <CardContent className="p-8 text-center">
              <div className="text-gray-400 mb-2">No agents found</div>
              <Button variant="outline" onClick={onCreateAgent}>
                Create your first agent
              </Button>
            </CardContent>
          </Card>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredAgents.map(agent => (
              <AgentCard key={agent.id} agent={agent} />
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="p-0">
              {filteredAgents.map(agent => (
                <AgentListItem key={agent.id} agent={agent} />
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default AgentPoolView;