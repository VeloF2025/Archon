import React from 'react';
import { Card } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { AgentV3, AgentState, ModelTier } from '../../types/agentTypes';

interface AgentCardProps {
  agent: AgentV3;
  onStateChange: (newState: AgentState) => void;
  onClick: () => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({ agent, onStateChange, onClick }) => {
  const getStateColor = (state: AgentState) => {
    switch (state) {
      case AgentState.ACTIVE: return 'bg-green-100 text-green-800';
      case AgentState.IDLE: return 'bg-blue-100 text-blue-800';
      case AgentState.HIBERNATED: return 'bg-purple-100 text-purple-800';
      case AgentState.CREATED: return 'bg-gray-100 text-gray-800';
      case AgentState.ARCHIVED: return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTierColor = (tier: ModelTier) => {
    switch (tier) {
      case ModelTier.OPUS: return 'bg-gradient-to-r from-purple-500 to-pink-500';
      case ModelTier.SONNET: return 'bg-gradient-to-r from-blue-500 to-cyan-500';
      case ModelTier.HAIKU: return 'bg-gradient-to-r from-green-500 to-emerald-500';
      default: return 'bg-gray-500';
    }
  };

  const getAgentTypeIcon = (type: string) => {
    const iconMap: Record<string, string> = {
      'CODE_IMPLEMENTER': '⚡',
      'SYSTEM_ARCHITECT': '🏗️',
      'CODE_QUALITY_REVIEWER': '🔍',
      'TEST_COVERAGE_VALIDATOR': '🧪',
      'SECURITY_AUDITOR': '🛡️',
      'PERFORMANCE_OPTIMIZER': '🚀',
      'DEPLOYMENT_AUTOMATION': '🚢',
      'ANTIHALLUCINATION_VALIDATOR': '🎯',
      'UI_UX_OPTIMIZER': '🎨',
      'DATABASE_ARCHITECT': '🗄️',
      'DOCUMENTATION_GENERATOR': '📝',
      'CODE_REFACTORING_OPTIMIZER': '🔧',
      'STRATEGIC_PLANNER': '📋',
      'API_DESIGN_ARCHITECT': '🌐',
      'GENERAL_PURPOSE': '🤖'
    };
    return iconMap[type] || '🤖';
  };

  const formatAgentType = (type: string) => {
    return type.toLowerCase()
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getQuickActions = () => {
    const actions = [];
    
    if (agent.state === AgentState.IDLE) {
      actions.push(
        <Button
          key="activate"
          size="sm"
          variant="outline"
          onClick={(e) => {
            e.stopPropagation();
            onStateChange(AgentState.ACTIVE);
          }}
          className="text-xs"
        >
          ▶️ Activate
        </Button>
      );
    }
    
    if (agent.state === AgentState.ACTIVE) {
      actions.push(
        <Button
          key="idle"
          size="sm"
          variant="outline"
          onClick={(e) => {
            e.stopPropagation();
            onStateChange(AgentState.IDLE);
          }}
          className="text-xs"
        >
          ⏸️ Make Idle
        </Button>
      );
    }
    
    if ([AgentState.IDLE, AgentState.CREATED].includes(agent.state)) {
      actions.push(
        <Button
          key="hibernate"
          size="sm"
          variant="outline"
          onClick={(e) => {
            e.stopPropagation();
            onStateChange(AgentState.HIBERNATED);
          }}
          className="text-xs"
        >
          💤 Hibernate
        </Button>
      );
    }
    
    if (agent.state === AgentState.HIBERNATED) {
      actions.push(
        <Button
          key="wake"
          size="sm"
          variant="outline"
          onClick={(e) => {
            e.stopPropagation();
            onStateChange(AgentState.IDLE);
          }}
          className="text-xs"
        >
          🌅 Wake Up
        </Button>
      );
    }
    
    return actions;
  };

  const formatDate = (date: Date | string) => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const getActivityLevel = () => {
    if (!agent.last_active_at) return 'Never';
    
    const lastActive = new Date(agent.last_active_at);
    const now = new Date();
    const diff = now.getTime() - lastActive.getTime();
    
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  return (
    <Card 
      className="group hover:shadow-md transition-all duration-200 cursor-pointer border-l-4 hover:border-l-blue-500"
      onClick={onClick}
    >
      <div className="p-4 space-y-3">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-2">
            <div className="text-2xl">{getAgentTypeIcon(agent.agent_type)}</div>
            <div className="min-w-0 flex-1">
              <h3 className="font-semibold text-gray-900 dark:text-white truncate">
                {agent.name}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {formatAgentType(agent.agent_type)}
              </p>
            </div>
          </div>
          <div className="flex flex-col items-end space-y-1">
            <Badge className={getStateColor(agent.state)}>
              {agent.state}
            </Badge>
          </div>
        </div>

        {/* Intelligence Tier */}
        <div className="flex items-center space-x-2">
          <div 
            className={`px-2 py-1 rounded text-xs font-medium text-white ${getTierColor(agent.model_tier)}`}
          >
            {agent.model_tier}
          </div>
          <div className="text-xs text-gray-500">
            Intelligence Tier
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-gray-50 dark:bg-gray-800 rounded p-2">
            <div className="font-medium text-gray-900 dark:text-white">
              {agent.tasks_completed}
            </div>
            <div className="text-gray-600 dark:text-gray-400">Tasks Done</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-800 rounded p-2">
            <div className="font-medium text-gray-900 dark:text-white">
              {Math.round(parseFloat(agent.success_rate.toString()) * 100)}%
            </div>
            <div className="text-gray-600 dark:text-gray-400">Success Rate</div>
          </div>
        </div>

        {/* Resource Usage */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-gray-600">Memory</span>
            <span className="font-medium">{agent.memory_usage_mb}MB</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1">
            <div 
              className="bg-blue-500 h-1 rounded-full"
              style={{ width: `${Math.min(agent.memory_usage_mb / 10, 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-gray-600">CPU</span>
            <span className="font-medium">{agent.cpu_usage_percent}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1">
            <div 
              className="bg-green-500 h-1 rounded-full"
              style={{ width: `${parseFloat(agent.cpu_usage_percent.toString())}%` }}
            />
          </div>
        </div>

        {/* Last Activity */}
        <div className="text-xs text-gray-500 border-t pt-2">
          <div className="flex justify-between items-center">
            <span>Last active:</span>
            <span>{getActivityLevel()}</span>
          </div>
        </div>

        {/* Quick Actions */}
        {getQuickActions().length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2 border-t opacity-0 group-hover:opacity-100 transition-opacity">
            {getQuickActions()}
          </div>
        )}

        {/* Status Indicators */}
        <div className="flex justify-between items-center text-xs pt-2 border-t">
          <div className="flex space-x-2">
            {agent.state === AgentState.ACTIVE && (
              <div className="flex items-center space-x-1 text-green-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span>Online</span>
              </div>
            )}
            {agent.state === AgentState.HIBERNATED && (
              <div className="flex items-center space-x-1 text-purple-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full" />
                <span>Hibernating</span>
              </div>
            )}
          </div>
          <div className="text-gray-500">
            Created {formatDate(agent.created_at)}
          </div>
        </div>
      </div>
    </Card>
  );
};