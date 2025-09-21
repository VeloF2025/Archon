import React, { memo } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { AgentNodeData } from '../../types/workflowTypes';
import { AgentState, ModelTier, AgentType, AGENT_TYPE_ICONS } from '../../types/agentTypes';

interface AgentNodeProps extends NodeProps<AgentNodeData> {}

export const AgentNode: React.FC<AgentNodeProps> = memo(({ data, selected }) => {
  const { agent, position, is_selected, communication_stats } = data;

  // Status-based styling
  const getStatusColor = (state: AgentState): string => {
    switch (state) {
      case AgentState.ACTIVE:
        return 'from-green-500/20 to-green-600/30 border-green-500/50';
      case AgentState.IDLE:
        return 'from-yellow-500/20 to-yellow-600/30 border-yellow-500/50';
      case AgentState.HIBERNATED:
        return 'from-gray-500/20 to-gray-600/30 border-gray-500/50';
      case AgentState.CREATED:
        return 'from-blue-500/20 to-blue-600/30 border-blue-500/50';
      case AgentState.ARCHIVED:
        return 'from-red-500/20 to-red-600/30 border-red-500/50';
      default:
        return 'from-gray-500/20 to-gray-600/30 border-gray-500/50';
    }
  };

  // Model tier badge color
  const getTierColor = (tier: ModelTier): string => {
    switch (tier) {
      case ModelTier.OPUS:
        return 'bg-purple-500/20 text-purple-300 border-purple-500/50';
      case ModelTier.SONNET:
        return 'bg-blue-500/20 text-blue-300 border-blue-500/50';
      case ModelTier.HAIKU:
        return 'bg-green-500/20 text-green-300 border-green-500/50';
      default:
        return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
    }
  };

  // Activity level indicator
  const getActivityLevel = (): { level: string; color: string; width: string } => {
    if (!communication_stats) return { level: 'low', color: 'bg-gray-500', width: 'w-1/4' };

    const totalActivity = communication_stats.messages_sent + communication_stats.messages_received;

    if (totalActivity > 20) return { level: 'high', color: 'bg-green-500', width: 'w-full' };
    if (totalActivity > 10) return { level: 'medium', color: 'bg-yellow-500', width: 'w-2/3' };
    if (totalActivity > 5) return { level: 'moderate', color: 'bg-orange-500', width: 'w-1/2' };
    return { level: 'low', color: 'bg-gray-500', width: 'w-1/4' };
  };

  const activity = getActivityLevel();

  // Performance indicator
  const getPerformanceColor = (successRate: number): string => {
    if (successRate >= 0.9) return 'text-green-400';
    if (successRate >= 0.7) return 'text-yellow-400';
    if (successRate >= 0.5) return 'text-orange-400';
    return 'text-red-400';
  };

  // Node size based on model tier
  const getNodeSize = (): { width: number; height: number } => {
    switch (agent.model_tier) {
      case ModelTier.OPUS:
        return { width: 200, height: 120 };
      case ModelTier.SONNET:
        return { width: 180, height: 110 };
      case ModelTier.HAIKU:
        return { width: 160, height: 100 };
      default:
        return { width: 170, height: 105 };
    }
  };

  const nodeSize = getNodeSize();

  return (
    <div
      className={`
        relative rounded-lg border-2 backdrop-blur-md transition-all duration-300
        ${getStatusColor(agent.state)}
        ${selected || is_selected ? 'ring-2 ring-blue-400 scale-105' : 'hover:scale-102'}
        ${agent.state === AgentState.ACTIVE ? 'shadow-lg shadow-green-500/20' : ''}
      `}
      style={{
        width: nodeSize.width,
        height: nodeSize.height,
        background: `linear-gradient(135deg,
          ${agent.state === AgentState.ACTIVE ? 'rgba(16, 185, 129, 0.1)' :
            agent.state === AgentState.IDLE ? 'rgba(245, 158, 11, 0.1)' :
            'rgba(107, 114, 128, 0.1)'} 0%,
          rgba(0, 0, 0, 0.2) 100%
        )`,
      }}
    >
      {/* Connection Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-gray-400 !border-gray-600"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-gray-400 !border-gray-600"
      />

      {/* Activity Indicator (Animated pulse for active agents) */}
      {agent.state === AgentState.ACTIVE && (
        <div className="absolute -top-1 -right-1">
          <div className="relative">
            <div className="absolute inset-0 bg-green-500 rounded-full animate-ping opacity-75 w-4 h-4"></div>
            <div className="relative w-4 h-4 bg-green-500 rounded-full"></div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="h-full flex flex-col p-3 space-y-2">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-2">
            <div className="text-2xl">{AGENT_TYPE_ICONS[agent.agent_type] || '🤖'}</div>
            <div className="flex-1 min-w-0">
              <div className="text-white font-medium text-sm leading-tight truncate">
                {agent.name}
              </div>
              <div className="text-gray-400 text-xs truncate">
                {agent.agent_type.replace(/_/g, ' ')}
              </div>
            </div>
          </div>

          {/* Model Tier Badge */}
          <div className={`
            px-2 py-1 rounded text-xs font-medium border
            ${getTierColor(agent.model_tier)}
          `}>
            {agent.model_tier}
          </div>
        </div>

        {/* Status and Metrics */}
        <div className="flex-1 space-y-2">
          {/* Status */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">Status:</span>
            <span className={`
              px-2 py-0.5 rounded text-xs font-medium
              ${agent.state === AgentState.ACTIVE ? 'bg-green-500/20 text-green-400' :
                agent.state === AgentState.IDLE ? 'bg-yellow-500/20 text-yellow-400' :
                agent.state === AgentState.HIBERNATED ? 'bg-gray-500/20 text-gray-400' :
                agent.state === AgentState.CREATED ? 'bg-blue-500/20 text-blue-400' :
                'bg-red-500/20 text-red-400'}
            `}>
              {agent.state}
            </span>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="text-gray-400">Tasks</div>
              <div className="text-white font-medium">{agent.tasks_completed}</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">Success</div>
              <div className={`font-medium ${getPerformanceColor(agent.success_rate)}`}>
                {(agent.success_rate * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Activity Level */}
          {communication_stats && (
            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Activity:</span>
                <span className="text-gray-300 capitalize">{activity.level}</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div
                  className={`${activity.color} h-1.5 rounded-full transition-all duration-300`}
                  style={{ width: activity.width }}
                ></div>
              </div>
              <div className="flex justify-between text-xs text-gray-500">
                <span>↗ {communication_stats.messages_sent}</span>
                <span>↘ {communication_stats.messages_received}</span>
              </div>
            </div>
          )}

          {/* Resource Usage (if active) */}
          {agent.state === AgentState.ACTIVE && (
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">CPU:</span>
                <span className="text-gray-300">{agent.cpu_usage_percent.toFixed(0)}%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Memory:</span>
                <span className="text-gray-300">{(agent.memory_usage_mb / 1024).toFixed(1)}GB</span>
              </div>
            </div>
          )}
        </div>

        {/* Footer with Active Connections */}
        {communication_stats && communication_stats.active_connections > 0 && (
          <div className="pt-2 border-t border-gray-600/50">
            <div className="flex items-center justify-center space-x-1">
              <div className="flex space-x-0.5">
                {[...Array(Math.min(communication_stats.active_connections, 5))].map((_, i) => (
                  <div
                    key={i}
                    className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  ></div>
                ))}
                {communication_stats.active_connections > 5 && (
                  <div className="text-xs text-gray-400">+{communication_stats.active_connections - 5}</div>
                )}
              </div>
              <span className="text-xs text-gray-400">
                {communication_stats.active_connections} active
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Selection Indicator */}
      {(selected || is_selected) && (
        <div className="absolute inset-0 rounded-lg border-2 border-blue-400 pointer-events-none"></div>
      )}
    </div>
  );
});

AgentNode.displayName = 'AgentNode';