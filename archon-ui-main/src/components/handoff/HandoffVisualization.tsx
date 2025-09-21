/**
 * Handoff Visualization Component
 *
 * Real-time visualization of agent handoffs with analytics dashboard
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  HandoffVisualizationProps,
  ActiveHandoff,
  AgentHandoffState,
  HandoffHistoryEntry,
  HandoffStrategy,
  HandoffStatus,
  HANDOFF_STRATEGY_LABELS,
  HANDOFF_STATUS_COLORS
} from '../../types/handoffTypes';
import { handoffService } from '../../services/handoffService';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

const HandoffVisualization: React.FC<HandoffVisualizationProps> = ({
  projectId,
  refreshInterval = 5000,
  maxHistoryItems = 20,
  showMetrics = true,
  showControls = true,
  onHandoffSelect,
  onAgentSelect
}) => {
  const [activeHandoffs, setActiveHandoffs] = useState<ActiveHandoff[]>([]);
  const [handoffHistory, setHandoffHistory] = useState<HandoffHistoryEntry[]>([]);
  const [agentStates, setAgentStates] = useState<AgentHandoffState[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [selectedHandoff, setSelectedHandoff] = useState<ActiveHandoff | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentHandoffState | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Load handoff data
  const loadHandoffData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [activeData, historyData, visualizationData] = await Promise.all([
        handoffService.getActiveHandoffs(projectId),
        handoffService.getHandoffHistory(projectId, maxHistoryItems),
        handoffService.getHandoffVisualization(projectId)
      ]);

      setActiveHandoffs(activeData);
      setHandoffHistory(historyData.slice(0, maxHistoryItems));
      setAgentStates(visualizationData.agent_states);
      setLastRefresh(new Date());
    } catch (err) {
      console.error('Error loading handoff data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load handoff data');
    } finally {
      setIsLoading(false);
    }
  }, [projectId, maxHistoryItems]);

  // Auto-refresh setup
  useEffect(() => {
    loadHandoffData();

    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(loadHandoffData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadHandoffData, autoRefresh, refreshInterval]);

  // Handle handoff selection
  const handleHandoffSelect = (handoff: ActiveHandoff) => {
    setSelectedHandoff(handoff);
    onHandoffSelect?.(handoff);
  };

  // Handle agent selection
  const handleAgentSelect = (agent: AgentHandoffState) => {
    setSelectedAgent(agent);
    onAgentSelect?.(agent);
  };

  // Get status color
  const getStatusColor = (status: HandoffStatus): string => {
    return HANDOFF_STATUS_COLORS[status] || '#64748b';
  };

  // Get agent status color
  const getAgentStatusColor = (status: string): string => {
    const colors: Record<string, string> = {
      available: '#10b981', // green
      busy: '#f59e0b', // amber
      handing_off: '#3b82f6', // blue
      receiving: '#8b5cf6', // purple
      offline: '#6b7280' // gray
    };
    return colors[status] || '#64748b';
  };

  // Format duration
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  // Calculate metrics
  const calculateMetrics = () => {
    const todayHandoffs = handoffHistory.filter(h => {
      const handoffDate = new Date(h.timestamp);
      const today = new Date();
      return handoffDate.toDateString() === today.toDateString();
    });

    const successRate = todayHandoffs.length > 0
      ? todayHandoffs.filter(h => h.success).length / todayHandoffs.length
      : 0;

    const avgHandoffTime = todayHandoffs.length > 0
      ? todayHandoffs.reduce((sum, h) => sum + h.duration, 0) / todayHandoffs.length
      : 0;

    const strategyCounts = todayHandoffs.reduce((acc, h) => {
      acc[h.strategy] = (acc[h.strategy] || 0) + 1;
      return acc;
    }, {} as Record<HandoffStrategy, number>);

    const mostUsedStrategy = Object.entries(strategyCounts).reduce((a, b) =>
      a[1] > b[1] ? a : b
    )?.[0] || HandoffStrategy.SEQUENTIAL;

    return {
      totalHandoffsToday: todayHandoffs.length,
      successRate: successRate * 100,
      avgHandoffTime,
      mostUsedStrategy,
      activeHandoffs: activeHandoffs.length
    };
  };

  const metrics = calculateMetrics();

  if (isLoading && activeHandoffs.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading handoff visualization...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error Loading Handoff Data</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <Button onClick={loadHandoffData} variant="outline" size="sm">
                Retry
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      {showControls && (
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Agent Handoff Visualization</h2>
            <p className="text-gray-600">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          </div>
          <div className="flex space-x-2">
            <Button
              onClick={loadHandoffData}
              variant="outline"
              size="sm"
            >
              Refresh Now
            </Button>
            <Button
              onClick={() => setAutoRefresh(!autoRefresh)}
              variant={autoRefresh ? "default" : "outline"}
              size="sm"
            >
              {autoRefresh ? 'Auto Refresh: ON' : 'Auto Refresh: OFF'}
            </Button>
          </div>
        </div>
      )}

      {/* Metrics Dashboard */}
      {showMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card>
            <div className="p-4">
              <div className="text-2xl font-bold text-blue-600">{metrics.totalHandoffsToday}</div>
              <div className="text-sm text-gray-600">Handoffs Today</div>
            </div>
          </Card>
          <Card>
            <div className="p-4">
              <div className="text-2xl font-bold text-green-600">{metrics.successRate.toFixed(1)}%</div>
              <div className="text-sm text-gray-600">Success Rate</div>
            </div>
          </Card>
          <Card>
            <div className="p-4">
              <div className="text-2xl font-bold text-purple-600">{formatDuration(metrics.avgHandoffTime)}</div>
              <div className="text-sm text-gray-600">Avg Duration</div>
            </div>
          </Card>
          <Card>
            <div className="p-4">
              <div className="text-2xl font-bold text-orange-600">{activeHandoffs.length}</div>
              <div className="text-sm text-gray-600">Active Handoffs</div>
            </div>
          </Card>
          <Card>
            <div className="p-4">
              <div className="text-sm font-semibold text-gray-800 truncate">
                {HANDOFF_STRATEGY_LABELS[metrics.mostUsedStrategy]}
              </div>
              <div className="text-sm text-gray-600">Most Used Strategy</div>
            </div>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Handoffs */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Handoffs</h3>
            {activeHandoffs.length === 0 ? (
              <div className="text-center py-8">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
                <p className="mt-2 text-gray-600">No active handoffs</p>
              </div>
            ) : (
              <div className="space-y-3">
                {activeHandoffs.map((handoff) => (
                  <div
                    key={handoff.handoff_id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedHandoff?.handoff_id === handoff.handoff_id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => handleHandoffSelect(handoff)}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="font-medium">{handoff.source_agent}</span>
                          <svg className="w-4 h-4 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                          </svg>
                          <span className="font-medium">{handoff.target_agent}</span>
                        </div>
                        <p className="text-sm text-gray-600 truncate">{handoff.task_description}</p>
                      </div>
                      <div className="flex items-center space-x-2 ml-4">
                        <span
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                          style={{ backgroundColor: `${getStatusColor(handoff.status)}20`, color: getStatusColor(handoff.status) }}
                        >
                          {handoff.status.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-500">
                      <span>{HANDOFF_STRATEGY_LABELS[handoff.strategy]}</span>
                      <span>{handoff.progress}% Complete</span>
                    </div>
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${handoff.progress}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>

        {/* Agent States */}
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent States</h3>
            {agentStates.length === 0 ? (
              <div className="text-center py-8">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                <p className="mt-2 text-gray-600">No agents available</p>
              </div>
            ) : (
              <div className="space-y-3">
                {agentStates.map((agent) => (
                  <div
                    key={agent.agent_id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedAgent?.agent_id === agent.agent_id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => handleAgentSelect(agent)}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="font-medium">{agent.agent_name}</span>
                          <span className="text-sm text-gray-500">({agent.agent_type})</span>
                        </div>
                        <div className="flex items-center space-x-4 text-sm text-gray-600">
                          <span>
                            In: {agent.handoff_stats.received_today}, Out: {agent.handoff_stats.initiated_today}
                          </span>
                          <span>Success: {(agent.handoff_stats.success_rate * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                          style={{ backgroundColor: `${getAgentStatusColor(agent.current_status)}20`, color: getAgentStatusColor(agent.current_status) }}
                        >
                          {agent.current_status.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-gray-500">Load: {Math.round(agent.load_factor * 100)}%</span>
                        <div className="w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-orange-500 h-2 rounded-full"
                            style={{ width: `${agent.load_factor * 100}%` }}
                          ></div>
                        </div>
                      </div>
                      <span className="text-sm text-gray-500">
                        Avg: {formatDuration(agent.handoff_stats.avg_response_time * 1000)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Handoff History */}
      <Card>
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Recent Handoffs</h3>
            <span className="text-sm text-gray-500">Last {handoffHistory.length} handoffs</span>
          </div>
          {handoffHistory.length === 0 ? (
            <div className="text-center py-8">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="mt-2 text-gray-600">No handoff history available</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Source → Target
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Strategy
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {handoffHistory.map((history) => (
                    <tr key={history.handoff_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {history.timestamp.toLocaleTimeString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        <div className="flex items-center space-x-1">
                          <span>{history.source_agent}</span>
                          <svg className="w-3 h-3 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                          </svg>
                          <span>{history.target_agent}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {HANDOFF_STRATEGY_LABELS[history.strategy]}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDuration(history.duration)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                          style={{
                            backgroundColor: history.success
                              ? `${HANDOFF_STATUS_COLORS[HandoffStatus.COMPLETED]}20`
                              : `${HANDOFF_STATUS_COLORS[HandoffStatus.FAILED]}20`,
                            color: history.success
                              ? HANDOFF_STATUS_COLORS[HandoffStatus.COMPLETED]
                              : HANDOFF_STATUS_COLORS[HandoffStatus.FAILED]
                          }}
                        >
                          {history.success ? 'Success' : 'Failed'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default HandoffVisualization;