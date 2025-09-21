/**
 * Handoff Analytics Dashboard Component
 *
 * Comprehensive analytics and insights for agent handoff performance
 */

import React, { useState, useEffect } from 'react';
import {
  HandoffAnalyticsViewProps,
  HandoffAnalytics,
  HandoffStrategy,
  HANDOFF_STRATEGY_LABELS
} from '../../types/handoffTypes';
import { handoffService } from '../../services/handoffService';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';

const HandoffAnalytics: React.FC<HandoffAnalyticsViewProps> = ({
  projectId,
  timeRange = 24,
  showAdvancedMetrics = true,
  onExportData
}) => {
  const [analytics, setAnalytics] = useState<HandoffAnalytics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<'overview' | 'strategies' | 'agents' | 'insights'>('overview');

  // Load analytics data
  const loadAnalytics = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await handoffService.getHandoffAnalytics(projectId, timeRange);
      setAnalytics(data);
    } catch (err) {
      console.error('Error loading analytics:', err);
      setError(err instanceof Error ? err.message : 'Failed to load analytics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAnalytics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, timeRange]);

  // Export data
  const handleExport = (format: 'csv' | 'json') => {
    if (!analytics) return;

    if (format === 'json') {
      const dataStr = JSON.stringify(analytics, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `handoff-analytics-${new Date().toISOString()}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } else {
      // CSV export logic
      let csvContent = 'Strategy,Usage Count,Success Rate,Avg Execution Time,Avg Confidence\n';
      Object.entries(analytics.strategy_performance).forEach(([strategy, perf]) => {
        csvContent += `${strategy},${perf.usage_count},${perf.success_rate},${perf.avg_execution_time},${perf.avg_confidence_score}\n`;
      });

      const dataBlob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `handoff-analytics-${new Date().toISOString()}.csv`;
      link.click();
      URL.revokeObjectURL(url);
    }

    onExportData?.(format);
  };

  // Get color for performance metrics
  const getPerformanceColor = (value: number, type: 'rate' | 'time' | 'score' = 'score'): string => {
    if (type === 'rate') {
      if (value >= 0.8) return 'text-green-600';
      if (value >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    } else if (type === 'time') {
      if (value <= 1000) return 'text-green-600';
      if (value <= 5000) return 'text-yellow-600';
      return 'text-red-600';
    } else {
      if (value >= 0.8) return 'text-green-600';
      if (value >= 0.6) return 'text-yellow-600';
      return 'text-red-600';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analytics...</p>
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
            <h3 className="text-sm font-medium text-red-800">Error Loading Analytics</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <Button onClick={loadAnalytics} variant="outline" size="sm">
                Retry
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!analytics) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Handoff Analytics</h2>
          <p className="text-gray-600">
            Last {timeRange} hours • Generated {analytics.generated_at.toLocaleString()}
          </p>
        </div>
        <div className="flex space-x-2">
          <Button onClick={loadAnalytics} variant="outline" size="sm">
            Refresh
          </Button>
          <Button onClick={() => handleExport('json')} variant="outline" size="sm">
            Export JSON
          </Button>
          <Button onClick={() => handleExport('csv')} variant="outline" size="sm">
            Export CSV
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <div className="p-4">
            <div className="text-2xl font-bold text-blue-600">{analytics.total_handoffs}</div>
            <div className="text-sm text-gray-600">Total Handoffs</div>
          </div>
        </Card>
        <Card>
          <div className="p-4">
            <div className={`text-2xl font-bold ${getPerformanceColor(analytics.success_rate, 'rate')}`}>
              {(analytics.success_rate * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Success Rate</div>
          </div>
        </Card>
        <Card>
          <div className="p-4">
            <div className="text-2xl font-bold text-purple-600">
              {Object.keys(analytics.strategy_performance).length}
            </div>
            <div className="text-sm text-gray-600">Strategies Used</div>
          </div>
        </Card>
        <Card>
          <div className="p-4">
            <div className="text-2xl font-bold text-orange-600">
              {Object.keys(analytics.agent_performance).length}
            </div>
            <div className="text-sm text-gray-600">Active Agents</div>
          </div>
        </Card>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { key: 'overview', label: 'Overview' },
            { key: 'strategies', label: 'Strategies' },
            { key: 'agents', label: 'Agents' },
            { key: 'insights', label: 'Insights' }
          ].map((tab) => (
            <button
              key={tab.key}
              onClick={() => setSelectedTab(tab.key as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                selectedTab === tab.key
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {selectedTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Strategy Performance */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategy Performance</h3>
              <div className="space-y-3">
                {Object.entries(analytics.strategy_performance).map(([strategy, perf]) => (
                  <div key={strategy} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium">{HANDOFF_STRATEGY_LABELS[strategy as HandoffStrategy]}</div>
                      <div className="text-sm text-gray-500">{perf.usage_count} uses</div>
                    </div>
                    <div className="text-right">
                      <div className={`font-semibold ${getPerformanceColor(perf.success_rate, 'rate')}`}>
                        {(perf.success_rate * 100).toFixed(0)}%
                      </div>
                      <div className="text-sm text-gray-500">
                        {perf.avg_execution_time.toFixed(0)}ms avg
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Top Agents */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Performing Agents</h3>
              <div className="space-y-3">
                {Object.entries(analytics.agent_performance)
                  .sort(([,a], [,b]) => (b.success_rate_initiated + b.success_rate_received) / 2 - (a.success_rate_initiated + a.success_rate_received) / 2)
                  .slice(0, 5)
                  .map(([agentId, perf]) => (
                    <div key={agentId} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">{agentId}</div>
                        <div className="text-sm text-gray-500">
                          Initiated: {perf.handoffs_initiated}, Received: {perf.handoffs_received}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-semibold ${getPerformanceColor((perf.success_rate_initiated + perf.success_rate_received) / 2, 'rate')}`}>
                          {(((perf.success_rate_initiated + perf.success_rate_received) / 2) * 100).toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-500">
                          {perf.avg_response_time.toFixed(0)}ms
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </Card>
        </div>
      )}

      {selectedTab === 'strategies' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Strategy Analysis</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Strategy
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Usage Count
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Time
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Best For
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(analytics.strategy_performance).map(([strategy, perf]) => (
                    <tr key={strategy}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {HANDOFF_STRATEGY_LABELS[strategy as HandoffStrategy]}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {perf.usage_count}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`font-semibold ${getPerformanceColor(perf.success_rate, 'rate')}`}>
                          {(perf.success_rate * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {perf.avg_execution_time.toFixed(0)}ms
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`font-semibold ${getPerformanceColor(perf.avg_confidence_score)}`}>
                          {(perf.avg_confidence_score * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="max-w-xs">
                          {perf.best_for_scenarios?.slice(0, 2).map((scenario, i) => (
                            <div key={i} className="text-xs">{scenario}</div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {selectedTab === 'agents' && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Agent Performance Details</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Agent
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Initiated
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Received
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate (In)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate (Out)
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Response
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Preferred Strategies
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(analytics.agent_performance).map(([agentId, perf]) => (
                    <tr key={agentId}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {agentId}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {perf.handoffs_initiated}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {perf.handoffs_received}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`font-semibold ${getPerformanceColor(perf.success_rate_initiated, 'rate')}`}>
                          {(perf.success_rate_initiated * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`font-semibold ${getPerformanceColor(perf.success_rate_received, 'rate')}`}>
                          {(perf.success_rate_received * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {perf.avg_response_time.toFixed(0)}ms
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex flex-wrap gap-1">
                          {perf.preferred_strategies?.slice(0, 2).map((strategy) => (
                            <span key={strategy} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                              {HANDOFF_STRATEGY_LABELS[strategy]}
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {selectedTab === 'insights' && showAdvancedMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Improved Patterns */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Improved Patterns</h3>
              {analytics.learning_insights.improved_patterns.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No improved patterns identified yet</p>
              ) : (
                <div className="space-y-3">
                  {analytics.learning_insights.improved_patterns.map((pattern) => (
                    <div key={pattern.pattern_id} className="p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-medium text-green-800">{pattern.description}</div>
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                          +{(pattern.success_rate_improvement * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="text-sm text-green-700">
                        Confidence: {(pattern.confidence_score * 100).toFixed(0)}% • {pattern.occurrence_count} occurrences
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>

          {/* Optimization Opportunities */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Optimization Opportunities</h3>
              {analytics.learning_insights.optimization_opportunities.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No optimization opportunities identified</p>
              ) : (
                <div className="space-y-3">
                  {analytics.learning_insights.optimization_opportunities.map((opp) => (
                    <div key={opp.opportunity_id} className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                      <div className="font-medium text-blue-800 mb-1">{opp.description}</div>
                      <div className="flex justify-between items-center text-sm text-blue-700">
                        <span>Impact: {(opp.potential_impact * 100).toFixed(0)}%</span>
                        <span>Complexity: {opp.implementation_complexity}/5</span>
                      </div>
                      <div className="text-xs text-blue-600 mt-1">{opp.estimated_benefit}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>

          {/* Capability Gaps */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Capability Gaps</h3>
              {analytics.learning_insights.capability_gaps.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No capability gaps identified</p>
              ) : (
                <div className="space-y-3">
                  {analytics.learning_insights.capability_gaps.map((gap) => (
                    <div key={`${gap.capability_type}-${gap.priority}`} className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <div className="font-medium text-orange-800">{gap.capability_type}</div>
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-800">
                          Priority {gap.priority}
                        </span>
                      </div>
                      <div className="text-sm text-orange-700 mb-2">
                        Severity: {(gap.gap_severity * 100).toFixed(0)}% • {gap.affected_agents.length} agents affected
                      </div>
                      <div className="text-xs text-orange-600">
                        <div className="font-medium mb-1">Recommended training:</div>
                        <ul className="list-disc list-inside">
                          {gap.recommended_training.slice(0, 2).map((training, i) => (
                            <li key={i}>{training}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>

          {/* Confidence Improvements */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Confidence Improvements</h3>
              {analytics.learning_insights.confidence_improvements.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No confidence improvements identified</p>
              ) : (
                <div className="space-y-3">
                  {analytics.learning_insights.confidence_improvements.map((imp) => (
                    <div key={imp.factor} className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <div className="font-medium text-purple-800">{imp.factor}</div>
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                          +{imp.improvement_factor}%
                        </span>
                      </div>
                      <div className="text-sm text-purple-700">
                        {imp.current_confidence.toFixed(0)}% → {imp.target_confidence.toFixed(0)}% ({imp.timeframe})
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default HandoffAnalytics;