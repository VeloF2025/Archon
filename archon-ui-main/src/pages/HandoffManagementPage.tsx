/**
 * Handoff Management Page
 *
 * Main page for managing agent handoffs with real-time visualization
 */

import React, { useState, useEffect } from 'react';
import {
  ActiveHandoff,
  AgentHandoffState,
  HandoffRequest,
  HandoffStrategy,
  PredefinedTask
} from '../types/handoffTypes';
import { handoffService } from '../services/handoffService';
import { agentManagementService } from '../services/agentManagementService';
import HandoffVisualization from '../components/handoff/HandoffVisualization';
import HandoffRequestForm from '../components/handoff/HandoffRequestForm';
import HandoffAnalytics from '../components/handoff/HandoffAnalytics';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

const HandoffManagementPage: React.FC = () => {
  const [projectId, setProjectId] = useState<string>('');
  const [availableAgents, setAvailableAgents] = useState<AgentHandoffState[]>([]);
  const [selectedTab, setSelectedTab] = useState<'visualization' | 'request' | 'analytics'>('visualization');
  const [selectedHandoff, setSelectedHandoff] = useState<ActiveHandoff | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentHandoffState | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Predefined tasks for quick handoff requests
  const predefinedTasks: PredefinedTask[] = [
    {
      id: 'code-review',
      title: 'Code Review',
      description: 'Review and provide feedback on code changes, identify bugs, suggest improvements',
      required_capabilities: ['code_analysis', 'quality_assessment', 'pattern_recognition'],
      recommended_strategy: HandoffStrategy.COLLABORATIVE,
      estimated_complexity: 3
    },
    {
      id: 'security-audit',
      title: 'Security Audit',
      description: 'Conduct comprehensive security analysis, identify vulnerabilities, suggest fixes',
      required_capabilities: ['security_analysis', 'vulnerability_assessment', 'compliance_checking'],
      recommended_strategy: HandoffStrategy.SEQUENTIAL,
      estimated_complexity: 4
    },
    {
      id: 'performance-optimization',
      title: 'Performance Optimization',
      description: 'Analyze performance bottlenecks, optimize code, improve response times',
      required_capabilities: ['performance_analysis', 'bottleneck_identification', 'optimization_techniques'],
      recommended_strategy: HandoffStrategy.PARALLEL,
      estimated_complexity: 4
    },
    {
      id: 'testing-validation',
      title: 'Testing & Validation',
      description: 'Create comprehensive tests, validate functionality, ensure coverage requirements',
      required_capabilities: ['test_planning', 'test_implementation', 'coverage_analysis'],
      recommended_strategy: HandoffStrategy.SEQUENTIAL,
      estimated_complexity: 3
    },
    {
      id: 'architecture-design',
      title: 'Architecture Design',
      description: 'Design system architecture, create technical specifications, plan implementation',
      required_capabilities: ['system_design', 'technical_planning', 'scalability_analysis'],
      recommended_strategy: HandoffStrategy.COLLABORATIVE,
      estimated_complexity: 5
    }
  ];

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Get project ID from local storage or URL
      const urlParams = new URLSearchParams(window.location.search);
      const projectParam = urlParams.get('project') || localStorage.getItem('currentProjectId') || '';
      setProjectId(projectParam);

      // Load agents data
      const [agentsData] = await Promise.all([
        handoffService.getHandoffVisualization(projectParam),
        // Load existing agents from agent management service
        agentManagementService.getAgents(projectParam)
      ]);

      // Transform agent data to handoff state format
      const agentStates: AgentHandoffState[] = agentsData.agent_states.map((state: any) => ({
        agent_id: state.agent_id,
        agent_name: state.agent_name,
        agent_type: state.agent_type,
        current_status: state.current_status,
        current_handoff_id: state.current_handoff_id,
        handoff_stats: state.handoff_stats || {
          initiated_today: 0,
          received_today: 0,
          success_rate: 0,
          avg_response_time: 0
        },
        capabilities: state.capabilities || [],
        load_factor: state.load_factor || 0
      }));

      setAvailableAgents(agentStates);

    } catch (err) {
      console.error('Error loading initial data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle handoff request
  const handleHandoffRequest = (_request: HandoffRequest) => {
    // Show success message or navigate to visualization
    setSelectedTab('visualization');
  };

  // Handle handoff selection
  const handleHandoffSelect = (handoff: ActiveHandoff) => {
    setSelectedHandoff(handoff);
  };

  // Handle agent selection
  const handleAgentSelect = (agent: AgentHandoffState) => {
    setSelectedAgent(agent);
  };

  // Run learning cycle
  const handleRunLearningCycle = async () => {
    try {
      setIsLoading(true);
      const result = await handoffService.runLearningCycle();

      // Show success message
      alert(`Learning cycle completed successfully!\nUpdated ${Object.keys(result.updated_insights).length} insights.`);

      // Refresh data
      await loadInitialData();
    } catch (err) {
      console.error('Error running learning cycle:', err);
      alert('Failed to run learning cycle. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading Handoff Management...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error Loading Handoff Management</h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error}</p>
              </div>
              <div className="mt-4">
                <Button onClick={loadInitialData} variant="outline" size="sm">
                  Retry
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Agent Handoff Management</h1>
              <p className="mt-1 text-sm text-gray-600">
                Intelligent agent handoffs with real-time visualization and analytics
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <Button
                onClick={handleRunLearningCycle}
                variant="outline"
                disabled={isLoading}
              >
                {isLoading ? 'Running...' : 'Run Learning Cycle'}
              </Button>
              {projectId && (
                <div className="text-sm text-gray-500">
                  Project: {projectId}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Navigation Tabs */}
        <div className="mb-8">
          <nav className="flex space-x-8" aria-label="Tabs">
            {[
              { key: 'visualization', label: 'Real-time Visualization', icon: '📊' },
              { key: 'request', label: 'Create Handoff Request', icon: '🔄' },
              { key: 'analytics', label: 'Analytics & Insights', icon: '📈' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setSelectedTab(tab.key as any)}
                className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                  selectedTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {selectedTab === 'visualization' && (
            <HandoffVisualization
              projectId={projectId}
              onHandoffSelect={handleHandoffSelect}
              onAgentSelect={handleAgentSelect}
            />
          )}

          {selectedTab === 'request' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <HandoffRequestForm
                  projectId={projectId}
                  onHandoffRequest={handleHandoffRequest}
                  availableAgents={availableAgents}
                  predefinedTasks={predefinedTasks}
                />
              </div>
              <div>
                <Card>
                  <div className="p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
                    <div className="space-y-3">
                      <Button
                        onClick={() => {
                          const _randomTask = predefinedTasks[Math.floor(Math.random() * predefinedTasks.length)];
                          setSelectedTab('request');
                        }}
                        variant="outline"
                        className="w-full"
                      >
                        🎲 Random Task
                      </Button>
                      <Button
                        onClick={() => {
                          const availableAgent = availableAgents.find(agent => agent.current_status === 'available');
                          if (availableAgent) {
                            setSelectedAgent(availableAgent);
                          }
                        }}
                        variant="outline"
                        className="w-full"
                      >
                        👤 Find Available Agent
                      </Button>
                    </div>
                  </div>
                </Card>

                {/* Selected Agent Details */}
                {selectedAgent && (
                  <Card className="mt-4">
                    <div className="p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Selected Agent</h3>
                      <div className="space-y-2">
                        <div>
                          <span className="font-medium">{selectedAgent.agent_name}</span>
                          <span className="text-sm text-gray-500 ml-2">({selectedAgent.agent_type})</span>
                        </div>
                        <div className="text-sm text-gray-600">
                          Status: <span className="font-medium">{selectedAgent.current_status.replace('_', ' ')}</span>
                        </div>
                        <div className="text-sm text-gray-600">
                          Load Factor: {Math.round(selectedAgent.load_factor * 100)}%
                        </div>
                        <div className="text-sm text-gray-600">
                          Success Rate: {(selectedAgent.handoff_stats.success_rate * 100).toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-600">
                          Avg Response: {(selectedAgent.handoff_stats.avg_response_time / 1000).toFixed(1)}s
                        </div>
                      </div>
                    </div>
                  </Card>
                )}
              </div>
            </div>
          )}

          {selectedTab === 'analytics' && (
            <HandoffAnalytics
              projectId={projectId}
              timeRange={24}
              showAdvancedMetrics={true}
              onExportData={(format) => {
                console.log(`Exporting data in ${format} format`);
              }}
            />
          )}
        </div>

        {/* Selected Handoff Details */}
        {selectedHandoff && (
          <Card className="mt-6">
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Handoff Details</h3>
                <Button
                  onClick={() => setSelectedHandoff(null)}
                  variant="outline"
                  size="sm"
                >
                  Close
                </Button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Basic Information</h4>
                  <div className="space-y-1 text-sm">
                    <div><span className="font-medium">ID:</span> {selectedHandoff.handoff_id}</div>
                    <div><span className="font-medium">Source:</span> {selectedHandoff.source_agent}</div>
                    <div><span className="font-medium">Target:</span> {selectedHandoff.target_agent}</div>
                    <div><span className="font-medium">Strategy:</span> {selectedHandoff.strategy}</div>
                    <div><span className="font-medium">Status:</span> {selectedHandoff.status}</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Progress & Performance</h4>
                  <div className="space-y-1 text-sm">
                    <div><span className="font-medium">Progress:</span> {selectedHandoff.progress}%</div>
                    <div><span className="font-medium">Confidence:</span> {(selectedHandoff.confidence_score * 100).toFixed(0)}%</div>
                    <div><span className="font-medium">Started:</span> {selectedHandoff.start_time.toLocaleString()}</div>
                    {selectedHandoff.estimated_completion && (
                      <div><span className="font-medium">Estimated:</span> {selectedHandoff.estimated_completion.toLocaleString()}</div>
                    )}
                  </div>
                </div>
              </div>
              <div className="mt-4">
                <h4 className="font-medium text-gray-900 mb-2">Task Description</h4>
                <p className="text-sm text-gray-600">{selectedHandoff.task_description}</p>
              </div>
            </div>
          </Card>
        )}
      </main>
    </div>
  );
};

export default HandoffManagementPage;