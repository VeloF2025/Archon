/**
 * Handoff Request Form Component
 *
 * Form for creating manual handoff requests between agents
 */

import React, { useState, useEffect } from 'react';
import {
  HandoffRequestFormProps,
  HandoffStrategy,
  HandoffTrigger,
  HANDOFF_STRATEGY_LABELS,
  HANDOFF_STRATEGY_DESCRIPTIONS
} from '../../types/handoffTypes';
import { handoffService } from '../../services/handoffService';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Textarea } from '../ui/textarea';

const HandoffRequestForm: React.FC<HandoffRequestFormProps> = ({
  projectId,
  sourceAgentId,
  onHandoffRequest,
  availableAgents,
  predefinedTasks = []
}) => {
  const [formData, setFormData] = useState({
    targetAgentId: '',
    message: '',
    taskDescription: '',
    strategy: HandoffStrategy.SEQUENTIAL,
    trigger: HandoffTrigger.MANUAL_REQUEST,
    priority: 3,
    context: {}
  });

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [_showRecommendations, setShowRecommendations] = useState(false);
  const [selectedPredefinedTask, setSelectedPredefinedTask] = useState<string>('');

  useEffect(() => {
    if (sourceAgentId) {
      setFormData(prev => ({ ...prev, targetAgentId: '' }));
    }
  }, [sourceAgentId]);

  // Get available target agents (excluding source agent)
  const getTargetAgents = () => {
    return availableAgents.filter(agent =>
      agent.agent_id !== sourceAgentId &&
      agent.current_status !== 'offline'
    );
  };

  // Handle input changes
  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError(null);
  };

  // Handle predefined task selection
  const handlePredefinedTaskSelect = (taskId: string) => {
    const task = predefinedTasks.find(t => t.id === taskId);
    if (task) {
      setFormData(prev => ({
        ...prev,
        taskDescription: task.description,
        strategy: task.recommended_strategy
      }));
      setSelectedPredefinedTask(taskId);
    }
  };

  // Get recommendations for the task
  const getRecommendations = async () => {
    if (!formData.taskDescription || !sourceAgentId) return;

    try {
      setIsLoading(true);
      const recommendation = await handoffService.getHandoffRecommendations(
        formData.taskDescription,
        sourceAgentId
      );

      setRecommendations(recommendation.recommended_agents);
      setShowRecommendations(true);

      // Auto-select best recommendation
      if (recommendation.recommended_agents.length > 0) {
        const bestAgent = recommendation.recommended_agents[0];
        handleInputChange('targetAgentId', bestAgent.agent_id);

        if (recommendation.strategy_recommendation) {
          handleInputChange('strategy', recommendation.strategy_recommendation.strategy);
        }
      }
    } catch (err) {
      console.error('Error getting recommendations:', err);
      setError('Failed to get recommendations');
    } finally {
      setIsLoading(false);
    }
  };

  // Submit handoff request
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validate form
    if (!formData.targetAgentId) {
      setError('Please select a target agent');
      return;
    }

    if (!formData.taskDescription.trim()) {
      setError('Please provide a task description');
      return;
    }

    if (!formData.message.trim()) {
      setError('Please provide a message for the handoff');
      return;
    }

    try {
      setIsLoading(true);

      const handoffRequest = {
        source_agent_id: sourceAgentId || 'manual',
        target_agent_id: formData.targetAgentId,
        message: formData.message,
        task_description: formData.taskDescription,
        strategy: formData.strategy,
        trigger: formData.trigger,
        confidence_score: 0.8, // Default confidence for manual requests
        priority: formData.priority,
        context: {
          ...formData.context,
          project_id: projectId,
          predefined_task_id: selectedPredefinedTask || undefined
        }
      };

      await handoffService.requestHandoff(handoffRequest);

      // Call parent callback
      onHandoffRequest(handoffRequest);

      // Reset form
      setFormData({
        targetAgentId: '',
        message: '',
        taskDescription: '',
        strategy: HandoffStrategy.SEQUENTIAL,
        trigger: HandoffTrigger.MANUAL_REQUEST,
        priority: 3,
        context: {}
      });
      setSelectedPredefinedTask('');
      setShowRecommendations(false);
      setRecommendations([]);

    } catch (err) {
      console.error('Error submitting handoff request:', err);
      setError(err instanceof Error ? err.message : 'Failed to submit handoff request');
    } finally {
      setIsLoading(false);
    }
  };

  const targetAgents = getTargetAgents();

  return (
    <Card>
      <div className="p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Create Handoff Request</h3>

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-800">{error}</p>
              </div>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Predefined Tasks */}
          {predefinedTasks.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Predefined Tasks
              </label>
              <select
                value={selectedPredefinedTask}
                onChange={(e) => handlePredefinedTaskSelect(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select a predefined task...</option>
                {predefinedTasks.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.title} (Complexity: {task.estimated_complexity}/5)
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Target Agent Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Agent *
            </label>
            <div className="space-y-2">
              <select
                value={formData.targetAgentId}
                onChange={(e) => handleInputChange('targetAgentId', e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              >
                <option value="">Select a target agent...</option>
                {targetAgents.map((agent) => (
                  <option key={agent.agent_id} value={agent.agent_id}>
                    {agent.agent_name} ({agent.agent_type}) - {agent.current_status}
                  </option>
                ))}
              </select>

              {recommendations.length > 0 && (
                <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm font-medium text-blue-800 mb-2">
                    AI Recommendations:
                  </p>
                  <div className="space-y-1">
                    {recommendations.map((rec, index) => (
                      <div key={index} className="flex justify-between items-center text-sm">
                        <span className="text-blue-700">
                          {rec.agent_name} (Match: {(rec.match_score * 100).toFixed(0)}%)
                        </span>
                        <span className="text-blue-600 text-xs">
                          {rec.reasoning}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Task Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Task Description *
            </label>
            <Textarea
              value={formData.taskDescription}
              onChange={(e) => handleInputChange('taskDescription', e.target.value)}
              placeholder="Describe the task that needs to be handed off..."
              rows={3}
              required
            />
          </div>

          {/* Message */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Message for Target Agent *
            </label>
            <Textarea
              value={formData.message}
              onChange={(e) => handleInputChange('message', e.target.value)}
              placeholder="Provide context and instructions for the target agent..."
              rows={3}
              required
            />
          </div>

          {/* Handoff Strategy */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Handoff Strategy *
            </label>
            <select
              value={formData.strategy}
              onChange={(e) => handleInputChange('strategy', e.target.value as HandoffStrategy)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {Object.entries(HANDOFF_STRATEGY_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label} - {HANDOFF_STRATEGY_DESCRIPTIONS[value as HandoffStrategy]}
                </option>
              ))}
            </select>
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Priority
            </label>
            <div className="flex space-x-4">
              {[1, 2, 3, 4, 5].map((priority) => (
                <label key={priority} className="flex items-center">
                  <input
                    type="radio"
                    name="priority"
                    value={priority}
                    checked={formData.priority === priority}
                    onChange={(e) => handleInputChange('priority', parseInt(e.target.value))}
                    className="mr-2"
                  />
                  <span className="text-sm">
                    {priority === 1 ? 'Low' :
                     priority === 2 ? 'Normal' :
                     priority === 3 ? 'Medium' :
                     priority === 4 ? 'High' : 'Critical'}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-between items-center pt-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <Button
                type="button"
                onClick={getRecommendations}
                disabled={isLoading || !formData.taskDescription.trim() || !sourceAgentId}
                variant="outline"
                size="sm"
              >
                {isLoading ? 'Getting Recommendations...' : 'Get AI Recommendations'}
              </Button>
            </div>

            <div className="flex space-x-2">
              <Button
                type="button"
                onClick={() => {
                  setFormData({
                    targetAgentId: '',
                    message: '',
                    taskDescription: '',
                    strategy: HandoffStrategy.SEQUENTIAL,
                    trigger: HandoffTrigger.MANUAL_REQUEST,
                    priority: 3,
                    context: {}
                  });
                  setSelectedPredefinedTask('');
                  setShowRecommendations(false);
                  setRecommendations([]);
                  setError(null);
                }}
                variant="outline"
                disabled={isLoading}
              >
                Clear
              </Button>

              <Button
                type="submit"
                disabled={isLoading || !formData.targetAgentId || !formData.taskDescription.trim() || !formData.message.trim()}
              >
                {isLoading ? 'Creating Handoff...' : 'Create Handoff Request'}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </Card>
  );
};

export default HandoffRequestForm;