import React, { useState } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { useAgentSystem } from '../../hooks/useAgentSystem';

interface AgentQuickActionsProps {
  className?: string;
}

export const AgentQuickActions: React.FC<AgentQuickActionsProps> = ({ className = '' }) => {
  const { triggerAgent, spawnAgent, executeAgentAction, isLoading } = useAgentSystem();
  const [activeAction, setActiveAction] = useState<string | null>(null);

  const quickActions = [
    {
      id: 'code-review',
      label: 'Code Review',
      description: 'Run security audit and code review',
      agents: ['security_auditor', 'code_reviewer'],
      icon: '🔍'
    },
    {
      id: 'test-generation',
      label: 'Generate Tests',
      description: 'Create comprehensive test suite',
      agents: ['test_generator'],
      icon: '🧪'
    },
    {
      id: 'documentation', 
      label: 'Update Docs',
      description: 'Generate and update documentation',
      agents: ['documentation_writer', 'technical_writer'],
      icon: '📝'
    },
    {
      id: 'performance-check',
      label: 'Performance Check',
      description: 'Analyze and optimize performance',
      agents: ['performance_optimizer'],
      icon: '⚡'
    }
  ];

  const handleQuickAction = async (actionId: string, agents: string[]) => {
    setActiveAction(actionId);
    
    try {
      // Execute agents in parallel for the quick action
      const promises = agents.map(agent => 
        triggerAgent(agent, {
          prompt: `Execute ${actionId} workflow`,
          context: { quickAction: true, actionId }
        })
      );
      
      await Promise.all(promises);
      
      // Show success feedback
      setTimeout(() => setActiveAction(null), 2000);
      
    } catch (error) {
      console.error(`Quick action ${actionId} failed:`, error);
      setActiveAction(null);
    }
  };

  return (
    <Card className={`p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Quick Actions
        </h3>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          One-click workflows
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-3">
        {quickActions.map((action) => (
          <Button
            key={action.id}
            variant="outline"
            size="sm"
            onClick={() => handleQuickAction(action.id, action.agents)}
            disabled={isLoading || activeAction === action.id}
            className="flex flex-col items-center p-3 h-auto space-y-1 hover:bg-blue-50 dark:hover:bg-blue-900/20"
          >
            <span className="text-xl">{action.icon}</span>
            <span className="text-xs font-medium">{action.label}</span>
            <span className="text-xs text-gray-500 text-center">
              {action.description}
            </span>
            {activeAction === action.id && (
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mt-1" />
            )}
          </Button>
        ))}
      </div>
      
      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => executeAgentAction('refresh_all')}
            disabled={isLoading}
            className="text-xs"
          >
            🔄 Refresh All Agents
          </Button>
          <Button
            variant="ghost" 
            size="sm"
            onClick={() => executeAgentAction('emergency_stop')}
            className="text-xs text-red-600 hover:text-red-700"
          >
            🛑 Emergency Stop
          </Button>
        </div>
      </div>
    </Card>
  );
};