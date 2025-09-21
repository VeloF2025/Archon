import React from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';

export interface Agent {
  id: string;
  name: string;
  type: string;
  description?: string;
  status: 'active' | 'inactive' | 'error' | 'running';
  capabilities?: string[];
  config?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
  metrics?: {
    tasks_completed: number;
    success_rate: number;
    avg_response_time: number;
  };
}

interface AgentDetailsModalProps {
  agent: Agent | null;
  isOpen: boolean;
  onClose: () => void;
  onEdit?: (agent: Agent) => void;
  onDelete?: (agentId: string) => void;
  onToggleStatus?: (agentId: string) => void;
}

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

export const AgentDetailsModal: React.FC<AgentDetailsModalProps> = ({
  agent,
  isOpen,
  onClose,
  onEdit,
  onDelete,
  onToggleStatus,
}) => {
  if (!agent) return null;

  const handleEdit = () => {
    if (onEdit) {
      onEdit(agent);
    }
  };

  const handleDelete = () => {
    if (onDelete && confirm(`Are you sure you want to delete agent "${agent.name}"?`)) {
      onDelete(agent.id);
      onClose();
    }
  };

  const handleToggleStatus = () => {
    if (onToggleStatus) {
      onToggleStatus(agent.id);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="flex items-center gap-3">
                {agent.name}
                <Badge className={getStatusColor(agent.status)}>
                  {agent.status}
                </Badge>
              </DialogTitle>
              <DialogDescription className="mt-1">
                {agent.type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())} Agent
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Basic Information */}
          <div>
            <h3 className="text-lg font-semibold mb-3">Basic Information</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Agent ID
                </label>
                <p className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                  {agent.id}
                </p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Type
                </label>
                <p className="text-sm">{agent.type}</p>
              </div>
              {agent.created_at && (
                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Created
                  </label>
                  <p className="text-sm">{new Date(agent.created_at).toLocaleString()}</p>
                </div>
              )}
              {agent.updated_at && (
                <div>
                  <label className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Last Updated
                  </label>
                  <p className="text-sm">{new Date(agent.updated_at).toLocaleString()}</p>
                </div>
              )}
            </div>
          </div>

          {/* Description */}
          {agent.description && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Description</h3>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {agent.description}
              </p>
            </div>
          )}

          {/* Capabilities */}
          {agent.capabilities && agent.capabilities.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Capabilities</h3>
              <div className="flex flex-wrap gap-2">
                {agent.capabilities.map((capability, index) => (
                  <Badge key={index} variant="outline">
                    {capability}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Metrics */}
          {agent.metrics && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Performance Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">
                    {agent.metrics.tasks_completed}
                  </div>
                  <div className="text-sm text-gray-500">Tasks Completed</div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">
                    {Math.round(agent.metrics.success_rate * 100)}%
                  </div>
                  <div className="text-sm text-gray-500">Success Rate</div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">
                    {agent.metrics.avg_response_time.toFixed(1)}s
                  </div>
                  <div className="text-sm text-gray-500">Avg Response Time</div>
                </div>
              </div>
            </div>
          )}

          {/* Configuration */}
          {agent.config && Object.keys(agent.config).length > 0 && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Configuration</h3>
              <pre className="text-xs bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-auto">
                {JSON.stringify(agent.config, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between pt-4 border-t">
          <div className="flex space-x-2">
            {onToggleStatus && (
              <Button
                variant="outline"
                onClick={handleToggleStatus}
              >
                {agent.status === 'active' ? 'Deactivate' : 'Activate'}
              </Button>
            )}
            {onEdit && (
              <Button
                variant="outline"
                onClick={handleEdit}
              >
                Edit
              </Button>
            )}
          </div>
          <div className="flex space-x-2">
            {onDelete && (
              <Button
                variant="destructive"
                onClick={handleDelete}
              >
                Delete
              </Button>
            )}
            <Button onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default AgentDetailsModal;