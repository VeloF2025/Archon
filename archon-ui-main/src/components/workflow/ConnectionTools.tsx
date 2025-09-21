import React, { useState, useCallback } from 'react';

import {
  ConnectionType,
  CommunicationType,
} from '../../types/workflowTypes';
import { AGENT_TYPE_ICONS } from '../../types/agentTypes';

interface ConnectionToolsProps {
  selectedConnectionType: ConnectionType;
  onConnectionTypeChange: (type: ConnectionType) => void;
  onConnectionCreate: (sourceId: string, targetId: string, type: ConnectionType) => void;
  isCreatingConnection: boolean;
  sourceNodeId?: string;
  className?: string;
}

interface ConnectionTypeOption {
  type: ConnectionType;
  name: string;
  description: string;
  icon: string;
  color: string;
  messageTypes: string[];
  useCases: string[];
}

const CONNECTION_TYPES: ConnectionTypeOption[] = [
  {
    type: ConnectionType.DIRECT,
    name: 'Direct Connection',
    description: 'One-to-one communication between agents',
    icon: '🔗',
    color: 'text-blue-400',
    messageTypes: ['task_assignment', 'status_update', 'data_request', 'result_delivery'],
    useCases: [
      'Sequential task processing',
      'Supervisor-worker relationships',
      'Peer-to-peer collaboration'
    ]
  },
  {
    type: ConnectionType.BROADCAST,
    name: 'Broadcast',
    description: 'One-to-many communication to multiple agents',
    icon: '📡',
    color: 'text-purple-400',
    messageTypes: ['announcement', 'status_broadcast', 'emergency_alert', 'data_distribution'],
    useCases: [
      'System-wide notifications',
      'Coordinator to multiple workers',
      'Emergency alerts'
    ]
  },
  {
    type: ConnectionType.CHAIN,
    name: 'Chain Processing',
    description: 'Sequential processing through multiple agents',
    icon: '⛓️',
    color: 'text-green-400',
    messageTypes: ['pipeline_data', 'sequential_task', 'workflow_step', 'handoff'],
    useCases: [
      'Multi-step workflows',
      'Data processing pipelines',
      'Quality assurance chains'
    ]
  },
  {
    type: ConnectionType.COLLABORATIVE,
    name: 'Collaborative',
    description: 'Many-to-many communication for shared tasks',
    icon: '🤝',
    color: 'text-orange-400',
    messageTypes: ['collaboration', 'shared_context', 'discussion', 'consensus'],
    useCases: [
      'Team problem-solving',
      'Joint decision making',
      'Shared project work'
    ]
  }
];

interface MessageTypeOption {
  type: string;
  name: string;
  description: string;
  icon: string;
}

const MESSAGE_TYPES: MessageTypeOption[] = [
  {
    type: 'task_assignment',
    name: 'Task Assignment',
    description: 'Assign specific tasks to agents',
    icon: '📋'
  },
  {
    type: 'status_update',
    name: 'Status Update',
    description: 'Share progress and status information',
    icon: '📊'
  },
  {
    type: 'data_request',
    name: 'Data Request',
    description: 'Request information or data from other agents',
    icon: '🔍'
  },
  {
    type: 'result_delivery',
    name: 'Result Delivery',
    description: 'Deliver completed work or results',
    icon: '📦'
  },
  {
    type: 'collaboration',
    name: 'Collaboration',
    description: 'Collaborative discussion and problem-solving',
    icon: '💬'
  },
  {
    type: 'notification',
    name: 'Notification',
    description: 'System notifications and alerts',
    icon: '🔔'
  },
  {
    type: 'validation_request',
    name: 'Validation Request',
    description: 'Request validation or review of work',
    icon: '✅'
  },
  {
    type: 'error_report',
    name: 'Error Report',
    description: 'Report errors or issues encountered',
    icon: '⚠️'
  }
];

export const ConnectionTools: React.FC<ConnectionToolsProps> = ({
  selectedConnectionType,
  onConnectionTypeChange,
  onConnectionCreate,
  isCreatingConnection,
  sourceNodeId,
  className = '',
}) => {
  const [selectedMessageType, setSelectedMessageType] = useState<string>('task_assignment');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleConnectionTypeSelect = useCallback((type: ConnectionType) => {
    onConnectionTypeChange(type);
  }, [onConnectionTypeChange]);

  const handleCreateConnection = useCallback(() => {
    if (sourceNodeId) {
      // This would typically trigger a mode where the user selects a target
      // For now, we'll show the interface and expect the target to be selected elsewhere
      console.log('Creating connection from', sourceNodeId, 'with type', selectedConnectionType);
    }
  }, [sourceNodeId, selectedConnectionType]);

  const selectedConnectionOption = CONNECTION_TYPES.find(option => option.type === selectedConnectionType);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="space-y-2">
        <h3 className="text-white text-lg font-semibold">Connection Tools</h3>
        <p className="text-gray-400 text-sm">
          {isCreatingConnection
            ? sourceNodeId
              ? 'Select target agent to complete connection'
              : 'Select source agent to start connection'
            : 'Choose connection type and configure communication'
          }
        </p>
      </div>

      {/* Connection Types */}
      <div className="space-y-3">
        <h4 className="text-gray-300 text-sm font-medium">Connection Type</h4>
        <div className="grid grid-cols-2 gap-2">
          {CONNECTION_TYPES.map((option) => (
            <button
              key={option.type}
              onClick={() => handleConnectionTypeSelect(option.type)}
              className={`
                p-3 rounded-lg border transition-all duration-200 text-left
                ${selectedConnectionType === option.type
                  ? `${option.color.replace('text-', 'bg-')}/20 border-${option.color.replace('text-', '')} ring-2 ring-${option.color.replace('text-', '')}/50`
                  : 'bg-gray-800 border-gray-700 hover:bg-gray-700 hover:border-gray-600'
                }
              `}
            >
              <div className="flex items-start space-x-2">
                <span className="text-lg">{option.icon}</span>
                <div className="flex-1">
                  <div className={`text-sm font-medium ${selectedConnectionType === option.type ? option.color : 'text-white'}`}>
                    {option.name}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {option.description}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Selected Connection Details */}
      {selectedConnectionOption && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-gray-300 text-sm font-medium">
              {selectedConnectionOption.name} Details
            </h4>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-gray-500 hover:text-gray-400 text-xs"
            >
              {showAdvanced ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          {/* Quick Info */}
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-xl">{selectedConnectionOption.icon}</span>
              <span className={`text-sm font-medium ${selectedConnectionOption.color}`}>
                {selectedConnectionOption.name}
              </span>
            </div>
            <p className="text-gray-400 text-xs">
              {selectedConnectionOption.description}
            </p>
          </div>

          {/* Message Types */}
          <div className="space-y-2">
            <label className="text-gray-300 text-sm font-medium">Message Type</label>
            <select
              value={selectedMessageType}
              onChange={(e) => setSelectedMessageType(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {selectedConnectionOption.messageTypes.map((messageType) => {
                const messageOption = MESSAGE_TYPES.find(opt => opt.type === messageType);
                return (
                  <option key={messageType} value={messageType}>
                    {messageOption?.name || messageType}
                  </option>
                );
              })}
            </select>
          </div>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="space-y-3">
              {/* Use Cases */}
              <div className="space-y-2">
                <h5 className="text-gray-400 text-xs font-medium uppercase">Common Use Cases</h5>
                <div className="space-y-1">
                  {selectedConnectionOption.useCases.map((useCase, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <div className="w-1 h-1 bg-gray-500 rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-gray-400 text-xs">{useCase}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Communication Pattern */}
              <div className="space-y-2">
                <h5 className="text-gray-400 text-xs font-medium uppercase">Communication Pattern</h5>
                <div className="grid grid-cols-3 gap-1 text-xs">
                  {selectedConnectionOption.type === ConnectionType.DIRECT && (
                    <>
                      <div className="bg-gray-800 p-2 rounded text-center">Agent A</div>
                      <div className="flex items-center justify-center text-gray-500">→</div>
                      <div className="bg-gray-800 p-2 rounded text-center">Agent B</div>
                    </>
                  )}
                  {selectedConnectionOption.type === ConnectionType.BROADCAST && (
                    <>
                      <div className="bg-gray-800 p-2 rounded text-center">Sender</div>
                      <div className="flex items-center justify-center text-gray-500">⟲</div>
                      <div className="bg-gray-800 p-2 rounded text-center">Multiple</div>
                    </>
                  )}
                  {selectedConnectionOption.type === ConnectionType.CHAIN && (
                    <>
                      <div className="bg-gray-800 p-2 rounded text-center">A</div>
                      <div className="flex items-center justify-center text-gray-500">→ B →</div>
                      <div className="bg-gray-800 p-2 rounded text-center">C</div>
                    </>
                  )}
                  {selectedConnectionOption.type === ConnectionType.COLLABORATIVE && (
                    <>
                      <div className="bg-gray-800 p-2 rounded text-center">All</div>
                      <div className="flex items-center justify-center text-gray-500">⟷</div>
                      <div className="bg-gray-800 p-2 rounded text-center">All</div>
                    </>
                  )}
                </div>
              </div>

              {/* Advanced Settings */}
              <div className="space-y-2">
                <h5 className="text-gray-400 text-xs font-medium uppercase">Advanced Settings</h5>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded bg-gray-700 border-gray-600" />
                    <span className="text-gray-300 text-xs">Enable message priority</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded bg-gray-700 border-gray-600" defaultChecked />
                    <span className="text-gray-300 text-xs">Allow asynchronous communication</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded bg-gray-700 border-gray-600" />
                    <span className="text-gray-300 text-xs">Enable message persistence</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="space-y-2">
            {isCreatingConnection ? (
              <div className="space-y-2">
                <button
                  onClick={handleCreateConnection}
                  disabled={!sourceNodeId}
                  className={`
                    w-full px-4 py-2 rounded-lg font-medium text-sm transition-colors
                    ${sourceNodeId
                      ? 'bg-blue-500 hover:bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                    }
                  `}
                >
                  {sourceNodeId ? 'Create Connection' : 'Select Source Agent'}
                </button>
                <button
                  onClick={() => onConnectionTypeChange(ConnectionType.DIRECT)}
                  className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg font-medium text-sm transition-colors"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => onConnectionTypeChange(selectedConnectionType)}
                className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium text-sm transition-colors"
              >
                Start Creating Connection
              </button>
            )}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
        <h4 className="text-white text-sm font-medium mb-2">How to create connections:</h4>
        <ul className="text-gray-400 text-xs space-y-1">
          <li>• Select a connection type above</li>
          <li>• Choose the message type for communication</li>
          <li>• Click "Start Creating Connection"</li>
          <li>• Click on source agent, then target agent</li>
          <li>• Connection will be automatically created</li>
        </ul>
      </div>
    </div>
  );
};