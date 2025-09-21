import React, { useState, useCallback } from 'react';

import {
  SelectedElement,
  AgentProperties,
  ConnectionProperties,
  AgentState,
  ModelTier,
  AgentType,
  CommunicationType,
} from '../../types/workflowTypes';
import { AGENT_TYPE_ICONS } from '../../types/agentTypes';

interface PropertyEditorProps {
  selectedElement: SelectedElement;
  onChange: (element: SelectedElement) => void;
  className?: string;
}

export const PropertyEditor: React.FC<PropertyEditorProps> = ({
  selectedElement,
  onChange,
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'basic' | 'advanced' | 'capabilities'>('basic');

  const handleAgentPropertyChange = useCallback((updates: Partial<AgentProperties>) => {
    if (selectedElement.type === 'agent') {
      onChange({
        type: 'agent',
        data: { ...selectedElement.data, ...updates }
      });
    }
  }, [selectedElement, onChange]);

  const handleConnectionPropertyChange = useCallback((updates: Partial<ConnectionProperties>) => {
    if (selectedElement.type === 'connection') {
      onChange({
        type: 'connection',
        data: { ...selectedElement.data, ...updates }
      });
    }
  }, [selectedElement, onChange]);

  const handleCapabilityChange = useCallback((key: string, value: any) => {
    if (selectedElement.type === 'agent') {
      const updatedCapabilities = {
        ...selectedElement.data.capabilities,
        [key]: value
      };
      handleAgentPropertyChange({ capabilities: updatedCapabilities });
    }
  }, [selectedElement, handleAgentPropertyChange]);

  if (selectedElement.type === 'none') {
    return (
      <div className={`text-center py-8 ${className}`}>
        <div className="text-gray-500 text-sm">No element selected</div>
        <div className="text-gray-600 text-xs mt-2">
          Select an agent or connection to edit its properties
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="space-y-2">
        <h3 className="text-white text-lg font-semibold">
          {selectedElement.type === 'agent' ? 'Agent Properties' : 'Connection Properties'}
        </h3>
        <div className="flex items-center space-x-2">
          {selectedElement.type === 'agent' && (
            <>
              <span className="text-xl">{AGENT_TYPE_ICONS[selectedElement.data.agent_type]}</span>
              <span className="text-gray-300 text-sm">
                {selectedElement.data.name}
              </span>
            </>
          )}
          {selectedElement.type === 'connection' && (
            <span className="text-gray-300 text-sm">
              {selectedElement.data.source_agent_id} → {selectedElement.data.target_agent_id}
            </span>
          )}
        </div>
      </div>

      {/* Tabs */}
      {selectedElement.type === 'agent' && (
        <div className="flex space-x-1 bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setActiveTab('basic')}
            className={`
              flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors
              ${activeTab === 'basic' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'}
            `}
          >
            Basic
          </button>
          <button
            onClick={() => setActiveTab('advanced')}
            className={`
              flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors
              ${activeTab === 'advanced' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'}
            `}
          >
            Advanced
          </button>
          <button
            onClick={() => setActiveTab('capabilities')}
            className={`
              flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors
              ${activeTab === 'capabilities' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'}
            `}
          >
            Capabilities
          </button>
        </div>
      )}

      {/* Agent Properties */}
      {selectedElement.type === 'agent' && (
        <div className="space-y-4">
          {/* Basic Properties */}
          {activeTab === 'basic' && (
            <div className="space-y-3">
              {/* Name */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Name</label>
                <input
                  type="text"
                  value={selectedElement.data.name}
                  onChange={(e) => handleAgentPropertyChange({ name: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Agent Type */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Agent Type</label>
                <select
                  value={selectedElement.data.agent_type}
                  onChange={(e) => handleAgentPropertyChange({ agent_type: e.target.value as AgentType })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled
                >
                  {Object.values(AgentType).map((type) => (
                    <option key={type} value={type}>
                      {type.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
                <div className="text-gray-500 text-xs">Agent type cannot be changed after creation</div>
              </div>

              {/* Model Tier */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Model Tier</label>
                <select
                  value={selectedElement.data.model_tier}
                  onChange={(e) => handleAgentPropertyChange({ model_tier: e.target.value as ModelTier })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {Object.values(ModelTier).map((tier) => (
                    <option key={tier} value={tier}>
                      {tier}
                    </option>
                  ))}
                </select>
              </div>

              {/* State */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">State</label>
                <select
                  value={selectedElement.data.state}
                  onChange={(e) => handleAgentPropertyChange({ state: e.target.value as AgentState })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {Object.values(AgentState).map((state) => (
                    <option key={state} value={state}>
                      {state}
                    </option>
                  ))}
                </select>
              </div>

              {/* Position */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Position</label>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-gray-500 text-xs">X</label>
                    <input
                      type="number"
                      value={selectedElement.data.position.x}
                      onChange={(e) => handleAgentPropertyChange({
                        position: { ...selectedElement.data.position, x: parseInt(e.target.value) || 0 }
                      })}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="text-gray-500 text-xs">Y</label>
                    <input
                      type="number"
                      value={selectedElement.data.position.y}
                      onChange={(e) => handleAgentPropertyChange({
                        position: { ...selectedElement.data.position, y: parseInt(e.target.value) || 0 }
                      })}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Advanced Properties */}
          {activeTab === 'advanced' && (
            <div className="space-y-3">
              {/* Rules Profile */}
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Rules Profile ID</label>
                <input
                  type="text"
                  value={selectedElement.data.rules_profile_id || ''}
                  onChange={(e) => handleAgentPropertyChange({ rules_profile_id: e.target.value || undefined })}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter rules profile ID..."
                />
              </div>

              {/* Performance Metrics (Read-only) */}
              <div className="space-y-2">
                <h4 className="text-gray-300 text-sm font-medium">Performance Metrics</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Tasks Completed:</span>
                    <span className="text-gray-300">0</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Success Rate:</span>
                    <span className="text-gray-300">0%</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Avg Completion Time:</span>
                    <span className="text-gray-300">0s</span>
                  </div>
                </div>
              </div>

              {/* Resource Usage (Read-only) */}
              <div className="space-y-2">
                <h4 className="text-gray-300 text-sm font-medium">Resource Usage</h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Memory:</span>
                    <span className="text-gray-300">0 MB</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">CPU:</span>
                    <span className="text-gray-300">0%</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Capabilities */}
          {activeTab === 'capabilities' && (
            <div className="space-y-3">
              <h4 className="text-gray-300 text-sm font-medium">Capabilities</h4>

              {Object.entries(selectedElement.data.capabilities || {}).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-2 bg-gray-800 rounded-lg">
                  <div>
                    <div className="text-gray-300 text-sm capitalize">
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div className="text-gray-500 text-xs">
                      {typeof value === 'boolean' ? 'Boolean' : typeof value}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {typeof value === 'boolean' ? (
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={(e) => handleCapabilityChange(key, e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
                      </label>
                    ) : (
                      <input
                        type={typeof value === 'number' ? 'number' : 'text'}
                        value={value}
                        onChange={(e) => handleCapabilityChange(key,
                          typeof value === 'number' ? parseInt(e.target.value) || 0 : e.target.value
                        )}
                        className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    )}
                  </div>
                </div>
              ))}

              {/* Add New Capability */}
              <div className="mt-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                <div className="text-gray-400 text-xs mb-2">Add custom capability</div>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    placeholder="Capability name"
                    className="flex-1 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <button className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-xs transition-colors">
                    Add
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Connection Properties */}
      {selectedElement.type === 'connection' && (
        <div className="space-y-4">
          {/* Connection Info */}
          <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
            <div className="text-gray-400 text-xs mb-1">Connection ID</div>
            <div className="text-gray-300 text-sm font-mono">{selectedElement.data.id}</div>
          </div>

          {/* Basic Properties */}
          <div className="space-y-3">
            {/* Communication Type */}
            <div className="space-y-1">
              <label className="text-gray-300 text-sm font-medium">Communication Type</label>
              <select
                value={selectedElement.data.communication_type}
                onChange={(e) => handleConnectionPropertyChange({
                  communication_type: e.target.value as CommunicationType
                })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {Object.values(CommunicationType).map((type) => (
                  <option key={type} value={type}>
                    {type.replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            </div>

            {/* Message Type */}
            <div className="space-y-1">
              <label className="text-gray-300 text-sm font-medium">Message Type</label>
              <input
                type="text"
                value={selectedElement.data.message_type}
                onChange={(e) => handleConnectionPropertyChange({ message_type: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Source and Target */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Source Agent</label>
                <div className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg">
                  <div className="text-gray-300 text-sm font-mono">
                    {selectedElement.data.source_agent_id}
                  </div>
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-gray-300 text-sm font-medium">Target Agent</label>
                <div className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg">
                  <div className="text-gray-300 text-sm font-mono">
                    {selectedElement.data.target_agent_id}
                  </div>
                </div>
              </div>
            </div>

            {/* Data Flow */}
            <div className="space-y-2">
              <label className="text-gray-300 text-sm font-medium">Data Flow Configuration</label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    placeholder="Input size"
                    value={selectedElement.data.data_flow?.input_size || ''}
                    onChange={(e) => handleConnectionPropertyChange({
                      data_flow: {
                        ...selectedElement.data.data_flow,
                        input_size: parseInt(e.target.value) || 0
                      }
                    })}
                    className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <span className="text-gray-400 text-sm">bytes</span>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    placeholder="Output size"
                    value={selectedElement.data.data_flow?.output_size || ''}
                    onChange={(e) => handleConnectionPropertyChange({
                      data_flow: {
                        ...selectedElement.data.data_flow,
                        output_size: parseInt(e.target.value) || 0
                      }
                    })}
                    className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <span className="text-gray-400 text-sm">bytes</span>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="number"
                    placeholder="Processing time"
                    value={selectedElement.data.data_flow?.processing_time_ms || ''}
                    onChange={(e) => handleConnectionPropertyChange({
                      data_flow: {
                        ...selectedElement.data.data_flow,
                        processing_time_ms: parseInt(e.target.value) || 0
                      }
                    })}
                    className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <span className="text-gray-400 text-sm">ms</span>
                </div>
              </div>
            </div>

            {/* Metadata */}
            <div className="space-y-2">
              <label className="text-gray-300 text-sm font-medium">Metadata</label>
              <textarea
                value={selectedElement.data.metadata ? JSON.stringify(selectedElement.data.metadata, null, 2) : ''}
                onChange={(e) => {
                  try {
                    const metadata = e.target.value ? JSON.parse(e.target.value) : undefined;
                    handleConnectionPropertyChange({ metadata });
                  } catch {
                    // Invalid JSON, ignore
                  }
                }}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={4}
                placeholder="Enter JSON metadata..."
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};