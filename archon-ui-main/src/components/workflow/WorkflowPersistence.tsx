import React, { useState, useRef } from 'react';

import {
  WorkflowConfiguration,
  WorkflowExportOptions,
  ImportValidationResult,
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
} from '../../types/workflowTypes';
import { useToast } from '../../hooks/useToast';

interface WorkflowPersistenceProps {
  workflowConfig: WorkflowConfiguration;
  onSave?: (config: WorkflowConfiguration) => void;
  onLoad?: (config: WorkflowConfiguration) => void;
  className?: string;
}

interface SavedWorkflow {
  id: string;
  name: string;
  description?: string;
  created_at: Date;
  updated_at: Date;
  version: string;
  tags: string[];
  agent_count: number;
  connection_count: number;
}

export const WorkflowPersistence: React.FC<WorkflowPersistenceProps> = ({
  workflowConfig,
  onSave,
  onLoad,
  className = '',
}) => {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'save' | 'load' | 'export'>('save');
  const [savedWorkflows, setSavedWorkflows] = useState<SavedWorkflow[]>([]);
  const [exportOptions, setExportOptions] = useState<WorkflowExportOptions>({
    format: 'json',
    include_metadata: true,
    include_execution_history: false,
    redact_sensitive_data: false,
  });

  // Load saved workflows from localStorage
  React.useEffect(() => {
    const saved = localStorage.getItem('archon-saved-workflows');
    if (saved) {
      try {
        const workflows = JSON.parse(saved);
        setSavedWorkflows(workflows.map((w: any) => ({
          ...w,
          created_at: new Date(w.created_at),
          updated_at: new Date(w.updated_at),
        })));
      } catch (error) {
        console.error('Failed to load saved workflows:', error);
      }
    }
  }, []);

  const handleSave = async (name: string, description?: string) => {
    const configToSave: WorkflowConfiguration = {
      ...workflowConfig,
      name,
      description,
      updated_at: new Date(),
    };

    // Save to localStorage
    const newSaved: SavedWorkflow = {
      id: configToSave.id,
      name: configToSave.name,
      description: configToSave.description,
      created_at: configToSave.created_at,
      updated_at: configToSave.updated_at,
      version: configToSave.version,
      tags: configToSave.metadata.tags,
      agent_count: configToSave.nodes.length,
      connection_count: configToSave.edges.length,
    };

    const updatedWorkflows = [...savedWorkflows.filter(w => w.id !== configToSave.id), newSaved];
    setSavedWorkflows(updatedWorkflows);
    localStorage.setItem('archon-saved-workflows', JSON.stringify(updatedWorkflows));
    localStorage.setItem(`archon-workflow-${configToSave.id}`, JSON.stringify(configToSave));

    onSave?.(configToSave);

    toast({
      title: "Workflow Saved",
      description: `"${name}" has been saved successfully`,
      variant: "success",
    });

    setIsModalOpen(false);
  };

  const handleLoad = (workflowId: string) => {
    const saved = localStorage.getItem(`archon-workflow-${workflowId}`);
    if (saved) {
      try {
        const config = JSON.parse(saved);
        // Convert date strings back to Date objects
        config.created_at = new Date(config.created_at);
        config.updated_at = new Date(config.updated_at);

        onLoad?.(config);

        toast({
          title: "Workflow Loaded",
          description: `"${config.name}" has been loaded successfully`,
          variant: "success",
        });

        setIsModalOpen(false);
      } catch (error) {
        toast({
          title: "Load Failed",
          description: "Failed to load the selected workflow",
          variant: "error",
        });
      }
    }
  };

  const handleDelete = (workflowId: string) => {
    if (window.confirm('Are you sure you want to delete this workflow?')) {
      const updatedWorkflows = savedWorkflows.filter(w => w.id !== workflowId);
      setSavedWorkflows(updatedWorkflows);
      localStorage.setItem('archon-saved-workflows', JSON.stringify(updatedWorkflows));
      localStorage.removeItem(`archon-workflow-${workflowId}`);

      toast({
        title: "Workflow Deleted",
        description: "The workflow has been deleted",
        variant: "success",
      });
    }
  };

  const handleExport = () => {
    let content: string;
    let filename: string;
    let mimeType: string;

    switch (exportOptions.format) {
      case 'json':
        content = JSON.stringify(workflowConfig, null, 2);
        filename = `${workflowConfig.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
        mimeType = 'application/json';
        break;
      case 'yaml':
        // Simple YAML conversion (in production, use a proper YAML library)
        content = convertToYAML(workflowConfig);
        filename = `${workflowConfig.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.yaml`;
        mimeType = 'text/yaml';
        break;
      case 'xml':
        content = convertToXML(workflowConfig);
        filename = `${workflowConfig.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.xml`;
        mimeType = 'text/xml';
        break;
      default:
        content = JSON.stringify(workflowConfig, null, 2);
        filename = `${workflowConfig.name.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json`;
        mimeType = 'application/json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: "Workflow Exported",
      description: `Workflow exported as ${exportOptions.format.toUpperCase()}`,
      variant: "success",
    });
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        let config: WorkflowConfiguration;

        if (file.name.endsWith('.json')) {
          config = JSON.parse(content);
        } else if (file.name.endsWith('.yaml') || file.name.endsWith('.yml')) {
          // Simple YAML parsing (in production, use a proper YAML library)
          config = parseYAML(content);
        } else {
          throw new Error('Unsupported file format');
        }

        // Validate the imported configuration
        const validation = validateImport(config);
        if (validation.is_valid) {
          // Convert date strings back to Date objects
          config.created_at = new Date(config.created_at);
          config.updated_at = new Date(config.updated_at);

          onLoad?.(config);
          toast({
            title: "Workflow Imported",
            description: `"${config.name}" has been imported successfully`,
            variant: "success",
          });
          setIsModalOpen(false);
        } else {
          toast({
            title: "Import Failed",
            description: "The imported file contains validation errors",
            variant: "error",
          });
        }
      } catch (error) {
        toast({
          title: "Import Failed",
          description: "Failed to parse the imported file",
          variant: "error",
        });
      }
    };
    reader.readAsText(file);

    // Reset the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Helper functions
  const convertToYAML = (obj: any): string => {
    const yaml = require('js-yaml');
    return yaml.dump(obj);
  };

  const convertToXML = (obj: any): string => {
    // Simple XML conversion (in production, use a proper XML library)
    const toXML = (o: any, name: string): string => {
      if (typeof o === 'string') return `<${name}>${o}</${name}>`;
      if (typeof o === 'number') return `<${name}>${o}</${name}>`;
      if (typeof o === 'boolean') return `<${name}>${o}</${name}>`;
      if (Array.isArray(o)) {
        return o.map(item => toXML(item, name.replace(/s$/, ''))).join('');
      }
      if (typeof o === 'object') {
        return `<${name}>\n${Object.entries(o).map(([k, v]) => toXML(v, k)).join('\n')}\n</${name}>`;
      }
      return '';
    };
    return `<?xml version="1.0" encoding="UTF-8"?>\n${toXML(obj, 'workflow')}`;
  };

  const parseYAML = (content: string): WorkflowConfiguration => {
    const yaml = require('js-yaml');
    return yaml.load(content) as WorkflowConfiguration;
  };

  const validateImport = (config: any): ImportValidationResult => {
    const errors = [];
    const warnings = [];

    // Basic validation
    if (!config.name || typeof config.name !== 'string') {
      errors.push({
        id: 'missing-name',
        type: 'error',
        element_type: 'workflow',
        message: 'Workflow name is required',
        severity: 'critical' as any,
      });
    }

    if (!config.nodes || !Array.isArray(config.nodes)) {
      errors.push({
        id: 'invalid-nodes',
        type: 'error',
        element_type: 'workflow',
        message: 'Workflow must have nodes array',
        severity: 'critical' as any,
      });
    }

    if (!config.edges || !Array.isArray(config.edges)) {
      errors.push({
        id: 'invalid-edges',
        type: 'error',
        element_type: 'workflow',
        message: 'Workflow must have edges array',
        severity: 'critical' as any,
      });
    }

    // Check node structure
    if (config.nodes) {
      config.nodes.forEach((node: any, index: number) => {
        if (!node.id || !node.type) {
          errors.push({
            id: `invalid-node-${index}`,
            type: 'error',
            element_type: 'agent',
            element_id: node.id,
            message: `Node at index ${index} is missing required fields`,
            severity: 'high' as any,
          });
        }
      });
    }

    // Check edge structure
    if (config.edges) {
      config.edges.forEach((edge: any, index: number) => {
        if (!edge.id || !edge.source || !edge.target) {
          errors.push({
            id: `invalid-edge-${index}`,
            type: 'error',
            element_type: 'connection',
            element_id: edge.id,
            message: `Edge at index ${index} is missing required fields`,
            severity: 'high' as any,
          });
        }
      });
    }

    return {
      is_valid: errors.length === 0,
      errors,
      warnings,
      compatibility_issues: [],
      estimated_migration_time: errors.length * 5, // 5 seconds per error
    };
  };

  if (!isModalOpen) {
    return (
      <div className={`space-y-2 ${className}`}>
        <button
          onClick={() => {
            setActiveTab('save');
            setIsModalOpen(true);
          }}
          className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
          </svg>
          <span>Save Workflow</span>
        </button>

        <button
          onClick={() => {
            setActiveTab('load');
            setIsModalOpen(true);
          }}
          className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
          </svg>
          <span>Load Workflow</span>
        </button>

        <button
          onClick={() => {
            setActiveTab('export');
            setIsModalOpen(true);
          }}
          className="w-full px-4 py-2 bg-purple-700 hover:bg-purple-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span>Export Workflow</span>
        </button>
      </div>
    );
  }

  return (
    <div className={`fixed inset-0 bg-black/50 flex items-center justify-center z-50 ${className}`}>
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-4xl h-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-white text-xl font-semibold">
              {activeTab === 'save' && 'Save Workflow'}
              {activeTab === 'load' && 'Load Workflow'}
              {activeTab === 'export' && 'Export Workflow'}
            </h2>
            <p className="text-gray-400 text-sm">
              {activeTab === 'save' && 'Save your current workflow for later use'}
              {activeTab === 'load' && 'Load a previously saved workflow'}
              {activeTab === 'export' && 'Export your workflow to share or backup'}
            </p>
          </div>
          <button
            onClick={() => setIsModalOpen(false)}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          {(['save', 'load', 'export'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`
                flex-1 px-4 py-3 text-sm font-medium transition-colors
                ${activeTab === tab
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
                }
              `}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Save Tab */}
          {activeTab === 'save' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-gray-300 text-sm font-medium">Workflow Name</label>
                <input
                  type="text"
                  defaultValue={workflowConfig.name}
                  placeholder="Enter workflow name..."
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div className="space-y-2">
                <label className="text-gray-300 text-sm font-medium">Description (Optional)</label>
                <textarea
                  placeholder="Enter workflow description..."
                  rows={3}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                />
              </div>

              <div className="p-4 bg-gray-800 rounded-lg">
                <h4 className="text-gray-300 font-medium mb-2">Save Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Agents:</span>
                    <span className="text-gray-300 ml-2">{workflowConfig.nodes.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Connections:</span>
                    <span className="text-gray-300 ml-2">{workflowConfig.edges.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Version:</span>
                    <span className="text-gray-300 ml-2">{workflowConfig.version}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Last Modified:</span>
                    <span className="text-gray-300 ml-2">
                      {workflowConfig.updated_at.toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    const name = (document.querySelector('input[type="text"]') as HTMLInputElement)?.value || workflowConfig.name;
                    const description = (document.querySelector('textarea') as HTMLTextAreaElement)?.value;
                    handleSave(name, description);
                  }}
                  className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
                >
                  Save Workflow
                </button>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg font-medium transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Load Tab */}
          {activeTab === 'load' && (
            <div className="space-y-4">
              {/* Import Section */}
              <div className="p-4 bg-gray-800 rounded-lg">
                <h4 className="text-gray-300 font-medium mb-3">Import from File</h4>
                <div className="flex items-center space-x-3">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json,.yaml,.yml"
                    onChange={handleImport}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                    </svg>
                    <span>Import File</span>
                  </button>
                  <div className="text-gray-500 text-sm">
                    Supports JSON, YAML
                  </div>
                </div>
              </div>

              {/* Saved Workflows */}
              <div className="space-y-3">
                <h4 className="text-gray-300 font-medium">Saved Workflows ({savedWorkflows.length})</h4>

                {savedWorkflows.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="text-gray-500 text-lg mb-2">No saved workflows</div>
                    <div className="text-gray-600 text-sm">
                      Save a workflow to see it here
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {savedWorkflows.map(workflow => (
                      <div
                        key={workflow.id}
                        className="p-4 bg-gray-800 border border-gray-700 rounded-lg hover:border-gray-600 transition-colors"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h5 className="text-white font-medium">{workflow.name}</h5>
                            {workflow.description && (
                              <p className="text-gray-400 text-sm mt-1">{workflow.description}</p>
                            )}
                            <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                              <span>{workflow.agent_count} agents</span>
                              <span>{workflow.connection_count} connections</span>
                              <span>Version {workflow.version}</span>
                              <span>Updated {workflow.updated_at.toLocaleDateString()}</span>
                            </div>
                            {workflow.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {workflow.tags.map((tag, index) => (
                                  <span
                                    key={index}
                                    className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>

                          <div className="flex items-center space-x-2 ml-4">
                            <button
                              onClick={() => handleLoad(workflow.id)}
                              className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-xs transition-colors"
                            >
                              Load
                            </button>
                            <button
                              onClick={() => handleDelete(workflow.id)}
                              className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-xs transition-colors"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Export Tab */}
          {activeTab === 'export' && (
            <div className="space-y-4">
              {/* Export Options */}
              <div className="space-y-4">
                <h4 className="text-gray-300 font-medium">Export Options</h4>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-gray-300 text-sm font-medium">Format</label>
                    <select
                      value={exportOptions.format}
                      onChange={(e) => setExportOptions(prev => ({
                        ...prev,
                        format: e.target.value as any
                      }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="json">JSON</option>
                      <option value="yaml">YAML</option>
                      <option value="xml">XML</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-gray-300 text-sm font-medium">Compatibility Version</label>
                    <select
                      value={exportOptions.compatibility_version || 'latest'}
                      onChange={(e) => setExportOptions(prev => ({
                        ...prev,
                        compatibility_version: e.target.value || undefined
                      }))}
                      className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="latest">Latest (v1.0.0)</option>
                      <option value="1.0.0">v1.0.0</option>
                      <option value="0.9.0">v0.9.0 (Legacy)</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-3">
                  <h5 className="text-gray-300 text-sm font-medium">Include</h5>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={exportOptions.include_metadata}
                        onChange={(e) => setExportOptions(prev => ({
                          ...prev,
                          include_metadata: e.target.checked
                        }))}
                        className="rounded bg-gray-700 border-gray-600"
                      />
                      <span className="text-gray-300 text-sm">Metadata (author, tags, etc.)</span>
                    </label>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={exportOptions.include_execution_history}
                        onChange={(e) => setExportOptions(prev => ({
                          ...prev,
                          include_execution_history: e.target.checked
                        }))}
                        className="rounded bg-gray-700 border-gray-600"
                      />
                      <span className="text-gray-300 text-sm">Execution history (if available)</span>
                    </label>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={exportOptions.redact_sensitive_data}
                        onChange={(e) => setExportOptions(prev => ({
                          ...prev,
                          redact_sensitive_data: e.target.checked
                        }))}
                        className="rounded bg-gray-700 border-gray-600"
                      />
                      <span className="text-gray-300 text-sm">Redact sensitive data</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Preview */}
              <div className="space-y-2">
                <h4 className="text-gray-300 font-medium">Preview</h4>
                <div className="p-4 bg-gray-800 rounded-lg">
                  <div className="text-gray-400 text-sm mb-2">
                    Export will include:
                  </div>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• {workflowConfig.nodes.length} agents with configurations</li>
                    <li>• {workflowConfig.edges.length} connections with settings</li>
                    <li>• Workflow metadata and version information</li>
                    {exportOptions.include_metadata && (
                      <li>• Tags, author information, and creation details</li>
                    )}
                    {exportOptions.include_execution_history && (
                      <li>• Execution history and performance metrics</li>
                    )}
                  </ul>
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={handleExport}
                  className="flex-1 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>Export {exportOptions.format.toUpperCase()}</span>
                </button>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg font-medium transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};