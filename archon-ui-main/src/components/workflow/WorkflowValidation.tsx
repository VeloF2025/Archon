import React, { useMemo, useState, useCallback } from 'react';

import {
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
  ValidationError,
  ValidationResult,
  AgentState,
  CommunicationType,
} from '../../types/workflowTypes';

interface WorkflowValidationProps {
  nodes: ExtendedAgentNode[];
  edges: ExtendedCommunicationEdge[];
  onClose: () => void;
  onFix?: (error: ValidationError) => void;
  className?: string;
}

interface ValidationRule {
  id: string;
  name: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  validate: (nodes: ExtendedAgentNode[], edges: ExtendedCommunicationEdge[]) => ValidationError[];
  autoFix?: (nodes: ExtendedAgentNode[], edges: ExtendedCommunicationEdge[]) => { nodes: ExtendedAgentNode[], edges: ExtendedCommunicationEdge[] };
}

// Validation rules implementation
const VALIDATION_RULES: ValidationRule[] = [
  {
    id: 'orphaned-nodes',
    name: 'Orphaned Agents',
    description: 'Agents without any connections may not be able to participate in the workflow',
    severity: 'medium',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      nodes.forEach(node => {
        const hasConnections = edges.some(edge =>
          edge.source === node.id || edge.target === node.id
        );

        if (!hasConnections) {
          errors.push({
            id: `orphaned-${node.id}`,
            type: 'warning',
            element_type: 'agent',
            element_id: node.id,
            message: `Agent "${node.data.agent.name}" has no connections`,
            severity: 'medium',
            auto_fixable: false,
            fix_suggestion: 'Connect this agent to other agents in the workflow or remove it if not needed'
          });
        }
      });

      return errors;
    }
  },

  {
    id: 'isolated-groups',
    name: 'Isolated Groups',
    description: 'Workflow contains disconnected groups of agents',
    severity: 'high',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      if (nodes.length === 0) return errors;

      // Build adjacency list
      const graph = new Map<string, string[]>();
      nodes.forEach(node => graph.set(node.id, []));
      edges.forEach(edge => {
        graph.get(edge.source)?.push(edge.target);
        graph.get(edge.target)?.push(edge.source);
      });

      // Find connected components
      const visited = new Set<string>();
      const components: string[][] = [];

      const dfs = (nodeId: string, component: string[]) => {
        visited.add(nodeId);
        component.push(nodeId);
        graph.get(nodeId)?.forEach(neighbor => {
          if (!visited.has(neighbor)) {
            dfs(neighbor, component);
          }
        });
      };

      nodes.forEach(node => {
        if (!visited.has(node.id)) {
          const component: string[] = [];
          dfs(node.id, component);
          components.push(component);
        }
      });

      if (components.length > 1) {
        errors.push({
          id: 'isolated-groups',
          type: 'error',
          element_type: 'workflow',
          message: `Workflow contains ${components.length} disconnected groups of agents`,
          severity: 'high',
          auto_fixable: false,
          fix_suggestion: 'Connect the isolated groups to create a unified workflow'
        });
      }

      return errors;
    }
  },

  {
    id: 'self-loops',
    name: 'Self-Loops',
    description: 'Agents connected to themselves may cause infinite loops',
    severity: 'critical',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      edges.forEach(edge => {
        if (edge.source === edge.target) {
          errors.push({
            id: `self-loop-${edge.id}`,
            type: 'error',
            element_type: 'connection',
            element_id: edge.id,
            message: `Self-loop detected on agent "${edge.source}"`,
            severity: 'critical',
            auto_fixable: true,
            fix_suggestion: 'Remove self-loop connection or redirect to a different agent'
          });
        }
      });

      return errors;
    },
    autoFix: (nodes, edges) => {
      return {
        nodes,
        edges: edges.filter(edge => edge.source !== edge.target)
      };
    }
  },

  {
    id: 'duplicate-connections',
    name: 'Duplicate Connections',
    description: 'Multiple connections between the same agents may cause confusion',
    severity: 'medium',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];
      const connectionPairs = new Set<string>();

      edges.forEach(edge => {
        const pair = [edge.source, edge.target].sort().join('->');
        if (connectionPairs.has(pair)) {
          errors.push({
            id: `duplicate-${edge.id}`,
            type: 'warning',
            element_type: 'connection',
            element_id: edge.id,
            message: `Duplicate connection between "${edge.source}" and "${edge.target}"`,
            severity: 'medium',
            auto_fixable: true,
            fix_suggestion: 'Remove duplicate connections or merge their functionality'
          });
        }
        connectionPairs.add(pair);
      });

      return errors;
    }
  },

  {
    id: 'inactive-agents',
    name: 'Inactive Agents',
    description: 'Agents in CREATED or HIBERNATED state may not participate in workflow execution',
    severity: 'medium',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      nodes.forEach(node => {
        if (node.data.agent.state === AgentState.CREATED || node.data.agent.state === AgentState.HIBERNATED) {
          errors.push({
            id: `inactive-${node.id}`,
            type: 'warning',
            element_type: 'agent',
            element_id: node.id,
            message: `Agent "${node.data.agent.name}" is in ${node.data.agent.state} state`,
            severity: 'medium',
            auto_fixable: true,
            fix_suggestion: 'Activate the agent or change its state to ACTIVE'
          });
        }
      });

      return errors;
    },
    autoFix: (nodes, edges) => {
      return {
        nodes: nodes.map(node => ({
          ...node,
          data: {
            ...node.data,
            agent: {
              ...node.data.agent,
              state: node.data.agent.state === AgentState.ARCHIVED ? AgentState.ARCHIVED : AgentState.ACTIVE
            }
          }
        })),
        edges
      };
    }
  },

  {
    id: 'missing-endpoints',
    name: 'Missing Connection Endpoints',
    description: 'Connections reference agents that don\'t exist in the workflow',
    severity: 'critical',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];
      const nodeIds = new Set(nodes.map(node => node.id));

      edges.forEach(edge => {
        if (!nodeIds.has(edge.source)) {
          errors.push({
            id: `missing-source-${edge.id}`,
            type: 'error',
            element_type: 'connection',
            element_id: edge.id,
            message: `Connection references missing source agent "${edge.source}"`,
            severity: 'critical',
            auto_fixable: true,
            fix_suggestion: 'Remove the connection or add the missing agent'
          });
        }

        if (!nodeIds.has(edge.target)) {
          errors.push({
            id: `missing-target-${edge.id}`,
            type: 'error',
            element_type: 'connection',
            element_id: edge.id,
            message: `Connection references missing target agent "${edge.target}"`,
            severity: 'critical',
            auto_fixable: true,
            fix_suggestion: 'Remove the connection or add the missing agent'
          });
        }
      });

      return errors;
    },
    autoFix: (nodes, edges) => {
      const nodeIds = new Set(nodes.map(node => node.id));
      return {
        nodes,
        edges: edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      };
    }
  },

  {
    id: 'circular-dependencies',
    name: 'Circular Dependencies',
    description: 'Circular dependencies may cause infinite loops during execution',
    severity: 'high',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      // Build adjacency list for directed graph
      const graph = new Map<string, string[]>();
      nodes.forEach(node => graph.set(node.id, []));
      edges.forEach(edge => {
        if (edge.source !== edge.target) {
          graph.get(edge.source)?.push(edge.target);
        }
      });

      // Detect cycles using DFS
      const visited = new Set<string>();
      const recursionStack = new Set<string>();
      const cycles: string[][] = [];

      const detectCycles = (nodeId: string, path: string[]): void => {
        visited.add(nodeId);
        recursionStack.add(nodeId);

        graph.get(nodeId)?.forEach(neighbor => {
          if (recursionStack.has(neighbor)) {
            // Found a cycle
            const cycleStart = path.indexOf(neighbor);
            const cycle = [...path.slice(cycleStart), neighbor];
            cycles.push(cycle);
          } else if (!visited.has(neighbor)) {
            detectCycles(neighbor, [...path, neighbor]);
          }
        });

        recursionStack.delete(nodeId);
      };

      nodes.forEach(node => {
        if (!visited.has(node.id)) {
          detectCycles(node.id, [node.id]);
        }
      });

      cycles.forEach((cycle, index) => {
        errors.push({
          id: `cycle-${index}`,
          type: 'error',
          element_type: 'workflow',
          message: `Circular dependency detected: ${cycle.join(' → ')} → ${cycle[0]}`,
          severity: 'high',
          auto_fixable: false,
          fix_suggestion: 'Break the cycle by removing or redirecting one of the connections'
        });
      });

      return errors;
    }
  },

  {
    id: 'broadcast-overuse',
    name: 'Broadcast Overuse',
    description: 'Too many broadcast connections may cause performance issues',
    severity: 'low',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      const broadcastCount = edges.filter(edge =>
        edge.data.communication.communication_type === CommunicationType.BROADCAST
      ).length;

      if (broadcastCount > nodes.length * 0.5) {
        errors.push({
          id: 'broadcast-overuse',
          type: 'warning',
          element_type: 'workflow',
          message: `High number of broadcast connections (${broadcastCount}) may impact performance`,
          severity: 'low',
          auto_fixable: false,
          fix_suggestion: 'Consider using direct connections where possible'
        });
      }

      return errors;
    }
  },

  {
    id: 'single-agent-workflow',
    name: 'Single Agent Workflow',
    description: 'Workflows with only one agent may not benefit from multi-agent collaboration',
    severity: 'low',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      if (nodes.length === 1) {
        errors.push({
          id: 'single-agent',
          type: 'warning',
          element_type: 'workflow',
          message: 'Workflow contains only one agent',
          severity: 'low',
          auto_fixable: false,
          fix_suggestion: 'Add more agents to leverage multi-agent collaboration benefits'
        });
      }

      return errors;
    }
  },

  {
    id: 'chain-length',
    name: 'Long Processing Chains',
    description: 'Very long processing chains may cause delays and complexity',
    severity: 'medium',
    validate: (nodes, edges) => {
      const errors: ValidationError[] = [];

      // Build adjacency list for chain detection
      const graph = new Map<string, string[]>();
      nodes.forEach(node => graph.set(node.id, []));
      edges.forEach(edge => {
        if (edge.source !== edge.target) {
          graph.get(edge.source)?.push(edge.target);
        }
      });

      // Find longest path
      const findLongestPath = (nodeId: string, visited: Set<string>): number => {
        visited.add(nodeId);
        let maxLength = 0;

        graph.get(nodeId)?.forEach(neighbor => {
          if (!visited.has(neighbor)) {
            maxLength = Math.max(maxLength, findLongestPath(neighbor, new Set(visited)));
          }
        });

        return maxLength + 1;
      };

      let maxChainLength = 0;
      nodes.forEach(node => {
        maxChainLength = Math.max(maxChainLength, findLongestPath(node.id, new Set()));
      });

      if (maxChainLength > 5) {
        errors.push({
          id: 'long-chain',
          type: 'warning',
          element_type: 'workflow',
          message: `Processing chain length (${maxChainLength}) may be too long`,
          severity: 'medium',
          auto_fixable: false,
          fix_suggestion: 'Consider parallelizing some steps or breaking into smaller workflows'
        });
      }

      return errors;
    }
  }
];

interface ValidationErrorItemProps {
  error: ValidationError;
  onFix?: (error: ValidationError) => void;
}

const ValidationErrorItem: React.FC<ValidationErrorItemProps> = ({ error, onFix }) => {
  const [expanded, setExpanded] = useState(false);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-300 border-red-500/50';
      case 'high': return 'bg-orange-500/20 text-orange-300 border-orange-500/50';
      case 'medium': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/50';
      case 'low': return 'bg-blue-500/20 text-blue-300 border-blue-500/50';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return '🚨';
      case 'high': return '⚠️';
      case 'medium': return '⚡';
      case 'low': return 'ℹ️';
      default: return '❓';
    }
  };

  return (
    <div className={`p-3 rounded-lg border ${getSeverityColor(error.severity)} mb-2`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-2 flex-1">
          <span className="text-lg mt-0.5">{getSeverityIcon(error.severity)}</span>
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-1">
              <span className="text-white font-medium text-sm capitalize">
                {error.severity}
              </span>
              {error.auto_fixable && (
                <span className="px-2 py-0.5 bg-green-500/20 text-green-300 rounded text-xs">
                  Auto-fixable
                </span>
              )}
            </div>
            <p className="text-gray-300 text-sm">{error.message}</p>

            {expanded && error.fix_suggestion && (
              <div className="mt-2 p-2 bg-gray-800/50 rounded border border-gray-700">
                <div className="text-gray-400 text-xs mb-1">Suggested fix:</div>
                <div className="text-gray-300 text-sm">{error.fix_suggestion}</div>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {error.fix_suggestion && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-gray-400 hover:text-gray-300 transition-colors"
              title={expanded ? 'Hide details' : 'Show details'}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d={expanded ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"}
                />
              </svg>
            </button>
          )}

          {error.auto_fixable && onFix && (
            <button
              onClick={() => onFix(error)}
              className="px-2 py-1 bg-green-500 hover:bg-green-600 text-white rounded text-xs transition-colors"
              title="Auto-fix"
            >
              Fix
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export const WorkflowValidation: React.FC<WorkflowValidationProps> = ({
  nodes,
  edges,
  onClose,
  onFix,
  className = '',
}) => {
  const [selectedRule, setSelectedRule] = useState<string>('all');
  const [autoFixEnabled, setAutoFixEnabled] = useState(false);

  // Run all validation rules
  const validationResult = useMemo(() => {
    const allErrors: ValidationError[] = [];

    VALIDATION_RULES.forEach(rule => {
      if (selectedRule === 'all' || selectedRule === rule.id) {
        const ruleErrors = rule.validate(nodes, edges);
        allErrors.push(...ruleErrors);
      }
    });

    const errors = allErrors.filter(e => e.type === 'error');
    const warnings = allErrors.filter(e => e.type === 'warning');

    const score = Math.max(0, 1 - (errors.length * 0.3 + warnings.length * 0.1) / Math.max(nodes.length, 1));

    return {
      is_valid: errors.length === 0,
      errors,
      warnings,
      score,
      can_execute: errors.filter(e => e.severity === 'critical').length === 0
    };
  }, [nodes, edges, selectedRule]);

  const handleAutoFix = useCallback((error: ValidationError) => {
    const rule = VALIDATION_RULES.find(r => r.id === error.id);
    if (rule?.autoFix) {
      const fixed = rule.autoFix(nodes, edges);
      onFix?.(error);
      // In a real implementation, you would update the nodes and edges state
      console.log('Auto-fixed:', error.message, fixed);
    }
  }, [nodes, edges, onFix]);

  const handleAutoFixAll = useCallback(() => {
    let currentNodes = [...nodes];
    let currentEdges = [...edges];
    let fixedCount = 0;

    VALIDATION_RULES.forEach(rule => {
      if (rule.autoFix) {
        const errors = rule.validate(currentNodes, currentEdges);
        if (errors.length > 0) {
          const fixed = rule.autoFix(currentNodes, currentEdges);
          currentNodes = fixed.nodes;
          currentEdges = fixed.edges;
          fixedCount += errors.length;
        }
      }
    });

    if (fixedCount > 0) {
      console.log(`Auto-fixed ${fixedCount} issues`);
      // In a real implementation, you would update the state
    }
  }, [nodes, edges]);

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-green-400';
    if (score >= 0.7) return 'text-yellow-400';
    if (score >= 0.5) return 'text-orange-400';
    return 'text-red-400';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.9) return 'Excellent';
    if (score >= 0.7) return 'Good';
    if (score >= 0.5) return 'Fair';
    return 'Poor';
  };

  return (
    <div className={`fixed inset-0 bg-black/50 flex items-center justify-center z-50 ${className}`}>
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-4xl h-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-white text-xl font-semibold">Workflow Validation</h2>
            <p className="text-gray-400 text-sm">
              Validate your workflow for potential issues and best practices
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Summary */}
        <div className="p-6 border-b border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className={`text-2xl font-bold ${getScoreColor(validationResult.score)}`}>
                {(validationResult.score * 100).toFixed(0)}%
              </div>
              <div className="text-gray-400 text-sm">Validation Score</div>
              <div className={`text-xs mt-1 ${getScoreColor(validationResult.score)}`}>
                {getScoreLabel(validationResult.score)}
              </div>
            </div>

            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-green-400">
                {validationResult.errors.length === 0 ? '✓' : validationResult.errors.length}
              </div>
              <div className="text-gray-400 text-sm">Errors</div>
            </div>

            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-yellow-400">
                {validationResult.warnings.length}
              </div>
              <div className="text-gray-400 text-sm">Warnings</div>
            </div>

            <div className="text-center p-4 bg-gray-800 rounded-lg">
              <div className="text-2xl font-bold text-blue-400">
                {nodes.length}
              </div>
              <div className="text-gray-400 text-sm">Agents</div>
            </div>
          </div>

          {/* Status */}
          <div className="mt-4 flex items-center justify-between">
            <div className={`px-3 py-2 rounded-lg text-sm font-medium ${
              validationResult.is_valid
                ? 'bg-green-500/20 text-green-300'
                : validationResult.can_execute
                ? 'bg-yellow-500/20 text-yellow-300'
                : 'bg-red-500/20 text-red-300'
            }`}>
              {validationResult.is_valid
                ? '✓ Workflow is valid and ready to execute'
                : validationResult.can_execute
                ? '⚠️ Workflow has issues but can execute'
                : '🚫 Workflow has critical issues that prevent execution'
              }
            </div>

            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={autoFixEnabled}
                  onChange={(e) => setAutoFixEnabled(e.target.checked)}
                  className="rounded bg-gray-700 border-gray-600"
                />
                <span className="text-gray-300">Enable auto-fix</span>
              </label>

              <button
                onClick={handleAutoFixAll}
                disabled={!autoFixEnabled}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  autoFixEnabled
                    ? 'bg-green-500 hover:bg-green-600 text-white'
                    : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
              >
                Auto-fix All
              </button>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <label className="text-gray-300 text-sm font-medium">Filter by rule:</label>
            <select
              value={selectedRule}
              onChange={(e) => setSelectedRule(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Rules</option>
              {VALIDATION_RULES.map(rule => (
                <option key={rule.id} value={rule.id}>
                  {rule.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Validation Results */}
        <div className="flex-1 overflow-y-auto p-6">
          {validationResult.errors.length === 0 && validationResult.warnings.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">✅</div>
              <div className="text-green-400 text-xl font-medium mb-2">
                No Issues Found
              </div>
              <div className="text-gray-400">
                Your workflow has passed all validation checks!
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Errors */}
              {validationResult.errors.length > 0 && (
                <div>
                  <h3 className="text-red-400 font-medium mb-3 flex items-center">
                    <span className="mr-2">🚨</span>
                    Errors ({validationResult.errors.length})
                  </h3>
                  {validationResult.errors.map(error => (
                    <ValidationErrorItem
                      key={error.id}
                      error={error}
                      onFix={autoFixEnabled ? handleAutoFix : undefined}
                    />
                  ))}
                </div>
              )}

              {/* Warnings */}
              {validationResult.warnings.length > 0 && (
                <div>
                  <h3 className="text-yellow-400 font-medium mb-3 flex items-center">
                    <span className="mr-2">⚠️</span>
                    Warnings ({validationResult.warnings.length})
                  </h3>
                  {validationResult.warnings.map(error => (
                    <ValidationErrorItem
                      key={error.id}
                      error={error}
                      onFix={autoFixEnabled ? handleAutoFix : undefined}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Rules Info */}
        <div className="p-4 border-t border-gray-700">
          <details className="text-sm">
            <summary className="cursor-pointer text-gray-400 hover:text-gray-300">
              View validation rules ({VALIDATION_RULES.length})
            </summary>
            <div className="mt-2 space-y-2 max-h-32 overflow-y-auto">
              {VALIDATION_RULES.map(rule => (
                <div key={rule.id} className="flex items-center justify-between p-2 bg-gray-800 rounded">
                  <div>
                    <div className="text-white font-medium">{rule.name}</div>
                    <div className="text-gray-400 text-xs">{rule.description}</div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    rule.severity === 'critical' ? 'bg-red-500/20 text-red-300' :
                    rule.severity === 'high' ? 'bg-orange-500/20 text-orange-300' :
                    rule.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-300' :
                    'bg-blue-500/20 text-blue-300'
                  }`}>
                    {rule.severity}
                  </span>
                </div>
              ))}
            </div>
          </details>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex items-center justify-between">
          <div className="text-gray-500 text-sm">
            Validation completed at {new Date().toLocaleTimeString()}
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => {
                // Refresh validation
                console.log('Refreshing validation...');
              }}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
            >
              Refresh
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};