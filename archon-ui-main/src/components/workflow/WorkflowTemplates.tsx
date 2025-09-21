import React, { useState, useMemo } from 'react';

import {
  WorkflowTemplate,
  AgentType,
  ModelTier,
  ExtendedAgentNode,
  ExtendedCommunicationEdge,
  CommunicationType,
  AgentState,
} from '../../types/workflowTypes';
import { AGENT_TYPE_ICONS } from '../../types/agentTypes';

interface WorkflowTemplatesProps {
  onClose: () => void;
  onTemplateSelect: (templateId: string) => void;
  projectId?: string;
  className?: string;
}

interface TemplateCardProps {
  template: WorkflowTemplate;
  onSelect: () => void;
}

// Pre-built workflow templates
const WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: 'development-workflow',
    name: 'Development Workflow',
    description: 'Complete software development pipeline from planning to deployment',
    category: 'development',
    agents: [
      {
        agent_type: AgentType.STRATEGIC_PLANNER,
        model_tier: ModelTier.OPUS,
        name: 'Strategic Planner',
        capabilities: { strategic_planning: true, roadmap_creation: true },
        position: { x: 100, y: 100 }
      },
      {
        agent_type: AgentType.SYSTEM_ARCHITECT,
        model_tier: ModelTier.OPUS,
        name: 'System Architect',
        capabilities: { architecture_design: true, technology_selection: true },
        position: { x: 300, y: 100 }
      },
      {
        agent_type: AgentType.CODE_IMPLEMENTER,
        model_tier: ModelTier.SONNET,
        name: 'Code Implementer',
        capabilities: { code_generation: true, error_handling: true },
        position: { x: 500, y: 100 }
      },
      {
        agent_type: AgentType.CODE_QUALITY_REVIEWER,
        model_tier: ModelTier.SONNET,
        name: 'Code Reviewer',
        capabilities: { code_analysis: true, best_practice_validation: true },
        position: { x: 500, y: 250 }
      },
      {
        agent_type: AgentType.TEST_COVERAGE_VALIDATOR,
        model_tier: ModelTier.SONNET,
        name: 'Test Validator',
        capabilities: { test_generation: true, coverage_analysis: true },
        position: { x: 300, y: 250 }
      },
      {
        agent_type: AgentType.DEPLOYMENT_AUTOMATION,
        model_tier: ModelTier.SONNET,
        name: 'Deployer',
        capabilities: { ci_cd_automation: true, deployment_management: true },
        position: { x: 100, y: 250 }
      }
    ],
    connections: [
      { source_index: 0, target_index: 1, communication_type: CommunicationType.DIRECT, message_type: 'task_assignment' },
      { source_index: 1, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'specification_delivery' },
      { source_index: 2, target_index: 3, communication_type: CommunicationType.DIRECT, message_type: 'code_submission' },
      { source_index: 3, target_index: 4, communication_type: CommunicationType.DIRECT, message_type: 'review_result' },
      { source_index: 4, target_index: 5, communication_type: CommunicationType.DIRECT, message_type: 'test_approval' },
      { source_index: 3, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'feedback' },
      { source_index: 4, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'test_results' }
    ],
    metadata: {
      created_by: 'Archon Team',
      created_at: new Date('2024-01-01'),
      usage_count: 156,
      rating: 4.8,
      tags: ['development', 'coding', 'testing', 'deployment']
    }
  },
  {
    id: 'security-audit-workflow',
    name: 'Security Audit Workflow',
    description: 'Comprehensive security audit and vulnerability assessment',
    category: 'testing',
    agents: [
      {
        agent_type: AgentType.SECURITY_AUDITOR,
        model_tier: ModelTier.OPUS,
        name: 'Security Lead',
        capabilities: { vulnerability_scanning: true, security_analysis: true },
        position: { x: 200, y: 150 }
      },
      {
        agent_type: AgentType.CODE_IMPLEMENTER,
        model_tier: ModelTier.SONNET,
        name: 'Security Scanner',
        capabilities: { code_generation: true, security_analysis: true },
        position: { x: 400, y: 100 }
      },
      {
        agent_type: AgentType.ANTIHALLUCINATION_VALIDATOR,
        model_tier: ModelTier.HAIKU,
        name: 'Validation Agent',
        capabilities: { fact_validation: true, output_verification: true },
        position: { x: 400, y: 200 }
      },
      {
        agent_type: AgentType.DATABASE_ARCHITECT,
        model_tier: ModelTier.SONNET,
        name: 'Database Security',
        capabilities: { security_analysis: true, vulnerability_scanning: true },
        position: { x: 200, y: 300 }
      }
    ],
    connections: [
      { source_index: 0, target_index: 1, communication_type: CommunicationType.BROADCAST, message_type: 'scan_request' },
      { source_index: 0, target_index: 2, communication_type: CommunicationType.BROADCAST, message_type: 'validation_request' },
      { source_index: 0, target_index: 3, communication_type: CommunicationType.BROADCAST, message_type: 'database_audit' },
      { source_index: 1, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'scan_results' },
      { source_index: 3, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'database_findings' },
      { source_index: 2, target_index: 0, communication_type: CommunicationType.DIRECT, message_type: 'validation_report' }
    ],
    metadata: {
      created_by: 'Security Team',
      created_at: new Date('2024-02-01'),
      usage_count: 89,
      rating: 4.6,
      tags: ['security', 'audit', 'vulnerability', 'compliance']
    }
  },
  {
    id: 'collaborative-planning',
    name: 'Collaborative Planning',
    description: 'Multi-agent collaborative planning and decision making',
    category: 'collaboration',
    agents: [
      {
        agent_type: AgentType.STRATEGIC_PLANNER,
        model_tier: ModelTier.OPUS,
        name: 'Lead Planner',
        capabilities: { strategic_planning: true, risk_assessment: true },
        position: { x: 300, y: 100 }
      },
      {
        agent_type: AgentType.SYSTEM_ARCHITECT,
        model_tier: ModelTier.SONNET,
        name: 'Technical Architect',
        capabilities: { architecture_design: true, scalability_planning: true },
        position: { x: 150, y: 200 }
      },
      {
        agent_type: AgentType.DATABASE_ARCHITECT,
        model_tier: ModelTier.SONNET,
        name: 'Data Architect',
        capabilities: { schema_design: true, data_modeling: true },
        position: { x: 300, y: 250 }
      },
      {
        agent_type: AgentType.API_DESIGN_ARCHITECT,
        model_tier: ModelTier.SONNET,
        name: 'API Architect',
        capabilities: { api_design: true, endpoint_planning: true },
        position: { x: 450, y: 200 }
      }
    ],
    connections: [
      { source_index: 0, target_index: 1, communication_type: CommunicationType.COLLABORATIVE, message_type: 'planning_session' },
      { source_index: 0, target_index: 2, communication_type: CommunicationType.COLLABORATIVE, message_type: 'planning_session' },
      { source_index: 0, target_index: 3, communication_type: CommunicationType.COLLABORATIVE, message_type: 'planning_session' },
      { source_index: 1, target_index: 2, communication_type: CommunicationType.COLLABORATIVE, message_type: 'technical_discussion' },
      { source_index: 2, target_index: 3, communication_type: CommunicationType.COLLABORATIVE, message_type: 'api_requirements' },
      { source_index: 1, target_index: 3, communication_type: CommunicationType.COLLABORATIVE, message_type: 'integration_points' }
    ],
    metadata: {
      created_by: 'Planning Team',
      created_at: new Date('2024-03-01'),
      usage_count: 234,
      rating: 4.9,
      tags: ['planning', 'collaboration', 'architecture', 'teamwork']
    }
  },
  {
    id: 'performance-optimization',
    name: 'Performance Optimization',
    description: 'System performance analysis and optimization pipeline',
    category: 'analysis',
    agents: [
      {
        agent_type: AgentType.PERFORMANCE_OPTIMIZER,
        model_tier: ModelTier.SONNET,
        name: 'Performance Lead',
        capabilities: { performance_analysis: true, optimization_suggestions: true },
        position: { x: 200, y: 150 }
      },
      {
        agent_type: AgentType.CODE_QUALITY_REVIEWER,
        model_tier: ModelTier.HAIKU,
        name: 'Code Analyzer',
        capabilities: { code_analysis: true, pattern_recognition: true },
        position: { x: 400, y: 100 }
      },
      {
        agent_type: AgentType.DATABASE_ARCHITECT,
        model_tier: ModelTier.SONNET,
        name: 'Database Optimizer',
        capabilities: { query_optimization: true, performance_tuning: true },
        position: { x: 400, y: 200 }
      },
      {
        agent_type: AgentType.SYSTEM_ARCHITECT,
        model_tier: ModelTier.OPUS,
        name: 'Architecture Reviewer',
        capabilities: { scalability_planning: true, performance_optimization: true },
        position: { x: 200, y: 300 }
      }
    ],
    connections: [
      { source_index: 0, target_index: 1, communication_type: CommunicationType.CHAIN, message_type: 'analysis_request' },
      { source_index: 1, target_index: 2, communication_type: CommunicationType.CHAIN, message_type: 'code_analysis_complete' },
      { source_index: 2, target_index: 3, communication_type: CommunicationType.CHAIN, message_type: 'database_analysis_complete' },
      { source_index: 3, target_index: 0, communication_type: CommunicationType.DIRECT, message_type: 'optimization_recommendations' },
      { source_index: 0, target_index: 3, communication_type: CommunicationType.DIRECT, message_type: 'performance_report' }
    ],
    metadata: {
      created_by: 'Performance Team',
      created_at: new Date('2024-04-01'),
      usage_count: 167,
      rating: 4.7,
      tags: ['performance', 'optimization', 'analysis', 'tuning']
    }
  },
  {
    id: 'deployment-pipeline',
    name: 'CI/CD Pipeline',
    description: 'Continuous integration and deployment automation',
    category: 'deployment',
    agents: [
      {
        agent_type: AgentType.DEPLOYMENT_AUTOMATION,
        model_tier: ModelTier.SONNET,
        name: 'Pipeline Coordinator',
        capabilities: { ci_cd_automation: true, deployment_management: true },
        position: { x: 300, y: 150 }
      },
      {
        agent_type: AgentType.CODE_IMPLEMENTER,
        model_tier: ModelTier.HAIKU,
        name: 'Build Agent',
        capabilities: { code_generation: true, build_automation: true },
        position: { x: 150, y: 100 }
      },
      {
        agent_type: AgentType.TEST_COVERAGE_VALIDATOR,
        model_tier: ModelTier.SONNET,
        name: 'Test Runner',
        capabilities: { test_generation: true, integration_testing: true },
        position: { x: 150, y: 200 }
      },
      {
        agent_type: AgentType.SECURITY_AUDITOR,
        model_tier: ModelTier.SONNET,
        name: 'Security Scanner',
        capabilities: { vulnerability_scanning: true, compliance_validation: true },
        position: { x: 450, y: 100 }
      },
      {
        agent_type: AgentType.PERFORMANCE_OPTIMIZER,
        model_tier: ModelTier.HAIKU,
        name: 'Performance Checker',
        capabilities: { performance_analysis: true, benchmarking: true },
        position: { x: 450, y: 200 }
      }
    ],
    connections: [
      { source_index: 0, target_index: 1, communication_type: CommunicationType.DIRECT, message_type: 'build_request' },
      { source_index: 0, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'test_request' },
      { source_index: 1, target_index: 2, communication_type: CommunicationType.DIRECT, message_type: 'build_complete' },
      { source_index: 0, target_index: 3, communication_type: CommunicationType.DIRECT, message_type: 'security_scan' },
      { source_index: 0, target_index: 4, communication_type: CommunicationType.DIRECT, message_type: 'performance_check' },
      { source_index: 2, target_index: 0, communication_type: CommunicationType.DIRECT, message_type: 'test_results' },
      { source_index: 3, target_index: 0, communication_type: CommunicationType.DIRECT, message_type: 'security_report' },
      { source_index: 4, target_index: 0, communication_type: CommunicationType.DIRECT, message_type: 'performance_metrics' }
    ],
    metadata: {
      created_by: 'DevOps Team',
      created_at: new Date('2024-05-01'),
      usage_count: 298,
      rating: 4.5,
      tags: ['deployment', 'ci-cd', 'automation', 'devops']
    }
  }
];

const TemplateCard: React.FC<TemplateCardProps> = ({ template, onSelect }) => {
  const [isHovered, setIsHovered] = useState(false);

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'development': return 'bg-blue-500/20 text-blue-300 border-blue-500/50';
      case 'testing': return 'bg-green-500/20 text-green-300 border-green-500/50';
      case 'deployment': return 'bg-purple-500/20 text-purple-300 border-purple-500/50';
      case 'collaboration': return 'bg-orange-500/20 text-orange-300 border-orange-500/50';
      case 'analysis': return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
    }
  };

  const totalAgents = template.agents.length;
  const totalConnections = template.connections.length;

  return (
    <div
      className={`
        p-4 bg-gray-800 border border-gray-700 rounded-lg cursor-pointer transition-all duration-200
        ${isHovered ? 'border-blue-500 shadow-lg scale-105' : 'hover:border-gray-600'}
      `}
      onClick={onSelect}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <h4 className="text-white font-medium text-lg">{template.name}</h4>
        <span className={`px-2 py-1 rounded text-xs font-medium border ${getCategoryColor(template.category)}`}>
          {template.category}
        </span>
      </div>

      {/* Description */}
      <p className="text-gray-400 text-sm mb-4 line-clamp-2">
        {template.description}
      </p>

      {/* Stats */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-1">
            <span className="text-gray-500">🤖</span>
            <span className="text-gray-300">{totalAgents} agents</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="text-gray-500">🔗</span>
            <span className="text-gray-300">{totalConnections} connections</span>
          </div>
        </div>
        <div className="flex items-center space-x-1">
          <span className="text-yellow-400">⭐</span>
          <span className="text-gray-300 text-sm">{template.metadata.rating}</span>
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-4">
        {template.metadata.tags.slice(0, 3).map((tag, index) => (
          <span
            key={index}
            className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300"
          >
            {tag}
          </span>
        ))}
        {template.metadata.tags.length > 3 && (
          <span className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-500">
            +{template.metadata.tags.length - 3}
          </span>
        )}
      </div>

      {/* Preview */}
      <div className="relative h-32 bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="grid grid-cols-3 gap-2">
            {template.agents.slice(0, 6).map((agent, index) => (
              <div
                key={index}
                className="w-8 h-8 bg-gray-700 rounded-full flex items-center justify-center text-xs"
                title={agent.name}
              >
                {AGENT_TYPE_ICONS[agent.agent_type] || '🤖'}
              </div>
            ))}
          </div>
        </div>

        {/* Overlay */}
        {isHovered && (
          <div className="absolute inset-0 bg-blue-500/20 flex items-center justify-center">
            <div className="text-white font-medium">Use Template</div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
        <span>Used {template.metadata.usage_count} times</span>
        <span>
          {new Date(template.metadata.created_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
};

export const WorkflowTemplates: React.FC<WorkflowTemplatesProps> = ({
  onClose,
  onTemplateSelect,
  projectId,
  className = '',
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowTemplate | null>(null);

  const filteredTemplates = useMemo(() => {
    return WORKFLOW_TEMPLATES.filter(template => {
      const matchesCategory = selectedCategory === 'all' || template.category === selectedCategory;
      const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           template.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           template.metadata.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));

      return matchesCategory && matchesSearch;
    });
  }, [selectedCategory, searchTerm]);

  const categories = useMemo(() => {
    const cats = Array.from(new Set(WORKFLOW_TEMPLATES.map(t => t.category)));
    return ['all', ...cats];
  }, []);

  const handleTemplateSelect = (template: WorkflowTemplate) => {
    setSelectedTemplate(template);
  };

  const handleUseTemplate = () => {
    if (selectedTemplate) {
      onTemplateSelect(selectedTemplate.id);
    }
  };

  return (
    <div className={`fixed inset-0 bg-black/50 flex items-center justify-center z-50 ${className}`}>
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-6xl h-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div>
            <h2 className="text-white text-xl font-semibold">Workflow Templates</h2>
            <p className="text-gray-400 text-sm">Choose a template to quickly start your workflow</p>
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

        {/* Controls */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="flex-1 relative">
              <input
                type="text"
                placeholder="Search templates..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 pl-10 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <svg
                className="absolute left-3 top-2.5 w-4 h-4 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>

            {/* Category Filter */}
            <div className="flex space-x-2">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    selectedCategory === category
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {selectedTemplate ? (
            // Template Detail View
            <div className="h-full flex">
              {/* Left Panel - Template Details */}
              <div className="flex-1 p-6 overflow-y-auto">
                <div className="space-y-6">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-white text-xl font-semibold">{selectedTemplate.name}</h3>
                      <p className="text-gray-400 mt-1">{selectedTemplate.description}</p>
                    </div>
                    <button
                      onClick={() => setSelectedTemplate(null)}
                      className="text-gray-400 hover:text-white transition-colors"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                      </svg>
                    </button>
                  </div>

                  {/* Agents */}
                  <div className="space-y-3">
                    <h4 className="text-gray-300 font-medium">Agents ({selectedTemplate.agents.length})</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {selectedTemplate.agents.map((agent, index) => (
                        <div key={index} className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                          <div className="flex items-center space-x-3">
                            <span className="text-lg">{AGENT_TYPE_ICONS[agent.agent_type]}</span>
                            <div className="flex-1">
                              <div className="text-white text-sm font-medium">{agent.name}</div>
                              <div className="text-gray-400 text-xs">{agent.agent_type.replace(/_/g, ' ')} • {agent.model_tier}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Connections */}
                  <div className="space-y-3">
                    <h4 className="text-gray-300 font-medium">Connections ({selectedTemplate.connections.length})</h4>
                    <div className="space-y-2">
                      {selectedTemplate.connections.map((connection, index) => {
                        const sourceAgent = selectedTemplate.agents[connection.source_index];
                        const targetAgent = selectedTemplate.agents[connection.target_index];
                        return (
                          <div key={index} className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                            <div className="flex items-center space-x-2 text-sm">
                              <span className="text-gray-300">{sourceAgent.name}</span>
                              <span className="text-gray-500">→</span>
                              <span className="text-gray-300">{targetAgent.name}</span>
                              <span className="text-gray-500">•</span>
                              <span className="text-blue-400">{connection.communication_type}</span>
                            </div>
                            <div className="text-gray-500 text-xs mt-1">
                              {connection.message_type}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Metadata */}
                  <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="text-gray-500 text-xs">Created By</div>
                        <div className="text-gray-300">{selectedTemplate.metadata.created_by}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">Usage Count</div>
                        <div className="text-gray-300">{selectedTemplate.metadata.usage_count}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">Rating</div>
                        <div className="text-gray-300">⭐ {selectedTemplate.metadata.rating}</div>
                      </div>
                      <div>
                        <div className="text-gray-500 text-xs">Created</div>
                        <div className="text-gray-300">
                          {new Date(selectedTemplate.metadata.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Panel - Actions */}
              <div className="w-80 p-6 border-l border-gray-700">
                <div className="space-y-4">
                  <h4 className="text-gray-300 font-medium">Actions</h4>

                  <button
                    onClick={handleUseTemplate}
                    className="w-full px-4 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium transition-colors"
                  >
                    Use This Template
                  </button>

                  <div className="p-4 bg-gray-800 rounded-lg">
                    <h5 className="text-gray-300 text-sm font-medium mb-2">Preview</h5>
                    <div className="h-48 bg-gray-900 rounded border border-gray-700 flex items-center justify-center">
                      <div className="text-gray-500 text-sm">Workflow visualization preview</div>
                    </div>
                  </div>

                  <div className="p-4 bg-gray-800 rounded-lg">
                    <h5 className="text-gray-300 text-sm font-medium mb-2">Tags</h5>
                    <div className="flex flex-wrap gap-1">
                      {selectedTemplate.metadata.tags.map((tag, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Template Grid View
            <div className="h-full overflow-y-auto p-6">
              {filteredTemplates.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-gray-500 text-lg mb-2">No templates found</div>
                  <div className="text-gray-600 text-sm mb-4">
                    Try adjusting your search or filters
                  </div>
                  <button
                    onClick={() => {
                      setSearchTerm('');
                      setSelectedCategory('all');
                    }}
                    className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm transition-colors"
                  >
                    Clear Filters
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredTemplates.map((template) => (
                    <TemplateCard
                      key={template.id}
                      template={template}
                      onSelect={() => handleTemplateSelect(template)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-700 flex items-center justify-between">
          <div className="text-gray-500 text-sm">
            {filteredTemplates.length} template{filteredTemplates.length !== 1 ? 's' : ''} available
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};