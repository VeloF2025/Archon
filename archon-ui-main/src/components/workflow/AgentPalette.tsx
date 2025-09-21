import React, { useMemo } from 'react';
import { useDrag } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { DndProvider } from 'react-dnd';

import {
  AgentType,
  ModelTier,
  AgentV3,
  AGENT_TYPE_ICONS,
} from '../../types/agentTypes';
import { DraggedAgent } from '../../types/workflowTypes';

interface AgentPaletteProps {
  onAgentDragStart?: () => void;
  onAgentDragEnd?: () => void;
  className?: string;
}

interface DraggableAgentItemProps {
  agentType: AgentType;
  modelTier: ModelTier;
  name: string;
  description: string;
  capabilities: Record<string, any>;
  onDragStart?: () => void;
  onDragEnd?: () => void;
}

const DraggableAgentItem: React.FC<DraggableAgentItemProps> = ({
  agentType,
  modelTier,
  name,
  description,
  capabilities,
  onDragStart,
  onDragEnd,
}) => {
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'agent',
    item: {
      agentType,
      modelTier,
      name,
      description,
      capabilities,
      defaultConfig: createDefaultAgentConfig(agentType, modelTier, name, capabilities),
    } as DraggedAgent,
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  }));

  const tierColor = useMemo(() => {
    switch (modelTier) {
      case ModelTier.OPUS:
        return 'border-purple-500/50 bg-purple-500/10';
      case ModelTier.SONNET:
        return 'border-blue-500/50 bg-blue-500/10';
      case ModelTier.HAIKU:
        return 'border-green-500/50 bg-green-500/10';
      default:
        return 'border-gray-500/50 bg-gray-500/10';
    }
  }, [modelTier]);

  const handleDragStart = () => {
    onDragStart?.();
  };

  const handleDragEnd = () => {
    onDragEnd?.();
  };

  return (
    <div
      ref={drag}
      className={`
        relative p-3 mb-2 rounded-lg border cursor-grab transition-all duration-200
        ${tierColor}
        ${isDragging ? 'opacity-50 cursor-grabbing' : 'hover:scale-105 hover:shadow-lg'}
        backdrop-blur-sm
      `}
      draggable
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
    >
      <div className="flex items-start space-x-3">
        <div className="text-2xl flex-shrink-0">
          {AGENT_TYPE_ICONS[agentType] || '🤖'}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-1">
            <h4 className="text-white font-medium text-sm truncate">
              {name}
            </h4>
            <span className={`
              px-2 py-0.5 rounded text-xs font-medium ml-2 flex-shrink-0
              ${modelTier === ModelTier.OPUS ? 'bg-purple-500/20 text-purple-300' :
                modelTier === ModelTier.SONNET ? 'bg-blue-500/20 text-blue-300' :
                'bg-green-500/20 text-green-300'}
            `}>
              {modelTier}
            </span>
          </div>

          <p className="text-gray-400 text-xs mb-2 line-clamp-2">
            {description}
          </p>

          {/* Capabilities */}
          <div className="space-y-1">
            {Object.entries(capabilities).slice(0, 3).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-gray-500 text-xs capitalize">
                  {key.replace(/_/g, ' ')}:
                </span>
                <span className="text-gray-300 text-xs">
                  {typeof value === 'boolean' ? (value ? '✓' : '✗') :
                   typeof value === 'number' ? value :
                   typeof value === 'string' && value.length > 10 ? `${value.substring(0, 10)}...` : value}
                </span>
              </div>
            ))}
            {Object.keys(capabilities).length > 3 && (
              <div className="text-gray-500 text-xs">
                +{Object.keys(capabilities).length - 3} more...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Drag handle */}
      <div className="absolute bottom-2 right-2 text-gray-600">
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z" />
        </svg>
      </div>

      {/* Dragging overlay */}
      {isDragging && (
        <div className="absolute inset-0 bg-white/10 rounded-lg flex items-center justify-center">
          <div className="text-white text-sm font-medium">Dragging...</div>
        </div>
      )}
    </div>
  );
};

// Helper function to create default agent configuration
const createDefaultAgentConfig = (
  agentType: AgentType,
  modelTier: ModelTier,
  name: string,
  capabilities: Record<string, any>
): Partial<AgentV3> => {
  return {
    name,
    agent_type: agentType,
    model_tier,
    project_id: '',
    state: 'CREATED' as any,
    state_changed_at: new Date(),
    tasks_completed: 0,
    success_rate: 0,
    avg_completion_time_seconds: 0,
    memory_usage_mb: 0,
    cpu_usage_percent: 0,
    capabilities,
    created_at: new Date(),
    updated_at: new Date(),
  };
};

// Agent definitions with descriptions and capabilities
const AGENT_DEFINITIONS = [
  {
    type: AgentType.CODE_IMPLEMENTER,
    name: 'Code Implementer',
    description: 'Writes high-quality, production-ready code following established patterns and conventions',
    capabilities: {
      code_generation: true,
      error_handling: true,
      documentation: true,
      type_safety: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.SYSTEM_ARCHITECT,
    name: 'System Architect',
    description: 'Designs scalable, maintainable system architectures and technical solutions',
    capabilities: {
      architecture_design: true,
      technology_selection: true,
      scalability_planning: true,
      performance_optimization: true,
    },
    defaultTier: ModelTier.OPUS,
  },
  {
    type: AgentType.CODE_QUALITY_REVIEWER,
    name: 'Code Quality Reviewer',
    description: 'Reviews code for quality, maintainability, and adherence to best practices',
    capabilities: {
      code_analysis: true,
      pattern_recognition: true,
      refactoring_suggestions: true,
      best_practice_validation: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.TEST_COVERAGE_VALIDATOR,
    name: 'Test Coverage Validator',
    description: 'Ensures comprehensive test coverage and validates test quality',
    capabilities: {
      test_generation: true,
      coverage_analysis: true,
      test_validation: true,
      integration_testing: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.SECURITY_AUDITOR,
    name: 'Security Auditor',
    description: 'Identifies security vulnerabilities and ensures secure coding practices',
    capabilities: {
      vulnerability_scanning: true,
      security_analysis: true,
      compliance_validation: true,
      threat_modeling: true,
    },
    defaultTier: ModelTier.OPUS,
  },
  {
    type: AgentType.PERFORMANCE_OPTIMIZER,
    name: 'Performance Optimizer',
    description: 'Analyzes and optimizes application performance and resource usage',
    capabilities: {
      performance_analysis: true,
      optimization_suggestions: true,
      benchmarking: true,
      resource_management: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.DEPLOYMENT_AUTOMATION,
    name: 'Deployment Automation',
    description: 'Automates deployment processes and manages CI/CD pipelines',
    capabilities: {
      ci_cd_automation: true,
      deployment_management: true,
      environment_configuration: true,
      rollback_planning: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.ANTIHALLUCINATION_VALIDATOR,
    name: 'Anti-Hallucination Validator',
    description: 'Validates AI outputs and prevents hallucinations in generated content',
    capabilities: {
      fact_validation: true,
      output_verification: true,
      source_validation: true,
      confidence_assessment: true,
    },
    defaultTier: ModelTier.HAIKU,
  },
  {
    type: AgentType.UI_UX_OPTIMIZER,
    name: 'UI/UX Optimizer',
    description: 'Optimizes user interfaces and user experience design',
    capabilities: {
      design_optimization: true,
      accessibility_validation: true,
      user_experience_analysis: true,
      responsive_design: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.DATABASE_ARCHITECT,
    name: 'Database Architect',
    description: 'Designs efficient database schemas and optimizes data access patterns',
    capabilities: {
      schema_design: true,
      query_optimization: true,
      data_modeling: true,
      performance_tuning: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.DOCUMENTATION_GENERATOR,
    name: 'Documentation Generator',
    description: 'Creates comprehensive documentation for code and systems',
    capabilities: {
      auto_documentation: true,
      api_documentation: true,
      user_guides: true,
      technical_writing: true,
    },
    defaultTier: ModelTier.HAIKU,
  },
  {
    type: AgentType.CODE_REFACTORING_OPTIMIZER,
    name: 'Code Refactoring Optimizer',
    description: 'Identifies and performs code refactoring to improve maintainability',
    capabilities: {
      refactoring_analysis: true,
      code_simplification: true,
      pattern_extraction: true,
      duplicate_removal: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.STRATEGIC_PLANNER,
    name: 'Strategic Planner',
    description: 'Creates strategic plans and roadmaps for projects and initiatives',
    capabilities: {
      strategic_planning: true,
      roadmap_creation: true,
      resource_allocation: true,
      risk_assessment: true,
    },
    defaultTier: ModelTier.OPUS,
  },
  {
    type: AgentType.API_DESIGN_ARCHITECT,
    name: 'API Design Architect',
    description: 'Designs RESTful APIs and API architecture patterns',
    capabilities: {
      api_design: true,
      endpoint_planning: true,
      documentation_generation: true,
      versioning_strategy: true,
    },
    defaultTier: ModelTier.SONNET,
  },
  {
    type: AgentType.GENERAL_PURPOSE,
    name: 'General Purpose Agent',
    description: 'Versatile agent capable of handling various tasks and workflows',
    capabilities: {
      task_execution: true,
      problem_solving: true,
      learning_adaptation: true,
      communication: true,
    },
    defaultTier: ModelTier.SONNET,
  },
];

export const AgentPalette: React.FC<AgentPaletteProps> = ({
  onAgentDragStart,
  onAgentDragEnd,
  className = '',
}) => {
  const [searchTerm, setSearchTerm] = React.useState('');
  const [selectedTier, setSelectedTier] = React.useState<ModelTier | 'all'>('all');
  const [selectedCategory, setSelectedCategory] = React.useState<string>('all');

  // Filter agents based on search and filters
  const filteredAgents = useMemo(() => {
    return AGENT_DEFINITIONS.filter(agent => {
      const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           agent.description.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesTier = selectedTier === 'all' || agent.defaultTier === selectedTier;

      return matchesSearch && matchesTier;
    });
  }, [searchTerm, selectedTier]);

  // Group agents by tier
  const agentsByTier = useMemo(() => {
    return {
      [ModelTier.OPUS]: filteredAgents.filter(agent => agent.defaultTier === ModelTier.OPUS),
      [ModelTier.SONNET]: filteredAgents.filter(agent => agent.defaultTier === ModelTier.SONNET),
      [ModelTier.HAIKU]: filteredAgents.filter(agent => agent.defaultTier === ModelTier.HAIKU),
    };
  }, [filteredAgents]);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Header */}
      <div className="space-y-3">
        <h3 className="text-white text-lg font-semibold">Agent Palette</h3>
        <p className="text-gray-400 text-sm">
          Drag agents onto the canvas to build your workflow
        </p>
      </div>

      {/* Search */}
      <div className="relative">
        <input
          type="text"
          placeholder="Search agents..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full px-3 py-2 pl-10 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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

      {/* Tier Filters */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setSelectedTier('all')}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            selectedTier === 'all'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          All Tiers
        </button>
        <button
          onClick={() => setSelectedTier(ModelTier.OPUS)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            selectedTier === ModelTier.OPUS
              ? 'bg-purple-500 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Opus
        </button>
        <button
          onClick={() => setSelectedTier(ModelTier.SONNET)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            selectedTier === ModelTier.SONNET
              ? 'bg-blue-500 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Sonnet
        </button>
        <button
          onClick={() => setSelectedTier(ModelTier.HAIKU)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            selectedTier === ModelTier.HAIKU
              ? 'bg-green-500 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          Haiku
        </button>
      </div>

      {/* Agent Lists by Tier */}
      <div className="space-y-4 max-h-96 overflow-y-auto">
        {/* Opus Agents */}
        {agentsByTier[ModelTier.OPUS].length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <h4 className="text-purple-300 text-sm font-medium">Opus Tier</h4>
              <span className="text-gray-500 text-xs">({agentsByTier[ModelTier.OPUS].length})</span>
            </div>
            {agentsByTier[ModelTier.OPUS].map((agent) => (
              <DraggableAgentItem
                key={agent.type}
                agentType={agent.type}
                modelTier={agent.defaultTier}
                name={agent.name}
                description={agent.description}
                capabilities={agent.capabilities}
                onDragStart={onAgentDragStart}
                onDragEnd={onAgentDragEnd}
              />
            ))}
          </div>
        )}

        {/* Sonnet Agents */}
        {agentsByTier[ModelTier.SONNET].length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <h4 className="text-blue-300 text-sm font-medium">Sonnet Tier</h4>
              <span className="text-gray-500 text-xs">({agentsByTier[ModelTier.SONNET].length})</span>
            </div>
            {agentsByTier[ModelTier.SONNET].map((agent) => (
              <DraggableAgentItem
                key={agent.type}
                agentType={agent.type}
                modelTier={agent.defaultTier}
                name={agent.name}
                description={agent.description}
                capabilities={agent.capabilities}
                onDragStart={onAgentDragStart}
                onDragEnd={onAgentDragEnd}
              />
            ))}
          </div>
        )}

        {/* Haiku Agents */}
        {agentsByTier[ModelTier.HAIKU].length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <h4 className="text-green-300 text-sm font-medium">Haiku Tier</h4>
              <span className="text-gray-500 text-xs">({agentsByTier[ModelTier.HAIKU].length})</span>
            </div>
            {agentsByTier[ModelTier.HAIKU].map((agent) => (
              <DraggableAgentItem
                key={agent.type}
                agentType={agent.type}
                modelTier={agent.defaultTier}
                name={agent.name}
                description={agent.description}
                capabilities={agent.capabilities}
                onDragStart={onAgentDragStart}
                onDragEnd={onAgentDragEnd}
              />
            ))}
          </div>
        )}

        {/* No results */}
        {filteredAgents.length === 0 && (
          <div className="text-center py-8">
            <div className="text-gray-500 text-sm">No agents found matching your search</div>
            <button
              onClick={() => {
                setSearchTerm('');
                setSelectedTier('all');
              }}
              className="mt-2 text-blue-400 text-sm hover:text-blue-300"
            >
              Clear filters
            </button>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="mt-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
        <h4 className="text-white text-sm font-medium mb-2">How to use:</h4>
        <ul className="text-gray-400 text-xs space-y-1">
          <li>• Search and filter agents by type or tier</li>
          <li>• Drag agents onto the canvas</li>
          <li>• Connect agents to create workflow</li>
          <li>• Configure properties in the editor</li>
        </ul>
      </div>
    </div>
  );
};

// Wrapper component that provides DnD context
export const AgentPaletteWithProvider: React.FC<AgentPaletteProps> = (props) => (
  <DndProvider backend={HTML5Backend}>
    <AgentPalette {...props} />
  </DndProvider>
);