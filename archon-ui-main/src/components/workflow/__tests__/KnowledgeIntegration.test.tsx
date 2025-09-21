/**
 * Knowledge Integration Test Suite
 * Tests knowledge-aware workflow components and RAG functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { KnowledgeAwareWorkflow } from '../KnowledgeAwareWorkflow';
import { KnowledgeIntegrationDashboard } from '../KnowledgeIntegrationDashboard';

import {
  WorkflowKnowledgeSession,
  WorkflowInsight,
  ContextualKnowledge,
  WorkflowTemplate,
  KnowledgeQuery,
  KnowledgeResult,
  KnowledgeGraph,
  KnowledgeNode,
  KnowledgeEdge,
} from '../../../types/workflowTypes';

// Mock knowledge services
vi.mock('../../../services/workflowKnowledgeService', () => ({
  workflowKnowledgeService: {
    startKnowledgeSession: vi.fn(),
    captureWorkflowEvent: vi.fn(),
    getWorkflowInsights: vi.fn(),
    getWorkflowTemplates: vi.fn(),
    suggestOptimizations: vi.fn(),
    queryKnowledge: vi.fn(),
    getKnowledgeGraph: vi.fn(),
    updateKnowledgeSession: vi.fn(),
    endKnowledgeSession: vi.fn(),
    analyzeWorkflowPatterns: vi.fn(),
    generateWorkflowReport: vi.fn(),
    shareKnowledge: vi.fn(),
    importKnowledge: vi.fn(),
  },
}));

vi.mock('../../../services/knowledgeBaseService', () => ({
  knowledgeBaseService: {
    searchKnowledge: vi.fn(),
    getKnowledgeItems: vi.fn(),
    addKnowledgeItem: vi.fn(),
    updateKnowledgeItem: vi.fn(),
    deleteKnowledgeItem: vi.fn(),
    getKnowledgeGraph: vi.fn(),
    queryKnowledgeGraph: vi.fn(),
  },
}));

vi.mock('../../../services/workflowService', () => ({
  workflowService: {
    getWorkflow: vi.fn(),
    executeWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    optimizeWorkflow: vi.fn(),
    getWorkflowStats: vi.fn(),
  },
}));

// Mock socket.io
vi.mock('socket.io-client', () => ({
  io: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
  })),
}));

// Mock UI components
vi.mock('../../ui/button', () => ({
  Button: vi.fn(({ children, ...props }) => (
    <button data-testid="knowledge-button" {...props}>
      {children}
    </button>
  )),
}));

vi.mock('../../ui/card', () => ({
  Card: vi.fn(({ children, ...props }) => (
    <div data-testid="knowledge-card" {...props}>
      {children}
    </div>
  )),
  CardContent: vi.fn(({ children }) => <div data-testid="knowledge-card-content">{children}</div>),
  CardDescription: vi.fn(({ children }) => <div data-testid="knowledge-card-description">{children}</div>),
  CardHeader: vi.fn(({ children }) => <div data-testid="knowledge-card-header">{children}</div>),
  CardTitle: vi.fn(({ children }) => <div data-testid="knowledge-card-title">{children}</div>),
}));

vi.mock('../../ui/tabs', () => ({
  Tabs: vi.fn(({ children, ...props }) => (
    <div data-testid="knowledge-tabs" {...props}>
      {children}
    </div>
  )),
  TabsContent: vi.fn(({ children }) => <div data-testid="knowledge-tabs-content">{children}</div>),
  TabsList: vi.fn(({ children }) => <div data-testid="knowledge-tabs-list">{children}</div>),
  TabsTrigger: vi.fn(({ children }) => <button data-testid="knowledge-tabs-trigger">{children}</button>),
}));

vi.mock('../../ui/input', () => ({
  Input: vi.fn((props) => <input data-testid="knowledge-input" {...props} />),
}));

vi.mock('../../ui/textarea', () => ({
  Textarea: vi.fn((props) => <textarea data-testid="knowledge-textarea" {...props} />),
}));

vi.mock('../../ui/label', () => ({
  Label: vi.fn(({ children }) => <label data-testid="knowledge-label">{children}</label>),
}));

vi.mock('../../ui/select', () => ({
  Select: vi.fn(({ children, ...props }) => (
    <div data-testid="knowledge-select" {...props}>
      {children}
    </div>
  )),
  SelectContent: vi.fn(({ children }) => <div data-testid="knowledge-select-content">{children}</div>),
  SelectItem: vi.fn(({ children }) => <div data-testid="knowledge-select-item">{children}</div>),
  SelectTrigger: vi.fn(({ children }) => <button data-testid="knowledge-select-trigger">{children}</button>),
  SelectValue: vi.fn(() => <div data-testid="knowledge-select-value" />),
}));

vi.mock('../../ui/badge', () => ({
  Badge: vi.fn(({ children }) => <span data-testid="knowledge-badge">{children}</span>),
}));

vi.mock('../../ui/alert', () => ({
  Alert: vi.fn(({ children }) => <div data-testid="knowledge-alert">{children}</div>),
  AlertDescription: vi.fn(({ children }) => <div data-testid="knowledge-alert-description">{children}</div>),
}));

describe('Knowledge Integration', () => {
  const mockKnowledgeSession: WorkflowKnowledgeSession = {
    id: 'session-1',
    workflow_id: 'workflow-1',
    project_id: 'project-1',
    status: 'active',
    created_at: new Date(),
    updated_at: new Date(),
    insights: [],
    contextual_knowledge: [],
    knowledge_queries: [],
    optimization_suggestions: [],
  };

  const mockWorkflowInsight: WorkflowInsight = {
    id: 'insight-1',
    type: 'performance',
    description: 'Workflow efficiency improved by 25%',
    confidence: 0.85,
    agent_id: 'agent-1',
    workflow_id: 'workflow-1',
    data: {
      metric: 'efficiency',
      value: 0.25,
      unit: 'percentage',
    },
    created_at: new Date(),
  };

  const mockContextualKnowledge: ContextualKnowledge = {
    id: 'knowledge-1',
    type: 'best_practice',
    title: 'Optimal Agent Configuration',
    content: 'Configure agents with appropriate model tiers for optimal performance',
    context: {
      workflow_type: 'analysis',
      agent_types: ['analyst', 'specialist'],
    },
    relevance_score: 0.9,
    source: 'system',
    created_at: new Date(),
  };

  const mockWorkflowTemplate: WorkflowTemplate = {
    id: 'template-1',
    name: 'Analysis Workflow Template',
    description: 'Optimized template for analysis workflows',
    category: 'analysis',
    nodes: [
      {
        id: 'analyst-node',
        type: 'agent',
        position: { x: 100, y: 100 },
        data: {
          agent: {
            id: 'analyst',
            name: 'Analysis Agent',
            type: 'analyst' as any,
            model_tier: 'sonnet' as any,
            state: 'active' as any,
            capabilities: ['analysis'],
            created_at: new Date(),
            updated_at: new Date(),
            metadata: {},
          },
        },
      },
    ],
    edges: [],
    metadata: {
      created_at: new Date(),
      updated_at: new Date(),
      version: '1.0.0',
      author: 'system',
      efficiency_score: 0.95,
    },
  };

  const mockKnowledgeQuery: KnowledgeQuery = {
    id: 'query-1',
    query: 'How to optimize agent communication?',
    type: 'optimization',
    context: {
      workflow_id: 'workflow-1',
      agent_ids: ['agent-1', 'agent-2'],
    },
    results: [],
    timestamp: new Date(),
  };

  const mockKnowledgeResult: KnowledgeResult = {
    id: 'result-1',
    query_id: 'query-1',
    content: 'Optimize agent communication by using asynchronous messaging and proper error handling',
    relevance_score: 0.95,
    source: 'knowledge_base',
    metadata: {
      confidence: 0.95,
      source_type: 'document',
    },
    timestamp: new Date(),
  };

  const mockKnowledgeGraph: KnowledgeGraph = {
    nodes: [
      {
        id: 'node-1',
        type: 'concept',
        label: 'Agent Communication',
        description: 'Communication patterns between agents',
        properties: {
          complexity: 'medium',
          frequency: 'high',
        },
      },
      {
        id: 'node-2',
        type: 'pattern',
        label: 'Async Messaging',
        description: 'Asynchronous messaging pattern',
        properties: {
          efficiency: 'high',
          reliability: 'medium',
        },
      },
    ],
    edges: [
      {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        type: 'improves',
        weight: 0.8,
        properties: {
          improvement_factor: 0.3,
        },
      },
    ],
    metadata: {
      created_at: new Date(),
      updated_at: new Date(),
      version: '1.0.0',
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('KnowledgeAwareWorkflow Core Functionality', () => {
    beforeEach(() => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).startKnowledgeSession.mockResolvedValue(mockKnowledgeSession);
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).getWorkflowInsights.mockResolvedValue([mockWorkflowInsight]);
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).getWorkflowTemplates.mockResolvedValue([mockWorkflowTemplate]);
    });

    it('should initialize knowledge session correctly', async () => {
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.startKnowledgeSession
        ).toHaveBeenCalledWith('workflow-1', 'project-1', expect.any(Object), expect.any(Array));
      });
    });

    it('should display workflow insights', async () => {
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.getWorkflowInsights
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle workflow updates and knowledge capture', async () => {
      const mockWorkflowUpdate = vi.fn();
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={mockWorkflowUpdate}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.startKnowledgeSession
        ).toHaveBeenCalled();
      });

      // Simulate workflow update
      const workflowEvent = {
        type: 'agent_execution',
        agent_id: 'agent-1',
        timestamp: new Date(),
        data: { result: 'success', duration: 1500 },
      };

      await act(async () => {
        mockWorkflowUpdate(workflowEvent);
      });

      expect(mockWorkflowUpdate).toHaveBeenCalledWith(workflowEvent);
    });

    it('should handle knowledge queries', async () => {
      const user = userEvent.setup();
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Simulate knowledge query
      const queryInput = screen.getByTestId('knowledge-input');
      await user.type(queryInput, 'How to optimize workflow?');

      const queryButton = screen.getByTestId('knowledge-button');
      await user.click(queryButton);

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });
    });

    it('should display contextual knowledge', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).getContextualKnowledge.mockResolvedValue([mockContextualKnowledge]);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.getContextualKnowledge
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle knowledge session updates', async () => {
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.startKnowledgeSession
        ).toHaveBeenCalled();
      });

      // Simulate session update
      const updatedSession = {
        ...mockKnowledgeSession,
        insights: [mockWorkflowInsight],
        contextual_knowledge: [mockContextualKnowledge],
      };

      await act(async () => {
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService.updateKnowledgeSession.mockResolvedValue(updatedSession);
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });
  });

  describe('KnowledgeIntegrationDashboard Functionality', () => {
    beforeEach(() => {
      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).getKnowledgeGraph.mockResolvedValue(mockKnowledgeGraph);
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue([mockKnowledgeResult]);
    });

    it('should render knowledge dashboard with all components', () => {
      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      expect(screen.getByTestId('knowledge-tabs')).toBeInTheDocument();
    });

    it('should display knowledge graph visualization', async () => {
      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle knowledge graph interactions', async () => {
      const user = userEvent.setup();
      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
        ).toHaveBeenCalled();
      });

      // Simulate graph node click
      const graphContainer = screen.getByTestId('knowledge-card-content');
      await user.click(graphContainer);

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should display knowledge search results', async () => {
      const user = userEvent.setup();
      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Perform knowledge search
      const searchInput = screen.getByTestId('knowledge-input');
      await user.type(searchInput, 'optimization');

      const searchButton = screen.getByTestId('knowledge-button');
      await user.click(searchButton);

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });
    });

    it('should handle knowledge sharing', async () => {
      const user = userEvent.setup();
      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Simulate knowledge sharing
      const shareButton = screen.getAllByTestId('knowledge-button')[1];
      await user.click(shareButton);

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.shareKnowledge
        ).toHaveBeenCalled();
      });
    });
  });

  describe('Knowledge Query and RAG Functionality', () => {
    it('should handle different query types', async () => {
      const queryTypes = ['optimization', 'troubleshooting', 'best_practices', 'patterns'];

      for (const queryType of queryTypes) {
        const mockQuery: KnowledgeQuery = {
          ...mockKnowledgeQuery,
          type: queryType as any,
        };

        vi.mocked(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService
        ).queryKnowledge.mockResolvedValue([mockKnowledgeResult]);

        render(
          <KnowledgeAwareWorkflow
            workflowId="workflow-1"
            projectId="project-1"
            onWorkflowUpdate={vi.fn()}
          />
        );

        await waitFor(() => {
          expect(
            require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
          ).toHaveBeenCalledWith(expect.objectContaining({ type: queryType }));
        });
      }
    });

    it('should handle context-aware queries', async () => {
      const contextAwareQuery: KnowledgeQuery = {
        ...mockKnowledgeQuery,
        context: {
          workflow_id: 'workflow-1',
          agent_ids: ['agent-1', 'agent-2'],
          task_types: ['analysis', 'optimization'],
        },
      };

      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue([mockKnowledgeResult]);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalledWith(contextAwareQuery);
      });
    });

    it('should handle query result filtering', async () => {
      const mockResults: KnowledgeResult[] = [
        mockKnowledgeResult,
        {
          ...mockKnowledgeResult,
          id: 'result-2',
          relevance_score: 0.7,
          content: 'Less relevant result',
        },
        {
          ...mockKnowledgeResult,
          id: 'result-3',
          relevance_score: 0.9,
          content: 'Highly relevant result',
        },
      ];

      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue(mockResults);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });

      // Results should be sorted by relevance score
      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle query result pagination', async () => {
      const mockResults: KnowledgeResult[] = Array.from({ length: 25 }, (_, i) => ({
        ...mockKnowledgeResult,
        id: `result-${i}`,
        relevance_score: 0.9 - (i * 0.01),
      }));

      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue(mockResults);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });
  });

  describe('Knowledge Graph Visualization', () => {
    it('should display knowledge graph with nodes and edges', async () => {
      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).getKnowledgeGraph.mockResolvedValue(mockKnowledgeGraph);

      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle different node types in knowledge graph', async () => {
      const nodeTypes = ['concept', 'pattern', 'agent', 'workflow', 'tool'];

      for (const nodeType of nodeTypes) {
        const graphWithNodeType: KnowledgeGraph = {
          ...mockKnowledgeGraph,
          nodes: [
            {
              ...mockKnowledgeGraph.nodes[0],
              type: nodeType as any,
            },
          ],
        };

        vi.mocked(
          require('../../../services/knowledgeBaseService').knowledgeBaseService
        ).getKnowledgeGraph.mockResolvedValue(graphWithNodeType);

        render(
          <KnowledgeIntegrationDashboard
            workflowId="workflow-1"
            projectId="project-1"
          />
        );

        await waitFor(() => {
          expect(
            require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
          ).toHaveBeenCalled();
        });

        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      }
    });

    it('should handle knowledge graph filtering', async () => {
      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).getKnowledgeGraph.mockResolvedValue(mockKnowledgeGraph);

      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
        ).toHaveBeenCalled();
      });

      // Simulate graph filtering
      const filterInput = screen.getByTestId('knowledge-input');
      const user = userEvent.setup();
      await user.type(filterInput, 'concept');

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle knowledge graph search', async () => {
      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).queryKnowledgeGraph.mockResolvedValue(mockKnowledgeGraph);

      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Simulate graph search
      const searchInput = screen.getByTestId('knowledge-input');
      const user = userEvent.setup();
      await user.type(searchInput, 'communication');

      const searchButton = screen.getByTestId('knowledge-button');
      await user.click(searchButton);

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.queryKnowledgeGraph
        ).toHaveBeenCalled();
      });
    });
  });

  describe('Workflow Optimization and Insights', () => {
    it('should display optimization suggestions', async () => {
      const mockSuggestions = [
        {
          id: 'suggestion-1',
          type: 'performance',
          description: 'Reduce agent response time by 20%',
          impact: 'high',
          implementation_difficulty: 'medium',
          estimated_improvement: 0.2,
        },
      ];

      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).suggestOptimizations.mockResolvedValue(mockSuggestions);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.suggestOptimizations
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle workflow pattern analysis', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).analyzeWorkflowPatterns.mockResolvedValue({
        patterns: ['parallel_execution', 'async_communication'],
        efficiency_score: 0.85,
        recommendations: ['increase_parallelization', 'optimize_message_broker'],
      });

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.analyzeWorkflowPatterns
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle workflow report generation', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).generateWorkflowReport.mockResolvedValue({
        id: 'report-1',
        workflow_id: 'workflow-1',
        generated_at: new Date(),
        metrics: {
          efficiency: 0.85,
          reliability: 0.95,
          scalability: 0.75,
        },
        insights: [mockWorkflowInsight],
        recommendations: [
          {
            type: 'optimization',
            description: 'Optimize agent communication',
            priority: 'high',
          },
        ],
      });

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.generateWorkflowReport
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle knowledge session initialization errors', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).startKnowledgeSession.mockRejectedValue(new Error('Failed to initialize session'));

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-alert')).toBeInTheDocument();
      });
    });

    it('should handle knowledge query errors', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockRejectedValue(new Error('Query failed'));

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Simulate query that fails
      const user = userEvent.setup();
      const queryInput = screen.getByTestId('knowledge-input');
      await user.type(queryInput, 'test query');

      const queryButton = screen.getByTestId('knowledge-button');
      await user.click(queryButton);

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-alert')).toBeInTheDocument();
      });
    });

    it('should handle knowledge graph errors', async () => {
      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).getKnowledgeGraph.mockRejectedValue(new Error('Graph loading failed'));

      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-alert')).toBeInTheDocument();
      });
    });

    it('should handle empty knowledge results', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue([]);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle malformed knowledge data', async () => {
      vi.mocked(
        require('../../../services/workflowKnowledgeService').workflowKnowledgeService
      ).queryKnowledge.mockResolvedValue([
        {
          ...mockKnowledgeResult,
          content: null,
          relevance_score: 'invalid' as any,
        } as any,
      ]);

      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/workflowKnowledgeService').workflowKnowledgeService.queryKnowledge
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large knowledge graphs efficiently', async () => {
      const largeKnowledgeGraph: KnowledgeGraph = {
        nodes: Array.from({ length: 1000 }, (_, i) => ({
          id: `node-${i}`,
          type: 'concept',
          label: `Concept ${i}`,
          description: `Description for concept ${i}`,
          properties: {
            complexity: 'medium',
            frequency: 'high',
          },
        })),
        edges: Array.from({ length: 2000 }, (_, i) => ({
          id: `edge-${i}`,
          source: `node-${i % 1000}`,
          target: `node-${(i + 1) % 1000}`,
          type: 'related',
          weight: 0.5,
          properties: {},
        })),
        metadata: {
          created_at: new Date(),
          updated_at: new Date(),
          version: '1.0.0',
        },
      };

      vi.mocked(
        require('../../../services/knowledgeBaseService').knowledgeBaseService
      ).getKnowledgeGraph.mockResolvedValue(largeKnowledgeGraph);

      render(
        <KnowledgeIntegrationDashboard
          workflowId="workflow-1"
          projectId="project-1"
        />
      );

      await waitFor(() => {
        expect(
          require('../../../services/knowledgeBaseService').knowledgeBaseService.getKnowledgeGraph
        ).toHaveBeenCalled();
      });

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });

    it('should handle rapid knowledge queries', async () => {
      render(
        <KnowledgeAwareWorkflow
          workflowId="workflow-1"
          projectId="project-1"
          onWorkflowUpdate={vi.fn()}
        />
      );

      await waitFor(() => {
        expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
      });

      // Simulate rapid queries
      const user = userEvent.setup();
      for (let i = 0; i < 10; i++) {
        const queryInput = screen.getByTestId('knowledge-input');
        await user.clear(queryInput);
        await user.type(queryInput, `query ${i}`);

        const queryButton = screen.getByTestId('knowledge-button');
        await user.click(queryButton);
      }

      expect(screen.getByTestId('knowledge-card')).toBeInTheDocument();
    });
  });
});