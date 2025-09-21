/**
 * Tests for Workflow Knowledge Integration Service
 */

import { workflowKnowledgeService } from '../../services/workflowKnowledgeService';
import type {
  WorkflowKnowledgeSession,
  WorkflowInsight,
  ContextualKnowledge,
  WorkflowTemplate,
  PerformanceInsight,
  TemplateRecommendation
} from '../../services/workflowKnowledgeService';

// Mock fetch API
global.fetch = jest.fn() as jest.MockedFunction<typeof fetch>;

describe('WorkflowKnowledgeService', () => {
  const mockWorkflowId = 'test-workflow-id';
  const mockProjectId = 'test-project-id';
  const mockSessionId = 'test-session-id';

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('startKnowledgeSession', () => {
    it('should start a knowledge session successfully', async () => {
      const mockSession: WorkflowKnowledgeSession = {
        session_id: mockSessionId,
        workflow_id: mockWorkflowId,
        project_id: mockProjectId,
        capture_config: {
          auto_capture: true,
          capture_insights: true,
          capture_patterns: true,
          capture_errors: true,
          capture_successes: true,
          real_time_analysis: true,
          embedding_generation: true
        },
        context_tags: ['test', 'workflow'],
        started_at: new Date().toISOString(),
        status: 'active'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockSession)
      });

      const result = await workflowKnowledgeService.startKnowledgeSession(
        mockWorkflowId,
        mockProjectId,
        { auto_capture: true },
        ['test', 'workflow']
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/start-session'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            workflow_id: mockWorkflowId,
            project_id: mockProjectId,
            capture_config: {
              auto_capture: true,
              capture_insights: true,
              capture_patterns: true,
              capture_errors: true,
              capture_successes: true,
              real_time_analysis: true,
              embedding_generation: true
            },
            context_tags: ['test', 'workflow']
          })
        })
      );

      expect(result).toEqual(mockSession);
    });

    it('should handle API errors', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: jest.fn().mockResolvedValue({ error: 'Failed to start session' })
      });

      await expect(
        workflowKnowledgeService.startKnowledgeSession(mockWorkflowId, mockProjectId)
      ).rejects.toThrow('Failed to start session');
    });
  });

  describe('captureInsight', () => {
    it('should capture an insight successfully', async () => {
      const mockInsight: WorkflowInsight = {
        insight_id: 'test-insight-id',
        session_id: mockSessionId,
        insight_type: 'performance_optimization',
        insight_data: { observation: 'Step completed efficiently' },
        importance_score: 0.8,
        tags: ['performance'],
        captured_at: new Date().toISOString()
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockInsight)
      });

      const result = await workflowKnowledgeService.captureInsight(
        mockSessionId,
        'performance_optimization',
        { observation: 'Step completed efficiently' },
        { importanceScore: 0.8, tags: ['performance'] }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/capture-insight'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            session_id: mockSessionId,
            insight_type: 'performance_optimization',
            insight_data: { observation: 'Step completed efficiently' },
            importance_score: 0.8,
            tags: ['performance']
          })
        })
      );

      expect(result).toEqual(mockInsight);
    });
  });

  describe('getContextualKnowledge', () => {
    it('should retrieve contextual knowledge successfully', async () => {
      const mockKnowledge: ContextualKnowledge[] = [
        {
          knowledge_id: 'test-knowledge-id',
          content: 'Optimize batch processing for better performance',
          source: 'best-practices',
          relevance_score: 0.9,
          knowledge_type: 'best_practice',
          metadata: {},
          created_at: new Date().toISOString()
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockKnowledge)
      });

      const result = await workflowKnowledgeService.getContextualKnowledge(
        mockSessionId,
        'How to optimize batch processing?',
        'execution_context',
        { maxResults: 5 }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining(`/api/workflow-knowledge/contextual/${mockSessionId}`),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'How to optimize batch processing?',
            context_type: 'execution_context',
            max_results: 5,
            similarity_threshold: 0.7
          })
        })
      );

      expect(result).toEqual(mockKnowledge);
    });
  });

  describe('endKnowledgeSession', () => {
    it('should end knowledge session successfully', async () => {
      const mockSummary = {
        session_id: mockSessionId,
        session_summary: { total_insights: 5, patterns_found: 2 },
        extracted_patterns: [{ type: 'performance_optimization', frequency: 3 }],
        total_insights_captured: 5,
        ended_at: new Date().toISOString()
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockSummary)
      });

      const result = await workflowKnowledgeService.endKnowledgeSession(
        mockSessionId,
        { generateSummary: true, extractPatterns: true }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/end-session'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            session_id: mockSessionId,
            generate_summary: true,
            extract_patterns: true
          })
        })
      );

      expect(result).toEqual(mockSummary);
    });
  });

  describe('storeWorkflowTemplate', () => {
    it('should store workflow as template successfully', async () => {
      const mockResponse = {
        template_id: 'test-template-id',
        message: 'Template stored successfully'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockResponse)
      });

      const result = await workflowKnowledgeService.storeWorkflowTemplate(
        mockWorkflowId,
        'Test Template',
        'A template for testing',
        {
          useCases: ['testing', 'development'],
          bestPractices: ['modular', 'test-driven'],
          tags: ['test', 'template']
        }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/store-template'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            workflow_id: mockWorkflowId,
            template_name: 'Test Template',
            template_description: 'A template for testing',
            use_cases: ['testing', 'development'],
            best_practices: ['modular', 'test-driven'],
            tags: ['test', 'template'],
            is_public: false
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('searchWorkflowTemplates', () => {
    it('should search workflow templates successfully', async () => {
      const mockTemplates: WorkflowTemplate[] = [
        {
          template_id: 'test-template-1',
          name: 'Data Processing Template',
          description: 'Template for efficient data processing',
          category: 'data_processing',
          flow_data: { nodes: [], edges: [] },
          use_cases: ['batch-processing', 'etl'],
          best_practices: ['parallel-execution'],
          common_patterns: ['map-reduce'],
          tags: ['data', 'processing', 'batch'],
          metadata: {
            category: 'data_processing',
            complexity_score: 0.7,
            version: '1.0.0',
            created_by: 'test-user',
            usage_count: 10,
            success_rate: 0.95,
            average_duration: 30000,
            rating: 4.5
          },
          is_public: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockTemplates)
      });

      const result = await workflowKnowledgeService.searchWorkflowTemplates(
        'data processing',
        { projectId: mockProjectId, limit: 10 }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/templates/search?query=data%20processing&project_id=test-project-id&limit=10'),
        expect.objectContaining({
          method: 'GET'
        })
      );

      expect(result).toEqual(mockTemplates);
    });
  });

  describe('getTemplateRecommendations', () => {
    it('should get template recommendations successfully', async () => {
      const mockRecommendations: TemplateRecommendation[] = [
        {
          template_id: 'recommended-template-1',
          name: 'Recommended Workflow Template',
          description: 'AI-recommended template for your project',
          match_score: 0.85,
          reasons: ['Matches project complexity', 'Similar to successful workflows'],
          category: 'general',
          complexity_score: 0.6,
          estimated_duration: 25000
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockRecommendations)
      });

      const result = await workflowKnowledgeService.getTemplateRecommendations(
        mockProjectId,
        { complexityPreference: 'medium', maxRecommendations: 5 }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/templates/recommendations?project_id=test-project-id&complexity_preference=medium&max_recommendations=5'),
        expect.objectContaining({
          method: 'GET'
        })
      );

      expect(result).toEqual(mockRecommendations);
    });
  });

  describe('applyTemplate', () => {
    it('should apply template successfully', async () => {
      const mockResponse = {
        workflow_id: 'new-workflow-id',
        workflow_name: 'Template-based Workflow',
        message: 'Template applied successfully'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockResponse)
      });

      const result = await workflowKnowledgeService.applyTemplate(
        'test-template-id',
        mockProjectId,
        {
          workflowName: 'Custom Workflow Name',
          workflowDescription: 'Custom workflow description',
          customParameters: { param1: 'value1' },
          overrideTags: ['custom', 'tag']
        }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/templates/apply'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            template_id: 'test-template-id',
            project_id: mockProjectId,
            workflow_name: 'Custom Workflow Name',
            workflow_description: 'Custom workflow description',
            custom_parameters: { param1: 'value1' },
            override_tags: ['custom', 'tag']
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('capturePerformanceMetrics', () => {
    it('should capture performance metrics successfully', async () => {
      const mockResponse = {
        metrics_id: 'test-metrics-id',
        message: 'Performance metrics captured successfully'
      };

      const metrics = {
        stepMetrics: { step1: { duration: 1000, status: 'completed' } },
        overallMetrics: { total_duration: 2000, success_rate: 1.0 },
        resourceUsage: { memory: 512, cpu: 0.7 },
        bottlenecksIdentified: ['network-io'],
        optimizationOpportunities: ['parallel-execution']
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockResponse)
      });

      const result = await workflowKnowledgeService.capturePerformanceMetrics(
        'test-execution-id',
        metrics
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/performance/capture'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            execution_id: 'test-execution-id',
            step_metrics: metrics.stepMetrics,
            overall_metrics: metrics.overallMetrics,
            resource_usage: metrics.resourceUsage,
            bottlenecks_identified: metrics.bottlenecksIdentified,
            optimization_opportunities: metrics.optimizationOpportunities
          })
        })
      );

      expect(result).toEqual(mockResponse);
    });
  });

  describe('generatePerformanceInsights', () => {
    it('should generate performance insights successfully', async () => {
      const mockInsights: PerformanceInsight[] = [
        {
          insight_id: 'performance-insight-1',
          insight_type: 'efficiency',
          description: 'Workflow efficiency improved by 15%',
          confidence: 0.85,
          impact: 'medium',
          actionable: true,
          recommendation: 'Continue current optimization strategy',
          expected_improvement: '10-20% efficiency gain',
          metadata: {}
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockInsights)
      });

      const result = await workflowKnowledgeService.generatePerformanceInsights(
        mockWorkflowId,
        {
          insightTypes: ['efficiency', 'cost'],
          minConfidence: 0.7
        }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/performance/insights'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            workflow_id: mockWorkflowId,
            insight_types: ['efficiency', 'cost'],
            min_confidence: 0.7
          })
        })
      );

      expect(result).toEqual(mockInsights);
    });
  });

  describe('getExecutionContext', () => {
    it('should get execution context successfully', async () => {
      const mockContext = {
        execution_context: {
          execution_id: 'test-execution-id',
          workflow_id: mockWorkflowId,
          status: 'running',
          progress: 0.5,
          current_step_id: 'step-1'
        },
        relevant_knowledge: [
          {
            knowledge_id: 'context-knowledge-1',
            content: 'Relevant context information',
            source: 'execution-history',
            relevance_score: 0.8,
            knowledge_type: 'context',
            metadata: {},
            created_at: new Date().toISOString()
          }
        ]
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockContext)
      });

      const result = await workflowKnowledgeService.getExecutionContext(
        'test-execution-id',
        { includeKnowledge: true, maxContextItems: 15 }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/execution-context?execution_id=test-execution-id&include_knowledge=true&max_context_items=15'),
        expect.objectContaining({
          method: 'GET'
        })
      );

      expect(result).toEqual(mockContext);
    });
  });

  describe('trackPerformanceImprovements', () => {
    it('should track performance improvements successfully', async () => {
      const mockResult = {
        improvements: {
          duration: { baseline: 3000, current: 2500, improvement: 16.7 },
          cost: { baseline: 0.50, current: 0.40, improvement: 20.0 },
          success_rate: { baseline: 0.85, current: 0.92, improvement: 8.2 }
        },
        comparison_summary: 'Overall performance improved by 15% across key metrics'
      };

      const baselinePeriod = {
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-15T00:00:00Z'
      };

      const improvementPeriod = {
        start: '2024-01-16T00:00:00Z',
        end: '2024-01-30T00:00:00Z'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue(mockResult)
      });

      const result = await workflowKnowledgeService.trackPerformanceImprovements(
        mockWorkflowId,
        baselinePeriod,
        improvementPeriod,
        ['duration', 'cost', 'success_rate']
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/workflow-knowledge/performance/improvements'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            workflow_id: mockWorkflowId,
            baseline_period: baselinePeriod,
            improvement_period: improvementPeriod,
            metrics_to_track: ['duration', 'cost', 'success_rate']
          })
        })
      );

      expect(result).toEqual(mockResult);
    });
  });

  describe('error handling', () => {
    it('should handle network errors', async () => {
      (fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      await expect(
        workflowKnowledgeService.startKnowledgeSession(mockWorkflowId, mockProjectId)
      ).rejects.toThrow('Network error');
    });

    it('should handle HTTP errors with custom error messages', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: jest.fn().mockResolvedValue({ error: 'Custom error message' })
      });

      await expect(
        workflowKnowledgeService.startKnowledgeSession(mockWorkflowId, mockProjectId)
      ).rejects.toThrow('Custom error message');
    });

    it('should handle malformed JSON responses', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockRejectedValue(new Error('Invalid JSON'))
      });

      await expect(
        workflowKnowledgeService.startKnowledgeSession(mockWorkflowId, mockProjectId)
      ).rejects.toThrow('Invalid JSON');
    });
  });

  describe('request parameters', () => {
    it('should use default values when optional parameters are not provided', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({})
      });

      await workflowKnowledgeService.getContextualKnowledge(mockSessionId, 'test query');

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'test query',
            context_type: 'execution_context',
            max_results: 10,
            similarity_threshold: 0.7
          })
        })
      );
    });

    it('should merge provided options with defaults', async () => {
      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: jest.fn().mockResolvedValue({})
      });

      await workflowKnowledgeService.getContextualKnowledge(
        mockSessionId,
        'test query',
        'step_context',
        { maxResults: 20, similarityThreshold: 0.5 }
      );

      expect(fetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            query: 'test query',
            context_type: 'step_context',
            max_results: 20,
            similarity_threshold: 0.5
          })
        })
      );
    });
  });
});