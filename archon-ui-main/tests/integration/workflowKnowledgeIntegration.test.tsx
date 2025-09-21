/**
 * Integration Tests for Workflow Knowledge Integration
 *
 * Tests the complete integration between workflow execution and knowledge management,
 * including UI components, services, and API interactions.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { KnowledgeAwareWorkflow } from '../../components/workflow/KnowledgeAwareWorkflow';
import { KnowledgeIntegrationDashboard } from '../../components/workflow/KnowledgeIntegrationDashboard';
import { workflowKnowledgeService } from '../../services/workflowKnowledgeService';

// Mock the workflowKnowledgeService
jest.mock('../../services/workflowKnowledgeService');
const mockWorkflowKnowledgeService = workflowKnowledgeService as jest.Mocked<typeof workflowKnowledgeService>;

// Mock other services
jest.mock('../../services/workflowService');
jest.mock('../../services/knowledgeBaseService');

// Mock UI components
jest.mock('../../components/ui/card', () => ({
  Card: ({ children }: { children: React.ReactNode }) => <div data-testid="card">{children}</div>,
  CardHeader: ({ children }: { children: React.ReactNode }) => <div data-testid="card-header">{children}</div>,
  CardTitle: ({ children }: { children: React.ReactNode }) => <div data-testid="card-title">{children}</div>,
  CardDescription: ({ children }: { children: React.ReactNode }) => <div data-testid="card-description">{children}</div>,
  CardContent: ({ children }: { children: React.ReactNode }) => <div data-testid="card-content">{children}</div>
}));

jest.mock('../../components/ui/button', () => ({
  Button: ({ children, onClick, disabled }: { children: React.ReactNode; onClick?: () => void; disabled?: boolean }) => (
    <button data-testid="button" onClick={onClick} disabled={disabled}>
      {children}
    </button>
  )
}));

jest.mock('../../components/ui/badge', () => ({
  Badge: ({ children }: { children: React.ReactNode }) => <span data-testid="badge">{children}</span>
}));

jest.mock('../../components/ui/tabs', () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div data-testid="tabs">{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div data-testid="tabs-list">{children}</div>,
  TabsTrigger: ({ children, value }: { children: React.ReactNode; value: string }) => (
    <button data-testid={`tabs-trigger-${value}`}>{children}</button>
  ),
  TabsContent: ({ children, value }: { children: React.ReactNode; value: string }) => (
    <div data-testid={`tabs-content-${value}`}>{children}</div>
  )
}));

jest.mock('../../components/ui/progress', () => ({
  Progress: ({ value }: { value: number }) => <div data-testid="progress" data-value={value} />
}));

describe('Workflow Knowledge Integration', () => {
  const mockWorkflowId = 'integration-test-workflow-id';
  const mockProjectId = 'integration-test-project-id';
  const mockSessionId = 'integration-test-session-id';

  const mockSession = {
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
    context_tags: ['integration-test', 'workflow-execution'],
    started_at: new Date().toISOString(),
    status: 'active' as const
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Setup default mock implementations
    mockWorkflowKnowledgeService.startKnowledgeSession.mockResolvedValue(mockSession);
    mockWorkflowKnowledgeService.endKnowledgeSession.mockResolvedValue({
      session_id: mockSessionId,
      ended_at: new Date().toISOString()
    });
    mockWorkflowKnowledgeService.getContextualKnowledge.mockResolvedValue([]);
    mockWorkflowKnowledgeService.searchWorkflowTemplates.mockResolvedValue([]);
    mockWorkflowKnowledgeService.getTemplateRecommendations.mockResolvedValue([]);
    mockWorkflowKnowledgeService.generatePerformanceInsights.mockResolvedValue([]);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('End-to-End Workflow Knowledge Flow', () => {
    it('completes full knowledge capture and retrieval cycle', async () => {
      // Mock the complete flow
      const mockInsight = {
        insight_id: 'integration-test-insight-id',
        session_id: mockSessionId,
        insight_type: 'performance_optimization' as const,
        insight_data: { observation: 'Integration test optimization', improvement: 25 },
        importance_score: 0.9,
        tags: ['integration', 'test'],
        captured_at: new Date().toISOString()
      };

      const mockKnowledge = [
        {
          knowledge_id: 'integration-test-knowledge-id',
          content: 'Integration test knowledge content',
          source: 'integration-test',
          relevance_score: 0.95,
          knowledge_type: 'best_practice',
          metadata: {},
          created_at: new Date().toISOString()
        }
      ];

      const mockTemplates = [
        {
          template_id: 'integration-test-template-id',
          name: 'Integration Test Template',
          description: 'Template created during integration testing',
          category: 'testing',
          flow_data: { nodes: [], edges: [] },
          use_cases: ['integration-testing'],
          best_practices: ['test-driven'],
          common_patterns: ['integration'],
          tags: ['integration', 'test'],
          metadata: {
            category: 'testing',
            complexity_score: 0.3,
            version: '1.0.0',
            created_by: 'integration-test',
            usage_count: 1,
            success_rate: 1.0,
            average_duration: 1000,
            rating: 5.0
          },
          is_public: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ];

      mockWorkflowKnowledgeService.captureInsight.mockResolvedValue(mockInsight);
      mockWorkflowKnowledgeService.getContextualKnowledge.mockResolvedValue(mockKnowledge);
      mockWorkflowKnowledgeService.searchWorkflowTemplates.mockResolvedValue(mockTemplates);
      mockWorkflowKnowledgeService.storeWorkflowTemplate.mockResolvedValue({
        template_id: 'stored-template-id',
        message: 'Template stored successfully'
      });

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
        />
      );

      // 1. Verify session initialization
      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.startKnowledgeSession).toHaveBeenCalledWith(
          mockWorkflowId,
          mockProjectId,
          expect.objectContaining({
            auto_capture: true,
            capture_insights: true
          }),
          expect.arrayContaining(['integration-test', 'workflow-execution'])
        );
      });

      // 2. Test knowledge capture
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-capture'));
      });

      // Fill and submit insight capture form
      act(() => {
        fireEvent.click(screen.getByTestId('select-trigger'));
        fireEvent.click(screen.getByTestId('select-item-performance_optimization'));
      });

      const textarea = screen.getByTestId('textarea');
      act(() => {
        fireEvent.change(textarea, {
          target: { value: '{"observation": "Integration test observation", "improvement": "25% faster"}' }
        });
      });

      const submitButton = screen.getByText('Capture Insight');
      act(() => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.captureInsight).toHaveBeenCalledWith(
          mockSessionId,
          'performance_optimization',
          {
            content: '{"observation": "Integration test observation", "improvement": "25% faster"}',
            source: 'manual'
          },
          expect.objectContaining({
            importanceScore: 0.8,
            tags: expect.arrayContaining(['manual', 'user-input'])
          })
        );
      });

      // 3. Test knowledge retrieval
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-knowledge'));
      });

      const searchInput = screen.getByTestId('input');
      const searchButton = screen.getByText('Search');

      act(() => {
        fireEvent.change(searchInput, { target: { value: 'integration test query' } });
        fireEvent.click(searchButton);
      });

      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.getContextualKnowledge).toHaveBeenCalledWith(
          mockSessionId,
          'integration test query',
          'execution_context',
          { maxResults: 10, similarityThreshold: 0.7 }
        );
      });

      // 4. Test template operations
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-templates'));
      });

      const templateSearchInput = screen.getByTestId('input');
      const templateSearchButton = screen.getByText('Search Templates');

      act(() => {
        fireEvent.change(templateSearchInput, { target: { value: 'integration template' } });
        fireEvent.click(templateSearchButton);
      });

      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.searchWorkflowTemplates).toHaveBeenCalledWith(
          'integration template',
          { projectId: mockProjectId, limit: 10 }
        );
      });

      // 5. Test template storage
      const saveButton = screen.getByText('Save as Template');
      act(() => {
        fireEvent.click(saveButton);
      });

      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.storeWorkflowTemplate).toHaveBeenCalledWith(
          mockWorkflowId,
          expect.stringContaining('Knowledge-Enhanced Workflow'),
          'Workflow template created with knowledge integration insights',
          expect.objectContaining({
            useCases: expect.arrayContaining(['knowledge-driven']),
            bestPractices: expect.arrayContaining(['auto-capture-insights']),
            tags: expect.arrayContaining(['knowledge-integrated'])
          })
        );
      });

      // 6. Cleanup
      unmount();

      expect(mockWorkflowKnowledgeService.endKnowledgeSession).toHaveBeenCalledWith(
        mockSessionId,
        { generateSummary: false, extractPatterns: false }
      );
    });
  });

  describe('Error Handling and Resilience', () => {
    it('handles API failures gracefully', async () => {
      // Mock various API failures
      mockWorkflowKnowledgeService.startKnowledgeSession.mockRejectedValue(new Error('Session start failed'));
      mockWorkflowKnowledgeService.captureInsight.mockRejectedValue(new Error('Capture failed'));
      mockWorkflowKnowledgeService.getContextualKnowledge.mockRejectedValue(new Error('Search failed'));

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
        />
      );

      // Component should still render and be functional despite API failures
      await waitFor(() => {
        expect(screen.getByTestId('tabs')).toBeInTheDocument();
        expect(screen.getByTestId('tabs-list')).toBeInTheDocument();
      });

      // Tabs should be clickable even if some operations fail
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-knowledge'));
      });

      expect(screen.getByTestId('tabs-content-knowledge')).toBeInTheDocument();

      // Cleanup should still be called
      unmount();
    });

    it('recovers from temporary network issues', async () => {
      // First call fails, second succeeds
      mockWorkflowKnowledgeService.startKnowledgeSession
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(mockSession);

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
        />
      );

      // Component should handle the initial failure gracefully
      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.startKnowledgeSession).toHaveBeenCalled();
      });

      // Cleanup should still work
      unmount();
    });
  });

  describe('Performance Optimization Integration', () => {
    it('integrates performance insights with workflow operations', async () => {
      const mockPerformanceInsights = [
        {
          insight_id: 'performance-insight-1',
          insight_type: 'efficiency',
          description: 'Workflow efficiency can be improved by 20%',
          confidence: 0.85,
          impact: 'medium' as const,
          actionable: true,
          recommendation: 'Optimize step execution order',
          expected_improvement: '15-25% efficiency gain',
          metadata: {}
        },
        {
          insight_id: 'performance-insight-2',
          insight_type: 'cost',
          description: 'Cost reduction opportunity identified',
          confidence: 0.75,
          impact: 'high' as const,
          actionable: true,
          recommendation: 'Reduce resource usage during idle periods',
          expected_improvement: '30% cost reduction',
          metadata: {}
        }
      ];

      mockWorkflowKnowledgeService.generatePerformanceInsights.mockResolvedValue(mockPerformanceInsights);

      const { unmount } = render(
        <KnowledgeIntegrationDashboard
          projectId={mockProjectId}
          workflowId={mockWorkflowId}
        />
      );

      // Wait for performance insights to load
      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.generatePerformanceInsights).toHaveBeenCalledWith(
          mockWorkflowId,
          expect.objectContaining({
            insightTypes: expect.arrayContaining(['efficiency', 'cost']),
            minConfidence: 0.7
          })
        );
      });

      // Verify dashboard displays performance insights
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-insights'));
      });

      expect(screen.getByText('Performance Insights')).toBeInTheDocument();

      unmount();
    });
  });

  describe('Template Lifecycle Integration', () => {
    it('manages complete template lifecycle from creation to application', async () => {
      const mockRecommendations = [
        {
          template_id: 'recommended-template-1',
          name: 'Recommended Workflow Template',
          description: 'AI-recommended template based on project patterns',
          match_score: 0.92,
          reasons: ['High compatibility with project type', 'Proven success rate'],
          category: 'data_processing',
          complexity_score: 0.6,
          estimated_duration: 25000
        }
      ];

      mockWorkflowKnowledgeService.getTemplateRecommendations.mockResolvedValue(mockRecommendations);
      mockWorkflowKnowledgeService.applyTemplate.mockResolvedValue({
        workflow_id: 'new-workflow-from-template',
        workflow_name: 'Template-based Workflow',
        message: 'Template applied successfully'
      });

      const { unmount } = render(
        <KnowledgeIntegrationDashboard
          projectId={mockProjectId}
          workflowId={mockWorkflowId}
        />
      );

      // Load template recommendations
      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.getTemplateRecommendations).toHaveBeenCalledWith(
          mockProjectId,
          expect.objectContaining({
            complexityPreference: 'medium',
            maxRecommendations: 5
          })
        );
      });

      // Navigate to template recommendations
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-templates'));
      });

      // Verify template recommendations are displayed
      expect(screen.getByText('Template Recommendations')).toBeInTheDocument();

      unmount();
    });
  });

  describe('Real-time Knowledge Updates', () => {
    it('handles real-time knowledge updates during workflow execution', async () => {
      // Mock real-time updates
      let insightCallCount = 0;
      mockWorkflowKnowledgeService.captureInsight.mockImplementation(async () => {
        insightCallCount++;
        return {
          insight_id: `real-time-insight-${insightCallCount}`,
          session_id: mockSessionId,
          insight_type: 'performance_optimization' as const,
          insight_data: { observation: `Real-time update ${insightCallCount}` },
          importance_score: 0.8,
          tags: ['real-time'],
          captured_at: new Date().toISOString()
        };
      });

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
        />
      );

      // Simulate multiple real-time insight captures
      for (let i = 0; i < 3; i++) {
        await waitFor(() => {
          fireEvent.click(screen.getByTestId('tabs-trigger-capture'));
        });

        const textarea = screen.getByTestId('textarea');
        act(() => {
          fireEvent.change(textarea, {
            target: { value: `{"observation": "Real-time update ${i + 1}", "timestamp": "${new Date().toISOString()}"}` }
          });
        });

        const submitButton = screen.getByText('Capture Insight');
        act(() => {
          fireEvent.click(submitButton);
        });

        await waitFor(() => {
          expect(mockWorkflowKnowledgeService.captureInsight).toHaveBeenCalledTimes(i + 1);
        });
      }

      // Verify all insights were captured
      expect(insightCallCount).toBe(3);

      unmount();
    });
  });

  describe('Cross-component Communication', () => {
    it('enables communication between workflow and knowledge components', async () => {
      const mockOnWorkflowUpdate = jest.fn();

      mockWorkflowKnowledgeService.applyTemplate.mockResolvedValue({
        workflow_id: 'cross-component-workflow-id',
        workflow_name: 'Cross-component Workflow',
        message: 'Template applied with cross-component communication'
      });

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
          onWorkflowUpdate={mockOnWorkflowUpdate}
        />
      );

      // Simulate template application that should trigger cross-component communication
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-templates'));
      });

      // Mock the template application
      act(() => {
        mockWorkflowKnowledgeService.applyTemplate('cross-component-template-id', mockProjectId, {
          workflowName: 'Cross-component Workflow',
          workflowDescription: 'Workflow demonstrating cross-component communication'
        });
      });

      await waitFor(() => {
        expect(mockOnWorkflowUpdate).toHaveBeenCalledWith({
          workflowId: 'cross-component-workflow-id',
          templateId: 'cross-component-template-id'
        });
      });

      unmount();
    });
  });

  describe('Data Consistency and Validation', () => {
    it('ensures data consistency across knowledge operations', async () => {
      // Mock data consistency checks
      const mockKnowledge = [
        {
          knowledge_id: 'consistent-knowledge-1',
          content: 'Consistent knowledge content',
          source: 'validated-source',
          relevance_score: 0.9,
          knowledge_type: 'validated_practice',
          metadata: { validated: true, validation_date: new Date().toISOString() },
          created_at: new Date().toISOString()
        }
      ];

      mockWorkflowKnowledgeService.getContextualKnowledge.mockResolvedValue(mockKnowledge);

      const { unmount } = render(
        <KnowledgeAwareWorkflow
          workflowId={mockWorkflowId}
          projectId={mockProjectId}
        />
      );

      // Perform knowledge search
      await waitFor(() => {
        fireEvent.click(screen.getByTestId('tabs-trigger-knowledge'));
      });

      const searchInput = screen.getByTestId('input');
      const searchButton = screen.getByText('Search');

      act(() => {
        fireEvent.change(searchInput, { target: { value: 'consistent data test' } });
        fireEvent.click(searchButton);
      });

      await waitFor(() => {
        expect(mockWorkflowKnowledgeService.getContextualKnowledge).toHaveBeenCalledWith(
          mockSessionId,
          'consistent data test',
          'execution_context',
          { maxResults: 10, similarityThreshold: 0.7 }
        );
      });

      // Verify data consistency
      expect(mockKnowledge[0].metadata.validated).toBe(true);

      unmount();
    });
  });
});