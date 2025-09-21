import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Alert, AlertDescription } from '../ui/alert';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Textarea } from '../ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';

import { workflowKnowledgeService, type WorkflowKnowledgeSession, type WorkflowInsight, type ContextualKnowledge, type WorkflowTemplate } from '../../services/workflowKnowledgeService';
import { workflowService } from '../../services/workflowService';
import { knowledgeBaseService } from '../../services/knowledgeBaseService';

interface KnowledgeAwareWorkflowProps {
  workflowId: string;
  projectId: string;
  onWorkflowUpdate?: (workflowData: any) => void;
}

export const KnowledgeAwareWorkflow: React.FC<KnowledgeAwareWorkflowProps> = ({
  workflowId,
  projectId,
  onWorkflowUpdate
}) => {
  const [knowledgeSession, setKnowledgeSession] = useState<WorkflowKnowledgeSession | null>(null);
  const [insights, setInsights] = useState<WorkflowInsight[]>([]);
  const [contextualKnowledge, setContextualKnowledge] = useState<ContextualKnowledge[]>([]);
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedInsightType, setSelectedInsightType] = useState('');
  const [manualInsightData, setManualInsightData] = useState('');

  // Initialize knowledge session
  useEffect(() => {
    const initializeSession = async () => {
      try {
        const session = await workflowKnowledgeService.startKnowledgeSession(
          workflowId,
          projectId,
          {
            auto_capture: true,
            capture_insights: true,
            capture_patterns: true,
            real_time_analysis: true
          },
          ['workflow-execution', 'knowledge-integration']
        );
        setKnowledgeSession(session);
      } catch (error) {
        console.error('Failed to initialize knowledge session:', error);
      }
    };

    initializeSession();

    return () => {
      // Cleanup session when component unmounts
      if (knowledgeSession) {
        workflowKnowledgeService.endKnowledgeSession(knowledgeSession.session_id);
      }
    };
  }, [workflowId, projectId]);

  // Search for contextual knowledge
  const searchContextualKnowledge = useCallback(async (query: string) => {
    if (!knowledgeSession || !query.trim()) return;

    try {
      const knowledge = await workflowKnowledgeService.getContextualKnowledge(
        knowledgeSession.session_id,
        query,
        'execution_context',
        { maxResults: 10, similarityThreshold: 0.7 }
      );
      setContextualKnowledge(knowledge);
    } catch (error) {
      console.error('Failed to search contextual knowledge:', error);
    }
  }, [knowledgeSession]);

  // Search for relevant templates
  const searchTemplates = useCallback(async (query: string) => {
    if (!query.trim()) return;

    try {
      const templates = await workflowKnowledgeService.searchWorkflowTemplates(query, {
        projectId,
        limit: 10
      });
      setTemplates(templates);
    } catch (error) {
      console.error('Failed to search templates:', error);
    }
  }, [projectId]);

  // Capture manual insight
  const captureManualInsight = async () => {
    if (!knowledgeSession || !selectedInsightType || !manualInsightData.trim()) return;

    setIsCapturing(true);
    try {
      const insight = await workflowKnowledgeService.captureInsight(
        knowledgeSession.session_id,
        selectedInsightType as any,
        { content: manualInsightData, source: 'manual' },
        { importanceScore: 0.8, tags: ['manual', 'user-input'] }
      );
      setInsights(prev => [insight, ...prev]);
      setManualInsightData('');
      setSelectedInsightType('');
    } catch (error) {
      console.error('Failed to capture insight:', error);
    } finally {
      setIsCapturing(false);
    }
  };

  // Apply template to workflow
  const applyTemplate = async (templateId: string) => {
    try {
      const result = await workflowKnowledgeService.applyTemplate(
        templateId,
        projectId,
        {
          workflowName: `Template-based Workflow ${new Date().toISOString()}`,
          workflowDescription: 'Workflow created from template with knowledge integration'
        }
      );

      // Notify parent component of workflow update
      if (onWorkflowUpdate) {
        onWorkflowUpdate({ workflowId: result.workflow_id, templateId });
      }
    } catch (error) {
      console.error('Failed to apply template:', error);
    }
  };

  // Store current workflow as template
  const storeAsTemplate = async () => {
    try {
      await workflowKnowledgeService.storeWorkflowTemplate(
        workflowId,
        `Knowledge-Enhanced Workflow ${new Date().toISOString().slice(0, 10)}`,
        'Workflow template created with knowledge integration insights',
        {
          useCases: ['knowledge-driven', 'automated-insights', 'optimized-execution'],
          bestPractices: ['auto-capture-insights', 'contextual-awareness', 'pattern-recognition'],
          tags: ['knowledge-integrated', 'ai-optimized', 'template-ready']
        }
      );
    } catch (error) {
      console.error('Failed to store workflow as template:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Knowledge Session Status */}
      {knowledgeSession && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              Knowledge Session Active
            </CardTitle>
            <CardDescription>
              Automatically capturing insights and patterns during workflow execution
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <Label>Session ID</Label>
                <p className="font-mono text-xs">{knowledgeSession.session_id.slice(0, 8)}...</p>
              </div>
              <div>
                <Label>Started</Label>
                <p>{new Date(knowledgeSession.started_at).toLocaleTimeString()}</p>
              </div>
              <div>
                <Label>Auto Capture</Label>
                <Badge variant={knowledgeSession.capture_config.auto_capture ? 'default' : 'secondary'}>
                  {knowledgeSession.capture_config.auto_capture ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
              <div>
                <Label>Context Tags</Label>
                <div className="flex gap-1 flex-wrap">
                  {knowledgeSession.context_tags.map(tag => (
                    <Badge key={tag} variant="outline" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Knowledge Integration Tabs */}
      <Tabs defaultValue="insights" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="capture">Capture</TabsTrigger>
        </TabsList>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Workflow Insights</h3>
            <Button onClick={storeAsTemplate} variant="outline">
              Save as Template
            </Button>
          </div>

          <div className="space-y-3">
            {insights.length === 0 ? (
              <Alert>
                <AlertDescription>
                  No insights captured yet. Insights will appear automatically during workflow execution.
                </AlertDescription>
              </Alert>
            ) : (
              insights.map(insight => (
                <Card key={insight.insight_id}>
                  <CardContent className="pt-4">
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="secondary">{insight.insight_type}</Badge>
                          <Badge variant="outline">
                            {Math.round(insight.importance_score * 100)}% importance
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {new Date(insight.captured_at).toLocaleString()}
                        </p>
                        <pre className="text-sm bg-muted p-2 rounded overflow-x-auto">
                          {JSON.stringify(insight.insight_data, null, 2)}
                        </pre>
                        {insight.tags.length > 0 && (
                          <div className="flex gap-1 flex-wrap">
                            {insight.tags.map(tag => (
                              <Badge key={tag} variant="outline" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        {/* Knowledge Tab */}
        <TabsContent value="knowledge" className="space-y-4">
          <div className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Search knowledge base for relevant information..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchContextualKnowledge(searchQuery)}
              />
              <Button onClick={() => searchContextualKnowledge(searchQuery)}>
                Search
              </Button>
            </div>

            {contextualKnowledge.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-medium">Relevant Knowledge Items</h4>
                {contextualKnowledge.map(knowledge => (
                  <Card key={knowledge.knowledge_id}>
                    <CardContent className="pt-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Badge variant="outline">{knowledge.knowledge_type}</Badge>
                          <span className="text-sm text-muted-foreground">
                            {Math.round(knowledge.relevance_score * 100)}% relevant
                          </span>
                        </div>
                        <p className="text-sm">{knowledge.content}</p>
                        <p className="text-xs text-muted-foreground">
                          Source: {knowledge.source} • {new Date(knowledge.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </TabsContent>

        {/* Templates Tab */}
        <TabsContent value="templates" className="space-y-4">
          <div className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Search workflow templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && searchTemplates(searchQuery)}
              />
              <Button onClick={() => searchTemplates(searchQuery)}>
                Search Templates
              </Button>
            </div>

            {templates.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-medium">Available Templates</h4>
                {templates.map(template => (
                  <Card key={template.template_id}>
                    <CardContent className="pt-4">
                      <div className="space-y-3">
                        <div className="flex items-start justify-between">
                          <div>
                            <h5 className="font-medium">{template.name}</h5>
                            <p className="text-sm text-muted-foreground">{template.description}</p>
                          </div>
                          <Button
                            size="sm"
                            onClick={() => applyTemplate(template.template_id)}
                          >
                            Apply Template
                          </Button>
                        </div>

                        <div className="flex items-center gap-4 text-sm">
                          <span>Category: {template.category}</span>
                          <span>Complexity: {Math.round(template.metadata.complexity_score * 100)}%</span>
                          <span>Usage: {template.metadata.usage_count} times</span>
                          <span>Rating: {template.metadata.rating.toFixed(1)}/5</span>
                        </div>

                        {template.tags.length > 0 && (
                          <div className="flex gap-1 flex-wrap">
                            {template.tags.map(tag => (
                              <Badge key={tag} variant="outline" className="text-xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </TabsContent>

        {/* Capture Tab */}
        <TabsContent value="capture" className="space-y-4">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Manual Insight Capture</h3>

            <div className="grid gap-4">
              <div>
                <Label htmlFor="insight-type">Insight Type</Label>
                <Select value={selectedInsightType} onValueChange={setSelectedInsightType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select insight type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="performance_optimization">Performance Optimization</SelectItem>
                    <SelectItem value="error_pattern">Error Pattern</SelectItem>
                    <SelectItem value="success_pattern">Success Pattern</SelectItem>
                    <SelectItem value="best_practice">Best Practice</SelectItem>
                    <SelectItem value="bottleneck_identified">Bottleneck Identified</SelectItem>
                    <SelectItem value="efficiency_gain">Efficiency Gain</SelectItem>
                    <SelectItem value="cost_optimization">Cost Optimization</SelectItem>
                    <SelectItem value="quality_improvement">Quality Improvement</SelectItem>
                    <SelectItem value="risk_mitigation">Risk Mitigation</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="insight-data">Insight Data (JSON format)</Label>
                <Textarea
                  id="insight-data"
                  placeholder='{"observation": "Describe what you observed", "impact": "Describe the impact", "recommendation": "Suggested action"}'
                  value={manualInsightData}
                  onChange={(e) => setManualInsightData(e.target.value)}
                  rows={6}
                />
              </div>

              <Button
                onClick={captureManualInsight}
                disabled={!selectedInsightType || !manualInsightData.trim() || isCapturing}
              >
                {isCapturing ? 'Capturing...' : 'Capture Insight'}
              </Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default KnowledgeAwareWorkflow;