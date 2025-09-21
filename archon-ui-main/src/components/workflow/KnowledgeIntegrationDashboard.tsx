import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Progress } from '../ui/progress';
import { Alert, AlertDescription } from '../ui/alert';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';

import { workflowKnowledgeService, type PerformanceInsight, type TemplateRecommendation } from '../../services/workflowKnowledgeService';
import { workflowService } from '../../services/workflowService';

interface KnowledgeIntegrationDashboardProps {
  projectId: string;
  workflowId?: string;
}

interface KnowledgeMetrics {
  totalSessions: number;
  activeSessions: number;
  totalInsights: number;
  patternsIdentified: number;
  templatesCreated: number;
  averageImprovement: number;
}

interface PerformanceTrend {
  date: string;
  duration: number;
  cost: number;
  success_rate: number;
  resource_usage: number;
}

export const KnowledgeIntegrationDashboard: React.FC<KnowledgeIntegrationDashboardProps> = ({
  projectId,
  workflowId
}) => {
  const [metrics, setMetrics] = useState<KnowledgeMetrics>({
    totalSessions: 0,
    activeSessions: 0,
    totalInsights: 0,
    patternsIdentified: 0,
    templatesCreated: 0,
    averageImprovement: 0
  });

  const [performanceInsights, setPerformanceInsights] = useState<PerformanceInsight[]>([]);
  const [templateRecommendations, setTemplateRecommendations] = useState<TemplateRecommendation[]>([]);
  const [performanceTrends, setPerformanceTrends] = useState<PerformanceTrend[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('30d');
  const [loading, setLoading] = useState(false);

  // Load knowledge metrics
  useEffect(() => {
    loadKnowledgeMetrics();
    if (workflowId) {
      loadPerformanceInsights();
      loadPerformanceTrends();
    }
    loadTemplateRecommendations();
  }, [projectId, workflowId, selectedTimeRange]);

  const loadKnowledgeMetrics = async () => {
    try {
      // Mock data - in real implementation, this would come from API
      setMetrics({
        totalSessions: 45,
        activeSessions: 3,
        totalInsights: 234,
        patternsIdentified: 67,
        templatesCreated: 12,
        averageImprovement: 23.5
      });
    } catch (error) {
      console.error('Failed to load knowledge metrics:', error);
    }
  };

  const loadPerformanceInsights = async () => {
    if (!workflowId) return;

    setLoading(true);
    try {
      const insights = await workflowKnowledgeService.generatePerformanceInsights(workflowId, {
        insightTypes: ['efficiency', 'cost', 'reliability', 'scalability'],
        minConfidence: 0.7
      });
      setPerformanceInsights(insights);
    } catch (error) {
      console.error('Failed to load performance insights:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPerformanceTrends = async () => {
    if (!workflowId) return;

    try {
      // Mock trend data - in real implementation, this would come from API
      const trends: PerformanceTrend[] = [];
      const now = new Date();

      for (let i = 29; i >= 0; i--) {
        const date = new Date(now);
        date.setDate(date.getDate() - i);

        trends.push({
          date: date.toISOString().split('T')[0],
          duration: Math.random() * 1000 + 500, // 500-1500ms
          cost: Math.random() * 0.5 + 0.1, // $0.10-$0.60
          success_rate: 0.8 + Math.random() * 0.15, // 80-95%
          resource_usage: Math.random() * 50 + 30 // 30-80%
        });
      }

      setPerformanceTrends(trends);
    } catch (error) {
      console.error('Failed to load performance trends:', error);
    }
  };

  const loadTemplateRecommendations = async () => {
    try {
      const recommendations = await workflowKnowledgeService.getTemplateRecommendations(projectId, {
        complexityPreference: 'medium',
        maxRecommendations: 5
      });
      setTemplateRecommendations(recommendations);
    } catch (error) {
      console.error('Failed to load template recommendations:', error);
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Knowledge Integration Dashboard</h2>
          <p className="text-muted-foreground">
            Monitor knowledge capture, insights, and performance optimization
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline">Export Report</Button>
        </div>
      </div>

      {/* Knowledge Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Sessions</p>
                <p className="text-2xl font-bold">{metrics.totalSessions}</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                📊
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Sessions</p>
                <p className="text-2xl font-bold">{metrics.activeSessions}</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
                🟢
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Insights Captured</p>
                <p className="text-2xl font-bold">{metrics.totalInsights}</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-purple-100 flex items-center justify-center">
                💡
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Patterns Found</p>
                <p className="text-2xl font-bold">{metrics.patternsIdentified}</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-orange-100 flex items-center justify-center">
                🔍
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Templates</p>
                <p className="text-2xl font-bold">{metrics.templatesCreated}</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center">
                📋
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Avg Improvement</p>
                <p className="text-2xl font-bold">{metrics.averageImprovement}%</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-green-100 flex items-center justify-center">
                📈
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="insights" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="insights">Performance Insights</TabsTrigger>
          <TabsTrigger value="trends">Performance Trends</TabsTrigger>
          <TabsTrigger value="templates">Template Recommendations</TabsTrigger>
          <TabsTrigger value="optimization">Optimization Opportunities</TabsTrigger>
        </TabsList>

        {/* Performance Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Insights</CardTitle>
              <CardDescription>
                AI-generated insights for workflow optimization and improvement
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : performanceInsights.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No performance insights available. Start workflow executions to generate insights.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-4">
                  {performanceInsights.map(insight => (
                    <Card key={insight.insight_id}>
                      <CardContent className="pt-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">{insight.insight_type}</Badge>
                              <div className={`w-2 h-2 rounded-full ${getImpactColor(insight.impact)}`} />
                              <span className="text-sm capitalize">{insight.impact} impact</span>
                              {insight.actionable && (
                                <Badge variant="default">Actionable</Badge>
                              )}
                            </div>
                            <span className={`text-sm font-medium ${getConfidenceColor(insight.confidence)}`}>
                              {Math.round(insight.confidence * 100)}% confidence
                            </span>
                          </div>

                          <p className="text-sm">{insight.description}</p>

                          {insight.recommendation && (
                            <div className="bg-muted p-3 rounded-md">
                              <Label className="text-sm font-medium">Recommendation</Label>
                              <p className="text-sm mt-1">{insight.recommendation}</p>
                            </div>
                          )}

                          {insight.expected_improvement && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <span>Expected improvement:</span>
                              <Badge variant="secondary">{insight.expected_improvement}</Badge>
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Trends Tab */}
        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Trends</CardTitle>
              <CardDescription>
                Track workflow performance metrics over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {performanceTrends.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No performance data available. Execute workflows to see trends.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-6">
                  {/* Duration Trend */}
                  <div>
                    <Label className="text-sm font-medium">Execution Duration (ms)</Label>
                    <div className="mt-2 space-y-2">
                      {performanceTrends.slice(-7).map((trend, index) => (
                        <div key={trend.date} className="flex items-center gap-4">
                          <span className="text-sm w-20">{trend.date.slice(5)}</span>
                          <Progress value={(trend.duration / 1500) * 100} className="flex-1" />
                          <span className="text-sm w-16">{Math.round(trend.duration)}ms</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Success Rate Trend */}
                  <div>
                    <Label className="text-sm font-medium">Success Rate</Label>
                    <div className="mt-2 space-y-2">
                      {performanceTrends.slice(-7).map((trend, index) => (
                        <div key={trend.date} className="flex items-center gap-4">
                          <span className="text-sm w-20">{trend.date.slice(5)}</span>
                          <Progress value={trend.success_rate * 100} className="flex-1" />
                          <span className="text-sm w-16">{Math.round(trend.success_rate * 100)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Cost Trend */}
                  <div>
                    <Label className="text-sm font-medium">Cost per Execution ($)</Label>
                    <div className="mt-2 space-y-2">
                      {performanceTrends.slice(-7).map((trend, index) => (
                        <div key={trend.date} className="flex items-center gap-4">
                          <span className="text-sm w-20">{trend.date.slice(5)}</span>
                          <Progress value={(trend.cost / 0.6) * 100} className="flex-1" />
                          <span className="text-sm w-16">${trend.cost.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Template Recommendations Tab */}
        <TabsContent value="templates" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Template Recommendations</CardTitle>
              <CardDescription>
                AI-recommended workflow templates based on your project context
              </CardDescription>
            </CardHeader>
            <CardContent>
              {templateRecommendations.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No template recommendations available. Continue working on your project to get personalized recommendations.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-4">
                  {templateRecommendations.map(template => (
                    <Card key={template.template_id}>
                      <CardContent className="pt-4">
                        <div className="space-y-3">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h4 className="font-medium">{template.name}</h4>
                              <p className="text-sm text-muted-foreground">{template.description}</p>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">
                                {Math.round(template.match_score * 100)}% match
                              </Badge>
                              <Button size="sm">Use Template</Button>
                            </div>
                          </div>

                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span>{template.category}</span>
                            <span>Complexity: {Math.round(template.complexity_score * 100)}%</span>
                            {template.estimated_duration && (
                              <span>~{Math.round(template.estimated_duration / 1000)}s</span>
                            )}
                          </div>

                          <div>
                            <Label className="text-sm font-medium">Why recommended:</Label>
                            <ul className="text-sm text-muted-foreground mt-1 space-y-1">
                              {template.reasons.map((reason, index) => (
                                <li key={index}>• {reason}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Optimization Opportunities Tab */}
        <TabsContent value="optimization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Optimization Opportunities</CardTitle>
              <CardDescription>
                Identified opportunities for workflow optimization based on patterns and insights
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Mock optimization opportunities */}
                <Card>
                  <CardContent className="pt-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">Parallel Processing Opportunities</h4>
                        <Badge variant="default">High Impact</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Analysis shows 3 workflow steps that could be executed in parallel, potentially reducing execution time by 40%.
                      </p>
                      <Button size="sm" variant="outline">View Details</Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="pt-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">Resource Optimization</h4>
                        <Badge variant="secondary">Medium Impact</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Memory usage patterns suggest opportunities for resource optimization that could reduce costs by 25%.
                      </p>
                      <Button size="sm" variant="outline">View Details</Button>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardContent className="pt-4">
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium">Error Handling Improvements</h4>
                        <Badge variant="outline">Low Impact</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Enhanced error handling patterns could improve success rates by 8% based on historical data.
                      </p>
                      <Button size="sm" variant="outline">View Details</Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default KnowledgeIntegrationDashboard;