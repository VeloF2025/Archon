import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface QualityMetrics {
  period_days: number;
  start_date: string;
  end_date: string;
  metrics: {
    [key: string]: {
      current_value: number | null;
      trend_direction: 'improving' | 'declining' | 'stable';
      trend_strength: number;
      data_points: number;
    };
  };
}

interface QualityInsight {
  insight_type: 'improvement' | 'warning' | 'achievement';
  title: string;
  description: string;
  impact_level: 'low' | 'medium' | 'high';
  recommendations: string[];
  metrics: string[];
  timestamp: string;
}

interface ViolationData {
  rule: string;
  severity: string;
  count: number;
}

interface WorkflowStatus {
  request_id: string;
  status: 'running' | 'completed' | 'failed';
  stage: string;
  trigger: string;
  start_time: string;
  end_time?: string;
  total_violations?: number;
}

const QADashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<QualityMetrics | null>(null);
  const [insights, setInsights] = useState<QualityInsight[]>([]);
  const [topViolations, setTopViolations] = useState<ViolationData[]>([]);
  const [activeWorkflows, setActiveWorkflows] = useState<WorkflowStatus[]>([]);
  const [workflowHistory, setWorkflowHistory] = useState<WorkflowStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState(7);

  // Colors for charts
  const COLORS = {
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
    improving: '#10b981',
    declining: '#ef4444',
    stable: '#6b7280'
  };

  useEffect(() => {
    loadQualityData();
    const interval = setInterval(loadQualityData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [selectedPeriod]);

  const loadQualityData = async () => {
    setLoading(true);
    try {
      // Load quality metrics
      const metricsResponse = await fetch(`/api/quality/metrics?days=${selectedPeriod}`);
      const metricsData = await metricsResponse.json();
      setMetrics(metricsData);

      // Load insights
      const insightsResponse = await fetch(`/api/quality/insights?days=${selectedPeriod}`);
      const insightsData = await insightsResponse.json();
      setInsights(insightsData);

      // Load top violations
      const violationsResponse = await fetch(`/api/quality/violations/top?days=${selectedPeriod}`);
      const violationsData = await violationsResponse.json();
      setTopViolations(violationsData);

      // Load active workflows
      const activeResponse = await fetch('/api/quality/workflows/active');
      const activeData = await activeResponse.json();
      setActiveWorkflows(activeData);

      // Load workflow history
      const historyResponse = await fetch(`/api/quality/workflows/history?limit=20`);
      const historyData = await historyResponse.json();
      setWorkflowHistory(historyData);

    } catch (error) {
      console.error('Failed to load quality data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'improving':
        return '📈';
      case 'declining':
        return '📉';
      default:
        return '➡️';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'achievement':
        return '🏆';
      case 'warning':
        return '⚠️';
      case 'improvement':
        return '💡';
      default:
        return '📊';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return COLORS.error;
      case 'error':
        return COLORS.error;
      case 'warning':
        return COLORS.warning;
      default:
        return COLORS.info;
    }
  };

  const prepareQualityScoreData = () => {
    if (!metrics) return [];

    return Object.entries(metrics.metrics).map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: value.current_value || 0,
      trend: value.trend_direction,
      dataPoints: value.data_points
    }));
  };

  const prepareViolationsChartData = () => {
    return topViolations.map(violation => ({
      name: violation.rule.length > 20 ? violation.rule.substring(0, 20) + '...' : violation.rule,
      count: violation.count,
      severity: violation.severity
    }));
  };

  const prepareWorkflowTrendData = () => {
    // This would typically come from an API endpoint
    return workflowHistory.slice(0, 10).reverse().map((workflow, index) => ({
      date: new Date(workflow.start_time).toLocaleDateString(),
      successful: workflow.status === 'completed' ? 1 : 0,
      failed: workflow.status === 'failed' ? 1 : 0,
      violations: workflow.total_violations || 0
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading Quality Dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Quality Assurance Dashboard</h1>
          <p className="text-gray-600">Monitor code quality, test coverage, and compliance metrics</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(Number(e.target.value))}
            className="px-3 py-2 border rounded-md"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          <Button onClick={loadQualityData}>Refresh</Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics && Object.entries(metrics.metrics).map(([key, metric]) => (
          <Card key={key}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="text-2xl font-bold">
                  {metric.current_value !== null
                    ? (metric.current_value * 100).toFixed(1) + '%'
                    : 'N/A'
                  }
                </div>
                <div className="flex items-center space-x-1">
                  <span>{getTrendIcon(metric.trend_direction)}</span>
                  <Badge
                    variant="outline"
                    style={{
                      borderColor: COLORS[metric.trend_direction as keyof typeof COLORS],
                      color: COLORS[metric.trend_direction as keyof typeof COLORS]
                    }}
                  >
                    {metric.trend_direction}
                  </Badge>
                </div>
              </div>
              <div className="mt-2">
                <Progress
                  value={metric.current_value ? metric.current_value * 100 : 0}
                  className="h-2"
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {metric.data_points} data points
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="violations">Violations</TabsTrigger>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quality Scores Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Quality Metrics</CardTitle>
                <CardDescription>Current quality scores across different dimensions</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={prepareQualityScoreData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="name"
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      interval={0}
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="value" fill={COLORS.info} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Workflow Trend Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Workflow Success Trend</CardTitle>
                <CardDescription>Recent workflow execution results</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={prepareWorkflowTrendData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="successful" stroke={COLORS.success} name="Successful" />
                    <Line type="monotone" dataKey="failed" stroke={COLORS.error} name="Failed" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Active Workflows */}
          {activeWorkflows.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Active Workflows</CardTitle>
                <CardDescription>Currently running quality assurance workflows</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {activeWorkflows.map((workflow) => (
                    <div key={workflow.request_id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                        <div>
                          <p className="font-medium">{workflow.stage}</p>
                          <p className="text-sm text-gray-500">
                            {workflow.trigger} • Started {new Date(workflow.start_time).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant="outline">Running</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Violations Tab */}
        <TabsContent value="violations" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Violations Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Top Violations</CardTitle>
                <CardDescription>Most common quality violations</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={prepareViolationsChartData()} layout="horizontal">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={150} />
                    <Tooltip />
                    <Bar dataKey="count" fill={COLORS.error} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Violations List */}
            <Card>
              <CardHeader>
                <CardTitle>Violation Details</CardTitle>
                <CardDescription>Detailed breakdown of quality violations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {topViolations.map((violation, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium">{violation.rule}</p>
                        <p className="text-sm text-gray-500">Count: {violation.count}</p>
                      </div>
                      <Badge
                        style={{
                          backgroundColor: getSeverityColor(violation.severity),
                          color: 'white'
                        }}
                      >
                        {violation.severity}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Workflows Tab */}
        <TabsContent value="workflows" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Workflow History</CardTitle>
              <CardDescription>Recent quality assurance workflow executions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {workflowHistory.map((workflow) => (
                  <div key={workflow.request_id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <p className="font-medium">{workflow.stage}</p>
                        <Badge variant="outline">{workflow.trigger}</Badge>
                      </div>
                      <p className="text-sm text-gray-500">
                        {new Date(workflow.start_time).toLocaleString()}
                        {workflow.end_time && ` - ${new Date(workflow.end_time).toLocaleString()}`}
                      </p>
                      {workflow.total_violations !== undefined && (
                        <p className="text-sm text-gray-500">
                          Violations: {workflow.total_violations}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge
                        variant={workflow.status === 'completed' ? 'default' :
                               workflow.status === 'failed' ? 'destructive' : 'secondary'}
                      >
                        {workflow.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {insights.map((insight, index) => (
              <Card key={index}>
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">{getInsightIcon(insight.insight_type)}</span>
                    <CardTitle className="text-lg">{insight.title}</CardTitle>
                  </div>
                  <CardDescription>
                    <Badge
                      variant="outline"
                      style={{
                        borderColor: COLORS[insight.impact_level as keyof typeof COLORS],
                        color: COLORS[insight.impact_level as keyof typeof COLORS]
                      }}
                    >
                      {insight.impact_level} impact
                    </Badge>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-700 mb-4">{insight.description}</p>

                  {insight.recommendations.length > 0 && (
                    <div>
                      <h4 className="font-medium mb-2">Recommendations:</h4>
                      <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                        {insight.recommendations.map((rec, recIndex) => (
                          <li key={recIndex}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className="mt-4 pt-4 border-t">
                    <p className="text-xs text-gray-500">
                      Generated: {new Date(insight.timestamp).toLocaleString()}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default QADashboard;