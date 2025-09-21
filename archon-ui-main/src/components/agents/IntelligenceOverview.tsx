import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  Brain,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  BarChart3,
  PieChart,
  Activity
} from 'lucide-react';

export interface AgentPerformanceMetrics {
  agent_id: string;
  agent_name: string;
  agent_type: string;
  tasks_completed: number;
  success_rate: number;
  avg_response_time: number;
  efficiency_score: number;
  last_active: string;
  performance_trend: 'up' | 'down' | 'stable';
}

export interface ProjectIntelligenceOverview {
  total_agents: number;
  active_agents: number;
  total_tasks_completed: number;
  overall_success_rate: number;
  avg_system_response_time: number;
  intelligence_tiers: {
    opus: { count: number; utilization: number };
    sonnet: { count: number; utilization: number };
    haiku: { count: number; utilization: number };
  };
  performance_insights: {
    top_performers: string[];
    underperformers: string[];
    optimization_opportunities: string[];
  };
  cost_analysis: {
    total_tokens_used: number;
    estimated_monthly_cost: number;
    cost_per_task: number;
    efficiency_rating: 'excellent' | 'good' | 'fair' | 'poor';
  };
}

interface IntelligenceOverviewProps {
  overview: ProjectIntelligenceOverview | null;
  performanceMetrics: AgentPerformanceMetrics[];
}

const getPerformanceColor = (score: number) => {
  if (score >= 90) return 'text-green-600';
  if (score >= 75) return 'text-blue-600';
  if (score >= 60) return 'text-yellow-600';
  return 'text-red-600';
};

const getEfficiencyBadge = (rating: string) => {
  const variants = {
    excellent: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
    good: 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400',
    fair: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
    poor: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
  };
  
  return variants[rating as keyof typeof variants] || variants.fair;
};

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};

const formatNumber = (num: number) => {
  return new Intl.NumberFormat('en-US').format(num);
};

export const IntelligenceOverview: React.FC<IntelligenceOverviewProps> = ({
  overview,
  performanceMetrics
}) => {
  if (!overview) {
    return (
      <Card className="p-8">
        <div className="text-center text-gray-500">
          <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No Intelligence Data</h3>
          <p>Start using agents to generate intelligence insights.</p>
        </div>
      </Card>
    );
  }

  const topPerformers = performanceMetrics
    .sort((a, b) => b.efficiency_score - a.efficiency_score)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="w-8 h-8 text-purple-600" />
            Intelligence Overview
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            System-wide AI performance and insights
          </p>
        </div>
        <Button variant="outline" className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Export Report
        </Button>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">
                  {overview.total_agents}
                </div>
                <div className="text-sm text-gray-500">Total Agents</div>
              </div>
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {overview.active_agents}
                </div>
                <div className="text-sm text-gray-500">Active Now</div>
              </div>
              <Activity className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {Math.round(overview.overall_success_rate * 100)}%
                </div>
                <div className="text-sm text-gray-500">Success Rate</div>
              </div>
              <Target className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">
                  {overview.avg_system_response_time.toFixed(1)}s
                </div>
                <div className="text-sm text-gray-500">Avg Response</div>
              </div>
              <Clock className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="performance" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">🎯 Performance</TabsTrigger>
          <TabsTrigger value="tiers">🧠 Intelligence Tiers</TabsTrigger>
          <TabsTrigger value="insights">💡 Insights</TabsTrigger>
          <TabsTrigger value="cost">💰 Cost Analysis</TabsTrigger>
        </TabsList>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Performers */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  Top Performers
                </CardTitle>
                <CardDescription>Highest efficiency agents</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {topPerformers.map((agent, index) => (
                    <div key={agent.agent_id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-6 h-6 rounded-full bg-green-100 dark:bg-green-900/20 flex items-center justify-center text-xs font-bold text-green-600">
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium">{agent.agent_name}</div>
                          <div className="text-sm text-gray-500">{agent.agent_type}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`font-bold ${getPerformanceColor(agent.efficiency_score)}`}>
                          {agent.efficiency_score}%
                        </div>
                        <div className="text-sm text-gray-500">
                          {agent.tasks_completed} tasks
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* System Health */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-blue-600" />
                  System Health
                </CardTitle>
                <CardDescription>Overall system performance metrics</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Overall Success Rate</span>
                    <span>{Math.round(overview.overall_success_rate * 100)}%</span>
                  </div>
                  <Progress value={overview.overall_success_rate * 100} className="h-2" />
                </div>
                
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span>Agent Utilization</span>
                    <span>{Math.round((overview.active_agents / overview.total_agents) * 100)}%</span>
                  </div>
                  <Progress value={(overview.active_agents / overview.total_agents) * 100} className="h-2" />
                </div>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatNumber(overview.total_tasks_completed)}
                    </div>
                    <div className="text-sm text-gray-500">Total Tasks</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {overview.avg_system_response_time.toFixed(1)}s
                    </div>
                    <div className="text-sm text-gray-500">Avg Response</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Intelligence Tiers Tab */}
        <TabsContent value="tiers" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(overview.intelligence_tiers).map(([tier, data]) => (
              <Card key={tier}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="capitalize">{tier}</span>
                    <Badge variant="outline">{data.count} agents</Badge>
                  </CardTitle>
                  <CardDescription>
                    {tier === 'opus' && 'Highest intelligence, most expensive'}
                    {tier === 'sonnet' && 'Balanced performance and cost'}
                    {tier === 'haiku' && 'Fast and efficient, cost-effective'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Utilization</span>
                        <span>{Math.round(data.utilization * 100)}%</span>
                      </div>
                      <Progress value={data.utilization * 100} className="h-2" />
                    </div>
                    <div className="text-center pt-2">
                      <Zap className={`w-8 h-8 mx-auto ${
                        tier === 'opus' ? 'text-purple-600' :
                        tier === 'sonnet' ? 'text-blue-600' : 'text-green-600'
                      }`} />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-600" />
                  Optimization Opportunities
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {overview.performance_insights.optimization_opportunities.map((opportunity, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                      <AlertTriangle className="w-4 h-4 text-yellow-600 mt-0.5" />
                      <div className="text-sm">{opportunity}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  System Strengths
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {overview.performance_insights.top_performers.map((performer, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
                      <div className="text-sm">Agent "{performer}" exceeding performance targets</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Cost Analysis Tab */}
        <TabsContent value="cost" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-green-600" />
                  Cost Overview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {formatCurrency(overview.cost_analysis.estimated_monthly_cost)}
                    </div>
                    <div className="text-sm text-gray-500">Monthly Cost</div>
                  </div>
                  <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatCurrency(overview.cost_analysis.cost_per_task)}
                    </div>
                    <div className="text-sm text-gray-500">Cost per Task</div>
                  </div>
                </div>
                
                <div className="text-center pt-4">
                  <Badge className={getEfficiencyBadge(overview.cost_analysis.efficiency_rating)}>
                    {overview.cost_analysis.efficiency_rating.toUpperCase()} Efficiency
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-purple-600" />
                  Token Usage
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center space-y-4">
                  <div>
                    <div className="text-3xl font-bold text-purple-600">
                      {formatNumber(overview.cost_analysis.total_tokens_used)}
                    </div>
                    <div className="text-sm text-gray-500">Total Tokens Used</div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="p-2 bg-purple-50 dark:bg-purple-900/20 rounded">
                      <div className="font-medium">Opus</div>
                      <div className="text-purple-600">High Cost</div>
                    </div>
                    <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded">
                      <div className="font-medium">Sonnet</div>
                      <div className="text-blue-600">Balanced</div>
                    </div>
                    <div className="p-2 bg-green-50 dark:bg-green-900/20 rounded">
                      <div className="font-medium">Haiku</div>
                      <div className="text-green-600">Low Cost</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default IntelligenceOverview;