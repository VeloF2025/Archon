import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Select } from '../ui/Select';
import { 
  DollarSign,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Target,
  PieChart,
  BarChart3,
  Zap,
  Settings,
  Download,
  Calendar,
  CreditCard,
  Lightbulb
} from 'lucide-react';

export interface CostOptimizationRecommendation {
  id: string;
  type: 'tier_optimization' | 'usage_pattern' | 'scheduling' | 'resource_allocation';
  title: string;
  description: string;
  potential_savings: number;
  effort_level: 'low' | 'medium' | 'high';
  implementation_time: string;
  impact_score: number;
  affected_agents: string[];
}

export interface ProjectIntelligenceOverview {
  cost_analysis: {
    total_tokens_used: number;
    estimated_monthly_cost: number;
    cost_per_task: number;
    efficiency_rating: 'excellent' | 'good' | 'fair' | 'poor';
  };
  intelligence_tiers: {
    opus: { count: number; utilization: number; cost_share: number };
    sonnet: { count: number; utilization: number; cost_share: number };
    haiku: { count: number; utilization: number; cost_share: number };
  };
}

interface CostDashboardProps {
  recommendations: CostOptimizationRecommendation[];
  overview: ProjectIntelligenceOverview | null;
}

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  }).format(amount);
};

const formatNumber = (num: number) => {
  return new Intl.NumberFormat('en-US').format(num);
};

const getEfficiencyColor = (rating: string) => {
  switch (rating) {
    case 'excellent': return 'text-green-600';
    case 'good': return 'text-blue-600';
    case 'fair': return 'text-yellow-600';
    case 'poor': return 'text-red-600';
    default: return 'text-gray-600';
  }
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

const getRecommendationIcon = (type: string) => {
  switch (type) {
    case 'tier_optimization': return <Zap className="w-4 h-4" />;
    case 'usage_pattern': return <BarChart3 className="w-4 h-4" />;
    case 'scheduling': return <Calendar className="w-4 h-4" />;
    case 'resource_allocation': return <Settings className="w-4 h-4" />;
    default: return <Lightbulb className="w-4 h-4" />;
  }
};

const getEffortBadge = (effort: string) => {
  const variants = {
    low: 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
    medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
    high: 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
  };
  
  return variants[effort as keyof typeof variants] || variants.medium;
};

export const CostDashboard: React.FC<CostDashboardProps> = ({
  recommendations,
  overview
}) => {
  const [timeframe, setTimeframe] = useState('monthly');
  const [sortBy, setSortBy] = useState('savings');

  if (!overview) {
    return (
      <Card className="p-8">
        <div className="text-center text-gray-500">
          <DollarSign className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium mb-2">No Cost Data</h3>
          <p>Start using agents to generate cost analytics.</p>
        </div>
      </Card>
    );
  }

  const totalPotentialSavings = recommendations.reduce((sum, rec) => sum + rec.potential_savings, 0);
  const sortedRecommendations = [...recommendations].sort((a, b) => {
    if (sortBy === 'savings') return b.potential_savings - a.potential_savings;
    if (sortBy === 'impact') return b.impact_score - a.impact_score;
    return a.effort_level.localeCompare(b.effort_level);
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <DollarSign className="w-8 h-8 text-green-600" />
            Cost Management Dashboard
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Monitor and optimize AI agent costs
          </p>
        </div>
        <div className="flex gap-2">
          <Select value={timeframe} onValueChange={setTimeframe}>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="quarterly">Quarterly</option>
          </Select>
          <Button variant="outline" className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-green-600">
                  {formatCurrency(overview.cost_analysis.estimated_monthly_cost)}
                </div>
                <div className="text-sm text-gray-500">Monthly Cost</div>
              </div>
              <CreditCard className="w-8 h-8 text-green-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-blue-600">
                  {formatCurrency(overview.cost_analysis.cost_per_task)}
                </div>
                <div className="text-sm text-gray-500">Cost per Task</div>
              </div>
              <Target className="w-8 h-8 text-blue-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-purple-600">
                  {formatNumber(overview.cost_analysis.total_tokens_used)}
                </div>
                <div className="text-sm text-gray-500">Tokens Used</div>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-400" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold text-orange-600">
                  {formatCurrency(totalPotentialSavings)}
                </div>
                <div className="text-sm text-gray-500">Potential Savings</div>
              </div>
              <TrendingDown className="w-8 h-8 text-orange-400" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">📊 Overview</TabsTrigger>
          <TabsTrigger value="breakdown">🔍 Breakdown</TabsTrigger>
          <TabsTrigger value="recommendations">💡 Optimization</TabsTrigger>
          <TabsTrigger value="trends">📈 Trends</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Cost Efficiency */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5 text-blue-600" />
                  Cost Efficiency
                </CardTitle>
                <CardDescription>Overall system cost performance</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <Badge className={getEfficiencyBadge(overview.cost_analysis.efficiency_rating)}>
                    {overview.cost_analysis.efficiency_rating.toUpperCase()}
                  </Badge>
                </div>
                
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Cost Efficiency Score</span>
                      <span>{overview.cost_analysis.efficiency_rating === 'excellent' ? 95 : 
                            overview.cost_analysis.efficiency_rating === 'good' ? 80 :
                            overview.cost_analysis.efficiency_rating === 'fair' ? 60 : 40}%</span>
                    </div>
                    <Progress value={overview.cost_analysis.efficiency_rating === 'excellent' ? 95 : 
                                   overview.cost_analysis.efficiency_rating === 'good' ? 80 :
                                   overview.cost_analysis.efficiency_rating === 'fair' ? 60 : 40} className="h-2" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded">
                    <div className="text-lg font-bold text-green-600">
                      {formatCurrency(overview.cost_analysis.estimated_monthly_cost * 0.85)}
                    </div>
                    <div className="text-xs text-gray-500">Target Cost</div>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded">
                    <div className="text-lg font-bold text-red-600">
                      {formatCurrency(overview.cost_analysis.estimated_monthly_cost * 1.2)}
                    </div>
                    <div className="text-xs text-gray-500">Budget Limit</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-600" />
                  Quick Optimizations
                </CardTitle>
                <CardDescription>Immediate cost reduction opportunities</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {sortedRecommendations.slice(0, 3).map((rec) => (
                    <div key={rec.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="flex items-center gap-3">
                        {getRecommendationIcon(rec.type)}
                        <div>
                          <div className="font-medium text-sm">{rec.title}</div>
                          <div className="text-xs text-gray-500">{rec.implementation_time}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-600 text-sm">
                          {formatCurrency(rec.potential_savings)}
                        </div>
                        <Badge size="sm" className={getEffortBadge(rec.effort_level)}>
                          {rec.effort_level}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
                <Button className="w-full mt-4" variant="outline">
                  View All Recommendations
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Cost Breakdown Tab */}
        <TabsContent value="breakdown" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Intelligence Tier Costs */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PieChart className="w-5 h-5 text-purple-600" />
                  Cost by Intelligence Tier
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(overview.intelligence_tiers).map(([tier, data]) => (
                    <div key={tier} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-2">
                          <div className={`w-3 h-3 rounded-full ${
                            tier === 'opus' ? 'bg-purple-600' :
                            tier === 'sonnet' ? 'bg-blue-600' : 'bg-green-600'
                          }`}></div>
                          <span className="font-medium capitalize">{tier}</span>
                          <Badge variant="outline">{data.count} agents</Badge>
                        </div>
                        <span className="font-bold">
                          {Math.round(data.cost_share * 100)}%
                        </span>
                      </div>
                      <Progress value={data.cost_share * 100} className="h-2" />
                      <div className="text-sm text-gray-500 text-right">
                        {formatCurrency(overview.cost_analysis.estimated_monthly_cost * data.cost_share)}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Usage Patterns */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  Usage Patterns
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                      <div className="text-2xl font-bold text-purple-600">
                        {Math.round(overview.intelligence_tiers.opus.utilization * 100)}%
                      </div>
                      <div className="text-sm text-gray-500">Opus Utilization</div>
                    </div>
                    <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                      <div className="text-2xl font-bold text-blue-600">
                        {Math.round(overview.intelligence_tiers.sonnet.utilization * 100)}%
                      </div>
                      <div className="text-sm text-gray-500">Sonnet Utilization</div>
                    </div>
                  </div>
                  
                  <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded">
                    <div className="text-2xl font-bold text-green-600">
                      {Math.round(overview.intelligence_tiers.haiku.utilization * 100)}%
                    </div>
                    <div className="text-sm text-gray-500">Haiku Utilization</div>
                  </div>

                  <div className="pt-4 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Peak Hours Cost</span>
                      <span className="font-medium">+15% Premium</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Off-Peak Discount</span>
                      <span className="font-medium text-green-600">-10% Savings</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold">Optimization Recommendations</h3>
              <p className="text-sm text-gray-600">
                Total potential savings: <span className="font-bold text-green-600">{formatCurrency(totalPotentialSavings)}</span>
              </p>
            </div>
            <Select value={sortBy} onValueChange={setSortBy}>
              <option value="savings">Sort by Savings</option>
              <option value="impact">Sort by Impact</option>
              <option value="effort">Sort by Effort</option>
            </Select>
          </div>

          <div className="space-y-4">
            {sortedRecommendations.map((rec) => (
              <Card key={rec.id}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        {getRecommendationIcon(rec.type)}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold mb-2">{rec.title}</h4>
                        <p className="text-gray-600 dark:text-gray-400 mb-3">{rec.description}</p>
                        <div className="flex items-center gap-4 text-sm">
                          <span>Implementation: {rec.implementation_time}</span>
                          <span>Affected: {rec.affected_agents.length} agents</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right space-y-2">
                      <div className="text-2xl font-bold text-green-600">
                        {formatCurrency(rec.potential_savings)}
                      </div>
                      <div className="space-x-2">
                        <Badge className={getEffortBadge(rec.effort_level)}>
                          {rec.effort_level} effort
                        </Badge>
                        <Badge variant="outline">
                          {rec.impact_score}/10 impact
                        </Badge>
                      </div>
                      <Button size="sm" className="w-full">
                        Implement
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Trends Tab */}
        <TabsContent value="trends" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Cost Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  Cost Trends
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded">
                    <span>This Month vs Last Month</span>
                    <span className="font-bold text-green-600">-12.5%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                    <span>Cost per Task Trend</span>
                    <span className="font-bold text-blue-600">-8.3%</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                    <span>Efficiency Improvement</span>
                    <span className="font-bold text-purple-600">+15.2%</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Projections */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-blue-600" />
                  Cost Projections
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Q1 2025 Forecast</span>
                      <span>{formatCurrency(overview.cost_analysis.estimated_monthly_cost * 3 * 0.95)}</span>
                    </div>
                    <Progress value={85} className="h-2" />
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span>Annual Projection</span>
                      <span>{formatCurrency(overview.cost_analysis.estimated_monthly_cost * 12 * 0.92)}</span>
                    </div>
                    <Progress value={75} className="h-2" />
                  </div>
                  
                  <div className="pt-4 text-center">
                    <div className="text-sm text-gray-500">
                      Projected savings with optimizations
                    </div>
                    <div className="text-2xl font-bold text-green-600">
                      {formatCurrency(totalPotentialSavings * 12)}
                    </div>
                    <div className="text-sm text-gray-500">annually</div>
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

export default CostDashboard;