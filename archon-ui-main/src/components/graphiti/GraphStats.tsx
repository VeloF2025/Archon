import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  Database, 
  TrendingUp, 
  Clock, 
  Activity, 
  GitBranch,
  Users,
  Zap
} from 'lucide-react';

interface GraphData {
  entities: Array<{
    entity_id: string;
    entity_type: string;
    name: string;
    attributes: Record<string, any>;
    creation_time: number;
    modification_time: number;
    access_frequency: number;
    confidence_score: number;
    importance_weight: number;
    tags: string[];
  }>;
  relationships: Array<{
    relationship_id: string;
    source_id: string;
    target_id: string;
    relationship_type: string;
    confidence: number;
    creation_time: number;
    temporal_data: Record<string, any>;
  }>;
  metadata: {
    total_entities: number;
    total_relationships: number;
    entity_types: string[];
    relationship_types: string[];
    last_updated: number;
  };
}

interface GraphStatsProps {
  data: GraphData;
}

export const GraphStats: React.FC<GraphStatsProps> = ({ data }) => {
  // Calculate entity type distribution
  const entityTypeStats = data.metadata.entity_types.map(type => ({
    name: type,
    count: data.entities.filter(e => e.entity_type === type).length,
    percentage: (data.entities.filter(e => e.entity_type === type).length / data.entities.length) * 100
  }));

  // Calculate relationship type distribution
  const relationshipTypeStats = data.metadata.relationship_types.map(type => ({
    name: type,
    count: data.relationships.filter(r => r.relationship_type === type).length,
    percentage: (data.relationships.filter(r => r.relationship_type === type).length / data.relationships.length) * 100
  }));

  // Calculate confidence distribution
  const confidenceRanges = [
    { range: '90-100%', min: 0.9, max: 1.0 },
    { range: '80-89%', min: 0.8, max: 0.9 },
    { range: '70-79%', min: 0.7, max: 0.8 },
    { range: '60-69%', min: 0.6, max: 0.7 },
    { range: '<60%', min: 0.0, max: 0.6 }
  ];

  const confidenceStats = confidenceRanges.map(range => ({
    range: range.range,
    count: data.entities.filter(e => 
      e.confidence_score >= range.min && e.confidence_score < range.max
    ).length
  }));

  // Calculate activity trends (last 7 days)
  const last7Days = Array.from({ length: 7 }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (6 - i));
    return date;
  });

  const activityTrends = last7Days.map(date => {
    const dayStart = new Date(date).setHours(0, 0, 0, 0);
    const dayEnd = new Date(date).setHours(23, 59, 59, 999);
    
    const entitiesCreated = data.entities.filter(e => 
      e.creation_time >= dayStart && e.creation_time <= dayEnd
    ).length;

    const relationshipsCreated = data.relationships.filter(r => 
      r.creation_time >= dayStart && r.creation_time <= dayEnd
    ).length;

    return {
      date: date.toLocaleDateString('en-US', { weekday: 'short' }),
      entities: entitiesCreated,
      relationships: relationshipsCreated,
      total: entitiesCreated + relationshipsCreated
    };
  });

  // Calculate graph health metrics
  const avgConfidence = data.entities.reduce((sum, e) => sum + e.confidence_score, 0) / data.entities.length;
  const avgImportance = data.entities.reduce((sum, e) => sum + e.importance_weight, 0) / data.entities.length;
  const totalAccess = data.entities.reduce((sum, e) => sum + e.access_frequency, 0);
  const connectivityRatio = data.relationships.length / data.entities.length;

  // Colors for charts
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444', '#06b6d4', '#84cc16', '#ec4899'];

  // Format time ago
  const formatTimeAgo = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const hours = Math.floor(diff / (60 * 60 * 1000));
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    
    if (days > 0) return `${days} day${days !== 1 ? 's' : ''} ago`;
    return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
  };

  return (
    <div className="space-y-4">
      {/* Overview Cards */}
      <div className="grid grid-cols-2 gap-3">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4 text-blue-600" />
              <div>
                <div className="text-lg font-bold">{data.metadata.total_entities}</div>
                <div className="text-xs text-gray-500">Entities</div>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <GitBranch className="h-4 w-4 text-green-600" />
              <div>
                <div className="text-lg font-bold">{data.metadata.total_relationships}</div>
                <div className="text-xs text-gray-500">Relationships</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Health Metrics */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Graph Health</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span>Avg Confidence</span>
              <span>{(avgConfidence * 100).toFixed(1)}%</span>
            </div>
            <Progress value={avgConfidence * 100} className="h-2" />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span>Avg Importance</span>
              <span>{(avgImportance * 100).toFixed(1)}%</span>
            </div>
            <Progress value={avgImportance * 100} className="h-2" />
          </div>
          
          <div className="flex justify-between items-center text-xs">
            <span>Connectivity Ratio</span>
            <Badge variant="secondary">{connectivityRatio.toFixed(2)}</Badge>
          </div>
          
          <div className="flex justify-between items-center text-xs">
            <span>Total Access Count</span>
            <Badge variant="secondary">{totalAccess}</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Entity Types Distribution */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center space-x-2">
            <Users className="h-4 w-4" />
            <span>Entity Types</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={entityTypeStats}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={50}
                  paddingAngle={2}
                  dataKey="count"
                >
                  {entityTypeStats.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  formatter={(value: number, name: string) => [value, 'Count']}
                  labelFormatter={(label: string) => `${label} entities`}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-1 mt-2">
            {entityTypeStats.map((stat, index) => (
              <div key={stat.name} className="flex items-center space-x-1 text-xs">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="truncate">{stat.name}</span>
                <span className="text-gray-500">({stat.count})</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Activity Trends */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center space-x-2">
            <TrendingUp className="h-4 w-4" />
            <span>7-Day Activity</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activityTrends}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 10 }}
                  axisLine={false}
                />
                <YAxis 
                  tick={{ fontSize: 10 }}
                  axisLine={false}
                />
                <Tooltip 
                  formatter={(value: number, name: string) => [value, name === 'entities' ? 'Entities' : 'Relationships']}
                  labelFormatter={(label: string) => `${label}`}
                />
                <Bar dataKey="entities" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                <Bar dataKey="relationships" fill="#10b981" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Confidence Distribution */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center space-x-2">
            <Zap className="h-4 w-4" />
            <span>Confidence Ranges</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {confidenceStats.map((stat, index) => (
              <div key={stat.range} className="flex items-center justify-between text-xs">
                <span>{stat.range}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-gray-200 rounded-full h-1.5">
                    <div 
                      className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${data.entities.length > 0 ? (stat.count / data.entities.length) * 100 : 0}%` 
                      }}
                    />
                  </div>
                  <span className="text-gray-500 w-6">{stat.count}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Relationship Types */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center space-x-2">
            <GitBranch className="h-4 w-4" />
            <span>Relationship Types</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {relationshipTypeStats.map((stat, index) => (
              <div key={stat.name} className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-2 h-2 rounded-full" 
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="capitalize">{stat.name.replace('_', ' ')}</span>
                </div>
                <Badge variant="outline" className="text-xs">
                  {stat.count}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Last Updated */}
      <Card>
        <CardContent className="p-3">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center space-x-1">
              <Clock className="h-3 w-3" />
              <span>Last Updated</span>
            </div>
            <span>{formatTimeAgo(data.metadata.last_updated)}</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};