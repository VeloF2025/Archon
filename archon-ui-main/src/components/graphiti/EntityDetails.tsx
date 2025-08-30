import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/Button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  Code, 
  Clock, 
  Activity, 
  Tag, 
  GitBranch, 
  FileText, 
  Users, 
  Star,
  TrendingUp,
  Link,
  ExternalLink
} from 'lucide-react';

interface GraphEntity {
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
}

interface RelatedEntity {
  entity: GraphEntity;
  relationship_type: string;
  confidence: number;
  direction: 'incoming' | 'outgoing';
}

interface EntityDetailsProps {
  entity: GraphEntity;
}

export const EntityDetails: React.FC<EntityDetailsProps> = ({ entity }) => {
  const [relatedEntities, setRelatedEntities] = useState<RelatedEntity[]>([]);
  const [isLoadingRelated, setIsLoadingRelated] = useState(false);

  // Load related entities
  useEffect(() => {
    const loadRelatedEntities = async () => {
      setIsLoadingRelated(true);
      try {
        // Mock related entities - replace with actual API call
        const mockRelated: RelatedEntity[] = [
          {
            entity: {
              entity_id: 'class_user_manager',
              entity_type: 'class',
              name: 'UserManager',
              attributes: {},
              creation_time: Date.now() - 172800000,
              modification_time: Date.now() - 7200000,
              access_frequency: 8,
              confidence_score: 0.92,
              importance_weight: 0.9,
              tags: ['user', 'management']
            },
            relationship_type: 'calls',
            confidence: 0.89,
            direction: 'outgoing'
          },
          {
            entity: {
              entity_id: 'agent_code_implementer',
              entity_type: 'agent',
              name: 'code-implementer',
              attributes: {},
              creation_time: Date.now() - 259200000,
              modification_time: Date.now() - 1800000,
              access_frequency: 25,
              confidence_score: 0.97,
              importance_weight: 0.95,
              tags: ['agent', 'implementation']
            },
            relationship_type: 'implements',
            confidence: 0.92,
            direction: 'incoming'
          }
        ];
        
        setRelatedEntities(mockRelated);
        
        // TODO: Replace with actual API call
        // const response = await fetch(`/api/graphiti/entities/${entity.entity_id}/related`);
        // const data = await response.json();
        // setRelatedEntities(data);
        
      } catch (error) {
        console.error('Failed to load related entities:', error);
      } finally {
        setIsLoadingRelated(false);
      }
    };

    loadRelatedEntities();
  }, [entity.entity_id]);

  // Get entity icon
  const getEntityIcon = (entityType: string) => {
    const icons: Record<string, React.ReactNode> = {
      function: <Code className="h-4 w-4" />,
      class: <FileText className="h-4 w-4" />,
      module: <GitBranch className="h-4 w-4" />,
      concept: <Star className="h-4 w-4" />,
      agent: <Users className="h-4 w-4" />,
      project: <Activity className="h-4 w-4" />,
      requirement: <FileText className="h-4 w-4" />,
      pattern: <TrendingUp className="h-4 w-4" />,
    };
    return icons[entityType] || <FileText className="h-4 w-4" />;
  };

  // Get entity color
  const getEntityColor = (entityType: string): string => {
    const colors: Record<string, string> = {
      function: 'bg-blue-100 text-blue-800',
      class: 'bg-green-100 text-green-800',
      module: 'bg-yellow-100 text-yellow-800',
      concept: 'bg-purple-100 text-purple-800',
      agent: 'bg-red-100 text-red-800',
      project: 'bg-cyan-100 text-cyan-800',
      requirement: 'bg-lime-100 text-lime-800',
      pattern: 'bg-pink-100 text-pink-800',
    };
    return colors[entityType] || 'bg-gray-100 text-gray-800';
  };

  // Format time ago
  const formatTimeAgo = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const minutes = Math.floor(diff / (60 * 1000));
    const hours = Math.floor(diff / (60 * 60 * 1000));
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));

    if (days > 0) return `${days} day${days !== 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
  };

  // Render attribute value
  const renderAttributeValue = (key: string, value: any): React.ReactNode => {
    if (typeof value === 'boolean') {
      return <Badge variant={value ? 'default' : 'secondary'}>{value ? 'Yes' : 'No'}</Badge>;
    }
    
    if (Array.isArray(value)) {
      return (
        <div className="flex flex-wrap gap-1">
          {value.map((item, index) => (
            <Badge key={index} variant="outline" className="text-xs">
              {String(item)}
            </Badge>
          ))}
        </div>
      );
    }

    if (typeof value === 'object' && value !== null) {
      return <code className="text-xs bg-gray-100 px-1 rounded">{JSON.stringify(value)}</code>;
    }

    // Handle file paths
    if (key === 'file_path' && typeof value === 'string') {
      return (
        <div className="flex items-center space-x-1">
          <code className="text-xs bg-gray-100 px-1 rounded">{value}</code>
          <ExternalLink className="h-3 w-3 text-gray-400" />
        </div>
      );
    }

    return <span className="text-sm text-gray-900">{String(value)}</span>;
  };

  return (
    <div className="space-y-4">
      {/* Entity Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-2">
              {getEntityIcon(entity.entity_type)}
              <div>
                <CardTitle className="text-lg">{entity.name}</CardTitle>
                <Badge className={`mt-1 ${getEntityColor(entity.entity_type)}`}>
                  {entity.entity_type}
                </Badge>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">Confidence</div>
              <div className="text-lg font-semibold">
                {(entity.confidence_score * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="pt-0">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="flex items-center space-x-1 text-gray-500 mb-1">
                <Clock className="h-3 w-3" />
                <span>Created</span>
              </div>
              <div>{formatTimeAgo(entity.creation_time)}</div>
            </div>
            <div>
              <div className="flex items-center space-x-1 text-gray-500 mb-1">
                <Activity className="h-3 w-3" />
                <span>Accessed</span>
              </div>
              <div>{entity.access_frequency} times</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Information */}
      <Tabs defaultValue="attributes" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="attributes">Attributes</TabsTrigger>
          <TabsTrigger value="relationships">Relations</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>

        {/* Attributes Tab */}
        <TabsContent value="attributes" className="space-y-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center space-x-2">
                <FileText className="h-4 w-4" />
                <span>Properties</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48">
                <div className="space-y-3">
                  {Object.entries(entity.attributes).length === 0 ? (
                    <p className="text-sm text-gray-500 italic">No additional attributes</p>
                  ) : (
                    Object.entries(entity.attributes).map(([key, value]) => (
                      <div key={key} className="flex flex-col space-y-1">
                        <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                          {key.replace(/_/g, ' ')}
                        </div>
                        {renderAttributeValue(key, value)}
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Tags */}
          {entity.tags.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center space-x-2">
                  <Tag className="h-4 w-4" />
                  <span>Tags</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {entity.tags.map((tag, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Relationships Tab */}
        <TabsContent value="relationships" className="space-y-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center space-x-2">
                <Link className="h-4 w-4" />
                <span>Related Entities</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-48">
                {isLoadingRelated ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                  </div>
                ) : relatedEntities.length === 0 ? (
                  <p className="text-sm text-gray-500 italic text-center py-8">
                    No related entities found
                  </p>
                ) : (
                  <div className="space-y-3">
                    {relatedEntities.map((related, index) => (
                      <div key={index} className="border rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            {getEntityIcon(related.entity.entity_type)}
                            <span className="font-medium text-sm">{related.entity.name}</span>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {related.relationship_type}
                          </Badge>
                        </div>
                        
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span className={`px-2 py-1 rounded-full ${getEntityColor(related.entity.entity_type)}`}>
                            {related.entity.entity_type}
                          </span>
                          <div className="flex items-center space-x-2">
                            <span className={
                              related.direction === 'incoming' ? 'text-green-600' : 'text-blue-600'
                            }>
                              {related.direction === 'incoming' ? '← incoming' : 'outgoing →'}
                            </span>
                            <span>({(related.confidence * 100).toFixed(0)}%)</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Timeline Tab */}
        <TabsContent value="timeline" className="space-y-3">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center space-x-2">
                <Clock className="h-4 w-4" />
                <span>Activity Timeline</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">Entity Created</div>
                    <div className="text-xs text-gray-500">
                      {new Date(entity.creation_time).toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      Initial confidence: {(entity.confidence_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                {entity.modification_time !== entity.creation_time && (
                  <div className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <div className="flex-1">
                      <div className="text-sm font-medium">Last Modified</div>
                      <div className="text-xs text-gray-500">
                        {new Date(entity.modification_time).toLocaleString()}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        Accessed {entity.access_frequency} times
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full mt-2"></div>
                  <div className="flex-1">
                    <div className="text-sm font-medium">Current Status</div>
                    <div className="text-xs text-gray-500">
                      Importance: {(entity.importance_weight * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      Active entity in the knowledge graph
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Actions */}
      <div className="flex space-x-2">
        <Button variant="outline" size="sm" className="flex-1">
          <ExternalLink className="h-4 w-4 mr-1" />
          View Source
        </Button>
        <Button variant="outline" size="sm" className="flex-1">
          <TrendingUp className="h-4 w-4 mr-1" />
          Analytics
        </Button>
      </div>
    </div>
  );
};