import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  Connection,
  Panel,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Search, 
  Filter, 
  Download, 
  Maximize2, 
  RefreshCw, 
  Clock,
  Database,
  Activity,
  GitBranch,
  AlertCircle
} from 'lucide-react';
import { Input } from '@/components/ui/Input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select';
import { Switch } from '@/components/ui/switch';
import { TemporalFilter } from './TemporalFilter';
import { EntityDetails } from './EntityDetails';
import { GraphStats } from './GraphStats';
import graphExplorerService, { 
  GraphData, 
  GraphFilters, 
  TemporalFilter as TemporalFilterType,
  EntityDetails as EntityDetailsType 
} from '@/services/graphExplorerService';

// Legacy interface for backward compatibility with existing entity display logic
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

// Custom node types for different entity types
const EntityNode = ({ data }: { data: any }) => {
  const getNodeStyle = (entityType: string) => {
    const baseStyle = {
      padding: '10px',
      borderRadius: '8px',
      border: '2px solid',
      background: 'white',
      minWidth: '120px',
      textAlign: 'center' as const,
      cursor: 'pointer',
      fontSize: '12px',
    };

    const typeStyles: Record<string, any> = {
      function: { borderColor: '#3b82f6', background: '#eff6ff' },
      class: { borderColor: '#10b981', background: '#f0fdf4' },
      module: { borderColor: '#f59e0b', background: '#fffbeb' },
      concept: { borderColor: '#8b5cf6', background: '#faf5ff' },
      agent: { borderColor: '#ef4444', background: '#fef2f2' },
      project: { borderColor: '#06b6d4', background: '#f0fdff' },
      requirement: { borderColor: '#84cc16', background: '#f7fee7' },
      pattern: { borderColor: '#ec4899', background: '#fdf2f8' },
    };

    return { ...baseStyle, ...typeStyles[entityType] };
  };

  const getEntityIcon = (entityType: string) => {
    const icons: Record<string, string> = {
      function: '⚡',
      class: '🏗️',
      module: '📦',
      concept: '💡',
      agent: '🤖',
      project: '📋',
      requirement: '📝',
      pattern: '🎯',
    };
    return icons[entityType] || '📄';
  };

  return (
    <div style={getNodeStyle(data.entity_type)}>
      <div style={{ fontSize: '16px', marginBottom: '4px' }}>
        {getEntityIcon(data.entity_type)}
      </div>
      <div style={{ fontWeight: 'bold', marginBottom: '2px' }}>
        {data.name}
      </div>
      <div style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase' }}>
        {data.entity_type}
      </div>
      <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
        Score: {(data.confidence_score * 100).toFixed(0)}%
      </div>
    </div>
  );
};

const nodeTypes = {
  entity: EntityNode,
};

// Main GraphExplorer component
export const GraphExplorer: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedEntityDetails, setSelectedEntityDetails] = useState<EntityDetailsType | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<GraphEntity | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [entityTypeFilter, setEntityTypeFilter] = useState<string>('all');
  const [timeFilter, setTimeFilter] = useState<TemporalFilterType | null>(null);
  const [showTemporalFilter, setShowTemporalFilter] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoLayout, setAutoLayout] = useState(true);
  const [serviceHealth, setServiceHealth] = useState<'unknown' | 'healthy' | 'unhealthy'>('unknown');

  // Load graph data from backend
  const loadGraphData = useCallback(async (filters?: GraphFilters) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Apply current UI filters to the request
      const requestFilters: GraphFilters = {
        ...filters,
        search_term: searchTerm || undefined,
        entity_types: entityTypeFilter !== 'all' ? [entityTypeFilter] : undefined,
        limit: 100
      };

      // Use real service to fetch graph data
      const data = await graphExplorerService.getGraphData(requestFilters);
      setGraphData(data);
      setServiceHealth('healthy');
      
    } catch (error) {
      console.error('Failed to load graph data:', error);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
      setServiceHealth('unhealthy');
    } finally {
      setIsLoading(false);
    }
  }, [searchTerm, entityTypeFilter]);

  // Convert graph data to React Flow format
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!graphData) return { flowNodes: [], flowEdges: [] };

    // Apply additional client-side filtering (service handles most filtering)
    let filteredNodes = graphData.nodes;
    
    // Apply temporal filter if present (supplement server-side filtering)
    if (timeFilter) {
      filteredNodes = filteredNodes.filter(node => 
        node.properties.creation_time >= timeFilter.start_time && 
        node.properties.creation_time <= timeFilter.end_time
      );
    }

    // Create React Flow nodes
    const flowNodes: Node[] = filteredNodes.map((node, index) => ({
      id: node.id,
      type: 'entity',
      position: node.position || (autoLayout ? 
        { x: (index % 4) * 200, y: Math.floor(index / 4) * 150 } : 
        { x: Math.random() * 800, y: Math.random() * 600 }),
      data: {
        // Convert to legacy format for existing node component
        entity_id: node.id,
        entity_type: node.type,
        name: node.label,
        attributes: node.properties.attributes || {},
        creation_time: node.properties.creation_time,
        modification_time: node.properties.modification_time,
        access_frequency: node.properties.access_frequency,
        confidence_score: node.properties.confidence_score,
        importance_weight: node.properties.importance_weight,
        tags: node.properties.tags || []
      },
    }));

    // Create React Flow edges
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const flowEdges: Edge[] = graphData.edges
      .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.type,
        type: 'smoothstep',
        style: { 
          strokeWidth: 2, 
          stroke: `rgba(59, 130, 246, ${edge.properties.confidence || 0.8})`,
        },
        labelStyle: { 
          fontSize: 10, 
          fontWeight: 'bold',
          fill: '#374151'
        },
        data: edge.properties,
      }));

    return { flowNodes, flowEdges };
  }, [graphData, timeFilter, autoLayout]);

  // Update React Flow when data changes
  useEffect(() => {
    setNodes(flowNodes);
    setEdges(flowEdges);
  }, [flowNodes, flowEdges, setNodes, setEdges]);

  // Handle node click
  const onNodeClick = useCallback(async (event: React.MouseEvent, node: Node) => {
    const entity = node.data as GraphEntity;
    setSelectedEntity(entity);
    
    // Fetch detailed entity information
    try {
      const details = await graphExplorerService.getEntityDetails(entity.entity_id);
      setSelectedEntityDetails(details);
    } catch (error) {
      console.error('Failed to fetch entity details:', error);
      // Still show basic entity info even if details fetch fails
      setSelectedEntityDetails(null);
    }
  }, []);

  // Handle edge connect
  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Apply temporal filter
  const applyTemporalFilter = useCallback(async (filter: TemporalFilterType) => {
    setTimeFilter(filter);
    setShowTemporalFilter(false);
    
    try {
      setIsLoading(true);
      const data = await graphExplorerService.applyTemporalFilter(filter);
      setGraphData(data);
      setServiceHealth('healthy');
    } catch (error) {
      console.error('Failed to apply temporal filter:', error);
      setError(error instanceof Error ? error.message : 'Failed to apply temporal filter');
      setServiceHealth('unhealthy');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Reset filters
  const resetFilters = useCallback(() => {
    setSearchTerm('');
    setEntityTypeFilter('all');
    setTimeFilter(null);
    setError(null);
    // Reload data with no filters
    loadGraphData();
  }, [loadGraphData]);

  // Load data on component mount and setup WebSocket listeners
  useEffect(() => {
    // Initial data load
    loadGraphData();
    
    // Check service health
    const checkHealth = async () => {
      try {
        const health = await graphExplorerService.healthCheck();
        setServiceHealth(health.status === 'healthy' ? 'healthy' : 'unhealthy');
      } catch (error) {
        setServiceHealth('unhealthy');
      }
    };
    checkHealth();
    
    // Setup WebSocket listener for real-time updates
    const unsubscribe = graphExplorerService.onGraphUpdate((updatedData) => {
      console.log('Received real-time graph update');
      setGraphData(updatedData);
    });
    
    // Cleanup on unmount
    return () => {
      unsubscribe();
    };
  }, [loadGraphData]);
  
  // Handle search term changes with debouncing
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchTerm !== '') {
        loadGraphData();
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [searchTerm, loadGraphData]);
  
  // Handle entity type filter changes
  useEffect(() => {
    loadGraphData();
  }, [entityTypeFilter, loadGraphData]);

  // Export graph data function
  const exportGraph = useCallback(() => {
    try {
      graphExplorerService.exportGraphData();
    } catch (error) {
      console.error('Failed to export graph data:', error);
      setError('Failed to export graph data. No data available.');
    }
  }, []);

  return (
    <div className="w-full h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-white border-b shadow-sm">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <GitBranch className="h-5 w-5 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-900">Graphiti Explorer</h1>
            {/* Service Health Indicator */}
            <div className={`w-2 h-2 rounded-full ${
              serviceHealth === 'healthy' ? 'bg-green-500' : 
              serviceHealth === 'unhealthy' ? 'bg-red-500' : 'bg-gray-400'
            }`} title={`Service Status: ${serviceHealth}`} />
          </div>
          {graphData && (
            <Badge variant="secondary">
              {graphData.metadata.total_entities} entities • {graphData.metadata.total_relationships} relationships
            </Badge>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowTemporalFilter(true)}
            disabled={serviceHealth === 'unhealthy'}
          >
            <Clock className="h-4 w-4 mr-1" />
            Time Filter
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={exportGraph}
            disabled={!graphData}
          >
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => loadGraphData()}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4 mx-4 mt-4">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
            <div className="text-sm text-red-700">
              <p className="font-medium">Error loading graph data</p>
              <p>{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center space-x-4 p-4 bg-white border-b">
        <div className="flex items-center space-x-2">
          <Search className="h-4 w-4 text-gray-500" />
          <Input
            placeholder="Search entities..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-64"
          />
        </div>
        
        <Select value={entityTypeFilter} onValueChange={setEntityTypeFilter}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="Entity Type" />
          </SelectTrigger>
          <SelectContent>
            {graphExplorerService.getEntityTypeOptions().map(option => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex items-center space-x-2">
          <Switch
            checked={autoLayout}
            onCheckedChange={setAutoLayout}
          />
          <span className="text-sm text-gray-600">Auto Layout</span>
        </div>

        <Button variant="outline" size="sm" onClick={resetFilters}>
          <Filter className="h-4 w-4 mr-1" />
          Clear Filters
        </Button>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Graph View */}
        <div className="flex-1 relative">
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              fitView
              className="bg-gray-50"
            >
              <Controls />
              <MiniMap />
              <Background />
              
              {/* Performance Stats Panel */}
              <Panel position="top-right" className="bg-white p-3 rounded-lg shadow-lg border">
                <div className="text-xs space-y-1">
                  <div className="font-medium text-gray-900">Performance</div>
                  <div className="text-gray-600">
                    Nodes: {nodes.length} / {graphData?.metadata.total_entities || 0}
                  </div>
                  <div className="text-gray-600">
                    Edges: {edges.length} / {graphData?.metadata.total_relationships || 0}
                  </div>
                  {timeFilter && (
                    <div className="text-blue-600 text-xs">
                      Time filter active
                    </div>
                  )}
                </div>
              </Panel>
            </ReactFlow>
          </ReactFlowProvider>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-white border-l overflow-y-auto">
          <Tabs defaultValue="details" className="h-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="details">Details</TabsTrigger>
              <TabsTrigger value="stats">Statistics</TabsTrigger>
            </TabsList>
            
            <TabsContent value="details" className="p-4 space-y-4">
              {selectedEntity ? (
                <div className="space-y-4">
                  <EntityDetails entity={selectedEntity} />
                  
                  {/* Enhanced details from API if available */}
                  {selectedEntityDetails && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-sm font-medium">Related Entities</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        {selectedEntityDetails.related_entities.length > 0 ? (
                          selectedEntityDetails.related_entities.map((related, index) => (
                            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                              <div>
                                <div className="font-medium text-sm">{related.entity.name}</div>
                                <div className="text-xs text-gray-500">{related.entity.entity_type}</div>
                              </div>
                              <div className="text-right">
                                <div className="text-xs text-blue-600">{related.relationship.relationship_type}</div>
                                <div className="text-xs text-gray-500">
                                  {Math.round(related.relationship.confidence * 100)}% confidence
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <p className="text-sm text-gray-500">No related entities found</p>
                        )}
                      </CardContent>
                    </Card>
                  )}
                </div>
              ) : (
                <Card>
                  <CardContent className="p-6 text-center text-gray-500">
                    <Database className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                    <p>Click on an entity to view details</p>
                    {serviceHealth === 'unhealthy' && (
                      <p className="text-xs text-red-500 mt-2">Service unavailable - limited functionality</p>
                    )}
                  </CardContent>
                </Card>
              )}
            </TabsContent>
            
            <TabsContent value="stats" className="p-4">
              {graphData && <GraphStats data={graphData} />}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Temporal Filter Modal */}
      {showTemporalFilter && (
        <TemporalFilter
          onApply={applyTemporalFilter}
          onClose={() => setShowTemporalFilter(false)}
          currentFilter={timeFilter}
        />
      )}
    </div>
  );
};