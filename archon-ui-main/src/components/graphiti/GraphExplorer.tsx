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
const EntityNode = ({ data, selected }: { data: any; selected?: boolean }) => {
  const getNodeStyle = (entityType: string, isSelected: boolean = false) => {
    const baseStyle = {
      padding: '16px',
      borderRadius: '12px',
      border: '2px solid',
      background: 'white',
      minWidth: '140px',
      maxWidth: '180px',
      textAlign: 'center' as const,
      cursor: 'pointer',
      fontSize: '13px',
      boxShadow: isSelected 
        ? '0 8px 25px rgba(0, 0, 0, 0.15), 0 4px 10px rgba(0, 0, 0, 0.1)' 
        : '0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.06)',
      transform: isSelected ? 'scale(1.05)' : 'scale(1)',
      transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
      position: 'relative' as const,
      overflow: 'hidden' as const,
    };

    const typeStyles: Record<string, any> = {
      function: { 
        borderColor: '#3b82f6', 
        background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
        color: '#1e40af'
      },
      class: { 
        borderColor: '#10b981', 
        background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
        color: '#065f46'
      },
      module: { 
        borderColor: '#f59e0b', 
        background: 'linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)',
        color: '#92400e'
      },
      concept: { 
        borderColor: '#8b5cf6', 
        background: 'linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%)',
        color: '#6b21a8'
      },
      agent: { 
        borderColor: '#ef4444', 
        background: 'linear-gradient(135deg, #fef2f2 0%, #fecaca 100%)',
        color: '#991b1b'
      },
      project: { 
        borderColor: '#06b6d4', 
        background: 'linear-gradient(135deg, #f0fdff 0%, #cffafe 100%)',
        color: '#0e7490'
      },
      requirement: { 
        borderColor: '#84cc16', 
        background: 'linear-gradient(135deg, #f7fee7 0%, #ecfccb 100%)',
        color: '#365314'
      },
      pattern: { 
        borderColor: '#ec4899', 
        background: 'linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%)',
        color: '#be185d'
      },
    };

    const style = { ...baseStyle, ...typeStyles[entityType] };
    
    if (isSelected) {
      style.borderWidth = '3px';
      style.borderColor = typeStyles[entityType]?.borderColor || '#3b82f6';
    }

    return style;
  };

  const getEntityIcon = (entityType: string) => {
    const icons: Record<string, { icon: string; gradient: string }> = {
      function: { icon: '⚡', gradient: 'from-blue-500 to-blue-600' },
      class: { icon: '🏗️', gradient: 'from-green-500 to-green-600' },
      module: { icon: '📦', gradient: 'from-amber-500 to-amber-600' },
      concept: { icon: '💡', gradient: 'from-purple-500 to-purple-600' },
      agent: { icon: '🤖', gradient: 'from-red-500 to-red-600' },
      project: { icon: '📋', gradient: 'from-cyan-500 to-cyan-600' },
      requirement: { icon: '📝', gradient: 'from-lime-500 to-lime-600' },
      pattern: { icon: '🎯', gradient: 'from-pink-500 to-pink-600' },
    };
    return icons[entityType] || { icon: '📄', gradient: 'from-gray-500 to-gray-600' };
  };

  const entityInfo = getEntityIcon(data.entity_type);
  const confidenceScore = (data.confidence_score * 100).toFixed(0);
  const isHighConfidence = data.confidence_score > 0.8;

  return (
    <div style={getNodeStyle(data.entity_type, selected)} className="group">
      {/* Confidence indicator */}
      <div 
        style={{
          position: 'absolute',
          top: '6px',
          right: '6px',
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: isHighConfidence ? '#10b981' : data.confidence_score > 0.6 ? '#f59e0b' : '#ef4444',
          opacity: 0.8
        }}
        title={`Confidence: ${confidenceScore}%`}
      />
      
      {/* Icon with gradient background */}
      <div 
        style={{
          fontSize: '24px',
          marginBottom: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '40px',
          height: '40px',
          margin: '0 auto 8px',
          borderRadius: '10px',
          background: `linear-gradient(135deg, ${
            data.entity_type === 'function' ? '#3b82f6, #1d4ed8' :
            data.entity_type === 'class' ? '#10b981, #059669' :
            data.entity_type === 'module' ? '#f59e0b, #d97706' :
            data.entity_type === 'concept' ? '#8b5cf6, #7c3aed' :
            data.entity_type === 'agent' ? '#ef4444, #dc2626' :
            data.entity_type === 'project' ? '#06b6d4, #0891b2' :
            data.entity_type === 'requirement' ? '#84cc16, #65a30d' :
            data.entity_type === 'pattern' ? '#ec4899, #db2777' :
            '#6b7280, #4b5563'
          })`,
          color: 'white',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        }}
      >
        {entityInfo.icon}
      </div>
      
      {/* Entity name */}
      <div 
        style={{ 
          fontWeight: '600', 
          marginBottom: '4px',
          fontSize: '14px',
          lineHeight: '1.2',
          wordWrap: 'break-word',
          hyphens: 'auto'
        }}
      >
        {data.name}
      </div>
      
      {/* Entity type badge */}
      <div 
        style={{ 
          fontSize: '11px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          fontWeight: '500',
          opacity: 0.8,
          marginBottom: '6px'
        }}
      >
        {data.entity_type}
      </div>
      
      {/* Confidence score with visual indicator */}
      <div 
        style={{ 
          fontSize: '11px',
          opacity: 0.7,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '4px'
        }}
      >
        <div
          style={{
            width: '30px',
            height: '3px',
            background: '#e5e7eb',
            borderRadius: '2px',
            overflow: 'hidden'
          }}
        >
          <div
            style={{
              width: `${data.confidence_score * 100}%`,
              height: '100%',
              background: isHighConfidence ? '#10b981' : data.confidence_score > 0.6 ? '#f59e0b' : '#ef4444',
              borderRadius: '2px',
              transition: 'width 0.3s ease'
            }}
          />
        </div>
        <span>{confidenceScore}%</span>
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

    // Create React Flow edges with enhanced styling
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const flowEdges: Edge[] = graphData.edges
      .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map(edge => {
        const confidence = edge.properties.confidence || 0.8;
        const isHighConfidence = confidence > 0.8;
        const isMediumConfidence = confidence > 0.6;
        
        // Enhanced edge styling based on confidence
        const strokeWidth = isHighConfidence ? 3 : isMediumConfidence ? 2 : 1.5;
        const strokeColor = isHighConfidence 
          ? 'rgba(16, 185, 129, 0.8)' 
          : isMediumConfidence 
            ? 'rgba(59, 130, 246, 0.7)' 
            : 'rgba(156, 163, 175, 0.6)';
        
        const strokeDasharray = confidence < 0.5 ? '5,5' : undefined;
        
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: edge.type,
          type: 'smoothstep',
          style: { 
            strokeWidth,
            stroke: strokeColor,
            strokeDasharray,
          },
          labelStyle: { 
            fontSize: 11,
            fontWeight: '600',
            fill: '#1f2937',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            padding: '2px 6px',
            borderRadius: '4px',
            border: '1px solid rgba(0, 0, 0, 0.1)',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          },
          labelBgPadding: [8, 4],
          labelBgBorderRadius: 4,
          labelBgStyle: {
            fill: 'rgba(255, 255, 255, 0.95)',
            fillOpacity: 0.95,
            stroke: 'rgba(0, 0, 0, 0.1)',
            strokeWidth: 1,
          },
          data: edge.properties,
        };
      });

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
    <div className="w-full h-full flex flex-col bg-gradient-to-br from-slate-50 to-blue-50/30">
      {/* Header */}
      <div className="flex items-center justify-between p-6 bg-white/95 backdrop-blur-sm border-b border-gray-200/60 shadow-sm">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <GitBranch className="h-6 w-6 text-blue-600" />
              <div className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 opacity-80" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 via-blue-900 to-purple-900 bg-clip-text text-transparent">
                Graphiti Explorer
              </h1>
              <div className="text-xs text-gray-500 mt-0.5">Knowledge Graph Visualization</div>
            </div>
            {/* Enhanced Service Health Indicator */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                serviceHealth === 'healthy' ? 'bg-green-500 animate-pulse' : 
                serviceHealth === 'unhealthy' ? 'bg-red-500' : 'bg-gray-400'
              }`} title={`Service Status: ${serviceHealth}`} />
              <span className={`text-xs font-medium ${
                serviceHealth === 'healthy' ? 'text-green-700' : 
                serviceHealth === 'unhealthy' ? 'text-red-700' : 'text-gray-500'
              }`}>
                {serviceHealth}
              </span>
            </div>
          </div>
          {graphData && (
            <div className="flex items-center space-x-3">
              <Badge 
                variant="secondary" 
                className="bg-blue-100 text-blue-800 border-blue-200 font-medium px-3 py-1"
              >
                {graphData.metadata.total_entities} entities
              </Badge>
              <Badge 
                variant="secondary" 
                className="bg-purple-100 text-purple-800 border-purple-200 font-medium px-3 py-1"
              >
                {graphData.metadata.total_relationships} relationships
              </Badge>
            </div>
          )}
        </div>
        
        <div className="flex items-center space-x-3">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowTemporalFilter(true)}
            disabled={serviceHealth === 'unhealthy'}
            className="bg-white/50 backdrop-blur-sm border-gray-300 hover:bg-white/80 hover:border-blue-300 transition-all duration-200"
          >
            <Clock className="h-4 w-4 mr-2" />
            Time Filter
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={exportGraph}
            disabled={!graphData}
            className="bg-white/50 backdrop-blur-sm border-gray-300 hover:bg-white/80 hover:border-green-300 transition-all duration-200"
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => loadGraphData()}
            disabled={isLoading}
            className="bg-white/50 backdrop-blur-sm border-gray-300 hover:bg-white/80 hover:border-purple-300 transition-all duration-200"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-lg p-4 mb-4 mx-6 mt-4 shadow-sm">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <AlertCircle className="h-5 w-5 text-red-500" />
            </div>
            <div className="ml-3">
              <div className="text-sm font-medium text-red-800">
                Error loading graph data
              </div>
              <div className="text-sm text-red-700 mt-1">
                {error}
              </div>
            </div>
            <div className="ml-auto pl-3">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setError(null)}
                className="text-red-600 hover:text-red-800"
              >
                Dismiss
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Controls */}
      <div className="flex items-center justify-between p-4 bg-white/80 backdrop-blur-sm border-b border-gray-200/60">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-white rounded-lg px-3 py-2 shadow-sm border">
            <Search className="h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search entities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-64 border-0 focus:ring-0 bg-transparent placeholder:text-gray-400"
            />
          </div>
          
          <Select value={entityTypeFilter} onValueChange={setEntityTypeFilter}>
            <SelectTrigger className="w-48 bg-white shadow-sm border-gray-200">
              <SelectValue placeholder="Filter by type" />
            </SelectTrigger>
            <SelectContent>
              {graphExplorerService.getEntityTypeOptions().map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {timeFilter && (
            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
              Time filter active
            </Badge>
          )}
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-white rounded-lg px-3 py-2 shadow-sm border">
            <Switch
              checked={autoLayout}
              onCheckedChange={setAutoLayout}
            />
            <span className="text-sm text-gray-700 font-medium">Auto Layout</span>
          </div>

          <Button 
            variant="outline" 
            size="sm" 
            onClick={resetFilters}
            className="bg-white shadow-sm border-gray-200 hover:bg-gray-50"
          >
            <Filter className="h-4 w-4 mr-1" />
            Clear Filters
          </Button>
        </div>
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
              className="bg-gradient-to-br from-slate-100/50 to-blue-100/30"
              proOptions={{ hideAttribution: true }}
            >
              <Controls 
                className="bg-white/90 backdrop-blur-sm border border-gray-200/60 rounded-lg shadow-lg"
                showZoom={true}
                showFitView={true}
                showInteractive={true}
              />
              <MiniMap 
                className="bg-white/90 backdrop-blur-sm border border-gray-200/60 rounded-lg shadow-lg"
                maskColor="rgba(0, 0, 0, 0.1)"
                nodeColor={(node) => {
                  const typeColors: Record<string, string> = {
                    function: '#3b82f6',
                    class: '#10b981',
                    module: '#f59e0b',
                    concept: '#8b5cf6',
                    agent: '#ef4444',
                    project: '#06b6d4',
                    requirement: '#84cc16',
                    pattern: '#ec4899',
                  };
                  return typeColors[node.data?.entity_type] || '#6b7280';
                }}
              />
              <Background 
                variant="dots" 
                gap={20} 
                size={1.2} 
                color="rgba(0, 0, 0, 0.05)"
              />
              
              {/* Enhanced Performance Stats Panel */}
              <Panel position="top-right" className="bg-white/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-gray-200/60">
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Activity className="h-4 w-4 text-blue-600" />
                    <div className="font-semibold text-gray-900 text-sm">Graph Stats</div>
                  </div>
                  <div className="text-xs space-y-1.5">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Visible Nodes:</span>
                      <span className="font-medium text-gray-900">{nodes.length} / {graphData?.metadata.total_entities || 0}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Visible Edges:</span>
                      <span className="font-medium text-gray-900">{edges.length} / {graphData?.metadata.total_relationships || 0}</span>
                    </div>
                    {timeFilter && (
                      <div className="text-blue-600 text-xs font-medium pt-1 border-t border-blue-100">
                        📅 Time filter active
                      </div>
                    )}
                    {isLoading && (
                      <div className="text-orange-600 text-xs font-medium pt-1 border-t border-orange-100 flex items-center">
                        <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                        Loading...
                      </div>
                    )}
                  </div>
                </div>
              </Panel>

              {/* Legend Panel */}
              <Panel position="bottom-left" className="bg-white/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-gray-200/60">
                <div className="space-y-2">
                  <div className="font-semibold text-gray-900 text-sm mb-3">Entity Types</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries({
                      function: { color: '#3b82f6', icon: '⚡' },
                      class: { color: '#10b981', icon: '🏗️' },
                      module: { color: '#f59e0b', icon: '📦' },
                      concept: { color: '#8b5cf6', icon: '💡' },
                      agent: { color: '#ef4444', icon: '🤖' },
                      project: { color: '#06b6d4', icon: '📋' },
                      requirement: { color: '#84cc16', icon: '📝' },
                      pattern: { color: '#ec4899', icon: '🎯' },
                    }).map(([type, { color, icon }]) => (
                      <div key={type} className="flex items-center space-x-2">
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-gray-700 capitalize">{type}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </Panel>
            </ReactFlow>
          </ReactFlowProvider>
        </div>

        {/* Enhanced Side Panel */}
        <div className="w-80 bg-white/95 backdrop-blur-sm border-l border-gray-200/60 overflow-y-auto">
          <Tabs defaultValue="details" className="h-full">
            <TabsList className="grid w-full grid-cols-2 bg-gray-100/80 backdrop-blur-sm">
              <TabsTrigger 
                value="details" 
                className="data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                Details
              </TabsTrigger>
              <TabsTrigger 
                value="stats" 
                className="data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                Statistics
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="details" className="p-6 space-y-4">
              {selectedEntity ? (
                <div className="space-y-4">
                  <EntityDetails entity={selectedEntity} />
                  
                  {/* Enhanced details from API if available */}
                  {selectedEntityDetails && (
                    <Card className="border-gray-200/60 shadow-sm">
                      <CardHeader className="pb-3">
                        <CardTitle className="text-sm font-semibold text-gray-800 flex items-center">
                          <GitBranch className="h-4 w-4 mr-2 text-blue-600" />
                          Related Entities
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {selectedEntityDetails.related_entities.length > 0 ? (
                          selectedEntityDetails.related_entities.map((related, index) => (
                            <div key={index} className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-blue-50/30 rounded-lg border border-gray-200/40">
                              <div className="flex-1">
                                <div className="font-medium text-sm text-gray-900">{related.entity.name}</div>
                                <div className="text-xs text-gray-500 capitalize">{related.entity.entity_type}</div>
                              </div>
                              <div className="text-right">
                                <div className="text-xs font-medium text-blue-700 bg-blue-100 px-2 py-1 rounded-full mb-1">
                                  {related.relationship.relationship_type}
                                </div>
                                <div className="text-xs text-gray-600">
                                  {Math.round(related.relationship.confidence * 100)}% confidence
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          <div className="text-center py-8">
                            <div className="text-gray-400 mb-2">🔍</div>
                            <p className="text-sm text-gray-500">No related entities found</p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}
                </div>
              ) : (
                <Card className="border-gray-200/60 shadow-sm">
                  <CardContent className="p-8 text-center">
                    <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-gray-100 to-blue-100 rounded-full flex items-center justify-center">
                      <Database className="h-8 w-8 text-gray-400" />
                    </div>
                    <div className="space-y-2">
                      <p className="font-medium text-gray-700">Select an Entity</p>
                      <p className="text-sm text-gray-500">Click on any node in the graph to view detailed information</p>
                    </div>
                    {serviceHealth === 'unhealthy' && (
                      <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
                        <p className="text-xs text-red-600 font-medium">Service unavailable - limited functionality</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </TabsContent>
            
            <TabsContent value="stats" className="p-6">
              {graphData ? (
                <GraphStats data={graphData} />
              ) : (
                <Card className="border-gray-200/60 shadow-sm">
                  <CardContent className="p-8 text-center">
                    <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-gray-100 to-purple-100 rounded-full flex items-center justify-center">
                      <Activity className="h-8 w-8 text-gray-400" />
                    </div>
                    <div className="space-y-2">
                      <p className="font-medium text-gray-700">No Statistics Available</p>
                      <p className="text-sm text-gray-500">Load graph data to view statistics</p>
                    </div>
                  </CardContent>
                </Card>
              )}
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