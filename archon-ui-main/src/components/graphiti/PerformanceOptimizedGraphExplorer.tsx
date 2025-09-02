/**
 * Performance Optimized Graphiti Explorer
 * Implements viewport culling, memoization, and performance monitoring
 * Target: <1.5s initial load, 60fps interactions
 */

import React, { useState, useEffect, useCallback, useMemo, memo } from 'react';
import {
  ReactFlow,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  Panel,
  ReactFlowProvider,
  useReactFlow,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { 
  Search, 
  Filter, 
  RefreshCw, 
  Activity,
  Zap,
  Clock
} from 'lucide-react';
import { Input } from '@/components/ui/Input';

// Performance monitoring
class PerformanceMonitor {
  private metrics: { [key: string]: number } = {};
  private frameCount = 0;
  private lastFrameTime = 0;

  startTiming(operation: string) {
    this.metrics[`${operation}_start`] = performance.now();
  }

  endTiming(operation: string) {
    const start = this.metrics[`${operation}_start`];
    if (start) {
      this.metrics[operation] = performance.now() - start;
      console.log(`⚡ ${operation}: ${this.metrics[operation].toFixed(2)}ms`);
    }
  }

  trackFPS() {
    const now = performance.now();
    if (this.lastFrameTime) {
      const fps = 1000 / (now - this.lastFrameTime);
      this.frameCount++;
      if (this.frameCount % 60 === 0) {
        console.log(`📊 FPS: ${fps.toFixed(1)}`);
      }
    }
    this.lastFrameTime = now;
    requestAnimationFrame(() => this.trackFPS());
  }

  getMetrics() {
    return this.metrics;
  }
}

const perfMonitor = new PerformanceMonitor();

// Optimized Entity Node with memoization
const OptimizedEntityNode = memo(({ data, selected }: { data: any; selected?: boolean }) => {
  const nodeStyle = useMemo(() => {
    const confidence = data.confidence_score || 0;
    const importance = data.importance_weight || 0;
    
    // Simplified styling for better performance
    return {
      padding: '12px',
      borderRadius: '8px',
      border: selected ? '3px solid #3b82f6' : '2px solid #e2e8f0',
      background: selected ? '#eff6ff' : 'white',
      minWidth: '120px',
      maxWidth: '160px',
      textAlign: 'center' as const,
      cursor: 'pointer',
      fontSize: '12px',
      boxShadow: selected 
        ? '0 4px 12px rgba(59, 130, 246, 0.15)' 
        : '0 2px 6px rgba(0, 0, 0, 0.08)',
      transform: selected ? 'scale(1.02)' : 'scale(1)',
      transition: 'all 0.15s ease',
      opacity: importance > 0.7 ? 1 : 0.85,
    };
  }, [data.confidence_score, data.importance_weight, selected]);

  return (
    <div style={nodeStyle}>
      <div style={{ 
        fontWeight: 600, 
        marginBottom: '4px',
        color: '#1f2937',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap'
      }}>
        {data.name || data.entity_id}
      </div>
      <div style={{ 
        fontSize: '10px', 
        color: '#6b7280',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
      }}>
        {data.entity_type}
      </div>
      {data.confidence_score > 0 && (
        <div style={{ 
          fontSize: '10px', 
          color: data.confidence_score > 0.8 ? '#059669' : '#d97706',
          marginTop: '2px'
        }}>
          {(data.confidence_score * 100).toFixed(0)}%
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for better memoization
  return (
    prevProps.data?.entity_id === nextProps.data?.entity_id &&
    prevProps.selected === nextProps.selected &&
    prevProps.data?.confidence_score === nextProps.data?.confidence_score &&
    prevProps.data?.name === nextProps.data?.name
  );
});

// Viewport culling hook
function useViewportCulling(nodes: Node[], edges: Edge[], viewportBounds: any) {
  return useMemo(() => {
    if (!viewportBounds) return { nodes, edges };

    perfMonitor.startTiming('viewport_culling');
    
    // Only render nodes within viewport + buffer
    const buffer = 200;
    const visibleNodes = nodes.filter(node => {
      const nodeX = node.position.x;
      const nodeY = node.position.y;
      return (
        nodeX >= viewportBounds.x - buffer &&
        nodeX <= viewportBounds.x + viewportBounds.width + buffer &&
        nodeY >= viewportBounds.y - buffer &&
        nodeY <= viewportBounds.y + viewportBounds.height + buffer
      );
    });

    // Only render edges connected to visible nodes
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const visibleEdges = edges.filter(edge => 
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    perfMonitor.endTiming('viewport_culling');

    return {
      nodes: visibleNodes,
      edges: visibleEdges
    };
  }, [nodes, edges, viewportBounds]);
}

// Performance statistics component
const PerformanceStats = memo(({ isVisible }: { isVisible: boolean }) => {
  const [stats, setStats] = useState<any>({});
  
  useEffect(() => {
    if (!isVisible) return;
    
    const interval = setInterval(() => {
      setStats(perfMonitor.getMetrics());
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <Card className="absolute top-4 right-4 z-10 min-w-48">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Performance
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 text-xs">
        <div>Render: {stats.render?.toFixed(1) || 0}ms</div>
        <div>Culling: {stats.viewport_culling?.toFixed(1) || 0}ms</div>
        <div>Layout: {stats.layout?.toFixed(1) || 0}ms</div>
      </CardContent>
    </Card>
  );
});

// Main optimized component
const GraphExplorerInner = () => {
  const reactFlow = useReactFlow();
  const [isLoading, setIsLoading] = useState(true);
  const [showPerfStats, setShowPerfStats] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [rawNodes, setRawNodes] = useState<Node[]>([]);
  const [rawEdges, setRawEdges] = useState<Edge[]>([]);
  const [viewportBounds, setViewportBounds] = useState<any>(null);

  // Track viewport changes for culling
  const onViewportChange = useCallback(() => {
    const viewport = reactFlow.getViewport();
    const bounds = {
      x: -viewport.x / viewport.zoom,
      y: -viewport.y / viewport.zoom,
      width: window.innerWidth / viewport.zoom,
      height: window.innerHeight / viewport.zoom
    };
    setViewportBounds(bounds);
  }, [reactFlow]);

  // Apply viewport culling
  const { nodes: culledNodes, edges: culledEdges } = useViewportCulling(rawNodes, rawEdges, viewportBounds);

  // Filtered nodes based on search
  const filteredNodes = useMemo(() => {
    perfMonitor.startTiming('search_filter');
    
    if (!searchTerm) {
      perfMonitor.endTiming('search_filter');
      return culledNodes;
    }
    
    const filtered = culledNodes.filter(node => 
      node.data?.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.data?.entity_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      node.data?.entity_id?.toLowerCase().includes(searchTerm.toLowerCase())
    );
    
    perfMonitor.endTiming('search_filter');
    return filtered;
  }, [culledNodes, searchTerm]);

  // Mock data generation for demo
  useEffect(() => {
    perfMonitor.startTiming('data_load');
    
    // Generate optimized mock data
    const generateMockData = () => {
      const nodeCount = 50; // Reduced for better performance
      const nodes: Node[] = [];
      const edges: Edge[] = [];

      // Generate nodes
      for (let i = 0; i < nodeCount; i++) {
        nodes.push({
          id: `node-${i}`,
          type: 'default',
          position: { 
            x: Math.random() * 1200 - 600, 
            y: Math.random() * 800 - 400 
          },
          data: {
            entity_id: `entity-${i}`,
            name: `Entity ${i}`,
            entity_type: ['PERSON', 'ORGANIZATION', 'CONCEPT', 'EVENT'][i % 4],
            confidence_score: Math.random(),
            importance_weight: Math.random(),
            attributes: {},
          },
        });
      }

      // Generate edges (fewer for performance)
      for (let i = 0; i < Math.min(nodeCount * 0.8, 40); i++) {
        const source = Math.floor(Math.random() * nodeCount);
        const target = Math.floor(Math.random() * nodeCount);
        if (source !== target) {
          edges.push({
            id: `edge-${i}`,
            source: `node-${source}`,
            target: `node-${target}`,
            type: 'default',
            style: { stroke: '#94a3b8', strokeWidth: 1 },
          });
        }
      }

      return { nodes, edges };
    };

    const { nodes, edges } = generateMockData();
    setRawNodes(nodes);
    setRawEdges(edges);
    setIsLoading(false);
    
    perfMonitor.endTiming('data_load');
    
    // Start FPS tracking
    setTimeout(() => {
      perfMonitor.trackFPS();
    }, 1000);
  }, []);

  // Performance-optimized node types
  const nodeTypes = useMemo(() => ({
    default: OptimizedEntityNode,
  }), []);

  const [nodes, , onNodesChange] = useNodesState(filteredNodes);
  const [edges, , onEdgesChange] = useEdgesState(culledEdges);

  // Update nodes when filtered data changes
  useEffect(() => {
    perfMonitor.startTiming('render');
    
    // Update node components to use optimized version
    const nodesWithComponents = filteredNodes.map(node => ({
      ...node,
      type: 'default',
    }));
    
    onNodesChange([{
      type: 'reset',
      nodes: nodesWithComponents
    }] as any);
    
    perfMonitor.endTiming('render');
  }, [filteredNodes, onNodesChange]);

  // Update edges when culled data changes
  useEffect(() => {
    onEdgesChange([{
      type: 'reset',
      edges: culledEdges
    }] as any);
  }, [culledEdges, onEdgesChange]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span>Loading optimized graph...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      {/* Performance Stats */}
      <PerformanceStats isVisible={showPerfStats} />
      
      {/* Controls Panel */}
      <Panel position="top-left" className="bg-white p-4 rounded-lg shadow-lg m-4">
        <div className="flex items-center space-x-2 mb-3">
          <h3 className="font-semibold">Optimized Graph Explorer</h3>
          <Badge variant="secondary" className="text-xs">
            {nodes.length}/{rawNodes.length} nodes
          </Badge>
        </div>
        
        <div className="flex items-center space-x-2 mb-3">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search entities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-8 h-8 text-sm"
            />
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPerfStats(!showPerfStats)}
            className="h-7 text-xs"
          >
            <Zap className="w-3 h-3 mr-1" />
            Stats
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSearchTerm('');
              reactFlow.fitView({ padding: 0.2 });
            }}
            className="h-7 text-xs"
          >
            <RefreshCw className="w-3 h-3 mr-1" />
            Reset
          </Button>
        </div>
      </Panel>

      {/* React Flow */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onViewportChange={onViewportChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        defaultViewport={{ x: 0, y: 0, zoom: 1 }}
        minZoom={0.3}
        maxZoom={2}
        deleteKeyCode={null} // Disable delete for performance
        multiSelectionKeyCode={null} // Disable multi-selection for performance
        selectionKeyCode={null} // Disable selection rectangle for performance
        panOnDrag
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick
      >
        <Background color="#f1f5f9" />
        <Controls />
        <MiniMap 
          nodeColor="#3b82f6" 
          maskColor="rgba(0, 0, 0, 0.1)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
};

// Main component with ReactFlow provider
export const PerformanceOptimizedGraphExplorer: React.FC = () => {
  return (
    <div className="w-full h-screen bg-gray-50">
      <ReactFlowProvider>
        <GraphExplorerInner />
      </ReactFlowProvider>
    </div>
  );
};

export default PerformanceOptimizedGraphExplorer;