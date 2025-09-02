import React, { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import { ReactFlow, Node, Edge, useReactFlow, ReactFlowProvider } from '@xyflow/react';
import { FixedSizeList as List } from 'react-window';
import { debounce, throttle } from 'lodash-es';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import {
  Zap,
  Activity,
  Gauge,
  TrendingUp,
  TrendingDown,
  Eye,
  EyeOff,
  Settings
} from 'lucide-react';

interface PerformanceMetrics {
  renderTime: number;
  nodeCount: number;
  edgeCount: number;
  visibleNodes: number;
  visibleEdges: number;
  memoryUsage: number;
  fps: number;
  lastUpdate: number;
}

interface PerformanceOptimizedGraphProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (node: Node) => void;
  className?: string;
  maxVisibleNodes?: number;
  enableVirtualization?: boolean;
  enableLevelOfDetail?: boolean;
  performanceMode?: 'auto' | 'high-performance' | 'high-quality';
}

// Level of Detail (LOD) configuration
const LOD_LEVELS = {
  HIGH_DETAIL: { minZoom: 1.5, showLabels: true, showConnections: true, nodeSize: 1.0 },
  MEDIUM_DETAIL: { minZoom: 0.8, showLabels: true, showConnections: true, nodeSize: 0.8 },
  LOW_DETAIL: { minZoom: 0.3, showLabels: false, showConnections: false, nodeSize: 0.6 },
  MINIMAL: { minZoom: 0, showLabels: false, showConnections: false, nodeSize: 0.4 }
};

export const PerformanceOptimizedGraph: React.FC<PerformanceOptimizedGraphProps> = ({
  nodes,
  edges,
  onNodeClick,
  className,
  maxVisibleNodes = 500,
  enableVirtualization = true,
  enableLevelOfDetail = true,
  performanceMode = 'auto'
}) => {
  const reactFlowInstance = useReactFlow();
  const [viewport, setViewport] = useState({ x: 0, y: 0, zoom: 1 });
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    visibleNodes: 0,
    visibleEdges: 0,
    memoryUsage: 0,
    fps: 0,
    lastUpdate: Date.now()
  });
  
  const [lodLevel, setLodLevel] = useState<keyof typeof LOD_LEVELS>('MEDIUM_DETAIL');
  const [showPerformancePanel, setShowPerformancePanel] = useState(false);
  const frameCount = useRef(0);
  const lastFrameTime = useRef(Date.now());
  const renderStartTime = useRef(Date.now());

  // Performance monitoring
  useEffect(() => {
    const updateMetrics = () => {
      const now = Date.now();
      frameCount.current++;
      
      if (now - lastFrameTime.current >= 1000) {
        const fps = Math.round(frameCount.current * 1000 / (now - lastFrameTime.current));
        
        setPerformanceMetrics(prev => ({
          ...prev,
          fps,
          lastUpdate: now,
          renderTime: now - renderStartTime.current,
          memoryUsage: (performance as any).memory ? 
            Math.round((performance as any).memory.usedJSHeapSize / 1048576) : 0
        }));
        
        frameCount.current = 0;
        lastFrameTime.current = now;
      }
      
      requestAnimationFrame(updateMetrics);
    };
    
    updateMetrics();
  }, []);

  // Determine LOD level based on zoom and node count
  const determineLodLevel = useCallback((zoom: number, nodeCount: number) => {
    if (nodeCount > 1000 || zoom < 0.3) return 'MINIMAL';
    if (nodeCount > 500 || zoom < 0.8) return 'LOW_DETAIL';
    if (zoom < 1.5) return 'MEDIUM_DETAIL';
    return 'HIGH_DETAIL';
  }, []);

  // Update LOD level when viewport changes
  useEffect(() => {
    if (enableLevelOfDetail) {
      const newLodLevel = determineLodLevel(viewport.zoom, nodes.length);
      if (newLodLevel !== lodLevel) {
        setLodLevel(newLodLevel);
      }
    }
  }, [viewport.zoom, nodes.length, lodLevel, determineLodLevel, enableLevelOfDetail]);

  // Viewport-based node culling
  const visibleNodes = useMemo(() => {
    if (!enableVirtualization) return nodes;

    renderStartTime.current = Date.now();
    
    const viewportBounds = {
      x: -viewport.x / viewport.zoom - 200,
      y: -viewport.y / viewport.zoom - 200,
      width: (window.innerWidth / viewport.zoom) + 400,
      height: (window.innerHeight / viewport.zoom) + 400
    };

    // Filter nodes within viewport bounds
    let visibleNodeList = nodes.filter(node => {
      return (
        node.position.x >= viewportBounds.x &&
        node.position.x <= viewportBounds.x + viewportBounds.width &&
        node.position.y >= viewportBounds.y &&
        node.position.y <= viewportBounds.y + viewportBounds.height
      );
    });

    // Limit total visible nodes for performance
    if (visibleNodeList.length > maxVisibleNodes) {
      // Sort by importance and keep top nodes
      visibleNodeList = visibleNodeList
        .sort((a, b) => (b.data.importance_weight || 0) - (a.data.importance_weight || 0))
        .slice(0, maxVisibleNodes);
    }

    // Apply LOD configuration
    const lodConfig = LOD_LEVELS[lodLevel];
    
    return visibleNodeList.map(node => ({
      ...node,
      data: {
        ...node.data,
        showLabel: lodConfig.showLabels,
        size: lodConfig.nodeSize,
        simplified: lodLevel === 'MINIMAL' || lodLevel === 'LOW_DETAIL'
      }
    }));
  }, [nodes, viewport, lodLevel, maxVisibleNodes, enableVirtualization]);

  // Edge culling and simplification
  const visibleEdges = useMemo(() => {
    if (!enableVirtualization) return edges;

    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const lodConfig = LOD_LEVELS[lodLevel];

    if (!lodConfig.showConnections) {
      return [];
    }

    let visibleEdgeList = edges.filter(edge =>
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    // Simplify edges at lower detail levels
    if (lodLevel === 'LOW_DETAIL' || lodLevel === 'MINIMAL') {
      // Group parallel edges and show only the strongest connection
      const edgeGroups = new Map<string, Edge[]>();
      
      visibleEdgeList.forEach(edge => {
        const key = `${edge.source}-${edge.target}`;
        const reverseKey = `${edge.target}-${edge.source}`;
        const groupKey = edgeGroups.has(key) ? key : reverseKey;
        
        if (!edgeGroups.has(groupKey)) {
          edgeGroups.set(groupKey, []);
        }
        edgeGroups.get(groupKey)!.push(edge);
      });

      visibleEdgeList = Array.from(edgeGroups.values()).map(group => {
        if (group.length === 1) return group[0];
        
        // Return the edge with highest confidence
        return group.reduce((best, current) => 
          (current.data?.confidence || 0) > (best.data?.confidence || 0) ? current : best
        );
      });
    }

    return visibleEdgeList.map(edge => ({
      ...edge,
      style: {
        ...edge.style,
        strokeWidth: lodLevel === 'HIGH_DETAIL' ? edge.style?.strokeWidth || 2 : 1
      }
    }));
  }, [edges, visibleNodes, lodLevel, enableVirtualization]);

  // Update performance metrics
  useEffect(() => {
    setPerformanceMetrics(prev => ({
      ...prev,
      nodeCount: nodes.length,
      edgeCount: edges.length,
      visibleNodes: visibleNodes.length,
      visibleEdges: visibleEdges.length
    }));
  }, [nodes.length, edges.length, visibleNodes.length, visibleEdges.length]);

  // Debounced viewport change handler
  const handleViewportChange = useCallback(
    debounce((newViewport) => {
      setViewport(newViewport);
    }, 100),
    []
  );

  // Performance mode optimization
  const getOptimizedSettings = () => {
    if (performanceMode === 'high-performance') {
      return {
        nodesDraggable: false,
        nodesConnectable: false,
        elementsSelectable: true,
        panOnDrag: true,
        zoomOnDoubleClick: false,
        deleteKeyCode: null,
        multiSelectionKeyCode: null
      };
    } else if (performanceMode === 'high-quality') {
      return {
        nodesDraggable: true,
        nodesConnectable: true,
        elementsSelectable: true,
        panOnDrag: true,
        zoomOnDoubleClick: true,
        deleteKeyCode: 'Delete',
        multiSelectionKeyCode: 'Meta'
      };
    } else {
      // Auto mode - adjust based on node count
      const isHighPerformance = nodes.length > 300;
      return {
        nodesDraggable: !isHighPerformance,
        nodesConnectable: !isHighPerformance,
        elementsSelectable: true,
        panOnDrag: true,
        zoomOnDoubleClick: !isHighPerformance,
        deleteKeyCode: isHighPerformance ? null : 'Delete',
        multiSelectionKeyCode: isHighPerformance ? null : 'Meta'
      };
    }
  };

  const optimizedSettings = getOptimizedSettings();

  return (
    <div className={cn("relative w-full h-full", className)}>
      {/* Performance Panel */}
      {showPerformancePanel && (
        <Card className="absolute top-4 left-4 z-10 bg-white/95 backdrop-blur-sm shadow-lg">
          <CardContent className="p-3 space-y-2">
            <div className="flex items-center space-x-2 text-sm font-medium">
              <Activity className="h-4 w-4 text-blue-600" />
              <span>Performance</span>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <div className="text-gray-500">FPS</div>
                <div className={cn(
                  "font-medium",
                  performanceMetrics.fps > 30 ? "text-green-600" :
                  performanceMetrics.fps > 15 ? "text-yellow-600" : "text-red-600"
                )}>
                  {performanceMetrics.fps}
                </div>
              </div>
              
              <div>
                <div className="text-gray-500">Visible</div>
                <div className="font-medium">
                  {performanceMetrics.visibleNodes}/{performanceMetrics.nodeCount}
                </div>
              </div>
              
              <div>
                <div className="text-gray-500">LOD</div>
                <Badge variant="outline" className="text-xs px-1 py-0">
                  {lodLevel.replace('_', ' ')}
                </Badge>
              </div>
              
              <div>
                <div className="text-gray-500">Memory</div>
                <div className="font-medium">
                  {performanceMetrics.memoryUsage}MB
                </div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span>Render Performance</span>
                <span>
                  {performanceMetrics.fps > 30 ? 'Excellent' :
                   performanceMetrics.fps > 15 ? 'Good' : 'Poor'}
                </span>
              </div>
              <Progress 
                value={Math.min(performanceMetrics.fps * 100 / 60, 100)} 
                className="h-1"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Controls */}
      <div className="absolute top-4 right-4 z-10 flex items-center space-x-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowPerformancePanel(!showPerformancePanel)}
          className="bg-white/95 backdrop-blur-sm"
          title="Toggle performance panel"
        >
          <Gauge className="h-4 w-4" />
        </Button>
        
        <Card className="bg-white/95 backdrop-blur-sm">
          <CardContent className="p-2 flex items-center space-x-2">
            {/* LOD Indicator */}
            <div className="flex items-center space-x-1 text-xs">
              <span className={cn(
                "w-2 h-2 rounded-full",
                lodLevel === 'HIGH_DETAIL' ? 'bg-green-500' :
                lodLevel === 'MEDIUM_DETAIL' ? 'bg-yellow-500' :
                lodLevel === 'LOW_DETAIL' ? 'bg-orange-500' : 'bg-red-500'
              )} />
              <span className="text-gray-600">
                {lodLevel === 'HIGH_DETAIL' ? 'HD' :
                 lodLevel === 'MEDIUM_DETAIL' ? 'MD' :
                 lodLevel === 'LOW_DETAIL' ? 'LD' : 'MIN'}
              </span>
            </div>
            
            {/* Node Count */}
            <div className="text-xs text-gray-600">
              {performanceMetrics.visibleNodes}/{performanceMetrics.nodeCount}
            </div>
            
            {/* FPS Indicator */}
            <div className="flex items-center space-x-1">
              {performanceMetrics.fps > 30 ? (
                <TrendingUp className="h-3 w-3 text-green-500" />
              ) : (
                <TrendingDown className="h-3 w-3 text-red-500" />
              )}
              <span className="text-xs font-medium">
                {performanceMetrics.fps}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Graph */}
      <ReactFlow
        nodes={visibleNodes}
        edges={visibleEdges}
        onNodeClick={onNodeClick}
        onMove={handleViewportChange}
        onMoveEnd={handleViewportChange}
        fitView
        {...optimizedSettings}
        className="bg-gray-50"
      >
        {/* Performance warning overlay */}
        {performanceMetrics.fps < 15 && performanceMetrics.fps > 0 && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center z-20">
            <Card className="bg-yellow-50 border-yellow-200">
              <CardContent className="p-3 flex items-center space-x-2">
                <TrendingDown className="h-4 w-4 text-yellow-600" />
                <div className="text-sm text-yellow-800">
                  Performance may be degraded. Consider enabling high-performance mode.
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </ReactFlow>
    </div>
  );
};

// Wrapper component with ReactFlowProvider
export const PerformanceOptimizedGraphWrapper: React.FC<PerformanceOptimizedGraphProps> = (props) => {
  return (
    <ReactFlowProvider>
      <PerformanceOptimizedGraph {...props} />
    </ReactFlowProvider>
  );
};