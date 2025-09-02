import React, { useState, useCallback, useEffect } from 'react';
import { ReactFlow, Node, Edge, useReactFlow, Panel } from '@xyflow/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/Input';
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle, DrawerTrigger } from '@/components/ui/drawer';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { 
  Search, 
  Filter, 
  Menu,
  X,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Info,
  ChevronUp,
  ChevronDown,
  Touch,
  Smartphone,
  Tablet
} from 'lucide-react';

interface MobileGraphExplorerProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (node: Node) => void;
  selectedNode?: Node | null;
  isLoading?: boolean;
  className?: string;
}

type MobileViewMode = 'phone' | 'tablet' | 'desktop';
type TouchGesture = 'tap' | 'double-tap' | 'long-press' | 'pinch' | 'pan';

export const MobileGraphExplorer: React.FC<MobileGraphExplorerProps> = ({
  nodes,
  edges,
  onNodeClick,
  selectedNode,
  isLoading = false,
  className
}) => {
  const reactFlowInstance = useReactFlow();
  const [viewMode, setViewMode] = useState<MobileViewMode>('phone');
  const [showSearch, setShowSearch] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [showEntityDetails, setShowEntityDetails] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Touch gesture tracking
  const [touchState, setTouchState] = useState({
    startTime: 0,
    startTouch: { x: 0, y: 0 },
    isLongPress: false,
    gestureType: null as TouchGesture | null
  });

  // Detect device type and set appropriate view mode
  useEffect(() => {
    const detectViewMode = () => {
      const width = window.innerWidth;
      const isTouchDevice = 'ontouchstart' in window;
      
      if (width < 640) {
        setViewMode('phone');
      } else if (width < 1024) {
        setViewMode('tablet');
      } else {
        setViewMode('desktop');
      }
    };

    detectViewMode();
    window.addEventListener('resize', detectViewMode);
    
    return () => window.removeEventListener('resize', detectViewMode);
  }, []);

  // Handle touch gestures
  const handleTouchStart = useCallback((event: React.TouchEvent) => {
    const touch = event.touches[0];
    setTouchState({
      startTime: Date.now(),
      startTouch: { x: touch.clientX, y: touch.clientY },
      isLongPress: false,
      gestureType: null
    });

    // Set long press timer
    setTimeout(() => {
      setTouchState(prev => ({ ...prev, isLongPress: true, gestureType: 'long-press' }));
    }, 600);
  }, []);

  const handleTouchEnd = useCallback((event: React.TouchEvent) => {
    const endTime = Date.now();
    const duration = endTime - touchState.startTime;
    const touch = event.changedTouches[0];
    const deltaX = Math.abs(touch.clientX - touchState.startTouch.x);
    const deltaY = Math.abs(touch.clientY - touchState.startTouch.y);
    const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);

    if (duration < 200 && distance < 10) {
      // Quick tap
      setTouchState(prev => ({ ...prev, gestureType: 'tap' }));
    } else if (duration < 400 && distance < 10) {
      // Potential double tap (handle in component that uses this)
      setTouchState(prev => ({ ...prev, gestureType: 'double-tap' }));
    }
  }, [touchState]);

  // Enhanced zoom controls for mobile
  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(zoomLevel * 1.5, 3);
    setZoomLevel(newZoom);
    reactFlowInstance.zoomTo(newZoom, { duration: 300 });
  }, [zoomLevel, reactFlowInstance]);

  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(zoomLevel * 0.7, 0.1);
    setZoomLevel(newZoom);
    reactFlowInstance.zoomTo(newZoom, { duration: 300 });
  }, [zoomLevel, reactFlowInstance]);

  const handleFitView = useCallback(() => {
    reactFlowInstance.fitView({ padding: 0.1, duration: 300 });
    setZoomLevel(1);
  }, [reactFlowInstance]);

  // Filter nodes based on search
  const filteredNodes = nodes.filter(node =>
    node.data.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    node.data.entity_type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get layout classes based on view mode
  const getLayoutClasses = () => {
    const base = "w-full h-full flex flex-col";
    
    switch (viewMode) {
      case 'phone':
        return cn(base, "text-sm");
      case 'tablet':
        return cn(base, "text-base");
      default:
        return cn(base, "text-base");
    }
  };

  // Render mobile-optimized entity node
  const MobileEntityNode = ({ data, selected }: { data: any; selected?: boolean }) => {
    const size = viewMode === 'phone' ? 60 : 80;
    const iconSize = viewMode === 'phone' ? '16px' : '20px';
    
    return (
      <div
        className={cn(
          "rounded-xl bg-white border-2 shadow-lg transition-all duration-200 flex flex-col items-center justify-center p-2",
          selected ? "border-blue-500 shadow-xl scale-105" : "border-gray-300"
        )}
        style={{ width: `${size}px`, height: `${size}px` }}
      >
        <div className="text-lg mb-1">{data.icon || '📄'}</div>
        <div className="text-xs font-medium text-center leading-tight truncate w-full px-1">
          {data.name.length > 8 ? data.name.substring(0, 6) + '...' : data.name}
        </div>
        {viewMode !== 'phone' && (
          <div className="text-xs text-gray-500 mt-1">
            {Math.round(data.confidence_score * 100)}%
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={getLayoutClasses()}>
      {/* Mobile Header */}
      <header className="flex items-center justify-between p-3 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center space-x-2">
          <h1 className="font-semibold text-gray-900">
            {viewMode === 'phone' ? 'Graphiti' : 'Graphiti Explorer'}
          </h1>
          {viewMode !== 'phone' && (
            <Badge variant="secondary" className="text-xs">
              {nodes.length} entities
            </Badge>
          )}
        </div>

        <div className="flex items-center space-x-1">
          {/* Search Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSearch(!showSearch)}
            className={cn("p-2", showSearch && "bg-blue-100 text-blue-600")}
          >
            <Search className="h-4 w-4" />
          </Button>

          {/* Filters */}
          <Sheet open={showFilters} onOpenChange={setShowFilters}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="sm" className="p-2">
                <Filter className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80">
              <SheetHeader>
                <SheetTitle>Filter Options</SheetTitle>
                <SheetDescription>
                  Filter entities by type, confidence, or other criteria
                </SheetDescription>
              </SheetHeader>
              <div className="mt-6 space-y-4">
                {/* Filter content would go here */}
                <div className="text-sm text-gray-500">Filter controls coming soon...</div>
              </div>
            </SheetContent>
          </Sheet>

          {/* Menu */}
          <Button variant="ghost" size="sm" className="p-2">
            <Menu className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Search Bar */}
      {showSearch && (
        <div className="p-3 bg-gray-50 border-b border-gray-200">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              placeholder="Search entities..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 h-10 bg-white"
            />
            {searchTerm && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSearchTerm('')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-1 h-6 w-6"
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Main Graph Area */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={filteredNodes}
          edges={edges}
          onNodeClick={(event, node) => {
            setShowEntityDetails(true);
            onNodeClick?.(node);
          }}
          fitView
          minZoom={0.1}
          maxZoom={3}
          onTouchStart={handleTouchStart}
          onTouchEnd={handleTouchEnd}
          className="touch-pan-x touch-pan-y"
        >
          {/* Mobile-optimized controls panel */}
          <Panel position="bottom-right" className="mb-4 mr-4">
            <div className="flex flex-col space-y-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleZoomIn}
                className="bg-white shadow-lg w-10 h-10 p-0"
                aria-label="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleZoomOut}
                className="bg-white shadow-lg w-10 h-10 p-0"
                aria-label="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleFitView}
                className="bg-white shadow-lg w-10 h-10 p-0"
                aria-label="Fit to view"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </Panel>

          {/* Touch instructions for first-time users */}
          <Panel position="top-center" className="mt-4">
            <Card className="bg-white/90 backdrop-blur-sm shadow-sm">
              <CardContent className="p-2 text-center">
                <div className="flex items-center space-x-2 text-xs text-gray-600">
                  <Touch className="h-3 w-3" />
                  <span>Tap entities • Pinch to zoom • Drag to pan</span>
                </div>
              </CardContent>
            </Card>
          </Panel>

          {/* Loading overlay */}
          {isLoading && (
            <Panel position="center">
              <div className="bg-white rounded-lg p-4 shadow-lg flex items-center space-x-3">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                <span className="text-sm text-gray-700">Loading graph...</span>
              </div>
            </Panel>
          )}
        </ReactFlow>
      </div>

      {/* Bottom Entity Details Drawer */}
      <Drawer open={showEntityDetails} onOpenChange={setShowEntityDetails}>
        <DrawerContent className="max-h-[60vh]">
          <DrawerHeader className="text-left">
            <DrawerTitle>
              {selectedNode ? selectedNode.data.name : 'Entity Details'}
            </DrawerTitle>
            <DrawerDescription>
              {selectedNode 
                ? `${selectedNode.data.entity_type} • ${Math.round(selectedNode.data.confidence_score * 100)}% confidence`
                : 'No entity selected'
              }
            </DrawerDescription>
          </DrawerHeader>
          
          <div className="px-4 pb-6">
            {selectedNode ? (
              <ScrollArea className="h-full">
                <div className="space-y-4">
                  {/* Basic Info */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="text-gray-500 mb-1">Type</div>
                      <Badge variant="secondary">{selectedNode.data.entity_type}</Badge>
                    </div>
                    <div>
                      <div className="text-gray-500 mb-1">Confidence</div>
                      <div className="font-medium">
                        {Math.round(selectedNode.data.confidence_score * 100)}%
                      </div>
                    </div>
                  </div>

                  {/* Tags */}
                  {selectedNode.data.tags && selectedNode.data.tags.length > 0 && (
                    <div>
                      <div className="text-gray-500 mb-2 text-sm">Tags</div>
                      <div className="flex flex-wrap gap-1">
                        {selectedNode.data.tags.map((tag: string, index: number) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Connections */}
                  <div>
                    <div className="text-gray-500 mb-2 text-sm">Connections</div>
                    <div className="text-sm">
                      {edges.filter(e => 
                        e.source === selectedNode.id || e.target === selectedNode.id
                      ).length} connected entities
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex space-x-2 pt-4">
                    <Button variant="outline" size="sm" className="flex-1">
                      View Source
                    </Button>
                    <Button variant="outline" size="sm" className="flex-1">
                      Explore
                    </Button>
                  </div>
                </div>
              </ScrollArea>
            ) : (
              <div className="text-center py-8">
                <Info className="h-8 w-8 mx-auto text-gray-400 mb-2" />
                <p className="text-gray-500">Select an entity to view details</p>
              </div>
            )}
          </div>
        </DrawerContent>
      </Drawer>
    </div>
  );
};