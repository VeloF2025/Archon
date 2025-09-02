import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { ReactFlow, Node, Edge, useReactFlow } from '@xyflow/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { 
  Search, 
  ArrowUp, 
  ArrowDown, 
  ArrowLeft, 
  ArrowRight,
  Escape,
  Enter,
  Tab,
  Info,
  VolumeX,
  Volume2
} from 'lucide-react';

interface AccessibleGraphExplorerProps {
  nodes: Node[];
  edges: Edge[];
  onNodeClick?: (node: Node) => void;
  onNodeSelect?: (nodeId: string | null) => void;
  selectedNodeId?: string | null;
  className?: string;
}

interface KeyboardState {
  selectedNodeIndex: number;
  mode: 'navigation' | 'selection' | 'details';
  announcements: boolean;
}

export const AccessibleGraphExplorer: React.FC<AccessibleGraphExplorerProps> = ({
  nodes,
  edges,
  onNodeClick,
  onNodeSelect,
  selectedNodeId,
  className
}) => {
  const reactFlowInstance = useReactFlow();
  const [keyboardState, setKeyboardState] = useState<KeyboardState>({
    selectedNodeIndex: 0,
    mode: 'navigation',
    announcements: true
  });
  
  const [focusedElement, setFocusedElement] = useState<string | null>(null);
  const [screenReaderText, setScreenReaderText] = useState<string>('');
  const ariaLiveRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<HTMLDivElement>(null);

  // Get currently selected node
  const selectedNode = nodes[keyboardState.selectedNodeIndex];

  // Announce to screen reader
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!keyboardState.announcements) return;
    
    setScreenReaderText(message);
    
    // Also speak using speech synthesis if available
    if (window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.2;
      utterance.volume = 0.8;
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      window.speechSynthesis.speak(utterance);
    }
  }, [keyboardState.announcements]);

  // Navigate between nodes
  const navigateNodes = useCallback((direction: 'up' | 'down' | 'left' | 'right' | 'next' | 'prev') => {
    if (nodes.length === 0) {
      announce("No entities available in the graph");
      return;
    }

    let nextIndex = keyboardState.selectedNodeIndex;

    if (direction === 'next') {
      nextIndex = (keyboardState.selectedNodeIndex + 1) % nodes.length;
    } else if (direction === 'prev') {
      nextIndex = (keyboardState.selectedNodeIndex - 1 + nodes.length) % nodes.length;
    } else {
      // Spatial navigation - find closest node in direction
      const currentNode = nodes[keyboardState.selectedNodeIndex];
      const currentPos = currentNode.position;
      
      let closestNode = null;
      let closestDistance = Infinity;
      
      nodes.forEach((node, index) => {
        if (index === keyboardState.selectedNodeIndex) return;
        
        const dx = node.position.x - currentPos.x;
        const dy = node.position.y - currentPos.y;
        
        // Check if node is in the correct direction
        const isInDirection = 
          (direction === 'right' && dx > 0 && Math.abs(dy) < Math.abs(dx)) ||
          (direction === 'left' && dx < 0 && Math.abs(dy) < Math.abs(dx)) ||
          (direction === 'down' && dy > 0 && Math.abs(dx) < Math.abs(dy)) ||
          (direction === 'up' && dy < 0 && Math.abs(dx) < Math.abs(dy));
        
        if (isInDirection) {
          const distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestNode = index;
          }
        }
      });
      
      if (closestNode !== null) {
        nextIndex = closestNode;
      } else {
        announce(`No entity found ${direction} of current selection`);
        return;
      }
    }

    setKeyboardState(prev => ({ ...prev, selectedNodeIndex: nextIndex }));
    
    const nextNode = nodes[nextIndex];
    const nodeData = nextNode.data;
    
    // Center the view on the selected node
    reactFlowInstance.setCenter(nextNode.position.x, nextNode.position.y, { zoom: 1 });
    
    // Announce the selected node
    announce(
      `Selected ${nodeData.entity_type} entity: ${nodeData.name}. ` +
      `Confidence ${Math.round(nodeData.confidence_score * 100)}%. ` +
      `${nextIndex + 1} of ${nodes.length} entities. ` +
      `Press Enter to view details, or Space to explore connections.`
    );

    if (onNodeSelect) {
      onNodeSelect(nextNode.id);
    }
  }, [nodes, keyboardState.selectedNodeIndex, reactFlowInstance, announce, onNodeSelect]);

  // Handle keyboard shortcuts
  useHotkeys('tab', (e) => {
    e.preventDefault();
    navigateNodes('next');
  }, { scopes: ['graph'] });

  useHotkeys('shift+tab', (e) => {
    e.preventDefault();
    navigateNodes('prev');
  }, { scopes: ['graph'] });

  useHotkeys('arrowup', (e) => {
    e.preventDefault();
    navigateNodes('up');
  }, { scopes: ['graph'] });

  useHotkeys('arrowdown', (e) => {
    e.preventDefault();
    navigateNodes('down');
  }, { scopes: ['graph'] });

  useHotkeys('arrowleft', (e) => {
    e.preventDefault();
    navigateNodes('left');
  }, { scopes: ['graph'] });

  useHotkeys('arrowright', (e) => {
    e.preventDefault();
    navigateNodes('right');
  }, { scopes: ['graph'] });

  useHotkeys('enter', (e) => {
    e.preventDefault();
    if (selectedNode && onNodeClick) {
      onNodeClick(selectedNode);
      announce(`Opened details for ${selectedNode.data.name}`);
    }
  }, { scopes: ['graph'] });

  useHotkeys('space', (e) => {
    e.preventDefault();
    if (selectedNode) {
      const connections = edges.filter(
        edge => edge.source === selectedNode.id || edge.target === selectedNode.id
      );
      announce(
        `${selectedNode.data.name} has ${connections.length} connections. ` +
        `Use Alt+C to explore connections.`
      );
    }
  }, { scopes: ['graph'] });

  useHotkeys('alt+c', (e) => {
    e.preventDefault();
    if (selectedNode) {
      const connections = edges.filter(
        edge => edge.source === selectedNode.id || edge.target === selectedNode.id
      );
      
      if (connections.length > 0) {
        const connectionDetails = connections.map(edge => {
          const isOutgoing = edge.source === selectedNode.id;
          const connectedNodeId = isOutgoing ? edge.target : edge.source;
          const connectedNode = nodes.find(n => n.id === connectedNodeId);
          
          return `${isOutgoing ? 'Connected to' : 'Connected from'} ${connectedNode?.data.name || 'unknown'} via ${edge.label || edge.type}`;
        }).join('. ');
        
        announce(`Connections for ${selectedNode.data.name}: ${connectionDetails}`);
      } else {
        announce(`${selectedNode.data.name} has no connections`);
      }
    }
  }, { scopes: ['graph'] });

  useHotkeys('alt+s', (e) => {
    e.preventDefault();
    setKeyboardState(prev => ({
      ...prev,
      announcements: !prev.announcements
    }));
    announce(
      `Screen reader announcements ${!keyboardState.announcements ? 'enabled' : 'disabled'}`,
      'assertive'
    );
  }, { scopes: ['graph'] });

  useHotkeys('alt+h', (e) => {
    e.preventDefault();
    announce(
      `Keyboard shortcuts: ` +
      `Tab or Shift+Tab to navigate entities. ` +
      `Arrow keys for spatial navigation. ` +
      `Enter to view details. ` +
      `Space to check connections. ` +
      `Alt+C to explore connections. ` +
      `Alt+S to toggle announcements. ` +
      `Alt+H for help.`,
      'assertive'
    );
  }, { scopes: ['graph'] });

  // Set focus scope when component mounts
  useEffect(() => {
    if (graphRef.current) {
      graphRef.current.focus();
    }
  }, []);

  // Announce graph overview on mount
  useEffect(() => {
    if (nodes.length > 0) {
      const entityTypes = [...new Set(nodes.map(n => n.data.entity_type))];
      announce(
        `Graph loaded with ${nodes.length} entities and ${edges.length} connections. ` +
        `Entity types: ${entityTypes.join(', ')}. ` +
        `Use Tab to navigate or Alt+H for keyboard shortcuts.`
      );
    }
  }, [nodes, edges, announce]);

  // Create accessible graph description
  const getGraphDescription = () => {
    const entityCounts = nodes.reduce((acc, node) => {
      const type = node.data.entity_type;
      acc[type] = (acc[type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const description = Object.entries(entityCounts)
      .map(([type, count]) => `${count} ${type}${count === 1 ? '' : 's'}`)
      .join(', ');

    return `Knowledge graph containing ${description} with ${edges.length} relationships.`;
  };

  return (
    <div className={cn("relative w-full h-full", className)}>
      {/* Screen Reader Live Region */}
      <div
        ref={ariaLiveRef}
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
        role="status"
      >
        {screenReaderText}
      </div>

      {/* Hidden Graph Description */}
      <div className="sr-only">
        <h2>Knowledge Graph Visualization</h2>
        <p>{getGraphDescription()}</p>
        <p>Use keyboard navigation to explore entities. Press Alt+H for help.</p>
      </div>

      {/* Accessibility Controls */}
      <div className="absolute top-4 right-4 z-10 flex items-center space-x-2">
        <Card className="bg-white/95 backdrop-blur-sm">
          <CardContent className="p-2 flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setKeyboardState(prev => ({
                ...prev,
                announcements: !prev.announcements
              }))}
              className="p-2"
              title={`${keyboardState.announcements ? 'Disable' : 'Enable'} screen reader announcements`}
              aria-label={`${keyboardState.announcements ? 'Disable' : 'Enable'} screen reader announcements`}
            >
              {keyboardState.announcements ? (
                <Volume2 className="h-4 w-4" />
              ) : (
                <VolumeX className="h-4 w-4" />
              )}
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => announce(
                `Keyboard shortcuts: ` +
                `Tab or Shift+Tab to navigate entities. ` +
                `Arrow keys for spatial navigation. ` +
                `Enter to view details. ` +
                `Space to check connections. ` +
                `Alt+C to explore connections. ` +
                `Alt+S to toggle announcements.`,
                'assertive'
              )}
              className="p-2"
              title="Show keyboard shortcuts"
              aria-label="Show keyboard shortcuts"
            >
              <Info className="h-4 w-4" />
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Current Selection Indicator */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 z-10">
          <Card className="bg-white/95 backdrop-blur-sm">
            <CardContent className="p-3">
              <div className="flex items-center space-x-2 text-sm">
                <Badge variant="secondary">
                  {keyboardState.selectedNodeIndex + 1} / {nodes.length}
                </Badge>
                <span className="font-medium">{selectedNode.data.name}</span>
                <span className="text-gray-500">({selectedNode.data.entity_type})</span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Graph Area */}
      <div
        ref={graphRef}
        className="w-full h-full"
        tabIndex={0}
        role="application"
        aria-label="Interactive knowledge graph"
        aria-describedby="graph-description"
        onFocus={() => {
          // Set keyboard scope when focused
          document.body.setAttribute('data-scope', 'graph');
        }}
        onBlur={() => {
          document.body.removeAttribute('data-scope');
        }}
        onKeyDown={(e) => {
          // Prevent default behavior for navigation keys
          if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab', 'Enter', ' '].includes(e.key)) {
            e.preventDefault();
          }
        }}
      >
        <ReactFlow
          nodes={nodes.map((node, index) => ({
            ...node,
            data: {
              ...node.data,
              selected: index === keyboardState.selectedNodeIndex,
              'aria-label': `${node.data.entity_type} entity: ${node.data.name}, confidence ${Math.round(node.data.confidence_score * 100)}%`,
              'aria-describedby': `node-${node.id}-details`
            }
          }))}
          edges={edges.map(edge => ({
            ...edge,
            'aria-label': `Connection from ${edge.source} to ${edge.target} of type ${edge.type}`
          }))}
          fitView
          attributionPosition="bottom-left"
          className="bg-gray-50"
        >
          {/* Add custom controls or overlays here if needed */}
        </ReactFlow>
      </div>

      {/* Hidden details for current node (for screen readers) */}
      {selectedNode && (
        <div id={`node-${selectedNode.id}-details`} className="sr-only">
          <h3>{selectedNode.data.name} Details</h3>
          <p>Entity Type: {selectedNode.data.entity_type}</p>
          <p>Confidence Score: {Math.round(selectedNode.data.confidence_score * 100)}%</p>
          <p>Importance: {Math.round(selectedNode.data.importance_weight * 100)}%</p>
          {selectedNode.data.tags && selectedNode.data.tags.length > 0 && (
            <p>Tags: {selectedNode.data.tags.join(', ')}</p>
          )}
          {edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length > 0 && (
            <p>
              Connections: {edges.filter(e => e.source === selectedNode.id || e.target === selectedNode.id).length}
            </p>
          )}
        </div>
      )}
    </div>
  );
};