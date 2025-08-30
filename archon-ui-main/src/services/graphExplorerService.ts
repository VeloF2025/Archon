/**
 * Graph Explorer Service - Real API Integration
 * 
 * Provides comprehensive graph visualization and exploration functionality:
 * - Fetch graph data with filtering
 * - Temporal filtering and analytics
 * - Entity search and details
 * - Real-time updates via WebSocket
 * - Performance monitoring
 */

import { API_BASE } from './api';
import { knowledgeSocketIO } from './socketIOService';

// Graph Data Types
export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, any>;
}

export interface GraphMetadata {
  total_entities: number;
  total_relationships: number;
  entity_types: string[];
  relationship_types: string[];
  last_updated: number;
  query_time: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  metadata: GraphMetadata;
}

// Filter Types
export interface GraphFilters {
  entity_types?: string[];
  relationship_types?: string[];
  time_window?: string;
  search_term?: string;
  confidence_threshold?: number;
  importance_threshold?: number;
  limit?: number;
}

export interface TemporalFilter {
  start_time: number;
  end_time: number;
  granularity?: 'hour' | 'day' | 'week';
  entity_type?: string;
  pattern?: 'evolution' | 'trending';
}

export interface EntitySearchRequest {
  query: string;
  entity_types?: string[];
  limit?: number;
}

// Entity Details
export interface EntityDetails {
  entity: {
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
  };
  related_entities: Array<{
    entity: {
      entity_id: string;
      name: string;
      entity_type: string;
      confidence_score: number;
    };
    relationship: {
      relationship_type: string;
      confidence: number;
    };
  }>;
}

// Graph Statistics
export interface GraphStatistics {
  performance: {
    avg_query_time: number;
    max_query_time: number;
    total_queries: number;
    cached_entities: number;
    db_path: string;
  };
  entity_counts: Record<string, number>;
  total_entities: number;
  recent_activity: {
    entities_24h: number;
    recent_entities: Array<{
      entity_id: string;
      name: string;
      entity_type: string;
      creation_time: number;
    }>;
  };
  trending: Array<{
    entity_id: string;
    name: string;
    entity_type: string;
    access_frequency: number;
    importance_weight: number;
  }>;
}

// Available UI Actions
export interface GraphAction {
  name: string;
  label: string;
  description: string;
  shortcut: string;
}

class GraphExplorerService {
  private apiBase: string;
  private wsListeners: Set<(data: any) => void> = new Set();
  private lastGraphData: GraphData | null = null;

  constructor() {
    this.apiBase = `${API_BASE}/graphiti`;
    this.setupWebSocketListeners();
  }

  /**
   * Get graph data with optional filtering
   */
  async getGraphData(filters?: GraphFilters): Promise<GraphData> {
    try {
      const params = new URLSearchParams();
      
      if (filters?.entity_types?.length) {
        params.append('entity_types', filters.entity_types.join(','));
      }
      if (filters?.relationship_types?.length) {
        params.append('relationship_types', filters.relationship_types.join(','));
      }
      if (filters?.time_window) {
        params.append('time_window', filters.time_window);
      }
      if (filters?.search_term) {
        params.append('search_term', filters.search_term);
      }
      if (filters?.confidence_threshold !== undefined) {
        params.append('confidence_threshold', filters.confidence_threshold.toString());
      }
      if (filters?.importance_threshold !== undefined) {
        params.append('importance_threshold', filters.importance_threshold.toString());
      }
      if (filters?.limit !== undefined) {
        params.append('limit', filters.limit.toString());
      }

      const url = `${this.apiBase}/graph-data${params.toString() ? `?${params.toString()}` : ''}`;
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch graph data: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      this.lastGraphData = data;
      
      // Emit update to listeners
      this.emitGraphUpdate(data);
      
      return data;
    } catch (error) {
      console.error('Error fetching graph data:', error);
      throw error;
    }
  }

  /**
   * Apply temporal filter to graph data
   */
  async applyTemporalFilter(filter: TemporalFilter): Promise<GraphData> {
    try {
      const response = await fetch(`${this.apiBase}/temporal-filter`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(filter),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Temporal filter failed: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      this.lastGraphData = data;
      
      // Emit update to listeners
      this.emitGraphUpdate(data);
      
      return data;
    } catch (error) {
      console.error('Error applying temporal filter:', error);
      throw error;
    }
  }

  /**
   * Get detailed information about a specific entity
   */
  async getEntityDetails(entityId: string): Promise<EntityDetails> {
    try {
      const response = await fetch(`${this.apiBase}/entity/${entityId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Entity not found');
        }
        const errorText = await response.text();
        throw new Error(`Failed to fetch entity details: ${response.status} ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching entity details:', error);
      throw error;
    }
  }

  /**
   * Search entities by name or tags
   */
  async searchEntities(request: EntitySearchRequest): Promise<{
    entities: Array<{
      entity_id: string;
      entity_type: string;
      name: string;
      confidence_score: number;
      importance_weight: number;
      tags: string[];
      creation_time: number;
    }>;
    total: number;
  }> {
    try {
      const response = await fetch(`${this.apiBase}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Search failed: ${response.status} ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error searching entities:', error);
      throw error;
    }
  }

  /**
   * Get comprehensive graph statistics
   */
  async getGraphStatistics(): Promise<GraphStatistics> {
    try {
      const response = await fetch(`${this.apiBase}/stats`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch statistics: ${response.status} ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching graph statistics:', error);
      throw error;
    }
  }

  /**
   * Get available UI actions for the Graph Explorer
   */
  async getAvailableActions(): Promise<GraphAction[]> {
    try {
      const response = await fetch(`${this.apiBase}/available-actions`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to fetch actions: ${response.status} ${errorText}`);
      }

      const data = await response.json();
      return data.actions;
    } catch (error) {
      console.error('Error fetching available actions:', error);
      // Return default actions if API fails
      return this.getDefaultActions();
    }
  }

  /**
   * Check Graphiti service health
   */
  async healthCheck(): Promise<{
    status: string;
    checks: Record<string, any>;
    timestamp: number;
  }> {
    try {
      const response = await fetch(`${this.apiBase}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error checking Graphiti health:', error);
      return {
        status: 'unhealthy',
        checks: {
          connection: { status: 'failed', error: error.message }
        },
        timestamp: Date.now()
      };
    }
  }

  /**
   * Setup WebSocket listeners for real-time updates
   */
  private setupWebSocketListeners(): void {
    // Import the WebSocket service dynamically to avoid circular dependencies
    import('./graphitiWebSocketService').then(({ graphitiWebSocketService }) => {
      // Listen for graph update events
      graphitiWebSocketService.on('graph_update', (update) => {
        console.log('Received Graphiti graph update:', update);
        
        switch (update.type) {
          case 'entity_created':
          case 'entity_updated':
          case 'entity_deleted':
          case 'relationship_created':
          case 'relationship_updated':
          case 'relationship_deleted':
            // Refresh graph data for entity/relationship changes
            this.refreshGraphData();
            break;
            
          case 'graph_refreshed':
            // Full refresh requested
            console.log('Full graph refresh requested:', update.data.reason);
            this.refreshGraphData();
            break;
            
          case 'health_changed':
            // Health status changed - emit to listeners
            this.wsListeners.forEach(listener => {
              try {
                // Send health update to UI
                if (this.lastGraphData) {
                  listener({ ...this.lastGraphData, health: update.data });
                }
              } catch (error) {
                console.error('Error in health update listener:', error);
              }
            });
            break;
        }
      });

      // Listen for performance updates
      graphitiWebSocketService.on('performance_update', (data) => {
        console.log('Performance metrics updated:', data);
        // Could emit performance data to listeners if needed
      });

      // Listen for connection status
      graphitiWebSocketService.on('connection_status', (status) => {
        console.log('Graphiti WebSocket connection status:', status);
      });

      // Listen for batch updates (more efficient)
      graphitiWebSocketService.on('batch_update', (data) => {
        console.log('Batch update received:', data);
        this.refreshGraphData();
      });
    });
  }

  /**
   * Add listener for graph updates
   */
  onGraphUpdate(listener: (data: GraphData) => void): () => void {
    this.wsListeners.add(listener);
    
    // Return unsubscribe function
    return () => {
      this.wsListeners.delete(listener);
    };
  }

  /**
   * Emit graph update to all listeners
   */
  private emitGraphUpdate(data: GraphData): void {
    this.wsListeners.forEach(listener => {
      try {
        listener(data);
      } catch (error) {
        console.error('Error in graph update listener:', error);
      }
    });
  }

  /**
   * Refresh current graph data
   */
  private async refreshGraphData(): Promise<void> {
    try {
      // Use same filters as last request if available
      const data = await this.getGraphData();
      this.emitGraphUpdate(data);
    } catch (error) {
      console.error('Error refreshing graph data:', error);
    }
  }

  /**
   * Get default UI actions when API is unavailable
   */
  private getDefaultActions(): GraphAction[] {
    return [
      {
        name: 'zoom_in',
        label: 'Zoom In',
        description: 'Zoom into the graph visualization',
        shortcut: '+'
      },
      {
        name: 'zoom_out',
        label: 'Zoom Out', 
        description: 'Zoom out of the graph visualization',
        shortcut: '-'
      },
      {
        name: 'pan',
        label: 'Pan',
        description: 'Pan around the graph by dragging',
        shortcut: 'drag'
      },
      {
        name: 'filter_by_type',
        label: 'Filter by Type',
        description: 'Filter entities by their type',
        shortcut: 'F'
      },
      {
        name: 'temporal_filter',
        label: 'Time Filter',
        description: 'Filter by time range',
        shortcut: 'T'
      },
      {
        name: 'search',
        label: 'Search',
        description: 'Search entities by name or tags',
        shortcut: 'Ctrl+F'
      },
      {
        name: 'export_graph',
        label: 'Export Graph',
        description: 'Export graph data as JSON',
        shortcut: 'Ctrl+E'
      },
      {
        name: 'reset_layout',
        label: 'Reset Layout',
        description: 'Reset graph to default layout',
        shortcut: 'R'
      },
      {
        name: 'toggle_labels',
        label: 'Toggle Labels',
        description: 'Show/hide node labels',
        shortcut: 'L'
      },
      {
        name: 'fullscreen',
        label: 'Fullscreen',
        description: 'Enter fullscreen mode',
        shortcut: 'F11'
      }
    ];
  }

  /**
   * Export graph data as JSON
   */
  exportGraphData(): void {
    if (!this.lastGraphData) {
      throw new Error('No graph data available to export');
    }

    const dataStr = JSON.stringify(this.lastGraphData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `graph-data-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
  }

  /**
   * Generate time window options for filtering
   */
  getTimeWindowOptions(): Array<{ label: string; value: string }> {
    return [
      { label: 'Last Hour', value: '1h' },
      { label: 'Last 6 Hours', value: '6h' },
      { label: 'Last 24 Hours', value: '24h' },
      { label: 'Last 3 Days', value: '3d' },
      { label: 'Last Week', value: '7d' },
      { label: 'Last Month', value: '30d' },
      { label: 'Last 3 Months', value: '90d' },
      { label: 'Last 6 Months', value: '180d' },
      { label: 'Last Year', value: '365d' }
    ];
  }

  /**
   * Generate entity type options for filtering
   */
  getEntityTypeOptions(): Array<{ label: string; value: string }> {
    return [
      { label: 'All Types', value: 'all' },
      { label: 'Function', value: 'function' },
      { label: 'Class', value: 'class' },
      { label: 'Module', value: 'module' },
      { label: 'Concept', value: 'concept' },
      { label: 'Agent', value: 'agent' },
      { label: 'Project', value: 'project' },
      { label: 'Requirement', value: 'requirement' },
      { label: 'Pattern', value: 'pattern' },
      { label: 'Document', value: 'document' }
    ];
  }

  /**
   * Parse time window string to milliseconds
   */
  parseTimeWindow(timeWindow: string): number {
    const match = timeWindow.match(/^(\d+)([hdw])$/);
    if (!match) return 0;
    
    const [, amount, unit] = match;
    const num = parseInt(amount, 10);
    
    switch (unit) {
      case 'h': return num * 60 * 60 * 1000;
      case 'd': return num * 24 * 60 * 60 * 1000;
      case 'w': return num * 7 * 24 * 60 * 60 * 1000;
      default: return 0;
    }
  }
}

// Export singleton instance
export const graphExplorerService = new GraphExplorerService();
export default graphExplorerService;