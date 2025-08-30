/**
 * Graphiti WebSocket Service - Real-time Graph Updates
 * 
 * Handles WebSocket communication for real-time graph visualization updates:
 * - Entity creation/update/deletion events
 * - Relationship changes
 * - Performance metrics updates
 * - Health status changes
 */

import { knowledgeSocketIO } from './socketIOService';
import { GraphData } from './graphExplorerService';

export interface GraphUpdateEvent {
  type: 'entity_created' | 'entity_updated' | 'entity_deleted' | 
        'relationship_created' | 'relationship_updated' | 'relationship_deleted' |
        'graph_refreshed' | 'health_changed';
  data: any;
  timestamp: number;
}

export interface EntityEvent {
  entity_id: string;
  entity_type: string;
  name: string;
  change_type: 'created' | 'updated' | 'deleted';
  details?: Record<string, any>;
}

export interface RelationshipEvent {
  relationship_id: string;
  source_id: string;
  target_id: string;
  relationship_type: string;
  change_type: 'created' | 'updated' | 'deleted';
  confidence: number;
}

class GraphitiWebSocketService {
  private listeners: Map<string, Set<Function>> = new Map();
  private isConnected: boolean = false;

  constructor() {
    this.setupEventListeners();
  }

  /**
   * Setup WebSocket event listeners for graph updates
   */
  private setupEventListeners(): void {
    // Listen for connection status
    knowledgeSocketIO.onStateChange((state) => {
      this.isConnected = knowledgeSocketIO.isConnected();
      console.log('🔗 Graphiti WebSocket connection state:', state);
      this.emit('connection_status', { connected: this.isConnected, state });
    });

    // Graph-specific events from backend
    knowledgeSocketIO.onMessage('graphiti_entity_created', (data: EntityEvent) => {
      console.log('📈 Entity created:', data);
      this.emit('entity_created', data);
      this.emit('graph_update', { type: 'entity_created', data, timestamp: Date.now() });
    });

    knowledgeSocketIO.onMessage('graphiti_entity_updated', (data: EntityEvent) => {
      console.log('📝 Entity updated:', data);
      this.emit('entity_updated', data);
      this.emit('graph_update', { type: 'entity_updated', data, timestamp: Date.now() });
    });

    knowledgeSocketIO.onMessage('graphiti_entity_deleted', (data: EntityEvent) => {
      console.log('🗑️ Entity deleted:', data);
      this.emit('entity_deleted', data);
      this.emit('graph_update', { type: 'entity_deleted', data, timestamp: Date.now() });
    });

    knowledgeSocketIO.onMessage('graphiti_relationship_created', (data: RelationshipEvent) => {
      console.log('🔗 Relationship created:', data);
      this.emit('relationship_created', data);
      this.emit('graph_update', { type: 'relationship_created', data, timestamp: Date.now() });
    });

    knowledgeSocketIO.onMessage('graphiti_relationship_updated', (data: RelationshipEvent) => {
      console.log('🔄 Relationship updated:', data);
      this.emit('relationship_updated', data);
      this.emit('graph_update', { type: 'relationship_updated', data, timestamp: Date.now() });
    });

    knowledgeSocketIO.onMessage('graphiti_relationship_deleted', (data: RelationshipEvent) => {
      console.log('💔 Relationship deleted:', data);
      this.emit('relationship_deleted', data);
      this.emit('graph_update', { type: 'relationship_deleted', data, timestamp: Date.now() });
    });

    // Full graph refresh events
    knowledgeSocketIO.onMessage('graphiti_graph_refreshed', (data: { reason: string; metadata?: any }) => {
      console.log('🔄 Graph refreshed:', data);
      this.emit('graph_refreshed', data);
      this.emit('graph_update', { type: 'graph_refreshed', data, timestamp: Date.now() });
    });

    // Health status updates
    knowledgeSocketIO.onMessage('graphiti_health_changed', (data: { status: string; checks: any }) => {
      console.log('🏥 Graphiti health changed:', data);
      this.emit('health_changed', data);
      this.emit('graph_update', { type: 'health_changed', data, timestamp: Date.now() });
    });

    // Performance metrics updates
    knowledgeSocketIO.onMessage('graphiti_performance_update', (data: { metrics: any }) => {
      console.log('📊 Performance metrics updated:', data);
      this.emit('performance_update', data);
    });

    // Batch updates for efficiency
    knowledgeSocketIO.onMessage('graphiti_batch_update', (data: { 
      entities: EntityEvent[];
      relationships: RelationshipEvent[];
      metadata: any;
    }) => {
      console.log('📦 Batch update received:', data);
      this.emit('batch_update', data);
      this.emit('graph_update', { type: 'batch_update', data, timestamp: Date.now() });
    });
  }

  /**
   * Add event listener
   */
  on(event: string, callback: Function): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    
    this.listeners.get(event)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(event);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(event);
        }
      }
    };
  }

  /**
   * Remove event listener
   */
  off(event: string, callback: Function): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(callback);
      if (listeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Emit event to listeners
   */
  private emit(event: string, data: any): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in Graphiti WebSocket listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Request graph refresh from server
   */
  requestRefresh(reason: string = 'manual'): void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({ 
        type: 'graphiti_request_refresh', 
        data: { reason, timestamp: Date.now() }
      });
      console.log('🔄 Requested graph refresh:', reason);
    } else {
      console.warn('Cannot request refresh - WebSocket not connected');
    }
  }

  /**
   * Subscribe to entity updates for a specific entity
   */
  subscribeToEntity(entityId: string): () => void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({ 
        type: 'graphiti_subscribe_entity', 
        data: { entity_id: entityId }
      });
      console.log('👁️ Subscribed to entity updates:', entityId);
    }

    // Return unsubscribe function
    return () => {
      if (this.isConnected) {
        knowledgeSocketIO.sendMessage({ 
          type: 'graphiti_unsubscribe_entity', 
          data: { entity_id: entityId }
        });
        console.log('👁️‍🗨️ Unsubscribed from entity updates:', entityId);
      }
    };
  }

  /**
   * Subscribe to relationship updates for specific entities
   */
  subscribeToRelationships(entityIds: string[]): () => void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({ 
        type: 'graphiti_subscribe_relationships', 
        data: { entity_ids: entityIds }
      });
      console.log('🔗 Subscribed to relationship updates for entities:', entityIds);
    }

    // Return unsubscribe function
    return () => {
      if (this.isConnected) {
        knowledgeSocketIO.sendMessage({ 
          type: 'graphiti_unsubscribe_relationships', 
          data: { entity_ids: entityIds }
        });
        console.log('🔗💔 Unsubscribed from relationship updates');
      }
    };
  }

  /**
   * Request performance metrics update
   */
  requestPerformanceUpdate(): void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({ 
        type: 'graphiti_request_performance', 
        data: { timestamp: Date.now() }
      });
      console.log('📊 Requested performance metrics update');
    }
  }

  /**
   * Check connection status
   */
  isWebSocketConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Get active listeners count for debugging
   */
  getActiveListeners(): Record<string, number> {
    const counts: Record<string, number> = {};
    this.listeners.forEach((listeners, event) => {
      counts[event] = listeners.size;
    });
    return counts;
  }

  /**
   * Send graph filter preferences to server for optimized updates
   */
  updateFilterPreferences(filters: {
    entity_types?: string[];
    time_window?: string;
    confidence_threshold?: number;
    importance_threshold?: number;
  }): void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({ 
        type: 'graphiti_update_filters', 
        data: { filters, timestamp: Date.now() }
      });
      console.log('🔧 Updated server filter preferences:', filters);
    }
  }

  /**
   * Request specific graph data from server
   */
  requestGraphData(filters?: Record<string, any>): void {
    if (this.isConnected) {
      knowledgeSocketIO.sendMessage({
        type: 'graphiti_request_data',
        data: { filters: filters || {}, timestamp: Date.now() }
      });
      console.log('📡 Requested graph data with filters:', filters);
    }
  }

  /**
   * Cleanup all listeners and connections
   */
  cleanup(): void {
    console.log('🧹 Cleaning up Graphiti WebSocket service');
    this.listeners.clear();
    this.isConnected = false;
  }
}

// Export singleton instance
export const graphitiWebSocketService = new GraphitiWebSocketService();
export default graphitiWebSocketService;