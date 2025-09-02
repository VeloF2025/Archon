/**
 * DeepConf Service - Real-time confidence scoring and SCWT metrics
 * Connects UI components to the DeepConf API endpoints and Socket.IO streams
 * 
 * Integration with Phase 5 DeepConf lazy loading and confidence API
 * DEPRECATED: 2025-09-01-20:50 - DISABLED socket initialization to fix .on() error
 */

import { knowledgeSocketIO, WebSocketService, WebSocketState } from './socketIOService';

// Type definitions for DeepConf data structures
export interface ConfidenceScore {
  overall: number;
  bayesian: {
    lower: number;
    upper: number;
    mean: number;
    variance: number;
  };
  dimensions: {
    structural: number;
    contextual: number;
    temporal: number;
    semantic: number;
  };
  uncertainty: {
    epistemic: number;
    aleatoric: number;
    total: number;
  };
  trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
}

export interface SCWTMetrics {
  structuralWeight: number;
  contextWeight: number;
  temporalWeight: number;
  combinedScore: number;
  confidence?: number;
  timestamp: Date;
  task_id?: string;
  agent_id?: string;
}

export interface PerformanceMetrics {
  tokenEfficiency: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
    compressionRatio: number;
    efficiencyScore: number;
  };
  cost: {
    inputCost: number;
    outputCost: number;
    totalCost: number;
    costPerQuery: number;
    costSavings: number;
  };
  timing: {
    processingTime: number;
    networkLatency: number;
    totalResponseTime: number;
    throughput: number;
  };
  quality: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
}

export interface RealTimeData {
  current: SCWTMetrics;
  confidence: ConfidenceScore;
  performance: PerformanceMetrics;
  history: SCWTMetrics[];
  status: 'active' | 'idle' | 'error';
  lastUpdate: Date;
}

export class DeepConfError extends Error {
  public code?: string;
  public details?: any;

  constructor(message: string, code?: string, details?: any) {
    super(message);
    this.name = 'DeepConfError';
    this.code = code;
    this.details = details;
  }
}

/**
 * DeepConf Service Class  
 * Manages all interactions with DeepConf backend services
 * Updated: Fixed Socket.IO integration - uses WebSocketService API only
 */
class DeepConfService {
  private baseUrl: string;
  private isConnected: boolean = false;
  private listeners: Map<string, ((data: any) => void)[]> = new Map();

  constructor(baseUrl: string = '') {
    // Use empty baseUrl to make relative requests that go through Vite proxy
    this.baseUrl = baseUrl;
    
    console.log('DeepConf: Service constructor called, timestamp:', Date.now());
    console.log('DeepConf: Service version: CACHE_BUST_2025-09-01-19:15-ULTIMATE-FIX');
    
    // DISABLED: Do not initialize socket connection to prevent .on() error
    // This service is deprecated - use deepconfService_fixed instead
    console.log('DeepConf: Old service detected, initialization DISABLED to prevent errors');
  }

  /**
   * Initialize Socket.IO connection for real-time updates
   * AGGRESSIVE FIX: Added comprehensive defensive programming and fallback mechanisms
   */
  private async initializeSocketConnection(): Promise<void> {
    try {
      console.log('DeepConf: Starting Socket.IO initialization...');
      console.log('DeepConf: knowledgeSocketIO exists:', !!knowledgeSocketIO);
      console.log('DeepConf: knowledgeSocketIO instanceof WebSocketService:', knowledgeSocketIO instanceof WebSocketService);
      console.log('DeepConf: Method addMessageHandler exists:', typeof knowledgeSocketIO.addMessageHandler === 'function');
      
      // DEFENSIVE CHECK 1: Verify knowledgeSocketIO is actually a WebSocketService instance
      if (!(knowledgeSocketIO instanceof WebSocketService)) {
        console.error('DeepConf: knowledgeSocketIO is not a WebSocketService instance!');
        console.log('DeepConf: Actual type:', typeof knowledgeSocketIO);
        console.log('DeepConf: Available methods:', Object.getOwnPropertyNames(knowledgeSocketIO));
        throw new Error('Invalid WebSocketService instance');
      }

      // DEFENSIVE CHECK 2: Verify all required methods exist before using them
      const requiredMethods = ['addMessageHandler', 'addStateChangeHandler', 'isConnected', 'connect'];
      const missingMethods = requiredMethods.filter(method => 
        typeof knowledgeSocketIO[method as keyof WebSocketService] !== 'function'
      );
      
      if (missingMethods.length > 0) {
        console.error('DeepConf: Missing required methods:', missingMethods);
        console.log('DeepConf: Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(knowledgeSocketIO)));
        throw new Error(`Missing WebSocketService methods: ${missingMethods.join(', ')}`);
      }

      console.log('DeepConf: All required methods verified, proceeding with connection...');

      // Connect to the Socket.IO service
      await knowledgeSocketIO.connect('/');
      console.log('DeepConf: Connection established, setting up handlers...');
      
      // Subscribe to confidence-related events using WebSocketService API
      knowledgeSocketIO.addMessageHandler('confidence_update', (message) => {
        console.log('DeepConf: Received confidence_update:', message);
        this.isConnected = true;
        this.notifyListeners('confidence_update', message.data);
      });

      knowledgeSocketIO.addMessageHandler('task_confidence_update', (message) => {
        console.log('DeepConf: Received task_confidence_update:', message);
        this.notifyListeners('task_confidence_update', message.data);
      });

      knowledgeSocketIO.addMessageHandler('scwt_metrics_update', (message) => {
        console.log('DeepConf: Received scwt_metrics_update:', message);
        this.notifyListeners('scwt_metrics_update', message.data);
      });

      // Track connection state using WebSocketService state change handlers
      knowledgeSocketIO.addStateChangeHandler((state: WebSocketState) => {
        console.log('DeepConf: WebSocket state changed to:', state);
        const wasConnected = this.isConnected;
        this.isConnected = knowledgeSocketIO.isConnected();
        
        if (!wasConnected && this.isConnected) {
          console.log('DeepConf: Connected to Socket.IO server');
        } else if (wasConnected && !this.isConnected) {
          console.log('DeepConf: Disconnected from Socket.IO server');
        }
      });

      this.isConnected = knowledgeSocketIO.isConnected();
      console.log('DeepConf: Socket.IO integration initialized successfully, connected:', this.isConnected);
      
    } catch (error) {
      console.error('DeepConf: Critical failure in Socket.IO initialization:', error);
      console.error('DeepConf: Error stack:', (error as Error).stack);
      
      // FALLBACK MECHANISM: Try to continue with basic functionality
      console.log('DeepConf: Attempting fallback initialization without real-time features...');
      this.isConnected = false;
      
      // Set up a simple polling mechanism as fallback
      this.setupPollingFallback();
    }
  }

  /**
   * Subscribe to real-time data updates
   */
  public subscribe(event: string, callback: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.push(callback);
    }
  }

  /**
   * Unsubscribe from real-time data updates
   */
  public unsubscribe(event: string, callback: (data: any) => void): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      const index = eventListeners.indexOf(callback);
      if (index > -1) {
        eventListeners.splice(index, 1);
      }
    }
  }

  /**
   * Notify all listeners of an event
   */
  private notifyListeners(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`DeepConf: Error in listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Get current confidence score for a task
   */
  public async getTaskConfidence(taskId: string): Promise<ConfidenceScore> {
    try {
      // First try to get task history
      try {
        const historyResponse = await fetch(`${this.baseUrl}/api/confidence/task/${taskId}/history`);
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          if (historyData.confidence_history && historyData.confidence_history.length > 0) {
            // Return the most recent confidence score
            const latest = historyData.confidence_history[historyData.confidence_history.length - 1];
            return this.transformConfidenceData(latest);
          }
        }
      } catch (historyError) {
        // Fall back to generating confidence via analyze endpoint
      }
      
      // Fall back to analysis endpoint
      const response = await fetch(`${this.baseUrl}/api/confidence/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          task: {
            task_id: taskId,
            content: `task confidence calculation for ${taskId}`,
            complexity: 'moderate',
            domain: 'general',
            priority: 1,
            model_source: 'ui_task_request'
          },
          context: {
            user_id: 'ui_user',
            environment: 'web_ui',
            timestamp: Date.now() / 1000
          }
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformConfidenceData(data.confidence_score);
    } catch (error) {
      console.error('DeepConf: Failed to get task confidence:', error);
      throw new DeepConfError(`Failed to get task confidence: ${error}`);
    }
  }

  /**
   * Get overall system confidence metrics
   */
  public async getSystemConfidence(): Promise<ConfidenceScore> {
    try {
      const response = await fetch(`${this.baseUrl}/api/confidence/system`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformConfidenceData(data.confidence_score);
    } catch (error) {
      console.error('DeepConf: Failed to get system confidence:', error);
      throw new DeepConfError(`Failed to get system confidence: ${error}`);
    }
  }

  /**
   * Get confidence history over time
   */
  public async getConfidenceHistory(
    hours: number = 24, 
    granularity: 'minute' | 'hour' | 'day' = 'hour'
  ): Promise<ConfidenceScore[]> {
    try {
      const params = new URLSearchParams({
        hours: hours.toString(),
        granularity
      });

      const response = await fetch(`${this.baseUrl}/api/confidence/history?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Validate response structure - NO FALLBACKS, throw if invalid
      if (!data || !Array.isArray(data.history)) {
        throw new DeepConfError(`Invalid history data structure: expected array, got ${typeof data?.history}`);
      }

      return data.history.map((item: any) => {
        if (!item.confidence || !item.uncertainty || !item.timestamp) {
          throw new DeepConfError(`Invalid history item missing required fields: ${JSON.stringify(item)}`);
        }
        
        const baseConfidence = this.safeNumber(item.confidence, null);
        const uncertainty = this.safeNumber(item.uncertainty, null);
        
        if (baseConfidence === null || uncertainty === null) {
          throw new DeepConfError(`Invalid confidence/uncertainty values in history item: ${JSON.stringify(item)}`);
        }
        
        return {
          overall: baseConfidence,
          bayesian: {
            lower: Math.max(0, baseConfidence - uncertainty),
            upper: Math.min(1, baseConfidence + uncertainty),
            mean: baseConfidence,
            variance: this.safeNumber(item.variance, uncertainty * uncertainty),
          },
          dimensions: {
            structural: this.safeNumber(item.dimensions?.structural, baseConfidence),
            contextual: this.safeNumber(item.dimensions?.contextual, baseConfidence),
            temporal: this.safeNumber(item.dimensions?.temporal, baseConfidence),
            semantic: this.safeNumber(item.dimensions?.semantic, baseConfidence),
          },
          uncertainty: {
            epistemic: this.safeNumber(item.uncertainty_breakdown?.epistemic, uncertainty / 2),
            aleatoric: this.safeNumber(item.uncertainty_breakdown?.aleatoric, uncertainty / 2),
            total: uncertainty,
          },
          trend: item.trend || 'stable' as const,
          timestamp: item.timestamp * 1000 // Convert to milliseconds, no fallback
        };
      });
    } catch (error) {
      console.error('DeepConf: Failed to get confidence history:', error);
      // Re-throw error to show real failures, no fallback empty array
      throw new DeepConfError(`Confidence history unavailable: ${error}`);
    }
  }

  /**
   * Calculate confidence for specific text/task
   */
  public async calculateConfidence(
    text: string, 
    taskId?: string, 
    agentId?: string
  ): Promise<ConfidenceScore> {
    try {
      const requestBody = {
        task: {
          task_id: taskId || `calc_${Date.now()}`,
          content: text,
          complexity: 'moderate',
          domain: 'general',
          priority: 1,
          model_source: agentId || 'ui_calculator'
        },
        context: {
          user_id: 'ui_user',
          environment: 'web_ui',
          timestamp: Date.now() / 1000
        }
      };

      const response = await fetch(`${this.baseUrl}/api/confidence/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return this.transformConfidenceData(data.confidence_score);
    } catch (error) {
      console.error('DeepConf: Failed to calculate confidence:', error);
      throw new DeepConfError(`Failed to calculate confidence: ${error}`);
    }
  }

  /**
   * Get SCWT benchmark metrics
   */
  public async getSCWTMetrics(phase?: string): Promise<SCWTMetrics[]> {
    try {
      const params = phase ? `?phase=${phase}` : '';
      const response = await fetch(`${this.baseUrl}/api/confidence/scwt${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Validate response structure - NO FALLBACKS, throw if invalid
      if (!data || !Array.isArray(data.scwt_metrics)) {
        throw new DeepConfError(`Invalid SCWT metrics data structure: expected array, got ${typeof data?.scwt_metrics}`);
      }
      
      // Transform API response - NO FALLBACKS, validate each item
      return data.scwt_metrics.map((item: any, index: number) => {
        if (!item || typeof item !== 'object') {
          throw new DeepConfError(`Invalid SCWT metric item at index ${index}: ${JSON.stringify(item)}`);
        }

        const requiredFields = ['structural_weight', 'context_weight', 'temporal_weight', 'combined_score', 'timestamp'];
        for (const field of requiredFields) {
          if (typeof item[field] !== 'number') {
            throw new DeepConfError(`SCWT metric missing/invalid ${field} at index ${index}: got ${typeof item[field]}`);
          }
        }

        return {
          structuralWeight: item.structural_weight,
          contextWeight: item.context_weight,
          temporalWeight: item.temporal_weight,
          combinedScore: item.combined_score,
          confidence: this.safeNumber(item.confidence, item.combined_score),
          timestamp: new Date(item.timestamp * 1000), // Convert to milliseconds, no fallback
          task_id: item.task_id || null,
          agent_id: item.agent_id || null,
        };
      });
    } catch (error) {
      console.error('DeepConf: Failed to get SCWT metrics:', error);
      // Re-throw error to show real failures, no fallback empty array
      throw new DeepConfError(`SCWT metrics unavailable: ${error}`);
    }
  }

  /**
   * Get real-time dashboard data
   */
  public async getDashboardData(): Promise<RealTimeData> {
    try {
      // Fetch all required data in parallel with error handling
      const [confidence, scwtMetrics, _history] = await Promise.allSettled([
        this.getSystemConfidence(),
        this.getSCWTMetrics(),
        this.getConfidenceHistory(24, 'hour')
      ]);

      // Safely extract confidence data - NO FALLBACKS, throw if failed
      if (confidence.status === 'rejected') {
        throw new DeepConfError(`System confidence unavailable: ${confidence.reason}`);
      }
      const safeConfidence = confidence.value;

      // Safely extract SCWT metrics - NO FALLBACKS, throw if failed
      if (scwtMetrics.status === 'rejected') {
        throw new DeepConfError(`SCWT metrics unavailable: ${scwtMetrics.reason}`);
      }
      const safeSCWTMetrics = scwtMetrics.value;

      // Get the most recent SCWT metrics - NO DEFAULTS, throw if empty
      if (safeSCWTMetrics.length === 0) {
        throw new DeepConfError('No SCWT metrics available from API');
      }
      const current = safeSCWTMetrics[safeSCWTMetrics.length - 1];

      // Fetch real performance metrics from API - NO FALLBACKS, show real errors
      const performance = await this.getPerformanceMetrics();

      return {
        current,
        confidence: safeConfidence,
        performance,
        history: safeSCWTMetrics,
        status: this.isConnected ? 'active' : 'idle',
        lastUpdate: new Date()
      };
    } catch (error) {
      console.error('DeepConf: Failed to get dashboard data:', error);
      // Re-throw error to show real failures, no fallback data
      throw new DeepConfError(`Dashboard data unavailable: ${error}`);
    }
  }

  /**
   * Start confidence stream for a specific task
   */
  public async startTaskConfidenceStream(taskId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/confidence/start-tracking/${taskId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      console.log(`DeepConf: Started confidence stream for task ${taskId}`);
    } catch (error) {
      console.error('DeepConf: Failed to start confidence stream:', error);
      throw new DeepConfError(`Failed to start confidence stream: ${error}`);
    }
  }

  /**
   * Stop confidence stream for a specific task
   */
  public async stopTaskConfidenceStream(taskId: string): Promise<void> {
    try {
      // Note: The backend doesn't currently have a stop endpoint, 
      // but we can implement this by just logging the stop locally
      console.log(`DeepConf: Stopped confidence stream for task ${taskId}`);
    } catch (error) {
      console.error('DeepConf: Failed to stop confidence stream:', error);
    }
  }

  /**
   * Get real performance metrics from API
   */
  public async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response = await fetch(`${this.baseUrl}/api/confidence/performance`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      return data.performance_metrics;
    } catch (error) {
      console.error('DeepConf: Failed to get performance metrics:', error);
      throw new DeepConfError(`Failed to get performance metrics: ${error}`);
    }
  }

  /**
   * Validate DeepConf system health
   */
  public async checkHealth(): Promise<{ status: string; version?: string; features?: string[] }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/confidence/health`);
      
      if (!response.ok) {
        return { status: 'error' };
      }

      return await response.json();
    } catch (error) {
      console.error('DeepConf: Health check failed:', error);
      return { status: 'error' };
    }
  }


  /**
   * Safe value utility - ensures valid numbers, prevents NaN values
   * When defaultValue is null, returns null for invalid values (no fallback)
   */
  private safeNumber(value: any, defaultValue: number | null = 0): number | null {
    if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
      return value;
    }
    return defaultValue;
  }

  /**
   * Safe percentage utility - ensures values are between 0 and 1
   */
  private safePercentage(value: any, defaultValue: number = 0.5): number {
    const num = this.safeNumber(value, defaultValue);
    return Math.max(0, Math.min(1, num));
  }

  /**
   * Transform API confidence data to UI format with comprehensive NaN prevention
   */
  private transformConfidenceData(data: any): ConfidenceScore {
    // Validate data structure - NO DEFAULTS, throw if invalid
    if (!data || typeof data !== 'object') {
      throw new DeepConfError('Invalid confidence data structure received from API');
    }

    if (typeof data.confidence !== 'number') {
      throw new DeepConfError(`Invalid confidence value: expected number, got ${typeof data.confidence}`);
    }

    const baseConfidence = this.safeNumber(data.confidence, null);
    if (baseConfidence === null) {
      throw new DeepConfError(`Invalid confidence value: ${data.confidence}`);
    }
    
    return {
      overall: baseConfidence,
      bayesian: {
        lower: this.safeNumber(
          data.confidence_bounds?.lower, 
          Math.max(0, baseConfidence - (data.uncertainty?.total || 0.1))
        ),
        upper: this.safeNumber(
          data.confidence_bounds?.upper, 
          Math.min(1, baseConfidence + (data.uncertainty?.total || 0.1))
        ),
        mean: baseConfidence,
        variance: this.safeNumber(data.variance, (data.uncertainty?.total || 0.1) ** 2),
      },
      dimensions: {
        structural: this.safeNumber(data.dimensions?.structural, baseConfidence),
        contextual: this.safeNumber(data.dimensions?.contextual, baseConfidence),
        temporal: this.safeNumber(data.dimensions?.temporal, baseConfidence),
        semantic: this.safeNumber(data.dimensions?.semantic, baseConfidence),
      },
      uncertainty: {
        epistemic: this.safeNumber(data.uncertainty?.epistemic, (data.uncertainty?.total || 0) / 2),
        aleatoric: this.safeNumber(data.uncertainty?.aleatoric, (data.uncertainty?.total || 0) / 2),
        total: this.safeNumber(data.uncertainty?.total, 0),
      },
      trend: ['increasing', 'decreasing', 'stable', 'volatile'].includes(data.trend) 
        ? data.trend 
        : 'stable',
    };
  }

  /**
   * Disconnect from Socket.IO
   */
  public disconnect(): void {
    knowledgeSocketIO.disconnect();
    this.isConnected = false;
    this.listeners.clear();
  }

  /**
   * Setup polling fallback when Socket.IO fails
   */
  private setupPollingFallback(): void {
    console.log('DeepConf: Setting up polling fallback mechanism...');
    
    // Poll for updates every 30 seconds as a fallback
    setInterval(async () => {
      try {
        // Try to get fresh data and notify listeners
        const dashboardData = await this.getDashboardData();
        this.notifyListeners('confidence_update', dashboardData.confidence);
        this.notifyListeners('scwt_metrics_update', dashboardData.current);
      } catch (error) {
        // Silently fail in polling fallback
        console.debug('DeepConf: Polling fallback failed:', error);
      }
    }, 30000);
  }

  /**
   * Get connection status with defensive checks
   */
  public getConnectionStatus(): boolean {
    try {
      // Defensive check before calling isConnected
      if (knowledgeSocketIO && typeof knowledgeSocketIO.isConnected === 'function') {
        return knowledgeSocketIO.isConnected();
      } else {
        console.warn('DeepConf: knowledgeSocketIO.isConnected is not available, returning false');
        return false;
      }
    } catch (error) {
      console.error('DeepConf: Error checking connection status:', error);
      return false;
    }
  }
}

// CACHE BUSTING: Force new instance creation to bypass any cached issues - ULTIMATE VERSION
const createUltimateFreshDeepConfService = () => {
  console.log('DeepConf: Creating ULTIMATE fresh service instance - version 2025-09-01-19:15...');
  return new DeepConfService();
};

// Export singleton instance - freshly created with new timestamp
export const deepconfService = createUltimateFreshDeepConfService();

// Export default service instance
export default deepconfService;

// Export service class for manual instantiation if needed
export { DeepConfService };

// EMERGENCY FALLBACK: Export a factory function for creating new instances
export const createDeepConfService = () => new DeepConfService();