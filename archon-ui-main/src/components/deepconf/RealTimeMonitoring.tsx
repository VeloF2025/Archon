/**
 * Real-Time Monitoring Component
 * Phase 7 DeepConf Integration - Updated to use Socket.IO
 * 
 * Live metrics visualization with Socket.IO connectivity,
 * real-time updates, and streaming data display
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Button } from '../ui/Button';
import { Toggle } from '../ui/Toggle';
import { Badge } from '../ui/badge';
import { 
  RealTimeData, 
  WebSocketMessage, 
  WebSocketConfig,
  SCWTMetrics,
  MetricType,
  TimeRange,
  DashboardConfig,
  ChartDataPoint
} from './types';

// Import Socket.IO client directly
import { io, Socket } from 'socket.io-client';

// Icons
const WifiIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M5 13a10 10 0 0 1 14 0M8.5 16.5a5 5 0 0 1 7 0M12 20h.01"/>
  </svg>
);

const WifiOffIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m2 2 20 20M8.5 16.5a5 5 0 0 1 7 0M2 8.82a15 15 0 0 1 4.17-2.65M10.66 5.01a15 15 0 0 1 11.72 4.09M8 11.5a10 10 0 0 1 12 0"/>
  </svg>
);

const ActivityIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="m22 12-4-4v3a32 32 0 0 0-28 0v3z"/>
  </svg>
);

const PauseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect width="4" height="16" x="6" y="4"/>
    <rect width="4" height="16" x="14" y="4"/>
  </svg>
);

const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polygon points="5,3 19,12 5,21"/>
  </svg>
);

const TrendingUpIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22,7 13.5,15.5 8.5,10.5 2,17"/>
    <polyline points="16,7 22,7 22,13"/>
  </svg>
);

interface RealTimeMonitoringProps {
  data?: RealTimeData;
  timeRange: TimeRange;
  selectedMetrics: MetricType[];
  config: DashboardConfig;
  webSocketUrl?: string;
}

// 🟢 WORKING: Ultra-aggressive NaN protection for RealTimeMonitoring
const safeNumber = (value: any, defaultValue: number = 0): number => {
  if (value === null || value === undefined) return defaultValue;
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    if (isNaN(parsed) || !isFinite(parsed)) return defaultValue;
    return parsed;
  }
  if (typeof value !== 'number') return defaultValue;
  if (isNaN(value) || !isFinite(value) || value === Infinity || value === -Infinity) {
    console.warn(`[RealTimeMonitoring] Invalid numeric value: ${value}, using default: ${defaultValue}`);
    return defaultValue;
  }
  return value;
};

const safeSVGCoord = (value: any, defaultValue: number = 0, context: string = 'unknown'): number => {
  const safe = safeNumber(value, defaultValue);
  if (safe < -10000 || safe > 10000) {
    console.warn(`[RealTimeMonitoring] Extreme coordinate in ${context}: ${safe}`);
    return Math.max(-1000, Math.min(1000, safe));
  }
  return safe;
};

// Default configuration for Socket.IO connection
const defaultWebSocketConfig: WebSocketConfig = {
  reconnectAttempts: 5,
  reconnectDelay: 3000,
  heartbeatInterval: 30000,
  timeout: 10000,
};

export const RealTimeMonitoring: React.FC<RealTimeMonitoringProps> = ({
  data,
  timeRange,
  selectedMetrics,
  config,
  webSocketUrl,
}) => {
  // State management
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting' | 'error'>('disconnected');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamData, setStreamData] = useState<ChartDataPoint[]>([]);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [messageCount, setMessageCount] = useState(0);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  
  // Refs
  const currentTaskId = useRef<string>('scwt_realtime_' + Date.now());
  const socketRef = useRef<Socket | null>(null);
  
  // Socket.IO connection management
  const connect = useCallback(async () => {
    if (connectionStatus === 'connected' && socketRef.current?.connected) {
      return;
    }
    
    setConnectionStatus('connecting');
    
    try {
      // Create direct Socket.IO connection to backend
      const socket = io('http://localhost:8181', {
        transports: ['polling', 'websocket'],
        timeout: 10000,
        forceNew: true
      });
      
      socketRef.current = socket;
      
      socket.on('connect', () => {
        console.log('Socket.IO connected for SCWT metrics streaming');
        setConnectionStatus('connected');
        setReconnectAttempts(0);
        
        // Subscribe to global confidence updates
        socket.emit('subscribe_global_confidence', {});
      });
      
      socket.on('disconnect', () => {
        console.log('Socket.IO disconnected');
        setConnectionStatus('disconnected');
      });
      
      socket.on('connect_error', (error) => {
        console.error('Socket.IO connection error:', error);
        setConnectionStatus('error');
        
        // Retry connection
        if (reconnectAttempts < defaultWebSocketConfig.reconnectAttempts) {
          setTimeout(() => {
            setReconnectAttempts(prev => prev + 1);
            connect();
          }, defaultWebSocketConfig.reconnectDelay);
        }
      });
      
      // Listen for confidence updates 
      socket.on('confidence_update', (data) => {
        if (!isPaused) {
          console.log('Received confidence update:', data);
          
          // Transform confidence update to metrics format
          const confidenceMessage: WebSocketMessage = {
            type: 'confidence',
            data: data,
            timestamp: new Date().toISOString()
          };
          
          setLastMessage(confidenceMessage);
          setMessageCount(prev => prev + 1);
          handleConfidenceUpdate(confidenceMessage);
        }
      });
      
      // Listen for task confidence updates (includes SCWT metrics)
      socket.on('task_confidence_update', (data) => {
        if (!isPaused) {
          console.log('Received task confidence update:', data);
          
          const taskMessage: WebSocketMessage = {
            type: 'metrics',
            data: data,
            timestamp: new Date().toISOString()
          };
          
          setLastMessage(taskMessage);
          setMessageCount(prev => prev + 1);
          handleMetricsUpdate(taskMessage);
        }
      });
      
    } catch (error) {
      console.error('Failed to create Socket.IO connection:', error);
      setConnectionStatus('error');
    }
  }, [selectedMetrics, timeRange, isPaused, reconnectAttempts]);
  
  // Disconnect Socket.IO
  const disconnect = useCallback(() => {
    try {
      if (socketRef.current) {
        // Unsubscribe from confidence updates
        socketRef.current.emit('unsubscribe_global_confidence', {});
        
        // Disconnect the socket
        socketRef.current.disconnect();
        socketRef.current = null;
      }
      
      console.log('Disconnected from SCWT metrics streaming');
      setConnectionStatus('disconnected');
      setIsStreaming(false);
    } catch (error) {
      console.error('Error disconnecting from Socket.IO:', error);
    }
  }, []);
  
  // Handle metrics updates
  const handleMetricsUpdate = useCallback((message: WebSocketMessage) => {
    try {
      // Extract confidence data and transform to SCWT metrics
      const confidenceData = message.data;
      
      if (!confidenceData) {
        console.warn('No confidence data in message:', message);
        return;
      }
      
      // Transform confidence score to SCWT-like metrics
      const confidence = safeNumber(confidenceData.confidence_score || confidenceData.overall, 0.5);
      const combinedScore = confidence; // Use confidence as combined score
      
      const metrics: SCWTMetrics = {
        structuralWeight: safeNumber(confidenceData.dimensions?.structural, confidence),
        contextWeight: safeNumber(confidenceData.dimensions?.contextual, confidence), 
        temporalWeight: safeNumber(confidenceData.dimensions?.temporal, confidence),
        combinedScore: combinedScore,
        confidence: confidence,
        timestamp: new Date(message.timestamp || Date.now()),
        task_id: confidenceData.task_id || currentTaskId.current,
        agent_id: confidenceData.agent_id || 'real_time_monitor'
      };
      
      setStreamData(prev => {
        const maxPoints = config?.dataRetention?.maxHistoryPoints || 50;
        const newData = [
          ...prev.slice(-(maxPoints - 1)),
          {
            timestamp: metrics.timestamp,
            value: metrics.combinedScore,
            confidence: metrics.confidence,
            metadata: {
              structuralWeight: metrics.structuralWeight,
              contextWeight: metrics.contextWeight,
              temporalWeight: metrics.temporalWeight,
            },
          } as ChartDataPoint,
        ];
        
        return newData;
      });
    } catch (error) {
      console.error('Error handling metrics update:', error);
    }
  }, [config]);
  
  // Handle performance updates
  const handlePerformanceUpdate = useCallback((message: WebSocketMessage) => {
    // Update performance metrics in real-time
    console.log('Performance update:', message.data);
  }, []);
  
  // Handle confidence updates (same as metrics for real-time display)
  const handleConfidenceUpdate = useCallback((message: WebSocketMessage) => {
    // Treat confidence updates as metrics updates for real-time visualization
    handleMetricsUpdate(message);
  }, [handleMetricsUpdate]);
  
  // Handle status updates
  const handleStatusUpdate = useCallback((message: WebSocketMessage) => {
    console.log('Status update:', message.data);
  }, []);
  
  // Start backend streaming
  const startBackendStream = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      
      // Start the backend confidence streaming
      const response = await fetch('/api/confidence/stream/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      console.log('Backend streaming started:', result);
      
      // Now connect to Socket.IO to receive the stream
      await connect();
      setIsStreaming(true);
      
    } catch (error) {
      console.error('Failed to start backend streaming:', error);
      setConnectionStatus('error');
    }
  }, [connect]);

  // Toggle streaming
  const toggleStreaming = useCallback(() => {
    if (isStreaming) {
      disconnect();
    } else {
      startBackendStream();
    }
  }, [isStreaming, disconnect, startBackendStream]);
  
  // Calculate streaming statistics
  const streamStats = useMemo(() => {
    if (!streamData.length) return null;
    
    const values = streamData.map(d => d.value);
    const timestamps = streamData.map(d => d.timestamp);
    
    return {
      dataPoints: streamData.length,
      latestValue: values[values.length - 1],
      averageValue: values.reduce((a, b) => a + b, 0) / values.length,
      minValue: Math.min(...values),
      maxValue: Math.max(...values),
      updateRate: streamData.length > 1 ? 
        1000 / ((timestamps[timestamps.length - 1].getTime() - timestamps[0].getTime()) / (timestamps.length - 1)) : 0,
    };
  }, [streamData]);
  
  // Render real-time chart with bulletproof NaN protection
  const renderRealtimeChart = () => {
    if (!streamData.length) {
      return (
        <div className="h-32 flex items-center justify-center text-muted-foreground">
          {connectionStatus === 'connected' ? 'Waiting for data...' : 'No real-time data available'}
        </div>
      );
    }
    
    const width = safeNumber(400, 400);
    const height = safeNumber(120, 120);
    const padding = safeNumber(20, 20);
    
    // Ultra-safe scaling functions
    const xScale = (index: number) => {
      const safeIndex = safeNumber(index, 0);
      const maxIndex = Math.max(1, streamData.length - 1);
      const scaleFactor = maxIndex > 0 ? safeIndex / maxIndex : 0;
      const result = safeSVGCoord(scaleFactor * (width - 2 * padding) + padding, padding, 'realtime-xScale');
      return Math.max(padding, Math.min(width - padding, result));
    };
    
    const yScale = (value: number) => {
      const safeValue = safeNumber(value, 0.5);
      const normalizedValue = Math.max(0, Math.min(1, safeValue)); // Clamp to 0-1
      const result = safeSVGCoord(height - padding - (normalizedValue * (height - 2 * padding)), height - padding, 'realtime-yScale');
      return Math.max(padding, Math.min(height - padding, result));
    };
    
    try {
      // Generate ultra-safe polyline points
      const points = streamData.map((d, i) => {
        const x = safeSVGCoord(xScale(i), padding, `realtime-point-x-${i}`);
        const y = safeSVGCoord(yScale(safeNumber(d?.value, 0.5)), height - padding, `realtime-point-y-${i}`);
        return `${x},${y}`;
      }).filter(point => point && !point.includes('NaN') && !point.includes('Infinity')).join(' ');
      
      // Fallback if no valid points
      if (!points || points.trim() === '') {
        console.warn('[RealTimeMonitoring] No valid points generated, using fallback');
        const fallbackPoints = `${padding},${height - padding} ${width - padding},${height - padding}`;
        return (
          <div className="h-32 flex items-center justify-center">
            <svg width={width} height={height} className="w-full h-32">
              <polyline points={fallbackPoints} fill="none" stroke="rgb(156, 163, 175)" strokeWidth="2" />
              <text x={width/2} y={height/2} textAnchor="middle" className="text-xs fill-muted-foreground">Invalid data</text>
            </svg>
          </div>
        );
      }
      
      return (
        <svg width={width} height={height} className="w-full h-32">
          {/* Grid */}
          <defs>
            <pattern id="realtimeGrid" width="20" height="10" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#realtimeGrid)" />
          
          {/* Data line */}
          <polyline
            points={points}
            fill="none"
            stroke="rgb(34, 197, 94)"
            strokeWidth="2"
            className="drop-shadow-sm"
          />
          
          {/* Current value indicator */}
          {streamData.length > 0 && (() => {
            const lastIndex = streamData.length - 1;
            const lastData = streamData[lastIndex];
            const cx = safeSVGCoord(xScale(lastIndex), width - padding, 'realtime-circle-cx');
            const cy = safeSVGCoord(yScale(safeNumber(lastData?.value, 0.5)), height - padding, 'realtime-circle-cy');
            
            // Validate circle coordinates
            if (cx >= 0 && cx <= width && cy >= 0 && cy <= height) {
              return (
                <circle
                  cx={cx}
                  cy={cy}
                  r="4"
                  fill="rgb(34, 197, 94)"
                  stroke="white"
                  strokeWidth="2"
                  className="animate-pulse"
                />
              );
            }
            return null;
          })()}
          
          {/* Axes */}
          <line 
            x1={safeSVGCoord(padding, padding, 'axis-x1')} 
            y1={safeSVGCoord(height - padding, height - padding, 'axis-y1')} 
            x2={safeSVGCoord(width - padding, width - padding, 'axis-x2')} 
            y2={safeSVGCoord(height - padding, height - padding, 'axis-y2')} 
            stroke="currentColor" 
            opacity="0.3" 
          />
          <line 
            x1={safeSVGCoord(padding, padding, 'axis-v-x1')} 
            y1={safeSVGCoord(padding, padding, 'axis-v-y1')} 
            x2={safeSVGCoord(padding, padding, 'axis-v-x2')} 
            y2={safeSVGCoord(height - padding, height - padding, 'axis-v-y2')} 
            stroke="currentColor" 
            opacity="0.3" 
          />
        </svg>
      );
    } catch (error) {
      console.error('[RealTimeMonitoring] Error rendering chart:', error);
      return (
        <div className="h-32 flex items-center justify-center text-red-500">
          Chart rendering error - see console for details
        </div>
      );
    }
  };
  
  // Connection status indicator
  const getConnectionStatusBadge = () => {
    const statusConfig = {
      connected: { color: 'bg-green-500', text: 'Connected', icon: <WifiIcon /> },
      connecting: { color: 'bg-yellow-500', text: 'Connecting', icon: <WifiIcon /> },
      disconnected: { color: 'bg-gray-500', text: 'Disconnected', icon: <WifiOffIcon /> },
      error: { color: 'bg-red-500', text: 'Error', icon: <WifiOffIcon /> },
    };
    
    const config = statusConfig[connectionStatus];
    
    return (
      <Badge variant="outline" className={`${config.color} text-white border-transparent`}>
        <span className="flex items-center gap-1">
          {config.icon}
          {config.text}
        </span>
      </Badge>
    );
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, []);
  

  return (
    <Card accentColor="green">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="flex items-center gap-2">
              <ActivityIcon />
              Real-Time Monitoring
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Live SCWT metrics with WebSocket streaming
            </p>
          </div>
          
          <div className="flex items-center gap-2">
            {getConnectionStatusBadge()}
            
            <Toggle
              pressed={isPaused}
              onPressedChange={setIsPaused}
              size="sm"
              disabled={connectionStatus !== 'connected'}
            >
              {isPaused ? <PlayIcon /> : <PauseIcon />}
            </Toggle>
            
            <Button
              onClick={toggleStreaming}
              variant={isStreaming ? "destructive" : "default"}
              size="sm"
              className="gap-2"
            >
              {isStreaming ? 'Stop' : 'Start'} Stream
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Connection Info */}
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-md">
          <div className="flex items-center gap-4">
            <div>
              <span className="text-sm font-medium">Messages:</span>
              <span className="ml-2 text-lg font-bold text-green-600">
                {messageCount.toLocaleString()}
              </span>
            </div>
            
            {reconnectAttempts > 0 && (
              <div>
                <span className="text-sm font-medium">Reconnect Attempts:</span>
                <span className="ml-2 text-sm text-yellow-600">
                  {reconnectAttempts}/{defaultWebSocketConfig.reconnectAttempts}
                </span>
              </div>
            )}
          </div>
          
          {streamStats && (
            <div className="text-right text-sm">
              <div>Update Rate: {streamStats.updateRate.toFixed(1)} Hz</div>
              <div className="text-muted-foreground">
                {streamStats.dataPoints} data points
              </div>
            </div>
          )}
        </div>
        
        {/* Real-time Chart */}
        <div>
          <h4 className="font-medium mb-2 flex items-center gap-2">
            <TrendingUpIcon />
            Live Metrics Stream
          </h4>
          {renderRealtimeChart()}
        </div>
        
        {/* Current Values */}
        {streamStats && (
          <div>
            <h4 className="font-medium mb-3">Current Values</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {streamStats.latestValue.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">Latest</div>
              </div>
              
              <div className="text-center">
                <div className="text-xl font-semibold">
                  {streamStats.averageValue.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">Average</div>
              </div>
              
              <div className="text-center">
                <div className="text-xl font-semibold text-blue-600">
                  {streamStats.maxValue.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">Peak</div>
              </div>
              
              <div className="text-center">
                <div className="text-xl font-semibold text-orange-600">
                  {streamStats.minValue.toFixed(3)}
                </div>
                <div className="text-xs text-muted-foreground">Low</div>
              </div>
            </div>
          </div>
        )}
        
        {/* Last Message Info */}
        {lastMessage && (
          <div className="p-3 bg-muted/30 rounded-md">
            <div className="flex justify-between items-center mb-2">
              <h5 className="font-medium text-sm">Latest Message</h5>
              <span className="text-xs text-muted-foreground">
                {new Date(lastMessage.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-xs font-mono bg-black text-green-400 p-2 rounded overflow-x-auto">
              {JSON.stringify(lastMessage, null, 2)}
            </div>
          </div>
        )}
        
        {/* Connection Error */}
        {connectionStatus === 'error' && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <div className="text-red-600 dark:text-red-300 text-sm">
              Failed to connect to WebSocket server. Please check the connection and try again.
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RealTimeMonitoring;