import { useEffect, useRef } from 'react';
import { WebSocketService } from '../services/socketIOService';
import { API_CONFIG } from '../config/api';

/**
 * Hook for managing the main Socket.IO connection
 * Returns a stable socket instance that persists across re-renders
 * 
 * @example
 * const socket = useSocket();
 * 
 * useEffect(() => {
 *   socket.addMessageHandler('event', handler);
 *   return () => socket.removeMessageHandler('event', handler);
 * }, [socket]);
 */
export function useSocket(): WebSocketService {
  const socketRef = useRef<WebSocketService | null>(null);
  
  // Create socket instance if it doesn't exist
  if (!socketRef.current) {
    socketRef.current = new WebSocketService(API_CONFIG.SOCKET_URL || 'http://localhost:8181');
  }
  
  useEffect(() => {
    const socket = socketRef.current;
    
    // Connect on mount if not already connected
    if (socket && !socket.isConnected()) {
      socket.connect('default');
    }
    
    // Cleanup on unmount
    return () => {
      // Note: We don't disconnect here as other components might be using it
      // The socket will disconnect when the app unmounts
    };
  }, []);
  
  return socketRef.current;
}

// Export a singleton instance for components that need a shared socket
let sharedSocket: WebSocketService | null = null;

export function getSharedSocket(): WebSocketService {
  if (!sharedSocket) {
    sharedSocket = new WebSocketService(API_CONFIG.SOCKET_URL || 'http://localhost:8181');
    sharedSocket.connect('default');
  }
  return sharedSocket;
}