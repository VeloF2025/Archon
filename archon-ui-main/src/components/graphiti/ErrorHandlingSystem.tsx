/**
 * Error Handling System for Graphiti Explorer
 * Provides comprehensive error boundaries, empty states, and network failure handling
 */

import React, { Component, ReactNode } from 'react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { 
  AlertTriangle, 
  RefreshCw, 
  Wifi, 
  WifiOff, 
  Database,
  Search,
  FileX,
  ServerCrash,
  Settings,
  ExternalLink
} from 'lucide-react';

// Error types and interfaces
export type ErrorType = 
  | 'network' 
  | 'data_loading' 
  | 'empty_state' 
  | 'permission' 
  | 'timeout' 
  | 'server_error'
  | 'component_error';

export interface ErrorInfo {
  type: ErrorType;
  message: string;
  details?: string;
  retryable: boolean;
  timestamp: Date;
  component?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<{ error: ErrorInfo; onRetry: () => void }>;
  onError?: (error: ErrorInfo) => void;
}

// Main Error Boundary Component
export class GraphitiErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    const errorInfo: ErrorInfo = {
      type: 'component_error',
      message: error.message || 'An unexpected error occurred',
      details: error.stack,
      retryable: true,
      timestamp: new Date(),
      component: 'GraphitiErrorBoundary'
    };

    return {
      hasError: true,
      error,
      errorInfo
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const enhancedError: ErrorInfo = {
      type: 'component_error',
      message: error.message,
      details: `${error.stack}\n\nComponent Stack:\n${errorInfo.componentStack}`,
      retryable: true,
      timestamp: new Date(),
      component: errorInfo.componentStack.split('\n')[1]?.trim() || 'Unknown'
    };

    this.props.onError?.(enhancedError);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError && this.state.errorInfo) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return <FallbackComponent error={this.state.errorInfo} onRetry={this.handleRetry} />;
    }

    return this.props.children;
  }
}

// Network Status Hook
export const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = React.useState(navigator.onLine);
  const [lastOnline, setLastOnline] = React.useState<Date | null>(null);

  React.useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setLastOnline(new Date());
    };
    
    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, lastOnline };
};

// Loading State Component
interface LoadingStateProps {
  message?: string;
  showSpinner?: boolean;
  timeout?: number;
  onTimeout?: () => void;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ 
  message = "Loading graph data...",
  showSpinner = true,
  timeout = 30000,
  onTimeout
}) => {
  React.useEffect(() => {
    if (timeout && onTimeout) {
      const timer = setTimeout(onTimeout, timeout);
      return () => clearTimeout(timer);
    }
  }, [timeout, onTimeout]);

  return (
    <div className="flex flex-col items-center justify-center h-64 text-gray-500">
      {showSpinner && (
        <RefreshCw className="w-8 h-8 animate-spin mb-4 text-blue-500" />
      )}
      <p className="text-lg font-medium mb-2">{message}</p>
      <p className="text-sm opacity-70">This may take a few moments...</p>
    </div>
  );
};

// Empty State Component
interface EmptyStateProps {
  type: 'no_data' | 'no_search_results' | 'no_entities' | 'no_relationships';
  onAction?: () => void;
  actionLabel?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ type, onAction, actionLabel }) => {
  const getEmptyStateConfig = () => {
    switch (type) {
      case 'no_data':
        return {
          icon: <Database className="w-12 h-12 text-gray-400" />,
          title: "No Graph Data Available",
          message: "The knowledge graph appears to be empty. Try connecting to a data source or refreshing the connection.",
          action: actionLabel || "Refresh Data"
        };
      case 'no_search_results':
        return {
          icon: <Search className="w-12 h-12 text-gray-400" />,
          title: "No Results Found",
          message: "Your search didn't match any entities or relationships. Try adjusting your search terms or clearing filters.",
          action: actionLabel || "Clear Search"
        };
      case 'no_entities':
        return {
          icon: <FileX className="w-12 h-12 text-gray-400" />,
          title: "No Entities Available",
          message: "No entities are currently visible with the selected filters and view mode. Try changing your view settings.",
          action: actionLabel || "Reset Filters"
        };
      case 'no_relationships':
        return {
          icon: <FileX className="w-12 h-12 text-gray-400" />,
          title: "No Relationships Found",
          message: "No relationships are available for the current selection. Try selecting different entities or expanding your view.",
          action: actionLabel || "Expand View"
        };
    }
  };

  const config = getEmptyStateConfig();

  return (
    <div className="flex flex-col items-center justify-center h-64 text-center">
      <div className="mb-4">
        {config.icon}
      </div>
      <h3 className="text-xl font-semibold text-gray-700 mb-2">
        {config.title}
      </h3>
      <p className="text-gray-500 mb-6 max-w-md">
        {config.message}
      </p>
      {onAction && (
        <Button onClick={onAction} variant="outline">
          {config.action}
        </Button>
      )}
    </div>
  );
};

// Network Error Component
interface NetworkErrorProps {
  error: ErrorInfo;
  onRetry: () => void;
  showDetails?: boolean;
}

export const NetworkError: React.FC<NetworkErrorProps> = ({ error, onRetry, showDetails = false }) => {
  const { isOnline } = useNetworkStatus();

  return (
    <Card className="border-red-200 bg-red-50">
      <CardHeader className="pb-3">
        <CardTitle className="text-red-800 flex items-center gap-2">
          {isOnline ? <Wifi className="w-5 h-5" /> : <WifiOff className="w-5 h-5" />}
          Connection Problem
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-red-700">
            {error.message}
          </p>
          
          {!isOnline && (
            <div className="bg-red-100 border border-red-200 rounded p-3">
              <p className="text-red-800 text-sm">
                ⚠️ You appear to be offline. Please check your internet connection.
              </p>
            </div>
          )}

          {showDetails && error.details && (
            <details className="text-sm">
              <summary className="cursor-pointer text-red-600 font-medium">
                Technical Details
              </summary>
              <pre className="mt-2 p-2 bg-red-100 rounded text-xs overflow-auto">
                {error.details}
              </pre>
            </details>
          )}

          <div className="flex gap-2">
            <Button 
              onClick={onRetry} 
              variant="default"
              disabled={!isOnline}
              className="bg-red-600 hover:bg-red-700"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Try Again
            </Button>
            
            {!isOnline && (
              <Button 
                variant="outline" 
                onClick={() => window.location.reload()}
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Reload Page
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Server Error Component
interface ServerErrorProps {
  error: ErrorInfo;
  onRetry: () => void;
}

export const ServerError: React.FC<ServerErrorProps> = ({ error, onRetry }) => {
  return (
    <Card className="border-orange-200 bg-orange-50">
      <CardHeader className="pb-3">
        <CardTitle className="text-orange-800 flex items-center gap-2">
          <ServerCrash className="w-5 h-5" />
          Server Error
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-orange-700">
            {error.message}
          </p>
          
          <div className="bg-orange-100 border border-orange-200 rounded p-3">
            <p className="text-orange-800 text-sm">
              The server encountered an unexpected error. Our team has been notified.
            </p>
          </div>

          <Button onClick={onRetry} variant="outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry Request
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

// Default Error Fallback Component
interface DefaultErrorFallbackProps {
  error: ErrorInfo;
  onRetry: () => void;
}

export const DefaultErrorFallback: React.FC<DefaultErrorFallbackProps> = ({ error, onRetry }) => {
  // Route to appropriate error component based on error type
  switch (error.type) {
    case 'network':
    case 'timeout':
      return <NetworkError error={error} onRetry={onRetry} showDetails />;
    
    case 'server_error':
      return <ServerError error={error} onRetry={onRetry} />;
    
    default:
      return (
        <Card className="border-red-200 bg-red-50">
          <CardHeader className="pb-3">
            <CardTitle className="text-red-800 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Something went wrong
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-red-700">
                {error.message}
              </p>
              
              <details className="text-sm">
                <summary className="cursor-pointer text-red-600 font-medium">
                  Error Details
                </summary>
                <div className="mt-2 space-y-2">
                  <div><strong>Type:</strong> {error.type}</div>
                  <div><strong>Component:</strong> {error.component || 'Unknown'}</div>
                  <div><strong>Time:</strong> {error.timestamp.toLocaleString()}</div>
                  {error.details && (
                    <pre className="p-2 bg-red-100 rounded text-xs overflow-auto">
                      {error.details}
                    </pre>
                  )}
                </div>
              </details>

              <div className="flex gap-2">
                <Button onClick={onRetry} variant="default">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
                
                <Button 
                  variant="outline" 
                  onClick={() => window.location.reload()}
                >
                  <Settings className="w-4 h-4 mr-2" />
                  Reload Page
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      );
  }
};

// Timeout Handler Hook
export const useTimeout = (callback: () => void, delay: number | null) => {
  const timeoutRef = React.useRef<number | null>(null);
  const savedCallback = React.useRef(callback);

  React.useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  React.useEffect(() => {
    const tick = () => savedCallback.current();

    if (typeof delay === 'number') {
      timeoutRef.current = window.setTimeout(tick, delay);
      return () => {
        if (timeoutRef.current) {
          window.clearTimeout(timeoutRef.current);
        }
      };
    }
  }, [delay]);

  const resetTimeout = React.useCallback(() => {
    if (timeoutRef.current) {
      window.clearTimeout(timeoutRef.current);
    }
  }, []);

  return resetTimeout;
};

// Error Context for global error state management
interface ErrorContextValue {
  errors: ErrorInfo[];
  addError: (error: ErrorInfo) => void;
  removeError: (timestamp: Date) => void;
  clearErrors: () => void;
}

const ErrorContext = React.createContext<ErrorContextValue | null>(null);

export const useErrorHandler = () => {
  const context = React.useContext(ErrorContext);
  if (!context) {
    throw new Error('useErrorHandler must be used within an ErrorProvider');
  }
  return context;
};

export const ErrorProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [errors, setErrors] = React.useState<ErrorInfo[]>([]);

  const addError = React.useCallback((error: ErrorInfo) => {
    setErrors(prev => [...prev, error]);
    
    // Auto-remove non-critical errors after 10 seconds
    if (error.type !== 'component_error') {
      setTimeout(() => {
        setErrors(prev => prev.filter(e => e.timestamp !== error.timestamp));
      }, 10000);
    }
  }, []);

  const removeError = React.useCallback((timestamp: Date) => {
    setErrors(prev => prev.filter(e => e.timestamp !== timestamp));
  }, []);

  const clearErrors = React.useCallback(() => {
    setErrors([]);
  }, []);

  const value = { errors, addError, removeError, clearErrors };

  return (
    <ErrorContext.Provider value={value}>
      {children}
    </ErrorContext.Provider>
  );
};