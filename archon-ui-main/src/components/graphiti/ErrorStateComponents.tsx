import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { 
  AlertTriangle, 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  Search, 
  Database, 
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  FileX,
  Server,
  Zap,
  TrendingDown
} from 'lucide-react';

// Loading States
interface LoadingStateProps {
  type?: 'initial' | 'refresh' | 'search' | 'filter';
  message?: string;
  progress?: number;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  type = 'initial',
  message,
  progress,
  className
}) => {
  const getLoadingConfig = () => {
    switch (type) {
      case 'initial':
        return {
          icon: <Database className="h-8 w-8 text-blue-600" />,
          title: 'Loading Knowledge Graph',
          defaultMessage: 'Fetching entities and relationships...',
          color: 'blue'
        };
      case 'refresh':
        return {
          icon: <RefreshCw className="h-6 w-6 text-green-600 animate-spin" />,
          title: 'Refreshing Data',
          defaultMessage: 'Updating graph with latest changes...',
          color: 'green'
        };
      case 'search':
        return {
          icon: <Search className="h-6 w-6 text-purple-600" />,
          title: 'Searching',
          defaultMessage: 'Finding matching entities...',
          color: 'purple'
        };
      case 'filter':
        return {
          icon: <Zap className="h-6 w-6 text-orange-600" />,
          title: 'Applying Filters',
          defaultMessage: 'Processing filter criteria...',
          color: 'orange'
        };
      default:
        return {
          icon: <Loader2 className="h-6 w-6 text-gray-600 animate-spin" />,
          title: 'Loading',
          defaultMessage: 'Please wait...',
          color: 'gray'
        };
    }
  };

  const config = getLoadingConfig();
  const isCompact = type === 'refresh' || type === 'search' || type === 'filter';

  if (isCompact) {
    return (
      <div className={cn("flex items-center space-x-2 p-2 bg-white rounded-lg shadow-sm border", className)}>
        {config.icon}
        <span className="text-sm font-medium text-gray-700">
          {message || config.defaultMessage}
        </span>
      </div>
    );
  }

  return (
    <div className={cn("flex items-center justify-center w-full h-full min-h-[200px]", className)}>
      <Card className="w-full max-w-sm">
        <CardContent className="p-8 text-center">
          <div className="flex justify-center mb-4">
            {config.icon}
          </div>
          
          <CardTitle className="text-lg font-semibold text-gray-900 mb-2">
            {config.title}
          </CardTitle>
          
          <p className="text-sm text-gray-600 mb-6">
            {message || config.defaultMessage}
          </p>

          {progress !== undefined && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <div className="text-xs text-gray-500">
                {progress}% complete
              </div>
            </div>
          )}

          <div className="flex justify-center space-x-1 mt-4">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className={cn(
                  "w-2 h-2 rounded-full animate-pulse",
                  `bg-${config.color}-500`
                )}
                style={{ animationDelay: `${i * 0.2}s` }}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Error States
interface ErrorStateProps {
  type?: 'network' | 'server' | 'data' | 'timeout' | 'permission' | 'general';
  title?: string;
  message?: string;
  details?: string;
  onRetry?: () => void;
  onSupport?: () => void;
  showDetails?: boolean;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  type = 'general',
  title,
  message,
  details,
  onRetry,
  onSupport,
  showDetails = false,
  className
}) => {
  const [showFullDetails, setShowFullDetails] = React.useState(false);

  const getErrorConfig = () => {
    switch (type) {
      case 'network':
        return {
          icon: <WifiOff className="h-8 w-8 text-orange-500" />,
          title: 'Connection Error',
          message: 'Unable to connect to the Graphiti service. Please check your internet connection.',
          color: 'orange',
          variant: 'destructive' as const,
          actions: ['retry', 'support']
        };
      case 'server':
        return {
          icon: <Server className="h-8 w-8 text-red-500" />,
          title: 'Server Error',
          message: 'The Graphiti service is temporarily unavailable. Our team has been notified.',
          color: 'red',
          variant: 'destructive' as const,
          actions: ['retry', 'support']
        };
      case 'data':
        return {
          icon: <FileX className="h-8 w-8 text-purple-500" />,
          title: 'No Data Available',
          message: 'No entities or relationships found. Try adjusting your search criteria or filters.',
          color: 'purple',
          variant: 'default' as const,
          actions: ['retry']
        };
      case 'timeout':
        return {
          icon: <Clock className="h-8 w-8 text-yellow-500" />,
          title: 'Request Timeout',
          message: 'The request took too long to complete. This might be due to a large dataset.',
          color: 'yellow',
          variant: 'default' as const,
          actions: ['retry', 'support']
        };
      case 'permission':
        return {
          icon: <XCircle className="h-8 w-8 text-red-500" />,
          title: 'Access Denied',
          message: 'You do not have permission to access this graph data.',
          color: 'red',
          variant: 'destructive' as const,
          actions: ['support']
        };
      default:
        return {
          icon: <AlertTriangle className="h-8 w-8 text-gray-500" />,
          title: 'Something went wrong',
          message: 'An unexpected error occurred. Please try again.',
          color: 'gray',
          variant: 'default' as const,
          actions: ['retry', 'support']
        };
    }
  };

  const config = getErrorConfig();
  const errorTitle = title || config.title;
  const errorMessage = message || config.message;

  return (
    <div className={cn("flex items-center justify-center w-full h-full min-h-[200px] p-4", className)}>
      <Card className="w-full max-w-md">
        <CardHeader className="text-center pb-4">
          <div className="flex justify-center mb-4">
            {config.icon}
          </div>
          <CardTitle className="text-lg font-semibold text-gray-900">
            {errorTitle}
          </CardTitle>
        </CardHeader>
        
        <CardContent className="text-center space-y-4">
          <p className="text-gray-600">
            {errorMessage}
          </p>

          {details && showDetails && (
            <Alert variant={config.variant}>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Error Details</AlertTitle>
              <AlertDescription className="mt-2">
                <div className="text-left">
                  <button
                    onClick={() => setShowFullDetails(!showFullDetails)}
                    className="text-sm underline hover:no-underline mb-2"
                  >
                    {showFullDetails ? 'Hide' : 'Show'} technical details
                  </button>
                  
                  {showFullDetails && (
                    <pre className="text-xs bg-gray-100 p-2 rounded mt-2 overflow-auto max-h-32">
                      {details}
                    </pre>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}

          <div className="flex flex-col space-y-2">
            {config.actions.includes('retry') && onRetry && (
              <Button onClick={onRetry} className="w-full">
                <RefreshCw className="h-4 w-4 mr-2" />
                Try Again
              </Button>
            )}
            
            {config.actions.includes('support') && onSupport && (
              <Button variant="outline" onClick={onSupport} className="w-full">
                Contact Support
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Empty States
interface EmptyStateProps {
  type?: 'no-data' | 'no-results' | 'no-connections' | 'first-time';
  title?: string;
  message?: string;
  actionLabel?: string;
  onAction?: () => void;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'no-data',
  title,
  message,
  actionLabel,
  onAction,
  className
}) => {
  const getEmptyConfig = () => {
    switch (type) {
      case 'no-data':
        return {
          icon: <Database className="h-12 w-12 text-gray-300" />,
          title: 'No Graph Data',
          message: 'There are no entities or relationships to display. Start by adding some data to your knowledge graph.',
          actionLabel: 'Add Data',
          illustration: '📊'
        };
      case 'no-results':
        return {
          icon: <Search className="h-12 w-12 text-gray-300" />,
          title: 'No Results Found',
          message: 'No entities match your current search and filter criteria. Try adjusting your filters or search terms.',
          actionLabel: 'Clear Filters',
          illustration: '🔍'
        };
      case 'no-connections':
        return {
          icon: <TrendingDown className="h-12 w-12 text-gray-300" />,
          title: 'No Connections',
          message: 'This entity has no relationships with other entities in the graph.',
          actionLabel: 'Explore Other Entities',
          illustration: '🔗'
        };
      case 'first-time':
        return {
          icon: <CheckCircle className="h-12 w-12 text-blue-300" />,
          title: 'Welcome to Graphiti Explorer',
          message: 'Discover and explore relationships in your knowledge graph. Start by searching for entities or browsing the complete graph.',
          actionLabel: 'Take a Tour',
          illustration: '🌟'
        };
      default:
        return {
          icon: <FileX className="h-12 w-12 text-gray-300" />,
          title: 'Nothing Here',
          message: 'There\'s nothing to show right now.',
          actionLabel: 'Refresh',
          illustration: '📋'
        };
    }
  };

  const config = getEmptyConfig();
  const emptyTitle = title || config.title;
  const emptyMessage = message || config.message;
  const emptyActionLabel = actionLabel || config.actionLabel;

  return (
    <div className={cn("flex items-center justify-center w-full h-full min-h-[300px] p-8", className)}>
      <div className="text-center max-w-md">
        <div className="text-6xl mb-6">{config.illustration}</div>
        
        <h3 className="text-xl font-semibold text-gray-900 mb-3">
          {emptyTitle}
        </h3>
        
        <p className="text-gray-600 mb-6 leading-relaxed">
          {emptyMessage}
        </p>

        {onAction && (
          <Button onClick={onAction} variant={type === 'first-time' ? 'default' : 'outline'}>
            {emptyActionLabel}
          </Button>
        )}
      </div>
    </div>
  );
};

// Success States
interface SuccessStateProps {
  title?: string;
  message?: string;
  details?: string;
  onContinue?: () => void;
  autoHide?: boolean;
  duration?: number;
  className?: string;
}

export const SuccessState: React.FC<SuccessStateProps> = ({
  title = 'Success!',
  message = 'Operation completed successfully.',
  details,
  onContinue,
  autoHide = false,
  duration = 3000,
  className
}) => {
  const [visible, setVisible] = React.useState(true);

  React.useEffect(() => {
    if (autoHide) {
      const timer = setTimeout(() => {
        setVisible(false);
        onContinue?.();
      }, duration);
      
      return () => clearTimeout(timer);
    }
  }, [autoHide, duration, onContinue]);

  if (!visible) return null;

  return (
    <div className={cn("fixed top-4 right-4 z-50", className)}>
      <Alert className="bg-green-50 border-green-200 shadow-lg">
        <CheckCircle className="h-4 w-4 text-green-600" />
        <AlertTitle className="text-green-800">{title}</AlertTitle>
        <AlertDescription className="text-green-700">
          {message}
          {details && (
            <div className="text-xs text-green-600 mt-1">{details}</div>
          )}
        </AlertDescription>
        
        {onContinue && !autoHide && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onContinue}
            className="mt-2 text-green-700 hover:text-green-800 hover:bg-green-100"
          >
            Continue
          </Button>
        )}
      </Alert>
    </div>
  );
};

// Skeleton Loading Components
export const GraphSkeleton: React.FC<{ className?: string }> = ({ className }) => {
  return (
    <div className={cn("w-full h-full p-4", className)}>
      <div className="grid grid-cols-4 gap-4 h-full">
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} className="space-y-3">
            <div className="animate-pulse">
              <div className="w-16 h-16 bg-gray-200 rounded-xl mx-auto"></div>
              <div className="h-3 bg-gray-200 rounded mt-2"></div>
              <div className="h-2 bg-gray-200 rounded mt-1 w-3/4 mx-auto"></div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Fake connections */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        {Array.from({ length: 8 }).map((_, i) => (
          <line
            key={i}
            x1={`${Math.random() * 80 + 10}%`}
            y1={`${Math.random() * 80 + 10}%`}
            x2={`${Math.random() * 80 + 10}%`}
            y2={`${Math.random() * 80 + 10}%`}
            stroke="#e5e7eb"
            strokeWidth="2"
            className="animate-pulse"
          />
        ))}
      </svg>
    </div>
  );
};

// Connection Health Indicator
interface HealthIndicatorProps {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  lastCheck?: Date;
  className?: string;
}

export const HealthIndicator: React.FC<HealthIndicatorProps> = ({
  status,
  lastCheck,
  className
}) => {
  const getHealthConfig = () => {
    switch (status) {
      case 'healthy':
        return {
          icon: <Wifi className="h-4 w-4 text-green-500" />,
          label: 'Connected',
          color: 'green',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200'
        };
      case 'degraded':
        return {
          icon: <TrendingDown className="h-4 w-4 text-yellow-500" />,
          label: 'Slow Connection',
          color: 'yellow',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200'
        };
      case 'unhealthy':
        return {
          icon: <WifiOff className="h-4 w-4 text-red-500" />,
          label: 'Disconnected',
          color: 'red',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200'
        };
      default:
        return {
          icon: <AlertTriangle className="h-4 w-4 text-gray-500" />,
          label: 'Unknown',
          color: 'gray',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200'
        };
    }
  };

  const config = getHealthConfig();

  return (
    <div className={cn(
      "flex items-center space-x-2 px-3 py-2 rounded-lg border",
      config.bgColor,
      config.borderColor,
      className
    )}>
      {config.icon}
      <span className={cn("text-sm font-medium", `text-${config.color}-700`)}>
        {config.label}
      </span>
      {lastCheck && (
        <span className="text-xs text-gray-500">
          {lastCheck.toLocaleTimeString()}
        </span>
      )}
    </div>
  );
};