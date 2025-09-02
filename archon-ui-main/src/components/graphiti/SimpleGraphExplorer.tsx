import React, { useState, useEffect } from 'react';
import { OnboardingSystem, HelpButton, QuickTip } from './OnboardingSystem';
import { 
  ProgressiveDisclosureProvider,
  useProgressiveDisclosure,
  ViewModeSelector,
  ProgressiveEntityDisplay,
  ProgressiveRelationshipDisplay,
  ExpandableSection,
  ViewMode
} from './ProgressiveDisclosureSystem';
import {
  GraphitiErrorBoundary,
  ErrorProvider,
  useErrorHandler,
  useNetworkStatus,
  LoadingState,
  EmptyState,
  useTimeout,
  ErrorType,
  ErrorInfo
} from './ErrorHandlingSystem';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/badge';
import { Search, RefreshCw, Settings, Users, GitBranch, AlertCircle, Wifi, WifiOff } from 'lucide-react';

// Internal component that uses progressive disclosure context
const GraphExplorerInner: React.FC = () => {
  const { viewMode, config, setViewMode } = useProgressiveDisclosure();
  const { addError } = useErrorHandler();
  const { isOnline } = useNetworkStatus();
  
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [hasSeenOnboarding, setHasSeenOnboarding] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [hasLoadingError, setHasLoadingError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  // Auto-show onboarding for first-time users
  useEffect(() => {
    const hasSeenTour = localStorage.getItem('graphiti-onboarding-complete');
    if (!hasSeenTour) {
      setTimeout(() => setShowOnboarding(true), 1000);
    } else {
      setHasSeenOnboarding(true);
    }
  }, []);

  // Simulate data loading with error handling
  useEffect(() => {
    const loadGraphData = async () => {
      setIsLoading(true);
      setHasLoadingError(false);

      try {
        // Simulate network request delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Simulate occasional network failures for testing
        if (Math.random() < 0.1 && retryCount === 0) {
          throw new Error('Network connection failed');
        }

        setIsLoading(false);
      } catch (error) {
        setHasLoadingError(true);
        setIsLoading(false);
        
        const errorInfo: ErrorInfo = {
          type: 'network',
          message: 'Failed to load graph data',
          details: error instanceof Error ? error.message : 'Unknown network error',
          retryable: true,
          timestamp: new Date(),
          component: 'GraphExplorerInner'
        };
        
        addError(errorInfo);
      }
    };

    // Only load if online
    if (isOnline) {
      loadGraphData();
    } else {
      setIsLoading(false);
      setHasLoadingError(true);
      
      const offlineError: ErrorInfo = {
        type: 'network',
        message: 'You are currently offline',
        details: 'Graph data cannot be loaded without an internet connection',
        retryable: true,
        timestamp: new Date(),
        component: 'GraphExplorerInner'
      };
      
      addError(offlineError);
    }
  }, [isOnline, retryCount, addError]);

  // Handle data loading timeout
  const handleLoadingTimeout = () => {
    if (isLoading) {
      const timeoutError: ErrorInfo = {
        type: 'timeout',
        message: 'Loading timeout',
        details: 'Graph data took too long to load. The server may be experiencing issues.',
        retryable: true,
        timestamp: new Date(),
        component: 'GraphExplorerInner'
      };
      
      addError(timeoutError);
      setIsLoading(false);
      setHasLoadingError(true);
    }
  };

  // Set up loading timeout
  useTimeout(handleLoadingTimeout, isLoading ? 10000 : null);

  // Retry handler
  const handleRetry = () => {
    setRetryCount(prev => prev + 1);
    setHasLoadingError(false);
  };
  
  // Real entities fetched from API - no mock data
  const [entities, setEntities] = useState([]);

  // Real relationships fetched from API - no mock data
  const [relationships, setRelationships] = useState([]);

  // Fetch real data from API
  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch graph data from REAL API endpoint that builds from knowledge base
        const graphResponse = await fetch('/api/graphiti/graph-data');
        if (graphResponse.ok) {
          const graphData = await graphResponse.json();
          
          // Transform GraphNode format to component format
          const transformedEntities = graphData.nodes.map((node: any) => ({
            id: node.id,
            type: node.type,
            name: node.label,
            confidence: Math.round((node.properties.confidence_score || 0.9) * 100),
            attributes: node.properties,
            lastUpdated: new Date(node.properties.created_at || Date.now()),
            relatedCount: graphData.edges.filter((e: any) => 
              e.source === node.id || e.target === node.id
            ).length
          }));
          
          // Transform GraphEdge format to component format
          const transformedRelationships = graphData.edges.map((edge: any) => ({
            id: edge.id,
            from: edge.source,
            to: edge.target,
            type: edge.type,
            confidence: Math.round((edge.properties.confidence || 0.7) * 100),
            attributes: edge.properties
          }));
          
          setEntities(transformedEntities);
          setRelationships(transformedRelationships);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to fetch graph data:', error);
        setHasLoadingError(true);
        setIsLoading(false);
      }
    };

    fetchGraphData();
  }, [retryCount]);

  // Onboarding handlers
  const handleOnboardingComplete = () => {
    setShowOnboarding(false);
    setHasSeenOnboarding(true);
    localStorage.setItem('graphiti-onboarding-complete', 'true');
  };

  const handleOnboardingSkip = () => {
    setShowOnboarding(false);
    setHasSeenOnboarding(true);
    localStorage.setItem('graphiti-onboarding-complete', 'true');
  };

  const handleStartOnboarding = () => {
    setShowOnboarding(true);
  };

  // Filter entities based on search
  const filteredEntities = entities.filter(entity => 
    entity.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    entity.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getEntityColor = (type: string) => {
    const colors: Record<string, string> = {
      function: 'bg-blue-100 text-blue-800 border-blue-300',
      class: 'bg-green-100 text-green-800 border-green-300',
      agent: 'bg-red-100 text-red-800 border-red-300',
      concept: 'bg-purple-100 text-purple-800 border-purple-300',
    };
    return colors[type] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  return (
    <div className="min-h-screen bg-background text-foreground neon-grid">
      {/* Header */}
      <div className="relative glass-purple rounded-none border-l-0 border-r-0 border-t-0">
        <div className="p-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-purple-500/20 border border-purple-500/30">
                <svg className="h-6 w-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16l2.879-2.879m0 0a3 3 0 104.243-4.242 3 3 0 00-4.243 4.242zM21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Graphiti Explorer
              </span>
            </h1>
            
            <div className="flex items-center space-x-6">
              {/* Network Status Indicator */}
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-black/20 border border-zinc-700/50">
                  {isOnline ? (
                    <Wifi className="w-4 h-4 text-emerald-400" />
                  ) : (
                    <WifiOff className="w-4 h-4 text-red-400" />
                  )}
                  <div className="text-sm font-medium">
                    {!hasLoadingError && !isLoading && (
                      <span className="text-gray-300">
                        {filteredEntities.length} entities • {relationships.length} relationships
                      </span>
                    )}
                    {hasLoadingError && (
                      <span className="text-red-400 flex items-center gap-1">
                        <AlertCircle className="w-3 h-3" />
                        Connection Error
                      </span>
                    )}
                    {isLoading && (
                      <span className="text-blue-400 flex items-center gap-1">
                        <RefreshCw className="w-3 h-3 animate-spin" />
                        Loading...
                      </span>
                    )}
                  </div>
                </div>
                <HelpButton onStartOnboarding={handleStartOnboarding} />
              </div>
            </div>
          </div>
          
          {/* Search Bar */}
          <div className="mt-6 max-w-md">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-purple-400" />
              <Input
                placeholder="Search entities..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 bg-black/30 border-purple-500/30 text-white placeholder-gray-400 focus:border-purple-400 focus:ring-purple-400/30"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex gap-6 p-6">
        {/* Left Sidebar - View Mode Selector */}
        <div className="w-80 space-y-4">
          <div className="relative glass-purple rounded-lg p-4">
            <ViewModeSelector 
              currentMode={viewMode} 
              onModeChange={setViewMode}
              className="" 
            />
          </div>
          
          {config.features.showAdvancedFilters && (
            <div className="relative glass-blue rounded-lg p-4">
              <ExpandableSection
                title="Advanced Filters"
                icon={<Settings className="w-4 h-4 text-blue-400" />}
                defaultExpanded={false}
              >
                <div className="space-y-2">
                  <Button variant="outline" size="sm" className="w-full justify-start bg-black/20 border-zinc-700 text-gray-300 hover:bg-blue-500/20 hover:border-blue-500/50">
                    <GitBranch className="w-4 h-4 mr-2 text-blue-400" />
                    Filter by Type
                  </Button>
                  <Button variant="outline" size="sm" className="w-full justify-start bg-black/20 border-zinc-700 text-gray-300 hover:bg-blue-500/20 hover:border-blue-500/50">
                    <Users className="w-4 h-4 mr-2 text-blue-400" />
                    Filter by Confidence
                  </Button>
                </div>
              </ExpandableSection>
            </div>
          )}

          {config.features.showSystemInfo && (
            <div className="relative glass-green rounded-lg p-4">
              <ExpandableSection
                title="System Information"
                icon={<Settings className="w-4 h-4 text-emerald-400" />}
                defaultExpanded={false}
              >
                <div className="text-xs space-y-2 text-gray-300 font-mono">
                  <div className="flex justify-between">
                    <span className="text-emerald-400">View Mode:</span> 
                    <span className="text-purple-300">{config.label}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-emerald-400">Entities:</span> 
                    <span className="text-blue-300">{Math.min(filteredEntities.length, config.features.maxEntitiesDisplayed > 0 ? config.features.maxEntitiesDisplayed : filteredEntities.length)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-emerald-400">Relationships:</span> 
                    <span className="text-pink-300">{relationships.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-emerald-400">Last Updated:</span> 
                    <span className="text-gray-300">{new Date().toLocaleTimeString()}</span>
                  </div>
                </div>
              </ExpandableSection>
            </div>
          )}
        </div>

        {/* Main Graph Area */}
        <div className="flex-1">
          <div className="relative glass rounded-lg h-full">
            <div className="p-6 border-b border-zinc-800/50">
              <h2 className="text-xl font-semibold flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-cyan-500/20 border border-cyan-500/30">
                  <GitBranch className="w-4 h-4 text-cyan-400" />
                </div>
                <span className="bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                  Knowledge Graph
                </span>
                {config.features.showPerformanceMetrics && (
                  <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                    {config.label}
                  </Badge>
                )}
              </h2>
            </div>
            
            <div className="p-6">
              {/* Loading State */}
              {isLoading && (
                <LoadingState
                  message="Loading graph data..."
                  onTimeout={() => handleLoadingTimeout()}
                />
              )}

              {/* Error State */}
              {hasLoadingError && !isLoading && (
                <div className="space-y-4">
                  <EmptyState
                    type="no_data"
                    onAction={handleRetry}
                    actionLabel={retryCount > 0 ? `Retry (${retryCount})` : "Retry"}
                  />
                  {retryCount > 2 && (
                    <div className="text-center">
                      <Button 
                        variant="outline" 
                        onClick={() => window.location.reload()}
                        className="ml-2"
                      >
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Reload Page
                      </Button>
                    </div>
                  )}
                </div>
              )}

              {/* Normal Content State */}
              {!isLoading && !hasLoadingError && (
                <>
                  {/* No Search Results */}
                  {searchTerm && filteredEntities.length === 0 && (
                    <EmptyState
                      type="no_search_results"
                      onAction={() => setSearchTerm('')}
                      actionLabel="Clear Search"
                    />
                  )}

                  {/* No Entities */}
                  {!searchTerm && filteredEntities.length === 0 && (
                    <EmptyState
                      type="no_entities"
                      onAction={() => setViewMode(ViewMode.EXPERT)}
                      actionLabel="Show All Data"
                    />
                  )}

                  {/* Progressive Entity Display */}
                  {filteredEntities.length > 0 && (
                    <ProgressiveEntityDisplay
                      entities={filteredEntities}
                      viewMode={viewMode}
                      selectedEntity={selectedEntity}
                      onEntityClick={setSelectedEntity}
                      className="mb-8"
                    />
                  )}

                  {/* Progressive Relationship Display */}
                  {filteredEntities.length > 0 && (
                    <ExpandableSection
                      title="Relationships"
                      icon={<GitBranch className="w-4 h-4" />}
                      defaultExpanded={config.features.showRelationshipDetails}
                      disabled={!config.features.showRelationshipDetails}
                      badge={relationships.length}
                    >
                      {relationships.length === 0 ? (
                        <EmptyState
                          type="no_relationships"
                          onAction={() => setViewMode(ViewMode.DETAILED)}
                          actionLabel="Show Detailed View"
                        />
                      ) : (
                        <ProgressiveRelationshipDisplay
                          relationships={relationships}
                          entities={filteredEntities}
                          viewMode={viewMode}
                        />
                      )}
                    </ExpandableSection>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80">
          <div className="relative glass-pink rounded-lg p-6 h-full">
            {selectedEntity ? (
              <div>
                <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                  <div className="p-1.5 rounded-md bg-pink-500/20 border border-pink-500/30">
                    <Users className="w-4 h-4 text-pink-400" />
                  </div>
                  <span className="bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
                    Entity Details
                  </span>
                </h2>
                {(() => {
                  const entity = entities.find(e => e.id === selectedEntity);
                  if (!entity) return null;
                  return (
                    <div className="space-y-6">
                      <div>
                        <label className="text-sm font-medium text-pink-400 mb-2 block">Name</label>
                        <div className="font-semibold text-white text-lg">{entity.name}</div>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-pink-400 mb-2 block">Type</label>
                        <div className={`inline-block px-3 py-1.5 rounded-full text-sm font-medium bg-purple-500/20 text-purple-300 border border-purple-500/30`}>
                          {entity.type}
                        </div>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-pink-400 mb-2 block">Confidence Score</label>
                        <div className="flex items-center space-x-3">
                          <div className="flex-1 bg-black/30 rounded-full h-3 border border-zinc-700/50">
                            <div 
                              className="bg-gradient-to-r from-pink-500 to-purple-500 h-full rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(236,72,153,0.5)]" 
                              style={{ width: `${entity.confidence}%` }}
                            />
                          </div>
                          <span className="text-sm font-mono text-cyan-300 font-bold">{entity.confidence}%</span>
                        </div>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-pink-400 mb-3 block">Related Entities</label>
                        <div className="space-y-2">
                          {relationships
                            .filter(rel => rel.from === selectedEntity || rel.to === selectedEntity)
                            .map((rel, index) => {
                              const otherEntityId = rel.from === selectedEntity ? rel.to : rel.from;
                              const otherEntity = entities.find(e => e.id === otherEntityId);
                              return (
                                <div key={index} className="flex items-center space-x-2 p-2 rounded-lg bg-black/20 border border-zinc-800/50">
                                  <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]"></div>
                                  <span className="text-xs text-gray-400 font-mono">{rel.type}</span>
                                  <span className="text-cyan-400">→</span>
                                  <span className="text-gray-300 font-medium">{otherEntity?.name}</span>
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <div className="text-center mt-20">
                <div className="p-8 rounded-full bg-black/20 border border-zinc-800/50 w-24 h-24 flex items-center justify-center mx-auto mb-4">
                  <Search className="w-8 h-8 text-pink-400" />
                </div>
                <p className="text-gray-400 text-lg">Click on an entity to view details</p>
                <p className="text-gray-500 text-sm mt-2">Explore connections and relationships</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Onboarding System */}
      <OnboardingSystem
        isActive={showOnboarding}
        onComplete={handleOnboardingComplete}
        onSkip={handleOnboardingSkip}
      />
      
      {/* Quick Tips (always visible tooltips) */}
      {hasSeenOnboarding && (
        <>
          <QuickTip
            message="Click to explore entity details"
            targetSelector=".react-flow__node"
            position="top"
          />
          <QuickTip
            message="Type to search entities in real-time"
            targetSelector="input[placeholder*='Search']"
            position="bottom"
          />
          <QuickTip
            message="Start guided tour anytime"
            targetSelector="button:has(.w-4.h-4)"
            position="left"
          />
        </>
      )}
    </div>
  );
};

// Main exported component with error handling and progressive disclosure providers
export const SimpleGraphExplorer: React.FC = () => {
  const handleGlobalError = (error: ErrorInfo) => {
    console.error('Graphiti Explorer Error:', error);
    
    // Could send to error reporting service in production
    if (process.env.NODE_ENV === 'production') {
      // Example: sendToErrorReporting(error);
    }
  };

  return (
    <ErrorProvider>
      <GraphitiErrorBoundary onError={handleGlobalError}>
        <ProgressiveDisclosureProvider initialMode={ViewMode.STANDARD}>
          <GraphExplorerInner />
        </ProgressiveDisclosureProvider>
      </GraphitiErrorBoundary>
    </ErrorProvider>
  );
};