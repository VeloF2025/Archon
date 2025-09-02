import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  ChevronLeft, 
  ChevronRight, 
  Search, 
  Filter, 
  Settings,
  HelpCircle,
  Maximize2,
  MoreHorizontal
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface GraphitiLayoutProps {
  children: React.ReactNode;
  selectedEntity?: any;
  onSearch?: (term: string) => void;
  onFilter?: (filters: any) => void;
  isLoading?: boolean;
  graphStats?: {
    entities: number;
    relationships: number;
  };
}

type ViewMode = 'minimal' | 'standard' | 'advanced';
type PanelState = 'collapsed' | 'peek' | 'expanded';

export const GraphitiLayout: React.FC<GraphitiLayoutProps> = ({
  children,
  selectedEntity,
  onSearch,
  onFilter,
  isLoading = false,
  graphStats
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('standard');
  const [leftPanelState, setLeftPanelState] = useState<PanelState>('peek');
  const [rightPanelState, setRightPanelState] = useState<PanelState>('collapsed');
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [isFirstVisit, setIsFirstVisit] = useState(false);

  // Check if this is user's first visit
  useEffect(() => {
    const hasVisited = localStorage.getItem('graphiti-explorer-visited');
    if (!hasVisited) {
      setIsFirstVisit(true);
      setShowOnboarding(true);
      localStorage.setItem('graphiti-explorer-visited', 'true');
    }
  }, []);

  // Auto-expand right panel when entity is selected
  useEffect(() => {
    if (selectedEntity && rightPanelState === 'collapsed') {
      setRightPanelState('expanded');
    }
  }, [selectedEntity, rightPanelState]);

  const getLayoutClasses = () => {
    const baseClasses = "w-full h-screen flex flex-col bg-gradient-to-br from-slate-50 to-blue-50/30";
    
    if (viewMode === 'minimal') {
      return cn(baseClasses, "relative");
    }
    
    return baseClasses;
  };

  const getLeftPanelWidth = () => {
    switch (leftPanelState) {
      case 'collapsed': return 'w-0';
      case 'peek': return 'w-16';
      case 'expanded': return 'w-80';
      default: return 'w-16';
    }
  };

  const getRightPanelWidth = () => {
    switch (rightPanelState) {
      case 'collapsed': return 'w-0';
      case 'peek': return 'w-16';
      case 'expanded': return 'w-96';
      default: return 'w-0';
    }
  };

  return (
    <div className={getLayoutClasses()}>
      {/* Top Navigation Bar */}
      <header className="flex items-center justify-between p-4 bg-white/95 backdrop-blur-sm border-b border-gray-200/60 shadow-sm relative z-50">
        <div className="flex items-center space-x-4">
          {/* Logo and Title */}
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white text-sm font-bold">G</span>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Graphiti Explorer</h1>
              <p className="text-xs text-gray-500 hidden sm:block">Knowledge Graph Visualization</p>
            </div>
          </div>

          {/* Quick Stats - Only in standard/advanced mode */}
          {viewMode !== 'minimal' && graphStats && (
            <div className="flex items-center space-x-2 hidden md:flex">
              <Badge variant="secondary" className="bg-blue-100 text-blue-800 text-xs">
                {graphStats.entities} entities
              </Badge>
              <Badge variant="secondary" className="bg-purple-100 text-purple-800 text-xs">
                {graphStats.relationships} connections
              </Badge>
            </div>
          )}
        </div>

        {/* Top Action Bar */}
        <div className="flex items-center space-x-2">
          {/* View Mode Toggle */}
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <Button
              variant={viewMode === 'minimal' ? 'default' : 'ghost'}
              size="sm"
              className="text-xs px-3 h-8"
              onClick={() => setViewMode('minimal')}
              aria-label="Minimal view - Graph only"
            >
              Focus
            </Button>
            <Button
              variant={viewMode === 'standard' ? 'default' : 'ghost'}
              size="sm"
              className="text-xs px-3 h-8"
              onClick={() => setViewMode('standard')}
              aria-label="Standard view - Balanced interface"
            >
              Standard
            </Button>
            <Button
              variant={viewMode === 'advanced' ? 'default' : 'ghost'}
              size="sm"
              className="text-xs px-3 h-8"
              onClick={() => setViewMode('advanced')}
              aria-label="Advanced view - All features"
            >
              Advanced
            </Button>
          </div>

          {/* Help Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowOnboarding(true)}
            className="text-gray-600 hover:text-blue-600"
            aria-label="Show help and onboarding"
          >
            <HelpCircle className="h-4 w-4" />
          </Button>

          {/* Fullscreen Toggle */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              if (document.fullscreenElement) {
                document.exitFullscreen();
              } else {
                document.documentElement.requestFullscreen();
              }
            }}
            className="text-gray-600 hover:text-blue-600 hidden sm:flex"
            aria-label="Toggle fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Controls and Filters */}
        {viewMode !== 'minimal' && (
          <aside className={cn(
            "transition-all duration-300 ease-in-out bg-white/95 backdrop-blur-sm border-r border-gray-200/60 overflow-hidden",
            getLeftPanelWidth()
          )}>
            {/* Sidebar Toggle */}
            <div className="flex items-center justify-between p-3 border-b border-gray-200/60">
              {leftPanelState === 'expanded' && (
                <span className="text-sm font-medium text-gray-700">Controls</span>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (leftPanelState === 'collapsed') setLeftPanelState('peek');
                  else if (leftPanelState === 'peek') setLeftPanelState('expanded');
                  else setLeftPanelState('peek');
                }}
                className="text-gray-500 hover:text-gray-700 p-1 h-6 w-6"
                aria-label="Toggle control panel"
              >
                {leftPanelState === 'expanded' ? (
                  <ChevronLeft className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
            </div>

            {/* Sidebar Content */}
            <div className="p-3 space-y-4">
              {leftPanelState === 'peek' ? (
                // Compact toolbar view
                <div className="flex flex-col space-y-3">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-center p-2"
                    aria-label="Search entities"
                  >
                    <Search className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-center p-2"
                    aria-label="Filter options"
                  >
                    <Filter className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-center p-2"
                    aria-label="Settings"
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                // Expanded controls view
                <Tabs defaultValue="search" className="w-full">
                  <TabsList className="grid w-full grid-cols-3 mb-4">
                    <TabsTrigger value="search" className="text-xs">
                      <Search className="h-3 w-3 mr-1" />
                      Search
                    </TabsTrigger>
                    <TabsTrigger value="filter" className="text-xs">
                      <Filter className="h-3 w-3 mr-1" />
                      Filter
                    </TabsTrigger>
                    <TabsTrigger value="settings" className="text-xs">
                      <Settings className="h-3 w-3 mr-1" />
                      Settings
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="search" className="space-y-3">
                    {/* Search content will be injected here */}
                    <div className="text-xs text-gray-500">Search functionality</div>
                  </TabsContent>

                  <TabsContent value="filter" className="space-y-3">
                    {/* Filter content will be injected here */}
                    <div className="text-xs text-gray-500">Filter options</div>
                  </TabsContent>

                  <TabsContent value="settings" className="space-y-3">
                    {/* Settings content will be injected here */}
                    <div className="text-xs text-gray-500">Display settings</div>
                  </TabsContent>
                </Tabs>
              )}
            </div>
          </aside>
        )}

        {/* Main Graph Visualization */}
        <main className="flex-1 relative">
          {children}

          {/* Minimal View Floating Controls */}
          {viewMode === 'minimal' && (
            <div className="absolute top-4 left-4 z-10 flex items-center space-x-2">
              <Card className="bg-white/95 backdrop-blur-sm shadow-lg border border-gray-200/60">
                <CardContent className="p-2 flex items-center space-x-2">
                  <Button variant="ghost" size="sm" className="p-2" aria-label="Search">
                    <Search className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="sm" className="p-2" aria-label="Filter">
                    <Filter className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="sm" className="p-2" aria-label="More options">
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-white/60 backdrop-blur-sm flex items-center justify-center z-40">
              <div className="flex items-center space-x-3">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                <span className="text-sm font-medium text-gray-700">Loading graph data...</span>
              </div>
            </div>
          )}
        </main>

        {/* Right Sidebar - Entity Details */}
        {viewMode !== 'minimal' && (
          <aside className={cn(
            "transition-all duration-300 ease-in-out bg-white/95 backdrop-blur-sm border-l border-gray-200/60 overflow-hidden",
            getRightPanelWidth()
          )}>
            {/* Sidebar Toggle */}
            <div className="flex items-center justify-between p-3 border-b border-gray-200/60">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (rightPanelState === 'collapsed') setRightPanelState('expanded');
                  else if (rightPanelState === 'expanded') setRightPanelState('collapsed');
                }}
                className="text-gray-500 hover:text-gray-700 p-1 h-6 w-6"
                aria-label="Toggle details panel"
              >
                {rightPanelState === 'expanded' ? (
                  <ChevronRight className="h-4 w-4" />
                ) : (
                  <ChevronLeft className="h-4 w-4" />
                )}
              </Button>
              {rightPanelState === 'expanded' && (
                <span className="text-sm font-medium text-gray-700">Details</span>
              )}
            </div>

            {/* Details Content */}
            {rightPanelState === 'expanded' && (
              <div className="p-4">
                {selectedEntity ? (
                  <div className="space-y-3">
                    <div className="text-sm text-gray-500">Entity details will appear here</div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="text-gray-400 mb-2">👆</div>
                    <p className="text-sm text-gray-500">Select an entity to view details</p>
                  </div>
                )}
              </div>
            )}
          </aside>
        )}
      </div>

      {/* Onboarding Overlay */}
      {showOnboarding && (
        <OnboardingOverlay
          isFirstVisit={isFirstVisit}
          onClose={() => setShowOnboarding(false)}
          onComplete={() => setShowOnboarding(false)}
        />
      )}
    </div>
  );
};

// Onboarding Component
interface OnboardingOverlayProps {
  isFirstVisit: boolean;
  onClose: () => void;
  onComplete: () => void;
}

const OnboardingOverlay: React.FC<OnboardingOverlayProps> = ({
  isFirstVisit,
  onClose,
  onComplete
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      title: "Welcome to Graphiti Explorer",
      content: "Visualize and explore your knowledge graph with an intuitive interface.",
      target: null
    },
    {
      title: "View Modes",
      content: "Switch between Focus (minimal), Standard, and Advanced modes based on your needs.",
      target: ".view-mode-toggle"
    },
    {
      title: "Search & Filter",
      content: "Use the left panel to search entities and apply filters to focus on specific data.",
      target: ".left-panel"
    },
    {
      title: "Entity Details",
      content: "Click any entity to view detailed information in the right panel.",
      target: ".right-panel"
    }
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900">
                {steps[currentStep].title}
              </h3>
              <p className="text-sm text-gray-600 mt-2">
                {steps[currentStep].content}
              </p>
            </div>

            <div className="flex justify-between items-center">
              <div className="flex space-x-1">
                {steps.map((_, index) => (
                  <div
                    key={index}
                    className={cn(
                      "w-2 h-2 rounded-full",
                      index === currentStep ? "bg-blue-600" : "bg-gray-300"
                    )}
                  />
                ))}
              </div>

              <div className="flex space-x-2">
                {currentStep > 0 && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentStep(currentStep - 1)}
                  >
                    Previous
                  </Button>
                )}
                <Button
                  size="sm"
                  onClick={() => {
                    if (currentStep < steps.length - 1) {
                      setCurrentStep(currentStep + 1);
                    } else {
                      onComplete();
                    }
                  }}
                >
                  {currentStep < steps.length - 1 ? 'Next' : 'Get Started'}
                </Button>
              </div>
            </div>

            <div className="text-center">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="text-xs text-gray-500"
              >
                Skip tutorial
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};