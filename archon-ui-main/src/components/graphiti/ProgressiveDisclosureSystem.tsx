/**
 * Progressive Disclosure System for Graphiti Explorer
 * Reduces information overload by showing information progressively
 */

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/badge';
import { 
  ChevronDown, 
  ChevronUp, 
  Eye, 
  EyeOff,
  Info,
  MoreHorizontal,
  Settings,
  Layers,
  Filter,
  BarChart3,
  Clock,
  Users,
  Zap
} from 'lucide-react';

// View modes for progressive disclosure
export enum ViewMode {
  MINIMAL = 'minimal',      // Essential info only
  STANDARD = 'standard',    // Balanced view
  DETAILED = 'detailed',    // All information
  EXPERT = 'expert'         // Advanced features
}

export interface ViewModeConfig {
  label: string;
  description: string;
  icon: React.ReactNode;
  features: {
    showConfidenceScores: boolean;
    showRelationshipDetails: boolean;
    showEntityAttributes: boolean;
    showPerformanceMetrics: boolean;
    showAdvancedFilters: boolean;
    showSystemInfo: boolean;
    maxEntitiesDisplayed: number;
    showTimestamps: boolean;
    enableBulkOperations: boolean;
  };
}

// Predefined view mode configurations
export const VIEW_MODE_CONFIGS: Record<ViewMode, ViewModeConfig> = {
  [ViewMode.MINIMAL]: {
    label: 'Focus Mode',
    description: 'Clean, distraction-free view with essential information only',
    icon: <Eye className="w-4 h-4" />,
    features: {
      showConfidenceScores: false,
      showRelationshipDetails: false,
      showEntityAttributes: false,
      showPerformanceMetrics: false,
      showAdvancedFilters: false,
      showSystemInfo: false,
      maxEntitiesDisplayed: 12,
      showTimestamps: false,
      enableBulkOperations: false,
    }
  },
  [ViewMode.STANDARD]: {
    label: 'Standard View',
    description: 'Balanced information display suitable for most users',
    icon: <Layers className="w-4 h-4" />,
    features: {
      showConfidenceScores: true,
      showRelationshipDetails: true,
      showEntityAttributes: false,
      showPerformanceMetrics: false,
      showAdvancedFilters: true,
      showSystemInfo: false,
      maxEntitiesDisplayed: 24,
      showTimestamps: false,
      enableBulkOperations: false,
    }
  },
  [ViewMode.DETAILED]: {
    label: 'Detailed View',
    description: 'Comprehensive information for in-depth exploration',
    icon: <BarChart3 className="w-4 h-4" />,
    features: {
      showConfidenceScores: true,
      showRelationshipDetails: true,
      showEntityAttributes: true,
      showPerformanceMetrics: true,
      showAdvancedFilters: true,
      showSystemInfo: true,
      maxEntitiesDisplayed: 48,
      showTimestamps: true,
      enableBulkOperations: true,
    }
  },
  [ViewMode.EXPERT]: {
    label: 'Expert Mode',
    description: 'All features and advanced controls for power users',
    icon: <Settings className="w-4 h-4" />,
    features: {
      showConfidenceScores: true,
      showRelationshipDetails: true,
      showEntityAttributes: true,
      showPerformanceMetrics: true,
      showAdvancedFilters: true,
      showSystemInfo: true,
      maxEntitiesDisplayed: -1, // No limit
      showTimestamps: true,
      enableBulkOperations: true,
    }
  }
};

// Expandable section component
interface ExpandableSectionProps {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  disabled?: boolean;
  badge?: string | number;
  className?: string;
}

export const ExpandableSection: React.FC<ExpandableSectionProps> = ({
  title,
  icon,
  children,
  defaultExpanded = false,
  disabled = false,
  badge,
  className = ""
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  if (disabled) return null;

  return (
    <div className={`${className}`}>
      <div 
        className="flex items-center justify-between cursor-pointer hover:bg-white/5 transition-colors p-2 rounded-lg mb-3"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-gray-300">{title}</span>
          {badge && (
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30 text-xs">
              {badge}
            </Badge>
          )}
        </div>
        <Button variant="ghost" size="sm" className="h-6 w-6 p-0 hover:bg-white/10 text-gray-400 hover:text-white">
          {isExpanded ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )}
        </Button>
      </div>
      {isExpanded && (
        <div className="pl-2">
          {children}
        </div>
      )}
    </div>
  );
};

// View mode selector component
interface ViewModeSelectorProps {
  currentMode: ViewMode;
  onModeChange: (mode: ViewMode) => void;
  className?: string;
}

export const ViewModeSelector: React.FC<ViewModeSelectorProps> = ({
  currentMode,
  onModeChange,
  className = ""
}) => {
  return (
    <div className={`${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2 text-purple-300">
          <Layers className="w-5 h-5 text-purple-400" />
          View Mode
        </h3>
      </div>
      <div className="space-y-3">
        {Object.entries(VIEW_MODE_CONFIGS).map(([mode, config]) => (
          <Button
            key={mode}
            variant={currentMode === mode ? "default" : "outline"}
            size="sm"
            onClick={() => onModeChange(mode as ViewMode)}
            className={`w-full justify-start text-left transition-all duration-200 ${
              currentMode === mode 
                ? "bg-purple-500/30 border-purple-400 text-purple-200 shadow-[0_0_15px_rgba(168,85,247,0.3)]" 
                : "bg-black/20 border-zinc-700 text-gray-300 hover:bg-purple-500/10 hover:border-purple-500/50"
            }`}
          >
            <div className="flex items-center gap-3">
              <div className={`p-1.5 rounded-md ${currentMode === mode ? 'bg-purple-400/20' : 'bg-zinc-700/50'}`}>
                {config.icon}
              </div>
              <div>
                <div className="font-medium">{config.label}</div>
                <div className="text-xs text-gray-400 line-clamp-2">
                  {config.description}
                </div>
              </div>
            </div>
          </Button>
        ))}
      </div>
    </div>
  );
};

// Progressive entity display component
interface ProgressiveEntityDisplayProps {
  entities: Array<{
    id: string;
    name: string;
    type: string;
    confidence?: number;
    attributes?: Record<string, any>;
    lastUpdated?: Date;
    relatedCount?: number;
  }>;
  viewMode: ViewMode;
  selectedEntity?: string;
  onEntityClick?: (entityId: string) => void;
  className?: string;
}

export const ProgressiveEntityDisplay: React.FC<ProgressiveEntityDisplayProps> = ({
  entities,
  viewMode,
  selectedEntity,
  onEntityClick,
  className = ""
}) => {
  const config = VIEW_MODE_CONFIGS[viewMode];
  const [showAll, setShowAll] = useState(false);
  
  const displayEntities = config.features.maxEntitiesDisplayed > 0 && !showAll
    ? entities.slice(0, config.features.maxEntitiesDisplayed)
    : entities;

  const hasHiddenEntities = config.features.maxEntitiesDisplayed > 0 && 
    entities.length > config.features.maxEntitiesDisplayed && !showAll;

  const getEntityColor = (type: string) => {
    const colors: Record<string, string> = {
      function: 'bg-blue-500/10 border-blue-500/30 text-blue-300',
      class: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300',
      agent: 'bg-red-500/10 border-red-500/30 text-red-300',
      concept: 'bg-purple-500/10 border-purple-500/30 text-purple-300',
      person: 'bg-orange-500/10 border-orange-500/30 text-orange-300',
      organization: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300',
    };
    return colors[type] || 'bg-gray-500/10 border-gray-500/30 text-gray-300';
  };

  return (
    <div className={className}>
      <div className="grid gap-3" style={{
        gridTemplateColumns: `repeat(auto-fill, minmax(${
          viewMode === ViewMode.MINIMAL ? '200px' : 
          viewMode === ViewMode.STANDARD ? '240px' : '280px'
        }, 1fr))`
      }}>
        {displayEntities.map((entity) => (
          <div
            key={entity.id}
            className={`cursor-pointer transition-all duration-300 backdrop-blur-sm rounded-lg border p-4 hover:scale-[1.02] ${
              selectedEntity === entity.id 
                ? 'ring-2 ring-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.3)] bg-purple-500/20 border-purple-400' 
                : `${getEntityColor(entity.type)} hover:shadow-lg hover:border-opacity-60 bg-black/30`
            }`}
            onClick={() => onEntityClick?.(entity.id)}
          >
            <div className="space-y-3">
              {/* Entity name and type */}
              <div>
                <h3 className="font-semibold text-base text-white">{entity.name}</h3>
                <p className="text-xs uppercase tracking-wider font-medium opacity-80">
                  {entity.type}
                </p>
              </div>

              {/* Progressive information based on view mode */}
              {config.features.showConfidenceScores && entity.confidence && (
                <div className="flex items-center gap-3">
                  <div className="text-xs font-bold text-cyan-300 font-mono">
                    {Math.round(entity.confidence)}%
                  </div>
                  <div className="flex-1 bg-black/40 rounded-full h-2 border border-zinc-700/50">
                    <div 
                      className="bg-gradient-to-r from-cyan-500 to-purple-500 h-full rounded-full transition-all duration-500 shadow-[0_0_8px_rgba(34,211,238,0.5)]"
                      style={{ width: `${entity.confidence}%` }}
                    />
                  </div>
                </div>
              )}

                {config.features.showEntityAttributes && entity.attributes && (
                  <ExpandableSection
                    title="Attributes"
                    icon={<Info className="w-3 h-3" />}
                    defaultExpanded={false}
                    badge={Object.keys(entity.attributes).length}
                  >
                    <div className="space-y-1 text-xs">
                      {Object.entries(entity.attributes).slice(0, 3).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span className="opacity-70">{key}:</span>
                          <span>{String(value).substring(0, 20)}</span>
                        </div>
                      ))}
                    </div>
                  </ExpandableSection>
                )}

                {config.features.showTimestamps && entity.lastUpdated && (
                  <div className="flex items-center gap-1 text-xs opacity-70">
                    <Clock className="w-3 h-3" />
                    {entity.lastUpdated.toLocaleDateString()}
                  </div>
                )}

                {viewMode === ViewMode.DETAILED && entity.relatedCount && (
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <Users className="w-3 h-3 text-cyan-400" />
                    <span>{entity.relatedCount} related</span>
                  </div>
                )}
              </div>
            </div>
        ))}
      </div>

      {hasHiddenEntities && (
        <div className="mt-6 text-center">
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="flex items-center gap-2 bg-black/20 border-purple-500/30 text-purple-300 hover:bg-purple-500/20 hover:border-purple-400 transition-all duration-200"
          >
            <MoreHorizontal className="w-4 h-4" />
            Show {entities.length - displayEntities.length} more entities
          </Button>
        </div>
      )}
    </div>
  );
};

// Progressive relationship display
interface ProgressiveRelationshipDisplayProps {
  relationships: Array<{
    id: string;
    from: string;
    to: string;
    type: string;
    confidence?: number;
    attributes?: Record<string, any>;
  }>;
  entities: Array<{ id: string; name: string }>;
  viewMode: ViewMode;
  className?: string;
}

export const ProgressiveRelationshipDisplay: React.FC<ProgressiveRelationshipDisplayProps> = ({
  relationships,
  entities,
  viewMode,
  className = ""
}) => {
  const config = VIEW_MODE_CONFIGS[viewMode];

  if (!config.features.showRelationshipDetails) {
    return (
      <div className={`text-center py-4 text-sm text-gray-500 ${className}`}>
        <EyeOff className="w-4 h-4 mx-auto mb-2 opacity-50" />
        Relationship details hidden in {config.label}
      </div>
    );
  }

  const getEntityName = (entityId: string) => {
    return entities.find(e => e.id === entityId)?.name || entityId;
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {relationships.map((relationship) => (
        <Card key={relationship.id} className="p-3">
          <div className="flex items-center space-x-2 text-sm">
            <span className="font-medium text-blue-600">
              {getEntityName(relationship.from)}
            </span>
            <div className="flex items-center space-x-1">
              <div className="w-3 h-px bg-gray-400" />
              <Badge variant="outline" className="px-2 py-0 text-xs">
                {relationship.type}
              </Badge>
              <div className="w-3 h-px bg-gray-400" />
              <div className="text-gray-400">→</div>
            </div>
            <span className="font-medium text-green-600">
              {getEntityName(relationship.to)}
            </span>
            
            {config.features.showConfidenceScores && relationship.confidence && (
              <Badge variant="secondary" className="ml-auto text-xs">
                {Math.round(relationship.confidence)}%
              </Badge>
            )}
          </div>

          {config.features.showEntityAttributes && relationship.attributes && 
           Object.keys(relationship.attributes).length > 0 && (
            <div className="mt-2 pt-2 border-t border-gray-100">
              <div className="text-xs text-gray-600 space-y-1">
                {Object.entries(relationship.attributes).slice(0, 2).map(([key, value]) => (
                  <div key={key}>
                    <span className="font-medium">{key}:</span>{' '}
                    {String(value).substring(0, 30)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      ))}
    </div>
  );
};

// Main progressive disclosure context
interface ProgressiveDisclosureContextValue {
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  config: ViewModeConfig;
}

const ProgressiveDisclosureContext = React.createContext<ProgressiveDisclosureContextValue | null>(null);

export const useProgressiveDisclosure = () => {
  const context = React.useContext(ProgressiveDisclosureContext);
  if (!context) {
    throw new Error('useProgressiveDisclosure must be used within a ProgressiveDisclosureProvider');
  }
  return context;
};

// Provider component
interface ProgressiveDisclosureProviderProps {
  children: React.ReactNode;
  initialMode?: ViewMode;
}

export const ProgressiveDisclosureProvider: React.FC<ProgressiveDisclosureProviderProps> = ({
  children,
  initialMode = ViewMode.STANDARD
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>(initialMode);
  
  // Persist view mode preference
  React.useEffect(() => {
    const savedMode = localStorage.getItem('graphiti-view-mode') as ViewMode;
    if (savedMode && Object.values(ViewMode).includes(savedMode)) {
      setViewMode(savedMode);
    }
  }, []);

  const handleSetViewMode = useCallback((mode: ViewMode) => {
    setViewMode(mode);
    localStorage.setItem('graphiti-view-mode', mode);
  }, []);

  const config = VIEW_MODE_CONFIGS[viewMode];

  const value: ProgressiveDisclosureContextValue = {
    viewMode,
    setViewMode: handleSetViewMode,
    config
  };

  return (
    <ProgressiveDisclosureContext.Provider value={value}>
      {children}
    </ProgressiveDisclosureContext.Provider>
  );
};