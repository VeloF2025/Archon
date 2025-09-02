import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface EntityNodeData {
  entity_id: string;
  entity_type: string;
  name: string;
  confidence_score: number;
  importance_weight: number;
  access_frequency: number;
  tags?: string[];
  attributes?: Record<string, any>;
}

interface EntityNodeProps {
  data: EntityNodeData;
  selected?: boolean;
  isConnectable?: boolean;
}

// Enhanced entity type configuration with better visual hierarchy
const ENTITY_TYPES = {
  function: {
    icon: '⚡',
    color: {
      primary: '#3b82f6',
      light: '#dbeafe',
      dark: '#1e40af'
    },
    label: 'Function',
    priority: 1
  },
  class: {
    icon: '🏗️',
    color: {
      primary: '#10b981',
      light: '#d1fae5',
      dark: '#065f46'
    },
    label: 'Class',
    priority: 2
  },
  module: {
    icon: '📦',
    color: {
      primary: '#f59e0b',
      light: '#fef3c7',
      dark: '#92400e'
    },
    label: 'Module',
    priority: 3
  },
  concept: {
    icon: '💡',
    color: {
      primary: '#8b5cf6',
      light: '#ede9fe',
      dark: '#6b21a8'
    },
    label: 'Concept',
    priority: 1
  },
  agent: {
    icon: '🤖',
    color: {
      primary: '#ef4444',
      light: '#fee2e2',
      dark: '#991b1b'
    },
    label: 'Agent',
    priority: 1
  },
  project: {
    icon: '📋',
    color: {
      primary: '#06b6d4',
      light: '#cffafe',
      dark: '#0e7490'
    },
    label: 'Project',
    priority: 2
  },
  requirement: {
    icon: '📝',
    color: {
      primary: '#84cc16',
      light: '#ecfccb',
      dark: '#365314'
    },
    label: 'Requirement',
    priority: 3
  },
  pattern: {
    icon: '🎯',
    color: {
      primary: '#ec4899',
      light: '#fce7f3',
      dark: '#be185d'
    },
    label: 'Pattern',
    priority: 2
  }
} as const;

export const EnhancedEntityNode: React.FC<EntityNodeProps> = memo(({ 
  data, 
  selected = false,
  isConnectable = true 
}) => {
  const entityConfig = ENTITY_TYPES[data.entity_type as keyof typeof ENTITY_TYPES] || {
    icon: '📄',
    color: { primary: '#6b7280', light: '#f3f4f6', dark: '#374151' },
    label: 'Unknown',
    priority: 3
  };

  const confidenceScore = Math.round(data.confidence_score * 100);
  const importanceScore = Math.round(data.importance_weight * 100);
  
  // Determine node size based on importance and type priority
  const getNodeSize = () => {
    const baseSize = 120;
    const importanceMultiplier = 1 + (data.importance_weight - 0.5) * 0.4; // 0.6 to 1.4 range
    const priorityMultiplier = entityConfig.priority === 1 ? 1.2 : entityConfig.priority === 2 ? 1.0 : 0.9;
    
    return {
      width: Math.max(baseSize * importanceMultiplier * priorityMultiplier, 100),
      height: Math.max(80 * importanceMultiplier * priorityMultiplier, 70)
    };
  };

  const nodeSize = getNodeSize();
  
  // Dynamic styling based on state and properties
  const getNodeStyles = () => {
    const baseStyle = {
      width: `${nodeSize.width}px`,
      height: `${nodeSize.height}px`,
      background: `linear-gradient(135deg, ${entityConfig.color.light} 0%, white 50%, ${entityConfig.color.light} 100%)`,
      border: `2px solid ${entityConfig.color.primary}`,
      borderRadius: '16px',
      boxShadow: selected 
        ? `0 8px 32px rgba(0, 0, 0, 0.12), 0 0 0 3px ${entityConfig.color.primary}40`
        : '0 4px 16px rgba(0, 0, 0, 0.08)',
      transform: selected ? 'scale(1.05)' : 'scale(1)',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      cursor: 'pointer',
      position: 'relative' as const,
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      padding: '12px',
      overflow: 'hidden'
    };

    if (selected) {
      baseStyle.borderWidth = '3px';
      baseStyle.zIndex = 10;
    }

    return baseStyle;
  };

  // Confidence indicator style
  const getConfidenceIndicator = () => {
    const color = confidenceScore >= 80 ? '#10b981' : 
                 confidenceScore >= 60 ? '#f59e0b' : '#ef4444';
    
    return {
      position: 'absolute' as const,
      top: '8px',
      right: '8px',
      width: '8px',
      height: '8px',
      borderRadius: '50%',
      background: color,
      boxShadow: `0 0 0 2px white, 0 0 8px ${color}40`
    };
  };

  // Truncate long names for better display
  const truncateName = (name: string, maxLength: number = 20) => {
    if (name.length <= maxLength) return name;
    return name.substring(0, maxLength - 3) + '...';
  };

  return (
    <div style={getNodeStyles()} className="group hover:shadow-xl">
      {/* Connection Handles */}
      {isConnectable && (
        <>
          <Handle
            type="target"
            position={Position.Left}
            className="!w-3 !h-3 !bg-gray-400 !border-2 !border-white hover:!bg-blue-500 transition-colors"
            isConnectable={isConnectable}
          />
          <Handle
            type="source"
            position={Position.Right}
            className="!w-3 !h-3 !bg-gray-400 !border-2 !border-white hover:!bg-blue-500 transition-colors"
            isConnectable={isConnectable}
          />
          <Handle
            type="target"
            position={Position.Top}
            className="!w-3 !h-3 !bg-gray-400 !border-2 !border-white hover:!bg-blue-500 transition-colors"
            isConnectable={isConnectable}
          />
          <Handle
            type="source"
            position={Position.Bottom}
            className="!w-3 !h-3 !bg-gray-400 !border-2 !border-white hover:!bg-blue-500 transition-colors"
            isConnectable={isConnectable}
          />
        </>
      )}

      {/* Confidence Indicator */}
      <div style={getConfidenceIndicator()} title={`Confidence: ${confidenceScore}%`} />

      {/* Priority Indicator for high-priority entities */}
      {entityConfig.priority === 1 && importanceScore > 80 && (
        <div
          className="absolute top-2 left-2 w-2 h-2 bg-yellow-400 rounded-full animate-pulse"
          title="High Priority Entity"
        />
      )}

      {/* Entity Icon */}
      <div 
        className="text-3xl mb-2 p-2 rounded-xl transition-transform group-hover:scale-110"
        style={{
          background: `linear-gradient(135deg, ${entityConfig.color.primary}, ${entityConfig.color.dark})`,
          color: 'white',
          boxShadow: `0 4px 12px ${entityConfig.color.primary}40`
        }}
      >
        {entityConfig.icon}
      </div>

      {/* Entity Name */}
      <div 
        className="font-semibold text-center mb-1 leading-tight"
        style={{ 
          color: entityConfig.color.dark,
          fontSize: nodeSize.width > 140 ? '14px' : '12px'
        }}
        title={data.name}
      >
        {truncateName(data.name, nodeSize.width > 140 ? 25 : 18)}
      </div>

      {/* Entity Type Badge */}
      <Badge 
        variant="secondary" 
        className="text-xs px-2 py-1 mb-2"
        style={{ 
          backgroundColor: `${entityConfig.color.primary}20`,
          color: entityConfig.color.dark,
          border: `1px solid ${entityConfig.color.primary}40`
        }}
      >
        {entityConfig.label}
      </Badge>

      {/* Metrics Bar - Only show on larger nodes */}
      {nodeSize.width > 130 && (
        <div className="flex justify-center space-x-2 text-xs">
          {/* Confidence Score */}
          <div className="flex items-center space-x-1">
            <div 
              className="w-2 h-2 rounded-full"
              style={{ 
                backgroundColor: confidenceScore >= 80 ? '#10b981' : 
                               confidenceScore >= 60 ? '#f59e0b' : '#ef4444'
              }}
            />
            <span className="text-gray-600">{confidenceScore}%</span>
          </div>
          
          {/* Access Frequency Indicator */}
          {data.access_frequency > 5 && (
            <div className="flex items-center space-x-1">
              <span className="text-blue-600">🔥</span>
              <span className="text-gray-600">{data.access_frequency}</span>
            </div>
          )}
        </div>
      )}

      {/* Hover Tooltip Content - appears on hover */}
      <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs px-3 py-2 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-20 whitespace-nowrap">
        <div className="font-medium">{data.name}</div>
        <div className="text-gray-300">
          {entityConfig.label} • {confidenceScore}% confidence
        </div>
        {data.tags && data.tags.length > 0 && (
          <div className="text-gray-400 text-xs mt-1">
            {data.tags.slice(0, 2).join(', ')}
            {data.tags.length > 2 && '...'}
          </div>
        )}
      </div>

      {/* Selection Ring */}
      {selected && (
        <div 
          className="absolute inset-0 rounded-2xl pointer-events-none animate-pulse"
          style={{
            background: `linear-gradient(45deg, transparent 30%, ${entityConfig.color.primary}20 50%, transparent 70%)`,
          }}
        />
      )}
    </div>
  );
});

EnhancedEntityNode.displayName = 'EnhancedEntityNode';