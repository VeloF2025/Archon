import React, { memo, useMemo } from 'react';
import {
  BaseEdge,
  EdgeProps,
  getBezierPath,
  getSmoothStepPath,
  getStraightPath,
  EdgeLabelRenderer,
  useReactFlow,
} from '@xyflow/react';
import { CommunicationEdgeData, CommunicationType, CommunicationStatus } from '../../types/workflowTypes';

interface CommunicationFlowEdgeProps extends EdgeProps<CommunicationEdgeData> {}

export const CommunicationFlowEdge: React.FC<CommunicationFlowEdgeProps> = memo(({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  style = {},
  markerEnd,
  selected,
}) => {
  const { getEdge, getNodes } = useReactFlow();

  if (!data?.communication) {
    return null;
  }

  const { communication, is_animated, is_highlighted, message_flow } = data;

  // Get path based on edge style preference
  const getPath = () => {
    switch (communication.communication_type) {
      case CommunicationType.HIERARCHICAL:
        return getSmoothStepPath({
          sourceX,
          sourceY,
          targetX,
          targetY,
          sourcePosition,
          targetPosition,
          borderRadius: 8,
        });
      case CommunicationType.DIRECT:
        return getStraightPath({
          sourceX,
          sourceY,
          targetX,
          targetY,
        });
      case CommunicationType.COLLABORATIVE:
      case CommunicationType.BROADCAST:
      case CommunicationType.CHAIN:
      default:
        return getBezierPath({
          sourceX,
          sourceY,
          targetX,
          targetY,
          sourcePosition,
          targetPosition,
        });
    }
  };

  const [edgePath, labelX, labelY] = getPath();

  // Edge styling based on communication type and status
  const getEdgeStyle = (): React.CSSProperties => {
    const baseStyle = {
      ...style,
      strokeWidth: Math.max(1, Math.min(communication.message_count / 5, 6)),
      strokeDasharray: communication.status === CommunicationStatus.PENDING ? '5,5' : 'none',
      opacity: communication.status === CommunicationStatus.FAILED ? 0.5 : 1,
    };

    switch (communication.communication_type) {
      case CommunicationType.HIERARCHICAL:
        return {
          ...baseStyle,
          stroke: is_highlighted ? '#8b5cf6' : '#a855f7',
          strokeWidth: (baseStyle.strokeWidth || 2) + 1,
        };
      case CommunicationType.COLLABORATIVE:
        return {
          ...baseStyle,
          stroke: is_highlighted ? '#10b981' : '#059669',
        };
      case CommunicationType.BROADCAST:
        return {
          ...baseStyle,
          stroke: is_highlighted ? '#f59e0b' : '#d97706',
          strokeWidth: (baseStyle.strokeWidth || 2) - 0.5,
        };
      case CommunicationType.CHAIN:
        return {
          ...baseStyle,
          stroke: is_highlighted ? '#3b82f6' : '#2563eb',
        };
      case CommunicationType.DIRECT:
      default:
        return {
          ...baseStyle,
          stroke: is_highlighted ? '#6b7280' : '#4b5563',
        };
    }
  };

  // Status-based styling
  const getStatusStyle = (): React.CSSProperties => {
    switch (communication.status) {
      case CommunicationStatus.ACTIVE:
        return {
          filter: is_animated ? 'drop-shadow(0 0 3px currentColor)' : 'none',
          animation: is_animated ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none',
        };
      case CommunicationStatus.COMPLETED:
        return {
          opacity: 0.8,
        };
      case CommunicationStatus.FAILED:
        return {
          strokeDasharray: '8,4',
          opacity: 0.6,
        };
      case CommunicationStatus.PENDING:
        return {
          strokeDasharray: '4,4',
          opacity: 0.7,
        };
      default:
        return {};
    }
  };

  // Get icon for communication type
  const getCommunicationIcon = (): string => {
    switch (communication.communication_type) {
      case CommunicationType.HIERARCHICAL:
        return '🏗️';
      case CommunicationType.COLLABORATIVE:
        return '🤝';
      case CommunicationType.BROADCAST:
        return '📡';
      case CommunicationType.CHAIN:
        return '⛓️';
      case CommunicationType.DIRECT:
      default:
        return '➡️';
    }
  };

  // Get status indicator
  const getStatusIndicator = (): string => {
    switch (communication.status) {
      case CommunicationStatus.ACTIVE:
        return '🟢';
      case CommunicationStatus.COMPLETED:
        return '✅';
      case CommunicationStatus.FAILED:
        return '❌';
      case CommunicationStatus.PENDING:
        return '⏳';
      case CommunicationStatus.IDLE:
        return '⚪';
      default:
        return '⚫';
    }
  };

  // Calculate edge label
  const edgeStyle = useMemo(() => ({
    ...getEdgeStyle(),
    ...getStatusStyle(),
  }), [communication, is_animated, is_highlighted, selected]);

  // Animated message particles
  const MessageParticles = () => {
    if (!is_animated || communication.status !== CommunicationStatus.ACTIVE) {
      return null;
    }

    const particles = Math.min(communication.message_count, 5);

    return (
      <>
        {[...Array(particles)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-blue-400 rounded-full animate-pulse"
            style={{
              animationDelay: `${i * 0.3}s`,
              animationDuration: `${2000 + i * 200}ms`,
            }}
          ></div>
        ))}
      </>
    );
  };

  // Edge label with communication info
  const EdgeLabel = () => {
    if (communication.message_count === 0 && !is_highlighted && !selected) {
      return null;
    }

    return (
      <EdgeLabelRenderer>
        <div
          className={`
            absolute transform -translate-x-1/2 -translate-y-1/2
            bg-gray-800/90 backdrop-blur-sm border border-gray-600 rounded-lg
            px-2 py-1 text-xs text-white font-medium
            transition-all duration-200
            ${selected || is_highlighted ? 'scale-110 shadow-lg' : 'scale-100'}
          `}
          style={{
            left: labelX,
            top: labelY,
          }}
        >
          <div className="flex items-center space-x-2">
            <span>{getCommunicationIcon()}</span>
            <span>{getStatusIndicator()}</span>
            {communication.message_count > 0 && (
              <span className="text-blue-300">
                {communication.message_count}
              </span>
            )}
            {selected && (
              <span className="text-gray-400 text-xs">
                {communication.message_type}
              </span>
            )}
          </div>

          {/* Tooltip on hover */}
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block">
            <div className="bg-gray-900 border border-gray-600 rounded-lg p-2 text-xs whitespace-nowrap">
              <div><strong>Type:</strong> {communication.communication_type.replace(/_/g, ' ')}</div>
              <div><strong>Status:</strong> {communication.status.replace(/_/g, ' ')}</div>
              <div><strong>Messages:</strong> {communication.message_count}</div>
              {communication.last_message_at && (
                <div><strong>Last:</strong> {new Date(communication.last_message_at).toLocaleTimeString()}</div>
              )}
            </div>
          </div>
        </div>
      </EdgeLabelRenderer>
    );
  };

  // Custom marker for different communication types
  const getMarker = () => {
    switch (communication.communication_type) {
      case CommunicationType.HIERARCHICAL:
        return (
          <marker
            id={`arrow-hierarchical-${id}`}
            markerWidth="12"
            markerHeight="12"
            refX="12"
            refY="6"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path
              d="M0,0 L0,12 L12,6 z"
              fill="#a855f7"
              stroke="#a855f7"
              strokeWidth="1"
            />
          </marker>
        );
      case CommunicationType.COLLABORATIVE:
        return (
          <marker
            id={`arrow-collaborative-${id}`}
            markerWidth="10"
            markerHeight="10"
            refX="10"
            refY="5"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <circle
              cx="5"
              cy="5"
              r="4"
              fill="#059669"
              stroke="#059669"
              strokeWidth="1"
            />
          </marker>
        );
      case CommunicationType.BROADCAST:
        return (
          <marker
            id={`arrow-broadcast-${id}`}
            markerWidth="12"
            markerHeight="12"
            refX="12"
            refY="6"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path
              d="M0,2 L12,6 L0,10 L4,6 z"
              fill="#d97706"
              stroke="#d97706"
              strokeWidth="1"
            />
          </marker>
        );
      default:
        return markerEnd;
    }
  };

  return (
    <g className="group">
      {/* Define custom markers */}
      <defs>
        {getMarker()}

        {/* Glow effect for active communications */}
        {communication.status === CommunicationStatus.ACTIVE && is_animated && (
          <filter id={`glow-${id}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        )}
      </defs>

      {/* Main edge path */}
      <BaseEdge
        path={edgePath}
        markerEnd={`url(#arrow-${communication.communication_type}-${id})`}
        style={edgeStyle}
      />

      {/* Animated flow overlay for active communications */}
      {communication.status === CommunicationStatus.ACTIVE && is_animated && (
        <BaseEdge
          path={edgePath}
          style={{
            ...edgeStyle,
            stroke: 'rgba(59, 130, 246, 0.6)',
            strokeWidth: (edgeStyle.strokeWidth || 2) + 2,
            strokeDasharray: '10,10',
            animation: 'flow 1s linear infinite',
            filter: `url(#glow-${id})`,
          }}
        />
      )}

      {/* Edge label */}
      <EdgeLabel />

      {/* Message particles animation */}
      <MessageParticles />
    </g>
  );
});

CommunicationFlowEdge.displayName = 'CommunicationFlowEdge';

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  @keyframes flow {
    0% {
      strokeDashoffset: 0;
    }
    100% {
      strokeDashoffset: 20;
    }
  }
`;
document.head.appendChild(style);