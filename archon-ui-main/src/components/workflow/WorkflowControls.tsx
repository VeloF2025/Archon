import React, { useState } from 'react';
import { WorkflowControls as Controls, WorkflowVisualizationConfig, WorkflowStats, LayoutAlgorithm } from '../../types/workflowTypes';
import { Button } from '../ui/button';
import { Card } from '../ui/card';

interface WorkflowControlsProps {
  controls: Controls;
  config: WorkflowVisualizationConfig;
  stats?: WorkflowStats;
  isAnimating: boolean;
  currentLayout: LayoutAlgorithm;
  onConfigChange: (config: WorkflowVisualizationConfig) => void;
}

export const WorkflowControls: React.FC<WorkflowControlsProps> = ({
  controls,
  config,
  stats,
  isAnimating,
  currentLayout,
  onConfigChange,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Layout options
  const layoutOptions = [
    { value: 'force', label: 'Force', icon: '🌐' },
    { value: 'circular', label: 'Circular', icon: '⭕' },
    { value: 'hierarchical', label: 'Hierarchical', icon: '🏗️' },
    { value: 'grid', label: 'Grid', icon: '⬜' },
  ] as const;

  // Handle layout change
  const handleLayoutChange = (layout: LayoutAlgorithm) => {
    controls.apply_layout(layout);
  };

  // Toggle animation
  const handleToggleAnimation = () => {
    controls.toggle_animation();
  };

  // Handle zoom controls
  const ZoomControls = () => (
    <div className="flex items-center space-x-1">
      <Button
        size="sm"
        variant="outline"
        accentColor="blue"
        onClick={controls.zoom_in}
        title="Zoom In"
      >
        <span className="text-lg">➕</span>
      </Button>
      <Button
        size="sm"
        variant="outline"
        accentColor="blue"
        onClick={controls.zoom_out}
        title="Zoom Out"
      >
        <span className="text-lg">➖</span>
      </Button>
      <Button
        size="sm"
        variant="outline"
        accentColor="blue"
        onClick={controls.fit_to_screen}
        title="Fit to Screen"
      >
        <span className="text-lg">🔍</span>
      </Button>
      <Button
        size="sm"
        variant="outline"
        accentColor="blue"
        onClick={controls.center_view}
        title="Center View"
      >
        <span className="text-lg">⊕</span>
      </Button>
    </div>
  );

  // Layout controls
  const LayoutControls = () => (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-300 mb-2">Layout</h4>
      <div className="grid grid-cols-2 gap-1">
        {layoutOptions.map(({ value, label, icon }) => (
          <Button
            key={value}
            size="sm"
            variant={currentLayout === value ? 'primary' : 'outline'}
            accentColor="purple"
            onClick={() => handleLayoutChange(value)}
            className="text-xs"
            title={`${label} layout`}
          >
            <span className="mr-1">{icon}</span>
            {label}
          </Button>
        ))}
      </div>
    </div>
  );

  // Animation controls
  const AnimationControls = () => (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-300 mb-2">Animation</h4>
      <div className="flex items-center space-x-2">
        <Button
          size="sm"
          variant={isAnimating ? 'primary' : 'outline'}
          accentColor="green"
          onClick={handleToggleAnimation}
          className="text-xs"
        >
          <span className="mr-1">{isAnimating ? '⏸️' : '▶️'}</span>
          {isAnimating ? 'Pause' : 'Play'}
        </Button>
        <select
          value={config.animation_speed}
          onChange={(e) => onConfigChange({
            ...config,
            animation_speed: parseInt(e.target.value)
          })}
          className="text-xs bg-gray-700 border border-gray-600 rounded px-2 py-1 text-white"
        >
          <option value={500}>Fast</option>
          <option value={1000}>Normal</option>
          <option value={2000}>Slow</option>
        </select>
      </div>
    </div>
  );

  // View controls
  const ViewControls = () => (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-300 mb-2">View Options</h4>
      <div className="space-y-1">
        <label className="flex items-center space-x-2 text-xs">
          <input
            type="checkbox"
            checked={config.show_labels}
            onChange={(e) => onConfigChange({
              ...config,
              show_labels: e.target.checked
            })}
            className="rounded"
          />
          <span>Show Labels</span>
        </label>
        <label className="flex items-center space-x-2 text-xs">
          <input
            type="checkbox"
            checked={config.show_metrics}
            onChange={(e) => onConfigChange({
              ...config,
              show_metrics: e.target.checked
            })}
            className="rounded"
          />
          <span>Show Metrics</span>
        </label>
        <div className="flex items-center space-x-2">
          <span className="text-xs">Node Size:</span>
          <select
            value={config.node_size}
            onChange={(e) => onConfigChange({
              ...config,
              node_size: e.target.value as any
            })}
            className="text-xs bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-white"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </select>
        </div>
      </div>
    </div>
  );

  // Statistics display
  const StatisticsPanel = () => {
    if (!stats) return null;

    return (
      <Card className="p-3 bg-gray-800/50 border-gray-600">
        <h4 className="text-xs font-medium text-gray-300 mb-2">Statistics</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <div className="text-gray-400">Total Agents</div>
            <div className="text-white font-medium">{stats.total_agents}</div>
          </div>
          <div>
            <div className="text-gray-400">Active</div>
            <div className="text-green-400 font-medium">{stats.active_agents}</div>
          </div>
          <div>
            <div className="text-gray-400">Communications</div>
            <div className="text-white font-medium">{stats.total_communications}</div>
          </div>
          <div>
            <div className="text-gray-400">Active Flows</div>
            <div className="text-blue-400 font-medium">{stats.active_communications}</div>
          </div>
          <div className="col-span-2">
            <div className="text-gray-400">Avg Messages/Connection</div>
            <div className="text-white font-medium">
              {stats.avg_messages_per_connection.toFixed(1)}
            </div>
          </div>
        </div>
      </Card>
    );
  };

  // Action controls
  const ActionControls = () => (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-gray-300 mb-2">Actions</h4>
      <div className="space-y-1">
        <Button
          size="sm"
          variant="outline"
          accentColor="green"
          onClick={controls.refresh_data}
          className="w-full text-xs justify-start"
        >
          <span className="mr-2">🔄</span>
          Refresh Data
        </Button>
        <Button
          size="sm"
          variant="outline"
          accentColor="blue"
          onClick={controls.export_layout}
          className="w-full text-xs justify-start"
        >
          <span className="mr-2">💾</span>
          Export Layout
        </Button>
      </div>
    </div>
  );

  return (
    <div className="w-80 space-y-4 text-white">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Workflow Controls</h3>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs"
        >
          {showAdvanced ? 'Hide' : 'Show'} Advanced
        </Button>
      </div>

      {/* Zoom Controls */}
      <Card className="p-3 bg-gray-800/50 border-gray-600">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-medium text-gray-300">Zoom & View</h4>
        </div>
        <ZoomControls />
      </Card>

      {/* Layout Controls */}
      <Card className="p-3 bg-gray-800/50 border-gray-600">
        <LayoutControls />
      </Card>

      {/* Animation Controls */}
      <Card className="p-3 bg-gray-800/50 border-gray-600">
        <AnimationControls />
      </Card>

      {/* Statistics */}
      <StatisticsPanel />

      {/* Advanced Controls */}
      {showAdvanced && (
        <>
          <Card className="p-3 bg-gray-800/50 border-gray-600">
            <ViewControls />
          </Card>
          <Card className="p-3 bg-gray-800/50 border-gray-600">
            <ActionControls />
          </Card>
        </>
      )}

      {/* Legend */}
      <Card className="p-3 bg-gray-800/50 border-gray-600">
        <h4 className="text-xs font-medium text-gray-300 mb-2">Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Active Agent</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span>Idle Agent</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
            <span>Hibernated Agent</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-0.5 bg-blue-500"></div>
            <span>Active Communication</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-0.5 bg-gray-500 border-dashed"></div>
            <span>Pending Communication</span>
          </div>
        </div>
      </Card>
    </div>
  );
};