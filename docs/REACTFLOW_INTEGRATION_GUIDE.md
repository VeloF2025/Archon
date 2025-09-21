# ReactFlow Integration Guide

This guide provides comprehensive documentation for integrating ReactFlow components with the Archon workflow system.

## Overview

The Archon workflow system provides full integration with ReactFlow, allowing you to build visual workflow editors and executors with real-time updates and comprehensive analytics.

## Key Features

- **ReactFlow Data Compatibility**: Full support for ReactFlow node and edge data structures
- **Real-time Updates**: Socket.IO events for live workflow execution status
- **API Integration**: RESTful endpoints for workflow CRUD operations
- **Execution Engine**: Parallel step execution with agent orchestration
- **Analytics & Insights**: Performance metrics and optimization recommendations
- **Validation System**: Comprehensive integration validation tools

## Architecture

### Data Flow

```
ReactFlow UI → API Endpoints → Workflow Service → Execution Engine → Agents
     ↓                    ↓                ↓               ↓
Real-time Updates ← Socket.IO Events ← Analytics Service ← Database
```

### Components

1. **ReactFlow Components**: Frontend visual workflow editor
2. **API Layer**: RESTful endpoints and MCP tools
3. **Service Layer**: Business logic and workflow orchestration
4. **Execution Engine**: Parallel step execution with agent assignment
5. **Analytics Service**: Performance metrics and insights
6. **Validation System**: Integration testing and validation

## Quick Start

### 1. Setting up the Backend

```python
# Install dependencies
uv sync

# Start the services
docker-compose up --build -d
```

### 2. Running Integration Validation

```bash
# Run comprehensive validation
python scripts/validate_reactflow_integration.py

# Generate HTML report
python scripts/validate_reactflow_integration.py --format html --output validation_report.html

# Run with custom database
python scripts/validate_reactflow_integration.py --database-url "postgresql://user:pass@localhost/archon"
```

### 3. Basic ReactFlow Integration

```javascript
import React, { useState, useCallback } from 'react';
import ReactFlow, {
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  NodeChange,
  EdgeChange,
  Connection,
  Edge,
  Node,
} from 'reactflow';
import 'reactflow/dist/style.css';

const WorkflowEditor = () => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  // Load workflow from backend
  const loadWorkflow = async (workflowId) => {
    const response = await fetch(`/api/workflows/${workflowId}`);
    const workflow = await response.json();

    setNodes(workflow.reactflow_data.nodes);
    setEdges(workflow.reactflow_data.edges);
  };

  // Save workflow to backend
  const saveWorkflow = async () => {
    const workflowData = {
      name: 'My Workflow',
      description: 'Created with ReactFlow',
      reactflow_data: { nodes, edges }
    };

    const response = await fetch('/api/workflows', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(workflowData)
    });

    return await response.json();
  };

  // Handle node changes
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((nds) => applyNodeChanges(changes, nds));
    },
    []
  );

  // Handle edge changes
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => applyEdgeChanges(changes, eds));
    },
    []
  );

  // Handle new connections
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge(connection, eds));
    },
    []
  );

  return (
    <div style={{ width: '100%', height: '80vh' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
      >
        <Controls />
        <Background />
      </ReactFlow>

      <div style={{ padding: '20px' }}>
        <button onClick={saveWorkflow}>Save Workflow</button>
        <button onClick={() => loadWorkflow('workflow-id')}>Load Workflow</button>
      </div>
    </div>
  );
};

export default WorkflowEditor;
```

## API Reference

### Workflow Endpoints

#### Create Workflow
```http
POST /api/workflows
Content-Type: application/json

{
  "name": "My Workflow",
  "description": "Workflow description",
  "reactflow_data": {
    "nodes": [...],
    "edges": [...]
  }
}
```

#### Get Workflow
```http
GET /api/workflows/{id}
```

#### Update Workflow
```http
PUT /api/workflows/{id}
Content-Type: application/json

{
  "reactflow_data": {
    "nodes": [...],
    "edges": [...]
  }
}
```

#### Execute Workflow
```http
POST /api/workflows/{id}/execute
Content-Type: application/json

{
  "input_data": { "key": "value" }
}
```

### MCP Tools

#### Create Workflow
```python
result = await mcp_tools.create_workflow(
    name="My Workflow",
    description="Workflow description",
    reactflow_data={"nodes": [...], "edges": [...]}
)
```

#### Execute Workflow
```python
result = await mcp_tools.execute_workflow(
    workflow_id="workflow-id",
    input_data={"key": "value"}
)
```

#### Get Workflow Analytics
```python
result = await mcp_tools.get_workflow_analytics(
    workflow_id="workflow-id",
    time_range="7d"
)
```

## ReactFlow Data Format

### Node Structure

```javascript
{
  "id": "node-1",
  "position": { "x": 100, "y": 100 },
  "data": {
    "label": "Start Node",
    "type": "start",
    "config": {},
    "status": "idle"
  },
  "type": "input"
}
```

### Edge Structure

```javascript
{
  "id": "edge-1",
  "source": "node-1",
  "target": "node-2",
  "sourceHandle": "source",
  "targetHandle": "target",
  "type": "default",
  "animated": false,
  "style": { "stroke": "#555" }
}
```

### Node Types

| Type | ReactFlow Type | Description |
|------|----------------|-------------|
| `start` | `input` | Workflow starting point |
| `end` | `output` | Workflow ending point |
| `agent_task` | `default` | Agent execution task |
| `decision` | `decision` | Conditional logic |
| `api_call` | `api` | External API call |
| `data_transform` | `transform` | Data transformation |
| `parallel` | `parallel` | Parallel execution |
| `condition` | `condition` | Conditional branching |

## Real-time Updates

### Socket.IO Events

#### Workflow Execution Events
```javascript
// Workflow started
socket.on('workflow_execution_started', (data) => {
  console.log('Workflow started:', data);
  // Update UI to show running state
});

// Step execution update
socket.on('step_execution_update', (data) => {
  console.log('Step update:', data);
  // Update node status in ReactFlow
});

// Workflow completed
socket.on('workflow_execution_completed', (data) => {
  console.log('Workflow completed:', data);
  // Update UI to show completed state
});
```

#### Node Status Updates
```javascript
// Node status change
socket.on('node_status_update', (data) => {
  console.log('Node status:', data);
  // Update individual node in ReactFlow
  updateNodeStatus(data.node_id, data.status);
});
```

### Subscribing to Updates

```javascript
// Subscribe to workflow updates
socket.emit('subscribe_workflow', { workflow_id: 'workflow-id' });

// Unsubscribe from workflow updates
socket.emit('unsubscribe_workflow', { workflow_id: 'workflow-id' });
```

## Custom Node Components

### Basic Custom Node

```javascript
import React from 'react';
import { Handle, Position } from 'reactflow';

const CustomNode = ({ data, id }) => {
  const [status, setStatus] = useState(data.status || 'idle');

  // Listen for status updates
  useEffect(() => {
    socket.on('node_status_update', (updateData) => {
      if (updateData.node_id === id) {
        setStatus(updateData.status);
      }
    });
  }, [id]);

  const getStatusColor = () => {
    switch (status) {
      case 'running': return '#fbbf24';
      case 'completed': return '#10b981';
      case 'failed': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div style={{
      padding: '10px',
      border: `2px solid ${getStatusColor()}`,
      borderRadius: '5px',
      background: 'white'
    }}>
      <Handle type="target" position={Position.Top} />
      <div>{data.label}</div>
      <div style={{ fontSize: '12px', color: getStatusColor() }}>
        {status}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default CustomNode;
```

### Agent Task Node

```javascript
const AgentTaskNode = ({ data, id }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    socket.on('node_progress_update', (updateData) => {
      if (updateData.node_id === id) {
        setProgress(updateData.progress);
      }
    });
  }, [id]);

  return (
    <div style={{
      padding: '15px',
      border: '2px solid #3b82f6',
      borderRadius: '8px',
      background: 'white',
      minWidth: '150px'
    }}>
      <Handle type="target" position={Position.Top} />
      <div style={{ fontWeight: 'bold' }}>{data.label}</div>
      <div style={{ fontSize: '12px', color: '#6b7280' }}>
        Agent: {data.config?.agent_type || 'Unknown'}
      </div>
      {progress > 0 && (
        <div style={{ marginTop: '5px' }}>
          <div style={{
            height: '4px',
            background: '#e5e7eb',
            borderRadius: '2px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              background: '#3b82f6',
              width: `${progress}%`
            }} />
          </div>
          <div style={{ fontSize: '10px', textAlign: 'center' }}>
            {progress}%
          </div>
        </div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};
```

## Validation and Testing

### Running Tests

```bash
# Run all ReactFlow integration tests
python -m pytest tests/test_reactflow_integration.py -v

# Run specific test categories
python -m pytest tests/test_reactflow_integration.py::TestReactFlowDataFormatValidation -v
python -m pytest tests/test_reactflow_integration.py::TestReactFlowAPIIntegration -v
python -m pytest tests/test_reactflow_integration.py::TestReactFlowRealtimeUpdates -v
```

### Validation Script

```bash
# Generate comprehensive validation report
python scripts/validate_reactflow_integration.py --format html --output reactflow_validation.html

# Run validation with custom database
python scripts/validate_reactflow_integration.py --database-url "postgresql://user:pass@localhost/archon" --format json --output validation_results.json
```

## Performance Optimization

### Best Practices

1. **Lazy Loading**: Load large workflows in chunks
2. **Virtualization**: Use ReactFlow's built-in virtualization for large graphs
3. **Debouncing**: Debounce rapid changes during editing
4. **Caching**: Cache workflow data locally
5. **Optimistic Updates**: Update UI optimistically during execution

### Performance Monitoring

```javascript
// Monitor performance metrics
const performanceMetrics = {
  nodeCount: nodes.length,
  edgeCount: edges.length,
  renderTime: 0,
  updateFrequency: 0
};

// Track render performance
const startRender = performance.now();
// ... render operations ...
performanceMetrics.renderTime = performance.now() - startRender;
```

## Troubleshooting

### Common Issues

1. **Nodes Not Rendering**
   - Check that node IDs are unique
   - Verify position data is valid
   - Ensure node type is registered

2. **Edges Not Connecting**
   - Verify source and target node IDs exist
   - Check handle IDs match
   - Ensure proper node positioning

3. **Real-time Updates Not Working**
   - Verify Socket.IO connection
   - Check event subscription
   - Ensure proper event data structure

4. **Performance Issues**
   - Monitor node and edge count
   - Check for excessive re-renders
   - Optimize update frequency

### Debug Tools

```javascript
// Enable ReactFlow debug mode
<ReactFlow debugMode={true} />

// Log all events
socket.onAny((event, ...args) => {
  console.log(`Socket.IO Event: ${event}`, args);
});
```

## Advanced Features

### Workflow Templates

```javascript
const workflowTemplates = {
  'data-processing': {
    nodes: [
      { id: 'start', type: 'input', data: { label: 'Start' }, position: { x: 0, y: 0 } },
      { id: 'process', type: 'default', data: { label: 'Process Data' }, position: { x: 200, y: 0 } },
      { id: 'validate', type: 'decision', data: { label: 'Validate' }, position: { x: 400, y: 0 } },
      { id: 'end', type: 'output', data: { label: 'End' }, position: { x: 600, y: 0 } }
    ],
    edges: [
      { id: 'e1', source: 'start', target: 'process' },
      { id: 'e2', source: 'process', target: 'validate' },
      { id: 'e3', source: 'validate', target: 'end' }
    ]
  }
};
```

### Workflow Analytics

```javascript
// Load workflow analytics
const loadAnalytics = async (workflowId) => {
  const response = await fetch(`/api/workflows/${workflowId}/analytics`);
  const analytics = await response.json();

  // Display analytics in dashboard
  updateAnalyticsDashboard(analytics);
};
```

### Custom Validation

```javascript
// Validate workflow before saving
const validateWorkflow = (nodes, edges) => {
  const errors = [];

  // Check for disconnected nodes
  const connectedNodeIds = new Set();
  edges.forEach(edge => {
    connectedNodeIds.add(edge.source);
    connectedNodeIds.add(edge.target);
  });

  nodes.forEach(node => {
    if (!connectedNodeIds.has(node.id)) {
      errors.push(`Node ${node.id} is not connected`);
    }
  });

  // Check for cycles
  if (hasCycles(nodes, edges)) {
    errors.push('Workflow contains cycles');
  }

  return errors;
};
```

## Contributing

When contributing to ReactFlow integration:

1. Run all validation tests before submitting
2. Follow the established code patterns
3. Update documentation for new features
4. Test with different workflow sizes
5. Ensure backward compatibility

## License

This integration is part of the Archon project and follows the same license terms.

---

For more information, see the [Archon Documentation](../README.md) and [API Reference](../API_REFERENCE.md).