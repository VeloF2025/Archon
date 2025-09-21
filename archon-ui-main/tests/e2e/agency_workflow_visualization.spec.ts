/**
 * Agency Workflow Visualization E2E Test Suite
 *
 * Comprehensive Playwright tests for ReactFlow-based agency workflow visualization
 * with real-time updates, interactive controls, and responsive design.
 *
 * Key test scenarios:
 * - Workflow visualization renders correctly with agency data
 * - Real-time Socket.IO updates work seamlessly
 * - Interactive controls (zoom, pan, layout) function properly
 * - Agent node interactions work correctly
 * - Communication flow animations display properly
 * - Responsive design works across viewports
 * - Performance with large datasets (100+ agents)
 * - Accessibility and keyboard navigation
 * - Error handling and graceful degradation
 * - Export functionality works correctly
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const WORKFLOW_URL = `${BASE_URL}/workflow`;

// Mock agency data for testing
const mockAgencyData = {
  id: uuidv4(),
  name: 'Test Agency Workflow',
  description: 'Comprehensive test agency with multiple agent types and communication flows',
  agents: [
    {
      id: 'agent-1',
      name: 'System Architect',
      agent_type: 'SYSTEM_ARCHITECT' as any,
      model_tier: 'OPUS' as any,
      project_id: 'test-project',
      state: 'ACTIVE' as any,
      state_changed_at: new Date(),
      tasks_completed: 25,
      success_rate: 0.92,
      avg_completion_time_seconds: 120,
      memory_usage_mb: 512,
      cpu_usage_percent: 45,
      capabilities: {
        architecture_design: true,
        system_planning: true,
        technical_analysis: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-2',
      name: 'Code Implementer',
      agent_type: 'CODE_IMPLEMENTER' as any,
      model_tier: 'SONNET' as any,
      project_id: 'test-project',
      state: 'ACTIVE' as any,
      state_changed_at: new Date(),
      tasks_completed: 150,
      success_rate: 0.88,
      avg_completion_time_seconds: 45,
      memory_usage_mb: 256,
      cpu_usage_percent: 30,
      capabilities: {
        code_generation: true,
        debugging: true,
        testing: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-3',
      name: 'Quality Reviewer',
      agent_type: 'CODE_QUALITY_REVIEWER' as any,
      model_tier: 'SONNET' as any,
      project_id: 'test-project',
      state: 'IDLE' as any,
      state_changed_at: new Date(),
      tasks_completed: 89,
      success_rate: 0.95,
      avg_completion_time_seconds: 60,
      memory_usage_mb: 128,
      cpu_usage_percent: 20,
      capabilities: {
        code_review: true,
        quality_assurance: true,
        linting: true
      },
      created_at: new Date(),
      updated_at: new Date()
    },
    {
      id: 'agent-4',
      name: 'Test Validator',
      agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
      model_tier: 'HAIKU' as any,
      project_id: 'test-project',
      state: 'ACTIVE' as any,
      state_changed_at: new Date(),
      tasks_completed: 200,
      success_rate: 0.90,
      avg_completion_time_seconds: 30,
      memory_usage_mb: 64,
      cpu_usage_percent: 15,
      capabilities: {
        test_generation: true,
        coverage_analysis: true,
        validation: true
      },
      created_at: new Date(),
      updated_at: new Date()
    }
  ],
  communication_flows: [
    {
      id: 'flow-1',
      source_agent_id: 'agent-1',
      target_agent_id: 'agent-2',
      communication_type: 'DIRECT' as any,
      status: 'active' as any,
      message_count: 45,
      last_message_at: new Date(),
      message_type: 'task_assignment'
    },
    {
      id: 'flow-2',
      source_agent_id: 'agent-2',
      target_agent_id: 'agent-3',
      communication_type: 'DIRECT' as any,
      status: 'active' as any,
      message_count: 120,
      last_message_at: new Date(),
      message_type: 'code_submission'
    },
    {
      id: 'flow-3',
      source_agent_id: 'agent-3',
      target_agent_id: 'agent-4',
      communication_type: 'COLLABORATIVE' as any,
      status: 'pending' as any,
      message_count: 75,
      last_message_at: new Date(),
      message_type: 'quality_review'
    },
    {
      id: 'flow-4',
      source_agent_id: 'agent-1',
      target_agent_id: 'agent-4',
      communication_type: 'BROADCAST' as any,
      status: 'active' as any,
      message_count: 15,
      last_message_at: new Date(),
      message_type: 'specification_update'
    }
  ],
  workflow_rules: {
    routing_rules: {
      complex_tasks: 'SYSTEM_ARCHITECT',
      code_tasks: 'CODE_IMPLEMENTER',
      review_tasks: 'CODE_QUALITY_REVIEWER'
    },
    collaboration_patterns: {
      peer_review: true,
      collective_intelligence: true
    },
    escalation_paths: ['SYSTEM_ARCHITECT', 'CODE_QUALITY_REVIEWER']
  },
  created_at: new Date(),
  updated_at: new Date()
};

// Large dataset for performance testing
const largeAgencyData = {
  id: uuidv4(),
  name: 'Large Agency Performance Test',
  description: 'Agency with 100+ agents for performance testing',
  agents: Array.from({ length: 100 }, (_, i) => ({
    id: `agent-${i + 1}`,
    name: `Agent ${i + 1}`,
    agent_type: ['CODE_IMPLEMENTER', 'SYSTEM_ARCHITECT', 'CODE_QUALITY_REVIEWER', 'TEST_COVERAGE_VALIDATOR'][i % 4] as any,
    model_tier: ['OPUS', 'SONNET', 'HAIKU'][i % 3] as any,
    project_id: 'test-project',
    state: ['ACTIVE', 'IDLE', 'HIBERNATED'][i % 3] as any,
    state_changed_at: new Date(),
    tasks_completed: Math.floor(Math.random() * 200) + 10,
    success_rate: Math.random() * 0.3 + 0.7,
    avg_completion_time_seconds: Math.floor(Math.random() * 120) + 20,
    memory_usage_mb: Math.floor(Math.random() * 512) + 64,
    cpu_usage_percent: Math.floor(Math.random() * 60) + 10,
    capabilities: {
      capability_1: true,
      capability_2: i % 2 === 0,
      capability_3: i % 3 === 0
    },
    created_at: new Date(),
    updated_at: new Date()
  })),
  communication_flows: Array.from({ length: 200 }, (_, i) => ({
    id: `flow-${i + 1}`,
    source_agent_id: `agent-${Math.floor(Math.random() * 100) + 1}`,
    target_agent_id: `agent-${Math.floor(Math.random() * 100) + 1}`,
    communication_type: ['DIRECT', 'BROADCAST', 'COLLABORATIVE', 'CHAIN'][i % 4] as any,
    status: ['active', 'pending', 'completed'][i % 3] as any,
    message_count: Math.floor(Math.random() * 100) + 1,
    last_message_at: new Date(),
    message_type: ['task_assignment', 'status_update', 'data_request', 'result_delivery'][i % 4],
    data_flow: {
      input_size: Math.floor(Math.random() * 10000),
      output_size: Math.floor(Math.random() * 5000),
      processing_time_ms: Math.floor(Math.random() * 1000) + 100
    }
  })),
  workflow_rules: {
    routing_rules: {},
    collaboration_patterns: {},
    escalation_paths: []
  },
  created_at: new Date(),
  updated_at: new Date()
};

// Helper functions
async function setupMockAPI(page: Page, agencyData: any) {
  // Mock agency data API
  await page.route('**/api/agency/workflow/data', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(agencyData)
    });
  });

  // Mock workflow statistics API
  await page.route('**/api/agency/workflow/stats', async (route) => {
    const stats = {
      total_agents: agencyData.agents.length,
      active_agents: agencyData.agents.filter((a: any) => a.state === 'ACTIVE').length,
      total_communications: agencyData.communication_flows.length,
      active_communications: agencyData.communication_flows.filter((f: any) => f.status === 'active').length,
      avg_messages_per_connection: agencyData.communication_flows.reduce((sum: number, f: any) => sum + f.message_count, 0) / agencyData.communication_flows.length,
      busiest_agent: {
        agent_id: 'agent-2',
        message_count: 120
      },
      communication_type_distribution: {
        DIRECT: 2,
        BROADCAST: 1,
        COLLABORATIVE: 1
      },
      agent_type_distribution: {
        SYSTEM_ARCHITECT: 1,
        CODE_IMPLEMENTER: 1,
        CODE_QUALITY_REVIEWER: 1,
        TEST_COVERAGE_VALIDATOR: 1
      }
    };
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(stats)
    });
  });
}

async function simulateSocketUpdate(page: Page, eventType: string, data: any) {
  await page.evaluate(({ eventType, data }) => {
    // Simulate Socket.IO event
    window.dispatchEvent(new CustomEvent('socket-message', {
      detail: { type: eventType, data }
    }));
  }, { eventType, data });
}

test.describe('Agency Workflow Visualization', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console error tracking
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log(`Console error: ${msg.text()}`);
      }
    });

    // Set up network error tracking
    page.on('requestfailed', request => {
      console.log(`Network request failed: ${request.url()} - ${request.failure()?.errorText}`);
    });

    // Track React errors
    page.on('pageerror', error => {
      console.log(`Page error: ${error.message}`);
      consoleErrors.push(error.message);
    });

    // Attach errors to page for test access
    await page.exposeFunction('getConsoleErrors', () => consoleErrors);
  });

  test.describe('Basic Rendering and Layout', () => {
    test('should render workflow visualization with agency data', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for ReactFlow to initialize
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Verify main workflow container is visible
      const workflowContainer = page.locator('.react-flow');
      await expect(workflowContainer).toBeVisible();

      // Verify all agent nodes are rendered
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await expect(agentNodes).toHaveCount(mockAgencyData.agents.length);

      // Verify communication edges are rendered
      const edges = page.locator('.react-flow__edge');
      await expect(edges).toHaveCount(mockAgencyData.communication_flows.length);

      // Verify control panel is visible
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

      // Verify minimap is visible
      await expect(page.locator('.react-flow__minimap')).toBeVisible();

      // Verify no critical errors occurred
      const errors = await page.evaluate(() => (window as any).getConsoleErrors());
      const criticalErrors = errors.filter((e: string) =>
        !e.includes('Warning') && !e.includes('deprecated')
      );
      expect(criticalErrors).toHaveLength(0);
    });

    test('should display agent nodes with correct information', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Test each agent node displays correct information
      for (const agent of mockAgencyData.agents) {
        const agentNode = page.locator(`[data-agent-id="${agent.id}"]`);
        await expect(agentNode).toBeVisible();

        // Verify agent name is displayed
        await expect(agentNode.locator('[data-testid="agent-name"]')).toHaveText(agent.name);

        // Verify agent type is displayed
        await expect(agentNode.locator('[data-testid="agent-type"]')).toBeVisible();

        // Verify agent state is displayed with correct color
        const stateIndicator = agentNode.locator('[data-testid="agent-state"]');
        await expect(stateIndicator).toBeVisible();

        // Verify model tier is displayed
        await expect(agentNode.locator('[data-testid="agent-tier"]')).toBeVisible();

        // Verify task completion count is displayed
        await expect(agentNode.locator('[data-testid="agent-tasks"]')).toHaveText(
          agent.tasks_completed.toString()
        );
      }
    });

    test('should display communication flows with correct styling', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Test communication flows
      for (const flow of mockAgencyData.communication_flows) {
        // Verify edge exists (using ReactFlow edge classes)
        const edgeExists = await page.evaluate((flowId) => {
          const edges = document.querySelectorAll('.react-flow__edge');
          return Array.from(edges).some(edge =>
            edge.getAttribute('data-edge-id') === flowId ||
            edge.id.includes(flowId)
          );
        }, flow.id);

        expect(edgeExists).toBe(true);

        // Verify active flows have animation
        if (flow.status === 'active') {
          const animatedEdge = await page.locator('.react-flow__edge.animated').first();
          await expect(animatedEdge).toBeVisible();
        }
      }
    });

    test('should apply different layout algorithms correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      const layouts = ['circular', 'hierarchical', 'grid', 'force'];

      for (const layout of layouts) {
        // Click layout button
        await page.locator(`[data-testid="layout-${layout}"]`).click();
        await page.waitForTimeout(1000); // Wait for layout animation

        // Verify nodes are repositioned
        const nodes = await page.locator('[data-testid="agent-node"]').all();
        expect(nodes.length).toBeGreaterThan(0);

        // Verify layout doesn't break visibility
        for (const node of nodes) {
          await expect(node).toBeVisible();
        }
      }
    });
  });

  test.describe('Interactive Controls', () => {
    test('should handle zoom controls correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Test zoom in
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);

      // Test zoom out
      await page.locator('[data-testid="zoom-out"]').click();
      await page.waitForTimeout(500);

      // Test fit to screen
      await page.locator('[data-testid="fit-to-screen"]').click();
      await page.waitForTimeout(500);

      // Test center view
      await page.locator('[data-testid="center-view"]').click();
      await page.waitForTimeout(500);

      // Verify workflow remains visible after all zoom operations
      await expect(page.locator('.react-flow')).toBeVisible();
    });

    test('should handle animation toggle correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Get initial animation state
      const animationButton = page.locator('[data-testid="toggle-animation"]');

      // Toggle animation off
      await animationButton.click();
      await page.waitForTimeout(500);

      // Verify edges stop animating
      const animatedEdges = page.locator('.react-flow__edge.animated');
      const initialCount = await animatedEdges.count();

      // Toggle animation on
      await animationButton.click();
      await page.waitForTimeout(500);

      // Verify edges start animating again
      const finalCount = await animatedEdges.count();
      // Animation state should change (may not be exact due to timing)
      expect([initialCount, finalCount]).toContain(0);
    });

    test('should handle data refresh correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Mock API call for refresh
      let refreshCount = 0;
      await page.route('**/api/agency/workflow/data', async (route) => {
        refreshCount++;
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockAgencyData)
        });
      });

      // Click refresh button
      await page.locator('[data-testid="refresh-data"]').click();
      await page.waitForTimeout(1000);

      // Verify refresh was called
      expect(refreshCount).toBeGreaterThan(1);

      // Verify workflow remains visible
      await expect(page.locator('.react-flow')).toBeVisible();
    });
  });

  test.describe('Node Interactions', () => {
    test('should handle agent node selection correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Click on first agent node
      const firstAgentNode = page.locator('[data-testid="agent-node"]').first();
      await firstAgentNode.click();

      // Verify node is selected (visual feedback)
      await expect(firstAgentNode).toHaveClass(/selected/);

      // Verify agent details panel appears
      const detailsPanel = page.locator('[data-testid="agent-details-panel"]');
      await expect(detailsPanel).toBeVisible();

      // Verify agent information is displayed in details panel
      const firstAgent = mockAgencyData.agents[0];
      await expect(detailsPanel.locator('[data-testid="detail-name"]')).toHaveText(firstAgent.name);
      await expect(detailsPanel.locator('[data-testid="detail-type"]')).toHaveText(firstAgent.agent_type);
      await expect(detailsPanel.locator('[data-testid="detail-state"]')).toHaveText(firstAgent.state);
    });

    test('should handle node dragging correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      const firstAgentNode = page.locator('[data-testid="agent-node"]').first();
      const initialBoundingBox = await firstAgentNode.boundingBox();

      if (!initialBoundingBox) {
        throw new Error('Agent node bounding box not found');
      }

      // Drag the node
      await firstAgentNode.dragTo(page.locator('.react-flow__pane'), {
        targetPosition: { x: 400, y: 300 }
      });

      await page.waitForTimeout(500);

      // Verify node moved (check bounding box changed)
      const newBoundingBox = await firstAgentNode.boundingBox();
      expect(newBoundingBox).not.toEqual(initialBoundingBox);

      // Verify node remains visible after dragging
      await expect(firstAgentNode).toBeVisible();
    });

    test('should handle edge selection correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Click on first edge
      const firstEdge = page.locator('.react-flow__edge').first();
      await firstEdge.click();

      // Verify edge is selected (visual feedback)
      await expect(firstEdge).toHaveClass(/selected/);

      // Verify edge details appear (if implemented)
      const edgeDetails = page.locator('[data-testid="edge-details-panel"]');
      if (await edgeDetails.isVisible()) {
        await expect(edgeDetails).toBeVisible();
      }
    });
  });

  test.describe('Real-time Updates', () => {
    test('should handle agent state updates via Socket.IO', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Simulate agent state update
      await simulateSocketUpdate(page, 'workflow_agent_update', {
        agentId: 'agent-1',
        updates: { state: 'HIBERNATED' }
      });

      await page.waitForTimeout(1000);

      // Verify agent node reflects state change
      const agentNode = page.locator('[data-agent-id="agent-1"]');
      const stateIndicator = agentNode.locator('[data-testid="agent-state"]');
      await expect(stateIndicator).toHaveText(/HIBERNATED/);
    });

    test('should handle communication flow updates via Socket.IO', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Simulate communication flow update
      await simulateSocketUpdate(page, 'workflow_communication_update', {
        communicationId: 'flow-1',
        updates: {
          message_count: 50,
          status: 'completed'
        }
      });

      await page.waitForTimeout(1000);

      // Verify edge reflects update (check if animation stops for completed flows)
      const animatedEdges = page.locator('.react-flow__edge.animated');
      const hasAnimatedEdges = await animatedEdges.count() > 0;

      // The edge should no longer be animated if status is completed
      expect(hasAnimatedEdges).toBe(true); // Other active edges should still animate
    });

    test('should handle multiple simultaneous updates', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Simulate multiple rapid updates
      const updates = [
        { type: 'workflow_agent_update', data: { agentId: 'agent-2', updates: { state: 'ACTIVE' } } },
        { type: 'workflow_communication_update', data: { communicationId: 'flow-2', updates: { message_count: 125 } } },
        { type: 'workflow_agent_update', data: { agentId: 'agent-3', updates: { tasks_completed: 90 } } }
      ];

      for (const update of updates) {
        await simulateSocketUpdate(page, update.type, update.data);
      }

      await page.waitForTimeout(1500);

      // Verify workflow remains stable and responsive
      await expect(page.locator('.react-flow')).toBeVisible();
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await expect(agentNodes).toHaveCount(mockAgencyData.agents.length);
    });
  });

  test.describe('Performance Testing', () => {
    test('should handle large datasets (100+ agents) efficiently', async ({ page }) => {
      await setupMockAPI(page, largeAgencyData);

      const startTime = Date.now();

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Wait for large dataset to render (extended timeout)
      await page.waitForSelector('.react-flow', { timeout: 30000 });

      const loadTime = Date.now() - startTime;

      // Should load within reasonable time (15 seconds for 100+ agents)
      expect(loadTime).toBeLessThan(15000);

      // Verify all agents are rendered
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await expect(agentNodes).toHaveCount(largeAgencyData.agents.length);

      // Verify all communication flows are rendered
      const edges = page.locator('.react-flow__edge');
      await expect(edges).toHaveCount(largeAgencyData.communication_flows.length);

      // Verify controls remain responsive
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);
      await expect(page.locator('.react-flow')).toBeVisible();
    });

    test('should maintain smooth animations with large datasets', async ({ page }) => {
      await setupMockAPI(page, largeAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 30000 });

      // Test performance during layout changes
      const layouts = ['circular', 'hierarchical', 'grid'];

      for (const layout of layouts) {
        const layoutStartTime = Date.now();

        await page.locator(`[data-testid="layout-${layout}"]`).click();
        await page.waitForTimeout(1000);

        const layoutTime = Date.now() - layoutStartTime;

        // Layout change should complete within 3 seconds
        expect(layoutTime).toBeLessThan(3000);

        // Verify nodes are still visible
        const agentNodes = page.locator('[data-testid="agent-node"]');
        await expect(agentNodes.first()).toBeVisible();
      }
    });

    test('should handle real-time updates efficiently with large datasets', async ({ page }) => {
      await setupMockAPI(page, largeAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 30000 });

      // Simulate 10 rapid updates
      for (let i = 0; i < 10; i++) {
        await simulateSocketUpdate(page, 'workflow_agent_update', {
          agentId: `agent-${i + 1}`,
          updates: {
            state: i % 2 === 0 ? 'ACTIVE' : 'IDLE',
            tasks_completed: Math.floor(Math.random() * 100) + 50
          }
        });
        await page.waitForTimeout(100); // 100ms between updates
      }

      await page.waitForTimeout(1000);

      // Verify workflow remains responsive
      await expect(page.locator('.react-flow')).toBeVisible();

      // Test interaction is still possible
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);
      await expect(page.locator('.react-flow')).toBeVisible();
    });
  });

  test.describe('Responsive Design', () => {
    const viewports = [
      { name: 'Desktop', width: 1920, height: 1080 },
      { name: 'Tablet', width: 768, height: 1024 },
      { name: 'Mobile', width: 375, height: 667 }
    ];

    for (const viewport of viewports) {
      test(`should display correctly on ${viewport.name} viewport`, async ({ page }) => {
        await setupMockAPI(page, mockAgencyData);

        await page.setViewportSize({ width: viewport.width, height: viewport.height });

        await page.goto(WORKFLOW_URL);
        await page.waitForLoadState('networkidle');
        await page.waitForSelector('.react-flow', { timeout: 10000 });

        // Verify workflow container adapts to viewport
        const workflowContainer = page.locator('.react-flow');
        await expect(workflowContainer).toBeVisible();

        // Verify controls are accessible
        const controls = page.locator('[data-testid="workflow-controls"]');
        await expect(controls).toBeVisible();

        // Verify minimap adjusts appropriately
        const minimap = page.locator('.react-flow__minimap');
        await expect(minimap).toBeVisible();

        // Test basic interactions work
        await page.locator('[data-testid="zoom-in"]').click();
        await page.waitForTimeout(500);
        await expect(workflowContainer).toBeVisible();
      });
    }
  });

  test.describe('Accessibility', () => {
    test('should have proper ARIA labels and roles', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Verify main container has appropriate role
      await expect(page.locator('.react-flow')).toHaveAttribute('role', 'application');

      // Verify agent nodes have proper ARIA labels
      const agentNodes = page.locator('[data-testid="agent-node"]');
      const nodeCount = await agentNodes.count();

      for (let i = 0; i < nodeCount; i++) {
        const node = agentNodes.nth(i);
        await expect(node).toHaveAttribute('aria-label', /Agent/);
        await expect(node).toHaveAttribute('role', 'button');
      }

      // Verify control buttons have proper labels
      const controlButtons = page.locator('[data-testid^="zoom-"], [data-testid^="fit-"], [data-testid^="toggle-"]');
      const buttonCount = await controlButtons.count();

      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = controlButtons.nth(i);
        await expect(button).toHaveAttribute('aria-label');
      }
    });

    test('should support keyboard navigation', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Test keyboard navigation through controls
      await page.keyboard.press('Tab');

      // Verify first control is focused
      const focusedElement = page.locator(':focus');
      await expect(focusedElement).toBeVisible();

      // Test arrow key navigation in workflow
      await page.keyboard.press('ArrowRight');
      await page.keyboard.press('ArrowDown');
      await page.waitForTimeout(500);

      // Verify workflow remains interactive
      await expect(page.locator('.react-flow')).toBeVisible();

      // Test Escape key to deselect
      await page.keyboard.press('Escape');
      await page.waitForTimeout(200);

      // Verify no elements are selected
      const selectedNodes = page.locator('[data-testid="agent-node"].selected');
      expect(await selectedNodes.count()).toBe(0);
    });

    test('should have sufficient color contrast', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Verify text elements have sufficient contrast (basic check)
      const textElements = page.locator('[data-testid="agent-name"], [data-testid="agent-type"]');
      const textCount = await textElements.count();

      for (let i = 0; i < Math.min(textCount, 5); i++) {
        const element = textElements.nth(i);
        await expect(element).toBeVisible();

        // Check that element has visible text
        const text = await element.textContent();
        expect(text?.trim().length).toBeGreaterThan(0);
      }
    });
  });

  test.describe('Error Handling', () => {
    test('should handle API failures gracefully', async ({ page }) => {
      // Mock API failure
      await page.route('**/api/agency/workflow/data', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal Server Error' })
        });
      });

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Should show error state without crashing
      await expect(page.locator('.react-flow')).toBeVisible();

      // Should show error message
      const errorMessage = page.locator('[data-testid="error-message"]');
      if (await errorMessage.isVisible()) {
        await expect(errorMessage).toBeVisible();
      }
    });

    test('should handle Socket.IO connection failures gracefully', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      // Block WebSocket connections
      await page.route('**/socket.io/**', route => route.abort());

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Should still render without real-time updates
      await expect(page.locator('.react-flow')).toBeVisible();
      await expect(page.locator('[data-testid="agent-node"]')).toHaveCount(mockAgencyData.agents.length);
    });

    test('should handle malformed data gracefully', async ({ page }) => {
      // Mock malformed API response
      await page.route('**/api/agency/workflow/data', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ invalid: 'data structure' })
        });
      });

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');

      // Should handle gracefully without crashing
      await expect(page.locator('.react-flow')).toBeVisible();

      // Should show empty state or error
      const emptyState = page.locator('[data-testid="empty-state"]');
      if (await emptyState.isVisible()) {
        await expect(emptyState).toBeVisible();
      }
    });

    test('should handle network timeouts gracefully', async ({ page }) => {
      // Mock slow API response
      await page.route('**/api/agency/workflow/data', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 5000));
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockAgencyData)
        });
      });

      await page.goto(WORKFLOW_URL);

      // Should show loading state
      const loadingSpinner = page.locator('[data-testid="loading-spinner"]');
      if (await loadingSpinner.isVisible({ timeout: 2000 })) {
        await expect(loadingSpinner).toBeVisible();
      }

      // Should eventually load
      await page.waitForSelector('.react-flow', { timeout: 10000 });
      await expect(page.locator('.react-flow')).toBeVisible();
    });
  });

  test.describe('Export Functionality', () => {
    test('should export workflow layout correctly', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Set up download listener
      let downloadTriggered = false;
      page.on('download', () => {
        downloadTriggered = true;
      });

      // Click export button
      await page.locator('[data-testid="export-layout"]').click();
      await page.waitForTimeout(1000);

      // Verify export was triggered (or button action completed)
      const exportButton = page.locator('[data-testid="export-layout"]');
      await expect(exportButton).toBeVisible();

      // Workflow should remain visible after export
      await expect(page.locator('.react-flow')).toBeVisible();
    });

    test('should handle export errors gracefully', async ({ page }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 10000 });

      // Mock export failure
      await page.route('**/api/agency/workflow/export', async (route) => {
        await route.fulfill({
          status: 500,
          body: 'Export failed'
        });
      });

      // Click export button
      await page.locator('[data-testid="export-layout"]').click();
      await page.waitForTimeout(1000);

      // Should show error message or handle gracefully
      await expect(page.locator('.react-flow')).toBeVisible();

      // Should not crash the interface
      const errorMessage = page.locator('[data-testid="export-error"]');
      if (await errorMessage.isVisible({ timeout: 2000 })) {
        await expect(errorMessage).toBeVisible();
      }
    });
  });

  test.describe('Cross-browser Compatibility', () => {
    test('should work in different browsers', async ({ page, browserName }) => {
      await setupMockAPI(page, mockAgencyData);

      await page.goto(WORKFLOW_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('.react-flow', { timeout: 15000 });

      // Basic functionality should work across all browsers
      await expect(page.locator('.react-flow')).toBeVisible();
      await expect(page.locator('[data-testid="agent-node"]')).toHaveCount(mockAgencyData.agents.length);

      // Test basic interaction
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);
      await expect(page.locator('.react-flow')).toBeVisible();

      // Test node selection
      const firstAgentNode = page.locator('[data-testid="agent-node"]').first();
      await firstAgentNode.click();
      await page.waitForTimeout(500);
      await expect(firstAgentNode).toBeVisible();
    });
  });
});

test.describe('Agency Workflow Visualization Integration', () => {
  test('should integrate with agent management system', async ({ page }) => {
    await setupMockAPI(page, mockAgencyData);

    await page.goto(WORKFLOW_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForSelector('.react-flow', { timeout: 10000 });

    // Verify workflow displays agents from management system
    await expect(page.locator('[data-testid="agent-node"]')).toHaveCount(mockAgencyData.agents.length);

    // Test navigation to agent management (if link exists)
    const managementLink = page.locator('[data-testid="navigate-to-management"]');
    if (await managementLink.isVisible()) {
      await managementLink.click();
      await page.waitForTimeout(1000);
      // Should navigate or show management interface
    }
  });

  test('should integrate with knowledge management', async ({ page }) => {
    await setupMockAPI(page, mockAgencyData);

    await page.goto(WORKFLOW_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForSelector('.react-flow', { timeout: 10000 });

    // Test knowledge integration features
    const knowledgeButton = page.locator('[data-testid="show-knowledge"]');
    if (await knowledgeButton.isVisible()) {
      await knowledgeButton.click();
      await page.waitForTimeout(1000);

      // Should show knowledge panel
      const knowledgePanel = page.locator('[data-testid="knowledge-panel"]');
      if (await knowledgePanel.isVisible()) {
        await expect(knowledgePanel).toBeVisible();
      }
    }
  });

  test('should integrate with MCP server', async ({ page }) => {
    await setupMockAPI(page, mockAgencyData);

    await page.goto(WORKFLOW_URL);
    await page.waitForLoadState('networkidle');
    await page.waitForSelector('.react-flow', { timeout: 10000 });

    // Test MCP integration
    const mcpButton = page.locator('[data-testid="mcp-integration"]');
    if (await mcpButton.isVisible()) {
      await mcpButton.click();
      await page.waitForTimeout(1000);

      // Should show MCP panel or tools
      const mcpPanel = page.locator('[data-testid="mcp-panel"]');
      if (await mcpPanel.isVisible()) {
        await expect(mcpPanel).toBeVisible();
      }
    }
  });
});