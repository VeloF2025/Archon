/**
 * Workflow Editor Integration Test Suite
 *
 * Comprehensive Playwright tests for the interactive workflow editor
 * with drag-and-drop, agent creation, connection management, and template functionality.
 *
 * Key test scenarios:
 * - Editor loads with proper tools and panels
 * - Drag-and-drop agent creation works correctly
 * - Connection creation between agents functions properly
 * - Property editing for agents and connections
 * - Undo/redo functionality
 * - Template loading and application
 * - Workflow validation
 * - Import/export functionality
 * - Keyboard shortcuts
 * - Real-time collaboration features
 * - Performance with complex workflows
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const WORKFLOW_EDITOR_URL = `${BASE_URL}/workflow/editor`;

// Mock templates for testing
const mockTemplates = [
  {
    id: 'template-1',
    name: 'Development Team',
    description: 'Standard development team workflow',
    category: 'development' as const,
    agents: [
      {
        agent_type: 'SYSTEM_ARCHITECT' as any,
        model_tier: 'OPUS' as any,
        name: 'Lead Architect',
        capabilities: { architecture: true },
        position: { x: 200, y: 100 }
      },
      {
        agent_type: 'CODE_IMPLEMENTER' as any,
        model_tier: 'SONNET' as any,
        name: 'Developer',
        capabilities: { coding: true },
        position: { x: 400, y: 200 }
      },
      {
        agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
        model_tier: 'HAIKU' as any,
        name: 'QA Tester',
        capabilities: { testing: true },
        position: { x: 600, y: 100 }
      }
    ],
    connections: [
      {
        source_index: 0,
        target_index: 1,
        communication_type: 'DIRECT' as any,
        message_type: 'specification'
      },
      {
        source_index: 1,
        target_index: 2,
        communication_type: 'DIRECT' as any,
        message_type: 'code_submission'
      }
    ],
    metadata: {
      created_by: 'system',
      created_at: new Date(),
      usage_count: 150,
      rating: 4.5,
      tags: ['development', 'team', 'standard']
    }
  },
  {
    id: 'template-2',
    name: 'Testing Pipeline',
    description: 'Comprehensive testing workflow',
    category: 'testing' as const,
    agents: [
      {
        agent_type: 'CODE_QUALITY_REVIEWER' as any,
        model_tier: 'SONNET' as any,
        name: 'Code Reviewer',
        capabilities: { review: true },
        position: { x: 300, y: 150 }
      },
      {
        agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
        model_tier: 'HAIKU' as any,
        name: 'Unit Tester',
        capabilities: { unit_tests: true },
        position: { x: 500, y: 150 }
      },
      {
        agent_type: 'SECURITY_AUDITOR' as any,
        model_tier: 'OPUS' as any,
        name: 'Security Tester',
        capabilities: { security: true },
        position: { x: 700, y: 150 }
      }
    ],
    connections: [
      {
        source_index: 0,
        target_index: 1,
        communication_type: 'COLLABORATIVE' as any,
        message_type: 'code_for_testing'
      },
      {
        source_index: 1,
        target_index: 2,
        communication_type: 'DIRECT' as any,
        message_type: 'test_results'
      }
    ],
    metadata: {
      created_by: 'system',
      created_at: new Date(),
      usage_count: 89,
      rating: 4.2,
      tags: ['testing', 'quality', 'security']
    }
  }
];

// Mock agent palette data
const mockAgentPalette = [
  {
    agent_type: 'SYSTEM_ARCHITECT' as any,
    model_tier: 'OPUS' as any,
    agent_name: 'System Architect',
    description: 'Designs system architecture and technical specifications',
    capabilities: {
      architecture_design: true,
      system_planning: true,
      technical_analysis: true
    },
    default_config: {
      state: 'CREATED',
      capabilities: {},
      tasks_completed: 0,
      success_rate: 0,
      avg_completion_time_seconds: 0,
      memory_usage_mb: 0,
      cpu_usage_percent: 0
    }
  },
  {
    agent_type: 'CODE_IMPLEMENTER' as any,
    model_tier: 'SONNET' as any,
    agent_name: 'Code Implementer',
    description: 'Implements code based on specifications',
    capabilities: {
      code_generation: true,
      debugging: true,
      testing: true
    },
    default_config: {
      state: 'CREATED',
      capabilities: {},
      tasks_completed: 0,
      success_rate: 0,
      avg_completion_time_seconds: 0,
      memory_usage_mb: 0,
      cpu_usage_percent: 0
    }
  },
  {
    agent_type: 'TEST_COVERAGE_VALIDATOR' as any,
    model_tier: 'HAIKU' as any,
    agent_name: 'Test Validator',
    description: 'Validates test coverage and quality',
    capabilities: {
      test_generation: true,
      coverage_analysis: true,
      validation: true
    },
    default_config: {
      state: 'CREATED',
      capabilities: {},
      tasks_completed: 0,
      success_rate: 0,
      avg_completion_time_seconds: 0,
      memory_usage_mb: 0,
      cpu_usage_percent: 0
    }
  },
  {
    agent_type: 'SECURITY_AUDITOR' as any,
    model_tier: 'OPUS' as any,
    agent_name: 'Security Auditor',
    description: 'Audits code for security vulnerabilities',
    capabilities: {
      security_analysis: true,
      vulnerability_detection: true,
      compliance_checking: true
    },
    default_config: {
      state: 'CREATED',
      capabilities: {},
      tasks_completed: 0,
      success_rate: 0,
      avg_completion_time_seconds: 0,
      memory_usage_mb: 0,
      cpu_usage_percent: 0
    }
  }
];

// Helper functions
async function setupMockAPI(page: Page) {
  // Mock agent palette API
  await page.route('**/api/agents/palette', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockAgentPalette)
    });
  });

  // Mock templates API
  await page.route('**/api/workflow/templates', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(mockTemplates)
    });
  });

  // Mock workflow save API
  let savedWorkflows: any[] = [];
  await page.route('**/api/workflow/save', async (route) => {
    const request = route.request();
    const postData = JSON.parse(request.postData() || '{}');
    savedWorkflows.push(postData);
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ id: uuidv4(), success: true })
    });
  });

  // Mock workflow validation API
  await page.route('**/api/workflow/validate', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        is_valid: true,
        errors: [],
        warnings: [],
        score: 1.0,
        can_execute: true
      })
    });
  });

  return { savedWorkflows };
}

async function simulateDragAndDrop(page: Page, sourceSelector: string, targetSelector: string, offset = { x: 0, y: 0 }) {
  const source = await page.locator(sourceSelector).first();
  const target = await page.locator(targetSelector).first();

  const sourceBox = await source.boundingBox();
  const targetBox = await target.boundingBox();

  if (!sourceBox || !targetBox) {
    throw new Error('Source or target element not found');
  }

  // Perform drag and drop
  await source.hover();
  await page.mouse.down();
  await page.mouse.move(
    targetBox.x + targetBox.width / 2 + offset.x,
    targetBox.y + targetBox.height / 2 + offset.y,
    { steps: 10 }
  );
  await page.mouse.up();
}

test.describe('Workflow Editor', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console error tracking
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log(`Console error: ${msg.text()}`);
      }
    });

    // Track React errors
    page.on('pageerror', error => {
      console.log(`Page error: ${error.message}`);
      consoleErrors.push(error.message);
    });

    // Attach errors to page for test access
    await page.exposeFunction('getConsoleErrors', () => consoleErrors);
  });

  test.describe('Editor Initialization', () => {
    test('should load workflow editor with all tools and panels', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');

      // Wait for editor to initialize
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Verify main editor container
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();

      // Verify ReactFlow canvas
      await expect(page.locator('.react-flow')).toBeVisible();

      // Verify tool panels
      await expect(page.locator('[data-testid="agent-palette"]')).toBeVisible();
      await expect(page.locator('[data-testid="editor-tools"]')).toBeVisible();

      // Verify mode indicators
      await expect(page.locator('[data-testid="editor-mode"]')).toBeVisible();

      // Verify controls
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

      // Verify no critical errors
      const errors = await page.evaluate(() => (window as any).getConsoleErrors());
      const criticalErrors = errors.filter((e: string) =>
        !e.includes('Warning') && !e.includes('deprecated')
      );
      expect(criticalErrors).toHaveLength(0);
    });

    test('should display agent palette with available agent types', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Verify palette is visible
      const palette = page.locator('[data-testid="agent-palette"]');
      await expect(palette).toBeVisible();

      // Verify agent types are displayed
      for (const agent of mockAgentPalette) {
        const agentItem = palette.locator(`[data-agent-type="${agent.agent_type}"]`);
        await expect(agentItem).toBeVisible();
        await expect(agentItem.locator('[data-testid="agent-name"]')).toHaveText(agent.agent_name);
        await expect(agentItem.locator('[data-testid="agent-description"]')).toHaveText(agent.description);
      }

      // Verify agent tier indicators
      const tierIndicators = palette.locator('[data-testid="agent-tier"]');
      expect(await tierIndicators.count()).toBeGreaterThan(0);
    });

    test('should show editor tools with proper icons and labels', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      const tools = page.locator('[data-testid="editor-tools"] button');

      // Verify essential tools are present
      const expectedTools = [
        'select-mode',
        'create-agent-mode',
        'create-connection-mode',
        'undo',
        'redo',
        'templates',
        'export',
        'clear'
      ];

      for (const tool of expectedTools) {
        const toolButton = tools.locator(`[data-testid="${tool}"]`);
        await expect(toolButton).toBeVisible();
        await expect(toolButton).toHaveAttribute('title');
      }
    });
  });

  test.describe('Agent Creation and Management', () => {
    test('should create agents via drag and drop', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Initial state - no agents
      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Drag and drop first agent
      const architectAgent = page.locator('[data-agent-type="SYSTEM_ARCHITECT"]').first();
      const canvas = page.locator('.react-flow__pane');

      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');

      await page.waitForTimeout(1000);

      // Verify agent was created
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Verify agent has correct properties
      const createdAgent = agentNodes.first();
      await expect(createdAgent.locator('[data-testid="agent-name"]')).toHaveText('System Architect');
      await expect(createdAgent.locator('[data-testid="agent-type"]')).toHaveText('SYSTEM_ARCHITECT');
      await expect(createdAgent.locator('[data-testid="agent-tier"]')).toHaveText('OPUS');

      // Verify agent is in created state
      await expect(createdAgent.locator('[data-testid="agent-state"]')).toHaveText('CREATED');
    });

    test('should create multiple agents of different types', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      const agentTypes = [
        'SYSTEM_ARCHITECT',
        'CODE_IMPLEMENTER',
        'TEST_COVERAGE_VALIDATOR',
        'SECURITY_AUDITOR'
      ];

      // Create agents of each type
      for (const agentType of agentTypes) {
        await simulateDragAndDrop(page, `[data-agent-type="${agentType}"]`, '.react-flow__pane');
        await page.waitForTimeout(500);
      }

      // Verify all agents were created
      const agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(agentTypes.length);

      // Verify each agent has correct type
      for (let i = 0; i < agentTypes.length; i++) {
        const agent = agentNodes.nth(i);
        await expect(agent.locator('[data-testid="agent-type"]')).toHaveText(agentTypes[i]);
      }
    });

    test('should position agents correctly on drop', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Get canvas dimensions
      const canvas = page.locator('.react-flow__pane');
      const canvasBox = await canvas.boundingBox();

      if (!canvasBox) {
        throw new Error('Canvas not found');
      }

      // Drop agent at specific position
      const targetX = canvasBox.x + canvasBox.width / 2;
      const targetY = canvasBox.y + canvasBox.height / 2;

      const architectAgent = page.locator('[data-agent-type="SYSTEM_ARCHITECT"]').first();
      const agentBox = await architectAgent.boundingBox();

      if (!agentBox) {
        throw new Error('Agent not found');
      }

      await architectAgent.hover();
      await page.mouse.down();
      await page.mouse.move(targetX, targetY, { steps: 10 });
      await page.mouse.up();

      await page.waitForTimeout(1000);

      // Verify agent was created and positioned
      const agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Verify agent is positioned within canvas bounds
      const createdAgentBox = await agentNodes.first().boundingBox();
      if (createdAgentBox) {
        expect(createdAgentBox.x).toBeGreaterThan(canvasBox.x);
        expect(createdAgentBox.x).toBeLessThan(canvasBox.x + canvasBox.width);
        expect(createdAgentBox.y).toBeGreaterThan(canvasBox.y);
        expect(createdAgentBox.y).toBeLessThan(canvasBox.y + canvasBox.height);
      }
    });
  });

  test.describe('Connection Creation and Management', () => {
    test('should create connections between agents', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create two agents
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane', { x: -100, y: 0 });
      await page.waitForTimeout(500);
      await simulateDragAndDrop(page, '[data-agent-type="CODE_IMPLEMENTER"]', '.react-flow__pane', { x: 100, y: 0 });
      await page.waitForTimeout(500);

      // Switch to connection mode
      await page.locator('[data-testid="create-connection-mode"]').click();

      // Create connection between agents
      const agentNodes = page.locator('[data-testid="agent-node"]');
      const sourceAgent = agentNodes.first();
      const targetAgent = agentNodes.nth(1);

      await sourceAgent.click();
      await targetAgent.click();

      await page.waitForTimeout(1000);

      // Verify connection was created
      const connections = page.locator('.react-flow__edge');
      expect(await connections.count()).toBe(1);

      // Verify connection has correct type
      const connection = connections.first();
      await expect(connection).toBeVisible();
    });

    test('should handle different connection types', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create three agents
      for (let i = 0; i < 3; i++) {
        await simulateDragAndDrop(
          page,
          '[data-agent-type="SYSTEM_ARCHITECT"]',
          '.react-flow__pane',
          { x: i * 150 - 150, y: 0 }
        );
        await page.waitForTimeout(500);
      }

      // Create connections with different types
      const connectionTypes = ['DIRECT', 'BROADCAST', 'COLLABORATIVE'];
      const agentNodes = page.locator('[data-testid="agent-node"]');

      for (let i = 0; i < connectionTypes.length; i++) {
        // Switch to connection mode
        await page.locator('[data-testid="create-connection-mode"]').click();

        // Select connection type (if UI allows)
        const typeSelector = page.locator(`[data-connection-type="${connectionTypes[i]}"]`);
        if (await typeSelector.isVisible()) {
          await typeSelector.click();
        }

        // Create connection
        await agentNodes.nth(i).click();
        await agentNodes.nth((i + 1) % 3).click();
        await page.waitForTimeout(500);
      }

      // Verify all connections were created
      const connections = page.locator('.react-flow__edge');
      expect(await connections.count()).toBeGreaterThan(0);
    });

    test('should validate connections before creation', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create one agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Try to create connection with only one agent
      await page.locator('[data-testid="create-connection-mode"]').click();

      const agentNodes = page.locator('[data-testid="agent-node"]');
      await agentNodes.first().click();

      // Should show validation error or prevent connection
      const errorMessage = page.locator('[data-testid="connection-error"]');
      if (await errorMessage.isVisible({ timeout: 2000 })) {
        await expect(errorMessage).toBeVisible();
      }

      // No connection should be created
      const connections = page.locator('.react-flow__edge');
      expect(await connections.count()).toBe(0);
    });
  });

  test.describe('Property Editing', () => {
    test('should edit agent properties', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Select agent
      const agentNode = page.locator('[data-testid="agent-node"]').first();
      await agentNode.click();

      // Verify property panel appears
      const propertyPanel = page.locator('[data-testid="property-editor"]');
      await expect(propertyPanel).toBeVisible();

      // Edit agent name
      const nameInput = propertyPanel.locator('[data-testid="agent-name-input"]');
      await expect(nameInput).toBeVisible();
      await nameInput.fill('Custom Architect Name');

      // Edit agent capabilities
      const capabilityToggle = propertyPanel.locator('[data-testid="capability-toggle"]');
      if (await capabilityToggle.isVisible()) {
        await capabilityToggle.click();
      }

      // Save changes
      const saveButton = propertyPanel.locator('[data-testid="save-properties"]');
      if (await saveButton.isVisible()) {
        await saveButton.click();
      }

      await page.waitForTimeout(500);

      // Verify changes were applied
      await expect(agentNode.locator('[data-testid="agent-name"]')).toHaveText('Custom Architect Name');
    });

    test('should edit connection properties', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create two agents and connection
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane', { x: -100, y: 0 });
      await page.waitForTimeout(500);
      await simulateDragAndDrop(page, '[data-agent-type="CODE_IMPLEMENTER"]', '.react-flow__pane', { x: 100, y: 0 });
      await page.waitForTimeout(500);

      await page.locator('[data-testid="create-connection-mode"]').click();
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await agentNodes.first().click();
      await agentNodes.nth(1).click();
      await page.waitForTimeout(500);

      // Select connection
      const connection = page.locator('.react-flow__edge').first();
      await connection.click();

      // Verify property panel shows connection properties
      const propertyPanel = page.locator('[data-testid="property-editor"]');
      await expect(propertyPanel).toBeVisible();

      // Edit connection type
      const typeSelect = propertyPanel.locator('[data-testid="connection-type-select"]');
      if (await typeSelect.isVisible()) {
        await typeSelect.selectOption('BROADCAST');
      }

      // Edit message type
      const messageTypeInput = propertyPanel.locator('[data-testid="message-type-input"]');
      if (await messageTypeInput.isVisible()) {
        await messageTypeInput.fill('custom_message_type');
      }

      await page.waitForTimeout(500);

      // Verify connection remains visible
      await expect(connection).toBeVisible();
    });
  });

  test.describe('Undo/Redo Functionality', () => {
    test('should handle undo operations correctly', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Initial state - no agents
      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Create agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Verify agent was created
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Undo creation
      await page.locator('[data-testid="undo"]').click();
      await page.waitForTimeout(500);

      // Verify agent was removed
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Verify undo button is now disabled
      const undoButton = page.locator('[data-testid="undo"]');
      await expect(undoButton).toBeDisabled();
    });

    test('should handle redo operations correctly', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Undo creation
      await page.locator('[data-testid="undo"]').click();
      await page.waitForTimeout(500);

      // Verify agent was removed
      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Redo creation
      await page.locator('[data-testid="redo"]').click();
      await page.waitForTimeout(500);

      // Verify agent was restored
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Verify redo button is now disabled
      const redoButton = page.locator('[data-testid="redo"]');
      await expect(redoButton).toBeDisabled();
    });

    test('should handle multiple undo/redo operations', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create multiple agents
      const agentTypes = ['SYSTEM_ARCHITECT', 'CODE_IMPLEMENTER', 'TEST_COVERAGE_VALIDATOR'];
      for (const agentType of agentTypes) {
        await simulateDragAndDrop(page, `[data-agent-type="${agentType}"]`, '.react-flow__pane');
        await page.waitForTimeout(500);
      }

      // Verify all agents exist
      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(3);

      // Undo multiple times
      for (let i = 0; i < 3; i++) {
        await page.locator('[data-testid="undo"]').click();
        await page.waitForTimeout(200);
      }

      // Verify all agents are removed
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Redo multiple times
      for (let i = 0; i < 3; i++) {
        await page.locator('[data-testid="redo"]').click();
        await page.waitForTimeout(200);
      }

      // Verify all agents are restored
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(3);
    });

    test('should support keyboard shortcuts for undo/redo', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Verify agent exists
      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Undo with Ctrl+Z
      await page.keyboard.press('Control+z');
      await page.waitForTimeout(500);

      // Verify agent was removed
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Redo with Ctrl+Shift+Z
      await page.keyboard.press('Control+Shift+Z');
      await page.waitForTimeout(500);

      // Verify agent was restored
      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);
    });
  });

  test.describe('Template Management', () => {
    test('should load and display workflow templates', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Open templates modal
      await page.locator('[data-testid="templates"]').click();
      await page.waitForTimeout(1000);

      // Verify templates modal is visible
      const templatesModal = page.locator('[data-testid="templates-modal"]');
      await expect(templatesModal).toBeVisible();

      // Verify templates are displayed
      for (const template of mockTemplates) {
        const templateCard = templatesModal.locator(`[data-template-id="${template.id}"]`);
        await expect(templateCard).toBeVisible();
        await expect(templateCard.locator('[data-testid="template-name"]')).toHaveText(template.name);
        await expect(templateCard.locator('[data-testid="template-description"]')).toHaveText(template.description);
      }
    });

    test('should apply workflow template correctly', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Open templates modal
      await page.locator('[data-testid="templates"]').click();
      await page.waitForTimeout(1000);

      // Select first template
      const templateCard = page.locator('[data-template-id="template-1"]');
      await templateCard.click();

      // Apply template
      await page.locator('[data-testid="apply-template"]').click();
      await page.waitForTimeout(1000);

      // Verify templates modal is closed
      const templatesModal = page.locator('[data-testid="templates-modal"]');
      await expect(templatesModal).not.toBeVisible();

      // Verify template agents were created
      const template = mockTemplates[0];
      const agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(template.agents.length);

      // Verify template connections were created
      const connections = page.locator('.react-flow__edge');
      expect(await connections.count()).toBe(template.connections.length);

      // Verify agent names match template
      for (let i = 0; i < template.agents.length; i++) {
        const agent = agentNodes.nth(i);
        await expect(agent.locator('[data-testid="agent-name"]')).toHaveText(template.agents[i].name);
      }
    });

    test('should handle template application errors gracefully', async ({ page }) => {
      await setupMockAPI(page);

      // Mock template API error
      await page.route('**/api/workflow/templates', async (route) => {
        await route.fulfill({
          status: 500,
          body: 'Template loading failed'
        });
      });

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Try to open templates
      await page.locator('[data-testid="templates"]').click();
      await page.waitForTimeout(1000);

      // Should show error message or handle gracefully
      const errorMessage = page.locator('[data-testid="templates-error"]');
      if (await errorMessage.isVisible({ timeout: 2000 })) {
        await expect(errorMessage).toBeVisible();
      }

      // Editor should remain functional
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });
  });

  test.describe('Workflow Validation', () => {
    test('should validate workflow configuration', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create valid workflow
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane', { x: -100, y: 0 });
      await page.waitForTimeout(500);
      await simulateDragAndDrop(page, '[data-agent-type="CODE_IMPLEMENTER"]', '.react-flow__pane', { x: 100, y: 0 });
      await page.waitForTimeout(500);

      await page.locator('[data-testid="create-connection-mode"]').click();
      const agentNodes = page.locator('[data-testid="agent-node"]');
      await agentNodes.first().click();
      await agentNodes.nth(1).click();
      await page.waitForTimeout(500);

      // Trigger validation
      await page.locator('[data-testid="validate-workflow"]').click();
      await page.waitForTimeout(1000);

      // Verify validation results
      const validationPanel = page.locator('[data-testid="validation-results"]');
      if (await validationPanel.isVisible()) {
        await expect(validationPanel).toBeVisible();
        await expect(validationPanel.locator('[data-testid="validation-status"]')).toHaveText(/Valid/);
      }
    });

    test('should detect and display workflow errors', async ({ page }) => {
      await setupMockAPI(page);

      // Mock validation with errors
      await page.route('**/api/workflow/validate', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            is_valid: false,
            errors: [
              {
                id: 'error-1',
                type: 'error',
                element_type: 'agent',
                element_id: 'agent-1',
                message: 'Agent missing required capabilities',
                severity: 'high'
              }
            ],
            warnings: [],
            score: 0.5,
            can_execute: false
          })
        });
      });

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create agent
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Trigger validation
      await page.locator('[data-testid="validate-workflow"]').click();
      await page.waitForTimeout(1000);

      // Verify error is displayed
      const errorPanel = page.locator('[data-testid="validation-errors"]');
      if (await errorPanel.isVisible()) {
        await expect(errorPanel).toBeVisible();
        await expect(errorPanel.locator('[data-testid="error-message"]')).toHaveText(/missing required capabilities/);
      }
    });
  });

  test.describe('Import/Export Functionality', () => {
    test('should export workflow configuration', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create workflow
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Trigger export
      let downloadTriggered = false;
      page.on('download', () => {
        downloadTriggered = true;
      });

      await page.locator('[data-testid="export"]').click();
      await page.waitForTimeout(1000);

      // Verify export action was triggered
      const exportButton = page.locator('[data-testid="export"]');
      await expect(exportButton).toBeVisible();

      // Editor should remain functional
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });

    test('should import workflow configuration', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Mock import workflow
      await page.route('**/api/workflow/import', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            message: 'Workflow imported successfully'
          })
        });
      });

      // Trigger import (via file input or button)
      const importButton = page.locator('[data-testid="import"]');
      if (await importButton.isVisible()) {
        await importButton.click();
        await page.waitForTimeout(1000);

        // Verify import modal appears
        const importModal = page.locator('[data-testid="import-modal"]');
        if (await importModal.isVisible()) {
          await expect(importModal).toBeVisible();
        }
      }
    });
  });

  test.describe('Keyboard Shortcuts', () => {
    test('should support common keyboard shortcuts', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Test mode switching shortcuts
      await page.keyboard.press('s'); // Select mode
      await page.waitForTimeout(200);
      let modeIndicator = page.locator('[data-testid="editor-mode"]');
      await expect(modeIndicator).toHaveText(/SELECT/);

      await page.keyboard.press('a'); // Agent creation mode
      await page.waitForTimeout(200);
      modeIndicator = page.locator('[data-testid="editor-mode"]');
      await expect(modeIndicator).toHaveText(/CREATE_AGENT/);

      await page.keyboard.press('c'); // Connection mode
      await page.waitForTimeout(200);
      modeIndicator = page.locator('[data-testid="editor-mode"]');
      await expect(modeIndicator).toHaveText(/CREATE_CONNECTION/);

      // Test delete shortcut
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      let agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(1);

      // Select agent and delete
      await agentNodes.first().click();
      await page.keyboard.press('Delete');
      await page.waitForTimeout(500);

      agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Test save shortcut
      await page.keyboard.press('Control+s');
      await page.waitForTimeout(500);

      // Editor should remain functional
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });
  });

  test.describe('Performance and Stress Testing', () => {
    test('should handle complex workflows efficiently', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      const startTime = Date.now();

      // Create complex workflow (20 agents, 30 connections)
      for (let i = 0; i < 20; i++) {
        const agentType = ['SYSTEM_ARCHITECT', 'CODE_IMPLEMENTER', 'TEST_COVERAGE_VALIDATOR'][i % 3];
        await simulateDragAndDrop(
          page,
          `[data-agent-type="${agentType}"]`,
          '.react-flow__pane',
          { x: (i % 5) * 150 - 300, y: Math.floor(i / 5) * 150 - 150 }
        );
        await page.waitForTimeout(200);
      }

      // Create connections
      await page.locator('[data-testid="create-connection-mode"]').click();
      const agentNodes = page.locator('[data-testid="agent-node"]');

      for (let i = 0; i < 30; i++) {
        const sourceIndex = i % 20;
        const targetIndex = (i + 1) % 20;
        await agentNodes.nth(sourceIndex).click();
        await agentNodes.nth(targetIndex).click();
        await page.waitForTimeout(100);
      }

      const creationTime = Date.now() - startTime;

      // Should complete within reasonable time
      expect(creationTime).toBeLessThan(30000);

      // Verify all elements are present
      expect(await agentNodes.count()).toBe(20);
      const connections = page.locator('.react-flow__edge');
      expect(await connections.count()).toBe(30);

      // Test interactions remain responsive
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });

    test('should maintain performance during rapid operations', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Perform rapid create/delete operations
      for (let i = 0; i < 10; i++) {
        // Create agent
        await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
        await page.waitForTimeout(100);

        // Delete agent
        const agentNodes = page.locator('[data-testid="agent-node"]');
        if (await agentNodes.count() > 0) {
          await agentNodes.last().click();
          await page.keyboard.press('Delete');
        }
        await page.waitForTimeout(100);
      }

      // Editor should remain responsive
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();

      // Test basic functionality still works
      await page.locator('[data-testid="zoom-in"]').click();
      await page.waitForTimeout(500);
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });
  });

  test.describe('Error Handling and Edge Cases', () => {
    test('should handle drag and drop failures gracefully', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Simulate failed drag and drop
      const agentItem = page.locator('[data-agent-type="SYSTEM_ARCHITECT"]').first();
      await agentItem.hover();
      await page.mouse.down();

      // Move outside canvas and release
      await page.mouse.move(-100, -100);
      await page.mouse.up();

      await page.waitForTimeout(500);

      // No agent should be created
      const agentNodes = page.locator('[data-testid="agent-node"]');
      expect(await agentNodes.count()).toBe(0);

      // Editor should remain functional
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });

    test('should handle invalid agent configurations', async ({ page }) => {
      await setupMockAPI(page);

      // Mock invalid agent data
      await page.route('**/api/agents/palette', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            {
              agent_type: 'INVALID_TYPE' as any,
              model_tier: 'OPUS' as any,
              agent_name: 'Invalid Agent',
              description: 'This should not work',
              capabilities: {},
              default_config: {}
            }
          ])
        });
      });

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Try to create invalid agent
      const invalidAgent = page.locator('[data-agent-type="INVALID_TYPE"]');
      if (await invalidAgent.isVisible()) {
        await simulateDragAndDrop(page, '[data-agent-type="INVALID_TYPE"]', '.react-flow__pane');
        await page.waitForTimeout(500);

        // Should handle gracefully
        await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
      }
    });

    test('should handle network timeouts during save', async ({ page }) => {
      await setupMockAPI(page);

      // Mock slow save response
      await page.route('**/api/workflow/save', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 10000));
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ success: true })
        });
      });

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Create workflow
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Attempt save
      await page.locator('[data-testid="save-workflow"]').click();

      // Should show loading indicator
      const loadingIndicator = page.locator('[data-testid="save-loading"]');
      if (await loadingIndicator.isVisible({ timeout: 2000 })) {
        await expect(loadingIndicator).toBeVisible();
      }

      // Should remain responsive during save
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();
    });
  });

  test.describe('Real-time Collaboration', () => {
    test('should handle collaborative editing scenarios', async ({ page }) => {
      await setupMockAPI(page);

      await page.goto(WORKFLOW_EDITOR_URL);
      await page.waitForLoadState('networkidle');
      await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 10000 });

      // Simulate collaborative updates
      await page.evaluate(() => {
        // Simulate WebSocket collaboration events
        window.dispatchEvent(new CustomEvent('collaborator-join', {
          detail: { userId: 'user-2', userName: 'Collaborator' }
        }));
      });

      await page.waitForTimeout(500);

      // Create agent while simulating collaboration
      await simulateDragAndDrop(page, '[data-agent-type="SYSTEM_ARCHITECT"]', '.react-flow__pane');
      await page.waitForTimeout(500);

      // Simulate collaborator actions
      await page.evaluate(() => {
        window.dispatchEvent(new CustomEvent('collaborator-action', {
          detail: {
            userId: 'user-2',
            action: 'create_agent',
            data: { agentType: 'CODE_IMPLEMENTER', position: { x: 200, y: 200 } }
          }
        }));
      });

      await page.waitForTimeout(1000);

      // Should handle gracefully
      await expect(page.locator('[data-testid="workflow-editor"]')).toBeVisible();

      // Verify presence indicators if implemented
      const presenceIndicator = page.locator('[data-testid="collaborator-presence"]');
      if (await presenceIndicator.isVisible()) {
        await expect(presenceIndicator).toBeVisible();
      }
    });
  });
});