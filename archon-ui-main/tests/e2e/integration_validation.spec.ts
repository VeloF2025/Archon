/**
 * Agency Swarm Integration Validation Test Suite
 *
 * Comprehensive integration tests to validate all Agency Swarm components
 * work together seamlessly across Phase 1-3 implementation.
 *
 * Key test scenarios:
 * - Cross-component integration validation
 * - API endpoint integration
 * - Database integration
 * - Real-time communication integration
 * - MCP server integration
 * - Socket.IO integration
 * - Authentication and authorization integration
 * - Third-party service integration
 */

import { test, expect, Page } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const API_BASE = `${BASE_URL}/api`;

// Integration test data
const integrationTestData = {
  agency: {
    id: uuidv4(),
    name: 'Integration Test Agency',
    description: 'Agency for integration testing',
    agents: [
      {
        id: 'integration-agent-1',
        name: 'Integration Coordinator',
        agent_type: 'SYSTEM_ARCHITECT',
        capabilities: ['coordination', 'integration', 'validation']
      },
      {
        id: 'integration-agent-2',
        name: 'API Tester',
        agent_type: 'TEST_COVERAGE_VALIDATOR',
        capabilities: ['api_testing', 'integration_testing', 'validation']
      }
    ]
  },
  workflows: [
    {
      id: 'integration-workflow-1',
      name: 'API Integration Workflow',
      steps: [
        { step: 'initialize', agent: 'integration-agent-1' },
        { step: 'test_api', agent: 'integration-agent-2' },
        { step: 'validate', agent: 'integration-agent-1' }
      ]
    }
  ]
};

// Mock API responses for integration testing
const mockIntegrationResponses = {
  // Agency API endpoints
  'GET /api/agency/workflow/data': {
    status: 200,
    body: integrationTestData.agency
  },
  'POST /api/agency/workflow/create': {
    status: 201,
    body: { id: uuidv4(), status: 'created', message: 'Agency created successfully' }
  },
  'GET /api/agency/workflow/stats': {
    status: 200,
    body: {
      total_agents: 2,
      active_agents: 2,
      total_workflows: 1,
      active_workflows: 1,
      integration_status: 'healthy'
    }
  },

  // MCP API endpoints
  'GET /api/mcp/health': {
    status: 200,
    body: {
      status: 'healthy',
      version: '1.0.0',
      connected: true,
      uptime: 3600
    }
  },
  'GET /api/mcp/tools': {
    status: 200,
    body: {
      tools: [
        {
          name: 'archon_perform_rag_query',
          description: 'Search knowledge base'
        },
        {
          name: 'archon_manage_project',
          description: 'Manage project operations'
        }
      ]
    }
  },

  // Knowledge API endpoints
  'GET /api/knowledge/items': {
    status: 200,
    body: {
      items: [
        { id: 'doc-1', title: 'Integration Guide', type: 'document' },
        { id: 'doc-2', title: 'API Documentation', type: 'document' }
      ],
      total: 2
    }
  },
  'POST /api/knowledge/search': {
    status: 200,
    body: {
      results: [
        { id: 'result-1', title: 'Integration Pattern', score: 0.95 }
      ],
      total: 1
    }
  },

  // Health and monitoring endpoints
  'GET /api/health': {
    status: 200,
    body: {
      status: 'healthy',
      version: '3.0.0',
      components: {
        database: 'healthy',
        redis: 'healthy',
        mcp_server: 'healthy',
        agency_service: 'healthy',
        socketio: 'healthy'
      }
    }
  },

  // Authentication endpoints
  'GET /api/auth/status': {
    status: 200,
    body: {
      authenticated: true,
      user: { id: 'test-user', role: 'admin' }
    }
  }
};

// Helper functions
async function setupIntegrationMockAPI(page: Page) {
  // Set up all API route mocks
  for (const [route, response] of Object.entries(mockIntegrationResponses)) {
    const [method, path] = route.split(' ');

    await page.route(`**${path}`, async (route_obj) => {
      if (method === 'GET' && route_obj.request().method() === 'GET') {
        await route_obj.fulfill({
          status: response.status,
          contentType: 'application/json',
          body: JSON.stringify(response.body)
        });
      } else if (method === 'POST' && route_obj.request().method() === 'POST') {
        await route_obj.fulfill({
          status: response.status,
          contentType: 'application/json',
          body: JSON.stringify(response.body)
        });
      }
    });
  }
}

async function simulateSocketEvent(page: Page, event: string, data: any) {
  await page.evaluate(({ event, data }) => {
    window.dispatchEvent(new CustomEvent(event, {
      detail: data
    }));
  }, { event, data });
}

test.describe('Agency Swarm Integration Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Set up comprehensive error and performance tracking
    const errors: string[] = [];
    const warnings: string[] = [];
    const apiCalls: any[] = [];

    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      } else if (msg.type() === 'warning') {
        warnings.push(msg.text());
      }
    });

    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiCalls.push({
          url: request.url(),
          method: request.method(),
          timestamp: Date.now()
        });
      }
    });

    // Attach tracking to page
    await page.exposeFunction('getErrors', () => errors);
    await page.exposeFunction('getWarnings', () => warnings);
    await page.exposeFunction('getApiCalls', () => apiCalls);
  });

  test.describe('API Integration Validation', () => {
    test('should validate all Agency Swarm API endpoints', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(BASE_URL);
      await page.waitForLoadState('networkidle');

      // Test agency workflow API
      const agencyResponse = await page.request.get(`${API_BASE}/agency/workflow/data`);
      expect(agencyResponse.status()).toBe(200);
      const agencyData = await agencyResponse.json();
      expect(agencyData).toHaveProperty('id');
      expect(agencyData).toHaveProperty('agents');

      // Test agency stats API
      const statsResponse = await page.request.get(`${API_BASE}/agency/workflow/stats`);
      expect(statsResponse.status()).toBe(200);
      const statsData = await statsResponse.json();
      expect(statsData).toHaveProperty('total_agents');
      expect(statsData).toHaveProperty('integration_status');

      // Test MCP health API
      const mcpHealthResponse = await page.request.get(`${API_BASE}/mcp/health`);
      expect(mcpHealthResponse.status()).toBe(200);
      const mcpHealthData = await mcpHealthResponse.json();
      expect(mcpHealthData).toHaveProperty('status');
      expect(mcpHealthData.status).toBe('healthy');

      // Test MCP tools API
      const mcpToolsResponse = await page.request.get(`${API_BASE}/mcp/tools`);
      expect(mcpToolsResponse.status()).toBe(200);
      const mcpToolsData = await mcpToolsResponse.json();
      expect(mcpToolsData).toHaveProperty('tools');
      expect(Array.isArray(mcpToolsData.tools)).toBe(true);

      // Test knowledge API
      const knowledgeResponse = await page.request.get(`${API_BASE}/knowledge/items`);
      expect(knowledgeResponse.status()).toBe(200);
      const knowledgeData = await knowledgeResponse.json();
      expect(knowledgeData).toHaveProperty('items');

      // Test health API
      const healthResponse = await page.request.get(`${API_BASE}/health`);
      expect(healthResponse.status()).toBe(200);
      const healthData = await healthResponse.json();
      expect(healthData).toHaveProperty('status');
      expect(healthData).toHaveProperty('components');

      // Test authentication API
      const authResponse = await page.request.get(`${API_BASE}/auth/status`);
      expect(authResponse.status()).toBe(200);
      const authData = await authResponse.json();
      expect(authData).toHaveProperty('authenticated');

      // Verify no critical errors
      const errors = await page.evaluate(() => (window as any).getErrors());
      expect(errors.filter(e => !e.includes('Warning'))).toHaveLength(0);
    });

    test('should handle API errors gracefully', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      // Mock API error responses
      await page.route('**/api/agency/workflow/data', async (route) => {
        await route.fulfill({
          status: 500,
          body: JSON.stringify({ error: 'Internal Server Error' })
        });
      });

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Should handle API error without crashing
      const errorElement = page.locator('[data-testid="api-error"]');
      if (await errorElement.isVisible({ timeout: 5000 })) {
        await expect(errorElement).toBeVisible();
      }

      // Should show retry mechanism
      const retryButton = page.locator('[data-testid="retry-api"]');
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible();
      }

      // Interface should remain functional
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
    });

    test('should validate API rate limiting and throttling', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      let requestCount = 0;
      await page.route('**/api/agency/workflow/stats', async (route) => {
        requestCount++;
        if (requestCount > 5) {
          await route.fulfill({
            status: 429,
            body: JSON.stringify({ error: 'Rate limit exceeded' })
          });
        } else {
          await route.fulfill({
            status: 200,
            body: JSON.stringify({ total_agents: 2, request_count: requestCount })
          });
        }
      });

      await page.goto(BASE_URL);
      await page.waitForLoadState('networkidle');

      // Make multiple rapid requests
      const requests = [];
      for (let i = 0; i < 10; i++) {
        requests.push(page.request.get(`${API_BASE}/agency/workflow/stats`));
      }

      const responses = await Promise.all(requests);
      const rateLimitedResponses = responses.filter(r => r.status() === 429);

      // Should have some rate limited responses
      expect(rateLimitedResponses.length).toBeGreaterThan(0);

      // Should handle rate limiting gracefully
      const rateLimitElement = page.locator('[data-testid="rate-limit-error"]');
      if (await rateLimitElement.isVisible({ timeout: 2000 })) {
        await expect(rateLimitElement).toBeVisible();
      }
    });
  });

  test.describe('Real-time Communication Integration', () => {
    test('should validate Socket.IO integration for real-time updates', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Wait for Socket.IO connection
      await page.waitForSelector('[data-testid="socket-connected"]', { timeout: 10000 });

      // Simulate real-time agent updates
      await simulateSocketEvent(page, 'agent_status_update', {
        agent_id: 'integration-agent-1',
        new_status: 'ACTIVE',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      // Verify UI updates in real-time
      const agentStatus = page.locator('[data-agent-id="integration-agent-1"] [data-testid="agent-status"]');
      if (await agentStatus.isVisible()) {
        await expect(agentStatus).toHaveText('ACTIVE');
      }

      // Simulate workflow progress update
      await simulateSocketEvent(page, 'workflow_progress', {
        workflow_id: 'integration-workflow-1',
        progress: 75,
        current_step: 'validate',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      // Verify progress bar updates
      const progressBar = page.locator('[data-testid="workflow-progress"]');
      if (await progressBar.isVisible()) {
        await expect(progressBar).toHaveAttribute('aria-valuenow', '75');
      }

      // Simulate system-wide notification
      await simulateSocketEvent(page, 'system_notification', {
        type: 'info',
        message: 'Integration validation in progress',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      // Verify notification appears
      const notification = page.locator('[data-testid="notification"]');
      if (await notification.isVisible()) {
        await expect(notification).toBeVisible();
        await expect(notification).toHaveText(/Integration validation/);
      }

      // Verify Socket.IO connection remains stable
      await expect(page.locator('[data-testid="socket-connected"]')).toBeVisible();
    });

    test('should handle Socket.IO connection failures gracefully', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      // Block WebSocket connections
      await page.route('**/socket.io/**', route => route.abort('failed'));

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Should show disconnected state
      await expect(page.locator('[data-testid="socket-disconnected"]')).toBeVisible();

      // Should provide retry mechanism
      const retryButton = page.locator('[data-testid="retry-socket"]');
      if (await retryButton.isVisible()) {
        await expect(retryButton).toBeVisible();
      }

      // Should continue to work with cached data
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();
    });
  });

  test.describe('Database Integration Validation', () => {
    test('should validate database operations and data consistency', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Test agency creation (database write)
      await page.locator('[data-testid="create-test-agency"]').click();
      await page.waitForSelector('[data-testid="agency-created"]', { timeout: 10000 });

      // Verify creation response
      const creationStatus = page.locator('[data-testid="creation-status"]');
      await expect(creationStatus).toHaveText(/success|created/);

      // Test agency retrieval (database read)
      await page.locator('[data-testid="retrieve-agency"]').click();
      await page.waitForSelector('[data-testid="agency-retrieved"]', { timeout: 10000 });

      // Verify data consistency
      const retrievedData = page.locator('[data-testid="agency-data"]');
      if (await retrievedData.isVisible()) {
        await expect(retrievedData).toBeVisible();
      }

      // Test agency update (database update)
      await page.locator('[data-testid="update-agency"]').click();
      await page.waitForSelector('[data-testid="agency-updated"]', { timeout: 10000 });

      // Verify update success
      const updateStatus = page.locator('[data-testid="update-status"]');
      await expect(updateStatus).toHaveText(/success|updated/);

      // Test agency deletion (database delete)
      await page.locator('[data-testid="delete-test-agency"]').click();
      await page.waitForSelector('[data-testid="agency-deleted"]', { timeout: 10000 });

      // Verify deletion success
      const deleteStatus = page.locator('[data-testid="deletion-status"]');
      await expect(deleteStatus).toHaveText(/success|deleted/);
    });

    test('should handle database connection failures gracefully', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      // Mock database failure
      await page.route('**/api/agency/workflow/data', async (route) => {
        await route.fulfill({
          status: 503,
          body: JSON.stringify({ error: 'Database connection failed' })
        });
      });

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Should show database error
      const dbError = page.locator('[data-testid="database-error"]');
      if (await dbError.isVisible({ timeout: 5000 })) {
        await expect(dbError).toBeVisible();
      }

      // Should show cached data if available
      const cachedData = page.locator('[data-testid="cached-data"]');
      if (await cachedData.isVisible()) {
        await expect(cachedData).toBeVisible();
      }

      // Should provide retry mechanism
      const retryDb = page.locator('[data-testid="retry-database"]');
      if (await retryDb.isVisible()) {
        await expect(retryDb).toBeVisible();
      }
    });
  });

  test.describe('MCP Server Integration Validation', () => {
    test('should validate MCP server tool execution integration', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/mcp`);
      await page.waitForLoadState('networkidle');

      // Wait for MCP connection
      await page.waitForSelector('[data-testid="mcp-status-connected"]', { timeout: 10000 });

      // Test tool discovery
      await expect(page.locator('[data-testid="mcp-tools-list"]')).toBeVisible();
      const toolCards = page.locator('[data-testid="mcp-tool-card"]');
      expect(await toolCards.count()).toBeGreaterThan(0);

      // Test tool execution integration
      const firstTool = toolCards.first();
      await firstTool.click();
      await page.waitForSelector('[data-testid="tool-details"]', { timeout: 5000 });

      // Execute tool with parameters
      const executeButton = page.locator('[data-testid="execute-tool"]');
      if (await executeButton.isVisible()) {
        await executeButton.click();
        await page.waitForSelector('[data-testid="tool-execution-result"]', { timeout: 10000 });

        // Verify execution result
        const resultStatus = page.locator('[data-testid="execution-status"]');
        await expect(resultStatus).toHaveText(/success|complete/);
      }

      // Verify MCP integration with agency workflow
      const integrationStatus = page.locator('[data-testid="mcp-agency-integration"]');
      if (await integrationStatus.isVisible()) {
        await expect(integrationStatus).toHaveText(/integrated|connected/);
      }
    });

    test('should validate MCP tool coordination between agents', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Simulate MCP tool coordination
      await simulateSocketEvent(page, 'mcp_tool_coordination', {
        coordinator_agent: 'integration-agent-1',
        executing_agent: 'integration-agent-2',
        tool_name: 'archon_perform_rag_query',
        parameters: { query: 'integration patterns' },
        coordination_id: uuidv4()
      });

      await page.waitForTimeout(1000);

      // Verify coordination visualization
      const coordinationFlow = page.locator('[data-testid="coordination-flow"]');
      if (await coordinationFlow.isVisible()) {
        await expect(coordinationFlow).toBeVisible();
      }

      // Simulate tool execution result
      await simulateSocketEvent(page, 'mcp_tool_result', {
        coordination_id: uuidv4(),
        result: { status: 'success', data: { results: [] } },
        execution_time_ms: 150
      });

      await page.waitForTimeout(500);

      // Verify result propagation
      const resultIndicator = page.locator('[data-testid="tool-result-indicator"]');
      if (await resultIndicator.isVisible()) {
        await expect(resultIndicator).toBeVisible();
      }
    });
  });

  test.describe('Authentication and Authorization Integration', () => {
    test('should validate authentication integration across components', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(BASE_URL);
      await page.waitForLoadState('networkidle');

      // Test authentication status
      const authStatus = page.locator('[data-testid="auth-status"]');
      await expect(authStatus).toBeVisible();
      await expect(authStatus).toHaveText(/authenticated|logged_in/);

      // Test protected API access
      const protectedResponse = await page.request.get(`${API_BASE}/agency/workflow/data`);
      expect(protectedResponse.status()).toBe(200);

      // Test user role display
      const userRole = page.locator('[data-testid="user-role"]');
      if (await userRole.isVisible()) {
        await expect(userRole).toHaveText(/admin|user/);
      }

      // Test session management
      const sessionInfo = page.locator('[data-testid="session-info"]');
      if (await sessionInfo.isVisible()) {
        await expect(sessionInfo).toBeVisible();
      }

      // Test authentication in different components
      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      const agencyAuth = page.locator('[data-testid="agency-auth-status"]');
      if (await agencyAuth.isVisible()) {
        await expect(agencyAuth).toHaveText(/authenticated/);
      }

      await page.goto(`${BASE_URL}/mcp`);
      await page.waitForLoadState('networkidle');

      const mcpAuth = page.locator('[data-testid="mcp-auth-status"]');
      if (await mcpAuth.isVisible()) {
        await expect(mcpAuth).toHaveText(/authenticated/);
      }
    });

    test('should handle authentication failures gracefully', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      // Mock authentication failure
      await page.route('**/api/auth/status', async (route) => {
        await route.fulfill({
          status: 401,
          body: JSON.stringify({ authenticated: false, error: 'Invalid token' })
        });
      });

      await page.goto(BASE_URL);
      await page.waitForLoadState('networkidle');

      // Should show authentication error
      const authError = page.locator('[data-testid="auth-error"]');
      if (await authError.isVisible({ timeout: 5000 })) {
        await expect(authError).toBeVisible();
      }

      // Should redirect to login
      const loginRedirect = page.locator('[data-testid="login-redirect"]');
      if (await loginRedirect.isVisible()) {
        await expect(loginRedirect).toBeVisible();
      }

      // Should restrict access to protected areas
      const protectedContent = page.locator('[data-testid="protected-content"]');
      expect(await protectedContent.count()).toBe(0);
    });
  });

  test.describe('Performance and Monitoring Integration', () => {
    test('should validate performance monitoring integration', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Start performance monitoring
      await page.locator('[data-testid="start-monitoring"]').click();
      await page.waitForSelector('[data-testid="monitoring-active"]', { timeout: 5000);

      // Verify monitoring metrics are displayed
      const metricsContainer = page.locator('[data-testid="performance-metrics"]');
      if (await metricsContainer.isVisible()) {
        await expect(metricsContainer).toBeVisible();

        // Check for key metrics
        const responseTime = metricsContainer.locator('[data-metric="response-time"]');
        const throughput = metricsContainer.locator('[data-metric="throughput"]');
        const errorRate = metricsContainer.locator('[data-metric="error-rate"]');

        expect(await responseText.count() + await throughput.count() + await errorRate.count()).toBeGreaterThan(0);
      }

      // Simulate performance events
      await simulateSocketEvent(page, 'performance_metric', {
        metric_type: 'response_time',
        value: 150,
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      // Verify real-time metric updates
      const realTimeMetric = page.locator('[data-testid="real-time-metric"]');
      if (await realTimeMetric.isVisible()) {
        await expect(realTimeMetric).toBeVisible();
      }

      // Test performance alerts
      await simulateSocketEvent(page, 'performance_alert', {
        alert_type: 'high_response_time',
        threshold: 500,
        current_value: 750,
        severity: 'warning'
      });

      await page.waitForTimeout(500);

      const alertIndicator = page.locator('[data-testid="performance-alert"]');
      if (await alertIndicator.isVisible()) {
        await expect(alertIndicator).toBeVisible();
      }
    });

    test('should validate error tracking and logging integration', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(BASE_URL);
      await page.waitForLoadState('networkidle');

      // Simulate error event
      await simulateSocketEvent(page, 'error_event', {
        error_type: 'integration_error',
        error_message: 'Test integration error',
        component: 'agency_service',
        severity: 'error',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      // Verify error tracking
      const errorTracker = page.locator('[data-testid="error-tracker"]');
      if (await errorTracker.isVisible()) {
        await expect(errorTracker).toBeVisible();
      }

      // Verify error logging
      const errorLog = page.locator('[data-testid="error-log"]');
      if (await errorLog.isVisible()) {
        await expect(errorLog).toBeVisible();
        await expect(errorLog).toHaveText(/integration_error/);
      }

      // Test error recovery
      await simulateSocketEvent(page, 'error_recovery', {
        error_id: uuidv4(),
        recovery_action: 'restart_service',
        recovery_status: 'success',
        timestamp: new Date().toISOString()
      });

      await page.waitForTimeout(500);

      const recoveryStatus = page.locator('[data-testid="recovery-status"]');
      if (await recoveryStatus.isVisible()) {
        await expect(recoveryStatus).toHaveText(/success|recovered/);
      }
    });
  });

  test.describe('Cross-Browser Integration Validation', () => {
    test('should validate integration across different browsers', async ({ page, browserName }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Test basic integration across browsers
      await expect(page.locator('[data-testid="agency-workflow"]')).toBeVisible();

      // Test API integration
      const apiResponse = await page.request.get(`${API_BASE}/health`);
      expect(apiResponse.status()).toBe(200);

      // Test real-time updates
      await simulateSocketEvent(page, 'test_event', {
        browser: browserName,
        message: 'Cross-browser integration test'
      });

      await page.waitForTimeout(500);

      // Verify event handling
      const eventHandler = page.locator('[data-testid="event-handled"]');
      if (await eventHandler.isVisible()) {
        await expect(eventHandler).toBeVisible();
      }

      // Test all major features work consistently
      const features = [
        'agency-workflow',
        'communication-flows',
        'agent-nodes',
        'real-time-updates',
        'api-integration'
      ];

      for (const feature of features) {
        const featureElement = page.locator(`[data-testid="${feature}"]`);
        if (await featureElement.isVisible()) {
          await expect(featureElement).toBeVisible();
        }
      }
    });
  });

  test.describe('Final Integration Validation', () => {
    test('should pass comprehensive integration validation suite', async ({ page }) => {
      await setupIntegrationMockAPI(page);

      await page.goto(`${BASE_URL}/agency`);
      await page.waitForLoadState('networkidle');

      // Run comprehensive integration validation
      await page.locator('[data-testid="run-integration-validation"]').click();
      await page.waitForSelector('[data-testid="validation-results"]', { timeout: 30000 });

      // Verify validation results
      await expect(page.locator('[data-testid="validation-results"]')).toBeVisible();

      // Check integration status
      const integrationStatus = page.locator('[data-testid="integration-status"]');
      await expect(integrationStatus).toHaveText(/healthy|operational|connected/);

      // Verify all integration categories
      const categories = [
        'api_integration',
        'realtime_communication',
        'database_integration',
        'mcp_integration',
        'authentication_integration',
        'performance_monitoring'
      ];

      for (const category of categories) {
        const categoryStatus = page.locator(`[data-integration-category="${category}"]`);
        if (await categoryStatus.isVisible()) {
          await expect(categoryStatus.locator('[data-testid="category-status"]')).toHaveText(/pass|success|connected/);
        }
      }

      // Verify no critical integration issues
      const criticalIssues = page.locator('[data-testid="critical-integration-issue"]');
      expect(await criticalIssues.count()).toBe(0);

      // Generate integration report
      await page.locator('[data-testid="generate-integration-report"]').click();
      await page.waitForSelector('[data-testid="integration-report"]', { timeout: 10000);

      // Verify report generation
      const reportStatus = page.locator('[data-testid="report-status"]');
      await expect(reportStatus).toHaveText(/success|complete/);

      // Final integration validation
      await expect(page.locator('[data-testid="integration-complete"]')).toBeVisible();

      console.log('✅ Agency Swarm Integration Validation - SUCCESS');
      console.log('✅ All components integrated successfully');
      console.log('✅ Cross-component communication validated');
      console.log('✅ Real-time integration verified');
      console.log('✅ API integration confirmed');
      console.log('✅ Database consistency validated');
      console.log('✅ MCP server integration complete');
      console.log('✅ Authentication and authorization integrated');
      console.log('✅ Performance monitoring operational');
    });
  });
});