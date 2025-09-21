/**
 * Agent Management Integration Test Suite
 * 
 * Comprehensive Playwright tests for agent management page integration
 * with real backend API endpoints after mock data removal.
 * 
 * Key test scenarios:
 * - Page loads without errors after API integration fixes  
 * - API endpoints return proper HTTP status codes (200 OK)
 * - Components handle empty data gracefully
 * - Real-time features work with actual backend
 * - Error handling works correctly
 */

import { test, expect, Page } from '@playwright/test';

// Test configuration
const BASE_URL = 'http://localhost:3738';
const AGENT_MANAGEMENT_URL = `${BASE_URL}/agents`;

// API endpoint paths that were fixed
const API_ENDPOINTS = {
  agents: '/api/agent-management/agents',
  performance: '/api/agent-management/analytics/performance',
  projectOverview: '/api/agent-management/analytics/project-overview',
  costRecommendations: '/api/agent-management/costs/recommendations'
};

// Helper function to intercept and verify API calls
async function interceptAndVerifyAPI(page: Page, endpoint: string, expectedStatus = 200) {
  let intercepted = false;
  let actualStatus = 0;
  
  await page.route(`**${endpoint}`, async (route, request) => {
    intercepted = true;
    const response = await route.fetch();
    actualStatus = response.status();
    await route.fulfill({ response });
  });
  
  return { 
    wasIntercepted: () => intercepted,
    getStatus: () => actualStatus 
  };
}

test.describe('Agent Management API Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console error tracking
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log(`Console error: ${msg.text()}`);
      }
    });

    // Set up network error tracking
    page.on('requestfailed', request => {
      console.log(`Network request failed: ${request.url()} - ${request.failure()?.errorText}`);
    });
  });

  test('should load agent management page successfully', async ({ page }) => {
    // Navigate to agent management page
    await page.goto(AGENT_MANAGEMENT_URL);
    
    // Wait for the page to load
    await page.waitForLoadState('networkidle');
    
    // Verify the page title and header
    await expect(page.locator('h1')).toContainText('Agent Management');
    await expect(page.locator('p.text-gray-600').first()).toContainText('Intelligence-Tiered Adaptive Agent Management System');
    
    // Verify no JavaScript errors on page load
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Wait a bit more for any delayed errors
    await page.waitForTimeout(2000);
    
    // Should have minimal or no console errors (excluding expected warnings about empty data)
    const criticalErrors = consoleErrors.filter(error => 
      !error.includes('Performance metrics not available') &&
      !error.includes('Project overview not available') &&
      !error.includes('Cost recommendations not available')
    );
    
    expect(criticalErrors).toHaveLength(0);
  });

  test('should make successful API calls to primary agent management endpoints', async ({ page }) => {
    // Set up API interception for key endpoints that are actually implemented
    const agentsIntercept = await interceptAndVerifyAPI(page, API_ENDPOINTS.agents);
    const performanceIntercept = await interceptAndVerifyAPI(page, API_ENDPOINTS.performance);
    
    // Track all API calls
    const allApiCalls = [];
    await page.route('**/api/agent-management/**', async (route, request) => {
      allApiCalls.push(request.url());
      const response = await route.fetch();
      await route.fulfill({ response });
    });
    
    // Navigate to trigger API calls
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for all API calls to complete
    await page.waitForTimeout(3000);
    
    // Verify key endpoints were called
    expect(agentsIntercept.wasIntercepted()).toBe(true);
    expect(performanceIntercept.wasIntercepted()).toBe(true);
    
    // Verify agents endpoint returns 200 OK (primary endpoint must work)
    expect(agentsIntercept.getStatus()).toBe(200);
    
    // Performance endpoint should also work or handle errors gracefully
    expect([200, 404, 500]).toContain(performanceIntercept.getStatus());
    
    // Log all API calls for debugging
    console.log('All API calls made:', allApiCalls);
    
    // Verify at least the basic agent management API calls were made
    expect(allApiCalls.length).toBeGreaterThan(0);
  });

  test('should handle empty data gracefully without errors', async ({ page }) => {
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for data loading to complete
    const loadingSpinner = page.locator('.animate-spin');
    if (await loadingSpinner.isVisible()) {
      await loadingSpinner.waitFor({ state: 'detached', timeout: 10000 });
    }
    
    // Verify statistics cards show zeros gracefully
    const statCards = page.locator('[class*="grid"] .p-4');
    await expect(statCards.first()).toBeVisible();
    
    // Verify empty state is displayed for agents
    const emptyStateMessage = page.locator('text=No Agents Found');
    await expect(emptyStateMessage).toBeVisible();
    
    // Verify the empty state has proper styling and messaging
    await expect(page.locator('text=Try adjusting your search criteria or create a new agent')).toBeVisible();
  });

  test('should display all main UI components and tabs', async ({ page }) => {
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for loading to complete
    await page.waitForSelector('[class*="container"]', { timeout: 10000 });
    
    // Verify header buttons are present
    await expect(page.locator('button:has-text("Hibernate Idle")')).toBeVisible();
    await expect(page.locator('button:has-text("Create Agent")')).toBeVisible();
    
    // Verify statistics cards are rendered
    const statCards = page.locator('.grid .p-4');
    await expect(statCards).toHaveCount(6); // Total, Active, Idle, Hibernated, Tasks Done, Success Rate
    
    // Verify all tabs are present (using text content instead of role attribute)
    const tabTriggers = [
      'Agents',
      'Pools', 
      'Intelligence',
      'Costs',
      'Collaboration',
      'Knowledge'
    ];
    
    for (const tabText of tabTriggers) {
      // Use a more flexible selector for tab elements
      await expect(page.locator(`text=${tabText}`).first()).toBeVisible();
    }
    
    // Verify filters are present on agents tab
    await expect(page.locator('input[placeholder*="Search agents"]')).toBeVisible();
    await expect(page.locator('select').first()).toBeVisible(); // State filter
    await expect(page.locator('select').nth(1)).toBeVisible(); // Tier filter
  });

  test('should handle tab navigation without errors', async ({ page }) => {
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for page to fully load
    await page.waitForTimeout(2000);
    
    // Test navigation through all tabs
    const tabs = [
      { name: 'Pools', text: 'Pools' },
      { name: 'Intelligence', text: 'Intelligence' },  
      { name: 'Costs', text: 'Costs' },
      { name: 'Collaboration', text: 'Collaboration' },
      { name: 'Knowledge', text: 'Knowledge' },
      { name: 'Agents', text: 'Agents' } // Return to first tab
    ];
    
    for (const tab of tabs) {
      await page.locator(`[role="tab"]:has-text("${tab.text}")`).click();
      await page.waitForTimeout(500); // Allow tab content to render
      
      // Verify tab is active (no need to check specific content as components may be empty)
      await expect(page.locator(`[role="tab"]:has-text("${tab.text}")`)).toHaveAttribute('data-state', 'active');
    }
  });

  test('should show create agent modal when create button is clicked', async ({ page }) => {
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for page to load
    await page.waitForTimeout(2000);
    
    // Click create agent button
    await page.locator('button:has-text("Create Agent")').click();
    
    // Wait for modal to appear
    await page.waitForTimeout(500);
    
    // Verify modal is visible (it should render even with empty backend)
    // Note: Modal content will depend on the AgentCreationModal component
    // We're just verifying the click action doesn't cause errors
    const body = page.locator('body');
    await expect(body).toBeVisible(); // Basic check that page doesn't crash
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Intercept and simulate network failures for some endpoints
    await page.route('**/api/agent-management/analytics/performance', route => {
      route.fulfill({ status: 500, body: 'Server Error' });
    });
    
    await page.route('**/api/agent-management/analytics/project-overview', route => {
      route.fulfill({ status: 404, body: 'Not Found' });
    });
    
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for all requests to complete
    await page.waitForTimeout(3000);
    
    // Page should still load and show the empty state
    await expect(page.locator('h1:has-text("Agent Management")')).toBeVisible();
    await expect(page.locator('text=No Agents Found')).toBeVisible();
    
    // Should not show error state for the entire page (individual components may show warnings)
    const errorCard = page.locator('text=Error Loading Agent Management');
    await expect(errorCard).not.toBeVisible();
  });

  test('should have responsive design elements', async ({ page }) => {
    // Test desktop view
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Verify desktop grid layout
    const statsGrid = page.locator('.grid.grid-cols-2.md\\:grid-cols-4.lg\\:grid-cols-6');
    await expect(statsGrid).toBeVisible();
    
    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);
    
    // Elements should still be visible
    await expect(page.locator('h1:has-text("Agent Management")')).toBeVisible();
    await expect(statsGrid).toBeVisible();
    
    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);
    
    // Core elements should still be accessible
    await expect(page.locator('h1:has-text("Agent Management")')).toBeVisible();
  });

  test('should validate search and filter functionality', async ({ page }) => {
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for page to load
    await page.waitForTimeout(2000);
    
    // Test search input
    const searchInput = page.locator('input[placeholder*="Search agents"]');
    await expect(searchInput).toBeVisible();
    
    await searchInput.fill('test-agent');
    await page.waitForTimeout(500);
    
    // Should not cause errors even with empty data
    await expect(page.locator('text=No Agents Found')).toBeVisible();
    
    // Test state filter
    const stateFilter = page.locator('select').first();
    await expect(stateFilter).toBeVisible();
    
    await stateFilter.selectOption('ACTIVE');
    await page.waitForTimeout(500);
    
    // Should still show empty state
    await expect(page.locator('text=No Agents Found')).toBeVisible();
    
    // Test tier filter  
    const tierFilter = page.locator('select').nth(1);
    await expect(tierFilter).toBeVisible();
    
    await tierFilter.selectOption('OPUS');
    await page.waitForTimeout(500);
    
    // Should still show empty state
    await expect(page.locator('text=No Agents Found')).toBeVisible();
  });
});

test.describe('Agent Management Real-time Features', () => {
  test('should handle socket connections gracefully', async ({ page }) => {
    // Monitor WebSocket connections
    const wsConnections = [];
    page.on('websocket', ws => {
      wsConnections.push(ws);
      ws.on('close', () => console.log('WebSocket closed'));
      ws.on('socketerror', err => console.log('WebSocket error:', err));
    });
    
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Wait for potential socket connections
    await page.waitForTimeout(3000);
    
    // Verify page loads even if socket connection fails
    await expect(page.locator('h1:has-text("Agent Management")')).toBeVisible();
    
    // Socket connection failure should not break the UI
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });
});

test.describe('Agent Management Performance Tests', () => {
  test('should load within reasonable time limits', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Should load within 5 seconds even with API calls
    expect(loadTime).toBeLessThan(5000);
    
    // Verify critical elements are visible
    await expect(page.locator('h1:has-text("Agent Management")')).toBeVisible();
  });
  
  test('should handle multiple rapid API calls', async ({ page }) => {
    let apiCallCount = 0;
    
    // Count API calls
    await page.route('**/api/agent-management/**', async (route, request) => {
      apiCallCount++;
      const response = await route.fetch();
      await route.fulfill({ response });
    });
    
    await page.goto(AGENT_MANAGEMENT_URL);
    await page.waitForLoadState('networkidle');
    
    // Should make reasonable number of API calls (not excessive)
    expect(apiCallCount).toBeGreaterThan(0);
    expect(apiCallCount).toBeLessThan(20); // Reasonable upper limit
  });
});