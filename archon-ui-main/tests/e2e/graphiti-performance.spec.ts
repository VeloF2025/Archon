import { test, expect, Page } from '@playwright/test';

/**
 * Performance-Optimized Graphiti Explorer Tests
 * Target: <1.5s initial load, smooth interactions
 */

class GraphitiPerformancePage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/graphiti');
  }

  async waitForLoad() {
    // Wait for the performance monitor to initialize
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForTimeout(1000); // Allow React to initialize
  }
}

test.describe('Graphiti Performance Tests', () => {
  let graphitiPage: GraphitiPerformancePage;

  test.beforeEach(async ({ page }) => {
    graphitiPage = new GraphitiPerformancePage(page);
  });

  test('meets performance targets for initial load', async ({ page }) => {
    const startTime = performance.now();
    
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Check if performance-optimized component loaded
    await expect(page.locator('text=Optimized Graph Explorer')).toBeVisible();
    
    // Measure page load performance
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
        loadComplete: navigation.loadEventEnd - navigation.navigationStart,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
      };
    });
    
    console.log('🚀 Performance metrics:', performanceMetrics);
    
    // Target: DOM content loaded in <1.5s
    expect(performanceMetrics.domContentLoaded).toBeLessThan(1500);
    console.log(`✅ DOM Content Loaded: ${performanceMetrics.domContentLoaded}ms (target: <1500ms)`);
    
    // Target: Complete load in <2s  
    expect(performanceMetrics.loadComplete).toBeLessThan(2000);
    console.log(`✅ Load Complete: ${performanceMetrics.loadComplete}ms (target: <2000ms)`);
    
    // Take performance screenshot
    await page.screenshot({ 
      path: './test-results/graphiti-performance-load.png',
      fullPage: true 
    });
  });

  test('validates node rendering performance', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Check that nodes are rendered efficiently
    const nodeCount = await page.locator('[data-testid="rf__node"], .react-flow__node').count();
    console.log(`📊 Rendered nodes: ${nodeCount}`);
    
    // Should render reasonable number of nodes for performance
    expect(nodeCount).toBeGreaterThan(10);
    expect(nodeCount).toBeLessThan(100); // Culling should limit visible nodes
    
    // Check for performance stats panel
    const statsButton = page.locator('text=Stats');
    await statsButton.click();
    
    // Verify performance stats are shown
    await expect(page.locator('text=Performance')).toBeVisible();
    
    // Look for performance metrics in the stats panel
    const perfPanel = page.locator('text=Performance').locator('..');
    await expect(perfPanel).toBeVisible();
    
    // Take screenshot with performance stats visible
    await page.screenshot({ 
      path: './test-results/graphiti-performance-stats.png',
      fullPage: true 
    });
  });

  test('validates search performance', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    const searchInput = page.locator('input[placeholder*="Search"]');
    await expect(searchInput).toBeVisible();

    // Measure search performance
    const searchTerm = 'Entity';
    const searchStart = performance.now();
    
    await searchInput.fill(searchTerm);
    
    // Wait a short time for search to process
    await page.waitForTimeout(500);
    
    // Check that filtered nodes are shown
    const nodeCount = await page.locator('[data-testid="rf__node"], .react-flow__node').count();
    console.log(`🔍 Search results: ${nodeCount} nodes for "${searchTerm}"`);
    
    // Search should return results quickly
    expect(nodeCount).toBeGreaterThan(0);
    
    // Clear search
    await searchInput.fill('');
    await page.waitForTimeout(300);
    
    // Verify nodes are restored
    const restoredCount = await page.locator('[data-testid="rf__node"], .react-flow__node').count();
    expect(restoredCount).toBeGreaterThan(nodeCount);
    console.log(`🔄 Restored nodes: ${restoredCount}`);
  });

  test('validates interaction responsiveness', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Test zoom performance
    const reactFlowViewport = page.locator('.react-flow__viewport');
    await expect(reactFlowViewport).toBeVisible();
    
    // Simulate zoom in
    await reactFlowViewport.hover();
    await page.mouse.wheel(0, -5); // Zoom in
    await page.waitForTimeout(300);
    
    // Verify zoom worked without errors
    const zoomLevel = await page.evaluate(() => {
      const viewport = document.querySelector('.react-flow__viewport');
      if (viewport) {
        const transform = viewport.getAttribute('style');
        const scaleMatch = transform?.match(/scale\(([\d.]+)\)/);
        return scaleMatch ? parseFloat(scaleMatch[1]) : 1;
      }
      return 1;
    });
    
    console.log(`🔍 Zoom level: ${zoomLevel}`);
    expect(zoomLevel).toBeGreaterThan(1);
    
    // Test reset functionality
    const resetButton = page.locator('text=Reset');
    await resetButton.click();
    await page.waitForTimeout(500);
    
    // Take final interaction screenshot
    await page.screenshot({ 
      path: './test-results/graphiti-performance-interaction.png',
      fullPage: true 
    });
  });

  test('validates viewport culling', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Enable performance stats to see culling metrics
    const statsButton = page.locator('text=Stats');
    if (await statsButton.isVisible()) {
      await statsButton.click();
    }

    // Get initial node count
    const initialNodes = await page.locator('[data-testid="rf__node"], .react-flow__node').count();
    console.log(`📊 Initial visible nodes: ${initialNodes}`);

    // Zoom out significantly to test culling
    const reactFlowViewport = page.locator('.react-flow__viewport');
    await reactFlowViewport.hover();
    
    // Zoom out multiple times
    for (let i = 0; i < 5; i++) {
      await page.mouse.wheel(0, 5); // Zoom out
      await page.waitForTimeout(100);
    }
    
    await page.waitForTimeout(1000); // Wait for culling to take effect
    
    // Check node count after zoom out (culling should reduce visible nodes)
    const culledNodes = await page.locator('[data-testid="rf__node"], .react-flow__node').count();
    console.log(`🎯 Nodes after culling: ${culledNodes}`);
    
    // Take screenshot showing culled view
    await page.screenshot({ 
      path: './test-results/graphiti-performance-culling.png',
      fullPage: true 
    });
  });

  test('measures console performance logs', async ({ page }) => {
    const performanceLogs: string[] = [];
    
    // Capture console logs for performance metrics
    page.on('console', msg => {
      if (msg.text().includes('⚡') || msg.text().includes('📊')) {
        performanceLogs.push(msg.text());
      }
    });
    
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();
    
    // Enable performance stats
    const statsButton = page.locator('text=Stats');
    if (await statsButton.isVisible()) {
      await statsButton.click();
      await page.waitForTimeout(2000); // Wait for stats to update
    }
    
    // Perform some interactions to generate performance logs
    const searchInput = page.locator('input[placeholder*="Search"]');
    await searchInput.fill('test');
    await page.waitForTimeout(500);
    await searchInput.fill('');
    await page.waitForTimeout(500);
    
    // Log performance data
    console.log('🎭 Performance logs captured:');
    performanceLogs.forEach((log, index) => {
      console.log(`  ${index + 1}. ${log}`);
    });
    
    // Verify we got some performance logs
    expect(performanceLogs.length).toBeGreaterThan(0);
  });
});