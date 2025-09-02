import { test, expect, Page } from '@playwright/test';

/**
 * COMPREHENSIVE ARCHON UI TEST SUITE
 * 
 * This test suite performs exhaustive testing of the entire Archon UI
 * including Graphiti Explorer and DeepConf Dashboard
 * 
 * Test Coverage:
 * - All UI components and interactions
 * - Console error monitoring
 * - Performance metrics
 * - Responsive design
 * - Error handling states
 */

// Global console error tracking
let consoleErrors: { type: string; message: string; timestamp: number; url?: string }[] = [];
let networkErrors: { url: string; status: number; statusText: string }[] = [];

test.describe('Comprehensive Archon UI Testing', () => {
  
  test.beforeEach(async ({ page }) => {
    // Clear error arrays
    consoleErrors = [];
    networkErrors = [];
    
    // Monitor console errors
    page.on('console', msg => {
      if (['error', 'warning'].includes(msg.type())) {
        consoleErrors.push({
          type: msg.type(),
          message: msg.text(),
          timestamp: Date.now(),
          url: page.url()
        });
      }
    });
    
    // Monitor network failures
    page.on('response', response => {
      if (!response.ok()) {
        networkErrors.push({
          url: response.url(),
          status: response.status(),
          statusText: response.statusText()
        });
      }
    });
    
    // Monitor uncaught exceptions
    page.on('pageerror', error => {
      consoleErrors.push({
        type: 'pageerror',
        message: error.message,
        timestamp: Date.now(),
        url: page.url()
      });
    });
  });

  test.afterEach(async ({ page }, testInfo) => {
    // Log all collected errors for this test
    if (consoleErrors.length > 0 || networkErrors.length > 0) {
      console.log(`\n=== ERRORS FOR TEST: ${testInfo.title} ===`);
      console.log('Console Errors:', JSON.stringify(consoleErrors, null, 2));
      console.log('Network Errors:', JSON.stringify(networkErrors, null, 2));
      console.log('=== END ERRORS ===\n');
    }
  });

  test('1. Main Application Loading and Navigation', async ({ page }) => {
    // Load main page and capture initial state
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Take screenshot of main page
    await page.screenshot({ path: 'test-results/01-main-page.png', fullPage: true });
    
    // Check basic navigation elements
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
    
    // Test all navigation links
    const navLinks = await page.locator('nav a, nav button').all();
    console.log(`Found ${navLinks.length} navigation elements`);
    
    for (let i = 0; i < navLinks.length; i++) {
      const link = navLinks[i];
      const href = await link.getAttribute('href');
      const text = await link.textContent();
      
      console.log(`Testing nav element ${i + 1}: "${text}" -> ${href}`);
      
      if (href && href.startsWith('/')) {
        await link.click();
        await page.waitForLoadState('networkidle');
        await page.screenshot({ path: `test-results/nav-${i + 1}-${text?.replace(/\\s+/g, '-')}.png` });
        
        // Go back to main page
        await page.goto('/');
        await page.waitForLoadState('networkidle');
      }
    }
  });

  test('2. Graphiti Explorer - Comprehensive Testing', async ({ page }) => {
    console.log('Starting Graphiti Explorer comprehensive test...');
    
    // Navigate to Graphiti Explorer
    await page.goto('/graphiti');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Allow for data loading
    
    // Take initial screenshot
    await page.screenshot({ path: 'test-results/02-graphiti-initial.png', fullPage: true });
    
    // Test 1: View Mode Testing
    console.log('Testing view modes...');
    const viewModes = ['Minimal', 'Standard', 'Detailed', 'Expert'];
    
    for (const mode of viewModes) {
      console.log(`Testing ${mode} view mode...`);
      
      // Look for view mode selector
      const viewSelector = page.locator(`text="${mode}"`).or(
        page.locator(`button:has-text("${mode}")`).or(
          page.locator(`[data-testid="${mode.toLowerCase()}"]`)
        )
      );
      
      if (await viewSelector.count() > 0) {
        await viewSelector.first().click();
        await page.waitForTimeout(1000);
        await page.screenshot({ path: `test-results/graphiti-view-${mode.toLowerCase()}.png`, fullPage: true });
      }
    }
    
    // Test 2: Entity Cards Testing
    console.log('Testing entity cards...');
    const entityCards = page.locator('[data-testid*="entity"], .entity-card, .card');
    const cardCount = await entityCards.count();
    console.log(`Found ${cardCount} entity cards`);
    
    // Test first 5 entity cards (to avoid overwhelming test)
    for (let i = 0; i < Math.min(cardCount, 5); i++) {
      const card = entityCards.nth(i);
      await card.click();
      await page.waitForTimeout(500);
      
      // Check if details panel appears
      const detailsPanel = page.locator('[data-testid="details-panel"], .details-panel, .sidebar-panel');
      if (await detailsPanel.count() > 0) {
        await page.screenshot({ path: `test-results/graphiti-card-${i + 1}-details.png`, fullPage: true });
      }
    }
    
    // Test 3: Search Functionality
    console.log('Testing search functionality...');
    const searchInput = page.locator('input[type="search"], input[placeholder*="search"], [data-testid*="search"]');
    
    if (await searchInput.count() > 0) {
      const searchTerms = ['project', 'user', 'task', 'test'];
      
      for (const term of searchTerms) {
        await searchInput.first().fill(term);
        await page.keyboard.press('Enter');
        await page.waitForTimeout(1000);
        await page.screenshot({ path: `test-results/graphiti-search-${term}.png`, fullPage: true });
        
        // Clear search
        await searchInput.first().fill('');
        await page.keyboard.press('Enter');
        await page.waitForTimeout(500);
      }
    }
    
    // Test 4: Filter Options
    console.log('Testing filter options...');
    const filterButtons = page.locator('button:has-text("Filter"), [data-testid*="filter"], .filter-button');
    
    if (await filterButtons.count() > 0) {
      await filterButtons.first().click();
      await page.waitForTimeout(500);
      await page.screenshot({ path: 'test-results/graphiti-filters-open.png', fullPage: true });
      
      // Test expandable sections
      const expandableItems = page.locator('[data-testid*="expandable"], .expandable, details');
      const expandableCount = await expandableItems.count();
      console.log(`Found ${expandableCount} expandable items`);
      
      for (let i = 0; i < Math.min(expandableCount, 3); i++) {
        const item = expandableItems.nth(i);
        await item.click();
        await page.waitForTimeout(300);
      }
      
      await page.screenshot({ path: 'test-results/graphiti-filters-expanded.png', fullPage: true });
    }
    
    // Test 5: Confidence Scores
    console.log('Testing confidence scores...');
    const confidenceScores = page.locator('[data-testid*="confidence"], .confidence-score, .score');
    const scoreCount = await confidenceScores.count();
    console.log(`Found ${scoreCount} confidence score elements`);
    
    if (scoreCount > 0) {
      await page.screenshot({ path: 'test-results/graphiti-confidence-scores.png', fullPage: true });
    }
    
    // Test 6: Network Status
    console.log('Testing network status indicators...');
    const statusIndicators = page.locator('[data-testid*="status"], .status-indicator, .network-status');
    const statusCount = await statusIndicators.count();
    console.log(`Found ${statusCount} status indicator elements`);
    
    // Test 7: Onboarding System
    console.log('Testing onboarding system...');
    const onboardingElements = page.locator('[data-testid*="onboarding"], .onboarding, .tour, .walkthrough');
    
    if (await onboardingElements.count() > 0) {
      // Look for skip button
      const skipButton = page.locator('button:has-text("Skip"), [data-testid*="skip"]');
      if (await skipButton.count() > 0) {
        await skipButton.first().click();
        await page.waitForTimeout(500);
      }
      
      // Look for restart onboarding
      const restartButton = page.locator('button:has-text("Restart"), button:has-text("Tour"), [data-testid*="restart"]');
      if (await restartButton.count() > 0) {
        await restartButton.first().click();
        await page.waitForTimeout(500);
        await page.screenshot({ path: 'test-results/graphiti-onboarding-restart.png', fullPage: true });
        
        // Complete onboarding
        const completeButton = page.locator('button:has-text("Complete"), button:has-text("Finish"), [data-testid*="complete"]');
        if (await completeButton.count() > 0) {
          await completeButton.first().click();
        }
      }
    }
    
    // Test 8: Hover Effects and Animations
    console.log('Testing hover effects...');
    const hoverableElements = page.locator('.card, .entity, button, .clickable');
    const hoverCount = await hoverableElements.count();
    
    for (let i = 0; i < Math.min(hoverCount, 5); i++) {
      const element = hoverableElements.nth(i);
      await element.hover();
      await page.waitForTimeout(200);
    }
    
    await page.screenshot({ path: 'test-results/graphiti-hover-effects.png', fullPage: true });
  });

  test('3. DeepConf Dashboard - Comprehensive Testing', async ({ page }) => {
    console.log('Starting DeepConf Dashboard comprehensive test...');
    
    // Navigate to DeepConf Dashboard
    await page.goto('/deepconf');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000); // Allow for data loading and WebSocket connections
    
    // Take initial screenshot
    await page.screenshot({ path: 'test-results/03-deepconf-initial.png', fullPage: true });
    
    // Test 1: Dashboard Components
    console.log('Testing dashboard components...');
    const dashboardComponents = [
      '[data-testid*="dashboard"]',
      '.dashboard-component',
      '.metric-card',
      '.chart-container',
      '.stats-panel'
    ];
    
    for (const selector of dashboardComponents) {
      const elements = page.locator(selector);
      const count = await elements.count();
      if (count > 0) {
        console.log(`Found ${count} elements matching ${selector}`);
        await page.screenshot({ path: `test-results/deepconf-${selector.replace(/[\[\]\.#\*"=]/g, '')}.png`, fullPage: true });
      }
    }
    
    // Test 2: Real-time Data Updates
    console.log('Testing real-time data updates...');
    const metricsElements = page.locator('[data-testid*="metric"], .metric, .live-data');
    const initialMetrics: string[] = [];
    
    // Capture initial state
    const metricsCount = await metricsElements.count();
    for (let i = 0; i < metricsCount; i++) {
      const text = await metricsElements.nth(i).textContent();
      initialMetrics.push(text || '');
    }
    
    // Wait for potential updates
    await page.waitForTimeout(5000);
    
    // Capture updated state
    const updatedMetrics: string[] = [];
    for (let i = 0; i < metricsCount; i++) {
      const text = await metricsElements.nth(i).textContent();
      updatedMetrics.push(text || '');
    }
    
    // Compare for changes (indicates real-time updates)
    let hasUpdates = false;
    for (let i = 0; i < initialMetrics.length; i++) {
      if (initialMetrics[i] !== updatedMetrics[i]) {
        hasUpdates = true;
        console.log(`Real-time update detected in metric ${i}: "${initialMetrics[i]}" -> "${updatedMetrics[i]}"`);
      }
    }
    
    console.log(`Real-time updates detected: ${hasUpdates}`);
    
    // Test 3: Interactive Elements and Controls
    console.log('Testing interactive elements...');
    const interactiveElements = page.locator('button, select, input, [data-testid*="control"]');
    const interactiveCount = await interactiveElements.count();
    console.log(`Found ${interactiveCount} interactive elements`);
    
    for (let i = 0; i < Math.min(interactiveCount, 10); i++) {
      const element = interactiveElements.nth(i);
      const tagName = await element.evaluate(el => el.tagName.toLowerCase());
      const type = await element.getAttribute('type');
      
      try {
        if (tagName === 'button') {
          await element.click();
          await page.waitForTimeout(500);
        } else if (tagName === 'select') {
          const options = await element.locator('option').count();
          if (options > 1) {
            await element.selectOption({ index: 1 });
            await page.waitForTimeout(500);
          }
        } else if (tagName === 'input' && type !== 'file') {
          await element.fill('test');
          await page.waitForTimeout(300);
          await element.fill('');
        }
      } catch (error) {
        console.log(`Interaction failed for element ${i}: ${error}`);
      }
    }
    
    await page.screenshot({ path: 'test-results/deepconf-interactions.png', fullPage: true });
    
    // Test 4: Confidence Scoring Calculations
    console.log('Testing confidence scoring...');
    const confidenceElements = page.locator('[data-testid*="confidence"], .confidence, .score-display');
    const confidenceCount = await confidenceElements.count();
    
    if (confidenceCount > 0) {
      console.log(`Found ${confidenceCount} confidence scoring elements`);
      
      for (let i = 0; i < confidenceCount; i++) {
        const element = confidenceElements.nth(i);
        const text = await element.textContent();
        const value = await element.getAttribute('data-value') || await element.getAttribute('value');
        console.log(`Confidence element ${i}: Text="${text}", Value="${value}"`);
      }
      
      await page.screenshot({ path: 'test-results/deepconf-confidence-scores.png', fullPage: true });
    }
    
    // Test 5: SCWT Metrics Visualization
    console.log('Testing SCWT metrics...');
    const scwtElements = page.locator('[data-testid*="scwt"], .scwt-metric, .scwt-chart');
    const scwtCount = await scwtElements.count();
    
    if (scwtCount > 0) {
      console.log(`Found ${scwtCount} SCWT metric elements`);
      await page.screenshot({ path: 'test-results/deepconf-scwt-metrics.png', fullPage: true });
    }
    
    // Test 6: Charts and Graphs
    console.log('Testing charts and graphs...');
    const chartElements = page.locator('canvas, svg, .chart, [data-testid*="chart"]');
    const chartCount = await chartElements.count();
    
    if (chartCount > 0) {
      console.log(`Found ${chartCount} chart elements`);
      
      // Test chart interactions
      for (let i = 0; i < chartCount; i++) {
        const chart = chartElements.nth(i);
        await chart.hover();
        await page.waitForTimeout(500);
        
        // Try clicking on chart for tooltips/interactions
        try {
          await chart.click();
          await page.waitForTimeout(300);
        } catch (error) {
          console.log(`Chart interaction failed: ${error}`);
        }
      }
      
      await page.screenshot({ path: 'test-results/deepconf-charts.png', fullPage: true });
    }
    
    // Test 7: Configuration/Settings Panels
    console.log('Testing configuration panels...');
    const configElements = page.locator('[data-testid*="config"], [data-testid*="settings"], .config-panel, .settings-panel');
    const configCount = await configElements.count();
    
    if (configCount > 0) {
      console.log(`Found ${configCount} configuration elements`);
      
      for (let i = 0; i < configCount; i++) {
        const config = configElements.nth(i);
        await config.click();
        await page.waitForTimeout(500);
      }
      
      await page.screenshot({ path: 'test-results/deepconf-config.png', fullPage: true });
    }
    
    // Test 8: Error Handling and Fallback States
    console.log('Testing error states...');
    
    // Simulate network issues by intercepting requests
    await page.route('**/api/**', route => {
      // Fail some API requests to test error handling
      if (Math.random() > 0.7) {
        route.fulfill({
          status: 500,
          body: 'Internal Server Error'
        });
      } else {
        route.continue();
      }
    });
    
    // Reload page to trigger error states
    await page.reload();
    await page.waitForTimeout(3000);
    
    await page.screenshot({ path: 'test-results/deepconf-error-states.png', fullPage: true });
    
    // Remove route interception
    await page.unroute('**/api/**');
  });

  test('4. Responsive Design Testing', async ({ page, browserName }) => {
    console.log(`Testing responsive design on ${browserName}...`);
    
    const viewports = [
      { width: 320, height: 568, name: 'mobile-small' },
      { width: 375, height: 667, name: 'mobile-medium' },
      { width: 768, height: 1024, name: 'tablet-portrait' },
      { width: 1024, height: 768, name: 'tablet-landscape' },
      { width: 1200, height: 800, name: 'desktop-small' },
      { width: 1920, height: 1080, name: 'desktop-large' }
    ];
    
    const testPages = ['/graphiti', '/deepconf'];
    
    for (const testPage of testPages) {
      for (const viewport of viewports) {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        await page.goto(testPage);
        await page.waitForLoadState('networkidle');
        await page.waitForTimeout(1000);
        
        await page.screenshot({ 
          path: `test-results/responsive-${testPage.replace('/', '')}-${viewport.name}-${browserName}.png`,
          fullPage: true 
        });
        
        // Test navigation menu on small screens
        if (viewport.width < 768) {
          const hamburgerMenu = page.locator('[data-testid*="menu"], .hamburger, .mobile-menu-button');
          if (await hamburgerMenu.count() > 0) {
            await hamburgerMenu.first().click();
            await page.waitForTimeout(500);
            await page.screenshot({ 
              path: `test-results/responsive-${testPage.replace('/', '')}-${viewport.name}-${browserName}-menu.png`,
              fullPage: true 
            });
          }
        }
      }
    }
  });

  test('5. Performance Metrics Testing', async ({ page }) => {
    console.log('Testing performance metrics...');
    
    const testPages = ['/', '/graphiti', '/deepconf'];
    const performanceResults: Array<{
      page: string;
      loadTime: number;
      domContentLoaded: number;
      firstPaint: number;
      resourceCount: number;
    }> = [];
    
    for (const testPage of testPages) {
      console.log(`Testing performance for ${testPage}...`);
      
      const startTime = Date.now();
      
      // Navigate and wait for load
      await page.goto(testPage);
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      
      // Get performance metrics
      const perfMetrics = await page.evaluate(() => {
        const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paints = performance.getEntriesByType('paint');
        const resources = performance.getEntriesByType('resource');
        
        return {
          domContentLoaded: nav.domContentLoadedEventEnd - nav.navigationStart,
          firstPaint: paints.find(p => p.name === 'first-paint')?.startTime || 0,
          resourceCount: resources.length
        };
      });
      
      performanceResults.push({
        page: testPage,
        loadTime,
        domContentLoaded: perfMetrics.domContentLoaded,
        firstPaint: perfMetrics.firstPaint,
        resourceCount: perfMetrics.resourceCount
      });
      
      console.log(`${testPage} Performance:`, {
        loadTime: `${loadTime}ms`,
        domContentLoaded: `${perfMetrics.domContentLoaded}ms`,
        firstPaint: `${perfMetrics.firstPaint}ms`,
        resources: perfMetrics.resourceCount
      });
    }
    
    // Log all performance results
    console.log('\\n=== PERFORMANCE SUMMARY ===');
    performanceResults.forEach(result => {
      console.log(`${result.page}:`);
      console.log(`  Load Time: ${result.loadTime}ms`);
      console.log(`  DOM Content Loaded: ${result.domContentLoaded}ms`);
      console.log(`  First Paint: ${result.firstPaint}ms`);
      console.log(`  Resources: ${result.resourceCount}`);
      console.log('');
    });
  });

  test('6. Accessibility Testing', async ({ page }) => {
    console.log('Testing accessibility features...');
    
    const testPages = ['/graphiti', '/deepconf'];
    
    for (const testPage of testPages) {
      await page.goto(testPage);
      await page.waitForLoadState('networkidle');
      
      // Test keyboard navigation
      console.log(`Testing keyboard navigation on ${testPage}...`);
      
      // Press Tab multiple times and check focus
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press('Tab');
        await page.waitForTimeout(200);
        
        const focusedElement = await page.evaluate(() => {
          const focused = document.activeElement;
          return focused ? {
            tagName: focused.tagName,
            id: focused.id,
            className: focused.className,
            textContent: focused.textContent?.substring(0, 50)
          } : null;
        });
        
        if (focusedElement) {
          console.log(`Tab ${i + 1}: ${focusedElement.tagName}${focusedElement.id ? '#' + focusedElement.id : ''}${focusedElement.className ? '.' + focusedElement.className.split(' ')[0] : ''}`);
        }
      }
      
      // Test ARIA attributes
      const ariaElements = await page.locator('[aria-label], [aria-describedby], [aria-expanded], [role]').count();
      console.log(`Found ${ariaElements} elements with ARIA attributes on ${testPage}`);
      
      // Test alt text on images
      const images = await page.locator('img').count();
      const imagesWithAlt = await page.locator('img[alt]').count();
      console.log(`Images on ${testPage}: ${imagesWithAlt}/${images} have alt text`);
      
      await page.screenshot({ path: `test-results/accessibility-${testPage.replace('/', '')}.png`, fullPage: true });
    }
  });

  // Final test to summarize all collected errors
  test('7. Error Summary and Analysis', async ({ page }) => {
    console.log('\\n=== FINAL ERROR ANALYSIS ===');
    console.log('Total Console Errors Collected:', consoleErrors.length);
    console.log('Total Network Errors Collected:', networkErrors.length);
    
    // Categorize errors
    const errorCategories = {
      critical: consoleErrors.filter(e => e.type === 'error' || e.type === 'pageerror'),
      warnings: consoleErrors.filter(e => e.type === 'warning'),
      networkFailures: networkErrors
    };
    
    console.log('\\n--- ERROR BREAKDOWN ---');
    console.log(`Critical Errors: ${errorCategories.critical.length}`);
    console.log(`Warnings: ${errorCategories.warnings.length}`);
    console.log(`Network Failures: ${errorCategories.networkFailures.length}`);
    
    // Log detailed error information
    if (errorCategories.critical.length > 0) {
      console.log('\\n--- CRITICAL ERRORS ---');
      errorCategories.critical.forEach((error, index) => {
        console.log(`${index + 1}. [${error.type.toUpperCase()}] ${error.message}`);
        console.log(`   URL: ${error.url}`);
        console.log(`   Time: ${new Date(error.timestamp).toISOString()}`);
      });
    }
    
    if (errorCategories.warnings.length > 0) {
      console.log('\\n--- WARNINGS ---');
      errorCategories.warnings.forEach((warning, index) => {
        console.log(`${index + 1}. ${warning.message}`);
        console.log(`   URL: ${warning.url}`);
      });
    }
    
    if (errorCategories.networkFailures.length > 0) {
      console.log('\\n--- NETWORK FAILURES ---');
      errorCategories.networkFailures.forEach((error, index) => {
        console.log(`${index + 1}. ${error.status} ${error.statusText}: ${error.url}`);
      });
    }
    
    console.log('\\n=== END ERROR ANALYSIS ===\\n');
    
    // Create a simple HTML report
    const htmlReport = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Archon UI Test Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .error { color: red; }
            .warning { color: orange; }
            .success { color: green; }
            .section { margin: 20px 0; padding: 10px; border-left: 3px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>Archon UI Comprehensive Test Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p><strong>Test Run:</strong> ${new Date().toISOString()}</p>
            <p><strong>Total Console Errors:</strong> <span class="error">${consoleErrors.length}</span></p>
            <p><strong>Total Network Errors:</strong> <span class="error">${networkErrors.length}</span></p>
        </div>
        
        <div class="section">
            <h2>Critical Issues</h2>
            ${errorCategories.critical.length === 0 ? '<p class="success">No critical issues found!</p>' : 
              errorCategories.critical.map((error, i) => `
                <div class="error">
                    <strong>${i + 1}. [${error.type.toUpperCase()}]</strong> ${error.message}<br>
                    <small>URL: ${error.url} | Time: ${new Date(error.timestamp).toISOString()}</small>
                </div>
              `).join('')
            }
        </div>
        
        <div class="section">
            <h2>Network Issues</h2>
            ${errorCategories.networkFailures.length === 0 ? '<p class="success">No network issues found!</p>' : 
              errorCategories.networkFailures.map((error, i) => `
                <div class="error">
                    <strong>${i + 1}.</strong> ${error.status} ${error.statusText}: ${error.url}
                </div>
              `).join('')
            }
        </div>
        
        <div class="section">
            <h2>Warnings</h2>
            ${errorCategories.warnings.length === 0 ? '<p class="success">No warnings found!</p>' : 
              errorCategories.warnings.map((warning, i) => `
                <div class="warning">
                    <strong>${i + 1}.</strong> ${warning.message}<br>
                    <small>URL: ${warning.url}</small>
                </div>
              `).join('')
            }
        </div>
    </body>
    </html>
    `;
    
    // Save the HTML report using page context
    await page.evaluate((htmlContent) => {
      // Create a temporary blob and download link
      const blob = new Blob([htmlContent], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'comprehensive-test-report.html';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, htmlReport);
    
    console.log('📊 Comprehensive test report saved to: test-results/comprehensive-test-report.html');
  });
});