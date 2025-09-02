import { test, expect, Page } from '@playwright/test';

/**
 * Test the Current Graphiti Explorer Implementation
 * Focus on actual UX/UI assessment of existing functionality
 */

class GraphitiExplorerPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/graphiti');
  }

  // Wait for page to load
  async waitForLoad() {
    await this.page.waitForLoadState('networkidle');
    await this.page.waitForTimeout(2000); // Allow React to initialize
  }
}

test.describe('Current Graphiti Explorer UX Assessment', () => {
  let graphitiPage: GraphitiExplorerPage;

  test.beforeEach(async ({ page }) => {
    graphitiPage = new GraphitiExplorerPage(page);
  });

  test('loads the Graphiti Explorer page', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Check if page loads without errors
    await expect(page).toHaveTitle(/Archon|Graphiti/);
    
    // Look for main content
    const body = page.locator('body');
    await expect(body).toBeVisible();

    // Take screenshot for visual inspection
    await page.screenshot({ 
      path: './test-results/graphiti-homepage.png',
      fullPage: true 
    });
  });

  test('evaluates visual hierarchy and design', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Check for header
    const headers = page.locator('h1, h2, .header, [role="banner"]');
    if (await headers.count() > 0) {
      await expect(headers.first()).toBeVisible();
      
      // Evaluate header styling
      const headerStyles = await headers.first().evaluate(el => {
        const styles = getComputedStyle(el);
        return {
          fontSize: styles.fontSize,
          fontWeight: styles.fontWeight,
          color: styles.color,
          marginBottom: styles.marginBottom
        };
      });
      
      console.log('Header styling:', headerStyles);
    }

    // Look for main graph container
    const graphContainer = page.locator('.react-flow, [data-testid="react-flow"], .graph-container, canvas, svg');
    if (await graphContainer.count() > 0) {
      await expect(graphContainer.first()).toBeVisible();
      
      const containerSize = await graphContainer.first().boundingBox();
      console.log('Graph container size:', containerSize);
      
      // Should occupy reasonable screen space
      if (containerSize) {
        expect(containerSize.width).toBeGreaterThan(300);
        expect(containerSize.height).toBeGreaterThan(300);
      }
    }

    // Check for search/filter controls
    const searchInput = page.locator('input[type="text"], input[placeholder*="search" i]');
    const filterControls = page.locator('select, .select, [role="combobox"]');
    
    console.log('Search inputs found:', await searchInput.count());
    console.log('Filter controls found:', await filterControls.count());

    // Take detailed screenshot
    await page.screenshot({ 
      path: './test-results/graphiti-visual-assessment.png',
      fullPage: true 
    });
  });

  test('checks accessibility basics', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Check for keyboard navigation
    await page.keyboard.press('Tab');
    const focusedElement = await page.evaluate(() => {
      const activeEl = document.activeElement;
      return {
        tagName: activeEl?.tagName,
        className: activeEl?.className,
        id: activeEl?.id,
        role: activeEl?.getAttribute('role')
      };
    });
    
    console.log('First focused element:', focusedElement);

    // Check for ARIA labels on interactive elements
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    console.log('Total buttons found:', buttonCount);
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(buttonCount, 5); i++) {
        const button = buttons.nth(i);
        const ariaLabel = await button.getAttribute('aria-label');
        const title = await button.getAttribute('title');
        const textContent = await button.textContent();
        
        console.log(`Button ${i}:`, {
          ariaLabel,
          title,
          textContent: textContent?.trim(),
          hasAccessibleName: !!(ariaLabel || title || textContent?.trim())
        });
      }
    }

    // Check for headings structure
    const headings = page.locator('h1, h2, h3, h4, h5, h6');
    const headingCount = await headings.count();
    console.log('Headings found:', headingCount);
    
    if (headingCount > 0) {
      for (let i = 0; i < Math.min(headingCount, 3); i++) {
        const heading = headings.nth(i);
        const level = await heading.evaluate(el => el.tagName);
        const text = await heading.textContent();
        console.log(`${level}: ${text?.trim()}`);
      }
    }
  });

  test('tests responsive behavior', async ({ page }) => {
    const viewports = [
      { width: 375, height: 667, name: 'iPhone SE' },
      { width: 768, height: 1024, name: 'iPad' },
      { width: 1200, height: 800, name: 'Desktop' }
    ];

    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await graphitiPage.goto();
      await graphitiPage.waitForLoad();

      console.log(`Testing ${viewport.name} (${viewport.width}x${viewport.height})`);

      // Check if content fits viewport
      const body = page.locator('body');
      const bodySize = await body.boundingBox();
      
      if (bodySize) {
        const hasHorizontalScroll = bodySize.width > viewport.width;
        console.log(`  Horizontal scroll needed: ${hasHorizontalScroll}`);
        
        // Some horizontal scroll might be acceptable for graph interactions
        if (hasHorizontalScroll && viewport.width < 768) {
          console.log('  Mobile horizontal scroll detected - may need responsive improvements');
        }
      }

      // Check if main UI elements are accessible
      const mainElements = page.locator('button, input, select, .interactive, [role="button"]');
      const elementCount = await mainElements.count();
      console.log(`  Interactive elements found: ${elementCount}`);
      
      if (elementCount > 0) {
        const firstElement = mainElements.first();
        if (await firstElement.isVisible()) {
          const elementBox = await firstElement.boundingBox();
          if (elementBox && elementBox.width > 0 && elementBox.height > 0) {
            console.log(`  First interactive element size: ${elementBox.width}x${elementBox.height}`);
            
            // Check if touch-friendly (for mobile)
            if (viewport.width < 768 && elementBox.height < 44) {
              console.log('  WARNING: Interactive element may be too small for touch');
            }
          }
        }
      }

      // Take screenshot for each viewport
      await page.screenshot({ 
        path: `./test-results/graphiti-${viewport.name.toLowerCase().replace(' ', '-')}.png`,
        fullPage: true 
      });
    }
  });

  test('evaluates error handling', async ({ page }) => {
    await graphitiPage.goto();
    
    // Look for any console errors
    const errors: string[] = [];
    page.on('pageerror', error => {
      errors.push(error.message);
    });
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await graphitiPage.waitForLoad();
    
    console.log('Console errors detected:', errors.length);
    if (errors.length > 0) {
      console.log('Errors:', errors.slice(0, 5)); // Show first 5 errors
    }

    // Look for error states in UI
    const errorElements = page.locator('.error, [role="alert"], .alert-error, .text-red, [class*="error"]');
    const errorCount = await errorElements.count();
    console.log('Error UI elements found:', errorCount);

    // Look for loading states
    const loadingElements = page.locator('.loading, .spinner, .skeleton, [aria-busy="true"]');
    const loadingCount = await loadingElements.count();
    console.log('Loading UI elements found:', loadingCount);

    // Look for empty states
    const emptyElements = page.locator('.empty, .no-data, .placeholder, [class*="empty"]');
    const emptyCount = await emptyElements.count();
    console.log('Empty state elements found:', emptyCount);
  });

  test('assesses overall user experience', async ({ page }) => {
    await graphitiPage.goto();
    await graphitiPage.waitForLoad();

    // Measure page load performance
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.navigationStart,
        loadComplete: navigation.loadEventEnd - navigation.navigationStart,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0
      };
    });
    
    console.log('Performance metrics:', performanceMetrics);

    // Evaluate information density
    const textElements = page.locator('p, span, div, label, button, a, h1, h2, h3, h4, h5, h6');
    const textCount = await textElements.count();
    const visibleTextCount = await textElements.filter({ hasText: /.+/ }).count();
    
    console.log('Text elements:', { total: textCount, withContent: visibleTextCount });

    // Check for overwhelming UI (too many visible controls)
    const interactiveElements = page.locator('button, input, select, a, [role="button"], [tabindex="0"]');
    const interactiveCount = await interactiveElements.count();
    console.log('Interactive elements count:', interactiveCount);
    
    if (interactiveCount > 20) {
      console.log('WARNING: High number of interactive elements - may overwhelm users');
    }

    // Check for helpful features
    const helpElements = page.locator('[aria-label*="help" i], [title*="help" i], .tooltip, [data-tooltip]');
    const helpCount = await helpElements.count();
    console.log('Help/tooltip elements found:', helpCount);

    // Final UX assessment screenshot
    await page.screenshot({ 
      path: './test-results/graphiti-ux-assessment.png',
      fullPage: true 
    });

    // Summary assessment
    const assessment = {
      performanceGood: performanceMetrics.domContentLoaded < 3000,
      reasonableComplexity: interactiveCount <= 20 && interactiveCount >= 5,
      hasHelp: helpCount > 0,
      overallScore: 0
    };

    assessment.overallScore = [
      assessment.performanceGood,
      assessment.reasonableComplexity,
      assessment.hasHelp
    ].filter(Boolean).length;

    console.log('UX Assessment Score:', `${assessment.overallScore}/3`);
    console.log('Assessment details:', assessment);
  });
});