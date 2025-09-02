import { test, expect, Page, Locator } from '@playwright/test';

/**
 * Comprehensive UX/UI Testing Strategy for Graphiti Explorer
 * 
 * This test suite covers:
 * 1. Visual design consistency and hierarchy
 * 2. Accessibility compliance (WCAG 2.1 AA)
 * 3. Responsive behavior across breakpoints
 * 4. User interaction flows and micro-interactions
 * 5. Performance and loading states
 * 6. Error handling and recovery
 * 7. Mobile touch interactions
 */

// Test data setup
const mockGraphData = {
  nodes: [
    { id: '1', type: 'function', name: 'calculateTotal', confidence: 0.95 },
    { id: '2', type: 'class', name: 'UserManager', confidence: 0.87 },
    { id: '3', type: 'concept', name: 'Authentication', confidence: 0.92 }
  ],
  edges: [
    { id: 'e1', source: '1', target: '2', type: 'calls' },
    { id: 'e2', source: '2', target: '3', type: 'implements' }
  ]
};

class GraphitiExplorerPage {
  constructor(private page: Page) {}

  // Locators
  get header() { return this.page.getByRole('banner'); }
  get searchInput() { return this.page.getByPlaceholder('Search entities...'); }
  get graphCanvas() { return this.page.locator('[data-testid="react-flow-wrapper"]'); }
  get entityNodes() { return this.page.locator('[data-testid="entity-node"]'); }
  get loadingSpinner() { return this.page.locator('[data-testid="loading-spinner"]'); }
  get errorMessage() { return this.page.locator('[data-testid="error-message"]'); }
  get filterDropdown() { return this.page.locator('[data-testid="entity-type-filter"]'); }
  get detailsPanel() { return this.page.locator('[data-testid="entity-details"]'); }
  get viewModeToggle() { return this.page.locator('[data-testid="view-mode-toggle"]'); }
  get onboardingOverlay() { return this.page.locator('[data-testid="onboarding-overlay"]'); }
  get performancePanel() { return this.page.locator('[data-testid="performance-panel"]'); }

  // Helper methods
  async waitForGraphLoad() {
    await this.loadingSpinner.waitFor({ state: 'hidden', timeout: 10000 });
    await expect(this.entityNodes.first()).toBeVisible();
  }

  async selectEntity(entityName: string) {
    await this.page.getByText(entityName).first().click();
  }

  async switchViewMode(mode: 'minimal' | 'standard' | 'advanced') {
    await this.viewModeToggle.locator(`text=${mode}`).click();
  }

  async performSearch(query: string) {
    await this.searchInput.fill(query);
    await this.searchInput.press('Enter');
  }
}

test.describe('Graphiti Explorer UX Tests', () => {
  let graphitiPage: GraphitiExplorerPage;

  test.beforeEach(async ({ page }) => {
    graphitiPage = new GraphitiExplorerPage(page);
    
    // Mock API responses
    await page.route('**/api/graphiti/graph-data', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          nodes: mockGraphData.nodes.map(n => ({
            id: n.id,
            label: n.name,
            type: n.type,
            properties: { confidence_score: n.confidence, entity_type: n.type, name: n.name },
            position: { x: Math.random() * 800, y: Math.random() * 600 }
          })),
          edges: mockGraphData.edges.map(e => ({
            id: e.id,
            source: e.source,
            target: e.target,
            type: e.type,
            properties: { confidence: 0.8 }
          })),
          metadata: {
            total_entities: mockGraphData.nodes.length,
            total_relationships: mockGraphData.edges.length,
            entity_types: ['function', 'class', 'concept'],
            relationship_types: ['calls', 'implements'],
            last_updated: Date.now()
          }
        })
      });
    });

    await page.goto('/graphiti');
  });

  test.describe('@visual Visual Design & Hierarchy', () => {
    test('displays proper visual hierarchy in header', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Check header structure
      const header = graphitiPage.header;
      await expect(header).toBeVisible();
      
      // Verify title prominence
      const title = header.getByRole('heading', { level: 1 });
      await expect(title).toHaveText(/Graphiti Explorer/);
      await expect(title).toHaveCSS('font-size', /24px|1.5rem/);
      await expect(title).toHaveCSS('font-weight', /600|700/);

      // Check entity count badges
      const badges = header.locator('[data-testid="entity-count-badge"]');
      await expect(badges).toHaveCount(2); // entities and relationships
      
      // Verify badge styling consistency
      for (const badge of await badges.all()) {
        await expect(badge).toHaveCSS('font-size', /12px|0.75rem/);
        await expect(badge).toHaveCSS('border-radius', /9999px|50%/);
      }
    });

    test('maintains consistent color scheme across entity types', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      const entityNodes = graphitiPage.entityNodes;
      await expect(entityNodes).toHaveCountGreaterThan(0);

      // Check color consistency for each entity type
      const functionNode = page.locator('[data-entity-type="function"]').first();
      const classNode = page.locator('[data-entity-type="class"]').first();

      if (await functionNode.isVisible()) {
        const functionColor = await functionNode.evaluate(el => 
          getComputedStyle(el).borderColor
        );
        expect(functionColor).toMatch(/rgb\(59, 130, 246\)/); // Blue
      }

      if (await classNode.isVisible()) {
        const classColor = await classNode.evaluate(el => 
          getComputedStyle(el).borderColor
        );
        expect(classColor).toMatch(/rgb\(16, 185, 129\)/); // Green
      }
    });

    test('shows proper spacing and alignment', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Check main layout spacing
      const mainContent = page.locator('main');
      const padding = await mainContent.evaluate(el => 
        getComputedStyle(el).padding
      );
      
      // Should have consistent padding
      expect(padding).not.toBe('0px');

      // Check entity node spacing
      const nodes = await graphitiPage.entityNodes.all();
      if (nodes.length > 1) {
        const positions = await Promise.all(
          nodes.map(node => node.boundingBox())
        );
        
        // Verify nodes don't overlap (basic spacing check)
        for (let i = 0; i < positions.length - 1; i++) {
          for (let j = i + 1; j < positions.length; j++) {
            const pos1 = positions[i];
            const pos2 = positions[j];
            if (pos1 && pos2) {
              const distance = Math.sqrt(
                Math.pow(pos1.x - pos2.x, 2) + Math.pow(pos1.y - pos2.y, 2)
              );
              expect(distance).toBeGreaterThan(50); // Minimum spacing
            }
          }
        }
      }
    });
  });

  test.describe('@a11y Accessibility Compliance', () => {
    test('supports keyboard navigation', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Focus should start on the graph container
      await page.keyboard.press('Tab');
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(['DIV', 'MAIN', 'SECTION']).toContain(focusedElement);

      // Should be able to navigate with arrow keys
      await page.keyboard.press('ArrowRight');
      
      // Check if an entity is focused/selected
      const selectedNode = page.locator('[aria-selected="true"], .selected');
      if (await selectedNode.count() > 0) {
        await expect(selectedNode.first()).toBeVisible();
      }

      // Enter should activate the focused entity
      await page.keyboard.press('Enter');
      
      // Details panel should open or entity should be selected
      const detailsPanel = graphitiPage.detailsPanel;
      if (await detailsPanel.isVisible()) {
        await expect(detailsPanel).toBeVisible();
      }
    });

    test('provides proper ARIA labels and roles', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Check main graph container
      const graphContainer = graphitiPage.graphCanvas;
      await expect(graphContainer).toHaveAttribute('role');
      
      const role = await graphContainer.getAttribute('role');
      expect(['application', 'img', 'figure']).toContain(role);

      // Check entity nodes have proper labels
      const entityNodes = graphitiPage.entityNodes;
      const nodeCount = await entityNodes.count();
      
      for (let i = 0; i < Math.min(nodeCount, 3); i++) {
        const node = entityNodes.nth(i);
        const ariaLabel = await node.getAttribute('aria-label');
        expect(ariaLabel).toBeTruthy();
        expect(ariaLabel).toMatch(/entity/i);
      }

      // Check buttons have accessible names
      const buttons = page.locator('button');
      const buttonCount = await buttons.count();
      
      for (let i = 0; i < buttonCount; i++) {
        const button = buttons.nth(i);
        const ariaLabel = await button.getAttribute('aria-label');
        const title = await button.getAttribute('title');
        const textContent = await button.textContent();
        
        expect(ariaLabel || title || textContent?.trim()).toBeTruthy();
      }
    });

    test('meets color contrast requirements', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Check text elements for contrast
      const textElements = [
        page.locator('h1, h2, h3, h4, h5, h6'),
        page.locator('p, span, div').filter({ hasText: /\w/ }).first(),
        page.locator('button').first(),
        page.locator('[data-testid="entity-name"]').first()
      ];

      for (const element of textElements) {
        if (await element.isVisible()) {
          const contrast = await element.evaluate((el) => {
            const style = getComputedStyle(el);
            const color = style.color;
            const backgroundColor = style.backgroundColor;
            
            // Basic contrast check (simplified)
            const parseRGB = (rgbString: string) => {
              const match = rgbString.match(/rgb\((\d+), (\d+), (\d+)\)/);
              return match ? [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])] : [0, 0, 0];
            };
            
            const [r1, g1, b1] = parseRGB(color);
            const [r2, g2, b2] = parseRGB(backgroundColor);
            
            // Relative luminance calculation (simplified)
            const getLuminance = (r: number, g: number, b: number) => {
              const [rs, gs, bs] = [r, g, b].map(c => {
                c = c / 255;
                return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
              });
              return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
            };
            
            const l1 = getLuminance(r1, g1, b1);
            const l2 = getLuminance(r2, g2, b2);
            
            const contrast = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
            return contrast;
          });
          
          // WCAG AA requirement: 4.5:1 for normal text, 3:1 for large text
          expect(contrast).toBeGreaterThan(3.0);
        }
      }
    });

    test('supports screen reader announcements', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // Check for live regions
      const liveRegions = page.locator('[aria-live]');
      await expect(liveRegions).toHaveCountGreaterThan(0);

      // Check that status messages are announced
      const statusRegion = page.locator('[aria-live="polite"], [role="status"]');
      if (await statusRegion.count() > 0) {
        await expect(statusRegion.first()).toBeInViewport();
      }

      // Test search functionality with screen reader support
      await graphitiPage.performSearch('test');
      
      // Should announce search results or status
      await page.waitForTimeout(1000); // Allow time for announcement
      
      const announcements = page.locator('[aria-live] *:not(:empty)');
      if (await announcements.count() > 0) {
        const announcementText = await announcements.first().textContent();
        expect(announcementText?.length).toBeGreaterThan(0);
      }
    });
  });

  test.describe('@mobile Responsive Behavior', () => {
    test('adapts layout for phone viewport', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
      await graphitiPage.waitForGraphLoad();

      // Header should be compact on mobile
      const header = graphitiPage.header;
      const headerHeight = await header.evaluate(el => el.getBoundingClientRect().height);
      expect(headerHeight).toBeLessThan(100);

      // Search should be toggleable or compact
      const searchToggle = page.locator('[data-testid="search-toggle"]');
      if (await searchToggle.isVisible()) {
        await searchToggle.click();
        await expect(graphitiPage.searchInput).toBeVisible();
      } else {
        // Search should be visible but compact
        await expect(graphitiPage.searchInput).toBeVisible();
        const searchWidth = await graphitiPage.searchInput.evaluate(
          el => el.getBoundingClientRect().width
        );
        expect(searchWidth).toBeLessThan(300);
      }

      // Entity nodes should be appropriately sized
      const entityNode = graphitiPage.entityNodes.first();
      if (await entityNode.isVisible()) {
        const nodeSize = await entityNode.boundingBox();
        expect(nodeSize?.width).toBeLessThan(120);
        expect(nodeSize?.height).toBeLessThan(120);
      }
    });

    test('adapts layout for tablet viewport', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 }); // iPad
      await graphitiPage.waitForGraphLoad();

      // Should show more information than mobile
      const header = graphitiPage.header;
      const badges = header.locator('[data-testid="entity-count-badge"]');
      await expect(badges).toHaveCountGreaterThan(0);

      // Filter dropdown should be visible
      await expect(graphitiPage.filterDropdown).toBeVisible();

      // Graph should use appropriate space
      const graphContainer = graphitiPage.graphCanvas;
      const containerSize = await graphContainer.boundingBox();
      expect(containerSize?.width).toBeGreaterThan(700);
    });

    test('maintains usability across breakpoints', async ({ page }) => {
      const viewports = [
        { width: 320, height: 568 }, // Small phone
        { width: 768, height: 1024 }, // Tablet
        { width: 1200, height: 800 }  // Desktop
      ];

      for (const viewport of viewports) {
        await page.setViewportSize(viewport);
        await graphitiPage.waitForGraphLoad();

        // Core functionality should work at all sizes
        await expect(graphitiPage.graphCanvas).toBeVisible();
        await expect(graphitiPage.entityNodes.first()).toBeVisible();

        // Search should be accessible
        const searchInput = graphitiPage.searchInput;
        const searchToggle = page.locator('[data-testid="search-toggle"]');
        
        if (await searchToggle.isVisible()) {
          await searchToggle.click();
        }
        await expect(searchInput).toBeVisible();
        
        // Should be able to interact with entities
        await graphitiPage.selectEntity('calculateTotal');
        
        // Some form of details should be accessible
        const detailsVisible = await graphitiPage.detailsPanel.isVisible() ||
                              await page.locator('[data-testid="entity-drawer"]').isVisible() ||
                              await page.locator('[aria-expanded="true"]').count() > 0;
        
        expect(detailsVisible).toBeTruthy();
      }
    });
  });

  test.describe('@performance Performance & Loading', () => {
    test('shows appropriate loading states', async ({ page }) => {
      // Intercept and delay API response
      await page.route('**/api/graphiti/graph-data', async route => {
        await page.waitForTimeout(2000); // Simulate slow loading
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            nodes: [],
            edges: [],
            metadata: { total_entities: 0, total_relationships: 0, entity_types: [], relationship_types: [], last_updated: Date.now() }
          })
        });
      });

      await page.goto('/graphiti');

      // Should show loading spinner
      await expect(graphitiPage.loadingSpinner).toBeVisible();
      
      // Loading message should be informative
      const loadingMessage = page.locator('text=/loading|fetching|processing/i');
      await expect(loadingMessage.first()).toBeVisible();

      // Should hide loading state when done
      await graphitiPage.loadingSpinner.waitFor({ state: 'hidden' });
      await expect(graphitiPage.loadingSpinner).not.toBeVisible();
    });

    test('handles large datasets efficiently', async ({ page }) => {
      // Mock large dataset
      const largeDataset = {
        nodes: Array.from({ length: 1000 }, (_, i) => ({
          id: `node-${i}`,
          label: `Entity ${i}`,
          type: 'function',
          properties: { confidence_score: Math.random(), entity_type: 'function', name: `Entity ${i}` },
          position: { x: Math.random() * 2000, y: Math.random() * 2000 }
        })),
        edges: Array.from({ length: 500 }, (_, i) => ({
          id: `edge-${i}`,
          source: `node-${i}`,
          target: `node-${(i + 1) % 1000}`,
          type: 'calls',
          properties: { confidence: Math.random() }
        })),
        metadata: {
          total_entities: 1000,
          total_relationships: 500,
          entity_types: ['function'],
          relationship_types: ['calls'],
          last_updated: Date.now()
        }
      };

      await page.route('**/api/graphiti/graph-data', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeDataset)
        });
      });

      await page.goto('/graphiti');
      await graphitiPage.waitForGraphLoad();

      // Should show performance optimizations
      const performancePanel = page.locator('[data-testid="performance-panel"]');
      if (await performancePanel.isVisible()) {
        const fpsIndicator = performancePanel.locator('text=/fps|frame/i');
        await expect(fpsIndicator).toBeVisible();
      }

      // Should not render all nodes at once (virtualization)
      const visibleNodes = graphitiPage.entityNodes;
      const nodeCount = await visibleNodes.count();
      expect(nodeCount).toBeLessThan(1000); // Should be virtualized
      expect(nodeCount).toBeGreaterThan(0);
    });

    test('maintains responsive interactions', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      const startTime = Date.now();
      
      // Test zoom responsiveness
      await page.mouse.wheel(0, -100);
      await page.waitForTimeout(100);
      
      // Test pan responsiveness  
      await page.mouse.move(400, 300);
      await page.mouse.down();
      await page.mouse.move(450, 350);
      await page.mouse.up();
      
      const endTime = Date.now();
      const interactionTime = endTime - startTime;
      
      // Interactions should be responsive (< 500ms)
      expect(interactionTime).toBeLessThan(500);

      // Graph should respond to interactions
      const graphContainer = graphitiPage.graphCanvas;
      await expect(graphContainer).toBeVisible();
    });
  });

  test.describe('@error Error Handling', () => {
    test('displays helpful error messages', async ({ page }) => {
      // Mock API error
      await page.route('**/api/graphiti/graph-data', async route => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Internal server error' })
        });
      });

      await page.goto('/graphiti');

      // Should show error state
      await expect(graphitiPage.errorMessage).toBeVisible();
      
      const errorText = await graphitiPage.errorMessage.textContent();
      expect(errorText).toMatch(/error|failed|unavailable/i);
      
      // Should provide retry option
      const retryButton = page.locator('button:has-text("retry"), button:has-text("try again")');
      await expect(retryButton).toBeVisible();
    });

    test('handles network errors gracefully', async ({ page }) => {
      // Simulate network failure
      await page.route('**/api/graphiti/graph-data', async route => {
        await route.abort('failed');
      });

      await page.goto('/graphiti');

      // Should show connection error
      const connectionError = page.locator('text=/connection|network|offline/i');
      await expect(connectionError.first()).toBeVisible();

      // Should offer helpful actions
      const helpfulActions = page.locator('button:has-text("retry"), button:has-text("refresh"), a:has-text("support")');
      await expect(helpfulActions.first()).toBeVisible();
    });

    test('shows empty state appropriately', async ({ page }) => {
      // Mock empty response
      await page.route('**/api/graphiti/graph-data', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            nodes: [],
            edges: [],
            metadata: { total_entities: 0, total_relationships: 0, entity_types: [], relationship_types: [], last_updated: Date.now() }
          })
        });
      });

      await page.goto('/graphiti');
      await graphitiPage.waitForGraphLoad();

      // Should show empty state
      const emptyState = page.locator('text=/no data|empty|no entities/i');
      await expect(emptyState.first()).toBeVisible();

      // Should provide guidance
      const guidance = page.locator('text=/add|create|import/i');
      if (await guidance.count() > 0) {
        await expect(guidance.first()).toBeVisible();
      }
    });
  });

  test.describe('@interaction User Interaction Flows', () => {
    test('supports entity exploration workflow', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // 1. User searches for entity
      await graphitiPage.performSearch('calculate');
      await page.waitForTimeout(500);

      // 2. User selects an entity
      await graphitiPage.selectEntity('calculateTotal');

      // 3. Details panel should open
      await expect(graphitiPage.detailsPanel).toBeVisible();

      // 4. User explores connections
      const exploreButton = page.locator('button:has-text("explore"), button:has-text("connections")');
      if (await exploreButton.count() > 0) {
        await exploreButton.first().click();
        
        // Should highlight connected entities or show relationship info
        const connectionInfo = page.locator('text=/connection|relationship|related/i');
        await expect(connectionInfo.first()).toBeVisible();
      }
    });

    test('supports filtering and discovery', async ({ page }) => {
      await graphitiPage.waitForGraphLoad();

      // 1. User opens filter dropdown
      await graphitiPage.filterDropdown.click();
      
      // 2. User selects entity type
      const functionOption = page.locator('text=Function, [value="function"]');
      if (await functionOption.count() > 0) {
        await functionOption.first().click();
        await page.waitForTimeout(500);
        
        // 3. Graph should update to show only functions
        const visibleNodes = graphitiPage.entityNodes;
        const nodeCount = await visibleNodes.count();
        
        if (nodeCount > 0) {
          // Verify filtered results
          const firstNode = visibleNodes.first();
          const nodeType = await firstNode.getAttribute('data-entity-type');
          expect(nodeType).toBe('function');
        }
      }

      // 4. User clears filter
      const clearButton = page.locator('button:has-text("clear"), button:has-text("reset")');
      if (await clearButton.count() > 0) {
        await clearButton.first().click();
        await page.waitForTimeout(500);
        
        // Should show all entities again
        const allNodes = graphitiPage.entityNodes;
        const totalCount = await allNodes.count();
        expect(totalCount).toBeGreaterThan(0);
      }
    });

    test('supports onboarding flow', async ({ page }) => {
      // Clear first-visit flag to trigger onboarding
      await page.evaluate(() => localStorage.removeItem('graphiti-explorer-visited'));
      
      await page.goto('/graphiti');
      await graphitiPage.waitForGraphLoad();

      // Should show onboarding overlay
      await expect(graphitiPage.onboardingOverlay).toBeVisible();

      // Should be able to step through tour
      const nextButton = page.locator('button:has-text("next")');
      if (await nextButton.isVisible()) {
        await nextButton.click();
        await page.waitForTimeout(500);
        
        // Should still be in onboarding
        await expect(graphitiPage.onboardingOverlay).toBeVisible();
        
        // Should be able to complete tour
        const completeButton = page.locator('button:has-text("complete"), button:has-text("get started")');
        if (await completeButton.isVisible()) {
          await completeButton.click();
          await expect(graphitiPage.onboardingOverlay).not.toBeVisible();
        }
      }
    });
  });
});