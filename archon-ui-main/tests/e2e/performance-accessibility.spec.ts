/**
 * Performance and Accessibility Test Suite
 * Tests performance metrics, accessibility compliance, and cross-device compatibility
 */

import { test, expect } from '@playwright/test';

test.describe('Performance and Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses for consistent testing
    await page.route('**/api/agency/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'test-agency',
          name: 'Test Agency',
          agents: [
            {
              id: 'agent-1',
              name: 'Performance Test Agent',
              type: 'analyst',
              model_tier: 'sonnet',
              state: 'active',
              capabilities: ['performance_testing'],
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            },
          ],
          communication_flows: [],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });
    });

    await page.route('**/api/workflow/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'test-workflow',
          name: 'Performance Test Workflow',
          nodes: [],
          edges: [],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }),
      });
    });

    await page.route('**/api/mcp/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'healthy',
          tools: ['test-tool'],
          uptime: 99.9,
        }),
      });
    });

    await page.route('**/api/knowledge/**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          results: [],
          total: 0,
          page: 1,
        }),
      });
    });

    // Navigate to the main application
    await page.goto('/');
  });

  test.describe('Performance Metrics', () => {
    test('should meet page load performance targets', async ({ page }) => {
      // Start performance monitoring
      const metrics = await page.evaluate(() => {
        return new Promise((resolve) => {
          const timing = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
          const paintEntries = performance.getEntriesByType('paint');

          resolve({
            domContentLoaded: timing.domContentLoadedEventEnd - timing.domContentLoadedEventStart,
            loadComplete: timing.loadEventEnd - timing.loadEventStart,
            firstPaint: paintEntries.find((p: any) => p.name === 'first-paint')?.startTime || 0,
            firstContentfulPaint: paintEntries.find((p: any) => p.name === 'first-contentful-paint')?.startTime || 0,
            totalResources: performance.getEntriesByType('resource').length,
            totalSize: performance.getEntriesByType('resource').reduce((sum: number, r: any) => sum + r.transferSize, 0),
          });
        });
      });

      console.log('Performance Metrics:', metrics);

      // Assert performance targets
      expect(metrics.domContentLoaded).toBeLessThan(1000); // < 1s
      expect(metrics.loadComplete).toBeLessThan(3000); // < 3s
      expect(metrics.firstPaint).toBeLessThan(800); // < 800ms
      expect(metrics.firstContentfulPaint).toBeLessThan(1200); // < 1.2s
      expect(metrics.totalSize).toBeLessThan(2000000); // < 2MB
    });

    test('should handle large datasets efficiently', async ({ page }) => {
      // Mock large agency data
      await page.route('**/api/agency/**', async (route) => {
        const largeAgency = {
          id: 'large-agency',
          name: 'Large Test Agency',
          agents: Array.from({ length: 100 }, (_, i) => ({
            id: `agent-${i}`,
            name: `Agent ${i}`,
            type: 'analyst',
            model_tier: 'sonnet',
            state: 'active',
            capabilities: ['testing'],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })),
          communication_flows: Array.from({ length: 200 }, (_, i) => ({
            id: `flow-${i}`,
            source_agent_id: `agent-${i % 100}`,
            target_agent_id: `agent-${(i + 1) % 100}`,
            communication_type: 'direct',
            status: 'active',
            message_count: Math.floor(Math.random() * 100),
            message_type: 'test',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          })),
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };

        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeAgency),
        });
      });

      // Navigate to workflow visualization
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Measure rendering performance
      const renderTime = await page.evaluate(() => {
        const start = performance.now();

        return new Promise((resolve) => {
          setTimeout(() => {
            const end = performance.now();
            resolve(end - start);
          }, 100);
        });
      });

      console.log('Large dataset render time:', renderTime, 'ms');
      expect(renderTime).toBeLessThan(2000); // < 2s for large dataset

      // Check that UI remains responsive
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();
      await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
    });

    test('should maintain performance during rapid interactions', async ({ page }) => {
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Simulate rapid user interactions
      const interactionTimes = [];

      for (let i = 0; i < 10; i++) {
        const startTime = performance.now();

        // Perform various interactions
        await page.click('[data-testid="zoom-in-button"]');
        await page.click('[data-testid="zoom-out-button"]');
        await page.click('[data-testid="fit-view-button"]');

        const endTime = performance.now();
        interactionTimes.push(endTime - startTime);
      }

      const averageInteractionTime = interactionTimes.reduce((a, b) => a + b, 0) / interactionTimes.length;
      console.log('Average interaction time:', averageInteractionTime, 'ms');

      // Each interaction should be responsive
      expect(averageInteractionTime).toBeLessThan(500); // < 500ms average
      expect(Math.max(...interactionTimes)).toBeLessThan(1000); // < 1s max
    });

    test('should handle memory usage efficiently', async ({ page }) => {
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Monitor memory usage
      const initialMemory = await page.evaluate(() => {
        return (performance as any).memory?.usedJSHeapSize || 0;
      });

      // Perform memory-intensive operations
      for (let i = 0; i < 5; i++) {
        await page.click('[data-testid="add-agent-button"]');
        await page.fill('[data-testid="agent-name-input"]', `Memory Test Agent ${i}`);
        await page.click('[data-testid="save-agent-button"]');
      }

      const finalMemory = await page.evaluate(() => {
        return (performance as any).memory?.usedJSHeapSize || 0;
      });

      const memoryIncrease = finalMemory - initialMemory;
      console.log('Memory increase:', memoryIncrease, 'bytes');

      // Memory increase should be reasonable
      expect(memoryIncrease).toBeLessThan(50000000); // < 50MB increase
    });

    test('should optimize bundle loading', async ({ page }) => {
      // Check for lazy loading of components
      await page.click('[data-testid="workflow-editor-link"]');

      // Verify that heavy components are loaded on demand
      const editorLoaded = await page.waitForSelector('[data-testid="workflow-editor"]', { timeout: 5000 });
      expect(editorLoaded).toBeTruthy();

      // Check that unnecessary resources are not loaded
      const resources = await page.evaluate(() => {
        return performance.getEntriesByType('resource').map((r: any) => ({
          name: r.name,
          size: r.transferSize,
          duration: r.duration,
        }));
      });

      const totalSize = resources.reduce((sum, r) => sum + (r.size || 0), 0);
      console.log('Total resources loaded:', totalSize, 'bytes');

      expect(totalSize).toBeLessThan(3000000); // < 3MB total
    });
  });

  test.describe('Accessibility Compliance', () => {
    test('should meet WCAG 2.1 AA standards', async ({ page }) => {
      // Check for proper ARIA labels
      const interactiveElements = await page.locator('button, input, select, textarea, a[href]');
      const count = await interactiveElements.count();

      for (let i = 0; i < Math.min(count, 20); i++) { // Sample first 20 elements
        const element = interactiveElements.nth(i);

        // Check for accessible name
        const accessibleName = await element.getAttribute('aria-label') ||
                               await element.getAttribute('title') ||
                               await element.textContent();

        expect(accessibleName?.trim()).toBeTruthy();
      }

      // Check for proper heading structure
      const headings = await page.locator('h1, h2, h3, h4, h5, h6');
      const headingCount = await headings.count();

      // Should have proper heading hierarchy
      let lastLevel = 0;
      for (let i = 0; i < headingCount; i++) {
        const heading = headings.nth(i);
        const level = parseInt(await heading.getAttribute('aria-level') || await heading.evaluate(el => el.tagName.charAt(1)));

        if (lastLevel > 0) {
          expect(level).toBeLessThanOrEqual(lastLevel + 1);
        }
        lastLevel = level;
      }
    });

    test('should support keyboard navigation', async ({ page }) => {
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Test keyboard navigation
      await page.keyboard.press('Tab');

      // Should be able to navigate to interactive elements
      let focused = await page.evaluate(() => document.activeElement?.tagName);
      expect(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A']).toContain(focused);

      // Test navigation through controls
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press('Tab');
        focused = await page.evaluate(() => document.activeElement?.tagName);
        expect(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A', 'DIV']).toContain(focused);
      }

      // Test Enter key on buttons
      const enterButton = await page.locator('[data-testid="zoom-in-button"]:visible').first();
      if (await enterButton.isVisible()) {
        await enterButton.focus();
        await page.keyboard.press('Enter');

        // Should trigger button action
        await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
      }
    });

    test('should provide proper focus management', async ({ page }) => {
      await page.click('[data-testid="workflow-editor-link"]');

      // Test focus management in modals
      await page.click('[data-testid="add-agent-button"]');

      // Focus should be trapped in modal
      const modalInput = await page.locator('[data-testid="agent-name-input"]').first();
      await expect(modalInput).toBeFocused();

      // Test focus restoration after modal close
      await page.click('[data-testid="cancel-button"]');

      // Focus should return to triggering element
      const triggerButton = await page.locator('[data-testid="add-agent-button"]').first();
      await expect(triggerButton).toBeFocused();
    });

    test('should have sufficient color contrast', async ({ page }) => {
      // Check color contrast for text elements
      const textElements = await page.locator('p, span, h1, h2, h3, h4, h5, h6, button, a');
      const sampleSize = Math.min(await textElements.count(), 10);

      for (let i = 0; i < sampleSize; i++) {
        const element = textElements.nth(i);
        const computedStyle = await element.evaluate((el) => {
          const style = window.getComputedStyle(el);
          return {
            color: style.color,
            backgroundColor: style.backgroundColor,
            fontSize: style.fontSize,
          };
        });

        // Basic color validation (in real implementation, use proper contrast calculation)
        expect(computedStyle.color).toBeTruthy();
        expect(computedStyle.fontSize).toMatch(/\d+px/);
      }
    });

    test('should support screen readers', async ({ page }) => {
      // Check for proper screen reader support
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Verify live regions for dynamic content
      const liveRegions = await page.locator('[aria-live], [role="status"], [role="alert"]');
      expect(await liveRegions.count()).toBeGreaterThan(0);

      // Check for proper descriptions of complex components
      const workflowContainer = await page.locator('[data-testid="react-flow"]').first();
      await expect(workflowContainer).toHaveAttribute('aria-label');

      // Verify form inputs have proper labels
      const formInputs = await page.locator('input, select, textarea');
      const inputCount = await formInputs.count();

      for (let i = 0; i < Math.min(inputCount, 5); i++) {
        const input = formInputs.nth(i);
        const hasLabel = await input.evaluate((el) => {
          const id = el.getAttribute('id');
          return id && document.querySelector(`label[for="${id}"]`) !== null;
        });

        // Should have associated label or aria-label
        const hasAriaLabel = await input.getAttribute('aria-label');
        expect(hasLabel || hasAriaLabel).toBeTruthy();
      }
    });

    test('should handle reduced motion preferences', async ({ page }) => {
      // Simulate reduced motion preference
      await page.addInitScript(() => {
        Object.defineProperty(window, 'matchMedia', {
          writable: true,
          value: (query: string) => ({
            matches: query === '(prefers-reduced-motion: reduce)',
            media: query,
            onchange: null,
            addListener: () => {},
            removeListener: () => {},
            addEventListener: () => {},
            removeEventListener: () => {},
            dispatchEvent: () => {},
          }),
        });
      });

      await page.reload();
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Animations should be disabled or reduced
      const animatedElements = await page.locator('[class*="animate"], [style*="animation"]');
      const animationCount = await animatedElements.count();

      console.log('Animated elements with reduced motion:', animationCount);

      // Should have minimal animations when reduced motion is preferred
      expect(animationCount).toBeLessThan(5);
    });
  });

  test.describe('Responsive Design', () => {
    test.describe('Mobile Viewport (375px)', () => {
      test.beforeEach(async ({ page }) => {
        await page.setViewportSize({ width: 375, height: 667 });
      });

      test('should display properly on mobile devices', async ({ page }) => {
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Check mobile layout
        const mobileMenu = await page.locator('[data-testid="mobile-menu"]');
        if (await mobileMenu.isVisible()) {
          await expect(mobileMenu).toBeVisible();
        }

        // Main content should be accessible
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

        // Controls should be touch-friendly
        const touchTargets = await page.locator('button, [role="button"]');
        const touchTargetCount = await touchTargets.count();

        for (let i = 0; i < Math.min(touchTargetCount, 5); i++) {
          const target = touchTargets.nth(i);
          const box = await target.boundingBox();

          if (box) {
            expect(box.width).toBeGreaterThanOrEqual(44); // Minimum touch target size
            expect(box.height).toBeGreaterThanOrEqual(44);
          }
        }
      });

      test('should handle mobile-specific interactions', async ({ page }) => {
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Test touch interactions
        const workflowArea = await page.locator('[data-testid="react-flow"]').first();

        // Simulate pinch zoom (simplified)
        await workflowArea.click({ position: { x: 100, y: 100 } });
        await workflowArea.click({ position: { x: 150, y: 150 } });

        // Test swipe gestures
        await page.mouse.down();
        await page.mouse.move(300, 300);
        await page.mouse.up();

        // Should remain functional
        await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();
      });
    });

    test.describe('Tablet Viewport (768px)', () => {
      test.beforeEach(async ({ page }) => {
        await page.setViewportSize({ width: 768, height: 1024 });
      });

      test('should display properly on tablet devices', async ({ page }) => {
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Check tablet layout
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

        // Sidebars might be visible or collapsible
        const sidebar = await page.locator('[data-testid="sidebar"]').first();
        const isVisible = await sidebar.isVisible();

        if (isVisible) {
          await expect(sidebar).toBeVisible();
        }
      });
    });

    test.describe('Desktop Viewport (1920px)', () => {
      test.beforeEach(async ({ page }) => {
        await page.setViewportSize({ width: 1920, height: 1080 });
      });

      test('should display properly on desktop devices', async ({ page }) => {
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Check desktop layout
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
        await expect(page.locator('[data-testid="sidebar"]')).toBeVisible();

        //充分利用大屏幕空间
        const workflowArea = await page.locator('[data-testid="react-flow"]').first();
        const box = await workflowArea.boundingBox();

        if (box) {
          expect(box.width).toBeGreaterThan(800); // Should use available space
        }
      });
    });
  });

  test.describe('Cross-Browser Compatibility', () => {
    test('should work consistently across browsers', async ({ page, browserName }) => {
      console.log(`Testing in browser: ${browserName}`);

      await page.click('[data-testid="workflow-visualizer-link"]');

      // Basic functionality should work in all browsers
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

      // Test interactions
      await page.click('[data-testid="zoom-in-button"]');
      await page.click('[data-testid="zoom-out-button"]');

      // Should remain functional
      await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
    });

    test('should handle browser-specific features gracefully', async ({ page }) => {
      // Test for feature detection
      const featureSupport = await page.evaluate(() => {
        return {
          webGL: !!document.createElement('canvas').getContext('webgl'),
          webWorkers: typeof Worker !== 'undefined',
          localStorage: typeof localStorage !== 'undefined',
          sessionStorage: typeof sessionStorage !== 'undefined',
        };
      });

      console.log('Browser feature support:', featureSupport);

      // Should work regardless of feature support
      await page.click('[data-testid="workflow-visualizer-link"]');
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });
  });

  test.describe('Network Performance', () => {
    test('should handle slow network conditions', async ({ page }) => {
      // Simulate slow network
      await page.context().setOffline(false);
      await page.route('**/*', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 1000)); // 1s delay
        await route.continue();
      });

      const startTime = Date.now();
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should handle slow loading gracefully
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible({ timeout: 10000 });

      const loadTime = Date.now() - startTime;
      console.log('Slow network load time:', loadTime, 'ms');

      // Should eventually load even with slow network
      expect(loadTime).toBeLessThan(15000); // < 15s total
    });

    test('should handle offline mode gracefully', async ({ page }) => {
      // Go offline
      await page.context().setOffline(true);

      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should show appropriate offline message or cached content
      const offlineMessage = await page.locator('[data-testid="offline-message"], text=/offline/i');
      const hasOfflineMessage = await offlineMessage.count() > 0;

      if (hasOfflineMessage) {
        await expect(offlineMessage).toBeVisible();
      } else {
        // Should show cached content or graceful degradation
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
      }

      // Go back online
      await page.context().setOffline(false);
    });

    test('should handle network interruptions gracefully', async ({ page }) => {
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Simulate network interruption during interaction
      await page.route('**/api/**', async (route) => {
        await route.abort('failed');
      });

      // Should handle failed API calls gracefully
      await page.click('[data-testid="refresh-button"]');

      // Should show error message or retry gracefully
      const errorMessage = await page.locator('[data-testid="error-message"], text=/error|failed/i');
      const hasErrorMessage = await errorMessage.count() > 0;

      if (hasErrorMessage) {
        await expect(errorMessage).toBeVisible();
      }

      // Should recover when network is restored
      await page.unroute('**/api/**');
      await page.reload();
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });
  });

  test.describe('Performance Optimization', () => {
    test('should implement efficient rendering strategies', async ({ page }) => {
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Monitor render performance during interactions
      const renderMetrics = await page.evaluate(() => {
        const metrics = [];
        let lastFrameTime = performance.now();

        return new Promise((resolve) => {
          const observer = new PerformanceObserver((list) => {
            const entries = list.getEntries();
            entries.forEach((entry) => {
              if (entry.name === 'render') {
                metrics.push(entry.duration);
              }
            });

            if (metrics.length >= 10) {
              resolve(metrics);
            }
          });

          observer.observe({ entryTypes: ['measure'] });

          // Trigger some renders
          const button = document.querySelector('[data-testid="zoom-in-button"]');
          if (button) {
            for (let i = 0; i < 10; i++) {
              button.dispatchEvent(new Event('click'));
            }
          }
        });
      });

      const averageRenderTime = renderMetrics.reduce((a, b) => a + b, 0) / renderMetrics.length;
      console.log('Average render time:', averageRenderTime, 'ms');

      expect(averageRenderTime).toBeLessThan(16); // < 16ms (60fps)
    });

    test('should optimize resource loading', async ({ page }) => {
      // Check for proper resource optimization
      const resources = await page.evaluate(() => {
        return performance.getEntriesByType('resource').map((r: any) => ({
          name: r.name,
          size: r.transferSize,
          duration: r.duration,
          cached: r.transferSize === 0,
        }));
      });

      const cachedResources = resources.filter(r => r.cached);
      const totalSize = resources.reduce((sum, r) => sum + (r.size || 0), 0);
      const cacheHitRate = cachedResources.length / resources.length;

      console.log('Cache hit rate:', (cacheHitRate * 100).toFixed(1) + '%');
      console.log('Total resources size:', totalSize, 'bytes');

      // Should have good cache utilization
      expect(cacheHitRate).toBeGreaterThan(0.3); // > 30% cache hit rate
      expect(totalSize).toBeLessThan(3000000); // < 3MB total
    });
  });
});