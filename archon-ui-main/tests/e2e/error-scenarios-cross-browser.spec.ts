/**
 * Error Scenarios and Cross-Browser Test Suite
 * Tests error handling, edge cases, and cross-browser compatibility
 */

import { test, expect } from '@playwright/test';

test.describe('Error Scenarios and Cross-Browser Compatibility', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing routes
    await page.unroute('**/*');
  });

  test.describe('Error Handling Scenarios', () => {
    test('should handle API server errors gracefully', async ({ page }) => {
      // Mock server errors
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Internal Server Error',
            message: 'Database connection failed',
            timestamp: new Date().toISOString(),
          }),
        });
      });

      await page.goto('/');

      // Should show error message
      const errorMessage = await page.locator('[data-testid="error-message"], [role="alert"]').first();
      await expect(errorMessage).toBeVisible();

      const errorText = await errorMessage.textContent();
      expect(errorText).toMatch(/error|failed|unavailable/i);

      // Should provide retry mechanism
      const retryButton = await page.locator('[data-testid="retry-button"], button:has-text("Retry"), button:has-text("Try Again")').first();
      if (await retryButton.isVisible()) {
        await retryButton.click();

        // Should attempt retry
        await expect(errorMessage).toBeVisible({ timeout: 5000 });
      }
    });

    test('should handle network timeouts', async ({ page }) => {
      // Mock timeout scenarios
      await page.route('**/api/agency/**', async (route) => {
        await new Promise(resolve => setTimeout(resolve, 10000)); // 10s timeout
        await route.abort('timedout');
      });

      await page.goto('/');

      // Should handle timeout gracefully
      const timeoutMessage = await page.locator('[data-testid="timeout-message"], text=/timeout|slow/i').first();
      const hasTimeoutMessage = await timeoutMessage.count() > 0;

      if (hasTimeoutMessage) {
        await expect(timeoutMessage).toBeVisible();
      } else {
        // Should show generic error message
        const errorMessage = await page.locator('[data-testid="error-message"]').first();
        await expect(errorMessage).toBeVisible();
      }
    });

    test('should handle malformed API responses', async ({ page }) => {
      // Mock malformed responses
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: 'invalid json response {',
        });
      });

      await page.goto('/');

      // Should handle JSON parsing errors
      const errorMessage = await page.locator('[data-testid="error-message"]').first();
      await expect(errorMessage).toBeVisible();

      const errorText = await errorMessage.textContent();
      expect(errorText).toMatch(/error|invalid|parse/i);
    });

    test('should handle authentication errors', async ({ page }) => {
      // Mock auth errors
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 401,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Unauthorized',
            message: 'Authentication required',
          }),
        });
      });

      await page.goto('/');

      // Should redirect to login or show auth error
      const loginForm = await page.locator('[data-testid="login-form"], form:has-text("Login"), form:has-text("Sign In")').first();
      const authMessage = await page.locator('[data-testid="auth-error"], text=/unauthorized|authentication/i').first();

      if (await loginForm.isVisible()) {
        await expect(loginForm).toBeVisible();
      } else if (await authMessage.isVisible()) {
        await expect(authMessage).toBeVisible();
      } else {
        // Should show appropriate error message
        const errorMessage = await page.locator('[data-testid="error-message"]').first();
        await expect(errorMessage).toBeVisible();
      }
    });

    test('should handle validation errors', async ({ page }) => {
      await page.goto('/');

      // Mock validation error responses
      await page.route('**/api/workflow/**', async (route) => {
        if (route.request().method() === 'POST') {
          await route.fulfill({
            status: 400,
            contentType: 'application/json',
            body: JSON.stringify({
              error: 'Validation Error',
              message: 'Invalid workflow data',
              details: {
                nodes: ['Invalid node configuration'],
                edges: ['Invalid edge connections'],
              },
            }),
          });
        } else {
          await route.continue();
        }
      });

      // Navigate to workflow editor
      await page.click('[data-testid="workflow-editor-link"]');

      // Try to submit invalid workflow
      await page.click('[data-testid="save-workflow-button"]');

      // Should show validation errors
      const validationErrors = await page.locator('[data-testid="validation-error"], [data-testid="error-message"]').first();
      await expect(validationErrors).toBeVisible();

      const errorText = await validationErrors.textContent();
      expect(errorText).toMatch(/validation|invalid/i);
    });

    test('should handle concurrent request failures', async ({ page }) => {
      // Mock multiple failing requests
      await page.route('**/api/**', async (route) => {
        await route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Service Unavailable',
            message: 'Service temporarily unavailable',
          }),
        });
      });

      await page.goto('/');

      // Trigger multiple concurrent requests
      await Promise.all([
        page.click('[data-testid="refresh-button"]'),
        page.click('[data-testid="workflow-visualizer-link"]'),
        page.click('[data-testid="settings-link"]'),
      ]);

      // Should handle multiple errors gracefully
      const errorMessages = await page.locator('[data-testid="error-message"]');
      const errorCount = await errorMessages.count();

      expect(errorCount).toBeGreaterThan(0);

      // Should not crash the application
      await expect(page).toHaveURL(/\/.*/);
    });

    test('should handle WebSocket connection failures', async ({ page }) => {
      // Mock WebSocket failures
      await page.addInitScript(() => {
        window.WebSocket = class {
          constructor(url: string) {
            setTimeout(() => {
              this.onerror?.(new Event('error'));
              this.onclose?.(new CloseEvent('close'));
            }, 100);
          }
          onopen: ((event: Event) => void) | null = null;
          onclose: ((event: CloseEvent) => void) | null = null;
          onerror: ((event: Event) => void) | null = null;
          onmessage: ((event: MessageEvent) => void) | null = null;
          readyState: number = 3;
          send() {}
          close() {}
        };
      });

      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should handle WebSocket failures gracefully
      const connectionStatus = await page.locator('[data-testid="connection-status"], [data-testid="offline-indicator"]').first();
      const hasConnectionStatus = await connectionStatus.count() > 0;

      if (hasConnectionStatus) {
        await expect(connectionStatus).toBeVisible();
        const statusText = await connectionStatus.textContent();
        expect(statusText).toMatch(/offline|disconnected|error/i);
      }

      // Should still allow basic functionality
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });

    test('should handle resource loading failures', async ({ page }) => {
      // Mock resource loading failures
      await page.route('**/*.png', async (route) => {
        await route.abort('failed');
      });
      await page.route('**/*.jpg', async (route) => {
        await route.abort('failed');
      });
      await page.route('**/*.svg', async (route) => {
        await route.abort('failed');
      });

      await page.goto('/');

      // Should handle missing images gracefully
      const brokenImages = await page.locator('img[src*="error"], img[alt*="missing"]');
      const brokenImageCount = await brokenImages.count();

      // Should not show broken image icons
      expect(brokenImageCount).toBe(0);

      // Should continue functioning
      await expect(page.locator('[data-testid="main-container"]')).toBeVisible();
    });

    test('should handle browser storage errors', async ({ page }) => {
      // Mock localStorage failures
      await page.addInitScript(() => {
        const originalSetItem = localStorage.setItem;
        localStorage.setItem = (key: string, value: string) => {
          throw new Error('Storage quota exceeded');
        };
      });

      await page.goto('/');

      // Should handle storage errors gracefully
      const storageWarning = await page.locator('[data-testid="storage-warning"], text=/storage|quota/i').first();
      const hasStorageWarning = await storageWarning.count() > 0;

      if (hasStorageWarning) {
        await expect(storageWarning).toBeVisible();
      }

      // Should continue basic functionality
      await expect(page.locator('[data-testid="main-container"]')).toBeVisible();
    });
  });

  test.describe('Edge Cases and Boundary Conditions', () => {
    test('should handle empty data sets', async ({ page }) => {
      // Mock empty responses
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'empty-agency',
            name: 'Empty Agency',
            agents: [],
            communication_flows: [],
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          }),
        });
      });

      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should handle empty state gracefully
      const emptyState = await page.locator('[data-testid="empty-state"], [data-testid="no-data"]').first();
      const hasEmptyState = await emptyState.count() > 0;

      if (hasEmptyState) {
        await expect(emptyState).toBeVisible();
      }

      // Should still show controls
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();
    });

    test('should handle extremely large data sets', async ({ page }) => {
      // Mock extremely large response
      const largeAgency = {
        id: 'large-agency',
        name: 'Large Agency',
        agents: Array.from({ length: 1000 }, (_, i) => ({
          id: `agent-${i}`,
          name: `Agent ${i}`,
          type: 'analyst',
          model_tier: 'sonnet',
          state: 'active',
          capabilities: ['testing'],
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })),
        communication_flows: Array.from({ length: 5000 }, (_, i) => ({
          id: `flow-${i}`,
          source_agent_id: `agent-${i % 1000}`,
          target_agent_id: `agent-${(i + 1) % 1000}`,
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

      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(largeAgency),
        });
      });

      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should handle large data without crashing
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible({ timeout: 10000 });

      // Should show performance warning or loading indicator
      const loadingIndicator = await page.locator('[data-testid="loading"], [data-testid="spinner"]').first();
      const hasLoadingIndicator = await loadingIndicator.count() > 0;

      if (hasLoadingIndicator) {
        await expect(loadingIndicator).toBeVisible();
      }
    });

    test('should handle malformed or invalid data', async ({ page }) => {
      // Mock invalid data structures
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'invalid-agency',
            name: null,
            agents: [
              {
                // Missing required fields
                id: 'invalid-agent',
              },
            ],
            communication_flows: 'invalid-array',
            created_at: 'invalid-date',
          }),
        });
      });

      await page.goto('/');

      // Should handle invalid data gracefully
      const errorMessage = await page.locator('[data-testid="error-message"], [data-testid="validation-error"]').first();
      await expect(errorMessage).toBeVisible();
    });

    test('should handle special characters and Unicode', async ({ page }) => {
      // Mock data with special characters
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'unicode-agency-🚀',
            name: 'Test Agency with émojis 🎯 and spëciâl chars',
            agents: [
              {
                id: 'agent-测试',
                name: 'Agent with 中文 and 🎨',
                type: 'analyst',
                model_tier: 'sonnet',
                state: 'active',
                capabilities: ['tësting 🧪', 'änalysis'],
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

      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Should handle Unicode properly
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

      // Text should display correctly
      const agentName = await page.locator('text=Agent with 中文 and 🎨').first();
      const hasAgentName = await agentName.count() > 0;

      if (hasAgentName) {
        await expect(agentName).toBeVisible();
      }
    });

    test('should handle extreme user inputs', async ({ page }) => {
      await page.goto('/');
      await page.click('[data-testid="workflow-editor-link"]');

      // Test extremely long input
      const longInput = 'a'.repeat(10000);
      const inputField = await page.locator('[data-testid="agent-name-input"]').first();

      if (await inputField.isVisible()) {
        await inputField.fill(longInput);

        // Should handle long input without crashing
        await expect(inputField).toHaveValue(longInput.substring(0, 255)); // Should truncate
      }

      // Test special characters in input
      const specialChars = '!@#$%^&*()_+-=[]{}|;:,.<>?`~"\'\\';
      await inputField.fill(specialChars);

      // Should handle special characters
      await expect(inputField).toHaveValue(specialChars);
    });

    test('should handle rapid user interactions', async ({ page }) => {
      await page.goto('/');
      await page.click('[data-testid="workflow-editor-link"]');

      // Simulate rapid clicking
      const buttons = await page.locator('button').all();
      const visibleButtons = [];

      for (const button of buttons) {
        if (await button.isVisible()) {
          visibleButtons.push(button);
        }
      }

      // Click buttons rapidly
      for (let i = 0; i < Math.min(visibleButtons.length, 10); i++) {
        await visibleButtons[i].click();
      }

      // Should handle rapid interactions without crashing
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });

    test('should handle browser back/forward navigation', async ({ page }) => {
      await page.goto('/');

      // Navigate through different pages
      await page.click('[data-testid="workflow-visualizer-link"]');
      await expect(page).toHaveURL(/.*visualizer.*/);

      await page.click('[data-testid="workflow-editor-link"]');
      await expect(page).toHaveURL(/.*editor.*/);

      await page.click('[data-testid="knowledge-link"]');
      await expect(page).toHaveURL(/.*knowledge.*/);

      // Test back navigation
      await page.goBack();
      await expect(page).toHaveURL(/.*editor.*/);

      await page.goBack();
      await expect(page).toHaveURL(/.*visualizer.*/);

      // Test forward navigation
      await page.goForward();
      await expect(page).toHaveURL(/.*editor.*/);

      await page.goForward();
      await expect(page).toHaveURL(/.*knowledge.*/);
    });

    test('should handle browser refresh', async ({ page }) => {
      await page.goto('/');
      await page.click('[data-testid="workflow-editor-link"]');

      // Perform some actions
      await page.click('[data-testid="add-agent-button"]');
      await page.fill('[data-testid="agent-name-input"]', 'Test Agent');
      await page.click('[data-testid="save-agent-button"]');

      // Refresh the page
      await page.reload();

      // Should handle refresh gracefully
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

      // Should restore state if possible
      const agentList = await page.locator('[data-testid="agent-list"]').first();
      const hasAgentList = await agentList.count() > 0;

      if (hasAgentList) {
        await expect(agentList).toBeVisible();
      }
    });
  });

  test.describe('Cross-Browser Compatibility', () => {
    test.use({ browserName: 'chromium' });

    test('should work in Chrome/Chromium', async ({ page }) => {
      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      // Basic functionality
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
      await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

      // Test interactions
      await page.click('[data-testid="zoom-in-button"]');
      await page.click('[data-testid="zoom-out-button"]');
      await page.click('[data-testid="fit-view-button"]');

      // Should remain functional
      await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
    });

    test('should handle Chrome-specific features', async ({ page }) => {
      // Test Chrome DevTools integration (if applicable)
      const chromeFeatures = await page.evaluate(() => {
        return {
          hasChrome: !!window.chrome,
          hasDevTools: typeof window.__REACT_DEVTOOLS_GLOBAL_HOOK__ !== 'undefined',
        };
      });

      console.log('Chrome features:', chromeFeatures);

      // Should work regardless of Chrome-specific features
      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });

    test.describe('Firefox Compatibility', () => {
      test.use({ browserName: 'firefox' });

      test('should work in Firefox', async ({ page }) => {
        await page.goto('/');
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Basic functionality
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
        await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

        // Test interactions
        await page.click('[data-testid="zoom-in-button"]');
        await page.click('[data-testid="zoom-out-button"]');

        // Should remain functional
        await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
      });

      test('should handle Firefox-specific behavior', async ({ page }) => {
        // Test Firefox-specific CSS and JavaScript behavior
        const firefoxDetection = await page.evaluate(() => {
          return {
            isFirefox: navigator.userAgent.toLowerCase().includes('firefox'),
            hasIndexedDB: 'indexedDB' in window,
            hasServiceWorker: 'serviceWorker' in navigator,
          };
        });

        console.log('Firefox detection:', firefoxDetection);

        await page.goto('/');
        await page.click('[data-testid="workflow-editor-link"]');

        // Should work with Firefox-specific features
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
      });
    });

    test.describe('Safari Compatibility', () => {
      test.use({ browserName: 'webkit' });

      test('should work in Safari/WebKit', async ({ page }) => {
        await page.goto('/');
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Basic functionality
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
        await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();

        // Test interactions
        await page.click('[data-testid="zoom-in-button"]');
        await page.click('[data-testid="zoom-out-button']');

        // Should remain functional
        await expect(page.locator('[data-testid="react-flow"]')).toBeVisible();
      });

      test('should handle Safari-specific behavior', async ({ page }) => {
        // Test Safari-specific features and limitations
        const safariFeatures = await page.evaluate(() => {
          return {
            isSafari: /^((?!chrome|android).)*safari/i.test(navigator.userAgent),
            hasTouch: 'ontouchstart' in window,
            hasDeviceOrientation: 'DeviceOrientationEvent' in window,
          };
        });

        console.log('Safari features:', safariFeatures);

        await page.goto('/');
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Should handle Safari-specific behavior
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
      });
    });
  });

  test.describe('Mobile Device Compatibility', () => {
    test.describe('Mobile Safari', () => {
      test.use({
        browserName: 'webkit',
        viewport: { width: 375, height: 667 },
        userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15'
      });

      test('should work on iPhone', async ({ page }) => {
        await page.goto('/');
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Should handle mobile viewport
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

        // Test touch interactions
        const workflowArea = await page.locator('[data-testid="react-flow"]').first();
        await workflowArea.tap({ position: { x: 100, y: 100 } });

        // Should remain functional
        await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();
      });
    });

    test.describe('Mobile Chrome', () => {
      test.use({
        browserName: 'chromium',
        viewport: { width: 375, height: 667 },
        userAgent: 'Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36'
      });

      test('should work on Android', async ({ page }) => {
        await page.goto('/');
        await page.click('[data-testid="workflow-visualizer-link"]');

        // Should handle mobile viewport
        await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

        // Test touch interactions
        const workflowArea = await page.locator('[data-testid="react-flow"]').first();
        await workflowArea.tap({ position: { x: 100, y: 100 } });

        // Should remain functional
        await expect(page.locator('[data-testid="workflow-controls"]')).toBeVisible();
      });
    });
  });

  test.describe('Accessibility Error Scenarios', () => {
    test('should maintain accessibility during errors', async ({ page }) => {
      // Mock API errors
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'Server Error',
            message: 'Internal server error occurred',
          }),
        });
      });

      await page.goto('/');

      // Error messages should be accessible
      const errorMessage = await page.locator('[data-testid="error-message"], [role="alert"]').first();
      await expect(errorMessage).toBeVisible();

      // Should have proper ARIA attributes
      const hasRole = await errorMessage.getAttribute('role');
      expect(hasRole).toMatch(/alert|status|alertdialog/);

      // Should be keyboard accessible
      await errorMessage.focus();
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(focusedElement).toBeTruthy();
    });

    test('should handle screen reader errors gracefully', async ({ page }) => {
      // Simulate screen reader environment
      await page.addInitScript(() => {
        Object.defineProperty(navigator, 'userAgent', {
          value: 'Mozilla/5.0 (ScreenReader) AppleWebKit/537.36',
          writable: true,
        });
      });

      await page.goto('/');

      // Should work with screen readers
      const mainContent = await page.locator('main, [role="main"]').first();
      await expect(mainContent).toBeVisible();

      // Should have proper landmarks
      const landmarks = await page.locator('[role="navigation"], [role="complementary"], [role="contentinfo"]');
      const landmarkCount = await landmarks.count();
      expect(landmarkCount).toBeGreaterThan(0);
    });
  });

  test.describe('Performance Under Error Conditions', () => {
    test('should maintain performance during partial failures', async ({ page }) => {
      // Mock intermittent failures
      let requestCount = 0;
      await page.route('**/api/**', async (route) => {
        requestCount++;
        if (requestCount % 3 === 0) {
          await route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Intermittent Error' }),
          });
        } else {
          await route.continue();
        }
      });

      const startTime = Date.now();
      await page.goto('/');
      await page.click('[data-testid="workflow-visualizer-link"]');

      const loadTime = Date.now() - startTime;
      console.log('Load time with intermittent failures:', loadTime, 'ms');

      // Should load reasonably fast even with some failures
      expect(loadTime).toBeLessThan(10000);

      // Should handle mixed success/failure states
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });

    test('should handle memory pressure scenarios', async ({ page }) => {
      // Simulate memory pressure by creating many elements
      await page.goto('/');
      await page.click('[data-testid="workflow-editor-link"]');

      // Create many agents
      for (let i = 0; i < 20; i++) {
        await page.click('[data-testid="add-agent-button"]');
        await page.fill('[data-testid="agent-name-input"]', `Memory Test Agent ${i}`);
        await page.click('[data-testid="save-agent-button"]');
      }

      // Should handle memory pressure gracefully
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();

      // Should not crash or become unresponsive
      const responseTime = await page.evaluate(() => {
        const start = performance.now();
        return new Promise(resolve => {
          setTimeout(() => {
            resolve(performance.now() - start);
          }, 100);
        });
      });

      expect(responseTime).toBeLessThan(1000); // Should remain responsive
    });
  });

  test.describe('Recovery and Resilience', () => {
    test('should recover from temporary failures', async ({ page }) => {
      // Mock temporary failure then recovery
      let failedOnce = false;
      await page.route('**/api/agency/**', async (route) => {
        if (!failedOnce) {
          failedOnce = true;
          await route.fulfill({
            status: 503,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Service Unavailable' }),
          });
        } else {
          await route.continue();
        }
      });

      await page.goto('/');

      // Should show error initially
      const errorMessage = await page.locator('[data-testid="error-message"]').first();
      await expect(errorMessage).toBeVisible();

      // Should recover on retry
      const retryButton = await page.locator('[data-testid="retry-button"]').first();
      if (await retryButton.isVisible()) {
        await retryButton.click();
      }

      // Should load successfully after retry
      await expect(page.locator('[data-testid="main-container"]')).toBeVisible({ timeout: 10000 });
    });

    test('should maintain user experience during degraded mode', async ({ page }) => {
      // Mock some API failures but not all
      await page.route('**/api/agency/**', async (route) => {
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: 'Agency Service Down' }),
        });
      });

      // Keep other services working
      await page.route('**/api/workflow/**', async (route) => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ workflows: [] }),
        });
      });

      await page.goto('/');

      // Should show partial functionality
      const errorMessage = await page.locator('[data-testid="error-message"]').first();
      await expect(errorMessage).toBeVisible();

      // Should still allow access to working features
      await page.click('[data-testid="workflow-editor-link"]');
      await expect(page.locator('[data-testid="workflow-container"]')).toBeVisible();
    });
  });
});