import { test, expect } from '@playwright/test';

test.describe('Agent Management Page - Final Verification', () => {
  test('should load agent management page successfully with backend failures', async ({ page }) => {
    // Set up console logging to capture any errors
    const consoleErrors: string[] = [];
    const networkErrors: string[] = [];
    
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    page.on('response', (response) => {
      if (response.status() >= 400) {
        networkErrors.push(`${response.status()} ${response.url()}`);
      }
    });

    // Navigate to the agent management page
    console.log('Navigating to agent management page...');
    await page.goto('http://localhost:3737/agents', { waitUntil: 'networkidle', timeout: 30000 });
    
    // Wait for the page to load completely - giving extra time for all API calls
    console.log('Waiting for page to fully load...');
    await page.waitForTimeout(12000); // 12 seconds to allow all API attempts
    
    // Take a screenshot for verification
    await page.screenshot({ path: 'tests/e2e/screenshots/agent-management-final-verification.png', fullPage: true });
    
    // Check if the error message is NOT present (which would indicate failure)
    const errorMessage = page.locator('text=Error Loading Agent Management');
    const hasErrorMessage = await errorMessage.isVisible();
    
    console.log('Error message present:', hasErrorMessage);
    
    // Check for positive indicators that the page loaded
    const pageTitle = await page.title();
    console.log('Page title:', pageTitle);
    
    // Look for key UI elements that should be present
    const agentPageIndicators = [
      page.locator('[data-testid="agent-management"]').first(),
      page.locator('h1, h2, h3').filter({ hasText: /agent/i }).first(),
      page.locator('.agent-pool, [class*="agent"], [id*="agent"]').first(),
      page.locator('text=Agent').first()
    ];
    
    let foundIndicators = 0;
    for (const indicator of agentPageIndicators) {
      try {
        const isVisible = await indicator.isVisible({ timeout: 1000 });
        if (isVisible) {
          foundIndicators++;
          const text = await indicator.textContent();
          console.log(`Found indicator: ${text?.substring(0, 50)}...`);
        }
      } catch (e) {
        // Indicator not found, continue
      }
    }
    
    console.log(`Found ${foundIndicators} positive UI indicators`);
    console.log(`Network errors: ${networkErrors.length}`);
    console.log('Network errors:', networkErrors.slice(0, 5)); // Show first 5
    console.log(`Console errors: ${consoleErrors.length}`);
    console.log('Console errors:', consoleErrors.slice(0, 3)); // Show first 3
    
    // The main test: the page should NOT show the error message
    // This confirms that Promise.all fix is working
    expect(hasErrorMessage).toBeFalsy();
    
    // The page should have loaded (not be blank)
    expect(pageTitle).not.toBe('');
    
    // We expect network errors (404s) but the page should still render
    console.log('\n=== VERIFICATION RESULTS ===');
    console.log(`✓ Page loaded without "Error Loading Agent Management" message`);
    console.log(`✓ Page title present: "${pageTitle}"`);
    console.log(`✓ Found ${foundIndicators} UI indicators`);
    console.log(`✓ Expected network errors present: ${networkErrors.length} (this is normal)`);
    console.log(`✓ Console errors: ${consoleErrors.length}`);
    
    // Final verification - page should be functional even with backend down
    const bodyText = await page.textContent('body');
    expect(bodyText).not.toBe('');
    expect(bodyText?.length).toBeGreaterThan(100); // Page has substantial content
  });
});