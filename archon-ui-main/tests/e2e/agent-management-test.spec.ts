import { test, expect } from '@playwright/test';

test.describe('Agent Management Page', () => {
  test('should load agent management page without socket errors', async ({ page }) => {
    // Track console errors
    const consoleErrors: string[] = [];
    const socketErrors: string[] = [];
    
    page.on('console', (message) => {
      if (message.type() === 'error') {
        const errorText = message.text();
        consoleErrors.push(errorText);
        
        // Check for socket-related errors
        if (errorText.toLowerCase().includes('socket') || 
            errorText.toLowerCase().includes('websocket') ||
            errorText.toLowerCase().includes('socketio')) {
          socketErrors.push(errorText);
        }
      }
    });

    // Navigate to the agent management page
    await page.goto('http://localhost:3740/agents');
    
    // Wait for the page to load completely
    await page.waitForLoadState('networkidle');
    
    // Wait a bit more to ensure any async socket connections are established
    await page.waitForTimeout(3000);
    
    // Take a screenshot for verification
    await page.screenshot({ 
      path: 'tests/e2e/screenshots/agent-management-page.png',
      fullPage: true 
    });
    
    // Check that the page loaded successfully
    await expect(page).toHaveTitle(/Archon/);
    
    // Check for the agents page content (you may need to adjust this selector based on your actual page structure)
    const agentPageContent = page.locator('body');
    await expect(agentPageContent).toBeVisible();
    
    // Report any console errors
    if (consoleErrors.length > 0) {
      console.log('Console errors found:');
      consoleErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error}`);
      });
    }
    
    // Report socket-specific errors
    if (socketErrors.length > 0) {
      console.log('Socket errors found:');
      socketErrors.forEach((error, index) => {
        console.log(`${index + 1}. ${error}`);
      });
    }
    
    // Assert no socket errors (this will fail the test if socket errors are found)
    expect(socketErrors).toHaveLength(0);
    
    console.log(`Total console errors: ${consoleErrors.length}`);
    console.log(`Socket-related errors: ${socketErrors.length}`);
    console.log('Agent management page loaded successfully!');
  });
});