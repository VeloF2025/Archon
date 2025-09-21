import { test, expect } from '@playwright/test';

test.describe('Socket Bug Fix Test', () => {
  test('should not have socket.on errors on agent management page', async ({ page }) => {
    const consoleErrors: string[] = [];
    
    // Capture console errors
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log('Console Error:', msg.text());
      }
    });

    // Capture page errors
    page.on('pageerror', error => {
      console.log('Page Error:', error.message);
      consoleErrors.push(error.message);
    });

    // Navigate to the agent management page
    await page.goto('http://localhost:3739/agents');
    
    // Wait for page to load
    await page.waitForTimeout(5000);
    
    // Check if there are socket-related errors
    const socketErrors = consoleErrors.filter(error => 
      error.includes('socket.on is not a function') || 
      error.includes('socket.off is not a function') ||
      error.includes('socket.on') ||
      error.toLowerCase().includes('socket')
    );
    
    console.log('All console errors:', consoleErrors);
    console.log('Socket-related errors:', socketErrors);
    
    // Check if the page loaded successfully
    const title = await page.title();
    console.log('Page title:', title);
    
    // Take a screenshot for debugging
    await page.screenshot({ path: 'socket-bug-test.png', fullPage: true });
    
    // Report findings - should have no socket errors
    expect(socketErrors.length).toBe(0);
  });
});