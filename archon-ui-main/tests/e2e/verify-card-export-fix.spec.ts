import { test, expect } from '@playwright/test';

test('Verify CardDescription export error is fixed', async ({ page }) => {
  // Array to collect console messages
  const consoleMessages: string[] = [];
  const consoleErrors: string[] = [];

  // Listen for console messages
  page.on('console', msg => {
    const text = msg.text();
    consoleMessages.push(`${msg.type()}: ${text}`);
    
    if (msg.type() === 'error') {
      consoleErrors.push(text);
    }
  });

  // Listen for page errors (uncaught exceptions)
  page.on('pageerror', error => {
    consoleErrors.push(`Page Error: ${error.message}`);
  });

  try {
    // Navigate to the application
    console.log('Navigating to http://localhost:3737/...');
    await page.goto('http://localhost:3737/', { 
      waitUntil: 'networkidle',
      timeout: 30000
    });

    // Wait a bit more for any delayed errors to surface
    await page.waitForTimeout(3000);

    // Take a screenshot
    await page.screenshot({ 
      path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/card-export-verification.png',
      fullPage: true 
    });

    // Check for specific CardDescription export errors
    const cardDescriptionErrors = consoleErrors.filter(error => 
      error.includes('CardDescription') || 
      error.includes('card.tsx') ||
      error.includes('export named')
    );

    // Check for any module import/export errors
    const moduleErrors = consoleErrors.filter(error =>
      error.includes('does not provide an export') ||
      error.includes('SyntaxError') ||
      error.includes('import') ||
      error.includes('export')
    );

    // Log all console messages for debugging
    console.log('\n=== ALL CONSOLE MESSAGES ===');
    consoleMessages.forEach(msg => console.log(msg));

    console.log('\n=== CONSOLE ERRORS ===');
    consoleErrors.forEach(error => console.log(`ERROR: ${error}`));

    console.log('\n=== VERIFICATION RESULTS ===');
    console.log(`Total console errors: ${consoleErrors.length}`);
    console.log(`CardDescription-related errors: ${cardDescriptionErrors.length}`);
    console.log(`Module import/export errors: ${moduleErrors.length}`);

    // Assertions
    expect(cardDescriptionErrors.length, 'Should have no CardDescription export errors').toBe(0);
    expect(moduleErrors.length, 'Should have no module import/export errors').toBe(0);

    // Verify page loaded successfully by checking for a common element
    await expect(page.locator('body')).toBeVisible();

    console.log('\n✅ SUCCESS: No CardDescription export errors found!');
    console.log('✅ Page loaded successfully without console errors');

  } catch (error) {
    console.error('\n❌ ERROR during verification:', error);
    
    // Still take a screenshot even if there's an error
    try {
      await page.screenshot({ 
        path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/card-export-error.png',
        fullPage: true 
      });
    } catch (screenshotError) {
      console.error('Failed to take screenshot:', screenshotError);
    }
    
    throw error;
  }
});