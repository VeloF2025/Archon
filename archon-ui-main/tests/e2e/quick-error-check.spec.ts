import { test, expect } from '@playwright/test';

test('Quick JavaScript Error Check - Agent Management Page', async ({ page }) => {
  const consoleErrors: string[] = [];
  const specificErrors = {
    urlErrors: [] as string[],
    matchErrors: [] as string[],
    serviceErrors: [] as string[]
  };
  
  // Capture console errors
  page.on('console', msg => {
    if (msg.type() === 'error') {
      const text = msg.text();
      consoleErrors.push(text);
      
      // Check for specific error patterns we fixed
      if (text.includes('Failed to construct \'URL\'') || text.includes('Invalid URL')) {
        specificErrors.urlErrors.push(text);
      }
      if (text.includes('Cannot read properties of undefined (reading \'match\')')) {
        specificErrors.matchErrors.push(text);
      }
      if (text.includes('AgentManagementService') || text.includes('socketIOService')) {
        specificErrors.serviceErrors.push(text);
      }
      
      console.log(`🔴 Console Error: ${text}`);
    }
  });

  // Capture page errors
  page.on('pageerror', error => {
    const message = error.message;
    consoleErrors.push(`PAGE ERROR: ${message}`);
    console.log(`🔴 Page Error: ${message}`);
  });

  console.log('🚀 Navigating to agent management page...');
  
  // Navigate to the page
  await page.goto('http://localhost:3737/agents', { 
    waitUntil: 'domcontentloaded',
    timeout: 15000 
  });
  
  console.log('✅ Page loaded, waiting 5 seconds for JavaScript to execute...');
  await page.waitForTimeout(5000);
  
  // Take screenshot
  await page.screenshot({ 
    path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/tests/e2e/screenshots/quick-error-check.png',
    fullPage: true 
  });
  
  // Report results
  console.log('\n📊 ERROR ANALYSIS RESULTS:');
  console.log(`🔴 Total Console Errors: ${consoleErrors.length}`);
  console.log(`🎯 URL Construction Errors: ${specificErrors.urlErrors.length}`);
  console.log(`🎯 Match Property Errors: ${specificErrors.matchErrors.length}`);
  console.log(`🎯 Service-Related Errors: ${specificErrors.serviceErrors.length}`);
  
  if (consoleErrors.length > 0) {
    console.log('\n🔴 ALL CONSOLE ERRORS:');
    consoleErrors.forEach((error, index) => {
      console.log(`${index + 1}. ${error}`);
    });
  } else {
    console.log('\n✅ SUCCESS: No JavaScript console errors detected!');
  }
  
  // Check if the page shows the expected error state (since backend is down)
  const errorHeading = await page.locator('h3:has-text("Error Loading Agent Management")').count();
  console.log(`\n📄 Page shows expected error message: ${errorHeading > 0 ? 'Yes' : 'No'}`);
  
  // The main thing we're testing - no JavaScript runtime errors
  expect(specificErrors.urlErrors.length).toBe(0);
  expect(specificErrors.matchErrors.length).toBe(0);
});