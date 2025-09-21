import { test, expect } from '@playwright/test';

test.describe('JavaScript Error Verification', () => {
  test('should verify agent management page loads without errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    const consoleWarnings: string[] = [];
    
    // Capture all console messages
    page.on('console', msg => {
      const text = msg.text();
      const type = msg.type();
      
      if (type === 'error') {
        consoleErrors.push(`ERROR: ${text}`);
        console.log(`🔴 Console Error: ${text}`);
      } else if (type === 'warning') {
        consoleWarnings.push(`WARNING: ${text}`);
        console.log(`🟡 Console Warning: ${text}`);
      } else {
        console.log(`📝 Console ${type}: ${text}`);
      }
    });

    // Capture page errors (uncaught exceptions)
    page.on('pageerror', error => {
      consoleErrors.push(`PAGE ERROR: ${error.message}`);
      console.log(`🔴 Page Error: ${error.message}`);
      console.log(`Stack: ${error.stack}`);
    });

    console.log('🚀 Navigating to agent management page...');
    
    try {
      // Navigate to the agent management page
      await page.goto('http://localhost:3737/agents', { 
        waitUntil: 'networkidle',
        timeout: 30000 
      });
      
      console.log('✅ Page navigation completed');
      
      // Wait for the page to fully load
      console.log('⏳ Waiting for page to stabilize (15 seconds)...');
      await page.waitForTimeout(15000);
      
      // Check if the page loaded properly by looking for key elements
      console.log('🔍 Checking for key page elements...');
      
      // Look for the main content area or any agent-related elements
      const pageContent = await page.locator('body').textContent();
      console.log('📄 Page contains content:', pageContent ? 'Yes' : 'No');
      
      // Check for specific elements that should be present
      const agentElements = await page.locator('[class*="agent"], [id*="agent"], h1, h2, h3').count();
      console.log(`📊 Found ${agentElements} potential agent-related elements`);
      
      // Take a screenshot
      console.log('📸 Taking screenshot...');
      await page.screenshot({ 
        path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/tests/e2e/screenshots/error-verification.png',
        fullPage: true 
      });
      
      console.log('📊 FINAL RESULTS:');
      console.log(`🔴 Total Errors: ${consoleErrors.length}`);
      console.log(`🟡 Total Warnings: ${consoleWarnings.length}`);
      
      if (consoleErrors.length > 0) {
        console.log('\n🔴 CONSOLE ERRORS FOUND:');
        consoleErrors.forEach((error, index) => {
          console.log(`${index + 1}. ${error}`);
        });
      }
      
      if (consoleWarnings.length > 0) {
        console.log('\n🟡 CONSOLE WARNINGS FOUND:');
        consoleWarnings.forEach((warning, index) => {
          console.log(`${index + 1}. ${warning}`);
        });
      }
      
      // Check for specific error patterns we were trying to fix
      const urlErrors = consoleErrors.filter(error => 
        error.includes('Failed to construct \'URL\'') || 
        error.includes('Invalid URL')
      );
      
      const matchErrors = consoleErrors.filter(error => 
        error.includes('Cannot read properties of undefined (reading \'match\')')
      );
      
      const agentServiceErrors = consoleErrors.filter(error => 
        error.includes('AgentManagementService') || 
        error.includes('socketIOService')
      );
      
      console.log(`\n🎯 SPECIFIC ERROR ANALYSIS:`);
      console.log(`   URL Construction Errors: ${urlErrors.length}`);
      console.log(`   Match Property Errors: ${matchErrors.length}`);
      console.log(`   Agent Service Errors: ${agentServiceErrors.length}`);
      
      // Report status
      if (consoleErrors.length === 0) {
        console.log('\n✅ SUCCESS: No console errors detected!');
      } else {
        console.log('\n❌ ISSUES DETECTED: Console errors still present');
      }
      
      // Assertions for test framework
      expect(urlErrors.length).toBe(0);
      expect(matchErrors.length).toBe(0);
      expect(agentServiceErrors.length).toBe(0);
      
    } catch (error) {
      console.log(`❌ Test execution error: ${error}`);
      
      // Take screenshot even on error
      try {
        await page.screenshot({ 
          path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/tests/e2e/screenshots/error-verification-failed.png',
          fullPage: true 
        });
        console.log('📸 Error screenshot saved');
      } catch (screenshotError) {
        console.log(`📸 Could not save error screenshot: ${screenshotError}`);
      }
      
      throw error;
    }
  });
});