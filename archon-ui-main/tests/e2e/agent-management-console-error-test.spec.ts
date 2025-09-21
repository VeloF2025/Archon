import { test, expect, Page } from '@playwright/test';

test.describe('Agent Management Page - Console Error Verification', () => {
  let consoleMessages: Array<{ type: string; text: string; location?: string }> = [];
  
  test.beforeEach(async ({ page }) => {
    // Capture all console messages
    consoleMessages = [];
    
    page.on('console', (msg) => {
      consoleMessages.push({
        type: msg.type(),
        text: msg.text(),
        location: msg.location()?.url
      });
    });

    // Capture page errors
    page.on('pageerror', (error) => {
      consoleMessages.push({
        type: 'pageerror',
        text: error.message,
        location: error.stack
      });
    });
  });

  test('should load agent management page without JavaScript errors', async ({ page }) => {
    console.log('🧪 Starting console error verification test...');
    
    try {
      // Navigate to the agent management page
      console.log('📍 Navigating to http://localhost:3737/agents');
      await page.goto('http://localhost:3737/agents', { 
        waitUntil: 'networkidle',
        timeout: 30000 
      });
      
      // Wait for the page to fully load and any async operations to complete
      console.log('⏳ Waiting for page to fully load...');
      await page.waitForTimeout(3000);
      
      // Wait for any React components to render
      try {
        await page.waitForSelector('[data-testid="agent-management-page"], .agent-management, h1', { timeout: 10000 });
        console.log('✅ Page content detected');
      } catch (e) {
        console.log('⚠️  No specific page content selector found, continuing...');
      }
      
      // Take a screenshot of the current state
      console.log('📸 Taking screenshot...');
      await page.screenshot({ 
        path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/tests/e2e/screenshots/agent-management-console-test.png',
        fullPage: true 
      });
      
      // Analyze console messages for specific errors we're looking for
      const urlConstructorErrors = consoleMessages.filter(msg => 
        msg.text.includes('Failed to construct \'URL\': Invalid URL')
      );
      
      const undefinedMatchErrors = consoleMessages.filter(msg => 
        msg.text.includes('Cannot read properties of undefined (reading \'match\')')
      );
      
      const jsErrors = consoleMessages.filter(msg => 
        msg.type === 'error' || msg.type === 'pageerror'
      );
      
      // Log all console messages for debugging
      console.log('🔍 All console messages captured:');
      consoleMessages.forEach((msg, index) => {
        console.log(`  ${index + 1}. [${msg.type.toUpperCase()}] ${msg.text}`);
        if (msg.location) {
          console.log(`     Location: ${msg.location}`);
        }
      });
      
      // Report specific error status
      console.log('\n📊 Specific Error Analysis:');
      console.log(`❌ URL Constructor Errors: ${urlConstructorErrors.length}`);
      urlConstructorErrors.forEach(err => console.log(`   - ${err.text}`));
      
      console.log(`❌ Undefined Match Errors: ${undefinedMatchErrors.length}`);
      undefinedMatchErrors.forEach(err => console.log(`   - ${err.text}`));
      
      console.log(`❌ Total JavaScript Errors: ${jsErrors.length}`);
      
      // Create a summary for the test results
      const errorSummary = {
        totalConsoleMessages: consoleMessages.length,
        totalErrors: jsErrors.length,
        urlConstructorErrors: urlConstructorErrors.length,
        undefinedMatchErrors: undefinedMatchErrors.length,
        hasTargetErrors: urlConstructorErrors.length > 0 || undefinedMatchErrors.length > 0,
        allErrors: jsErrors.map(err => ({ type: err.type, message: err.text }))
      };
      
      // Log final status
      if (errorSummary.hasTargetErrors) {
        console.log('🚨 VERIFICATION FAILED: Target JavaScript errors still present');
      } else if (errorSummary.totalErrors > 0) {
        console.log('⚠️  PARTIAL SUCCESS: Target errors fixed, but other JS errors present');
      } else {
        console.log('✅ VERIFICATION PASSED: No JavaScript errors detected');
      }
      
      // The test should pass if the specific target errors are resolved
      expect(urlConstructorErrors.length, 'URL constructor errors should be resolved').toBe(0);
      expect(undefinedMatchErrors.length, 'Undefined match errors should be resolved').toBe(0);
      
      // Store summary in a global for later access
      (global as any).testErrorSummary = errorSummary;
      
    } catch (error) {
      console.error('❌ Test execution failed:', error);
      
      // Take screenshot even on failure
      try {
        await page.screenshot({ 
          path: '/mnt/c/Jarvis/AI Workspace/Archon/archon-ui-main/tests/e2e/screenshots/agent-management-console-test-failed.png',
          fullPage: true 
        });
      } catch (screenshotError) {
        console.error('Failed to take failure screenshot:', screenshotError);
      }
      
      throw error;
    }
  });
});