import { test, expect } from '@playwright/test';

test.describe('Debug Cache Issue - Simple', () => {
  test('check served files and API endpoints', async ({ page }) => {
    console.log('=== DEBUGGING CACHE ISSUE ===\n');
    
    // Test 1: Check what's being served for AgentManagementPage
    console.log('TEST 1: Checking AgentManagementPage import');
    const pageFileResponse = await page.goto('http://localhost:3737/src/pages/AgentManagementPage.tsx');
    const pageContent = await pageFileResponse?.text() || '';
    
    const importMatch = pageContent.match(/import.*agentManagement[^;]*/);
    console.log('Import found:', importMatch?.[0] || 'NONE');
    
    if (pageContent.includes('agentManagementServiceV2')) {
      console.log('✅ Page is importing V2 service (fixed version)');
    } else {
      console.log('❌ Page is NOT importing V2 service');
    }
    
    // Test 2: Check API endpoints
    console.log('\nTEST 2: Testing API endpoints directly');
    const endpoints = [
      '/api/agent-management/agents',
      '/api/agent-management/analytics/performance', 
      '/api/agent-management/analytics/project-overview',
      '/api/agent-management/costs/recommendations'
    ];
    
    let allApisWorking = true;
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(`http://localhost:8181${endpoint}`);
        const status = response.status;
        const ok = response.ok;
        console.log(`  ${endpoint}: ${status} ${ok ? '✅' : '❌'}`);
        if (!ok) allApisWorking = false;
      } catch (error) {
        console.log(`  ${endpoint}: ERROR ❌`, error.message);
        allApisWorking = false;
      }
    }
    
    // Test 3: Navigate to agent page and capture errors
    console.log('\nTEST 3: Loading agent management page');
    
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    // Navigate with cache bypass
    await page.goto('http://localhost:3737/agents', {
      waitUntil: 'networkidle',
      // Force cache bypass
      bypassCSP: true
    });
    
    // Wait for any errors to appear
    await page.waitForTimeout(3000);
    
    console.log(`Console errors found: ${consoleErrors.length}`);
    consoleErrors.slice(0, 3).forEach(err => {
      console.log(`  Error: ${err.substring(0, 100)}...`);
    });
    
    // Test 4: Check the actual served JavaScript
    console.log('\nTEST 4: Checking actual served JS files');
    
    // Check if old service file exists
    const oldServiceResponse = await fetch('http://localhost:3737/src/services/agentManagementService.ts');
    console.log(`Old service file status: ${oldServiceResponse.status}`);
    
    // Check if new service file exists  
    const newServiceResponse = await fetch('http://localhost:3737/src/services/agentManagementServiceV2.ts');
    console.log(`New V2 service file status: ${newServiceResponse.status}`);
    
    if (newServiceResponse.ok) {
      const v2Content = await newServiceResponse.text();
      const hasNewURL = v2Content.includes('new URL');
      console.log(`V2 service uses 'new URL': ${hasNewURL ? 'YES ❌' : 'NO ✅'}`);
    }
    
    // Test 5: Test the simple HTML page
    console.log('\nTEST 5: Testing simple HTML test page');
    await page.goto('http://localhost:3737/test-agent-page.html');
    await page.waitForTimeout(2000);
    
    const testPageErrors = await page.$$eval('.status.error', elements => 
      elements.map(el => el.textContent)
    );
    const testPageSuccess = await page.$$eval('.status.success', elements =>
      elements.map(el => el.textContent)  
    );
    
    console.log(`Test page - Success: ${testPageSuccess.length}, Errors: ${testPageErrors.length}`);
    
    // Take debug screenshot
    await page.screenshot({ 
      path: 'tests/e2e/screenshots/cache-debug-simple.png',
      fullPage: true 
    });
    
    // Summary
    console.log('\n=== SUMMARY ===');
    console.log(`Server is serving V2 import: ${pageContent.includes('agentManagementServiceV2') ? 'YES ✅' : 'NO ❌'}`);
    console.log(`All APIs working: ${allApisWorking ? 'YES ✅' : 'NO ❌'}`);
    console.log(`Page has errors: ${consoleErrors.length > 0 ? `YES (${consoleErrors.length}) ❌` : 'NO ✅'}`);
    
    // Focus on the real issue
    const urlConstructorErrors = consoleErrors.filter(e => e.includes('Failed to construct'));
    if (urlConstructorErrors.length > 0) {
      console.log('\n⚠️ URL Constructor errors still present!');
      console.log('This indicates severe browser caching.');
      console.log('SOLUTION: The user needs to:');
      console.log('1. Close ALL browser tabs');
      console.log('2. Clear browser cache completely');
      console.log('3. Or use a different browser/incognito mode');
    }
    
    expect(allApisWorking).toBe(true);
  });
});