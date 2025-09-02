import { chromium } from 'playwright';

async function simpleGraphitiTest() {
  console.log('🚀 Simple Graphiti Explorer Test at localhost:3738/graphiti');
  
  const browser = await chromium.launch({ headless: false, devtools: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  // Listen for console messages
  page.on('console', msg => {
    const type = msg.type();
    if (type === 'error' || type === 'warning') {
      console.log(`🔧 Browser ${type.toUpperCase()}: ${msg.text()}`);
    }
  });
  
  // Listen for page errors
  page.on('pageerror', error => {
    console.log(`❌ Page Error: ${error.message}`);
  });
  
  try {
    // Navigate with shorter timeout and no network wait
    console.log('📍 Navigating to localhost:3738/graphiti');
    await page.goto('http://localhost:3738/graphiti', { 
      waitUntil: 'domcontentloaded',
      timeout: 10000 
    });
    
    // Wait for any React components to render
    console.log('⏳ Waiting for page to load...');
    await page.waitForTimeout(5000);
    
    // Take screenshot of what we actually see
    console.log('📸 Taking screenshot of current state');
    await page.screenshot({ path: 'graphiti_current_state.png', fullPage: true });
    
    // Get page info
    const title = await page.title();
    const url = page.url();
    console.log(`📄 Page Title: "${title}"`);
    console.log(`🔗 Current URL: ${url}`);
    
    // Check if we're on a React app
    const hasReactRoot = await page.locator('#root').isVisible().catch(() => false);
    console.log(`⚛️ React root element found: ${hasReactRoot}`);
    
    // Look for any content in the body
    const bodyText = await page.locator('body').textContent();
    console.log(`📝 Body content length: ${bodyText ? bodyText.length : 0} characters`);
    
    if (bodyText && bodyText.length > 50) {
      console.log(`📄 First 200 chars: "${bodyText.substring(0, 200)}..."`);
    }
    
    // Look for common UI elements
    const buttons = await page.$$('button');
    const inputs = await page.$$('input');
    const divs = await page.$$('div');
    
    console.log(`🔘 Found ${buttons.length} buttons`);
    console.log(`📝 Found ${inputs.length} input fields`);
    console.log(`📦 Found ${divs.length} div elements`);
    
    // Check for our specific entity names
    const entityNames = ['login', 'UserManager', 'code-implementer', 'Authentication'];
    console.log('\n🔍 Searching for expected entities:');
    for (const entity of entityNames) {
      const found = bodyText && bodyText.includes(entity);
      console.log(`  ${found ? '✅' : '❌'} ${entity}: ${found ? 'FOUND' : 'NOT FOUND'}`);
    }
    
    // Check for dark theme indicators
    const darkThemeElements = await page.$$('[class*="dark"], [class*="black"], [class*="bg-gray"], [class*="bg-slate"]');
    console.log(`🎨 Found ${darkThemeElements.length} potential dark theme elements`);
    
    // Look for search functionality
    const searchInputs = await page.$$('input[placeholder*="search"], input[type="search"], input[placeholder*="Search"]');
    console.log(`🔍 Found ${searchInputs.length} search inputs`);
    
    if (searchInputs.length > 0) {
      console.log('🔍 Testing search input...');
      await searchInputs[0].fill('login');
      await page.waitForTimeout(2000);
      await page.screenshot({ path: 'graphiti_after_search.png', fullPage: true });
      console.log('✅ Search test completed');
    }
    
    // Keep browser open for manual inspection
    console.log('\n🔍 Browser is open for manual inspection...');
    console.log('Press Enter to close browser and complete test');
    
    // Wait for user input
    await new Promise(resolve => {
      process.stdin.once('data', resolve);
    });
    
    console.log('\n✅ Test completed!');
    console.log('Screenshots saved: graphiti_current_state.png, graphiti_after_search.png');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
    await page.screenshot({ path: 'graphiti_error_state.png', fullPage: true });
  } finally {
    await browser.close();
  }
}

simpleGraphitiTest();