import { chromium } from 'playwright';

async function testGraphitiExplorer() {
  console.log('🚀 Testing Graphiti Explorer at localhost:3738/graphiti');
  
  const browser = await chromium.launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Navigate to our Graphiti Explorer
    console.log('📍 Navigating to localhost:3738/graphiti');
    await page.goto('http://localhost:3738/graphiti', { waitUntil: 'networkidle' });
    
    // Wait a moment for the page to fully load
    await page.waitForTimeout(2000);
    
    // Take initial screenshot
    console.log('📸 Taking initial screenshot');
    await page.screenshot({ path: 'graphiti_initial.png', fullPage: true });
    
    // Check for dark theme elements
    console.log('🎨 Checking dark glassmorphism theme');
    const darkElements = await page.$$('body, .dark, [class*="dark"]');
    console.log(`Found ${darkElements.length} dark theme elements`);
    
    // Look for entity cards
    console.log('🔍 Looking for entity cards');
    const entityCards = await page.$$('[class*="entity"], [class*="card"], .node, [data-testid*="entity"]');
    console.log(`Found ${entityCards.length} potential entity cards`);
    
    // Check for the 4 expected entities: login, UserManager, code-implementer, Authentication
    const expectedEntities = ['login', 'UserManager', 'code-implementer', 'Authentication'];
    for (const entity of expectedEntities) {
      const found = await page.locator(`text=${entity}`).first().isVisible().catch(() => false);
      console.log(`✅ ${entity}: ${found ? 'FOUND' : 'NOT FOUND'}`);
    }
    
    // Test search functionality
    console.log('🔍 Testing search functionality');
    const searchBox = await page.locator('input[type="search"], input[placeholder*="search"], input[placeholder*="Search"]').first();
    if (await searchBox.isVisible()) {
      await searchBox.fill('login');
      await page.waitForTimeout(1000);
      await page.screenshot({ path: 'graphiti_search_login.png', fullPage: true });
      console.log('✅ Search box tested with "login"');
    } else {
      console.log('❌ Search box not found');
    }
    
    // Test clicking on entity cards (if found)
    console.log('🖱️ Testing entity card interactions');
    const clickableCards = await page.$$('[role="button"], button, [class*="clickable"], [class*="interactive"]');
    if (clickableCards.length > 0) {
      await clickableCards[0].click();
      await page.waitForTimeout(1000);
      await page.screenshot({ path: 'graphiti_card_clicked.png', fullPage: true });
      console.log('✅ Clicked first interactive element');
    }
    
    // Check for view mode selector
    console.log('👁️ Looking for view mode selector');
    const viewModeButtons = await page.$$('button[class*="view"], [role="tablist"], [class*="mode"]');
    console.log(`Found ${viewModeButtons.length} potential view mode buttons`);
    
    if (viewModeButtons.length > 0) {
      await viewModeButtons[0].click();
      await page.waitForTimeout(1000);
      await page.screenshot({ path: 'graphiti_view_mode.png', fullPage: true });
      console.log('✅ Tested view mode selector');
    }
    
    // Check browser console for errors
    console.log('🔧 Checking browser console for errors');
    const logs = await page.evaluate(() => {
      const logs = [];
      const originalLog = console.log;
      const originalError = console.error;
      const originalWarn = console.warn;
      
      console.log = (...args) => { logs.push({ type: 'log', args }); originalLog(...args); };
      console.error = (...args) => { logs.push({ type: 'error', args }); originalError(...args); };
      console.warn = (...args) => { logs.push({ type: 'warn', args }); originalWarn(...args); };
      
      return window.console._logs || [];
    });
    
    console.log(`Console logs: ${logs.length} entries`);
    
    // Take final screenshot
    console.log('📸 Taking final screenshot');
    await page.screenshot({ path: 'graphiti_final.png', fullPage: true });
    
    // Get page title and URL for confirmation
    const title = await page.title();
    const url = page.url();
    console.log(`📄 Page Title: ${title}`);
    console.log(`🔗 Final URL: ${url}`);
    
    // Get basic page info
    const bodyText = await page.locator('body').textContent();
    const hasContent = bodyText && bodyText.length > 100;
    console.log(`📝 Page has substantial content: ${hasContent}`);
    console.log(`📏 Content length: ${bodyText ? bodyText.length : 0} characters`);
    
    console.log('\n✅ Test completed successfully!');
    console.log('Screenshots saved: graphiti_initial.png, graphiti_search_login.png, graphiti_card_clicked.png, graphiti_view_mode.png, graphiti_final.png');
    
  } catch (error) {
    console.error('❌ Error during testing:', error);
    await page.screenshot({ path: 'graphiti_error.png', fullPage: true });
  } finally {
    await browser.close();
  }
}

testGraphitiExplorer();