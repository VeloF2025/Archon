import puppeteer from 'puppeteer';
import fs from 'fs';

async function testGraphitiExplorer() {
  let browser;
  try {
    console.log('🚀 Starting Graphiti Explorer test...');
    
    browser = await puppeteer.launch({
      headless: false, // Run in visible mode
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    
    // Enable console logging
    page.on('console', msg => console.log(`Console ${msg.type()}: ${msg.text()}`));
    page.on('pageerror', error => console.log(`Page Error: ${error.message}`));
    
    console.log('📍 Navigating to localhost:3738/graphiti...');
    await page.goto('http://localhost:3738/graphiti', { waitUntil: 'networkidle0', timeout: 30000 });
    
    // Wait for React to render and components to mount
    console.log('⏳ Waiting for React components to load...');
    try {
      // Wait for the main Graphiti Explorer header to appear
      await page.waitForSelector('h1', { timeout: 15000 });
      console.log('✅ Main header found!');
      
      // Wait a bit more for all components to fully render
      await page.waitForTimeout(5000);
      
      // Check if we have the glass theme classes
      const hasGlassTheme = await page.$('.glass-purple, .glass, .neon-grid');
      console.log(`🎨 Glass theme elements found: ${!!hasGlassTheme}`);
      
    } catch (waitError) {
      console.log('⚠️ Some elements not found, continuing anyway...');
    }
    
    console.log('📸 Taking initial screenshot...');
    await page.screenshot({ path: 'graphiti_initial.png', fullPage: true });
    
    // Get page info
    const title = await page.title();
    const url = page.url();
    console.log(`📄 Page title: ${title}`);
    console.log(`🌐 URL: ${url}`);
    
    // Check for main elements
    const mainContent = await page.$('div[id="root"]');
    console.log(`📦 Root element found: ${!!mainContent}`);
    
    // Look for Graphiti-specific elements
    const searchInput = await page.$('input[placeholder*="search" i]');
    console.log(`🔍 Search input found: ${!!searchInput}`);
    
    if (searchInput) {
      console.log('🔍 Testing search functionality...');
      await searchInput.type('login');
      await page.waitForTimeout(1000);
      await page.screenshot({ path: 'graphiti_search_test.png', fullPage: true });
    }
    
    // Look for entity cards or nodes
    const nodes = await page.$$('[data-testid*="entity"], .react-flow__node, [class*="node"], [class*="entity"]');
    console.log(`📋 Found ${nodes.length} potential entity/node elements`);
    
    // Try to click on first node if found
    if (nodes.length > 0) {
      console.log('🖱️ Clicking first node/entity...');
      await nodes[0].click();
      await page.waitForTimeout(2000);
      await page.screenshot({ path: 'graphiti_node_clicked.png', fullPage: true });
    }
    
    // Look for buttons and controls
    const buttons = await page.$$('button');
    console.log(`🔘 Found ${buttons.length} buttons`);
    
    // Check for sidebars
    const sidebars = await page.$$('[class*="sidebar"], .sidebar, [data-testid*="sidebar"]');
    console.log(`📱 Found ${sidebars.length} sidebar elements`);
    
    // Check for React Flow elements
    const reactFlowElements = await page.$$('.react-flow, [class*="react-flow"]');
    console.log(`🌐 React Flow elements: ${reactFlowElements.length}`);
    
    // Check for dark theme
    const bodyStyle = await page.evaluate(() => {
      return window.getComputedStyle(document.body).backgroundColor;
    });
    console.log(`🎨 Body background color: ${bodyStyle}`);
    
    // Check for purple accents
    const purpleElements = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      let purpleCount = 0;
      for (const el of elements) {
        const style = window.getComputedStyle(el);
        if (style.color.includes('rgb(139, 92, 246)') || 
            style.backgroundColor.includes('rgb(139, 92, 246)') ||
            style.color.includes('purple') ||
            style.backgroundColor.includes('purple')) {
          purpleCount++;
        }
      }
      return purpleCount;
    });
    console.log(`🟣 Purple themed elements: ${purpleElements}`);
    
    // Get page content for analysis
    const pageText = await page.evaluate(() => document.body.innerText);
    const hasGraphitiContent = ['graphiti', 'entity', 'node', 'relationship', 'explore', 'knowledge'].some(
      keyword => pageText.toLowerCase().includes(keyword)
    );
    console.log(`📝 Contains Graphiti-related content: ${hasGraphitiContent}`);
    
    // Check for specific UI components
    const components = {
      searchBox: await page.$('input[placeholder*="search" i]'),
      entityCards: await page.$$('[class*="entity"], [data-testid*="entity"]'),
      controls: await page.$$('.react-flow__controls'),
      minimap: await page.$$('.react-flow__minimap'),
      panels: await page.$$('.react-flow__panel'),
      tabs: await page.$$('[role="tab"], [data-state]')
    };
    
    console.log('📊 UI Component Analysis:');
    Object.entries(components).forEach(([name, elements]) => {
      const count = Array.isArray(elements) ? elements.length : (elements ? 1 : 0);
      console.log(`  ${name}: ${count} elements`);
    });
    
    // Take final comprehensive screenshot
    await page.screenshot({ path: 'graphiti_final_test.png', fullPage: true });
    
    // Save page HTML for analysis
    const html = await page.content();
    fs.writeFileSync('graphiti_page_analysis.html', html);
    
    console.log('✅ Test completed successfully!');
    console.log('📸 Screenshots saved: graphiti_initial.png, graphiti_search_test.png, graphiti_node_clicked.png, graphiti_final_test.png');
    console.log('💾 HTML saved: graphiti_page_analysis.html');
    
  } catch (error) {
    console.error('❌ Error during test:', error.message);
    
    if (browser) {
      try {
        const page = (await browser.pages())[0];
        await page.screenshot({ path: 'graphiti_error.png', fullPage: true });
        console.log('📸 Error screenshot saved: graphiti_error.png');
      } catch (screenshotError) {
        console.error('Failed to take error screenshot:', screenshotError.message);
      }
    }
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

// Run the test
testGraphitiExplorer();