import puppeteer from 'puppeteer';
import fs from 'fs';

async function quickTest() {
  let browser;
  try {
    console.log('🚀 Quick Graphiti test...');
    
    browser = await puppeteer.launch({
      headless: false,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
      defaultViewport: { width: 1920, height: 1080 }
    });
    
    const page = await browser.newPage();
    
    console.log('📍 Going to localhost:3738/graphiti...');
    await page.goto('http://localhost:3738/graphiti');
    
    // Wait for React to start
    await new Promise(resolve => setTimeout(resolve, 8000));
    
    console.log('📸 Taking screenshot...');
    await page.screenshot({ path: 'graphiti_quick_test.png', fullPage: true });
    
    // Check what's on the page
    const title = await page.title();
    console.log(`📄 Title: ${title}`);
    
    // Look for text content
    const bodyText = await page.evaluate(() => document.body.innerText);
    const hasGraphitiContent = bodyText.includes('Graphiti') || bodyText.includes('Explorer');
    console.log(`📝 Has Graphiti content: ${hasGraphitiContent}`);
    
    // Check for specific elements
    const elements = {
      headers: await page.$$('h1, h2, h3').then(els => els.length),
      buttons: await page.$$('button').then(els => els.length),
      inputs: await page.$$('input').then(els => els.length),
      cards: await page.$$('[class*="glass"], [class*="card"]').then(els => els.length)
    };
    
    console.log('📊 Elements found:', elements);
    
    // Try to interact with search if it exists
    const searchInput = await page.$('input[placeholder*="Search"]');
    if (searchInput) {
      console.log('🔍 Testing search...');
      await searchInput.type('login');
      await new Promise(resolve => setTimeout(resolve, 1000));
      await page.screenshot({ path: 'graphiti_search.png', fullPage: true });
    }
    
    console.log('✅ Quick test complete!');
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    
    if (browser) {
      try {
        const page = (await browser.pages())[0];
        await page.screenshot({ path: 'graphiti_quick_error.png', fullPage: true });
        console.log('📸 Error screenshot saved');
      } catch (e) {}
    }
  } finally {
    if (browser) {
      // Keep browser open for 10 seconds to see the result
      console.log('🔍 Keeping browser open for 10 seconds...');
      await new Promise(resolve => setTimeout(resolve, 10000));
      await browser.close();
    }
  }
}

quickTest();