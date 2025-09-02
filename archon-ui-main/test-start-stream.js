import { chromium } from 'playwright';

async function testStartStreamButton() {
    const browser = await chromium.launch({ 
        headless: false,
        slowMo: 1000 // Slow down actions to see what's happening
    });
    
    const context = await browser.newContext();
    const page = await context.newPage();
    
    try {
        console.log('Navigating to Archon UI...');
        await page.goto('http://localhost:3738', { timeout: 30000 });
        await page.waitForLoadState('networkidle');
        
        // Take initial screenshot
        await page.screenshot({ path: 'archon-initial.png', fullPage: true });
        console.log('Initial screenshot saved as archon-initial.png');
        
        // Get page title and URL
        const title = await page.title();
        const url = page.url();
        console.log(`Page title: ${title}`);
        console.log(`Current URL: ${url}`);
        
        // Look for DeepConf or Real-Time Monitoring section
        console.log('Looking for DeepConf/Real-Time Monitoring section...');
        
        // Try to find navigation elements
        const navItems = await page.$$eval('nav a, .nav a, [role="navigation"] a, a', 
            elements => elements.map(el => ({ text: el.textContent?.trim(), href: el.href }))
        );
        
        console.log('Found navigation items:');
        navItems.forEach((item, i) => {
            if (item.text) console.log(`${i}: "${item.text}" -> ${item.href}`);
        });
        
        // Look for monitoring-related links
        const monitoringLink = await page.$('text=/monitoring|deepconf|real-time/i');
        if (monitoringLink) {
            console.log('Found monitoring-related link, clicking...');
            await monitoringLink.click();
            await page.waitForTimeout(2000);
        } else {
            console.log('No monitoring link found, checking current page for buttons...');
        }
        
        // Look for all buttons on the page
        const buttons = await page.$$eval('button, input[type="button"], input[type="submit"]', 
            elements => elements.map((el, i) => ({
                index: i,
                text: el.textContent?.trim() || el.value || `Button ${i}`,
                id: el.id,
                className: el.className,
                disabled: el.disabled,
                type: el.type
            }))
        );
        
        console.log(`\nFound ${buttons.length} buttons on page:`);
        buttons.forEach(btn => {
            console.log(`  ${btn.index}: "${btn.text}" (id: ${btn.id}, class: ${btn.className}, disabled: ${btn.disabled})`);
        });
        
        // Look specifically for Start Stream button
        console.log('\nLooking for Start Stream button...');
        
        const startStreamSelectors = [
            'text="Start Stream"',
            'text="Start Stream (Native)"',
            'button:has-text("Start Stream")',
            '[data-testid*="start-stream"]',
            '[id*="start-stream"]',
            'input[value*="Start Stream"]',
            '.start-stream',
            '#start-stream'
        ];
        
        let startButton = null;
        let foundSelector = null;
        
        for (const selector of startStreamSelectors) {
            try {
                startButton = await page.$(selector);
                if (startButton) {
                    foundSelector = selector;
                    console.log(`Found Start Stream button with selector: ${selector}`);
                    break;
                }
            } catch (error) {
                // Continue to next selector
            }
        }
        
        if (startButton) {
            // Check button properties
            const isDisabled = await startButton.isDisabled();
            const isVisible = await startButton.isVisible();
            const isEnabled = await startButton.isEnabled();
            
            console.log('\nButton properties:');
            console.log(`  Disabled: ${isDisabled}`);
            console.log(`  Visible: ${isVisible}`);
            console.log(`  Enabled: ${isEnabled}`);
            
            // Get button text and attributes
            const buttonText = await startButton.textContent();
            const buttonValue = await startButton.getAttribute('value');
            const buttonId = await startButton.getAttribute('id');
            const buttonClass = await startButton.getAttribute('class');
            
            console.log(`  Text: "${buttonText}"`);
            console.log(`  Value: "${buttonValue}"`);
            console.log(`  ID: "${buttonId}"`);
            console.log(`  Class: "${buttonClass}"`);
            
            // Listen for console messages and errors
            const consoleMessages = [];
            const pageErrors = [];
            
            page.on('console', msg => {
                consoleMessages.push(`Console ${msg.type()}: ${msg.text()}`);
            });
            
            page.on('pageerror', error => {
                pageErrors.push(`Page Error: ${error.message}`);
            });
            
            // Try to click the button
            console.log('\nAttempting to click Start Stream button...');
            
            try {
                if (isEnabled && isVisible) {
                    await startButton.click();
                    console.log('✅ Button clicked successfully');
                } else {
                    console.log('⚠️  Button is not clickable (disabled or not visible)');
                    console.log('Attempting force click...');
                    await startButton.click({ force: true });
                    console.log('✅ Force click completed');
                }
                
                // Wait for any response
                console.log('Waiting for response...');
                await page.waitForTimeout(3000);
                
                // Take screenshot after click
                await page.screenshot({ path: 'archon-after-click.png', fullPage: true });
                console.log('Screenshot saved as archon-after-click.png');
                
                // Check for any changes or new elements
                const newElements = await page.$$eval('[class*="stream"], [id*="stream"], [data-testid*="stream"]',
                    elements => elements.map(el => ({ 
                        tagName: el.tagName, 
                        text: el.textContent?.trim(),
                        id: el.id,
                        className: el.className
                    }))
                );
                
                if (newElements.length > 0) {
                    console.log('\nNew stream-related elements found:');
                    newElements.forEach((el, i) => {
                        console.log(`  ${i}: <${el.tagName}> "${el.text}" (id: ${el.id}, class: ${el.className})`);
                    });
                }
                
            } catch (error) {
                console.log(`❌ Error clicking button: ${error.message}`);
            }
            
            // Report console messages and errors
            if (consoleMessages.length > 0) {
                console.log('\n📋 Console messages:');
                consoleMessages.forEach(msg => console.log(`  ${msg}`));
            }
            
            if (pageErrors.length > 0) {
                console.log('\n❌ Page errors:');
                pageErrors.forEach(error => console.log(`  ${error}`));
            }
            
        } else {
            console.log('❌ Start Stream button not found');
            
            // Check if there are any buttons with "stream" in their text
            const streamButtons = buttons.filter(btn => 
                btn.text.toLowerCase().includes('stream') ||
                btn.id.toLowerCase().includes('stream') ||
                btn.className.toLowerCase().includes('stream')
            );
            
            if (streamButtons.length > 0) {
                console.log('\nFound buttons with "stream" in them:');
                streamButtons.forEach(btn => {
                    console.log(`  "${btn.text}" (id: ${btn.id}, class: ${btn.className})`);
                });
            }
        }
        
        // Take final screenshot
        await page.screenshot({ path: 'archon-final.png', fullPage: true });
        console.log('Final screenshot saved as archon-final.png');
        
    } catch (error) {
        console.log(`❌ Error during test: ${error.message}`);
        await page.screenshot({ path: 'archon-error.png', fullPage: true });
        console.log('Error screenshot saved as archon-error.png');
    } finally {
        await browser.close();
        console.log('\nTest completed. Check the screenshots for visual confirmation.');
    }
}

testStartStreamButton().catch(console.error);