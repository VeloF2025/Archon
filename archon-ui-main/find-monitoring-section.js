import { chromium } from 'playwright';

async function findMonitoringSection() {
    const browser = await chromium.launch({ 
        headless: false,
        slowMo: 2000 // Slower for better visibility
    });
    
    const context = await browser.newContext();
    const page = await context.newPage();
    
    try {
        console.log('Navigating to Archon UI...');
        await page.goto('http://localhost:3738', { timeout: 30000 });
        await page.waitForLoadState('networkidle');
        
        // Take initial screenshot
        await page.screenshot({ path: 'step1-initial.png', fullPage: true });
        console.log('Step 1: Initial page loaded');
        
        // Look for sidebar navigation icons
        console.log('Looking for sidebar navigation...');
        const sidebarIcons = await page.$$('nav [role="button"], nav button, [data-testid*="nav"], .sidebar button, .nav-item');
        console.log(`Found ${sidebarIcons.length} potential navigation elements`);
        
        // Try to find monitoring/deepconf related navigation
        const navElements = await page.$$eval('nav *, [class*="nav"] *, [class*="sidebar"] *', 
            elements => elements
                .filter(el => el.textContent || el.title || el.getAttribute('aria-label'))
                .map((el, i) => ({
                    index: i,
                    text: el.textContent?.trim(),
                    title: el.title,
                    ariaLabel: el.getAttribute('aria-label'),
                    className: el.className,
                    tagName: el.tagName,
                    id: el.id
                }))
                .filter(item => 
                    (item.text && (item.text.toLowerCase().includes('monitor') || 
                                   item.text.toLowerCase().includes('deepconf') ||
                                   item.text.toLowerCase().includes('stream') ||
                                   item.text.toLowerCase().includes('real-time'))) ||
                    (item.title && (item.title.toLowerCase().includes('monitor') || 
                                    item.title.toLowerCase().includes('deepconf'))) ||
                    (item.ariaLabel && (item.ariaLabel.toLowerCase().includes('monitor') || 
                                        item.ariaLabel.toLowerCase().includes('deepconf')))
                )
        );
        
        console.log('Found monitoring-related elements:');
        navElements.forEach(item => {
            console.log(`  ${item.tagName}: "${item.text}" (title: ${item.title}, aria: ${item.ariaLabel})`);
        });
        
        // Check all sidebar navigation icons systematically
        console.log('\nChecking all sidebar icons...');
        const sidebar = await page.$('nav, [class*="sidebar"], [class*="nav"]');
        if (sidebar) {
            const icons = await sidebar.$$('button, [role="button"], a, [class*="icon"]');
            console.log(`Found ${icons.length} clickable sidebar elements`);
            
            for (let i = 0; i < Math.min(icons.length, 10); i++) {
                try {
                    const icon = icons[i];
                    const isVisible = await icon.isVisible();
                    const title = await icon.getAttribute('title');
                    const ariaLabel = await icon.getAttribute('aria-label');
                    const text = await icon.textContent();
                    const className = await icon.getAttribute('class');
                    
                    console.log(`Icon ${i}: visible=${isVisible}, title="${title}", aria="${ariaLabel}", text="${text?.trim()}", class="${className}"`);
                    
                    // Check if this might be monitoring/deepconf related
                    const isMonitoringRelated = [title, ariaLabel, text, className].some(attr => 
                        attr && (
                            attr.toLowerCase().includes('monitor') ||
                            attr.toLowerCase().includes('deepconf') ||
                            attr.toLowerCase().includes('stream') ||
                            attr.toLowerCase().includes('real-time') ||
                            attr.toLowerCase().includes('activity') ||
                            attr.toLowerCase().includes('live')
                        )
                    );
                    
                    if (isMonitoringRelated || i < 6) { // Check first 6 icons or any monitoring-related ones
                        console.log(`Clicking on icon ${i}...`);
                        await icon.click();
                        await page.waitForTimeout(3000);
                        
                        // Take screenshot after clicking
                        await page.screenshot({ path: `step2-clicked-icon-${i}.png`, fullPage: true });
                        
                        // Look for "Start Stream" button on this page
                        const startStreamButton = await page.$('text="Start Stream"') || 
                                                 await page.$('text="Start Stream (Native)"') ||
                                                 await page.$('button:has-text("Start Stream")') ||
                                                 await page.$('[id*="start-stream"]') ||
                                                 await page.$('[data-testid*="start-stream"]');
                        
                        if (startStreamButton) {
                            console.log(`🎉 FOUND Start Stream button on page after clicking icon ${i}!`);
                            
                            // Check button properties
                            const isDisabled = await startStreamButton.isDisabled();
                            const isEnabled = await startStreamButton.isEnabled();
                            const buttonText = await startStreamButton.textContent();
                            
                            console.log(`Button text: "${buttonText}"`);
                            console.log(`Button enabled: ${isEnabled}`);
                            console.log(`Button disabled: ${isDisabled}`);
                            
                            // Try to click it
                            console.log('Attempting to click Start Stream button...');
                            
                            // Listen for console messages
                            const consoleMessages = [];
                            page.on('console', msg => consoleMessages.push(`${msg.type()}: ${msg.text()}`));
                            
                            try {
                                if (isEnabled) {
                                    await startStreamButton.click();
                                    console.log('✅ Start Stream button clicked successfully!');
                                } else {
                                    console.log('⚠️ Button is disabled, trying force click...');
                                    await startStreamButton.click({ force: true });
                                    console.log('✅ Force click completed');
                                }
                                
                                // Wait for response
                                await page.waitForTimeout(5000);
                                
                                // Take screenshot after clicking
                                await page.screenshot({ path: 'step3-after-stream-click.png', fullPage: true });
                                
                                // Check for console messages
                                if (consoleMessages.length > 0) {
                                    console.log('\n📋 Console messages after clicking:');
                                    consoleMessages.forEach(msg => console.log(`  ${msg}`));
                                }
                                
                                // Look for streaming indicators
                                const streamingIndicators = await page.$$eval('[class*="stream"], [id*="stream"], [data-testid*="stream"]',
                                    elements => elements.map(el => ({
                                        text: el.textContent?.trim(),
                                        className: el.className,
                                        id: el.id
                                    }))
                                );
                                
                                if (streamingIndicators.length > 0) {
                                    console.log('\nFound streaming-related elements after click:');
                                    streamingIndicators.forEach(el => {
                                        console.log(`  "${el.text}" (class: ${el.className}, id: ${el.id})`);
                                    });
                                }
                                
                                return; // Exit after finding and testing the button
                                
                            } catch (clickError) {
                                console.log(`❌ Error clicking Start Stream button: ${clickError.message}`);
                            }
                        }
                        
                        // Look for any stream-related content on this page
                        const streamContent = await page.$$eval('*', elements => 
                            Array.from(elements)
                                .filter(el => {
                                    const text = el.textContent?.toLowerCase() || '';
                                    return text.includes('stream') || text.includes('real-time') || 
                                           text.includes('monitor') || text.includes('deepconf');
                                })
                                .slice(0, 10)
                                .map(el => ({
                                    tagName: el.tagName,
                                    text: el.textContent?.trim()?.substring(0, 100),
                                    className: el.className
                                }))
                        );
                        
                        if (streamContent.length > 0) {
                            console.log(`Found stream-related content on page ${i}:`);
                            streamContent.forEach(content => {
                                console.log(`  <${content.tagName}>: "${content.text}" (${content.className})`);
                            });
                        }
                    }
                } catch (error) {
                    console.log(`Error with icon ${i}: ${error.message}`);
                }
            }
        }
        
        console.log('❌ Start Stream button not found in any of the checked sections');
        
        // Final attempt - search entire page for any button with "stream" in text
        const allStreamButtons = await page.$$eval('button, input[type="button"]', 
            buttons => buttons
                .filter(btn => {
                    const text = btn.textContent?.toLowerCase() || btn.value?.toLowerCase() || '';
                    return text.includes('stream') || text.includes('start');
                })
                .map((btn, i) => ({
                    index: i,
                    text: btn.textContent || btn.value,
                    className: btn.className,
                    id: btn.id,
                    disabled: btn.disabled
                }))
        );
        
        if (allStreamButtons.length > 0) {
            console.log('\nFound buttons with "stream" or "start" in text:');
            allStreamButtons.forEach(btn => {
                console.log(`  "${btn.text}" (disabled: ${btn.disabled}, class: ${btn.className})`);
            });
        }
        
    } catch (error) {
        console.log(`❌ Error during test: ${error.message}`);
        await page.screenshot({ path: 'error-state.png', fullPage: true });
    } finally {
        await browser.close();
        console.log('\n✅ Test completed. Check screenshots for visual confirmation.');
    }
}

findMonitoringSection().catch(console.error);