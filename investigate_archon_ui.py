#!/usr/bin/env python3
"""
Simple browser automation script to investigate Archon UI issues
Focuses on DeepConf dashboard and knowledge base connectivity
"""
import asyncio
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright

async def investigate_archon_ui():
    """Comprehensive UI investigation"""
    print("Starting Archon UI Investigation...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    findings = {
        'timestamp': datetime.now().isoformat(),
        'frontend_url': 'http://localhost:3737',
        'issues': [],
        'console_errors': [],
        'network_failures': [],
        'screenshots': [],
        'pages_tested': []
    }
    
    async with async_playwright() as p:
        # Launch browser in headed mode as requested
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=1000,  # Slow down for visibility
            args=['--start-maximized']
        )
        
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Collect console errors
        def handle_console(msg):
            if msg.type in ['error', 'warning']:
                findings['console_errors'].append({
                    'type': msg.type,
                    'text': msg.text,
                    'url': page.url
                })
                print(f"CONSOLE {msg.type.upper()}: {msg.text}")
        
        # Collect network failures
        def handle_response(response):
            if response.status >= 400:
                findings['network_failures'].append({
                    'url': response.url,
                    'status': response.status,
                    'method': response.request.method
                })
                print(f"NETWORK ERROR: {response.status} {response.request.method} {response.url}")
        
        page.on('console', handle_console)
        page.on('response', handle_response)
        
        try:
            # 1. Navigate to main application
            print("\n1. Navigating to Archon frontend...")
            await page.goto('http://localhost:3737', wait_until='networkidle')
            await page.wait_for_timeout(2000)
            
            # Take initial screenshot
            screenshot_path = f"C:\\Jarvis\\AI Workspace\\Archon\\screenshots\\main_page_{int(time.time())}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            findings['screenshots'].append(screenshot_path)
            findings['pages_tested'].append('Main Page')
            
            print(f"Screenshot saved: {screenshot_path}")
            
            # Check page title and basic content
            title = await page.title()
            print(f"Page title: {title}")
            
            # 2. Look for DeepConf dashboard
            print("\n2. Checking for DeepConf dashboard...")
            deepconf_elements = await page.query_selector_all('text=DeepConf')
            if deepconf_elements:
                print(f"Found {len(deepconf_elements)} DeepConf references")
                
                # Try to navigate to DeepConf section
                try:
                    await page.click('text=DeepConf', timeout=5000)
                    await page.wait_for_timeout(2000)
                    
                    deepconf_screenshot = f"C:\\Jarvis\\AI Workspace\\Archon\\screenshots\\deepconf_dashboard_{int(time.time())}.png"
                    await page.screenshot(path=deepconf_screenshot, full_page=True)
                    findings['screenshots'].append(deepconf_screenshot)
                    findings['pages_tested'].append('DeepConf Dashboard')
                    print(f"DeepConf screenshot saved: {deepconf_screenshot}")
                    
                except Exception as e:
                    print(f"Could not navigate to DeepConf: {e}")
                    findings['issues'].append(f"DeepConf navigation failed: {e}")
            else:
                print("No DeepConf elements found on main page")
                findings['issues'].append("DeepConf dashboard not visible on main page")
            
            # 3. Check knowledge base functionality
            print("\n3. Testing knowledge base...")
            
            # Look for search or knowledge base elements
            search_selectors = [
                'input[placeholder*="search"]',
                'input[placeholder*="Search"]', 
                '[data-testid*="search"]',
                'text=Knowledge',
                'text=Search'
            ]
            
            search_found = False
            for selector in search_selectors:
                elements = await page.query_selector_all(selector)
                if elements:
                    print(f"Found search element with selector: {selector}")
                    search_found = True
                    
                    try:
                        # Try to interact with search
                        if 'input' in selector:
                            await page.fill(selector, "test query")
                            await page.wait_for_timeout(1000)
                        elif 'text=' in selector:
                            await page.click(selector)
                            await page.wait_for_timeout(2000)
                        
                        kb_screenshot = f"C:\\Jarvis\\AI Workspace\\Archon\\screenshots\\knowledge_base_{int(time.time())}.png"
                        await page.screenshot(path=kb_screenshot, full_page=True)
                        findings['screenshots'].append(kb_screenshot)
                        findings['pages_tested'].append('Knowledge Base')
                        print(f"Knowledge base screenshot saved: {kb_screenshot}")
                        
                    except Exception as e:
                        print(f"Knowledge base interaction failed: {e}")
                        findings['issues'].append(f"Knowledge base interaction error: {e}")
                    break
            
            if not search_found:
                print("No search/knowledge base interface found")
                findings['issues'].append("Knowledge base interface not accessible")
            
            # 4. Check for any error messages on page
            print("\n4. Checking for visible error messages...")
            error_selectors = [
                '.error',
                '[class*="error"]',
                'text=Error',
                'text=Failed',
                'text=Connection',
                '[aria-live="polite"]',
                '[role="alert"]'
            ]
            
            for selector in error_selectors:
                elements = await page.query_selector_all(selector)
                if elements:
                    for element in elements:
                        text = await element.text_content()
                        if text and text.strip():
                            print(f"Found error message: {text}")
                            findings['issues'].append(f"Visible error message: {text}")
            
            # 5. Check network tab for API failures
            print("\n5. Testing API endpoints...")
            
            api_endpoints = [
                'http://localhost:8181/api/health',
                'http://localhost:8181/api/knowledge/items',
                'http://localhost:8181/api/deepconf/status'  # Assuming this exists
            ]
            
            for endpoint in api_endpoints:
                try:
                    response = await page.evaluate(f"""
                        fetch('{endpoint}')
                        .then(r => ({{status: r.status, ok: r.ok}}))
                        .catch(e => ({{error: e.message}}))
                    """)
                    print(f"API {endpoint}: {response}")
                    
                    if 'error' in response or not response.get('ok'):
                        findings['issues'].append(f"API endpoint {endpoint} failed: {response}")
                        
                except Exception as e:
                    print(f"Failed to test {endpoint}: {e}")
                    findings['issues'].append(f"Could not test API {endpoint}: {e}")
            
            # 6. Navigate through main sections
            print("\n6. Exploring navigation...")
            
            nav_selectors = [
                'nav a',
                '[role="navigation"] a',
                '.nav-link',
                '.menu-item'
            ]
            
            nav_links = []
            for selector in nav_selectors:
                links = await page.query_selector_all(selector)
                for link in links:
                    href = await link.get_attribute('href')
                    text = await link.text_content()
                    if href and text:
                        nav_links.append({'href': href, 'text': text.strip()})
            
            print(f"Found {len(nav_links)} navigation links")
            
            # Test a few key navigation items
            for link in nav_links[:3]:  # Test first 3 links
                try:
                    print(f"Testing navigation to: {link['text']}")
                    await page.click(f'text="{link["text"]}"')
                    await page.wait_for_timeout(2000)
                    
                    nav_screenshot = f"C:\\Jarvis\\AI Workspace\\Archon\\screenshots\\nav_{link['text'].replace(' ', '_')}_{int(time.time())}.png"
                    await page.screenshot(path=nav_screenshot, full_page=True)
                    findings['screenshots'].append(nav_screenshot)
                    findings['pages_tested'].append(f"Navigation: {link['text']}")
                    
                except Exception as e:
                    print(f"Navigation to {link['text']} failed: {e}")
                    findings['issues'].append(f"Navigation error for {link['text']}: {e}")
                    
                # Go back to main page
                try:
                    await page.goto('http://localhost:3737', wait_until='networkidle')
                    await page.wait_for_timeout(1000)
                except:
                    pass
        
        except Exception as e:
            print(f"Critical error during investigation: {e}")
            findings['issues'].append(f"Critical investigation error: {e}")
        
        finally:
            # Keep browser open for 10 seconds for manual inspection
            print("\n" + "="*60)
            print("Investigation complete! Browser will remain open for 10 seconds...")
            print("You can manually inspect the current state.")
            await page.wait_for_timeout(10000)
            
            await browser.close()
    
    # Save findings to file
    findings_file = f"C:\\Jarvis\\AI Workspace\\Archon\\ui_investigation_{int(time.time())}.json"
    with open(findings_file, 'w') as f:
        json.dump(findings, f, indent=2)
    
    print(f"\nFindings saved to: {findings_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("INVESTIGATION SUMMARY")
    print("="*60)
    print(f"Pages tested: {len(findings['pages_tested'])}")
    print(f"Screenshots captured: {len(findings['screenshots'])}")
    print(f"Console errors: {len(findings['console_errors'])}")
    print(f"Network failures: {len(findings['network_failures'])}")
    print(f"Issues identified: {len(findings['issues'])}")
    
    if findings['issues']:
        print("\nKEY ISSUES FOUND:")
        for i, issue in enumerate(findings['issues'], 1):
            print(f"{i}. {issue}")
    
    if findings['console_errors']:
        print("\nCONSOLE ERRORS:")
        for error in findings['console_errors']:
            print(f"- {error['type'].upper()}: {error['text']}")
    
    return findings

if __name__ == "__main__":
    # Create screenshots directory
    import os
    screenshots_dir = "C:\\Jarvis\\AI Workspace\\Archon\\screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # Run investigation
    findings = asyncio.run(investigate_archon_ui())