#!/usr/bin/env python3
"""
Simple UI investigation for Archon frontend
Fixed Unicode encoding issues and focused on DeepConf + knowledge base
"""
import asyncio
import json
import time
import os
import sys
from datetime import datetime
from playwright.async_api import async_playwright

# Fix Unicode encoding for Windows console
os.environ['PYTHONIOENCODING'] = 'utf-8'

async def test_archon_frontend():
    """Test Archon frontend with focus on DeepConf and knowledge base"""
    print("=" * 50)
    print("ARCHON UI INVESTIGATION")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    findings = {
        'timestamp': datetime.now().isoformat(),
        'issues': [],
        'console_errors': [],
        'screenshots': [],
        'test_results': {}
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # Headed mode as requested
            slow_mo=1500,
            args=['--start-maximized', '--disable-web-security']
        )
        
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Capture console messages without Unicode issues
        console_messages = []
        def capture_console(msg):
            try:
                console_messages.append({
                    'type': msg.type,
                    'text': str(msg.text).encode('ascii', 'ignore').decode('ascii'),
                    'timestamp': datetime.now().isoformat()
                })
                print(f"CONSOLE {msg.type}: {msg.text}")
            except Exception as e:
                print(f"Console capture error: {e}")
        
        page.on('console', capture_console)
        
        try:
            print("\n1. Testing frontend accessibility...")
            await page.goto('http://localhost:3737', wait_until='domcontentloaded', timeout=15000)
            await page.wait_for_timeout(3000)
            
            # Take initial screenshot
            screenshot_dir = "C:\\Jarvis\\AI Workspace\\Archon\\investigation_screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            
            main_screenshot = os.path.join(screenshot_dir, f"main_page_{int(time.time())}.png")
            await page.screenshot(path=main_screenshot, full_page=True)
            findings['screenshots'].append(main_screenshot)
            print(f"Screenshot: {main_screenshot}")
            
            # Check page title
            title = await page.title()
            print(f"Page title: {title}")
            findings['test_results']['title'] = title
            
            print("\n2. Looking for DeepConf dashboard...")
            
            # Search for DeepConf elements using various selectors
            deepconf_found = False
            deepconf_selectors = [
                'text=DeepConf',
                'text=Deep Conf',
                '[data-testid*="deepconf"]',
                '[class*="deepconf"]',
                'a[href*="deepconf"]',
                'button:has-text("DeepConf")'
            ]
            
            for selector in deepconf_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        print(f"Found DeepConf element: {selector} ({len(elements)} elements)")
                        deepconf_found = True
                        
                        # Try clicking on first element
                        try:
                            await elements[0].click(timeout=3000)
                            await page.wait_for_timeout(2000)
                            
                            deepconf_screenshot = os.path.join(screenshot_dir, f"deepconf_{int(time.time())}.png")
                            await page.screenshot(path=deepconf_screenshot, full_page=True)
                            findings['screenshots'].append(deepconf_screenshot)
                            print(f"DeepConf screenshot: {deepconf_screenshot}")
                            
                        except Exception as e:
                            print(f"Could not interact with DeepConf: {e}")
                            findings['issues'].append(f"DeepConf interaction failed: {e}")
                        break
                        
                except Exception as e:
                    print(f"Selector {selector} failed: {e}")
            
            if not deepconf_found:
                print("No DeepConf elements found")
                findings['issues'].append("DeepConf dashboard not found")
            
            findings['test_results']['deepconf_found'] = deepconf_found
            
            print("\n3. Testing knowledge base functionality...")
            
            # Go back to main page for knowledge base testing
            await page.goto('http://localhost:3737', wait_until='domcontentloaded')
            await page.wait_for_timeout(2000)
            
            # Look for knowledge base or search functionality
            kb_selectors = [
                'input[placeholder*="search" i]',
                'input[placeholder*="Search" i]',
                'text=Knowledge',
                'text=Search',
                'text=RAG',
                '[data-testid*="search"]',
                'button:has-text("Search")'
            ]
            
            kb_found = False
            for selector in kb_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        print(f"Found knowledge base element: {selector}")
                        kb_found = True
                        
                        # Try interacting with knowledge base
                        try:
                            if 'input' in selector.lower():
                                await elements[0].fill('test search')
                                await page.wait_for_timeout(1000)
                                await page.keyboard.press('Enter')
                            else:
                                await elements[0].click()
                            
                            await page.wait_for_timeout(2000)
                            
                            kb_screenshot = os.path.join(screenshot_dir, f"knowledge_base_{int(time.time())}.png")
                            await page.screenshot(path=kb_screenshot, full_page=True)
                            findings['screenshots'].append(kb_screenshot)
                            print(f"Knowledge base screenshot: {kb_screenshot}")
                            
                        except Exception as e:
                            print(f"Knowledge base interaction failed: {e}")
                            findings['issues'].append(f"Knowledge base interaction failed: {e}")
                        break
                        
                except Exception as e:
                    print(f"KB selector {selector} failed: {e}")
            
            if not kb_found:
                print("No knowledge base interface found")
                findings['issues'].append("Knowledge base interface not found")
                
            findings['test_results']['knowledge_base_found'] = kb_found
            
            print("\n4. Testing API connectivity...")
            
            # Test key API endpoints through browser
            api_tests = {
                'health': 'http://localhost:8181/api/health',
                'knowledge_items': 'http://localhost:8181/api/knowledge/items'
            }
            
            for test_name, endpoint in api_tests.items():
                try:
                    response = await page.evaluate(f'''
                        fetch('{endpoint}')
                        .then(r => ({{
                            status: r.status, 
                            ok: r.ok,
                            statusText: r.statusText
                        }}))
                        .catch(e => ({{error: e.message}}))
                    ''')
                    print(f"API {test_name}: {response}")
                    findings['test_results'][f'api_{test_name}'] = response
                    
                    if not response.get('ok', False):
                        findings['issues'].append(f"API {test_name} failed: {response}")
                        
                except Exception as e:
                    print(f"API test {test_name} failed: {e}")
                    findings['issues'].append(f"API test {test_name} failed: {e}")
            
            print("\n5. Checking for error messages on page...")
            
            # Look for visible error messages
            error_text_found = []
            error_indicators = [
                'text=Error',
                'text=Failed',
                'text=Connection',
                'text=Unable',
                '[class*="error"]',
                '[role="alert"]',
                '.text-red-500',
                '.text-danger'
            ]
            
            for indicator in error_indicators:
                try:
                    elements = await page.query_selector_all(indicator)
                    for element in elements:
                        text = await element.text_content()
                        if text and text.strip():
                            error_text_found.append(text.strip())
                            print(f"Error message found: {text.strip()}")
                except:
                    pass
            
            findings['test_results']['visible_errors'] = error_text_found
            
            print("\n6. Navigation testing...")
            
            # Look for main navigation
            nav_elements = await page.query_selector_all('nav a, .nav-link, [role="navigation"] a')
            nav_links = []
            
            for element in nav_elements:
                try:
                    text = await element.text_content()
                    href = await element.get_attribute('href')
                    if text and text.strip():
                        nav_links.append({'text': text.strip(), 'href': href})
                except:
                    pass
            
            print(f"Found {len(nav_links)} navigation links: {[link['text'] for link in nav_links[:5]]}")
            findings['test_results']['navigation_links'] = len(nav_links)
            
            # Final screenshot showing current state
            final_screenshot = os.path.join(screenshot_dir, f"final_state_{int(time.time())}.png")
            await page.screenshot(path=final_screenshot, full_page=True)
            findings['screenshots'].append(final_screenshot)
            
        except Exception as e:
            print(f"Major error during testing: {e}")
            findings['issues'].append(f"Critical test error: {e}")
        
        finally:
            # Add console messages to findings
            findings['console_errors'] = console_messages
            
            # Keep browser open for manual inspection as requested
            print("\n" + "=" * 50)
            print("INVESTIGATION COMPLETE")
            print("Browser will stay open for 15 seconds for manual inspection...")
            print("=" * 50)
            
            await page.wait_for_timeout(15000)
            await browser.close()
    
    # Save findings
    findings_file = f"C:\\Jarvis\\AI Workspace\\Archon\\ui_findings_{int(time.time())}.json"
    with open(findings_file, 'w', encoding='utf-8') as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    
    print(f"\nFindings saved to: {findings_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Screenshots captured: {len(findings['screenshots'])}")
    print(f"Issues identified: {len(findings['issues'])}")
    print(f"Console messages: {len(findings['console_errors'])}")
    print(f"DeepConf found: {findings['test_results'].get('deepconf_found', False)}")
    print(f"Knowledge base found: {findings['test_results'].get('knowledge_base_found', False)}")
    
    if findings['issues']:
        print("\nKEY ISSUES:")
        for i, issue in enumerate(findings['issues'], 1):
            print(f"{i}. {issue}")
    
    return findings

if __name__ == "__main__":
    # Set console encoding
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    
    results = asyncio.run(test_archon_frontend())