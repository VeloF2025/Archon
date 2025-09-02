#!/usr/bin/env python3
"""
Quick frontend test for knowledge base functionality
"""
import asyncio
import time
import os
from datetime import datetime
from playwright.async_api import async_playwright

async def quick_test_frontend():
    """Quick test of the knowledge base frontend"""
    print("=" * 50)
    print("QUICK KNOWLEDGE BASE FRONTEND TEST")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    results = {
        'frontend_loads': False,
        'api_calls_made': False,
        'knowledge_items_displayed': False,
        'search_available': False,
        'errors_found': [],
        'screenshots': []
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # Headed mode
            slow_mo=1000,
            args=['--start-maximized']
        )
        
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Capture console errors
        console_errors = []
        def capture_console(msg):
            if msg.type == 'error':
                console_errors.append(msg.text)
                print(f"CONSOLE ERROR: {msg.text}")
        
        page.on('console', capture_console)
        
        try:
            print("\n1. Loading frontend...")
            await page.goto('http://localhost:3737', wait_until='domcontentloaded', timeout=10000)
            await page.wait_for_timeout(3000)
            results['frontend_loads'] = True
            
            # Take screenshot
            screenshot_path = f"C:\\Jarvis\\AI Workspace\\Archon\\quick_test_{int(time.time())}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            results['screenshots'].append(screenshot_path)
            print(f"Screenshot: {screenshot_path}")
            
            print(f"Page title: {await page.title()}")
            
            print("\n2. Checking for knowledge items...")
            
            # Look for knowledge items or data on page
            knowledge_selectors = [
                '[data-testid*="knowledge"]',
                '.knowledge-item',
                'div:has-text("knowledge")',
                'div:has-text("items")',
                'div:has-text("sources")',
                '.grid > div',  # Generic grid items
                'ul > li',      # List items
                'table tr'      # Table rows
            ]
            
            items_found = 0
            for selector in knowledge_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        items_found = max(items_found, len(elements))
                        print(f"Found {len(elements)} elements with selector: {selector}")
                except:
                    pass
            
            results['knowledge_items_displayed'] = items_found > 0
            print(f"Total potential knowledge items found: {items_found}")
            
            print("\n3. Checking for search functionality...")
            
            # Look for search input
            search_selectors = [
                'input[type="search"]',
                'input[placeholder*="search" i]',
                'input[placeholder*="Search"]',
                'input[name*="search"]',
                '[data-testid*="search"]'
            ]
            
            search_found = False
            for selector in search_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        search_found = True
                        print(f"Found search input: {selector}")
                        
                        # Try to use search
                        await element.fill('test')
                        await page.wait_for_timeout(1000)
                        break
                except:
                    pass
            
            results['search_available'] = search_found
            
            print("\n4. Checking API calls in network...")
            
            # Check if API calls are being made by examining network requests or console logs
            api_keywords = ['api', 'knowledge', 'fetch', 'xhr']
            api_calls_detected = False
            
            for error in console_errors:
                for keyword in api_keywords:
                    if keyword.lower() in error.lower():
                        api_calls_detected = True
                        break
            
            # Also check for loading indicators
            loading_selectors = [
                'text=Loading',
                'text=loading',
                '[class*="loading"]',
                '[class*="spinner"]',
                '.animate-spin'
            ]
            
            for selector in loading_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        api_calls_detected = True
                        print(f"Found loading indicator: {selector}")
                        break
                except:
                    pass
            
            results['api_calls_made'] = api_calls_detected
            results['errors_found'] = console_errors
            
            print("\n5. Final page state...")
            await page.wait_for_timeout(2000)
            
            # Take final screenshot
            final_screenshot = f"C:\\Jarvis\\AI Workspace\\Archon\\final_state_{int(time.time())}.png"
            await page.screenshot(path=final_screenshot, full_page=True)
            results['screenshots'].append(final_screenshot)
            
            print("Test complete. Browser will stay open for 10 seconds...")
            await page.wait_for_timeout(10000)
            
        except Exception as e:
            print(f"Test error: {e}")
            results['errors_found'].append(str(e))
        
        finally:
            await browser.close()
    
    return results

async def main():
    results = await quick_test_frontend()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Frontend loads: {results['frontend_loads']}")
    print(f"API calls detected: {results['api_calls_made']}")
    print(f"Knowledge items displayed: {results['knowledge_items_displayed']}")
    print(f"Search functionality: {results['search_available']}")
    print(f"Console errors: {len(results['errors_found'])}")
    print(f"Screenshots taken: {len(results['screenshots'])}")
    
    if results['errors_found']:
        print("\nERRORS FOUND:")
        for i, error in enumerate(results['errors_found'][:5], 1):
            print(f"{i}. {error}")
    
    # Basic analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)
    
    if not results['frontend_loads']:
        print("❌ CRITICAL: Frontend failed to load")
    elif len(results['errors_found']) > 5:
        print("⚠️  WARNING: Many JavaScript errors detected")
    elif not results['api_calls_made']:
        print("⚠️  WARNING: No API activity detected")
    elif not results['knowledge_items_displayed']:
        print("⚠️  WARNING: No knowledge items visible on page")
    else:
        print("✅ Frontend appears to be working")
    
    print("\nScreenshots available for manual inspection:")
    for screenshot in results['screenshots']:
        print(f"  - {screenshot}")

if __name__ == "__main__":
    asyncio.run(main())