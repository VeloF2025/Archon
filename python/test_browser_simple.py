#!/usr/bin/env python3
"""Simple test of Browser MCP with Archon UI"""

import asyncio
from playwright.async_api import async_playwright
import base64
from datetime import datetime

async def test_archon_ui():
    """Test Archon UI with Playwright"""
    print("=" * 60)
    print("BROWSER MCP - ARCHON UI TEST")
    print("=" * 60)
    
    async with async_playwright() as p:
        # Launch browser
        print("\nLaunching browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 720})
        page = await context.new_page()
        
        # Navigate to Archon UI (use container name for inter-container communication)
        url = "http://archon-ui:3737"
        print(f"Navigating to: {url}")
        
        try:
            await page.goto(url, wait_until='networkidle', timeout=30000)
            print("✓ Page loaded successfully")
            
            # Get page title
            title = await page.title()
            print(f"  Page title: {title}")
            
            # Take screenshot
            print("\nTaking screenshot...")
            screenshot_bytes = await page.screenshot(full_page=True)
            filename = f"archon_ui_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(filename, 'wb') as f:
                f.write(screenshot_bytes)
            print(f"✓ Screenshot saved: {filename}")
            
            # Check for main elements
            print("\nChecking UI elements...")
            
            # Check if root element exists
            root = await page.query_selector('#root')
            if root:
                print("✓ Root element found")
            
            # Get text content
            text_content = await page.text_content('body')
            if text_content:
                print(f"✓ Page has content (length: {len(text_content)} chars)")
            
            # Check accessibility
            print("\nChecking accessibility...")
            accessibility_tree = await page.accessibility.snapshot()
            if accessibility_tree:
                print(f"✓ Accessibility tree available")
                print(f"  Root role: {accessibility_tree.get('role', 'unknown')}")
                print(f"  Root name: {accessibility_tree.get('name', 'unnamed')}")
            
            # Get performance metrics
            print("\nGetting performance metrics...")
            metrics = await page.evaluate("""() => {
                const perf = performance.getEntriesByType('navigation')[0];
                return {
                    domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
                    loadComplete: perf.loadEventEnd - perf.loadEventStart,
                    domInteractive: perf.domInteractive - perf.fetchStart,
                    responseTime: perf.responseEnd - perf.requestStart
                };
            }""")
            
            if metrics:
                print("✓ Performance metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.2f}ms")
            
            print("\n" + "=" * 60)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Take error screenshot
            try:
                error_screenshot = await page.screenshot()
                error_filename = f"archon_ui_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(error_filename, 'wb') as f:
                    f.write(error_screenshot)
                print(f"Error screenshot saved: {error_filename}")
            except:
                pass
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_archon_ui())