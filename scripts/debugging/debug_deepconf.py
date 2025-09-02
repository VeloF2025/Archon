#!/usr/bin/env python3
"""
DeepConf Dashboard Debugging Script
Uses Playwright to troubleshoot real-time dashboard errors
"""

import os
import sys
import json
import time
from datetime import datetime
from playwright.sync_api import sync_playwright
import urllib.parse

def debug_deepconf_dashboard():
    """Debug DeepConf dashboard with headed browser and error capture"""
    
    print("DEEPCONF DASHBOARD DEBUG SESSION STARTING")
    print("=" * 50)
    
    with sync_playwright() as p:
        # Launch browser in headed mode for visual debugging
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--start-maximized',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            # Set user agent to avoid detection
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        )
        
        page = context.new_page()
        
        # Store captured data
        console_errors = []
        network_failures = []
        api_calls = []
        
        # Console message handler (safe for Windows)
        def handle_console(msg):
            try:
                msg_data = {
                    'type': msg.type,
                    'text': str(msg.text).encode('ascii', 'ignore').decode('ascii'),
                    'timestamp': time.time()
                }
                console_errors.append(msg_data)
                
                # Print safely to console
                safe_text = str(msg.text).encode('ascii', 'ignore').decode('ascii')
                print(f"[CONSOLE {msg.type.upper()}] {safe_text}")
                
            except Exception as e:
                print(f"Error capturing console message: {e}")
        
        # Network monitoring
        def handle_request(request):
            try:
                api_calls.append({
                    'url': request.url,
                    'method': request.method,
                    'timestamp': time.time()
                })
                
                if '/api/' in request.url:
                    print(f"[API REQUEST] {request.method} {request.url}")
                    
            except Exception as e:
                print(f"Error capturing request: {e}")
        
        def handle_response(response):
            try:
                if response.status >= 400:
                    failure_data = {
                        'url': response.url,
                        'status': response.status,
                        'method': 'unknown',
                        'timestamp': time.time()
                    }
                    network_failures.append(failure_data)
                    print(f"[NETWORK FAILURE] {response.status} {response.url}")
                    
            except Exception as e:
                print(f"Error capturing response: {e}")
        
        # Set up event listeners
        page.on('console', handle_console)
        page.on('request', handle_request)
        page.on('response', handle_response)
        
        try:
            print(f"\nNavigating to DeepConf dashboard...")
            print(f"URL: http://localhost:3737/deepconf")
            
            # Navigate with longer timeout
            page.goto('http://localhost:3737/deepconf', wait_until='domcontentloaded', timeout=15000)
            print("SUCCESS: Initial navigation completed")
            
            # Take initial screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot1 = f"deepconf_initial_{timestamp}.png"
            page.screenshot(path=screenshot1, full_page=True)
            print(f"SCREENSHOT: Initial screenshot: {screenshot1}")
            
            # Wait for any async loading
            print("\nWaiting for async content to load...")
            page.wait_for_timeout(8000)
            
            # Take second screenshot after loading
            screenshot2 = f"deepconf_loaded_{timestamp}.png"
            page.screenshot(path=screenshot2, full_page=True)
            print(f"SCREENSHOT: Loaded screenshot: {screenshot2}")
            
            # Try to interact with elements to trigger API calls
            print("\nAttempting to interact with dashboard elements...")
            
            # Look for any buttons or interactive elements
            try:
                # Check if there are any buttons to click
                buttons = page.locator('button').all()
                print(f"Found {len(buttons)} buttons on page")
                
                if buttons:
                    # Click first button to potentially trigger API calls
                    buttons[0].click()
                    page.wait_for_timeout(3000)
                    
                    # Take screenshot after interaction
                    screenshot3 = f"deepconf_interaction_{timestamp}.png"
                    page.screenshot(path=screenshot3, full_page=True)
                    print(f"SCREENSHOT: Interaction screenshot: {screenshot3}")
                    
            except Exception as e:
                print(f"Could not interact with elements: {e}")
            
            # Test direct API calls from browser console
            print("\nTesting API endpoints directly from browser...")
            
            api_test_results = []
            endpoints = [
                '/api/confidence/system',
                '/api/confidence/health', 
                '/api/confidence/history',
                '/api/confidence/scwt'
            ]
            
            for endpoint in endpoints:
                try:
                    # Test via browser fetch
                    result = page.evaluate(f"""
                        fetch('{endpoint}')
                            .then(response => {{
                                return {{
                                    status: response.status,
                                    ok: response.ok,
                                    url: response.url
                                }};
                            }})
                            .catch(error => {{
                                return {{
                                    error: error.message,
                                    url: '{endpoint}'
                                }};
                            }});
                    """)
                    
                    api_test_results.append({
                        'endpoint': endpoint,
                        'result': result
                    })
                    print(f"[API TEST] {endpoint}: {result}")
                    
                except Exception as e:
                    print(f"[API TEST ERROR] {endpoint}: {e}")
            
            # Final analysis
            print("\n" + "="*50)
            print("DEBUGGING ANALYSIS COMPLETE")
            print("="*50)
            
            print(f"\nCONSOLE ERRORS ({len(console_errors)} total):")
            for i, error in enumerate(console_errors[-10:], 1):  # Last 10 errors
                print(f"  {i}. [{error['type'].upper()}] {error['text'][:100]}...")
            
            print(f"\nNETWORK FAILURES ({len(network_failures)} total):")
            for i, failure in enumerate(network_failures, 1):
                print(f"  {i}. {failure['status']} {failure['url']}")
            
            print(f"\nAPI CALLS ({len(api_calls)} total):")
            api_only = [call for call in api_calls if '/api/' in call['url']]
            for i, call in enumerate(api_only, 1):
                print(f"  {i}. {call['method']} {call['url']}")
            
            print(f"\nDIRECT API TESTS:")
            for test in api_test_results:
                print(f"  {test['endpoint']}: {test['result']}")
            
            # Save debug data to file
            debug_data = {
                'console_errors': console_errors,
                'network_failures': network_failures,
                'api_calls': api_calls,
                'api_test_results': api_test_results,
                'screenshots': [screenshot1, screenshot2],
                'timestamp': timestamp
            }
            
            with open(f'deepconf_debug_{timestamp}.json', 'w') as f:
                json.dump(debug_data, f, indent=2)
            
            print(f"\nDEBUG DATA SAVED: deepconf_debug_{timestamp}.json")
            print("\nBrowser will remain open for 30 seconds for manual inspection...")
            
            # Keep browser open for manual inspection
            page.wait_for_timeout(30000)
            
            return debug_data
            
        except Exception as e:
            print(f"ERROR during debugging: {e}")
            # Still take a screenshot even if there's an error
            try:
                error_screenshot = f"deepconf_error_{timestamp}.png"
                page.screenshot(path=error_screenshot)
                print(f"SCREENSHOT: Error screenshot: {error_screenshot}")
            except:
                pass
            return {'error': str(e)}
        
        finally:
            try:
                browser.close()
            except:
                pass  # Ignore close errors

if __name__ == "__main__":
    result = debug_deepconf_dashboard()
    print(f"\nFINAL RESULT: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")