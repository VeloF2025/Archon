#!/usr/bin/env python3
"""
Test Socket.IO timeout fixes by monitoring console messages and connection behavior
"""

from playwright.sync_api import sync_playwright
import time
import json
from datetime import datetime
import sys

def test_socket_timeout_fixes():
    """
    Test the Socket.IO timeout fixes by monitoring console messages
    for 60 seconds and testing basic functionality
    """
    
    results = {
        "test_start": datetime.now().isoformat(),
        "console_messages": [],
        "socket_errors": [],
        "connection_events": [],
        "functionality_tests": {},
        "duration_seconds": 60
    }
    
    with sync_playwright() as p:
        # Launch browser with console access
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        # Collect console messages
        def handle_console(msg):
            timestamp = datetime.now().isoformat()
            message_data = {
                "timestamp": timestamp,
                "type": msg.type,
                "text": msg.text
            }
            results["console_messages"].append(message_data)
            
            # Check for Socket.IO specific messages
            text = msg.text.lower()
            if 'socket' in text or 'websocket' in text or 'io' in text:
                results["connection_events"].append(message_data)
                
            if 'error' in text and ('timeout' in text or 'socket' in text):
                results["socket_errors"].append(message_data)
                
            print(f"[{msg.type.upper()}] {msg.text}")
        
        # Collect network failures  
        def handle_response(response):
            if response.status >= 400:
                error_data = {
                    "timestamp": datetime.now().isoformat(),
                    "url": response.url,
                    "status": response.status,
                    "status_text": response.status_text
                }
                if 'socket' in response.url.lower():
                    results["socket_errors"].append(error_data)
        
        page.on("console", handle_console)
        page.on("response", handle_response)
        
        try:
            print("Navigating to http://localhost:3737...")
            page.goto("http://localhost:3737", wait_until="domcontentloaded", timeout=30000)
            
            print("Page loaded successfully")
            results["functionality_tests"]["page_load"] = {"status": "success", "timestamp": datetime.now().isoformat()}
            
            # Wait for initial Socket.IO connection
            print("Waiting for Socket.IO initialization...")
            time.sleep(5)
            
            # Monitor console for 60 seconds
            print("Starting 60-second monitoring period...")
            start_time = time.time()
            
            while time.time() - start_time < 60:
                # Check for Socket.IO status
                try:
                    socket_status = page.evaluate("""
                        () => {
                            const result = { timestamp: new Date().toISOString() };
                            
                            if (typeof window.socket !== 'undefined') {
                                result.socket_found = true;
                                result.socket_connected = window.socket.connected;
                                result.socket_id = window.socket.id;
                            } else {
                                result.socket_found = false;
                            }
                            
                            return result;
                        }
                    """)
                    
                    if socket_status.get("socket_found"):
                        results["functionality_tests"]["socket_connection"] = {
                            "status": "found", 
                            "details": socket_status,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        if socket_status.get("socket_connected"):
                            print("Socket.IO connected successfully")
                        
                except Exception as e:
                    pass
                
                time.sleep(2)
                elapsed = time.time() - start_time
                if int(elapsed) % 15 == 0:
                    print(f"Monitoring progress: {int(elapsed)}/60 seconds")
            
            print("60-second monitoring completed")
            
        except Exception as e:
            print(f"Test execution error: {e}")
            results["functionality_tests"]["execution_error"] = {"error": str(e), "timestamp": datetime.now().isoformat()}
        
        finally:
            print("Closing browser...")
            browser.close()
    
    results["test_end"] = datetime.now().isoformat()
    return results

def main():
    print("Starting Socket.IO timeout fix testing...")
    test_results = test_socket_timeout_fixes()

    print("\n" + "="*80)
    print("SOCKET.IO TIMEOUT FIX TEST RESULTS")
    print("="*80)

    print(f"\nTest Duration: {test_results['test_start']} to {test_results['test_end']}")
    print(f"Console Messages Captured: {len(test_results['console_messages'])}")
    print(f"Socket Connection Events: {len(test_results['connection_events'])}")
    print(f"Socket/Timeout Errors: {len(test_results['socket_errors'])}")

    print(f"\nFunctionality Test Results:")
    for test_name, test_result in test_results['functionality_tests'].items():
        status = test_result.get('status', 'unknown')
        print(f"  - {test_name}: {status}")

    print(f"\nAnalysis:")
    
    # Check for timeout errors
    timeout_errors = [err for err in test_results['socket_errors'] if 'timeout' in str(err).lower()]
    if timeout_errors:
        print(f"TIMEOUT ERRORS STILL PRESENT ({len(timeout_errors)} found):")
        for error in timeout_errors[:3]:
            print(f"  - {error}")
    else:
        print("NO TIMEOUT ERRORS DETECTED - Fix appears successful!")

    # Check for Socket.IO connection events
    if test_results['connection_events']:
        print(f"\nSocket Connection Events ({len(test_results['connection_events'])} total):")
        for event in test_results['connection_events'][:5]:
            print(f"  - [{event['timestamp']}] [{event['type']}] {event['text']}")

    # Save results
    with open('socket_timeout_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
        
    print(f"\nDetailed results saved to: socket_timeout_test_results.json")

    # Summary
    print(f"\nCONCLUSION:")
    if not timeout_errors and test_results['connection_events']:
        print("Socket.IO timeout fixes appear to be working correctly!")
    elif not timeout_errors:
        print("No timeout errors detected")
    else:
        print("Timeout errors still present - needs investigation")

    print("="*80)

if __name__ == "__main__":
    main()