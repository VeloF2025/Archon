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
        browser = p.chromium.launch(headless=False, args=['--disable-web-security'])
        context = browser.new_context()
        page = context.new_page()
        
        # Collect console messages
        def handle_console(msg):
            timestamp = datetime.now().isoformat()
            message_data = {
                "timestamp": timestamp,
                "type": msg.type,
                "text": msg.text,
                "location": str(msg.location) if hasattr(msg, 'location') and msg.location else None
            }
            results["console_messages"].append(message_data)
            
            # Check for Socket.IO specific messages
            text = msg.text.lower()
            if 'socket' in text or 'websocket' in text or 'io' in text:
                results["connection_events"].append(message_data)
                
            if 'error' in text and ('timeout' in text or 'socket' in text or 'websocket' in text):
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
                if 'socket' in response.url.lower() or response.url.endswith('/socket.io/'):
                    results["socket_errors"].append(error_data)
        
        page.on("console", handle_console)
        page.on("response", handle_response)
        
        try:
            print("🚀 Navigating to http://localhost:3737...")
            page.goto("http://localhost:3737", wait_until="domcontentloaded", timeout=30000)
            
            print("✅ Page loaded successfully")
            results["functionality_tests"]["page_load"] = {"status": "success", "timestamp": datetime.now().isoformat()}
            
            # Wait for initial Socket.IO connection
            print("⏳ Waiting for Socket.IO initialization...")
            time.sleep(5)
            
            # Test 1: Check if knowledge base section loads
            print("🧪 Testing knowledge base functionality...")
            try:
                # Look for any main content elements
                main_content = page.locator('main, .main, [role="main"], #root').first
                if main_content.is_visible(timeout=10000):
                    results["functionality_tests"]["main_content_ui"] = {"status": "visible", "timestamp": datetime.now().isoformat()}
                    print("✅ Main content UI elements are visible")
                else:
                    results["functionality_tests"]["main_content_ui"] = {"status": "not_visible", "timestamp": datetime.now().isoformat()}
                    print("⚠️  Main content UI elements not immediately visible")
            except Exception as e:
                results["functionality_tests"]["main_content_ui"] = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
                print(f"❌ Main content test error: {e}")
            
            # Monitor console for 60 seconds
            print("📊 Starting 60-second monitoring period...")
            start_time = time.time()
            socket_found = False
            
            while time.time() - start_time < 60:
                # Check for specific Socket.IO patterns in the page
                try:
                    # Execute JavaScript to check Socket.IO status
                    socket_status = page.evaluate("""
                        () => {
                            const result = { timestamp: new Date().toISOString() };
                            
                            // Check for Socket.IO library
                            if (typeof io !== 'undefined') {
                                result.io_library = 'found';
                            } else {
                                result.io_library = 'not_found';
                            }
                            
                            // Check for socket instance
                            if (typeof window !== 'undefined' && window.socket) {
                                result.socket_instance = 'found';
                                result.socket_connected = window.socket.connected;
                                result.socket_id = window.socket.id;
                                
                                if (window.socket.io && window.socket.io.engine) {
                                    result.transport = window.socket.io.engine.transport.name;
                                }
                            } else {
                                result.socket_instance = 'not_found';
                            }
                            
                            return result;
                        }
                    """)
                    
                    if not socket_found and socket_status.get("socket_instance") == "found":
                        socket_found = True
                        results["functionality_tests"]["socket_connection"] = {
                            "status": "found", 
                            "details": socket_status,
                            "timestamp": datetime.now().isoformat()
                        }
                        print(f"✅ Socket.IO instance found: {socket_status}")
                        
                    if socket_found and socket_status.get("socket_connected"):
                        results["functionality_tests"]["socket_connection"]["connected"] = True
                        print(f"🔌 Socket.IO connected successfully")
                        
                except Exception as e:
                    print(f"Debug: Socket status check error: {e}")
                
                # Wait 2 seconds between checks
                time.sleep(2)
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:  # Progress update every 10 seconds
                    print(f"⏱️  Monitoring progress: {int(elapsed)}/60 seconds")
            
            print("✅ 60-second monitoring completed")
            
            # Final comprehensive check
            print("🔍 Performing final Socket.IO status check...")
            try:
                final_status = page.evaluate("""
                    () => {
                        const result = {
                            timestamp: new Date().toISOString(),
                            window_properties: []
                        };
                        
                        // Check all window properties for socket-related items
                        for (let prop in window) {
                            if (prop.toLowerCase().includes('socket') || prop.toLowerCase().includes('io')) {
                                result.window_properties.push(prop);
                            }
                        }
                        
                        // Specific checks
                        if (typeof io !== 'undefined') {
                            result.io_available = true;
                        }
                        
                        if (window.socket) {
                            result.socket = {
                                connected: window.socket.connected,
                                id: window.socket.id,
                                transport: window.socket.io?.engine?.transport?.name || 'unknown'
                            };
                        }
                        
                        return result;
                    }
                """)
                
                results["functionality_tests"]["final_comprehensive_check"] = {
                    "status": "completed",
                    "details": final_status,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"📋 Final check results: {json.dumps(final_status, indent=2)}")
                
            except Exception as e:
                results["functionality_tests"]["final_comprehensive_check"] = {
                    "status": "error", 
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"❌ Final check error: {e}")
            
        except Exception as e:
            print(f"❌ Test execution error: {e}")
            results["functionality_tests"]["execution_error"] = {"error": str(e), "timestamp": datetime.now().isoformat()}
        
        finally:
            print("🔄 Closing browser...")
            browser.close()
    
    results["test_end"] = datetime.now().isoformat()
    return results

def main():
    print("Starting Socket.IO timeout fix testing...")
    test_results = test_socket_timeout_fixes()

    # Analyze results
    print("\n" + "="*80)
    print("📋 SOCKET.IO TIMEOUT FIX TEST RESULTS")
    print("="*80)

    print(f"\n🕐 Test Duration: {test_results['test_start']} to {test_results['test_end']}")
    print(f"📊 Console Messages Captured: {len(test_results['console_messages'])}")
    print(f"🔌 Socket Connection Events: {len(test_results['connection_events'])}")
    print(f"❌ Socket/Timeout Errors: {len(test_results['socket_errors'])}")

    print(f"\n🧪 Functionality Test Results:")
    for test_name, test_result in test_results['functionality_tests'].items():
        status = test_result.get('status', 'unknown')
        print(f"  - {test_name}: {status}")
        if 'details' in test_result:
            print(f"    Details: {test_result['details']}")

    print(f"\n📝 Analysis:")
    
    # Check for timeout errors
    timeout_errors = [err for err in test_results['socket_errors'] if 'timeout' in str(err).lower()]
    if timeout_errors:
        print(f"❌ TIMEOUT ERRORS STILL PRESENT ({len(timeout_errors)} found):")
        for error in timeout_errors[:3]:  # Show first 3
            print(f"  - {error}")
    else:
        print(f"✅ NO TIMEOUT ERRORS DETECTED - Fix appears successful!")

    # Check for Socket.IO connection events
    if test_results['connection_events']:
        print(f"\n🔌 Socket Connection Events ({len(test_results['connection_events'])} total):")
        for event in test_results['connection_events'][:5]:  # Show first 5 events
            print(f"  - [{event['timestamp']}] [{event['type']}] {event['text']}")

    # Check overall socket functionality
    socket_tests = [name for name in test_results['functionality_tests'].keys() if 'socket' in name.lower()]
    if socket_tests:
        print(f"\n🔧 Socket Functionality:")
        for test_name in socket_tests:
            result = test_results['functionality_tests'][test_name]
            print(f"  - {test_name}: {result.get('status', 'unknown')}")

    # Save detailed results
    with open('socket_timeout_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
        
    print(f"\n💾 Detailed results saved to: socket_timeout_test_results.json")

    # Summary conclusion
    print(f"\n🎯 CONCLUSION:")
    if not timeout_errors and test_results['connection_events']:
        print("✅ Socket.IO timeout fixes appear to be working correctly!")
        print("✅ Connection events detected without timeout errors")
    elif not timeout_errors:
        print("⚠️  No timeout errors, but limited Socket.IO activity detected")
        print("   This could indicate the fixes are working or connections aren't being established")
    else:
        print("❌ Timeout errors still present - further investigation needed")

    print("="*80)

if __name__ == "__main__":
    main()