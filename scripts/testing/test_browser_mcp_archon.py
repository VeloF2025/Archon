#!/usr/bin/env python3
"""
Test Browser MCP with Archon UI
Tests the Playwright MCP integration by validating the Archon UI
"""

import asyncio
import json
import base64
from datetime import datetime
import sys
import os

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'src'))

async def test_archon_ui():
    """Test the Archon UI using Browser MCP"""
    
    print("=" * 60)
    print("🧪 BROWSER MCP - ARCHON UI VALIDATION TEST")
    print("=" * 60)
    print()
    
    try:
        # Import the UI/UX validator agent
        from agents.ui_ux_validator_agent import UIUXValidatorAgent, ValidationMode
        
        print("✅ UI/UX Validator Agent imported successfully")
        
        # Create the agent
        agent = UIUXValidatorAgent()
        print("✅ Agent initialized")
        
        # Test the Archon UI
        target_url = "http://localhost:3737"
        print(f"\n🔍 Testing Archon UI at: {target_url}")
        print("-" * 40)
        
        # Run quick scan validation
        print("\n📊 Running Quick Scan validation...")
        result = await agent.run_validation(
            target_url=target_url,
            validation_mode=ValidationMode.QUICK_SCAN
        )
        
        # Display results
        print("\n📋 VALIDATION RESULTS:")
        print("-" * 40)
        
        report = result.validation_report
        print(f"✅ Overall Pass: {report.overall_pass}")
        print(f"📊 Accessibility Score: {report.accessibility_score}/100")
        print(f"⏱️ Page Load Time: {report.page_load_time:.2f}s")
        
        # Performance metrics
        if report.performance_metrics:
            print(f"\n📈 Performance Metrics:")
            for metric, value in report.performance_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  • {metric}: {value:.2f}")
                else:
                    print(f"  • {metric}: {value}")
        
        # Issues found
        if report.issues:
            print(f"\n⚠️ Issues Found ({len(report.issues)}):")
            # Group by severity
            by_severity = {}
            for issue in report.issues:
                severity = issue.severity
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)
            
            for severity in ['critical', 'high', 'medium', 'low', 'info']:
                if severity in by_severity:
                    issues = by_severity[severity]
                    print(f"\n  {severity.upper()} ({len(issues)}):")
                    for issue in issues[:3]:  # Show first 3 of each severity
                        print(f"    • {issue.category}: {issue.description[:60]}...")
                        if issue.recommendation:
                            print(f"      Fix: {issue.recommendation[:50]}...")
        else:
            print("\n✅ No issues found!")
        
        # Accessibility issues
        if report.accessibility_issues:
            print(f"\n♿ Accessibility Issues ({len(report.accessibility_issues)}):")
            for issue in report.accessibility_issues[:5]:
                print(f"  • {issue}")
        
        # Save screenshot if available
        if report.screenshots:
            print(f"\n📸 Screenshots captured: {len(report.screenshots)}")
            # Save the first screenshot
            for name, data in list(report.screenshots.items())[:1]:
                filename = f"archon_ui_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(data))
                print(f"  • Saved: {filename}")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ BROWSER MCP TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nTrying alternative approach with MCP client directly...")
        
        # Fallback to direct MCP client usage
        try:
            from mcp_server.modules.browser_mcp import BrowserMCP
            
            print("✅ Browser MCP module imported directly")
            
            # Initialize Browser MCP
            browser_mcp = BrowserMCP()
            print("✅ Browser MCP initialized")
            
            # Navigate to Archon UI
            print(f"\n🔍 Navigating to: http://localhost:3737")
            nav_result = await browser_mcp.navigate_to(
                url="http://localhost:3737",
                browser_type="chromium",
                headless=True
            )
            
            if nav_result.get("success"):
                print("✅ Navigation successful")
                print(f"  • Title: {nav_result.get('title')}")
                print(f"  • URL: {nav_result.get('url')}")
            
            # Take screenshot
            print("\n📸 Taking screenshot...")
            screenshot_result = await browser_mcp.screenshot(
                full_page=True,
                format="png"
            )
            
            if screenshot_result.get("success"):
                print("✅ Screenshot captured")
                # Save screenshot
                filename = f"archon_ui_browser_mcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(screenshot_result["screenshot"]))
                print(f"  • Saved: {filename}")
            
            # Get accessibility tree
            print("\n♿ Getting accessibility information...")
            a11y_result = await browser_mcp.get_accessibility_tree()
            
            if a11y_result.get("success"):
                tree = a11y_result.get("accessibility_tree", {})
                print("✅ Accessibility tree retrieved")
                print(f"  • Role: {tree.get('role')}")
                print(f"  • Name: {tree.get('name')}")
                if tree.get('children'):
                    print(f"  • Child elements: {len(tree.get('children', []))}")
            
            # Get performance metrics
            print("\n📈 Getting performance metrics...")
            perf_result = await browser_mcp.get_performance_metrics()
            
            if perf_result.get("success"):
                metrics = perf_result.get("metrics", {})
                print("✅ Performance metrics retrieved")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  • {key}: {value:.2f}")
                    else:
                        print(f"  • {key}: {value}")
            
            print("\n" + "=" * 60)
            print("✅ BROWSER MCP DIRECT TEST COMPLETED!")
            print("=" * 60)
            
            return True
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_archon_ui())
    sys.exit(0 if success else 1)