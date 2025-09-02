#!/usr/bin/env python3
"""
UI/UX Validation Agent Example Usage

This script demonstrates how to use the UI/UX Validator Agent to perform
comprehensive testing of web applications.

Usage:
    python ui_ux_validation_example.py [URL] [--mode MODE] [--output-dir DIR]

Examples:
    # Quick scan of a website
    python ui_ux_validation_example.py https://example.com --mode quick_scan

    # Comprehensive validation with output directory
    python ui_ux_validation_example.py https://example.com --mode comprehensive --output-dir ./validation_results

    # Focused validation on specific component
    python ui_ux_validation_example.py https://example.com --mode focused --component "navigation menu"
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import our agent
sys.path.append(str(Path(__file__).parent.parent))

from ui_ux_validator_agent import (
    UIUXValidatorAgent, 
    ValidationMode, 
    ViewportConfig,
    validate_ui_ux
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_quick_validation_example():
    """Example: Quick validation scan"""
    print("\n" + "="*60)
    print("🚀 EXAMPLE 1: Quick UI/UX Validation Scan")
    print("="*60)
    
    url = "https://example.com"
    print(f"Validating: {url}")
    
    try:
        # Run quick validation
        result = await validate_ui_ux(
            target_url=url,
            mode=ValidationMode.QUICK_SCAN
        )
        
        print(f"\n✅ Validation completed!")
        print(f"Status: {result.pass_fail_status}")
        print(f"Issues found: {sum(result.issues_count.values())}")
        
        if result.critical_issues:
            print(f"\n🚨 Critical Issues ({len(result.critical_issues)}):")
            for issue in result.critical_issues[:3]:  # Show first 3
                print(f"  - {issue.title}: {issue.description}")
                
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        logger.error(f"Quick validation error: {e}")


async def run_comprehensive_validation_example():
    """Example: Comprehensive validation with multiple viewports"""
    print("\n" + "="*60)
    print("🔍 EXAMPLE 2: Comprehensive UI/UX Validation")
    print("="*60)
    
    url = "https://github.com"
    print(f"Validating: {url}")
    
    # Create custom viewport configurations
    custom_viewports = [
        ViewportConfig(name="Desktop", width=1920, height=1080, device_type="desktop"),
        ViewportConfig(name="Tablet", width=768, height=1024, device_type="tablet"),
        ViewportConfig(name="Mobile", width=375, height=667, device_type="mobile"),
    ]
    
    try:
        # Initialize agent
        agent = UIUXValidatorAgent()
        
        # Run comprehensive validation
        result = await agent.run_validation(
            target_url=url,
            validation_mode=ValidationMode.COMPREHENSIVE,
            viewports_to_test=custom_viewports,
            browsers_to_test=["chromium"],
            wcag_level="AA"
        )
        
        print(f"\n✅ Comprehensive validation completed!")
        print(f"Status: {result.pass_fail_status}")
        print(f"Total issues: {sum(result.issues_count.values())}")
        
        if result.validation_report:
            report = result.validation_report
            print(f"Accessibility score: {report.accessibility_score}/100")
            print(f"Viewports tested: {', '.join(report.viewports_tested)}")
            print(f"Browsers tested: {', '.join(report.browsers_tested)}")
            
            if report.recommendations:
                print("\n💡 Top Recommendations:")
                for rec in report.recommendations[:3]:
                    print(f"  - {rec}")
                    
    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")
        logger.error(f"Comprehensive validation error: {e}")


async def run_focused_validation_example():
    """Example: Focused validation on specific components"""
    print("\n" + "="*60)
    print("🎯 EXAMPLE 3: Focused Component Validation")
    print("="*60)
    
    url = "https://www.w3.org/WAI/WCAG21/quickref/"
    print(f"Validating forms on: {url}")
    
    try:
        # Initialize agent
        agent = UIUXValidatorAgent()
        
        # Run focused validation on forms
        result = await agent.run_validation(
            target_url=url,
            validation_mode=ValidationMode.FOCUSED,
            focus_component="search and filter forms"
        )
        
        print(f"\n✅ Focused validation completed!")
        print(f"Status: {result.pass_fail_status}")
        
        # Show form-specific issues
        if result.validation_report:
            form_issues = [
                issue for issue in result.validation_report.issues 
                if "form" in issue.category.lower()
            ]
            
            if form_issues:
                print(f"\n📝 Form Issues Found ({len(form_issues)}):")
                for issue in form_issues[:3]:
                    print(f"  - {issue.severity.upper()}: {issue.title}")
                    print(f"    {issue.description}")
                    if issue.recommendations:
                        print(f"    💡 Fix: {issue.recommendations[0]}")
                    print()
                    
    except Exception as e:
        print(f"❌ Focused validation failed: {e}")
        logger.error(f"Focused validation error: {e}")


async def run_visual_regression_example():
    """Example: Visual regression testing"""
    print("\n" + "="*60)
    print("👀 EXAMPLE 4: Visual Regression Testing")
    print("="*60)
    
    url = "https://example.com"
    baseline_path = "/tmp/baseline_screenshot.png"
    
    print(f"Testing visual regression for: {url}")
    print(f"Baseline: {baseline_path}")
    
    try:
        # Initialize agent
        agent = UIUXValidatorAgent()
        
        # First, create a baseline (in real usage, this would be done once)
        print("📸 Creating baseline screenshot...")
        
        async with agent:
            # Navigate and take baseline screenshot
            from ..mcp_client import get_mcp_client
            
            async with await get_mcp_client() as mcp:
                await mcp.call_tool(
                    "navigate_to",
                    url=url,
                    browser_type="chromium",
                    headless=True
                )
                
                baseline_result = await mcp.call_tool(
                    "screenshot",
                    full_page=True,
                    format="png",
                    path=baseline_path
                )
                
                print("✅ Baseline created")
                
                # Now perform comparison (simulating a later run)
                print("🔍 Performing visual comparison...")
                
                comparison_result = await mcp.call_tool(
                    "visual_compare",
                    baseline_path=baseline_path,
                    threshold=0.1
                )
                
                print("✅ Visual comparison completed")
                print(f"Result: {comparison_result}")
                
    except Exception as e:
        print(f"❌ Visual regression testing failed: {e}")
        logger.error(f"Visual regression error: {e}")


async def run_accessibility_audit_example():
    """Example: Detailed accessibility audit"""
    print("\n" + "="*60)
    print("♿ EXAMPLE 5: Accessibility Audit")
    print("="*60)
    
    url = "https://www.w3.org/WAI/demos/bad/"  # W3C's intentionally inaccessible demo
    print(f"Auditing accessibility of: {url}")
    
    try:
        # Initialize agent
        agent = UIUXValidatorAgent()
        
        # Run validation focused on accessibility
        result = await agent.run_validation(
            target_url=url,
            validation_mode=ValidationMode.COMPREHENSIVE,
            wcag_level="AAA"  # Highest standard
        )
        
        print(f"\n✅ Accessibility audit completed!")
        
        if result.validation_report:
            report = result.validation_report
            print(f"Accessibility Score: {report.accessibility_score}/100")
            
            # Show accessibility-specific issues
            a11y_issues = [
                issue for issue in report.issues 
                if issue.wcag_criteria
            ]
            
            if a11y_issues:
                print(f"\n♿ Accessibility Issues ({len(a11y_issues)}):")
                
                # Group by WCAG criteria
                wcag_groups = {}
                for issue in a11y_issues:
                    wcag = issue.wcag_criteria
                    if wcag not in wcag_groups:
                        wcag_groups[wcag] = []
                    wcag_groups[wcag].append(issue)
                
                for wcag_criteria, issues in wcag_groups.items():
                    print(f"\n  📋 WCAG {wcag_criteria}:")
                    for issue in issues[:2]:  # Show first 2 per criteria
                        print(f"    🔸 {issue.severity.upper()}: {issue.title}")
                        print(f"      {issue.description}")
                        
    except Exception as e:
        print(f"❌ Accessibility audit failed: {e}")
        logger.error(f"Accessibility audit error: {e}")


async def run_performance_analysis_example():
    """Example: Performance analysis"""
    print("\n" + "="*60)
    print("⚡ EXAMPLE 6: Performance Analysis")
    print("="*60)
    
    url = "https://web.dev"  # Google's web performance site
    print(f"Analyzing performance of: {url}")
    
    try:
        # Initialize agent
        agent = UIUXValidatorAgent()
        
        # Run validation focused on performance
        result = await agent.run_validation(
            target_url=url,
            validation_mode=ValidationMode.COMPREHENSIVE
        )
        
        print(f"\n✅ Performance analysis completed!")
        
        if result.validation_report and result.validation_report.performance_metrics:
            metrics = result.validation_report.performance_metrics
            print(f"\n⚡ Performance Metrics:")
            
            # Display key metrics
            if "performance_metrics" in metrics:
                perf_data = metrics["performance_metrics"]
                if isinstance(perf_data, str):
                    perf_data = json.loads(perf_data)
                
                if "performance_metrics" in perf_data:
                    core_metrics = perf_data["performance_metrics"]
                    print(f"  🕐 Load Time: {core_metrics.get('total_load_time', 'N/A')}ms")
                    print(f"  🎨 First Paint: {core_metrics.get('first_paint', 'N/A')}ms")
                    print(f"  📄 First Contentful Paint: {core_metrics.get('first_contentful_paint', 'N/A')}ms")
                    
            # Show performance-related issues
            perf_issues = [
                issue for issue in result.validation_report.issues 
                if "performance" in issue.category.lower()
            ]
            
            if perf_issues:
                print(f"\n⚠️  Performance Issues ({len(perf_issues)}):")
                for issue in perf_issues[:3]:
                    print(f"  - {issue.severity.upper()}: {issue.title}")
                    
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        logger.error(f"Performance analysis error: {e}")


async def main():
    """Main function to run all examples"""
    parser = argparse.ArgumentParser(
        description="UI/UX Validation Agent Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "url", 
        nargs="?",
        help="URL to validate (if not provided, runs all examples)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick_scan", "comprehensive", "focused", "continuous"],
        default="comprehensive",
        help="Validation mode"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Directory to save validation results"
    )
    
    parser.add_argument(
        "--component",
        help="Component to focus on (for focused mode)"
    )
    
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run all example scenarios"
    )
    
    args = parser.parse_args()
    
    print("🎯 UI/UX Validation Agent - Example Usage")
    print("=" * 60)
    
    try:
        if args.examples or not args.url:
            # Run all examples
            print("Running all example scenarios...\n")
            
            await run_quick_validation_example()
            await run_comprehensive_validation_example()
            await run_focused_validation_example()
            await run_visual_regression_example()
            await run_accessibility_audit_example()
            await run_performance_analysis_example()
            
        else:
            # Run validation on provided URL
            print(f"Running {args.mode} validation on: {args.url}")
            
            kwargs = {}
            if args.output_dir:
                kwargs["output_directory"] = args.output_dir
            if args.component and args.mode == "focused":
                kwargs["focus_component"] = args.component
            
            result = await validate_ui_ux(
                target_url=args.url,
                mode=ValidationMode(args.mode),
                **kwargs
            )
            
            print(f"\n✅ Validation completed!")
            print(f"Status: {result.pass_fail_status}")
            print(f"Issues found: {sum(result.issues_count.values())}")
            
            if result.critical_issues:
                print(f"\n🚨 Critical Issues:")
                for issue in result.critical_issues:
                    print(f"  - {issue.title}")
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.error(f"Main error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())