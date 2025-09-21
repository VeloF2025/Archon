#!/usr/bin/env python3
"""
Security and Compliance Testing Runner

This script runs the complete security and compliance test suite with comprehensive reporting.
"""

import pytest
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_security_tests():
    """Run all security and compliance tests with comprehensive reporting"""

    print("Security and Compliance Test Suite")
    print("=" * 60)

    # Test results storage
    test_results = {
        "start_time": datetime.now().isoformat(),
        "framework_tests": {},
        "compliance_tests": {},
        "api_tests": {},
        "performance_tests": {},
        "summary": {}
    }

    # Test directories
    test_dirs = [
        "tests/security/test_security_framework.py",
        "tests/compliance/test_compliance_engine.py",
        "tests/security/test_security_api.py",
        "tests/security/test_security_performance.py"
    ]

    # Run individual test suites
    for test_file in test_dirs:
        print(f"\nRunning {test_file}...")

        try:
            # Run pytest with coverage
            start_time = time.time()

            # Configure pytest arguments
            pytest_args = [
                test_file,
                "-v",
                "--tb=short",
                "--cov=src/agents/security",
                "--cov=src/agents/compliance",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/security_compliance",
                "--junit-xml=security_compliance_results.xml"
            ]

            # Add performance test markers
            if "performance" in test_file:
                pytest_args.extend(["-m", "performance"])

            # Run tests
            result = pytest.main(pytest_args)
            execution_time = time.time() - start_time

            # Store results
            category = test_file.split("/")[-1].replace("test_", "").replace(".py", "")
            test_results["summary"][category] = {
                "result": "PASSED" if result == 0 else "FAILED",
                "execution_time": execution_time,
                "exit_code": result
            }

            print(f"PASS {test_file} completed in {execution_time:.2f}s")

        except Exception as e:
            print(f"FAIL Error running {test_file}: {e}")
            test_results["summary"][category] = {
                "result": "ERROR",
                "execution_time": 0,
                "error": str(e)
            }

    # Generate comprehensive report
    generate_comprehensive_report(test_results)

    # Print summary
    print("\n" + "=" * 60)
    print("Test Suite Summary")
    print("=" * 60)

    total_tests = len(test_results["summary"])
    passed_tests = sum(1 for result in test_results["summary"].values() if result["result"] == "PASSED")

    for category, result in test_results["summary"].items():
        status = "PASS" if result["result"] == "PASSED" else "FAIL"
        print(f"{status} {category}: {result['result']} ({result.get('execution_time', 0):.2f}s)")

    print(f"\nOverall Results:")
    print(f"   Total Test Suites: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")

    # Return exit code
    return 0 if passed_tests == total_tests else 1

def generate_comprehensive_report(test_results: Dict[str, Any]):
    """Generate a comprehensive test report"""

    report = {
        "title": "Security and Compliance Test Report",
        "generated_at": datetime.now().isoformat(),
        "test_results": test_results,
        "recommendations": generate_recommendations(test_results)
    }

    # Save report
    report_file = "security_compliance_test_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Comprehensive report saved to {report_file}")

    # Generate HTML report
    generate_html_report(report)

def generate_recommendations(test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate recommendations based on test results"""

    recommendations = []

    # Analyze test results
    failed_suites = [name for name, result in test_results["summary"].items()
                   if result["result"] != "PASSED"]

    if failed_suites:
        recommendations.append({
            "priority": "HIGH",
            "category": "Test Failures",
            "issue": f"Failed test suites: {', '.join(failed_suites)}",
            "recommendation": "Review and fix failing tests before production deployment",
            "impact": "Critical - Security compliance cannot be verified"
        })

    # Check performance
    performance_results = test_results.get("performance_tests", {})
    if performance_results:
        slow_tests = []
        for test_name, result in performance_results.items():
            if result.get("execution_time", 0) > 5:  # 5 second threshold
                slow_tests.append(test_name)

        if slow_tests:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Performance",
                "issue": f"Slow performing tests: {', '.join(slow_tests)}",
                "recommendation": "Optimize slow performing security operations",
                "impact": "Performance - May affect system responsiveness under load"
            })

    # Coverage recommendations
    recommendations.append({
        "priority": "MEDIUM",
        "category": "Coverage",
        "issue": "Ensure comprehensive test coverage",
        "recommendation": "Maintain >95% test coverage for all security and compliance components",
        "impact": "Security - Untested code may contain vulnerabilities"
    })

    return recommendations

def generate_html_report(report_data: Dict[str, Any]):
    """Generate HTML report for better visualization"""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security and Compliance Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-suite {{
            background-color: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .passed {{
            border-left-color: #27ae60;
        }}
        .failed {{
            border-left-color: #e74c3c;
        }}
        .error {{
            border-left-color: #f39c12;
        }}
        .recommendations {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security and Compliance Test Report</h1>
        <p>Generated: {report_data['generated_at']}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{len(report_data['test_results']['summary'])}</div>
                <div>Test Suites</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sum(1 for r in report_data['test_results']['summary'].values() if r['result'] == 'PASSED')}</div>
                <div>Passed</div>
            </div>
            <div class="metric">
                <div class="metric">{sum(1 for r in report_data['test_results']['summary'].values() if r['result'] != 'PASSED')}</div>
                <div>Failed</div>
            </div>
        </div>
    </div>

    <h2>Test Suite Results</h2>
    """

    # Add test suite results
    for suite_name, result in report_data["test_results"]["summary"].items():
        status_class = result["result"].lower()
        html_content += f"""
        <div class="test-suite {status_class}">
            <h3>{suite_name.replace('_', ' ').title()}</h3>
            <p><strong>Status:</strong> {result["result"]}</p>
            <p><strong>Execution Time:</strong> {result.get('execution_time', 0):.2f}s</p>
        </div>
        """

    # Add recommendations
    if report_data.get("recommendations"):
        html_content += """
        <div class="recommendations">
            <h2>Recommendations</h2>
        """

        for rec in report_data["recommendations"]:
            priority_class = rec["priority"].lower()
            html_content += f"""
            <div>
                <h3 class="{priority_class}">{rec["priority"]}: {rec["category"]}</h3>
                <p><strong>Issue:</strong> {rec["issue"]}</p>
                <p><strong>Recommendation:</strong> {rec["recommendation"]}</p>
                <p><strong>Impact:</strong> {rec["impact"]}</p>
            </div>
            """

        html_content += "</div>"

    html_content += """
</body>
</html>
    """

    # Save HTML report
    html_file = "security_compliance_test_report.html"
    with open(html_file, "w") as f:
        f.write(html_content)

    print(f"HTML report saved to {html_file}")

def main():
    """Main function to run the test suite"""

    try:
        exit_code = run_security_tests()
        return exit_code

    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        return 1

    except Exception as e:
        print(f"\nUnexpected error running test suite: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)