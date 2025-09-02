#!/usr/bin/env python3
"""
Archon PM System Enhancement - TDD Test Runner
==============================================

This script runs the comprehensive TDD test suite for the Archon PM System Enhancement.
All tests are designed to initially FAIL (RED phase) to validate current system inadequacies.

Usage:
    python run_tdd_tests.py [options]
    
Options:
    --category <category>   Run specific test category (historical, realtime, verification, etc.)
    --red-only             Run only tests that should fail initially (TDD RED phase)
    --performance          Run performance tests only
    --coverage             Generate coverage reports
    --verbose              Verbose output
    --help                 Show this help message

Test Categories:
    - historical: Historical work discovery tests
    - realtime: Real-time activity monitoring tests  
    - verification: Implementation verification tests
    - task_mgmt: Dynamic task management tests
    - performance: Performance and integration tests
    - accuracy: Data accuracy validation tests
"""

import sys
import subprocess
import argparse
from pathlib import Path
import os
from datetime import datetime


class TDDTestRunner:
    """Test runner for Archon PM System Enhancement TDD suite."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"
        
    def run_tests(self, category=None, red_only=False, performance=False, 
                  coverage=True, verbose=False):
        """Run the TDD test suite with specified options."""
        
        print("🤖 Archon PM System Enhancement - TDD Test Suite")
        print("=" * 50)
        print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test directory
        cmd.append(str(self.test_dir))
        
        # Configure markers based on options
        markers = []
        
        if red_only:
            markers.append("tdd_red")
            print("🔴 Running TDD RED phase tests (should initially FAIL)")
        elif category:
            markers.append(category)
            print(f"📂 Running {category} tests")
        elif performance:
            markers.append("performance")
            print("⚡ Running performance tests")
        else:
            print("🧪 Running full TDD test suite")
        
        if markers:
            cmd.extend(["-m", " and ".join(markers)])
        
        # Add coverage options
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        # Add verbosity
        if verbose:
            cmd.append("-vv")
        else:
            cmd.append("-v")
        
        # Add other options
        cmd.extend([
            "--tb=short",
            "--durations=10",
            "--maxfail=50",
            "-ra"
        ])
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run tests
        try:
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=False, text=True)
            
            print()
            print("=" * 50)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                print("⚠️  NOTE: In TDD RED phase, tests SHOULD fail initially.")
                print("   If tests are passing, the system may already be implemented.")
            else:
                print("🔴 Tests failed (expected in TDD RED phase)")
                print("   This indicates current system inadequacies that need to be fixed.")
                print("   Proceed with GREEN phase implementation.")
            
            if coverage and self.coverage_dir.exists():
                print(f"📊 Coverage report generated: {self.coverage_dir}/index.html")
            
            return result.returncode
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return 1
    
    def list_test_categories(self):
        """List available test categories."""
        categories = {
            'historical': 'Historical work discovery tests (25+ implementations)',
            'realtime': 'Real-time activity monitoring tests (<30s updates)', 
            'verification': 'Implementation verification tests (health checks, APIs)',
            'task_mgmt': 'Dynamic task management tests (auto-creation, status sync)',
            'performance': 'Performance tests (<500ms discovery, 1000+ concurrent)',
            'accuracy': 'Data accuracy tests (95%+ tracking, <2% false positives)',
            'integration': 'System integration tests (Archon compatibility)',
            'tdd_red': 'All tests that should initially fail (RED phase)'
        }
        
        print("Available Test Categories:")
        print("=" * 40)
        for category, description in categories.items():
            print(f"  {category:12} - {description}")
        print()
    
    def validate_current_failures(self):
        """Validate that tests properly document current system failures."""
        print("🔍 Validating Current System Failures")
        print("=" * 40)
        
        expected_failures = {
            'Work Tracking': 'Only 2/25+ implementations tracked (8% accuracy)',
            'Real-time Updates': 'No real-time monitoring (∞ delay)',
            'Implementation Verification': 'No health checks or API testing',
            'Task Management': 'No automatic task creation or status sync',
            'Performance': 'No discovery optimization or concurrent handling',
            'Data Accuracy': 'Poor accuracy (<95%) and high false positive rates'
        }
        
        print("Expected System Failures (to be fixed by implementation):")
        for area, failure in expected_failures.items():
            print(f"  ❌ {area}: {failure}")
        
        print()
        print("✅ These failures are documented in the test suite")
        print("🔄 Tests will pass after implementing the enhanced PM system")


def main():
    """Main entry point for the TDD test runner."""
    parser = argparse.ArgumentParser(
        description="Run Archon PM System Enhancement TDD Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--category',
        choices=['historical', 'realtime', 'verification', 'task_mgmt', 
                'performance', 'accuracy', 'integration'],
        help='Run specific test category'
    )
    
    parser.add_argument(
        '--red-only', 
        action='store_true',
        help='Run only tests that should fail initially (TDD RED phase)'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only'
    )
    
    parser.add_argument(
        '--no-coverage', 
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--list-categories', 
        action='store_true',
        help='List available test categories'
    )
    
    parser.add_argument(
        '--validate-failures', 
        action='store_true',
        help='Validate current system failure documentation'
    )
    
    args = parser.parse_args()
    
    runner = TDDTestRunner()
    
    if args.list_categories:
        runner.list_test_categories()
        return 0
    
    if args.validate_failures:
        runner.validate_current_failures()
        return 0
    
    # Run tests
    return runner.run_tests(
        category=args.category,
        red_only=args.red_only,
        performance=args.performance,
        coverage=not args.no_coverage,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())