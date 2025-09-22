#!/usr/bin/env python3
"""
ReactFlow Integration Validation Script

This script validates the integration between the Archon workflow system
and ReactFlow components, providing comprehensive validation reports
and actionable recommendations.
"""

import asyncio
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from src.database.workflow_models import Base
from src.validation.reactflow_integration_validator import ReactFlowIntegrationValidator
from src.server.config.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReactFlowIntegrationReporter:
    """
    Generates comprehensive reports for ReactFlow integration validation.
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.timestamp = results.get("timestamp", datetime.utcnow().isoformat())

    def generate_text_report(self) -> str:
        """Generate a detailed text report."""
        report = []
        report.append("=" * 80)
        report.append("REACTFLOW INTEGRATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.timestamp}")
        report.append("")

        # Overall status
        overall_status = "PASSED" if self.results["overall_passed"] else "FAILED"
        report.append(f"Overall Status: {overall_status}")
        report.append("")

        # Validation summary
        report.append("VALIDATION SUMMARY")
        report.append("-" * 40)
        for category_name, category_result in self.results["validation_summary"].items():
            status = "PASSED" if category_result["passed"] else "FAILED"
            report.append(f"  {category_name.replace('_', ' ').title()}: {status}")

            if category_result["errors"]:
                report.append("    Errors:")
                for error in category_result["errors"]:
                    report.append(f"      - {error}")

            if category_result["warnings"]:
                report.append("    Warnings:")
                for warning in category_result["warnings"]:
                    report.append(f"      - {warning}")

        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for i, recommendation in enumerate(self.results["recommendations"], 1):
            report.append(f"{i}. {recommendation}")

        report.append("")

        # Detailed test results
        report.append("DETAILED TEST RESULTS")
        report.append("-" * 40)
        for category_name, category_result in self.results["validation_summary"].items():
            report.append(f"\n{category_name.replace('_', ' ').title()}:")
            report.append("-" * 20)

            if "test_cases" in category_result:
                for test_case in category_result["test_cases"]:
                    report.append(f"  Test: {test_case['name']}")
                    report.append(f"  Description: {test_case['description']}")
                    report.append(f"  Status: {test_case['status'].upper()}")
                    if "details" in test_case:
                        report.append(f"  Details: {test_case['details']}")
                    report.append("")

            if "api_tests" in category_result:
                for api_test in category_result["api_tests"]:
                    report.append(f"  API Test: {api_test['name']}")
                    report.append(f"  Endpoint: {api_test['endpoint']}")
                    report.append(f"  Description: {api_test['description']}")
                    report.append(f"  Status: {api_test['status'].upper()}")
                    if "details" in api_test:
                        report.append(f"  Details: {api_test['details']}")
                    report.append("")

            if "realtime_tests" in category_result:
                for realtime_test in category_result["realtime_tests"]:
                    report.append(f"  Real-time Test: {realtime_test['name']}")
                    report.append(f"  Event Type: {realtime_test['event_type']}")
                    report.append(f"  Description: {realtime_test['description']}")
                    report.append(f"  Status: {realtime_test['status'].upper()}")
                    if "details" in realtime_test:
                        report.append(f"  Details: {realtime_test['details']}")
                    report.append("")

            if "frontend_tests" in category_result:
                for frontend_test in category_result["frontend_tests"]:
                    report.append(f"  Frontend Test: {frontend_test['name']}")
                    report.append(f"  Description: {frontend_test['description']}")
                    report.append(f"  Status: {frontend_test['status'].upper()}")
                    if "details" in frontend_test:
                        report.append(f"  Details: {frontend_test['details']}")
                    report.append("")

        return "\n".join(report)

    def generate_json_report(self) -> str:
        """Generate a JSON report."""
        return json.dumps(self.results, indent=2, default=str)

    def generate_html_report(self) -> str:
        """Generate an HTML report."""
        html = []
        html.append(f"""
<!DOCTYPE html>
<html>
<head>
    <title>ReactFlow Integration Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .section {{ margin: 20px 0; }}
        .test-case {{ margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }}
        .error {{ color: red; }}
        .warning {{ color: orange; }}
        .recommendation {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ReactFlow Integration Validation Report</h1>
        <p>Generated: {self.timestamp}</p>
    </div>
""")

        # Overall status
        overall_status = "PASSED" if self.results["overall_passed"] else "FAILED"
        status_class = "status-passed" if self.results["overall_passed"] else "status-failed"
        html.append(f"""
    <div class="section">
        <h2>Overall Status: <span class="{status_class}">{overall_status}</span></h2>
    </div>
""")

        # Validation summary
        html.append("""
    <div class="section">
        <h2>Validation Summary</h2>
""")
        for category_name, category_result in self.results["validation_summary"].items():
            status = "PASSED" if category_result["passed"] else "FAILED"
            status_class = "status-passed" if category_result["passed"] else "status-failed"
            html.append(f"        <p><strong>{category_name.replace('_', ' ').title()}:</strong> <span class=\"{status_class}\">{status}</span></p>")

            if category_result["errors"]:
                html.append("        <ul class=\"error\">")
                for error in category_result["errors"]:
                    html.append(f"            <li>{error}</li>")
                html.append("        </ul>")

            if category_result["warnings"]:
                html.append("        <ul class=\"warning\">")
                for warning in category_result["warnings"]:
                    html.append(f"            <li>{warning}</li>")
                html.append("        </ul>")

        html.append("    </div>")

        # Recommendations
        html.append("""
    <div class="section">
        <h2>Recommendations</h2>
""")
        for recommendation in self.results["recommendations"]:
            html.append(f"        <div class=\"recommendation\">{recommendation}</div>")
        html.append("    </div>")

        html.append("""
</body>
</html>
""")

        return "\n".join(html)

    def save_report(self, filename: str, format: str = "text") -> None:
        """Save the report to a file."""
        if format == "text":
            content = self.generate_text_report()
        elif format == "json":
            content = self.generate_json_report()
        elif format == "html":
            content = self.generate_html_report()
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Report saved to {filename}")


async def create_database_session(database_url: str) -> AsyncSession:
    """Create a database session for validation."""
    logger.info(f"Creating database session for: {database_url}")

    # Create async engine
    engine = create_async_engine(database_url, echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    session = async_session()
    return session


async def run_validation(database_url: str, output_format: str = "text", output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run the ReactFlow integration validation."""
    logger.info("Starting ReactFlow integration validation...")

    try:
        # Create database session
        session = await create_database_session(database_url)

        # Create validator
        validator = ReactFlowIntegrationValidator(session)

        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()

        # Close session
        await session.close()

        logger.info(f"Validation completed. Overall status: {'PASSED' if results['overall_passed'] else 'FAILED'}")

        # Generate and save report
        reporter = ReactFlowIntegrationReporter(results)

        if output_file:
            reporter.save_report(output_file, output_format)
        else:
            # Print report to console
            if output_format == "json":
                print(reporter.generate_json_report())
            elif output_format == "html":
                print(reporter.generate_html_report())
            else:
                print(reporter.generate_text_report())

        return results

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate ReactFlow integration")
    parser.add_argument(
        "--database-url",
        default="sqlite+aiosqlite:///:memory:",
        help="Database URL for validation (default: in-memory SQLite)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: print to console)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    try:
        results = asyncio.run(run_validation(
            database_url=args.database_url,
            output_format=args.format,
            output_file=args.output
        ))

        # Exit with appropriate code
        sys.exit(0 if results["overall_passed"] else 1)

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()