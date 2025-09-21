#!/usr/bin/env python3
"""
Test Automation Framework for Agency Swarm
Automated test execution, scheduling, and reporting
"""

import asyncio
import aiohttp
import json
import logging
import argparse
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import yaml

# Import test modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "tests"))
import agency_swarm_e2e_tests
import integration_validation
import security_compliance_tests
import deployment_tests

logger = logging.getLogger(__name__)

class TestAutomationFramework:
    """Automated test execution and reporting system"""

    def __init__(self):
        self.config = self.load_config()
        self.test_results = {}
        self.test_schedules = self.config.get("schedules", {})
        self.notifications = self.config.get("notifications", {})
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    def load_config(self):
        """Load test automation configuration"""
        config_path = Path("config/test_automation_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                "schedules": {
                    "unit_tests": "0 6 * * *",  # Daily at 6 AM
                    "integration_tests": "0 2 * * 0",  # Weekly on Sunday at 2 AM
                    "performance_tests": "0 1 * * 0",  # Weekly on Sunday at 1 AM
                    "security_tests": "0 3 * * 0",  # Weekly on Sunday at 3 AM
                    "e2e_tests": "0 4 * * 0",  # Weekly on Sunday at 4 AM
                    "deployment_tests": "0 5 * * 0"  # Weekly on Sunday at 5 AM
                },
                "notifications": {
                    "email": {
                        "enabled": False,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "sender_email": "",
                        "sender_password": "",
                        "recipients": []
                    },
                    "slack": {
                        "enabled": False,
                        "webhook_url": "",
                        "channel": "#testing"
                    }
                },
                "quality_gates": {
                    "min_pass_rate": 85,
                    "critical_tests": ["security", "e2e", "deployment"],
                    "max_duration_minutes": 60
                },
                "parallel_execution": {
                    "enabled": True,
                    "max_parallel_jobs": 4
                }
            }

    async def run_test_suite(self, suite_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test suite"""
        logger.info(f"Running test suite: {suite_name}")

        start_time = datetime.now()
        result = {
            "suite_name": suite_name,
            "start_time": start_time.isoformat(),
            "status": "running",
            "results": {}
        }

        try:
            if suite_name == "unit_tests":
                result["results"] = await self.run_unit_tests(**kwargs)
            elif suite_name == "integration_tests":
                result["results"] = await self.run_integration_tests(**kwargs)
            elif suite_name == "performance_tests":
                result["results"] = await self.run_performance_tests(**kwargs)
            elif suite_name == "security_tests":
                result["results"] = await self.run_security_tests(**kwargs)
            elif suite_name == "e2e_tests":
                result["results"] = await self.run_e2e_tests(**kwargs)
            elif suite_name == "deployment_tests":
                result["results"] = await self.run_deployment_tests(**kwargs)
            elif suite_name == "all_tests":
                result["results"] = await self.run_all_tests(**kwargs)
            else:
                raise ValueError(f"Unknown test suite: {suite_name}")

            result["status"] = "completed"
            result["end_time"] = datetime.now().isoformat()
            result["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
            result["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60

        # Store result
        self.test_results[suite_name] = result

        # Generate report
        await self.generate_suite_report(suite_name, result)

        # Check quality gates
        await self.check_quality_gates(suite_name, result)

        return result

    async def run_unit_tests(self, **kwargs) -> Dict[str, Any]:
        """Run unit tests"""
        logger.info("Running unit tests...")

        try:
            # Run Python unit tests
            python_result = subprocess.run(
                ["pytest", "python/tests/", "-v", "--cov=src", "--cov-report=xml"],
                capture_output=True, text=True, cwd=Path("..")
            )

            # Run Node.js unit tests
            nodejs_result = subprocess.run(
                ["npm", "test", "--", "--coverage", "--watchAll=false"],
                capture_output=True, text=True, cwd=Path("../archon-ui-main")
            )

            return {
                "python_tests": {
                    "exit_code": python_result.returncode,
                    "stdout": python_result.stdout,
                    "stderr": python_result.stderr,
                    "passed": python_result.returncode == 0
                },
                "nodejs_tests": {
                    "exit_code": nodejs_result.returncode,
                    "stdout": nodejs_result.stdout,
                    "stderr": nodejs_result.stderr,
                    "passed": nodejs_result.returncode == 0
                },
                "overall_passed": python_result.returncode == 0 and nodejs_result.returncode == 0
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_integration_tests(self, **kwargs) -> Dict[str, Any]:
        """Run integration tests"""
        logger.info("Running integration tests...")

        try:
            validator = integration_validation.IntegrationValidator()
            results = await validator.run_all_integration_tests()
            report = validator.generate_integration_report()

            return {
                "results": results,
                "report": report,
                "passed": report["integration_status"] == "healthy"
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_performance_tests(self, **kwargs) -> Dict[str, Any]:
        """Run performance tests"""
        logger.info("Running performance tests...")

        try:
            benchmark = performance_benchmarks.PerformanceBenchmark()
            report = await benchmark.run_complete_benchmark()

            return {
                "report": report,
                "passed": report["overall_assessment"]["ready_for_production"]
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_security_tests(self, **kwargs) -> Dict[str, Any]:
        """Run security tests"""
        logger.info("Running security tests...")

        try:
            tester = security_compliance_tests.SecurityComplianceTester()
            report = await tester.run_complete_security_test()

            return {
                "report": report,
                "passed": report["overall_assessment"]["ready_for_production"]
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_e2e_tests(self, **kwargs) -> Dict[str, Any]:
        """Run end-to-end tests"""
        logger.info("Running end-to-end tests...")

        try:
            test_suite = agency_swarm_e2e_tests.AgencySwarmE2ETestSuite()
            report = await test_suite.run_complete_test_suite()

            return {
                "report": report,
                "passed": report["overall_summary"]["overall_success_rate"] >= 90
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_deployment_tests(self, **kwargs) -> Dict[str, Any]:
        """Run deployment tests"""
        logger.info("Running deployment tests...")

        try:
            tester = deployment_tests.DeploymentTester()
            report = await tester.run_complete_deployment_test()

            return {
                "report": report,
                "passed": report["overall_assessment"]["deployment_ready"]
            }
        except Exception as e:
            return {"error": str(e), "passed": False}

    async def run_all_tests(self, **kwargs) -> Dict[str, Any]:
        """Run all test suites"""
        logger.info("Running all test suites...")

        if self.config.get("parallel_execution", {}).get("enabled", True):
            # Run tests in parallel
            tasks = [
                self.run_unit_tests(**kwargs),
                self.run_integration_tests(**kwargs),
                self.run_performance_tests(**kwargs),
                self.run_security_tests(**kwargs),
                self.run_e2e_tests(**kwargs),
                self.run_deployment_tests(**kwargs)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run tests sequentially
            results = []
            for test_func in [
                self.run_unit_tests,
                self.run_integration_tests,
                self.run_performance_tests,
                self.run_security_tests,
                self.run_e2e_tests,
                self.run_deployment_tests
            ]:
                try:
                    result = await test_func(**kwargs)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "passed": False})

        return {
            "unit_tests": results[0],
            "integration_tests": results[1],
            "performance_tests": results[2],
            "security_tests": results[3],
            "e2e_tests": results[4],
            "deployment_tests": results[5],
            "overall_passed": all(
                result.get("passed", False) if isinstance(result, dict) else False
                for result in results
            )
        }

    async def generate_suite_report(self, suite_name: str, result: Dict[str, Any]):
        """Generate report for a specific test suite"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"{suite_name}_{timestamp}.json"

        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        # Generate HTML report
        html_report_path = self.reports_dir / f"{suite_name}_{timestamp}.html"
        await self.generate_html_report(suite_name, result, html_report_path)

        return report_path

    async def generate_html_report(self, suite_name: str, result: Dict[str, Any], output_path: Path):
        """Generate HTML report for test results"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{suite_name} Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .warning {{ color: orange; }}
                .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; flex: 1; }}
                .details {{ margin-top: 20px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{suite_name} Test Report</h1>
                <p>Generated: {result['start_time']}</p>
                <p>Status: <span class="{result['status']}">{result['status']}</span></p>
                <p>Duration: {result.get('duration_minutes', 0):.2f} minutes</p>
            </div>
        """

        if "results" in result and "overall_passed" in result["results"]:
            html_content += f"""
            <div class="metrics">
                <div class="metric">
                    <h3>Overall Status</h3>
                    <p class="{'passed' if result['results']['overall_passed'] else 'failed'}">
                        {'PASSED' if result['results']['overall_passed'] else 'FAILED'}
                    </p>
                </div>
            </div>
            """

        # Add detailed results
        if "results" in result:
            html_content += '<div class="details"><h2>Detailed Results</h2><pre>'
            html_content += json.dumps(result["results"], indent=2)
            html_content += '</pre></div>'

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")

    async def check_quality_gates(self, suite_name: str, result: Dict[str, Any]):
        """Check if test results meet quality gates"""
        quality_gates = self.config.get("quality_gates", {})
        min_pass_rate = quality_gates.get("min_pass_rate", 85)
        critical_tests = quality_gates.get("critical_tests", [])
        max_duration = quality_gates.get("max_duration_minutes", 60)

        issues = []

        # Check pass rate
        if "results" in result and "overall_passed" in result["results"]:
            if not result["results"]["overall_passed"]:
                issues.append(f"Test suite failed overall")

        # Check duration
        duration = result.get("duration_minutes", 0)
        if duration > max_duration:
            issues.append(f"Test duration ({duration:.2f} min) exceeds maximum ({max_duration} min)")

        # Check critical tests
        if suite_name in critical_tests:
            if result.get("status") != "completed":
                issues.append(f"Critical test suite {suite_name} did not complete successfully")

        # Send alerts if issues found
        if issues:
            await self.send_quality_alert(suite_name, result, issues)

    async def send_quality_alert(self, suite_name: str, result: Dict[str, Any], issues: List[str]):
        """Send quality alert notifications"""
        logger.warning(f"Quality gate violations detected for {suite_name}: {issues}")

        alert_message = f"""
🚨 Quality Gate Violation Alert

Test Suite: {suite_name}
Timestamp: {result['start_time']}
Status: {result['status']}

Issues Found:
{chr(10).join(f'• {issue}' for issue in issues)}

Please review the test results and take appropriate action.
        """

        # Send email notification
        if self.notifications.get("email", {}).get("enabled", False):
            await self.send_email_alert(f"Quality Gate Violation - {suite_name}", alert_message)

        # Send Slack notification
        if self.notifications.get("slack", {}).get("enabled", False):
            await self.send_slack_alert(alert_message)

    async def send_email_alert(self, subject: str, message: str):
        """Send email alert"""
        try:
            email_config = self.notifications["email"]
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = subject

            msg.attach(MimeText(message, 'plain'))

            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()

            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def send_slack_alert(self, message: str):
        """Send Slack alert"""
        try:
            import requests

            slack_config = self.notifications["slack"]
            payload = {
                "text": message,
                "channel": slack_config["channel"]
            }

            response = requests.post(slack_config["webhook_url"], json=payload)
            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(f"Failed to send Slack alert: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def run_scheduled_tests(self):
        """Run scheduled tests based on configuration"""
        logger.info("Starting scheduled test execution...")

        # Schedule tests
        for suite_name, cron_schedule in self.test_schedules.items():
            schedule.every().day.at(cron_schedule.split()[1]).do(
                lambda sn=suite_name: asyncio.create_task(self.run_test_suite(sn))
            )

        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)

    async def generate_daily_summary(self):
        """Generate daily test summary report"""
        logger.info("Generating daily test summary...")

        today = datetime.now().date()
        today_results = {}

        # Collect today's results
        for suite_name, result in self.test_results.items():
            result_date = datetime.fromisoformat(result['start_time']).date()
            if result_date == today:
                today_results[suite_name] = result

        # Generate summary
        summary = {
            "date": today.isoformat(),
            "total_suites": len(today_results),
            "passed_suites": sum(1 for r in today_results.values() if r.get("status") == "completed"),
            "failed_suites": sum(1 for r in today_results.values() if r.get("status") == "failed"),
            "results": today_results
        }

        # Save summary
        summary_path = self.reports_dir / f"daily_summary_{today.isoformat()}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Daily summary saved to {summary_path}")

        # Send daily summary notification
        if self.notifications.get("email", {}).get("enabled", False):
            await self.send_daily_summary_email(summary)

    async def send_daily_summary_email(self, summary: Dict[str, Any]):
        """Send daily summary email"""
        subject = f"Daily Test Summary - {summary['date']}"
        message = f"""
📊 Daily Test Summary Report

Date: {summary['date']}
Total Test Suites: {summary['total_suites']}
Passed: {summary['passed_suites']}
Failed: {summary['failed_suites']}

Detailed Results:
"""

        for suite_name, result in summary['results'].items():
            message += f"\n• {suite_name}: {result['status'].upper()}"
            if result.get("duration_minutes"):
                message += f" ({result['duration_minutes']:.2f} min)"

        await self.send_email_alert(subject, message)

    async def cleanup_old_reports(self, days_to_keep: int = 30):
        """Clean up old test reports"""
        logger.info(f"Cleaning up reports older than {days_to_keep} days...")

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for report_file in self.reports_dir.glob("*"):
            if report_file.is_file():
                try:
                    file_date = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        report_file.unlink()
                        logger.info(f"Deleted old report: {report_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {report_file}: {e}")

    async def run_health_check(self):
        """Run system health check"""
        logger.info("Running system health check...")

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "test_framework": "healthy"
        }

        # Check service health
        services = {
            "frontend": "http://localhost:3737",
            "api": "http://localhost:8181",
            "mcp": "http://localhost:8051",
            "agents": "http://localhost:8052"
        }

        for service_name, service_url in services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_url}/health") as response:
                        if response.status == 200:
                            health_status["services"][service_name] = "healthy"
                        else:
                            health_status["services"][service_name] = "unhealthy"
                            health_status["test_framework"] = "degraded"
            except Exception as e:
                health_status["services"][service_name] = "unhealthy"
                health_status["test_framework"] = "degraded"

        # Save health status
        health_path = self.reports_dir / f"health_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(health_path, 'w') as f:
            json.dump(health_status, f, indent=2)

        return health_status

    async def main_loop(self):
        """Main loop for continuous test automation"""
        logger.info("Starting test automation framework...")

        # Start health check scheduler
        schedule.every(5).minutes.do(lambda: asyncio.create_task(self.run_health_check()))

        # Start daily summary scheduler
        schedule.every().day.at("23:59").do(lambda: asyncio.create_task(self.generate_daily_summary()))

        # Start cleanup scheduler
        schedule.every().week.do(lambda: asyncio.create_task(self.cleanup_old_reports()))

        # Run initial health check
        await self.run_health_check()

        # Start scheduled test execution
        await self.run_scheduled_tests()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Agency Swarm Test Automation")
    parser.add_argument("--suite", choices=[
        "unit_tests", "integration_tests", "performance_tests",
        "security_tests", "e2e_tests", "deployment_tests", "all_tests"
    ], help="Test suite to run")
    parser.add_argument("--schedule", action="store_true", help="Run scheduled tests")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--cleanup", type=int, help="Clean up reports older than N days")

    args = parser.parse_args()

    framework = TestAutomationFramework()

    async def run_command():
        if args.suite:
            result = await framework.run_test_suite(args.suite)
            print(f"Test suite {args.suite} completed with status: {result['status']}")
        elif args.schedule:
            await framework.run_scheduled_tests()
        elif args.health_check:
            health_status = await framework.run_health_check()
            print(f"Health check completed: {health_status['test_framework']}")
        elif args.cleanup:
            await framework.cleanup_old_reports(args.cleanup)
            print(f"Cleaned up reports older than {args.cleanup} days")
        else:
            # Default: run all tests
            result = await framework.run_test_suite("all_tests")
            print(f"All tests completed with status: {result['status']}")

    asyncio.run(run_command())

if __name__ == "__main__":
    main()