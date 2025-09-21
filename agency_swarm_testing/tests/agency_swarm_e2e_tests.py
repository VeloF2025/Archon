#!/usr/bin/env python3
"""
Agency Swarm End-to-End Test Suite
Comprehensive testing across all Phase 1-3 components
"""

import asyncio
import pytest
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import aiohttp
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agency_swarm_e2e.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AgencySwarmE2ETestSuite:
    """Comprehensive E2E test suite for Agency Swarm system"""

    def __init__(self):
        self.base_url = "http://localhost:3737"
        self.api_base_url = "http://localhost:8181"
        self.mcp_base_url = "http://localhost:8051"
        self.agents_base_url = "http://localhost:8052"
        self.test_results = []
        self.session = None

    async def setup_test_environment(self):
        """Setup test environment and verify services are running"""
        logger.info("Setting up test environment...")

        # Check if all required services are running
        services = [
            ("Frontend", self.base_url),
            ("API Server", self.api_base_url),
            ("MCP Server", self.mcp_base_url),
            ("Agents Service", self.agents_base_url)
        ]

        for service_name, url in services:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✓ {service_name} is running")
                else:
                    logger.error(f"✗ {service_name} is not responding correctly")
                    return False
            except Exception as e:
                logger.error(f"✗ {service_name} is not accessible: {e}")
                return False

        return True

    async def test_phase1_knowledge_integration(self):
        """Test Phase 1: Knowledge Management System"""
        logger.info("Testing Phase 1: Knowledge Management System")

        test_cases = [
            {
                "name": "Knowledge Base Upload",
                "method": "POST",
                "url": f"{self.api_base_url}/api/knowledge/upload",
                "data": {"url": "https://example.com/test"},
                "expected_status": 200
            },
            {
                "name": "Knowledge Search",
                "method": "POST",
                "url": f"{self.api_base_url}/api/knowledge/search",
                "data": {"query": "test query"},
                "expected_status": 200
            },
            {
                "name": "Knowledge Items List",
                "method": "GET",
                "url": f"{self.api_base_url}/api/knowledge/items",
                "expected_status": 200
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.run_api_test(test_case)
            results.append(result)

        return self.analyze_results("Phase 1 Knowledge Integration", results)

    async def test_phase2_template_management(self):
        """Test Phase 2: Template Management System"""
        logger.info("Testing Phase 2: Template Management System")

        test_cases = [
            {
                "name": "Template Creation",
                "method": "POST",
                "url": f"{self.api_base_url}/api/templates",
                "data": {
                    "name": "Test Template",
                    "description": "E2E Test Template",
                    "type": "agent_template"
                },
                "expected_status": 201
            },
            {
                "name": "Template Listing",
                "method": "GET",
                "url": f"{self.api_base_url}/api/templates",
                "expected_status": 200
            },
            {
                "name": "Template Validation",
                "method": "POST",
                "url": f"{self.api_base_url}/api/templates/validate",
                "data": {"template_id": "test_template"},
                "expected_status": 200
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.run_api_test(test_case)
            results.append(result)

        return self.analyze_results("Phase 2 Template Management", results)

    async def test_phase3_pattern_library(self):
        """Test Phase 3: Pattern Library System"""
        logger.info("Testing Phase 3: Pattern Library System")

        test_cases = [
            {
                "name": "Pattern Registration",
                "method": "POST",
                "url": f"{self.api_base_url}/api/patterns",
                "data": {
                    "name": "Test Pattern",
                    "description": "E2E Test Pattern",
                    "category": "agent_communication"
                },
                "expected_status": 201
            },
            {
                "name": "Pattern Discovery",
                "method": "GET",
                "url": f"{self.api_base_url}/api/patterns/discover",
                "expected_status": 200
            },
            {
                "name": "Pattern Validation",
                "method": "POST",
                "url": f"{self.api_base_url}/api/patterns/validate",
                "data": {"pattern_id": "test_pattern"},
                "expected_status": 200
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.run_api_test(test_case)
            results.append(result)

        return self.analyze_results("Phase 3 Pattern Library", results)

    async def test_mcp_integration(self):
        """Test MCP integration and tool execution"""
        logger.info("Testing MCP Integration")

        test_cases = [
            {
                "name": "MCP Health Check",
                "method": "GET",
                "url": f"{self.mcp_base_url}/health",
                "expected_status": 200
            },
            {
                "name": "MCP Tools List",
                "method": "GET",
                "url": f"{self.mcp_base_url}/tools",
                "expected_status": 200
            },
            {
                "name": "MCP Tool Execution",
                "method": "POST",
                "url": f"{self.mcp_base_url}/tools/rag_query",
                "data": {"query": "test query"},
                "expected_status": 200
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.run_api_test(test_case)
            results.append(result)

        return self.analyze_results("MCP Integration", results)

    async def test_agents_service(self):
        """Test Agents service functionality"""
        logger.info("Testing Agents Service")

        test_cases = [
            {
                "name": "Agent Creation",
                "method": "POST",
                "url": f"{self.agents_base_url}/agents",
                "data": {
                    "name": "Test Agent",
                    "type": "knowledge_agent",
                    "config": {"model": "gpt-4"}
                },
                "expected_status": 201
            },
            {
                "name": "Agent Execution",
                "method": "POST",
                "url": f"{self.agents_base_url}/agents/execute",
                "data": {"agent_id": "test_agent", "task": "test task"},
                "expected_status": 200
            },
            {
                "name": "Agent Status",
                "method": "GET",
                "url": f"{self.agents_base_url}/agents/status",
                "expected_status": 200
            }
        ]

        results = []
        for test_case in test_cases:
            result = await self.run_api_test(test_case)
            results.append(result)

        return self.analyze_results("Agents Service", results)

    async def test_websocket_realtime_updates(self):
        """Test WebSocket real-time updates"""
        logger.info("Testing WebSocket Real-time Updates")

        try:
            import websockets

            async with websockets.connect(f"ws://{self.api_base_url.split('://')[1]}/ws") as websocket:
                # Subscribe to updates
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "channel": "agency_swarm_updates"
                }))

                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(response)

                return {
                    "test_name": "WebSocket Real-time Updates",
                    "status": "passed" if data.get("status") == "subscribed" else "failed",
                    "details": data
                }
        except Exception as e:
            return {
                "test_name": "WebSocket Real-time Updates",
                "status": "failed",
                "error": str(e)
            }

    async def test_full_agency_swarm_workflow(self):
        """Test complete Agency Swarm workflow"""
        logger.info("Testing Complete Agency Swarm Workflow")

        workflow_steps = [
            {
                "step": "Initialize Knowledge Base",
                "api_call": {
                    "method": "POST",
                    "url": f"{self.api_base_url}/api/knowledge/upload",
                    "data": {"url": "https://example.com/workflow-test"}
                }
            },
            {
                "step": "Create Agent Template",
                "api_call": {
                    "method": "POST",
                    "url": f"{self.api_base_url}/api/templates",
                    "data": {
                        "name": "Workflow Test Template",
                        "type": "agent_template"
                    }
                }
            },
            {
                "step": "Register Pattern",
                "api_call": {
                    "method": "POST",
                    "url": f"{self.api_base_url}/api/patterns",
                    "data": {
                        "name": "Workflow Test Pattern",
                        "category": "collaboration"
                    }
                }
            },
            {
                "step": "Execute Agent",
                "api_call": {
                    "method": "POST",
                    "url": f"{self.agents_base_url}/agents/execute",
                    "data": {
                        "agent_id": "workflow_agent",
                        "task": "Execute complete workflow"
                    }
                }
            }
        ]

        workflow_results = []
        for step in workflow_steps:
            result = await self.run_api_test(step["api_call"])
            workflow_results.append({
                "step": step["step"],
                "result": result
            })

        return {
            "test_name": "Complete Agency Swarm Workflow",
            "status": "passed" if all(r["result"]["status"] == "passed" for r in workflow_results) else "failed",
            "workflow_results": workflow_results
        }

    async def run_api_test(self, test_case):
        """Run a single API test case"""
        try:
            async with aiohttp.ClientSession() as session:
                if test_case["method"] == "GET":
                    async with session.get(test_case["url"]) as response:
                        data = await response.json()
                elif test_case["method"] == "POST":
                    async with session.post(test_case["url"], json=test_case["data"]) as response:
                        data = await response.json()

                success = response.status == test_case["expected_status"]

                return {
                    "test_name": test_case["name"],
                    "status": "passed" if success else "failed",
                    "expected_status": test_case["expected_status"],
                    "actual_status": response.status,
                    "response_data": data,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "test_name": test_case["name"],
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def analyze_results(self, test_category, results):
        """Analyze test results and generate summary"""
        passed = sum(1 for r in results if r["status"] == "passed")
        total = len(results)

        return {
            "test_category": test_category,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": (passed / total) * 100 if total > 0 else 0,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            "test_suite": "Agency Swarm E2E Test Suite",
            "execution_timestamp": datetime.now().isoformat(),
            "total_test_categories": len(self.test_results),
            "overall_summary": self.calculate_overall_summary(),
            "detailed_results": self.test_results,
            "recommendations": self.generate_recommendations()
        }

        # Save report to file
        report_path = Path("agency_swarm_e2e_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Test report saved to {report_path}")
        return report

    def calculate_overall_summary(self):
        """Calculate overall test summary"""
        total_tests = sum(r["total_tests"] for r in self.test_results)
        passed_tests = sum(r["passed_tests"] for r in self.test_results)

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "overall_success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        }

    def generate_recommendations(self):
        """Generate test recommendations based on results"""
        recommendations = []

        for result in self.test_results:
            if result["success_rate"] < 100:
                recommendations.append({
                    "category": result["test_category"],
                    "issue": f"Success rate {result['success_rate']:.1f}% below 100%",
                    "recommendation": "Investigate failed tests and fix underlying issues"
                })

        if not recommendations:
            recommendations.append({
                "category": "General",
                "issue": "All tests passing",
                "recommendation": "System is ready for production deployment"
            })

        return recommendations

    async def run_complete_test_suite(self):
        """Run the complete E2E test suite"""
        logger.info("Starting complete Agency Swarm E2E test suite...")

        # Setup test environment
        if not await self.setup_test_environment():
            logger.error("Test environment setup failed")
            return False

        # Run all test categories
        test_functions = [
            self.test_phase1_knowledge_integration,
            self.test_phase2_template_management,
            self.test_phase3_pattern_library,
            self.test_mcp_integration,
            self.test_agents_service,
            self.test_websocket_realtime_updates,
            self.test_full_agency_swarm_workflow
        ]

        for test_func in test_functions:
            try:
                result = await test_func()
                self.test_results.append(result)
                logger.info(f"✓ {test_func.__name__} completed")
            except Exception as e:
                logger.error(f"✗ {test_func.__name__} failed: {e}")
                self.test_results.append({
                    "test_category": test_func.__name__,
                    "status": "failed",
                    "error": str(e)
                })

        # Generate final report
        report = await self.generate_test_report()

        # Log summary
        summary = report["overall_summary"]
        logger.info(f"\n=== TEST SUMMARY ===")
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['overall_success_rate']:.1f}%")

        return report

async def main():
    """Main function to run the E2E test suite"""
    test_suite = AgencySwarmE2ETestSuite()
    await test_suite.run_complete_test_suite()

if __name__ == "__main__":
    asyncio.run(main())