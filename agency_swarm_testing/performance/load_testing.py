#!/usr/bin/env python3
"""
Load Testing Framework for Agency Swarm System
Enterprise-grade load testing with concurrent users and scenarios
"""

import asyncio
import aiohttp
import json
import time
import statistics
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)

class LoadTester:
    """Enterprise load testing system"""

    def __init__(self):
        self.base_url = "http://localhost:3737"
        self.api_url = "http://localhost:8181"
        self.mcp_url = "http://localhost:8051"
        self.agents_url = "http://localhost:8052"
        self.results = {}
        self.session_pool = None
        self.load_config = self.load_load_config()

    def load_load_config(self):
        """Load load testing configuration"""
        return {
            "scenarios": {
                "knowledge_workload": {
                    "duration_minutes": 10,
                    "concurrent_users": [10, 50, 100, 200, 500],
                    "ramp_up_minutes": 2,
                    "think_time_seconds": 1,
                    "endpoints": [
                        ("GET", "/api/knowledge/items"),
                        ("POST", "/api/knowledge/search"),
                        ("POST", "/api/knowledge/upload")
                    ]
                },
                "agent_workload": {
                    "duration_minutes": 15,
                    "concurrent_users": [5, 25, 50, 100],
                    "ramp_up_minutes": 3,
                    "think_time_seconds": 2,
                    "endpoints": [
                        ("POST", "/api/agents/execute"),
                        ("GET", "/api/agents/status"),
                        ("POST", "/api/agents/create")
                    ]
                },
                "mcp_workload": {
                    "duration_minutes": 8,
                    "concurrent_users": [20, 100, 200],
                    "ramp_up_minutes": 1,
                    "think_time_seconds": 0.5,
                    "endpoints": [
                        ("GET", "/tools"),
                        ("POST", "/tools/rag_query"),
                        ("POST", "/tools/search_code_examples")
                    ]
                },
                "mixed_workload": {
                    "duration_minutes": 20,
                    "concurrent_users": [100, 500, 1000],
                    "ramp_up_minutes": 5,
                    "think_time_seconds": 1.5,
                    "endpoints": [
                        ("GET", "/api/knowledge/items"),
                        ("POST", "/api/knowledge/search"),
                        ("POST", "/api/agents/execute"),
                        ("GET", "/tools"),
                        ("POST", "/tools/rag_query")
                    ]
                }
            },
            "thresholds": {
                "max_response_time": 5000,  # ms
                "max_error_rate": 0.05,   # 5%
                "min_throughput": 100,    # requests/second
                "max_cpu_usage": 80,      # %
                "max_memory_usage": 90     # %
            },
            "reporting": {
                "generate_charts": True,
                "save_raw_data": True,
                "interval_seconds": 30
            }
        }

    async def run_load_scenario(self, scenario_name: str, concurrent_users: int) -> Dict[str, Any]:
        """Run a specific load scenario"""
        logger.info(f"Running load scenario: {scenario_name} with {concurrent_users} concurrent users")

        scenario = self.load_config["scenarios"][scenario_name]
        duration = scenario["duration_minutes"] * 60
        ramp_up = scenario["ramp_up_minutes"] * 60
        think_time = scenario["think_time_seconds"]

        results = {
            "scenario_name": scenario_name,
            "concurrent_users": concurrent_users,
            "start_time": datetime.now().isoformat(),
            "requests": [],
            "errors": [],
            "throughput_data": [],
            "response_times": [],
            "system_metrics": []
        }

        # Create user tasks
        user_tasks = []
        for user_id in range(concurrent_users):
            task = asyncio.create_task(
                self.simulate_user(user_id, scenario, duration, ramp_up, think_time, results)
            )
            user_tasks.append(task)

        # Start monitoring
        monitor_task = asyncio.create_task(
            self.monitor_system_metrics(duration, results)
        )

        # Start throughput monitoring
        throughput_task = asyncio.create_task(
            self.monitor_throughput(duration, results)
        )

        # Wait for all user tasks to complete
        await asyncio.gather(*user_tasks)

        # Stop monitoring tasks
        monitor_task.cancel()
        throughput_task.cancel()

        # Calculate final metrics
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = duration
        results["total_requests"] = len(results["requests"])
        results["total_errors"] = len(results["errors"])
        results["success_rate"] = (
            (results["total_requests"] - results["total_errors"]) / results["total_requests"]
            if results["total_requests"] > 0 else 0
        )
        results["avg_response_time"] = statistics.mean(results["response_times"]) if results["response_times"] else 0
        results["p95_response_time"] = np.percentile(results["response_times"], 95) if results["response_times"] else 0
        results["p99_response_time"] = np.percentile(results["response_times"], 99) if results["response_times"] else 0
        results["throughput_rps"] = results["total_requests"] / duration if duration > 0 else 0

        return results

    async def simulate_user(self, user_id: int, scenario: Dict, duration: int, ramp_up: int, think_time: float, results: Dict):
        """Simulate a user's behavior"""
        start_time = time.time()
        end_time = start_time + duration

        # Calculate ramp-up delay
        if ramp_up > 0:
            user_delay = (ramp_up / scenario["concurrent_users"][-1]) * user_id
            await asyncio.sleep(user_delay)

        endpoints = scenario["endpoints"]

        while time.time() < end_time:
            try:
                # Select random endpoint
                method, endpoint = endpoints[np.random.randint(0, len(endpoints))]

                # Prepare request data
                request_data = self.prepare_request_data(method, endpoint)

                # Execute request
                start_request = time.time()
                result = await self.execute_request(method, endpoint, request_data)
                response_time = (time.time() - start_request) * 1000  # Convert to ms

                # Record result
                results["requests"].append({
                    "user_id": user_id,
                    "method": method,
                    "endpoint": endpoint,
                    "response_time": response_time,
                    "status_code": result.get("status_code", 0),
                    "timestamp": time.time()
                })

                if result.get("status_code", 0) >= 400:
                    results["errors"].append({
                        "user_id": user_id,
                        "method": method,
                        "endpoint": endpoint,
                        "error": result.get("error", "Unknown error"),
                        "timestamp": time.time()
                    })
                else:
                    results["response_times"].append(response_time)

                # Think time
                if think_time > 0:
                    await asyncio.sleep(think_time * (0.5 + np.random.random()))

            except Exception as e:
                results["errors"].append({
                    "user_id": user_id,
                    "method": method,
                    "endpoint": endpoint,
                    "error": str(e),
                    "timestamp": time.time()
                })

                # Wait before retry
                await asyncio.sleep(1)

    def prepare_request_data(self, method: str, endpoint: str) -> Dict:
        """Prepare request data for different endpoints"""
        if endpoint == "/api/knowledge/search":
            return {"query": f"test_query_{np.random.randint(1, 1000)}"}
        elif endpoint == "/api/knowledge/upload":
            return {"url": f"https://example.com/test_{np.random.randint(1, 1000)}"}
        elif endpoint == "/api/agents/execute":
            return {
                "agent_type": "knowledge_agent",
                "task": f"test_task_{np.random.randint(1, 1000)}"
            }
        elif endpoint == "/tools/rag_query":
            return {"query": f"rag_query_{np.random.randint(1, 1000)}"}
        elif endpoint == "/tools/search_code_examples":
            return {"search_term": f"code_{np.random.randint(1, 1000)}"}
        else:
            return {}

    async def execute_request(self, method: str, endpoint: str, data: Dict) -> Dict:
        """Execute HTTP request"""
        try:
            # Determine base URL based on endpoint
            if endpoint.startswith("/api/knowledge") or endpoint.startswith("/api/agents"):
                base_url = self.api_url
            elif endpoint.startswith("/tools"):
                base_url = self.mcp_url
            else:
                base_url = self.base_url

            url = f"{base_url}{endpoint}"

            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url, params=data) as response:
                        return {"status_code": response.status, "data": await response.text()}
                else:
                    async with session.post(url, json=data) as response:
                        return {"status_code": response.status, "data": await response.text()}

        except Exception as e:
            return {"status_code": 0, "error": str(e)}

    async def monitor_system_metrics(self, duration: int, results: Dict):
        """Monitor system metrics during load test"""
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            try:
                # Collect CPU and memory usage
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                results["system_metrics"].append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent
                })

                await asyncio.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(5)

    async def monitor_throughput(self, duration: int, results: Dict):
        """Monitor throughput during load test"""
        start_time = time.time()
        end_time = start_time + duration
        interval = self.load_config["reporting"]["interval_seconds"]

        while time.time() < end_time:
            interval_start = time.time()
            interval_end = interval_start + interval

            # Count requests in this interval
            request_count = sum(
                1 for req in results["requests"]
                if interval_start <= req["timestamp"] < interval_end
            )

            throughput = request_count / interval  # requests per second

            results["throughput_data"].append({
                "timestamp": interval_start,
                "throughput_rps": throughput,
                "request_count": request_count
            })

            await asyncio.sleep(interval)

    async def run_stress_test(self, scenario_name: str) -> Dict[str, Any]:
        """Run stress test to find breaking point"""
        logger.info(f"Running stress test for scenario: {scenario_name}")

        scenario = self.load_config["scenarios"][scenario_name]
        user_counts = [10, 50, 100, 200, 500, 1000, 2000, 5000]
        stress_results = []

        for user_count in user_counts:
            logger.info(f"Testing with {user_count} concurrent users...")

            # Run test for shorter duration for stress testing
            scenario_copy = scenario.copy()
            scenario_copy["duration_minutes"] = 2
            scenario_copy["ramp_up_minutes"] = 0.5

            result = await self.run_load_scenario(scenario_name, user_count)
            stress_results.append(result)

            # Check if system is overloaded
            if (result["success_rate"] < 0.95 or
                result["avg_response_time"] > 5000 or
                any(metric["cpu_percent"] > 90 for metric in result["system_metrics"])):
                logger.info(f"System breaking point detected at {user_count} users")
                break

        return {
            "scenario_name": scenario_name,
            "stress_results": stress_results,
            "breaking_point": self.find_breaking_point(stress_results),
            "max_sustainable_users": self.find_max_sustainable_users(stress_results)
        }

    def find_breaking_point(self, results: List[Dict]) -> Optional[int]:
        """Find the breaking point where system performance degrades"""
        thresholds = self.load_config["thresholds"]

        for result in results:
            if (result["success_rate"] < (1 - thresholds["max_error_rate"]) or
                result["avg_response_time"] > thresholds["max_response_time"]):
                return result["concurrent_users"]

        return None

    def find_max_sustainable_users(self, results: List[Dict]) -> int:
        """Find maximum number of sustainable users"""
        thresholds = self.load_config["thresholds"]
        max_users = 0

        for result in results:
            if (result["success_rate"] >= (1 - thresholds["max_error_rate"]) and
                result["avg_response_time"] <= thresholds["max_response_time"] and
                result["throughput_rps"] >= thresholds["min_throughput"]):
                max_users = result["concurrent_users"]
            else:
                break

        return max_users

    async def run_endurance_test(self, scenario_name: str, duration_hours: int = 24) -> Dict[str, Any]:
        """Run endurance test for extended period"""
        logger.info(f"Running endurance test for scenario: {scenario_name} ({duration_hours} hours)")

        scenario = self.load_config["scenarios"][scenario_name]
        concurrent_users = scenario["concurrent_users"][2]  # Use medium user count

        # Modify scenario for endurance test
        scenario_copy = scenario.copy()
        scenario_copy["duration_minutes"] = duration_hours * 60
        scenario_copy["ramp_up_minutes"] = 10

        results = await self.run_load_scenario(scenario_name, concurrent_users)

        # Add endurance-specific analysis
        results["endurance_analysis"] = {
            "duration_hours": duration_hours,
            "memory_trend": self.analyze_memory_trend(results["system_metrics"]),
            "response_time_trend": self.analyze_response_time_trend(results["requests"]),
            "error_trend": self.analyze_error_trend(results["errors"])
        }

        return results

    def analyze_memory_trend(self, system_metrics: List[Dict]) -> Dict:
        """Analyze memory usage trend"""
        if not system_metrics:
            return {"trend": "no_data"}

        memory_values = [m["memory_percent"] for m in system_metrics]
        first_half = memory_values[:len(memory_values)//2]
        second_half = memory_values[len(memory_values)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.1:
            return {"trend": "increasing", "first_avg": first_avg, "second_avg": second_avg}
        elif second_avg < first_avg * 0.9:
            return {"trend": "decreasing", "first_avg": first_avg, "second_avg": second_avg}
        else:
            return {"trend": "stable", "first_avg": first_avg, "second_avg": second_avg}

    def analyze_response_time_trend(self, requests: List[Dict]) -> Dict:
        """Analyze response time trend"""
        if not requests:
            return {"trend": "no_data"}

        # Group requests by time intervals
        interval_duration = 300  # 5 minutes
        intervals = {}
        for req in requests:
            interval = int(req["timestamp"] // interval_duration)
            if interval not in intervals:
                intervals[interval] = []
            intervals[interval].append(req["response_time"])

        # Calculate average response time per interval
        interval_avgs = []
        for interval in sorted(intervals.keys()):
            if intervals[interval]:
                interval_avgs.append(statistics.mean(intervals[interval]))

        if len(interval_avgs) < 2:
            return {"trend": "insufficient_data"}

        first_half = interval_avgs[:len(interval_avgs)//2]
        second_half = interval_avgs[len(interval_avgs)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.2:
            return {"trend": "degrading", "first_avg": first_avg, "second_avg": second_avg}
        elif second_avg < first_avg * 0.8:
            return {"trend": "improving", "first_avg": first_avg, "second_avg": second_avg}
        else:
            return {"trend": "stable", "first_avg": first_avg, "second_avg": second_avg}

    def analyze_error_trend(self, errors: List[Dict]) -> Dict:
        """Analyze error trend"""
        if not errors:
            return {"trend": "no_errors"}

        # Group errors by time intervals
        interval_duration = 300  # 5 minutes
        intervals = {}
        for error in errors:
            interval = int(error["timestamp"] // interval_duration)
            if interval not in intervals:
                intervals[interval] = 0
            intervals[interval] += 1

        # Calculate error rate per interval
        interval_error_rates = list(intervals.values())

        if len(interval_error_rates) < 2:
            return {"trend": "insufficient_data"}

        first_half = interval_error_rates[:len(interval_error_rates)//2]
        second_half = interval_error_rates[len(interval_error_rates)//2:]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        if second_avg > first_avg * 1.5:
            return {"trend": "increasing", "first_avg": first_avg, "second_avg": second_avg}
        elif second_avg < first_avg * 0.5:
            return {"trend": "decreasing", "first_avg": first_avg, "second_avg": second_avg}
        else:
            return {"trend": "stable", "first_avg": first_avg, "second_avg": second_avg}

    async def generate_load_test_report(self, test_results: Dict) -> Dict:
        """Generate comprehensive load test report"""
        timestamp = datetime.now().isoformat()

        # Analyze all results
        analysis = self.analyze_load_test_results(test_results)

        report = {
            "test_suite": "Agency Swarm Load Testing",
            "timestamp": timestamp,
            "test_configuration": self.load_config,
            "test_results": test_results,
            "analysis": analysis,
            "recommendations": self.generate_load_test_recommendations(analysis)
        }

        # Save report
        report_path = Path("agency_swarm_load_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate visualizations
        if self.load_config["reporting"]["generate_charts"]:
            await self.generate_load_test_charts(test_results)

        logger.info(f"Load test report saved to {report_path}")
        return report

    def analyze_load_test_results(self, test_results: Dict) -> Dict:
        """Analyze load test results"""
        analysis = {
            "scenario_summaries": {},
            "overall_assessment": {},
            "performance_trends": {},
            "bottlenecks": []
        }

        thresholds = self.load_config["thresholds"]

        for scenario_name, results in test_results.items():
            if isinstance(results, list):
                # Stress test results
                analysis["scenario_summaries"][scenario_name] = {
                    "type": "stress_test",
                    "max_users_tested": max(r["concurrent_users"] for r in results),
                    "breaking_point": self.find_breaking_point(results),
                    "max_sustainable_users": self.find_max_sustainable_users(results)
                }
            else:
                # Regular load test
                analysis["scenario_summaries"][scenario_name] = {
                    "type": "load_test",
                    "concurrent_users": results["concurrent_users"],
                    "success_rate": results["success_rate"],
                    "avg_response_time": results["avg_response_time"],
                    "throughput_rps": results["throughput_rps"],
                    "meets_thresholds": self.meets_thresholds(results, thresholds)
                }

                # Check for bottlenecks
                if not self.meets_thresholds(results, thresholds):
                    analysis["bottlenecks"].append({
                        "scenario": scenario_name,
                        "issues": self.identify_bottlenecks(results, thresholds)
                    })

        # Overall assessment
        total_scenarios = len(analysis["scenario_summaries"])
        passing_scenarios = sum(
            1 for summary in analysis["scenario_summaries"].values()
            if summary.get("meets_thresholds", True)
        )

        analysis["overall_assessment"] = {
            "total_scenarios": total_scenarios,
            "passing_scenarios": passing_scenarios,
            "success_rate": (passing_scenarios / total_scenarios) * 100 if total_scenarios > 0 else 0,
            "ready_for_production": passing_scenarios == total_scenarios
        }

        return analysis

    def meets_thresholds(self, results: Dict, thresholds: Dict) -> bool:
        """Check if results meet performance thresholds"""
        return (
            results["success_rate"] >= (1 - thresholds["max_error_rate"]) and
            results["avg_response_time"] <= thresholds["max_response_time"] and
            results["throughput_rps"] >= thresholds["min_throughput"]
        )

    def identify_bottlenecks(self, results: Dict, thresholds: Dict) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        if results["success_rate"] < (1 - thresholds["max_error_rate"]):
            bottlenecks.append(f"High error rate: {results['success_rate']:.2%}")

        if results["avg_response_time"] > thresholds["max_response_time"]:
            bottlenecks.append(f"Slow response time: {results['avg_response_time']:.2f}ms")

        if results["throughput_rps"] < thresholds["min_throughput"]:
            bottlenecks.append(f"Low throughput: {results['throughput_rps']:.2f} rps")

        # Check system metrics
        if results.get("system_metrics"):
            max_cpu = max(m["cpu_percent"] for m in results["system_metrics"])
            max_memory = max(m["memory_percent"] for m in results["system_metrics"])

            if max_cpu > thresholds["max_cpu_usage"]:
                bottlenecks.append(f"High CPU usage: {max_cpu:.1f}%")

            if max_memory > thresholds["max_memory_usage"]:
                bottlenecks.append(f"High memory usage: {max_memory:.1f}%")

        return bottlenecks

    def generate_load_test_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate performance optimization recommendations"""
        recommendations = []

        for bottleneck in analysis["bottlenecks"]:
            for issue in bottleneck["issues"]:
                if "error rate" in issue:
                    recommendations.append({
                        "priority": "High",
                        "category": "Reliability",
                        "issue": issue,
                        "recommendation": "Implement better error handling, retry mechanisms, and circuit breakers"
                    })
                elif "response time" in issue:
                    recommendations.append({
                        "priority": "High",
                        "category": "Performance",
                        "issue": issue,
                        "recommendation": "Optimize database queries, implement caching, or scale horizontally"
                    })
                elif "throughput" in issue:
                    recommendations.append({
                        "priority": "Medium",
                        "category": "Scalability",
                        "issue": issue,
                        "recommendation": "Increase server resources, optimize algorithms, or implement load balancing"
                    })
                elif "CPU usage" in issue:
                    recommendations.append({
                        "priority": "Medium",
                        "category": "Resource",
                        "issue": issue,
                        "recommendation": "Optimize CPU-intensive operations, add more CPU cores, or use more efficient algorithms"
                    })
                elif "memory usage" in issue:
                    recommendations.append({
                        "priority": "Medium",
                        "category": "Resource",
                        "issue": issue,
                        "recommendation": "Optimize memory usage, implement memory pooling, or add more RAM"
                    })

        return recommendations

    async def generate_load_test_charts(self, test_results: Dict):
        """Generate load test visualization charts"""
        try:
            charts_dir = Path("load_test_charts")
            charts_dir.mkdir(exist_ok=True)

            # Generate charts for each scenario
            for scenario_name, results in test_results.items():
                if isinstance(results, dict) and "response_times" in results:
                    await self.generate_scenario_charts(scenario_name, results, charts_dir)

        except Exception as e:
            logger.error(f"Failed to generate load test charts: {e}")

    async def generate_scenario_charts(self, scenario_name: str, results: Dict, charts_dir: Path):
        """Generate charts for a specific scenario"""
        try:
            # Response time distribution
            plt.figure(figsize=(12, 6))
            plt.hist(results["response_times"], bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(results["avg_response_time"], color='red', linestyle='--', label=f'Avg: {results["avg_response_time"]:.2f}ms')
            plt.axvline(results["p95_response_time"], color='orange', linestyle='--', label=f'P95: {results["p95_response_time"]:.2f}ms')
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Frequency')
            plt.title(f'Response Time Distribution - {scenario_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(charts_dir / f"{scenario_name}_response_times.png")
            plt.close()

            # Throughput over time
            if results["throughput_data"]:
                plt.figure(figsize=(12, 6))
                timestamps = [d["timestamp"] for d in results["throughput_data"]]
                throughputs = [d["throughput_rps"] for d in results["throughput_data"]]

                plt.plot(timestamps, throughputs, marker='o', linewidth=2, markersize=4)
                plt.xlabel('Time (seconds from start)')
                plt.ylabel('Throughput (requests/second)')
                plt.title(f'Throughput Over Time - {scenario_name}')
                plt.grid(True, alpha=0.3)
                plt.savefig(charts_dir / f"{scenario_name}_throughput.png")
                plt.close()

            # System metrics
            if results["system_metrics"]:
                plt.figure(figsize=(12, 8))
                timestamps = [m["timestamp"] for m in results["system_metrics"]]
                cpu_values = [m["cpu_percent"] for m in results["system_metrics"]]
                memory_values = [m["memory_percent"] for m in results["system_metrics"]]

                plt.subplot(2, 1, 1)
                plt.plot(timestamps, cpu_values, color='red', linewidth=2)
                plt.ylabel('CPU Usage (%)')
                plt.title(f'System Metrics - {scenario_name}')
                plt.grid(True, alpha=0.3)

                plt.subplot(2, 1, 2)
                plt.plot(timestamps, memory_values, color='blue', linewidth=2)
                plt.xlabel('Time (seconds from start)')
                plt.ylabel('Memory Usage (%)')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(charts_dir / f"{scenario_name}_system_metrics.png")
                plt.close()

        except Exception as e:
            logger.error(f"Failed to generate charts for {scenario_name}: {e}")

    async def run_complete_load_test_suite(self) -> Dict:
        """Run complete load test suite"""
        logger.info("Starting complete load test suite...")

        test_results = {}

        # Run load tests for each scenario
        for scenario_name, scenario_config in self.load_config["scenarios"].items():
            logger.info(f"Running load test for scenario: {scenario_name}")

            # Test different user counts
            for user_count in scenario_config["concurrent_users"]:
                result = await self.run_load_scenario(scenario_name, user_count)
                key = f"{scenario_name}_{user_count}_users"
                test_results[key] = result

        # Run stress tests
        for scenario_name in ["knowledge_workload", "mixed_workload"]:
            logger.info(f"Running stress test for scenario: {scenario_name}")
            stress_result = await self.run_stress_test(scenario_name)
            test_results[f"{scenario_name}_stress_test"] = stress_result

        # Generate final report
        return await self.generate_load_test_report(test_results)

async def main():
    """Main function to run load tests"""
    tester = LoadTester()

    # Run complete load test suite
    report = await tester.run_complete_load_test_suite()
    print(f"Load test completed. Report saved to agency_swarm_load_test_report.json")

if __name__ == "__main__":
    asyncio.run(main())