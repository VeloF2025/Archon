#!/usr/bin/env python3
"""
Performance Benchmarks for Agency Swarm System
Enterprise-grade performance validation and monitoring
"""

import asyncio
import aiohttp
import time
import json
import statistics
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Enterprise performance benchmarking system"""

    def __init__(self):
        self.services = {
            "frontend": "http://localhost:3737",
            "api": "http://localhost:8181",
            "mcp": "http://localhost:8051",
            "agents": "http://localhost:8052"
        }
        self.benchmark_results = {}
        self.enterprise_thresholds = {
            "api_response_time": 200,  # ms
            "page_load_time": 1500,   # ms
            "throughput": 1000,       # requests/second
            "error_rate": 0.01,        # 1%
            "memory_usage": 512,       # MB
            "cpu_usage": 70,           # %
            "database_query_time": 50, # ms
            "websocket_latency": 100   # ms
        }

    async def benchmark_api_endpoints(self):
        """Benchmark all API endpoints"""
        logger.info("Benchmarking API endpoints...")

        endpoints = [
            ("GET", "/api/knowledge/items"),
            ("POST", "/api/knowledge/search"),
            ("GET", "/api/templates"),
            ("POST", "/api/templates"),
            ("GET", "/api/patterns"),
            ("POST", "/api/patterns"),
            ("GET", "/api/agents"),
            ("POST", "/api/agents/execute")
        ]

        results = []
        for method, endpoint in endpoints:
            try:
                response_times = []
                error_count = 0

                # Run 100 requests for statistical significance
                for i in range(100):
                    start_time = time.time()
                    try:
                        async with aiohttp.ClientSession() as session:
                            if method == "GET":
                                async with session.get(f"{self.services['api']}{endpoint}") as response:
                                    await response.json()
                            else:
                                async with session.post(f"{self.services['api']}{endpoint}", json={"test": True}) as response:
                                    await response.json()

                            response_time = (time.time() - start_time) * 1000  # Convert to ms
                            response_times.append(response_time)
                    except Exception as e:
                        error_count += 1
                        logger.warning(f"Request {i} to {endpoint} failed: {e}")

                # Calculate statistics
                if response_times:
                    avg_time = statistics.mean(response_times)
                    median_time = statistics.median(response_times)
                    p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                    p99_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
                    error_rate = error_count / 100

                    threshold_met = avg_time <= self.enterprise_thresholds["api_response_time"]

                    results.append({
                        "endpoint": endpoint,
                        "method": method,
                        "avg_response_time": avg_time,
                        "median_response_time": median_time,
                        "p95_response_time": p95_time,
                        "p99_response_time": p99_time,
                        "error_rate": error_rate,
                        "threshold_met": threshold_met,
                        "response_times": response_times
                    })
            except Exception as e:
                logger.error(f"Failed to benchmark {endpoint}: {e}")

        return {
            "test_name": "API Endpoint Performance",
            "threshold": self.enterprise_thresholds["api_response_time"],
            "results": results,
            "summary": self.calculate_api_summary(results)
        }

    async def benchmark_knowledge_operations(self):
        """Benchmark knowledge management operations"""
        logger.info("Benchmarking knowledge operations...")

        operations = [
            ("Document Upload", "/api/knowledge/upload", {"url": "https://example.com/test"}),
            ("Knowledge Search", "/api/knowledge/search", {"query": "test query"}),
            ("Knowledge Retrieval", "/api/knowledge/items", {}),
            ("Embedding Generation", "/api/knowledge/embed", {"text": "Test text for embedding"})
        ]

        results = []
        for op_name, endpoint, data in operations:
            try:
                operation_times = []
                success_count = 0

                for i in range(50):  # Fewer iterations for expensive operations
                    start_time = time.time()
                    try:
                        async with aiohttp.ClientSession() as session:
                            if data:
                                async with session.post(f"{self.services['api']}{endpoint}", json=data) as response:
                                    await response.json()
                            else:
                                async with session.get(f"{self.services['api']}{endpoint}") as response:
                                    await response.json()

                            operation_time = (time.time() - start_time) * 1000
                            operation_times.append(operation_time)
                            success_count += 1
                    except Exception as e:
                        logger.warning(f"Knowledge operation {op_name} attempt {i} failed: {e}")

                if operation_times:
                    avg_time = statistics.mean(operation_times)
                    throughput = success_count / sum(operation_times) * 1000 * 60  # operations per minute

                    results.append({
                        "operation": op_name,
                        "avg_time_ms": avg_time,
                        "throughput_per_min": throughput,
                        "success_rate": success_count / 50,
                        "operation_times": operation_times
                    })
            except Exception as e:
                logger.error(f"Failed to benchmark {op_name}: {e}")

        return {
            "test_name": "Knowledge Operations Performance",
            "results": results
        }

    async def benchmark_agent_execution(self):
        """Benchmark agent execution performance"""
        logger.info("Benchmarking agent execution...")

        agent_types = [
            ("Knowledge Agent", "knowledge_agent", {"query": "test knowledge query"}),
            ("Pattern Agent", "pattern_agent", {"pattern": "test_pattern"}),
            ("Collaboration Agent", "collaboration_agent", {"team": ["agent1", "agent2"]})
        ]

        results = []
        for agent_name, agent_type, task in agent_types:
            try:
                execution_times = []
                memory_usage = []

                for i in range(30):  # Agent execution is resource-intensive
                    start_time = time.time()
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f"{self.services['agents']}/agents/execute", json={
                                "agent_type": agent_type,
                                "task": task
                            }) as response:
                                result = await response.json()

                            execution_time = (time.time() - start_time) * 1000
                            execution_times.append(execution_time)

                            # Extract memory usage if available
                            if "memory_usage" in result:
                                memory_usage.append(result["memory_usage"])

                    except Exception as e:
                        logger.warning(f"Agent execution {agent_name} attempt {i} failed: {e}")

                if execution_times:
                    avg_time = statistics.mean(execution_times)
                    max_memory = max(memory_usage) if memory_usage else 0

                    results.append({
                        "agent_type": agent_name,
                        "avg_execution_time_ms": avg_time,
                        "max_memory_usage_mb": max_memory,
                        "total_executions": len(execution_times),
                        "execution_times": execution_times
                    })
            except Exception as e:
                logger.error(f"Failed to benchmark {agent_name}: {e}")

        return {
            "test_name": "Agent Execution Performance",
            "results": results
        }

    async def benchmark_mcp_tools(self):
        """Benchmark MCP tool performance"""
        logger.info("Benchmarking MCP tools...")

        # Get available tools
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.services['mcp']}/tools") as response:
                    tools_data = await response.json()
                    available_tools = tools_data.get("tools", [])[:5]  # Test first 5 tools
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            return {"test_name": "MCP Tools Performance", "error": str(e)}

        results = []
        for tool in available_tools:
            try:
                tool_times = []
                success_count = 0

                for i in range(20):  # Tool execution benchmark
                    start_time = time.time()
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f"{self.services['mcp']}/tools/{tool['name']}", json=tool.get("parameters", {})) as response:
                                await response.json()

                            tool_time = (time.time() - start_time) * 1000
                            tool_times.append(tool_time)
                            success_count += 1
                    except Exception as e:
                        logger.warning(f"MCP tool {tool['name']} attempt {i} failed: {e}")

                if tool_times:
                    avg_time = statistics.mean(tool_times)
                    success_rate = success_count / 20

                    results.append({
                        "tool_name": tool["name"],
                        "avg_execution_time_ms": avg_time,
                        "success_rate": success_rate,
                        "total_executions": len(tool_times),
                        "execution_times": tool_times
                    })
            except Exception as e:
                logger.error(f"Failed to benchmark MCP tool {tool['name']}: {e}")

        return {
            "test_name": "MCP Tools Performance",
            "results": results
        }

    async def benchmark_websocket_performance(self):
        """Benchmark WebSocket performance"""
        logger.info("Benchmarking WebSocket performance...")

        try:
            import websockets

            latency_results = []
            throughput_results = []

            # Test latency
            for i in range(50):
                start_time = time.time()
                try:
                    ws_url = self.services["api"].replace("http", "ws") + "/ws"
                    async with websockets.connect(ws_url) as websocket:
                        await websocket.send(json.dumps({"type": "ping", "id": i}))
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        latency = (time.time() - start_time) * 1000
                        latency_results.append(latency)
                except Exception as e:
                    logger.warning(f"WebSocket latency test {i} failed: {e}")

            # Test throughput (messages per second)
            try:
                ws_url = self.services["api"].replace("http", "ws") + "/ws"
                async with websockets.connect(ws_url) as websocket:
                    start_time = time.time()
                    message_count = 0

                    # Send 1000 messages as fast as possible
                    for i in range(1000):
                        await websocket.send(json.dumps({"type": "test", "id": i}))
                        message_count += 1

                    throughput_time = time.time() - start_time
                    throughput = message_count / throughput_time  # messages per second
                    throughput_results.append(throughput)

            except Exception as e:
                logger.error(f"WebSocket throughput test failed: {e}")

            return {
                "test_name": "WebSocket Performance",
                "avg_latency_ms": statistics.mean(latency_results) if latency_results else 0,
                "p95_latency_ms": statistics.quantiles(latency_results, n=20)[18] if len(latency_results) >= 20 else 0,
                "throughput_msgs_per_sec": throughput_results[0] if throughput_results else 0,
                "latency_threshold_met": statistics.mean(latency_results) <= self.enterprise_thresholds["websocket_latency"] if latency_results else False,
                "latency_results": latency_results
            }

        except ImportError:
            return {
                "test_name": "WebSocket Performance",
                "status": "skipped",
                "reason": "websockets library not available"
            }

    async def benchmark_database_operations(self):
        """Benchmark database operations"""
        logger.info("Benchmarking database operations...")

        operations = [
            ("INSERT", "/api/db/insert", {"table": "test", "data": {"test": True}}),
            ("SELECT", "/api/db/select", {"table": "test", "where": {"test": True}}),
            ("UPDATE", "/api/db/update", {"table": "test", "data": {"updated": True}}),
            ("DELETE", "/api/db/delete", {"table": "test", "where": {"test": True}})
        ]

        results = []
        for op_name, endpoint, data in operations:
            try:
                operation_times = []
                success_count = 0

                for i in range(100):
                    start_time = time.time()
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f"{self.services['api']}{endpoint}", json=data) as response:
                                await response.json()

                            operation_time = (time.time() - start_time) * 1000
                            operation_times.append(operation_time)
                            success_count += 1
                    except Exception as e:
                        logger.warning(f"DB operation {op_name} attempt {i} failed: {e}")

                if operation_times:
                    avg_time = statistics.mean(operation_times)
                    threshold_met = avg_time <= self.enterprise_thresholds["database_query_time"]

                    results.append({
                        "operation": op_name,
                        "avg_time_ms": avg_time,
                        "success_rate": success_count / 100,
                        "threshold_met": threshold_met,
                        "operation_times": operation_times
                    })
            except Exception as e:
                logger.error(f"Failed to benchmark {op_name}: {e}")

        return {
            "test_name": "Database Operations Performance",
            "threshold": self.enterprise_thresholds["database_query_time"],
            "results": results
        }

    def calculate_api_summary(self, results):
        """Calculate API performance summary"""
        if not results:
            return {"total_endpoints": 0, "threshold_met": 0, "avg_response_time": 0}

        total_endpoints = len(results)
        threshold_met = sum(1 for r in results if r["threshold_met"])
        avg_response_time = statistics.mean([r["avg_response_time"] for r in results])
        overall_error_rate = statistics.mean([r["error_rate"] for r in results])

        return {
            "total_endpoints": total_endpoints,
            "threshold_met": threshold_met,
            "threshold_met_percentage": (threshold_met / total_endpoints) * 100,
            "avg_response_time": avg_response_time,
            "overall_error_rate": overall_error_rate
        }

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        timestamp = datetime.now().isoformat()

        report = {
            "test_suite": "Agency Swarm Performance Benchmark",
            "timestamp": timestamp,
            "enterprise_thresholds": self.enterprise_thresholds,
            "benchmark_results": self.benchmark_results,
            "overall_assessment": self.calculate_overall_assessment(),
            "recommendations": self.generate_performance_recommendations()
        }

        # Save detailed report
        report_path = Path("agency_swarm_performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate performance charts
        self.generate_performance_charts()

        logger.info(f"Performance report saved to {report_path}")
        return report

    def calculate_overall_assessment(self):
        """Calculate overall performance assessment"""
        total_tests = len(self.benchmark_results)
        passing_tests = 0

        api_results = self.benchmark_results.get("API Endpoint Performance", {}).get("results", [])
        if api_results:
            threshold_met = sum(1 for r in api_results if r["threshold_met"])
            if threshold_met / len(api_results) >= 0.9:  # 90% of endpoints meet threshold
                passing_tests += 1

        db_results = self.benchmark_results.get("Database Operations Performance", {}).get("results", [])
        if db_results:
            threshold_met = sum(1 for r in db_results if r["threshold_met"])
            if threshold_met / len(db_results) >= 0.9:
                passing_tests += 1

        ws_results = self.benchmark_results.get("WebSocket Performance", {})
        if ws_results.get("latency_threshold_met", False):
            passing_tests += 1

        return {
            "total_tests": total_tests,
            "passing_tests": passing_tests,
            "performance_grade": self.calculate_performance_grade(passing_tests, total_tests),
            "ready_for_production": passing_tests >= total_tests * 0.8  # 80% pass rate
        }

    def calculate_performance_grade(self, passing_tests, total_tests):
        """Calculate performance grade"""
        if total_tests == 0:
            return "N/A"

        percentage = (passing_tests / total_tests) * 100
        if percentage >= 90:
            return "A+"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B"
        elif percentage >= 60:
            return "C"
        else:
            return "F"

    def generate_performance_recommendations(self):
        """Generate performance optimization recommendations"""
        recommendations = []

        # Check API performance
        api_results = self.benchmark_results.get("API Endpoint Performance", {}).get("results", [])
        for endpoint in api_results:
            if not endpoint["threshold_met"]:
                recommendations.append({
                    "component": "API",
                    "endpoint": endpoint["endpoint"],
                    "issue": f"Response time {endpoint['avg_response_time']:.2f}ms exceeds threshold {self.enterprise_thresholds['api_response_time']}ms",
                    "recommendation": "Implement caching, optimize database queries, or scale horizontally"
                })

        # Check database performance
        db_results = self.benchmark_results.get("Database Operations Performance", {}).get("results", [])
        for operation in db_results:
            if not operation["threshold_met"]:
                recommendations.append({
                    "component": "Database",
                    "operation": operation["operation"],
                    "issue": f"Operation time {operation['avg_time_ms']:.2f}ms exceeds threshold {self.enterprise_thresholds['database_query_time']}ms",
                    "recommendation": "Add database indexes, optimize queries, or consider read replicas"
                })

        # Check WebSocket performance
        ws_results = self.benchmark_results.get("WebSocket Performance", {})
        if ws_results.get("avg_latency_ms", 0) > self.enterprise_thresholds["websocket_latency"]:
            recommendations.append({
                "component": "WebSocket",
                "issue": f"Latency {ws_results['avg_latency_ms']:.2f}ms exceeds threshold {self.enterprise_thresholds['websocket_latency']}ms",
                "recommendation": "Optimize WebSocket handlers, consider connection pooling"
            })

        return recommendations

    def generate_performance_charts(self):
        """Generate performance visualization charts"""
        try:
            # Create charts directory
            charts_dir = Path("performance_charts")
            charts_dir.mkdir(exist_ok=True)

            # API Response Times Chart
            api_results = self.benchmark_results.get("API Endpoint Performance", {}).get("results", [])
            if api_results:
                endpoints = [r["endpoint"] for r in api_results]
                avg_times = [r["avg_response_time"] for r in api_results]
                p95_times = [r["p95_response_time"] for r in api_results]

                plt.figure(figsize=(12, 6))
                x = range(len(endpoints))
                plt.bar([i - 0.2 for i in x], avg_times, 0.4, label='Average', alpha=0.7)
                plt.bar([i + 0.2 for i in x], p95_times, 0.4, label='95th Percentile', alpha=0.7)
                plt.axhline(y=self.enterprise_thresholds["api_response_time"], color='r', linestyle='--', label='Threshold')
                plt.xlabel('API Endpoints')
                plt.ylabel('Response Time (ms)')
                plt.title('API Response Time Performance')
                plt.xticks(x, endpoints, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(charts_dir / "api_response_times.png")
                plt.close()

            # Throughput Chart
            throughput_data = {}
            for test_name, result in self.benchmark_results.items():
                if "results" in result:
                    for item in result["results"]:
                        if "throughput_per_min" in item:
                            throughput_data[item["operation"]] = item["throughput_per_min"]

            if throughput_data:
                plt.figure(figsize=(10, 6))
                operations = list(throughput_data.keys())
                throughputs = list(throughput_data.values())
                plt.bar(operations, throughputs, alpha=0.7)
                plt.xlabel('Operations')
                plt.ylabel('Throughput (operations/minute)')
                plt.title('System Throughput Performance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(charts_dir / "throughput_performance.png")
                plt.close()

            logger.info("Performance charts generated in performance_charts/ directory")

        except Exception as e:
            logger.error(f"Failed to generate performance charts: {e}")

    async def run_complete_benchmark(self):
        """Run complete performance benchmark suite"""
        logger.info("Starting complete performance benchmark...")

        # Run all benchmarks
        benchmark_functions = [
            self.benchmark_api_endpoints,
            self.benchmark_knowledge_operations,
            self.benchmark_agent_execution,
            self.benchmark_mcp_tools,
            self.benchmark_websocket_performance,
            self.benchmark_database_operations
        ]

        for benchmark_func in benchmark_functions:
            try:
                result = await benchmark_func()
                self.benchmark_results[result["test_name"]] = result
                logger.info(f"✓ {benchmark_func.__name__} completed")
            except Exception as e:
                logger.error(f"✗ {benchmark_func.__name__} failed: {e}")

        # Generate final report
        return self.generate_performance_report()

async def main():
    """Main function to run performance benchmarks"""
    benchmark = PerformanceBenchmark()
    return await benchmark.run_complete_benchmark()

if __name__ == "__main__":
    asyncio.run(main())