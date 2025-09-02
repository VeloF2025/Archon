"""
SCWT (Structured Contextual Workflow Testing) Benchmark for Phase 5 External Validator
Tests all PRD requirements with real-world scenarios
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import httpx
from pathlib import Path

# Test configuration
VALIDATOR_URL = "http://localhost:8053"
BENCHMARK_RESULTS_DIR = Path("benchmark_results")
BENCHMARK_RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class TestCase:
    """Individual test case for SCWT benchmark"""
    id: str
    category: str
    description: str
    input_code: str
    expected_issues: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    validation_type: str = "code"
    should_pass: bool = True
    contains_hallucination: bool = False
    contains_gaming: bool = False
    
    
@dataclass
class BenchmarkResult:
    """Result of a single benchmark test"""
    test_id: str
    category: str
    passed: bool
    execution_time_ms: float
    detected_issues: List[str]
    expected_issues: List[str]
    confidence_score: float
    hallucination_detected: bool
    gaming_detected: bool
    token_count: int = 0
    error_message: Optional[str] = None


class SCWTBenchmark:
    """SCWT Benchmark Suite for External Validator"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results: List[BenchmarkResult] = []
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering all PRD requirements"""
        
        return [
            # ========== HALLUCINATION DETECTION TESTS ==========
            TestCase(
                id="HALL-001",
                category="hallucination_detection",
                description="Non-existent function call",
                input_code="""
def process_data(items):
    result = magic_transform(items)  # Non-existent function
    return result
""",
                expected_issues=["undefined function 'magic_transform'"],
                contains_hallucination=True,
                should_pass=False
            ),
            
            TestCase(
                id="HALL-002",
                category="hallucination_detection",
                description="Non-existent module import",
                input_code="""
import super_ai_module  # Doesn't exist

def analyze():
    return super_ai_module.process()
""",
                expected_issues=["module 'super_ai_module' not found"],
                contains_hallucination=True,
                should_pass=False
            ),
            
            TestCase(
                id="HALL-003",
                category="hallucination_detection",
                description="Valid code with real functions",
                input_code="""
import json

def parse_json(data):
    return json.loads(data)
""",
                expected_issues=[],
                contains_hallucination=False,
                should_pass=True
            ),
            
            # ========== GAMING DETECTION TESTS ==========
            TestCase(
                id="GAME-001",
                category="gaming_detection",
                description="Mock data return pattern",
                input_code="""
def get_user_data(user_id):
    # TODO: Implement actual database query
    return "mock_user_data"
""",
                expected_issues=["Mock data return", "Gaming pattern detected"],
                contains_gaming=True,
                should_pass=False
            ),
            
            TestCase(
                id="GAME-002",
                category="gaming_detection",
                description="Fake test implementation",
                input_code="""
def test_feature():
    assert True  # Always passes
    return "test_passed"
""",
                expected_issues=["Meaningless assertion", "Gaming pattern"],
                contains_gaming=True,
                should_pass=False
            ),
            
            TestCase(
                id="GAME-003",
                category="gaming_detection",
                description="Commented validation",
                input_code="""
def validate_input(data):
    # validation_required
    # if not data:
    #     raise ValueError("Invalid data")
    return True
""",
                expected_issues=["Commented validation", "Gaming pattern"],
                contains_gaming=True,
                should_pass=False
            ),
            
            # ========== KNOWLEDGE REUSE TESTS ==========
            TestCase(
                id="KNOW-001",
                category="knowledge_reuse",
                description="Code matching known patterns",
                input_code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
                expected_issues=[],
                context={"pattern": "recursive_fibonacci"},
                should_pass=True
            ),
            
            TestCase(
                id="KNOW-002",
                category="knowledge_reuse",
                description="API endpoint with context",
                input_code="""
@app.post("/users")
async def create_user(user: UserModel):
    db_user = await database.users.insert_one(user.dict())
    return {"id": str(db_user.inserted_id)}
""",
                expected_issues=[],
                context={
                    "framework": "FastAPI",
                    "database": "MongoDB",
                    "pattern": "REST_CRUD"
                },
                should_pass=True
            ),
            
            # ========== DETERMINISTIC CHECKS ==========
            TestCase(
                id="DET-001",
                category="deterministic",
                description="Python syntax error",
                input_code="""
def broken_function()
    return "missing colon"
""",
                expected_issues=["syntax error", "missing colon"],
                should_pass=False
            ),
            
            TestCase(
                id="DET-002",
                category="deterministic",
                description="Type annotation errors",
                input_code="""
def add_numbers(a: int, b: int) -> str:
    return a + b  # Returns int, not str
""",
                expected_issues=["type mismatch", "returns int"],
                should_pass=False
            ),
            
            # ========== CROSS-CHECK VALIDATION ==========
            TestCase(
                id="CROSS-001",
                category="cross_check",
                description="Implementation matches requirements",
                input_code="""
def calculate_discount(price: float, discount_percent: float) -> float:
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")
    return price * (1 - discount_percent / 100)
""",
                expected_issues=[],
                context={
                    "requirements": "Calculate discounted price with validation",
                    "constraints": "Discount between 0-100%"
                },
                should_pass=True
            ),
            
            TestCase(
                id="CROSS-002",
                category="cross_check",
                description="Missing requirement implementation",
                input_code="""
def calculate_discount(price: float, discount_percent: float) -> float:
    return price * (1 - discount_percent / 100)
""",
                expected_issues=["missing validation", "no boundary check"],
                context={
                    "requirements": "Calculate discounted price with validation",
                    "constraints": "Must validate discount between 0-100%"
                },
                should_pass=False
            ),
            
            # ========== PERFORMANCE EDGE CASES ==========
            TestCase(
                id="PERF-001",
                category="performance",
                description="Large code file",
                input_code="\n".join([
                    f"def function_{i}():\n    return {i}"
                    for i in range(100)
                ]),
                expected_issues=[],
                should_pass=True
            ),
            
            TestCase(
                id="PERF-002",
                category="performance",
                description="Complex nested structure",
                input_code="""
def complex_function(data):
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = {
                k: [v * 2 if isinstance(v, (int, float)) else v 
                    for v in (val if isinstance(val, list) else [val])]
                for k, val in value.items()
            }
        else:
            result[key] = value
    return result
""",
                expected_issues=[],
                should_pass=True
            ),
            
            # ========== SECURITY PATTERNS ==========
            TestCase(
                id="SEC-001",
                category="security",
                description="SQL injection vulnerability",
                input_code="""
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return database.execute(query)
""",
                expected_issues=["SQL injection", "unsafe query"],
                should_pass=False
            ),
            
            TestCase(
                id="SEC-002",
                category="security",
                description="Hardcoded credentials",
                input_code="""
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "admin123"

def connect():
    return Database(password=DATABASE_PASSWORD)
""",
                expected_issues=["hardcoded credentials", "security risk"],
                should_pass=False
            ),
            
            # ========== REAL-WORLD SCENARIOS ==========
            TestCase(
                id="REAL-001",
                category="real_world",
                description="React component with hooks",
                input_code="""
import React, { useState, useEffect } from 'react';

function UserProfile({ userId }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUser(userId).then(data => {
            setUser(data);
            setLoading(false);
        });
    }, [userId]);
    
    if (loading) return <div>Loading...</div>;
    return <div>{user?.name}</div>;
}
""",
                expected_issues=[],
                context={"framework": "React", "language": "JavaScript"},
                should_pass=True
            ),
            
            TestCase(
                id="REAL-002",
                category="real_world",
                description="FastAPI endpoint with validation",
                input_code="""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

class UserCreate(BaseModel):
    email: str
    age: int
    
    @validator('age')
    def validate_age(cls, v):
        if v < 18:
            raise ValueError('Must be 18 or older')
        return v

@app.post("/users")
async def create_user(user: UserCreate):
    if await user_exists(user.email):
        raise HTTPException(status_code=400, detail="User already exists")
    return await save_user(user)
""",
                expected_issues=[],
                context={"framework": "FastAPI", "validation": "Pydantic"},
                should_pass=True
            ),
        ]
    
    async def run_test_case(self, test_case: TestCase) -> BenchmarkResult:
        """Run a single test case against the validator"""
        
        start_time = time.time()
        result = BenchmarkResult(
            test_id=test_case.id,
            category=test_case.category,
            passed=False,
            execution_time_ms=0,
            detected_issues=[],
            expected_issues=test_case.expected_issues,
            confidence_score=0,
            hallucination_detected=False,
            gaming_detected=False
        )
        
        try:
            # Prepare validation request
            request_data = {
                "output": test_case.input_code,
                "validation_type": test_case.validation_type,
                "context": test_case.context
            }
            
            # Call validator
            response = await self.client.post(
                f"{VALIDATOR_URL}/validate",
                json=request_data
            )
            
            if response.status_code != 200:
                result.error_message = f"HTTP {response.status_code}: {response.text}"
                return result
            
            validation_result = response.json()
            
            # Extract results
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.confidence_score = validation_result.get("metrics", {}).get("confidence_score", 0)
            result.token_count = validation_result.get("metrics", {}).get("token_count", 0)
            
            # Check detected issues
            issues = validation_result.get("issues", [])
            result.detected_issues = [issue.get("message", "") for issue in issues]
            
            # Check for hallucination detection
            for issue in issues:
                if "hallucination" in issue.get("category", "").lower():
                    result.hallucination_detected = True
                if "gaming" in issue.get("category", "").lower():
                    result.gaming_detected = True
            
            # Determine if test passed
            if test_case.should_pass:
                result.passed = validation_result.get("status") == "PASS"
            else:
                result.passed = validation_result.get("status") == "FAIL"
            
            # Additional checks for specific detection
            if test_case.contains_hallucination:
                result.passed = result.passed and result.hallucination_detected
            
            if test_case.contains_gaming:
                result.passed = result.passed and result.gaming_detected
                
        except Exception as e:
            result.error_message = str(e)
            
        return result
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete SCWT benchmark suite"""
        
        print("=" * 80)
        print("SCWT BENCHMARK FOR PHASE 5 EXTERNAL VALIDATOR")
        print("=" * 80)
        print(f"Starting benchmark with {len(self.test_cases)} test cases...")
        print()
        
        # Run all tests
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] Running {test_case.id}: {test_case.description}...", end=" ")
            result = await self.run_test_case(test_case)
            self.results.append(result)
            
            if result.passed:
                print("[PASS]")
            else:
                print(f"[FAIL] - {result.error_message or 'Detection mismatch'}")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Generate report
        report = self._generate_report(metrics)
        
        # Save results
        self._save_results(report)
        
        return report
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate benchmark metrics"""
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        # Category-specific metrics
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {
                    "total": 0,
                    "passed": 0,
                    "execution_times": [],
                    "confidence_scores": []
                }
            
            cat = categories[result.category]
            cat["total"] += 1
            if result.passed:
                cat["passed"] += 1
            cat["execution_times"].append(result.execution_time_ms)
            cat["confidence_scores"].append(result.confidence_score)
        
        # Calculate category metrics
        for cat_name, cat_data in categories.items():
            cat_data["pass_rate"] = cat_data["passed"] / cat_data["total"] * 100
            cat_data["avg_execution_time"] = statistics.mean(cat_data["execution_times"])
            cat_data["avg_confidence"] = statistics.mean(cat_data["confidence_scores"])
        
        # Hallucination detection metrics
        hall_tests = [r for r in self.results if any(tc.id == r.test_id and tc.contains_hallucination for tc in self.test_cases)]
        hall_detected = sum(1 for r in hall_tests if r.hallucination_detected)
        hall_rate = (hall_detected / len(hall_tests) * 100) if hall_tests else 0
        
        # Gaming detection metrics
        game_tests = [r for r in self.results if any(tc.id == r.test_id and tc.contains_gaming for tc in self.test_cases)]
        game_detected = sum(1 for r in game_tests if r.gaming_detected)
        game_rate = (game_detected / len(game_tests) * 100) if game_tests else 0
        
        # Performance metrics
        all_execution_times = [r.execution_time_ms for r in self.results]
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_pass_rate": passed_tests / total_tests * 100,
            "categories": categories,
            "hallucination_detection_rate": hall_rate,
            "gaming_detection_rate": game_rate,
            "avg_execution_time_ms": statistics.mean(all_execution_times),
            "median_execution_time_ms": statistics.median(all_execution_times),
            "p95_execution_time_ms": sorted(all_execution_times)[int(len(all_execution_times) * 0.95)],
            "total_tokens": sum(r.token_count for r in self.results)
        }
    
    def _generate_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        
        report = {
            "benchmark": "SCWT Phase 5 External Validator",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": metrics["total_tests"],
                "passed": metrics["passed_tests"],
                "failed": metrics["total_tests"] - metrics["passed_tests"],
                "pass_rate": f"{metrics['overall_pass_rate']:.1f}%"
            },
            "prd_compliance": {
                "hallucination_reduction": {
                    "target": "≤10%",
                    "achieved": f"{100 - metrics['hallucination_detection_rate']:.1f}%",
                    "status": "PASS" if metrics["hallucination_detection_rate"] >= 90 else "FAIL"
                },
                "gaming_detection": {
                    "target": "100%",
                    "achieved": f"{metrics['gaming_detection_rate']:.1f}%",
                    "status": "PASS" if metrics["gaming_detection_rate"] == 100 else "FAIL"
                },
                "validation_speed": {
                    "target": "<2000ms",
                    "achieved": f"{metrics['avg_execution_time_ms']:.0f}ms",
                    "status": "PASS" if metrics["avg_execution_time_ms"] < 2000 else "FAIL"
                }
            },
            "category_breakdown": metrics["categories"],
            "performance": {
                "avg_response_time": f"{metrics['avg_execution_time_ms']:.0f}ms",
                "median_response_time": f"{metrics['median_execution_time_ms']:.0f}ms",
                "p95_response_time": f"{metrics['p95_execution_time_ms']:.0f}ms",
                "total_tokens_processed": metrics["total_tokens"]
            },
            "failed_tests": [
                {
                    "id": r.test_id,
                    "category": r.category,
                    "expected": r.expected_issues,
                    "detected": r.detected_issues,
                    "error": r.error_message
                }
                for r in self.results if not r.passed
            ],
            "metrics": metrics
        }
        
        return report
    
    def _save_results(self, report: Dict[str, Any]):
        """Save benchmark results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = BENCHMARK_RESULTS_DIR / f"scwt_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of the benchmark results"""
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} | Failed: {summary['failed']}")
        print(f"Overall Pass Rate: {summary['pass_rate']}")
        
        print("\n" + "-" * 40)
        print("PRD COMPLIANCE")
        print("-" * 40)
        
        for metric, data in report["prd_compliance"].items():
            status_symbol = "[✓]" if data["status"] == "PASS" else "[✗]"
            print(f"{status_symbol} {metric.replace('_', ' ').title()}")
            print(f"    Target: {data['target']} | Achieved: {data['achieved']}")
        
        print("\n" + "-" * 40)
        print("CATEGORY PERFORMANCE")
        print("-" * 40)
        
        for cat_name, cat_data in report["category_breakdown"].items():
            print(f"\n{cat_name.replace('_', ' ').title()}:")
            print(f"  Pass Rate: {cat_data['pass_rate']:.1f}%")
            print(f"  Avg Time: {cat_data['avg_execution_time']:.0f}ms")
            print(f"  Avg Confidence: {cat_data['avg_confidence']:.2f}")
        
        if report["failed_tests"]:
            print("\n" + "-" * 40)
            print("FAILED TESTS")
            print("-" * 40)
            for test in report["failed_tests"][:5]:  # Show first 5 failures
                print(f"\n{test['id']} ({test['category']}):")
                print(f"  Expected: {test['expected']}")
                print(f"  Detected: {test['detected']}")
                if test['error']:
                    print(f"  Error: {test['error']}")
            
            if len(report["failed_tests"]) > 5:
                print(f"\n... and {len(report['failed_tests']) - 5} more failures")
        
        print("\n" + "=" * 80)
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Run the SCWT benchmark"""
    
    benchmark = SCWTBenchmark()
    
    try:
        # Check if validator is running
        response = await benchmark.client.get(f"{VALIDATOR_URL}/health")
        if response.status_code != 200:
            print("ERROR: Validator is not running or unhealthy")
            print("Please start the validator with: docker compose --profile validator up -d")
            return
        
        # Run benchmark
        report = await benchmark.run_benchmark()
        
        # Print summary
        benchmark.print_summary(report)
        
        # Check overall pass/fail
        if report["summary"]["pass_rate"].rstrip('%').replace('.', '').isdigit():
            pass_rate = float(report["summary"]["pass_rate"].rstrip('%'))
            if pass_rate >= 90:
                print("\n✅ BENCHMARK PASSED - Validator meets PRD requirements!")
            else:
                print(f"\n❌ BENCHMARK FAILED - Pass rate {pass_rate}% is below 90% threshold")
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
    finally:
        await benchmark.cleanup()


if __name__ == "__main__":
    print("SCWT Benchmark for Phase 5 External Validator")
    print("This will test all PRD requirements including:")
    print("- Hallucination detection")
    print("- Gaming pattern detection")
    print("- Knowledge reuse")
    print("- Performance metrics")
    print("- Security validation")
    print("\nTo run: python test_phase5_scwt_benchmark.py")
    print("\nNOT RUNNING YET as requested.")
    # asyncio.run(main())  # Uncommented when ready to run