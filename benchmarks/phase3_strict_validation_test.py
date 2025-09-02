#!/usr/bin/env python
"""
Phase 3 Strict Validation Test - DGTS/NLNH Compliant
Tests External Validator with proper context for strict validation
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import httpx
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrictValidationTester:
    """Tests External Validator with comprehensive context"""
    
    def __init__(self):
        self.validator_url = "http://localhost:8053"
        self.results = []
        
    async def test_code_with_full_context(self):
        """Test code validation with complete context - NLNH compliant"""
        
        test_cases = [
            {
                "name": "Secure Authentication Implementation",
                "code": """
import hashlib
import secrets
import hmac
from typing import Optional

class SecureAuthenticator:
    '''Production-ready authentication with proper security measures'''
    
    def __init__(self):
        self.pepper = secrets.token_hex(32)  # Server-side pepper
        
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        '''Hash password with salt and pepper using PBKDF2'''
        if not salt:
            salt = secrets.token_hex(32)
        
        # Validate input
        if not isinstance(password, str):
            raise TypeError("Password must be a string")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
            
        # Use PBKDF2 with 100,000 iterations
        key = hashlib.pbkdf2_hmac(
            'sha256',
            (password + self.pepper).encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return key.hex(), salt
        
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        '''Verify password against stored hash'''
        computed_hash, _ = self.hash_password(password, salt)
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed_hash, stored_hash)
""",
                "context": {
                    "files": ["auth.py", "security_config.py", "requirements.txt"],
                    "docs": [
                        "Security Requirements: Use PBKDF2 or bcrypt for password hashing",
                        "Minimum 100,000 iterations for PBKDF2",
                        "Use constant-time comparison for hash verification",
                        "Implement salt and pepper for additional security"
                    ],
                    "prp": [
                        "Implement secure password hashing",
                        "Prevent timing attacks",
                        "Use industry-standard cryptographic functions",
                        "Validate all inputs"
                    ],
                    "entities": ["SecureAuthenticator", "PBKDF2", "hashlib", "secrets", "hmac"]
                },
                "expected_status": "pass",
                "reason": "Implements all security best practices with proper context"
            },
            {
                "name": "SQL Injection Vulnerable Code",
                "code": """
import sqlite3

class UserDatabase:
    '''Database handler with SQL injection vulnerability'''
    
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        
    def get_user(self, username: str):
        '''UNSAFE: Direct string interpolation in SQL query'''
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor = self.conn.execute(query)  # SQL INJECTION VULNERABILITY
        return cursor.fetchone()
        
    def authenticate(self, username: str, password: str):
        '''UNSAFE: Multiple injection points'''
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        cursor = self.conn.execute(query)  # CRITICAL SECURITY FLAW
        return cursor.fetchone() is not None
""",
                "context": {
                    "files": ["database.py", "models.py"],
                    "docs": [
                        "Security Policy: Never use string interpolation in SQL queries",
                        "Always use parameterized queries",
                        "SQL injection is a critical vulnerability"
                    ],
                    "prp": [
                        "Implement secure database queries",
                        "Prevent SQL injection attacks",
                        "Use parameterized queries"
                    ],
                    "entities": ["UserDatabase", "sqlite3", "SQL injection"]
                },
                "expected_status": "fail",
                "reason": "Contains critical SQL injection vulnerability"
            },
            {
                "name": "Rate Limiter with Proper Implementation",
                "code": """
import time
from collections import defaultdict
from typing import Dict, Optional
import threading

class RateLimiter:
    '''Thread-safe rate limiter using token bucket algorithm'''
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.buckets: Dict[str, Dict] = defaultdict(lambda: {
            'tokens': max_requests,
            'last_refill': time.time()
        })
        self.lock = threading.Lock()
        
    def _refill_bucket(self, bucket: Dict) -> None:
        '''Refill tokens based on elapsed time'''
        now = time.time()
        elapsed = now - bucket['last_refill']
        tokens_to_add = elapsed * (self.max_requests / self.window_seconds)
        
        bucket['tokens'] = min(
            self.max_requests,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_refill'] = now
        
    def is_allowed(self, client_id: str) -> bool:
        '''Check if request is allowed for client'''
        with self.lock:
            bucket = self.buckets[client_id]
            self._refill_bucket(bucket)
            
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                return True
            return False
            
    def get_retry_after(self, client_id: str) -> float:
        '''Get seconds until next request is allowed'''
        with self.lock:
            bucket = self.buckets[client_id]
            if bucket['tokens'] >= 1:
                return 0.0
            
            tokens_needed = 1 - bucket['tokens']
            seconds_per_token = self.window_seconds / self.max_requests
            return tokens_needed * seconds_per_token
""",
                "context": {
                    "files": ["rate_limiter.py", "middleware.py", "config.py"],
                    "docs": [
                        "Implement token bucket algorithm for rate limiting",
                        "Must be thread-safe for production use",
                        "Support per-client rate limiting",
                        "Provide retry-after information"
                    ],
                    "prp": [
                        "Implement rate limiting to prevent abuse",
                        "Use token bucket algorithm",
                        "Ensure thread safety",
                        "Track per-client limits"
                    ],
                    "entities": ["RateLimiter", "token bucket", "threading", "defaultdict"]
                },
                "expected_status": "pass",
                "reason": "Correctly implements thread-safe token bucket rate limiting"
            }
        ]
        
        logger.info("Starting strict validation tests with full context...")
        
        for test_case in test_cases:
            logger.info(f"\nTesting: {test_case['name']}")
            
            # Call External Validator with full context
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.validator_url}/validate",
                    json={
                        "output": test_case["code"],
                        "context": test_case["context"]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "unknown")
                    
                    # Check if validation matches expectation
                    passed = (status.lower() == test_case["expected_status"])
                    
                    logger.info(f"  Status: {status} (expected: {test_case['expected_status']})")
                    logger.info(f"  Result: {'✅ PASS' if passed else '❌ FAIL'}")
                    
                    if result.get("issues"):
                        logger.info(f"  Issues found: {len(result['issues'])}")
                        for issue in result["issues"][:2]:  # Show first 2 issues
                            logger.info(f"    - {issue.get('category')}: {issue.get('message')[:100]}...")
                    
                    self.results.append({
                        "test": test_case["name"],
                        "expected": test_case["expected_status"],
                        "actual": status.lower(),
                        "passed": passed,
                        "reason": test_case["reason"],
                        "issues_count": len(result.get("issues", []))
                    })
                else:
                    logger.error(f"  Failed to call validator: {response.status_code}")
                    self.results.append({
                        "test": test_case["name"],
                        "expected": test_case["expected_status"],
                        "actual": "error",
                        "passed": False,
                        "reason": f"Validator call failed: {response.status_code}"
                    })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "strict_validation_with_context",
            "principle": "DGTS/NLNH Compliant",
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "results": self.results,
            "validation_metrics": {
                "precision": passed_tests / total_tests if total_tests > 0 else 0,
                "strict_mode": True,
                "context_provided": True
            }
        }
        
        return report

async def main():
    """Run strict validation tests"""
    
    logger.info("=" * 80)
    logger.info("Phase 3 Strict Validation Test - DGTS/NLNH Compliant")
    logger.info("=" * 80)
    
    tester = StrictValidationTester()
    
    # Run tests
    await tester.test_code_with_full_context()
    
    # Generate report
    report = tester.generate_report()
    
    # Save report
    output_dir = Path("scwt-results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"phase3_strict_validation_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {report['summary']['total_tests']}")
    logger.info(f"Passed: {report['summary']['passed']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    logger.info(f"Success Rate: {report['summary']['success_rate']:.1%}")
    logger.info(f"Validation Precision: {report['validation_metrics']['precision']:.1%}")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    for result in report['results']:
        status = "✅" if result['passed'] else "❌"
        logger.info(f"{status} {result['test']}")
        logger.info(f"   Expected: {result['expected']}, Got: {result['actual']}")
        logger.info(f"   Reason: {result['reason']}")
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Return success if precision meets target
    target_precision = 0.92
    achieved_precision = report['validation_metrics']['precision']
    
    if achieved_precision >= target_precision:
        logger.info(f"\n✅ Validation precision target met: {achieved_precision:.1%} >= {target_precision:.1%}")
        return 0
    else:
        logger.warning(f"\n⚠️ Validation precision below target: {achieved_precision:.1%} < {target_precision:.1%}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)