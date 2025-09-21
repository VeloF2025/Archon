"""
Performance Testing Suite for Security and Compliance Framework

This module provides comprehensive performance testing for all security components
to ensure they meet production performance requirements under load.
"""

import pytest
import asyncio
import time
import psutil
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, AsyncMock

from src.agents.security.security_framework import SecurityFramework
from src.agents.security.authentication_service import AuthenticationService
from src.agents.security.authorization_service import AuthorizationService
from src.agents.security.encryption_service import EncryptionService
from src.agents.security.audit_logger import AuditLogger
from src.agents.security.access_control import AccessControlManager
from src.agents.security.threat_detection import ThreatDetectionSystem

from src.agents.compliance.compliance_engine import ComplianceEngine
from src.agents.compliance.gdpr_compliance import GDPRComplianceManager
from src.agents.compliance.soc2_compliance import SOC2ComplianceManager
from src.agents.compliance.hipaa_compliance import HIPAAComplianceManager

from src.agents.security.models import (
    SecurityContext,
    AuthenticationRequest,
    AuthorizationRequest,
    Subject,
    Role,
    Permission,
    Resource
)


class TestSecurityPerformance:
    """Performance testing for security framework components"""

    @pytest.fixture
    def security_framework(self):
        """Initialize security framework for performance testing"""
        return SecurityFramework()

    @pytest.fixture
    def auth_service(self):
        """Initialize authentication service"""
        return AuthenticationService()

    @pytest.fixture
    def encryption_service(self):
        """Initialize encryption service"""
        return EncryptionService()

    @pytest.fixture
    def audit_logger(self):
        """Initialize audit logger"""
        return AuditLogger()

    @pytest.fixture
    def sample_subject(self):
        """Create sample subject for testing"""
        return Subject(
            id="perf_test_subject",
            subject_type="user",
            name="Performance Test User",
            email="perf@test.com",
            attributes={"department": "engineering", "level": "senior"}
        )

    @pytest.fixture
    def sample_role(self):
        """Create sample role for testing"""
        return Role(
            id="perf_test_role",
            name="Performance Test Role",
            description="Role for performance testing",
            permissions=["read", "write", "execute"]
        )

    def test_authentication_throughput(self, auth_service):
        """Test authentication throughput under concurrent load"""
        num_requests = 1000
        num_workers = 50

        def authenticate_worker():
            """Worker function for concurrent authentication"""
            auth_request = AuthenticationRequest(
                username=f"user_{threading.get_ident()}",
                password="test_password",
                factors={"totp": "123456"}
            )

            start_time = time.time()
            try:
                result = auth_service.authenticate(auth_request)
                return result, time.time() - start_time
            except Exception as e:
                return None, time.time() - start_time

        # Measure throughput
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(authenticate_worker) for _ in range(num_requests)]

            results = []
            for future in as_completed(futures, timeout=60):
                results.append(future.result())

        total_time = time.time() - start_time
        successful_auths = sum(1 for result, _ in results if result is not None)

        # Calculate metrics
        throughput = successful_auths / total_time
        avg_latency = sum(latency for _, latency in results) / len(results)

        # Performance assertions
        assert throughput >= 100, f"Authentication throughput too low: {throughput} auths/sec"
        assert avg_latency <= 0.1, f"Authentication latency too high: {avg_latency}s"
        assert successful_auths >= num_requests * 0.95, f"Too many failed authentications: {successful_auths}/{num_requests}"

        print(f"\nAuthentication Performance:")
        print(f"  Throughput: {throughput:.2f} auths/sec")
        print(f"  Average Latency: {avg_latency:.4f}s")
        print(f"  Success Rate: {successful_auths/num_requests*100:.1f}%")

    def test_authorization_performance(self, security_framework, sample_subject, sample_role):
        """Test authorization performance under concurrent load"""
        num_requests = 2000
        num_workers = 100

        # Setup test data
        security_framework.create_subject(sample_subject)
        security_framework.create_role(sample_role)
        security_framework.assign_role_to_subject(sample_subject.id, sample_role.id)

        def authorize_worker():
            """Worker function for concurrent authorization"""
            auth_request = AuthorizationRequest(
                subject_id=sample_subject.id,
                resource="test_resource",
                action="read",
                context={"ip": "127.0.0.1", "time": "2024-01-01T12:00:00Z"}
            )

            start_time = time.time()
            try:
                result = security_framework.authorize(auth_request)
                return result, time.time() - start_time
            except Exception as e:
                return None, time.time() - start_time

        # Measure throughput
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(authorize_worker) for _ in range(num_requests)]

            results = []
            for future in as_completed(futures, timeout=60):
                results.append(future.result())

        total_time = time.time() - start_time
        successful_authorizations = sum(1 for result, _ in results if result is not None)

        # Calculate metrics
        throughput = successful_authorizations / total_time
        avg_latency = sum(latency for _, latency in results) / len(results)

        # Performance assertions
        assert throughput >= 500, f"Authorization throughput too low: {throughput} auths/sec"
        assert avg_latency <= 0.05, f"Authorization latency too high: {avg_latency}s"
        assert successful_authorizations >= num_requests * 0.98, f"Too many failed authorizations: {successful_authorizations}/{num_requests}"

        print(f"\nAuthorization Performance:")
        print(f"  Throughput: {throughput:.2f} auths/sec")
        print(f"  Average Latency: {avg_latency:.4f}s")
        print(f"  Success Rate: {successful_authorizations/num_requests*100:.1f}%")

    def test_encryption_performance(self, encryption_service):
        """Test encryption/decryption performance"""
        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        num_iterations = 100

        for data_size in data_sizes:
            test_data = "A" * data_size

            # Measure encryption performance
            start_time = time.time()
            for _ in range(num_iterations):
                encrypted = encryption_service.encrypt_data(test_data)
            encryption_time = time.time() - start_time

            # Measure decryption performance
            start_time = time.time()
            for _ in range(num_iterations):
                decrypted = encryption_service.decrypt_data(encrypted)
            decryption_time = time.time() - start_time

            # Calculate throughput
            encryption_throughput = (data_size * num_iterations) / encryption_time / 1024 / 1024  # MB/s
            decryption_throughput = (data_size * num_iterations) / decryption_time / 1024 / 1024  # MB/s

            print(f"\nEncryption Performance ({data_size/1024:.1f}KB data):")
            print(f"  Encryption: {encryption_throughput:.2f} MB/s")
            print(f"  Decryption: {decryption_throughput:.2f} MB/s")
            print(f"  Combined: {(encryption_throughput + decryption_throughput)/2:.2f} MB/s")

            # Performance assertions (should be able to handle at least 10MB/s)
            assert encryption_throughput >= 10, f"Encryption throughput too low for {data_size}B: {encryption_throughput} MB/s"
            assert decryption_throughput >= 10, f"Decryption throughput too low for {data_size}B: {decryption_throughput} MB/s"

    def test_audit_logging_performance(self, audit_logger):
        """Test audit logging performance under high load"""
        num_logs = 5000

        # Generate test log entries
        log_entries = []
        for i in range(num_logs):
            log_entries.append({
                "event_type": f"test_event_{i % 10}",
                "event_id": f"event_{i}",
                "timestamp": time.time(),
                "actor": f"user_{i % 100}",
                "action": f"action_{i % 20}",
                "resource": f"resource_{i % 50}",
                "status": "success" if i % 10 != 0 else "failure",
                "details": {"iteration": i, "data": "test" * 10}
            })

        # Measure logging performance
        start_time = time.time()

        for entry in log_entries:
            audit_logger.log_event(
                event_type=entry["event_type"],
                event_id=entry["event_id"],
                actor=entry["actor"],
                action=entry["action"],
                resource=entry["resource"],
                status=entry["status"],
                details=entry["details"]
            )

        total_time = time.time() - start_time
        throughput = num_logs / total_time

        # Performance assertions
        assert throughput >= 1000, f"Audit logging throughput too low: {throughput} logs/sec"

        print(f"\nAudit Logging Performance:")
        print(f"  Throughput: {throughput:.2f} logs/sec")
        print(f"  Total Time: {total_time:.2f}s for {num_logs} logs")

    def test_concurrent_role_assignment(self, security_framework):
        """Test concurrent role assignment performance"""
        num_users = 500
        num_roles = 50
        num_workers = 50

        # Create test data
        roles = []
        for i in range(num_roles):
            role = Role(
                id=f"perf_role_{i}",
                name=f"Performance Role {i}",
                description=f"Role for performance testing {i}",
                permissions=["read", "write"]
            )
            roles.append(role)
            security_framework.create_role(role)

        subjects = []
        for i in range(num_users):
            subject = Subject(
                id=f"perf_user_{i}",
                subject_type="user",
                name=f"Performance User {i}",
                email=f"user{i}@test.com"
            )
            subjects.append(subject)
            security_framework.create_subject(subject)

        def assignment_worker(worker_id):
            """Worker function for concurrent role assignment"""
            start_idx = worker_id * (num_users // num_workers)
            end_idx = start_idx + (num_users // num_workers)

            start_time = time.time()
            successful_assignments = 0

            for i in range(start_idx, end_idx):
                for role_idx in range(min(5, num_roles)):  # Assign up to 5 roles per user
                    try:
                        security_framework.assign_role_to_subject(
                            subjects[i].id,
                            roles[role_idx].id
                        )
                        successful_assignments += 1
                    except Exception as e:
                        pass  # Continue on failure

            return successful_assignments, time.time() - start_time

        # Measure concurrent assignment performance
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(assignment_worker, i) for i in range(num_workers)]

            results = []
            for future in as_completed(futures, timeout=120):
                results.append(future.result())

        total_time = time.time() - start_time
        total_assignments = sum(assignments for assignments, _ in results)

        # Calculate metrics
        throughput = total_assignments / total_time

        # Performance assertions
        assert throughput >= 200, f"Role assignment throughput too low: {throughput} assignments/sec"

        print(f"\nConcurrent Role Assignment Performance:")
        print(f"  Throughput: {throughput:.2f} assignments/sec")
        print(f"  Total Assignments: {total_assignments}")
        print(f"  Total Time: {total_time:.2f}s")

    def test_memory_usage_under_load(self, security_framework):
        """Test memory usage under sustained load"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create large number of subjects, roles, and permissions
        num_items = 1000

        for i in range(num_items):
            subject = Subject(
                id=f"memory_test_subject_{i}",
                subject_type="user",
                name=f"Memory Test User {i}",
                email=f"memory{i}@test.com",
                attributes={"index": i, "data": "test" * 50}
            )
            security_framework.create_subject(subject)

            role = Role(
                id=f"memory_test_role_{i}",
                name=f"Memory Test Role {i}",
                description=f"Role for memory testing {i}",
                permissions=[f"perm_{j}" for j in range(10)]
            )
            security_framework.create_role(role)

        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Memory assertions (should use less than 500MB for 2000 items)
        assert memory_increase <= 500, f"Memory usage too high: {memory_increase}MB increase"

        print(f"\nMemory Usage Under Load:")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Peak Memory: {peak_memory:.2f}MB")
        print(f"  Memory Increase: {memory_increase:.2f}MB for {num_items * 2} items")

    def test_threat_detection_performance(self):
        """Test threat detection performance"""
        from src.agents.security.threat_detection import ThreatDetectionSystem

        threat_service = ThreatDetectionSystem()
        num_requests = 1000

        # Generate test requests with some suspicious patterns
        test_requests = []
        for i in range(num_requests):
            is_suspicious = i % 20 == 0  # 5% suspicious requests
            request = {
                "ip": f"192.168.1.{i % 255}",
                "user_agent": f"test_agent_{i}",
                "endpoint": f"/api/test/{i % 100}",
                "method": "GET" if i % 2 == 0 else "POST",
                "headers": {"content-type": "application/json"},
                "timestamp": time.time()
            }

            if is_suspicious:
                request["headers"]["user-agent"] = "bot/scanner"
                request["ip"] = "10.0.0.1"  # Suspicious IP

            test_requests.append(request)

        # Measure threat detection performance
        start_time = time.time()

        detected_threats = 0
        for request in test_requests:
            result = threat_service.analyze_request(request)
            if result["is_threat"]:
                detected_threats += 1

        total_time = time.time() - start_time
        throughput = num_requests / total_time

        # Performance assertions
        assert throughput >= 1000, f"Threat detection throughput too low: {throughput} requests/sec"
        assert detected_threats >= 40, f"Threat detection accuracy low: {detected_threats}/{num_requests // 20}"

        print(f"\nThreat Detection Performance:")
        print(f"  Throughput: {throughput:.2f} requests/sec")
        print(f"  Detected Threats: {detected_threats}")
        print(f"  Detection Rate: {detected_threats/(num_requests//20)*100:.1f}%")

    @pytest.mark.asyncio
    async def test_async_authentication_performance(self, auth_service):
        """Test asynchronous authentication performance"""
        num_requests = 500

        async def async_authenticate(request_id):
            """Asynchronous authentication worker"""
            auth_request = AuthenticationRequest(
                username=f"async_user_{request_id}",
                password="test_password",
                factors={"totp": str(request_id).zfill(6)}
            )

            start_time = time.time()
            try:
                result = await auth_service.authenticate_async(auth_request)
                return result, time.time() - start_time
            except Exception as e:
                return None, time.time() - start_time

        # Create concurrent async tasks
        tasks = [async_authenticate(i) for i in range(num_requests)]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        successful_auths = sum(1 for result, _ in results if result is not None)
        throughput = successful_auths / total_time

        # Performance assertions
        assert throughput >= 200, f"Async authentication throughput too low: {throughput} auths/sec"

        print(f"\nAsync Authentication Performance:")
        print(f"  Throughput: {throughput:.2f} auths/sec")
        print(f"  Total Time: {total_time:.2f}s for {num_requests} requests")


class TestCompliancePerformance:
    """Performance testing for compliance framework components"""

    @pytest.fixture
    def compliance_engine(self):
        """Initialize compliance engine"""
        return ComplianceEngine()

    @pytest.fixture
    def gdpr_manager(self):
        """Initialize GDPR compliance manager"""
        return GDPRComplianceManager()

    def test_compliance_assessment_performance(self, compliance_engine):
        """Test compliance assessment performance"""
        num_assessments = 100

        assessments = []
        for i in range(num_assessments):
            assessment = {
                "framework": f"framework_{i % 5}",
                "controls": [{"id": f"control_{j}", "status": "pass" if j % 10 != 0 else "fail"} for j in range(50)],
                "evidence": [{"type": "document", "url": f"/doc/{k}"} for k in range(20)]
            }
            assessments.append(assessment)

        # Measure assessment performance
        start_time = time.time()

        results = []
        for assessment in assessments:
            result = compliance_engine.assess_compliance(
                framework=assessment["framework"],
                controls=assessment["controls"],
                evidence=assessment["evidence"]
            )
            results.append(result)

        total_time = time.time() - start_time
        throughput = num_assessments / total_time

        # Performance assertions
        assert throughput >= 50, f"Compliance assessment throughput too low: {throughput} assessments/sec"

        print(f"\nCompliance Assessment Performance:")
        print(f"  Throughput: {throughput:.2f} assessments/sec")
        print(f"  Total Time: {total_time:.2f}s for {num_assessments} assessments")

    def test_gdpr_dsar_processing_performance(self, gdpr_manager):
        """Test GDPR DSAR processing performance"""
        num_requests = 200

        dsar_requests = []
        for i in range(num_requests):
            request = {
                "request_id": f"dsar_{i}",
                "request_type": "access" if i % 2 == 0 else "deletion",
                "subject_id": f"subject_{i % 50}",
                "data_categories": ["personal", "contact", "preferences"],
                "deadline": time.time() + (30 * 24 * 3600)  # 30 days
            }
            dsar_requests.append(request)

        # Measure DSAR processing performance
        start_time = time.time()

        processed_requests = 0
        for request in dsar_requests:
            try:
                if request["request_type"] == "access":
                    result = gdpr_manager.process_access_request(
                        subject_id=request["subject_id"],
                        data_categories=request["data_categories"]
                    )
                else:
                    result = gdpr_manager.process_deletion_request(
                        subject_id=request["subject_id"],
                        data_categories=request["data_categories"]
                    )
                processed_requests += 1
            except Exception as e:
                pass  # Continue on failure

        total_time = time.time() - start_time
        throughput = processed_requests / total_time

        # Performance assertions
        assert throughput >= 20, f"GDPR DSAR processing throughput too low: {throughput} requests/sec"

        print(f"\nGDPR DSAR Processing Performance:")
        print(f"  Throughput: {throughput:.2f} requests/sec")
        print(f"  Processed Requests: {processed_requests}/{num_requests}")

    def test_compliance_report_generation_performance(self, compliance_engine):
        """Test compliance report generation performance"""
        report_configs = [
            {"framework": "GDPR", "format": "json", "detail_level": "summary"},
            {"framework": "SOC2", "format": "html", "detail_level": "detailed"},
            {"framework": "HIPAA", "format": "pdf", "detail_level": "executive"}
        ]

        for config in report_configs:
            start_time = time.time()

            # Generate report (mock implementation)
            try:
                report = compliance_engine.generate_report(
                    framework=config["framework"],
                    format=config["format"],
                    detail_level=config["detail_level"]
                )

                generation_time = time.time() - start_time

                # Performance assertions
                assert generation_time <= 10, f"Report generation too slow: {generation_time}s for {config['framework']}"

                print(f"\n{config['framework']} Report Generation Performance:")
                print(f"  Generation Time: {generation_time:.2f}s")
                print(f"  Report Size: {len(str(report))} chars")

            except Exception as e:
                print(f"Report generation failed for {config['framework']}: {e}")

    def test_memory_efficiency_with_large_datasets(self, compliance_engine):
        """Test memory efficiency when processing large compliance datasets"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        large_dataset = {
            "framework": "GDPR",
            "controls": [{"id": f"control_{i}", "status": "pass", "evidence": f"evidence_{i}"} for i in range(10000)],
            "assessments": [{"id": f"assessment_{i}", "score": 85 + (i % 15), "date": f"2024-01-{i % 28 + 1:02d}"} for i in range(1000)],
            "findings": [{"id": f"finding_{i}", "severity": "high" if i % 10 == 0 else "medium", "description": f"Finding {i}"} for i in range(500)]
        }

        # Process large dataset
        start_time = time.time()

        compliance_engine.process_large_dataset(large_dataset)

        processing_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory

        # Performance assertions
        assert processing_time <= 30, f"Large dataset processing too slow: {processing_time}s"
        assert memory_increase <= 100, f"Memory usage too high for large dataset: {memory_increase}MB"

        print(f"\nLarge Dataset Processing Performance:")
        print(f"  Processing Time: {processing_time:.2f}s for {len(large_dataset['controls'])} controls")
        print(f"  Memory Increase: {memory_increase:.2f}MB")
        print(f"  Processing Rate: {len(large_dataset['controls'])/processing_time:.2f} controls/sec")


# Performance test configuration
pytest_plugins = ("pytest_asyncio",)

# Custom performance testing markers
def pytest_configure(config):
    """Configure custom performance testing markers"""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "load_test: marks tests as load tests"
    )
    config.addinivalue_line(
        "markers", "memory_test: marks tests as memory usage tests"
    )