#!/usr/bin/env python3
"""
Security and Compliance Tests for Agency Swarm System
Enterprise-grade security validation and compliance checking
"""

import asyncio
import aiohttp
import json
import logging
import hashlib
import ssl
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import jwt
import bcrypt
from cryptography.fernet import Fernet
import re

logger = logging.getLogger(__name__)

class SecurityComplianceTester:
    """Enterprise security and compliance testing system"""

    def __init__(self):
        self.services = {
            "frontend": "http://localhost:3737",
            "api": "http://localhost:8181",
            "mcp": "http://localhost:8051",
            "agents": "http://localhost:8052"
        }
        self.security_results = {}
        self.compliance_standards = {
            "owasp_top_10": True,
            "gdpr": True,
            "soc2": True,
            "hipaa": False,  # Not applicable unless handling health data
            "pci_dss": False  # Not applicable unless handling payment data
        }
        self.vulnerability_patterns = [
            r"sql injection|sqli",
            r"xss|cross.*site.*scripting",
            r"csrf|cross.*site.*request.*forgery",
            r"buffer.*overflow",
            r"remote.*code.*execution|rce",
            r"directory.*traversal",
            r"authentication.*bypass",
            r"authorization.*bypass"
        ]

    async def test_authentication_security(self):
        """Test authentication security measures"""
        logger.info("Testing authentication security...")

        security_tests = []

        # Test password policy
        password_tests = [
            ("weak_password", "123456", False),
            ("short_password", "short", False),
            ("no_special_chars", "password123", False),
            ("strong_password", "Str0ngP@ssw0rd!", True),
            ("very_strong", "V3ry$tr0ng&P@ssw0rd!WithNumbers123", True)
        ]

        for test_name, password, should_pass in password_tests:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.services['api']}/api/auth/validate_password", json={
                        "password": password
                    }) as response:
                        result = await response.json()
                        is_valid = result.get("valid", False)

                        test_result = {
                            "test_name": f"Password Policy - {test_name}",
                            "status": "passed" if is_valid == should_pass else "failed",
                            "expected": should_pass,
                            "actual": is_valid,
                            "password_strength": result.get("strength_score", 0)
                        }
                        security_tests.append(test_result)
            except Exception as e:
                security_tests.append({
                    "test_name": f"Password Policy - {test_name}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test JWT token security
        jwt_tests = []
        try:
            # Test token expiration
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.services['api']}/api/auth/login", json={
                    "username": "test_user",
                    "password": "test_password"
                }) as response:
                    login_result = await response.json()
                    token = login_result.get("token")

                    if token:
                        # Decode token to check expiration
                        decoded = jwt.decode(token, options={"verify_signature": False})
                        expiration_time = decoded.get("exp", 0)
                        current_time = datetime.now().timestamp()

                        jwt_tests.append({
                            "test_name": "JWT Token Expiration",
                            "status": "passed" if expiration_time > current_time else "failed",
                            "expiration_time": expiration_time,
                            "current_time": current_time
                        })

                        # Test token tampering
                        try:
                            tampered_token = token[:-10] + "tampered"
                            async with session.get(f"{self.services['api']}/api/protected", headers={
                                "Authorization": f"Bearer {tampered_token}"
                            }) as response:
                                jwt_tests.append({
                                    "test_name": "JWT Token Tampering Protection",
                                    "status": "passed" if response.status == 401 else "failed",
                                    "response_status": response.status
                                })
                        except Exception as e:
                            jwt_tests.append({
                                "test_name": "JWT Token Tampering Protection",
                                "status": "failed",
                                "error": str(e)
                            })
        except Exception as e:
            jwt_tests.append({
                "test_name": "JWT Token Security",
                "status": "failed",
                "error": str(e)
            })

        # Test session management
        session_tests = []
        try:
            async with aiohttp.ClientSession() as session:
                # Test session timeout
                async with session.post(f"{self.services['api']}/api/auth/login", json={
                    "username": "test_user",
                    "password": "test_password"
                }) as response:
                    if response.status == 200:
                        session_tests.append({
                            "test_name": "Session Creation",
                            "status": "passed"
                        })

                # Test session invalidation
                async with session.post(f"{self.services['api']}/api/auth/logout") as response:
                    session_tests.append({
                        "test_name": "Session Invalidation",
                        "status": "passed" if response.status == 200 else "failed"
                    })
        except Exception as e:
            session_tests.append({
                "test_name": "Session Management",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Authentication Security",
            "password_tests": security_tests,
            "jwt_tests": jwt_tests,
            "session_tests": session_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in security_tests + jwt_tests + session_tests
            ) else "failed"
        }

    async def test_authorization_security(self):
        """Test authorization and access control"""
        logger.info("Testing authorization security...")

        auth_tests = []

        # Test role-based access control
        roles = ["admin", "user", "guest", "agent"]
        protected_endpoints = [
            ("GET", "/api/admin/users"),
            ("POST", "/api/admin/config"),
            ("GET", "/api/users/profile"),
            ("POST", "/api/agents/execute")
        ]

        for role in roles:
            for method, endpoint in protected_endpoints:
                try:
                    # Get token for role
                    async with aiohttp.ClientSession() as session:
                        async with session.post(f"{self.services['api']}/api/auth/login", json={
                            "username": f"{role}_user",
                            "password": "test_password"
                        }) as response:
                            login_result = await response.json()
                            token = login_result.get("token")

                            if token:
                                # Test access to protected endpoint
                                headers = {"Authorization": f"Bearer {token}"}
                                if method == "GET":
                                    async with session.get(f"{self.services['api']}{endpoint}", headers=headers) as response:
                                        status = response.status
                                else:
                                    async with session.post(f"{self.services['api']}{endpoint}", headers=headers, json={}) as response:
                                        status = response.status

                                # Expected status based on role and endpoint
                                expected_status = self.get_expected_access_status(role, endpoint)
                                auth_tests.append({
                                    "test_name": f"RBAC - {role} access to {endpoint}",
                                    "status": "passed" if status == expected_status else "failed",
                                    "expected_status": expected_status,
                                    "actual_status": status
                                })
                except Exception as e:
                    auth_tests.append({
                        "test_name": f"RBAC - {role} access to {endpoint}",
                        "status": "failed",
                        "error": str(e)
                    })

        # Test privilege escalation
        escalation_tests = []
        try:
            async with aiohttp.ClientSession() as session:
                # Login as regular user
                async with session.post(f"{self.services['api']}/api/auth/login", json={
                    "username": "regular_user",
                    "password": "test_password"
                }) as response:
                    login_result = await response.json()
                    user_token = login_result.get("token")

                    if user_token:
                        # Try to access admin endpoint with user token
                        headers = {"Authorization": f"Bearer {user_token}"}
                        async with session.get(f"{self.services['api']}/api/admin/users", headers=headers) as response:
                            escalation_tests.append({
                                "test_name": "Privilege Escalation Protection",
                                "status": "passed" if response.status == 403 else "failed",
                                "response_status": response.status
                            })
        except Exception as e:
            escalation_tests.append({
                "test_name": "Privilege Escalation Protection",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Authorization Security",
            "rbac_tests": auth_tests,
            "escalation_tests": escalation_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in auth_tests + escalation_tests
            ) else "failed"
        }

    def get_expected_access_status(self, role, endpoint):
        """Get expected access status based on role and endpoint"""
        if role == "admin":
            return 200
        elif role == "user":
            return 200 if "/users/profile" in endpoint else 403
        elif role == "agent":
            return 200 if "/api/agents/execute" in endpoint else 403
        else:  # guest
            return 403

    async def test_input_validation_security(self):
        """Test input validation and sanitization"""
        logger.info("Testing input validation security...")

        validation_tests = []

        # Test SQL injection
        sql_injection_payloads = [
            "SELECT * FROM users",
            "1' OR '1'='1",
            "1; DROP TABLE users; --",
            "1' UNION SELECT username, password FROM users--",
            "admin'--"
        ]

        for payload in sql_injection_payloads:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.services['api']}/api/knowledge/search", json={
                        "query": payload
                    }) as response:
                        result = await response.json()

                        # Check if SQL injection was successful (should return error or empty results)
                        is_vulnerable = "error" not in result and len(result.get("results", [])) > 0

                        validation_tests.append({
                            "test_name": f"SQL Injection - {payload[:20]}...",
                            "status": "passed" if not is_vulnerable else "failed",
                            "vulnerable": is_vulnerable,
                            "response_status": response.status
                        })
            except Exception as e:
                validation_tests.append({
                    "test_name": f"SQL Injection - {payload[:20]}...",
                    "status": "failed",
                    "error": str(e)
                })

        # Test XSS attacks
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83));//"
        ]

        for payload in xss_payloads:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.services['api']}/api/knowledge/upload", json={
                        "title": payload,
                        "content": "Test content"
                    }) as response:
                        result = await response.json()

                        # Check if XSS payload was sanitized
                        is_sanitized = payload not in result.get("title", "")

                        validation_tests.append({
                            "test_name": f"XSS Protection - {payload[:20]}...",
                            "status": "passed" if is_sanitized else "failed",
                            "sanitized": is_sanitized,
                            "response_status": response.status
                        })
            except Exception as e:
                validation_tests.append({
                    "test_name": f"XSS Protection - {payload[:20]}...",
                    "status": "failed",
                    "error": str(e)
                })

        # Test file upload security
        file_upload_tests = []
        malicious_files = [
            ("test.php", "<?php echo 'malicious'; ?>"),
            ("test.exe", b"fake executable content"),
            ("test.sh", "#!/bin/bash\necho 'malicious'"),
            ("../../etc/passwd", "attempted directory traversal")
        ]

        for filename, content in malicious_files:
            try:
                async with aiohttp.ClientSession() as session:
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', content, filename=filename)

                    async with session.post(f"{self.services['api']}/api/upload", data=form_data) as response:
                        result = await response.json()

                        # Check if malicious file was rejected
                        is_rejected = response.status != 200 or "error" in result

                        file_upload_tests.append({
                            "test_name": f"File Upload Security - {filename}",
                            "status": "passed" if is_rejected else "failed",
                            "rejected": is_rejected,
                            "response_status": response.status
                        })
            except Exception as e:
                file_upload_tests.append({
                    "test_name": f"File Upload Security - {filename}",
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "test_name": "Input Validation Security",
            "sql_injection_tests": validation_tests[:len(sql_injection_payloads)],
            "xss_tests": validation_tests[len(sql_injection_payloads):],
            "file_upload_tests": file_upload_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in validation_tests + file_upload_tests
            ) else "failed"
        }

    async def test_network_security(self):
        """Test network security and encryption"""
        logger.info("Testing network security...")

        network_tests = []

        # Test SSL/TLS configuration
        try:
            # Check if services use HTTPS in production
            for service_name, service_url in self.services.items():
                if service_url.startswith("https://"):
                    # Test SSL configuration
                    import ssl
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE

                    with ssl.wrap_socket(ssl.SSLSocket(), context=context) as s:
                        try:
                            s.connect((service_url.replace("https://", "").split(":")[0], 443))
                            cert = s.getpeercert()

                            network_tests.append({
                                "test_name": f"SSL Configuration - {service_name}",
                                "status": "passed",
                                "certificate_valid": True,
                                "certificate_expiry": cert.get("notAfter")
                            })
                        except Exception as e:
                            network_tests.append({
                                "test_name": f"SSL Configuration - {service_name}",
                                "status": "failed",
                                "error": str(e)
                            })
                else:
                    network_tests.append({
                        "test_name": f"SSL Configuration - {service_name}",
                        "status": "warning",
                        "message": "Service not using HTTPS (development mode)"
                    })
        except Exception as e:
            network_tests.append({
                "test_name": "SSL Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy"
        ]

        header_tests = []
        for service_name, service_url in self.services.items():
            try:
                response = requests.get(service_url, timeout=5)
                headers = response.headers

                missing_headers = []
                for header in security_headers:
                    if header not in headers:
                        missing_headers.append(header)

                header_tests.append({
                    "test_name": f"Security Headers - {service_name}",
                    "status": "passed" if not missing_headers else "failed",
                    "missing_headers": missing_headers,
                    "present_headers": [h for h in security_headers if h in headers]
                })
            except Exception as e:
                header_tests.append({
                    "test_name": f"Security Headers - {service_name}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test CORS configuration
        cors_tests = []
        for service_name, service_url in self.services.items():
            try:
                response = requests.options(service_url, timeout=5)
                cors_headers = {
                    "Access-Control-Allow-Origin": response.headers.get("Access-Control-Allow-Origin"),
                    "Access-Control-Allow-Methods": response.headers.get("Access-Control-Allow-Methods"),
                    "Access-Control-Allow-Headers": response.headers.get("Access-Control-Allow-Headers")
                }

                # Check if CORS is properly configured
                cors_secure = (
                    cors_headers["Access-Control-Allow-Origin"] != "*" or
                    service_url.startswith("http://localhost")
                )

                cors_tests.append({
                    "test_name": f"CORS Configuration - {service_name}",
                    "status": "passed" if cors_secure else "failed",
                    "cors_headers": cors_headers,
                    "is_secure": cors_secure
                })
            except Exception as e:
                cors_tests.append({
                    "test_name": f"CORS Configuration - {service_name}",
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "test_name": "Network Security",
            "ssl_tests": network_tests,
            "header_tests": header_tests,
            "cors_tests": cors_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in network_tests + header_tests + cors_tests
            ) else "failed"
        }

    async def test_data_protection_security(self):
        """Test data protection and privacy measures"""
        logger.info("Testing data protection security...")

        data_tests = []

        # Test data encryption
        try:
            # Test encryption of sensitive data
            sensitive_data = {
                "password": "sensitive_password",
                "api_key": "secret_api_key_123",
                "personal_info": "personal@example.com"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.services['api']}/api/test/encryption", json=sensitive_data) as response:
                    result = await response.json()

                    # Check if sensitive data is encrypted in response
                    data_encrypted = all(
                        key not in str(result) or "encrypted" in str(result).lower()
                        for key in sensitive_data.keys()
                    )

                    data_tests.append({
                        "test_name": "Data Encryption",
                        "status": "passed" if data_encrypted else "failed",
                        "data_encrypted": data_encrypted
                    })
        except Exception as e:
            data_tests.append({
                "test_name": "Data Encryption",
                "status": "failed",
                "error": str(e)
            })

        # Test data masking
        try:
            # Test PII masking in logs/responses
            pii_data = {
                "email": "user@example.com",
                "phone": "+1234567890",
                "ssn": "123-45-6789",
                "credit_card": "4111111111111111"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.services['api']}/api/test/pii", json=pii_data) as response:
                    result = await response.json()

                    # Check if PII is masked
                    pii_masked = all(
                        self.is_pii_masked(str(result), value)
                        for value in pii_data.values()
                    )

                    data_tests.append({
                        "test_name": "PII Data Masking",
                        "status": "passed" if pii_masked else "failed",
                        "pii_masked": pii_masked
                    })
        except Exception as e:
            data_tests.append({
                "test_name": "PII Data Masking",
                "status": "failed",
                "error": str(e)
            })

        # Test data retention policies
        try:
            # Test automatic data deletion
            async with aiohttp.ClientSession() as session:
                # Create test data with expiration
                await session.post(f"{self.services['api']}/api/test/expiring_data", json={
                    "data": "test_data",
                    "ttl": 1  # 1 second
                })

                # Wait for expiration
                await asyncio.sleep(2)

                # Try to retrieve expired data
                async with session.get(f"{self.services['api']}/api/test/expiring_data") as response:
                    result = await response.json()

                    data_deleted = len(result.get("data", [])) == 0

                    data_tests.append({
                        "test_name": "Data Retention Policy",
                        "status": "passed" if data_deleted else "failed",
                        "data_deleted": data_deleted
                    })
        except Exception as e:
            data_tests.append({
                "test_name": "Data Retention Policy",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Data Protection Security",
            "tests": data_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in data_tests
            ) else "failed"
        }

    def is_pii_masked(self, text, original_value):
        """Check if PII value is properly masked"""
        return original_value not in text or any(mask in text for mask in ["***", "MASKED", "REDACTED"])

    async def test_audit_logging_security(self):
        """Test audit logging and monitoring"""
        logger.info("Testing audit logging security...")

        audit_tests = []

        # Test security event logging
        security_events = [
            "login_attempt",
            "permission_denied",
            "data_access",
            "configuration_change",
            "suspicious_activity"
        ]

        for event in security_events:
            try:
                async with aiohttp.ClientSession() as session:
                    # Trigger security event
                    await session.post(f"{self.services['api']}/api/test/security_event", json={
                        "event_type": event,
                        "details": {"test": True}
                    })

                    # Check if event was logged
                    async with session.get(f"{self.services['api']}/api/test/audit_logs") as response:
                        logs = await response.json()
                        event_logged = any(
                            log.get("event_type") == event
                            for log in logs.get("logs", [])
                        )

                        audit_tests.append({
                            "test_name": f"Audit Logging - {event}",
                            "status": "passed" if event_logged else "failed",
                            "event_logged": event_logged
                        })
            except Exception as e:
                audit_tests.append({
                    "test_name": f"Audit Logging - {event}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test log integrity
        try:
            async with aiohttp.ClientSession() as session:
                # Get log hash
                async with session.get(f"{self.services['api']}/api/test/log_hash") as response:
                    hash_result = await response.json()
                    original_hash = hash_result.get("hash")

                    # Modify logs (should be prevented)
                    await session.post(f"{self.services['api']}/api/test/tamper_logs", json={
                        "modification": "unauthorized_change"
                    })

                    # Get new hash
                    async with session.get(f"{self.services['api']}/api/test/log_hash") as response:
                        new_hash_result = await response.json()
                        new_hash = new_hash_result.get("hash")

                    # Check if logs are protected
                    logs_protected = original_hash == new_hash

                    audit_tests.append({
                        "test_name": "Log Integrity Protection",
                        "status": "passed" if logs_protected else "failed",
                        "logs_protected": logs_protected
                    })
        except Exception as e:
            audit_tests.append({
                "test_name": "Log Integrity Protection",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Audit Logging Security",
            "tests": audit_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in audit_tests
            ) else "failed"
        }

    async def test_compliance_standards(self):
        """Test compliance with various standards"""
        logger.info("Testing compliance standards...")

        compliance_tests = []

        # Test GDPR compliance
        gdpr_tests = []
        gdpr_requirements = [
            ("Data Consent", "consent_management"),
            ("Data Access", "data_access_requests"),
            ("Data Portability", "data_export"),
            ("Right to be Forgotten", "data_deletion"),
            ("Data Breach Notification", "breach_notification")
        ]

        for requirement, endpoint in gdpr_requirements:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.services['api']}/api/gdpr/{endpoint}") as response:
                        result = await response.json()
                        compliant = result.get("compliant", False)

                        gdpr_tests.append({
                            "requirement": requirement,
                            "status": "passed" if compliant else "failed",
                            "compliant": compliant
                        })
            except Exception as e:
                gdpr_tests.append({
                    "requirement": requirement,
                    "status": "failed",
                    "error": str(e)
                })

        # Test SOC2 compliance
        soc2_tests = []
        soc2_trust_services = [
            ("Security", "security_controls"),
            ("Availability", "availability_monitoring"),
            ("Confidentiality", "data_classification"),
            ("Privacy", "privacy_controls")
        ]

        for service, endpoint in soc2_trust_services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.services['api']}/api/soc2/{endpoint}") as response:
                        result = await response.json()
                        implemented = result.get("implemented", False)

                        soc2_tests.append({
                            "trust_service": service,
                            "status": "passed" if implemented else "failed",
                            "implemented": implemented
                        })
            except Exception as e:
                soc2_tests.append({
                    "trust_service": service,
                    "status": "failed",
                    "error": str(e)
                })

        compliance_tests.append({
            "standard": "GDPR",
            "tests": gdpr_tests,
            "overall_status": "passed" if all(test["status"] == "passed" for test in gdpr_tests) else "failed"
        })

        compliance_tests.append({
            "standard": "SOC2",
            "tests": soc2_tests,
            "overall_status": "passed" if all(test["status"] == "passed" for test in soc2_tests) else "failed"
        })

        return {
            "test_name": "Compliance Standards",
            "compliance_tests": compliance_tests,
            "overall_status": "passed" if all(
                test["overall_status"] == "passed" for test in compliance_tests
            ) else "failed"
        }

    async def run_complete_security_test(self):
        """Run complete security and compliance test suite"""
        logger.info("Starting complete security and compliance test...")

        # Run all security tests
        test_functions = [
            self.test_authentication_security,
            self.test_authorization_security,
            self.test_input_validation_security,
            self.test_network_security,
            self.test_data_protection_security,
            self.test_audit_logging_security,
            self.test_compliance_standards
        ]

        for test_func in test_functions:
            try:
                result = await test_func()
                self.security_results[result["test_name"]] = result
                logger.info(f"✓ {test_func.__name__}: {result['overall_status']}")
            except Exception as e:
                logger.error(f"✗ {test_func.__name__} failed: {e}")

        # Generate final report
        return self.generate_security_report()

    def generate_security_report(self):
        """Generate comprehensive security report"""
        timestamp = datetime.now().isoformat()

        # Calculate overall security score
        total_tests = len(self.security_results)
        passed_tests = sum(1 for result in self.security_results.values() if result["overall_status"] == "passed")
        security_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        report = {
            "test_suite": "Agency Swarm Security & Compliance Test",
            "timestamp": timestamp,
            "overall_assessment": {
                "security_score": security_score,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "security_grade": self.calculate_security_grade(security_score),
                "ready_for_production": security_score >= 80
            },
            "compliance_standards": self.compliance_standards,
            "detailed_results": self.security_results,
            "vulnerability_summary": self.generate_vulnerability_summary(),
            "recommendations": self.generate_security_recommendations()
        }

        # Save report
        report_path = Path("agency_swarm_security_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Security report saved to {report_path}")
        return report

    def calculate_security_grade(self, score):
        """Calculate security grade based on score"""
        if score >= 95:
            return "A+ (Excellent)"
        elif score >= 85:
            return "A (Good)"
        elif score >= 75:
            return "B (Satisfactory)"
        elif score >= 65:
            return "C (Needs Improvement)"
        elif score >= 50:
            return "D (Poor)"
        else:
            return "F (Critical)"

    def generate_vulnerability_summary(self):
        """Generate vulnerability summary"""
        vulnerabilities = []
        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0

        for test_name, result in self.security_results.items():
            if result["overall_status"] == "failed":
                # Analyze failed tests to determine severity
                if "Authentication" in test_name or "Authorization" in test_name:
                    severity = "Critical"
                    critical_count += 1
                elif "Input Validation" in test_name:
                    severity = "High"
                    high_count += 1
                elif "Network Security" in test_name:
                    severity = "Medium"
                    medium_count += 1
                else:
                    severity = "Low"
                    low_count += 1

                vulnerabilities.append({
                    "category": test_name,
                    "severity": severity,
                    "status": "failed"
                })

        return {
            "total_vulnerabilities": len(vulnerabilities),
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "vulnerability_details": vulnerabilities
        }

    def generate_security_recommendations(self):
        """Generate security recommendations"""
        recommendations = []

        for test_name, result in self.security_results.items():
            if result["overall_status"] == "failed":
                if "Authentication" in test_name:
                    recommendations.append({
                        "priority": "Critical",
                        "category": "Authentication",
                        "issue": "Authentication security measures failed",
                        "recommendation": "Implement strong password policies, multi-factor authentication, and secure session management"
                    })
                elif "Authorization" in test_name:
                    recommendations.append({
                        "priority": "Critical",
                        "category": "Authorization",
                        "issue": "Authorization controls failed",
                        "recommendation": "Implement proper role-based access control and privilege escalation protection"
                    })
                elif "Input Validation" in test_name:
                    recommendations.append({
                        "priority": "High",
                        "category": "Input Validation",
                        "issue": "Input validation vulnerabilities detected",
                        "recommendation": "Implement proper input sanitization, parameterized queries, and file upload security"
                    })
                elif "Network Security" in test_name:
                    recommendations.append({
                        "priority": "Medium",
                        "category": "Network Security",
                        "issue": "Network security issues detected",
                        "recommendation": "Configure SSL/TLS, security headers, and CORS properly"
                    })

        return recommendations

async def main():
    """Main function to run security tests"""
    tester = SecurityComplianceTester()
    return await tester.run_complete_security_test()

if __name__ == "__main__":
    asyncio.run(main())