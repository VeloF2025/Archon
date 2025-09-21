#!/usr/bin/env python3
"""
Agency Swarm Production Readiness Check
Comprehensive validation system for production deployment readiness
"""

import asyncio
import logging
import json
import subprocess
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import asyncpg
import redis.asyncio as redis
import yaml
import hashlib
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_readiness.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReadinessStatus(Enum):
    READY = "ready"
    WARNING = "warning"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"

class CheckCategory(Enum):
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"
    BACKUP = "backup"
    COMPLIANCE = "compliance"
    DEPLOYMENT = "deployment"

@dataclass
class ReadinessCheck:
    """Individual readiness check result"""
    name: str
    category: CheckCategory
    status: ReadinessStatus
    message: str
    details: Dict[str, Any] = None
    critical: bool = True
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.details is None:
            self.details = {}

@dataclass
class ProductionReadinessReport:
    """Complete production readiness report"""
    deployment_name: str
    environment: str
    overall_status: ReadinessStatus
    readiness_score: float  # 0-100
    checks: List[ReadinessCheck]
    critical_issues: List[str]
    recommendations: List[str]
    timestamp: datetime
    generated_by: str = "production_readiness_check.py"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class ProductionReadinessChecker:
    """Production readiness checker"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checks: List[ReadinessCheck] = []
        self.critical_issues: List[str] = []
        self.recommendations: List[str] = []

        # Initialize connections
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize readiness checker"""
        logger.info("Initializing Agency Swarm Production Readiness Checker...")

        # Initialize connections
        db_config = self.config.get('database')
        if db_config:
            try:
                self.db_pool = await asyncpg.create_pool(
                    db_config.get('url'),
                    min_size=1,
                    max_size=5
                )
            except Exception as e:
                logger.warning(f"Database connection failed: {e}")

        redis_config = self.config.get('redis')
        if redis_config:
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")

        logger.info("Production readiness checker initialized successfully")

    async def run_comprehensive_check(self) -> ProductionReadinessReport:
        """Run comprehensive production readiness check"""
        logger.info("Running comprehensive production readiness check...")

        # Clear previous results
        self.checks.clear()
        self.critical_issues.clear()
        self.recommendations.clear()

        # Run all readiness checks
        await self.check_infrastructure_readiness()
        await self.check_security_readiness()
        await self.check_performance_readiness()
        await self.check_monitoring_readiness()
        await self.check_backup_readiness()
        await self.check_compliance_readiness()
        await self.check_deployment_readiness()

        # Calculate overall readiness
        overall_status, readiness_score = self.calculate_overall_readiness()

        # Generate recommendations
        await self.generate_recommendations()

        return ProductionReadinessReport(
            deployment_name=self.config.get('deployment_name', 'Agency Swarm'),
            environment=self.config.get('environment', 'production'),
            overall_status=overall_status,
            readiness_score=readiness_score,
            checks=self.checks.copy(),
            critical_issues=self.critical_issues.copy(),
            recommendations=self.recommendations.copy(),
            timestamp=datetime.utcnow()
        )

    async def check_infrastructure_readiness(self):
        """Check infrastructure readiness"""
        logger.info("Checking infrastructure readiness...")

        # Kubernetes cluster health
        await self.check_kubernetes_cluster()

        # Resource availability
        await self.check_resource_availability()

        # Network connectivity
        await self.check_network_connectivity()

        # Storage capacity
        await self.check_storage_capacity()

        # Database connectivity
        await self.check_database_connectivity()

        # Cache connectivity
        await self.check_cache_connectivity()

    async def check_kubernetes_cluster(self):
        """Check Kubernetes cluster health"""
        try:
            # Check kubectl connection
            result = subprocess.run(
                ['kubectl', 'cluster-info'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Check node status
                nodes_result = subprocess.run(
                    ['kubectl', 'get', 'nodes', '-o', 'jsonpath={.items[*].status.conditions[?(@.type==\"Ready\")].status}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if 'True' in nodes_result.stdout:
                    # Check cluster version
                    version_result = subprocess.run(
                        ['kubectl', 'version', '--short'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    self.checks.append(ReadinessCheck(
                        name="Kubernetes Cluster Health",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.READY,
                        message="Kubernetes cluster is healthy and ready",
                        details={
                            'kubectl_output': result.stdout,
                            'nodes_ready': nodes_result.stdout.count('True'),
                            'version_info': version_result.stdout
                        },
                        critical=True
                    ))
                else:
                    self.checks.append(ReadinessCheck(
                        name="Kubernetes Cluster Health",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.NOT_READY,
                        message="Some Kubernetes nodes are not ready",
                        details={'nodes_status': nodes_result.stdout},
                        critical=True
                    ))
                    self.critical_issues.append("Kubernetes nodes not ready")
            else:
                self.checks.append(ReadinessCheck(
                    name="Kubernetes Cluster Health",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=ReadinessStatus.NOT_READY,
                    message=f"Kubernetes cluster connection failed: {result.stderr}",
                    details={'error': result.stderr},
                    critical=True
                ))
                self.critical_issues.append("Kubernetes cluster connection failed")

        except subprocess.TimeoutExpired:
            self.checks.append(ReadinessCheck(
                name="Kubernetes Cluster Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.NOT_READY,
                message="Kubernetes cluster check timed out",
                critical=True
            ))
            self.critical_issues.append("Kubernetes cluster check timed out")
        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Kubernetes Cluster Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.UNKNOWN,
                message=f"Kubernetes cluster check error: {str(e)}",
                critical=True
            ))

    async def check_resource_availability(self):
        """Check resource availability"""
        try:
            # Check available resources
            result = subprocess.run(
                ['kubectl', 'get', 'nodes', '-o', 'jsonpath={range .items[*]}{.status.capacity}{"\\n"}{end}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse resource information
                total_cpu = 0
                total_memory = 0

                for line in result.stdout.strip().split('\n'):
                    if line and 'cpu' in line and 'memory' in line:
                        # Simple parsing - in production you'd want more robust parsing
                        try:
                            if 'cpu:' in line:
                                cpu_part = line.split('cpu:')[1].split()[0]
                                total_cpu += int(cpu_part)
                            if 'memory:' in line:
                                memory_part = line.split('memory:')[1].split()[0]
                                total_memory += int(memory_part)
                        except (ValueError, IndexError):
                            pass

                # Check against requirements
                required_cpu = self.config.get('resource_requirements', {}).get('cpu', 16)
                required_memory = self.config.get('resource_requirements', {}).get('memory_gb', 32)

                if total_cpu >= required_cpu and total_memory >= required_memory * 1024:  # Convert GB to Mi
                    self.checks.append(ReadinessCheck(
                        name="Resource Availability",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.READY,
                        message="Sufficient resources available",
                        details={
                            'available_cpu_cores': total_cpu,
                            'available_memory_gb': total_memory / 1024,
                            'required_cpu_cores': required_cpu,
                            'required_memory_gb': required_memory
                        },
                        critical=True
                    ))
                else:
                    self.checks.append(ReadinessCheck(
                        name="Resource Availability",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.NOT_READY,
                        message="Insufficient resources available",
                        details={
                            'available_cpu_cores': total_cpu,
                            'available_memory_gb': total_memory / 1024,
                            'required_cpu_cores': required_cpu,
                            'required_memory_gb': required_memory
                        },
                        critical=True
                    ))
                    self.critical_issues.append("Insufficient cluster resources")
            else:
                self.checks.append(ReadinessCheck(
                    name="Resource Availability",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=ReadinessStatus.UNKNOWN,
                    message=f"Failed to check resource availability: {result.stderr}",
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Resource Availability",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.UNKNOWN,
                message=f"Resource availability check error: {str(e)}",
                critical=True
            ))

    async def check_network_connectivity(self):
        """Check network connectivity"""
        connectivity_checks = [
            ("Google DNS", "8.8.8.8", 53),
            ("Cloudflare DNS", "1.1.1.1", 53),
            ("GitHub", "github.com", 443),
            ("Container Registry", "ghcr.io", 443)
        ]

        successful_checks = 0
        total_checks = len(connectivity_checks)

        for name, host, port in connectivity_checks:
            try:
                # Use ping for ICMP connectivity
                if port == 53:  # DNS
                    result = subprocess.run(
                        ['ping', '-c', '1', '-W', '5', host],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    success = result.returncode == 0
                else:
                    # Use TCP connectivity check
                    result = subprocess.run(
                        ['nc', '-z', '-w', '5', host, str(port)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    success = result.returncode == 0

                if success:
                    successful_checks += 1

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        success_rate = successful_checks / total_checks

        if success_rate >= 0.75:  # 75% success rate
            status = ReadinessStatus.READY
            message = f"Network connectivity good ({successful_checks}/{total_checks})"
        elif success_rate >= 0.5:
            status = ReadinessStatus.WARNING
            message = f"Network connectivity degraded ({successful_checks}/{total_checks})"
        else:
            status = ReadinessStatus.NOT_READY
            message = f"Network connectivity poor ({successful_checks}/{total_checks})"
            self.critical_issues.append("Poor network connectivity")

        self.checks.append(ReadinessCheck(
            name="Network Connectivity",
            category=CheckCategory.INFRASTRUCTURE,
            status=status,
            message=message,
            details={
                'successful_checks': successful_checks,
                'total_checks': total_checks,
                'success_rate': success_rate
            },
            critical=True
        ))

    async def check_storage_capacity(self):
        """Check storage capacity"""
        try:
            # Check persistent volume claims
            result = subprocess.run(
                ['kubectl', 'get', 'pvc', '-o', 'jsonpath={range .items[*]}{.spec.resources.requests.storage}{"\\n"}{end}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                total_storage = 0
                for line in result.stdout.strip().split('\n'):
                    if line:
                        # Parse storage size (e.g., "100Gi")
                        try:
                            if 'Gi' in line:
                                size = int(line.split('Gi')[0])
                                total_storage += size
                            elif 'Mi' in line:
                                size = int(line.split('Mi')[0]) / 1024
                                total_storage += size
                        except ValueError:
                            pass

                required_storage = self.config.get('resource_requirements', {}).get('storage_gb', 100)

                if total_storage >= required_storage:
                    self.checks.append(ReadinessCheck(
                        name="Storage Capacity",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.READY,
                        message="Sufficient storage available",
                        details={
                            'available_storage_gb': total_storage,
                            'required_storage_gb': required_storage
                        },
                        critical=True
                    ))
                else:
                    self.checks.append(ReadinessCheck(
                        name="Storage Capacity",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=ReadinessStatus.NOT_READY,
                        message="Insufficient storage available",
                        details={
                            'available_storage_gb': total_storage,
                            'required_storage_gb': required_storage
                        },
                        critical=True
                    ))
                    self.critical_issues.append("Insufficient storage capacity")
            else:
                self.checks.append(ReadinessCheck(
                    name="Storage Capacity",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=ReadinessStatus.UNKNOWN,
                    message=f"Failed to check storage capacity: {result.stderr}",
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Storage Capacity",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.UNKNOWN,
                message=f"Storage capacity check error: {str(e)}",
                critical=True
            ))

    async def check_database_connectivity(self):
        """Check database connectivity"""
        if not self.db_pool:
            self.checks.append(ReadinessCheck(
                name="Database Connectivity",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.NOT_READY,
                message="Database pool not configured",
                critical=True
            ))
            self.critical_issues.append("Database not configured")
            return

        try:
            async with self.db_pool.acquire() as conn:
                # Test basic connection
                await conn.fetchval("SELECT 1")

                # Check database size and capacity
                size_result = await conn.fetch("""
                    SELECT pg_database_size(current_database()) as db_size,
                           pg_size_pretty(pg_database_size(current_database())) as db_size_pretty
                """)

                db_size = size_result[0]['db_size']
                max_db_size = self.config.get('database', {}).get('max_size_gb', 100) * 1024 * 1024 * 1024

                # Check connection limits
                conn_result = await conn.fetch("""
                    SELECT count(*) as current_connections,
                           setting as max_connections
                    FROM pg_settings
                    WHERE name = 'max_connections'
                """)

                current_connections = conn_result[0]['current_connections']
                max_connections = int(conn_result[0]['max_connections'])

                # Determine status
                connection_usage = current_connections / max_connections
                size_usage = db_size / max_db_size

                if connection_usage < 0.8 and size_usage < 0.8:
                    status = ReadinessStatus.READY
                    message = "Database connectivity and capacity healthy"
                elif connection_usage < 0.9 and size_usage < 0.9:
                    status = ReadinessStatus.WARNING
                    message = "Database resources approaching limits"
                else:
                    status = ReadinessStatus.NOT_READY
                    message = "Database resources critical"
                    self.critical_issues.append("Database capacity critical")

                self.checks.append(ReadinessCheck(
                    name="Database Connectivity",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=status,
                    message=message,
                    details={
                        'db_size_bytes': db_size,
                        'db_size_pretty': size_result[0]['db_size_pretty'],
                        'current_connections': current_connections,
                        'max_connections': max_connections,
                        'connection_usage_percent': connection_usage * 100,
                        'size_usage_percent': size_usage * 100
                    },
                    critical=True
                ))

        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Database Connectivity",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.NOT_READY,
                message=f"Database connection failed: {str(e)}",
                critical=True
            ))
            self.critical_issues.append("Database connection failed")

    async def check_cache_connectivity(self):
        """Check cache connectivity"""
        if not self.redis_client:
            self.checks.append(ReadinessCheck(
                name="Cache Connectivity",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.NOT_READY,
                message="Redis client not configured",
                critical=False
            ))
            return

        try:
            # Test basic connection
            ping_result = await self.redis_client.ping()

            # Get Redis info
            info = await self.redis_client.info()

            # Check memory usage
            used_memory = int(info.get('used_memory', 0))
            max_memory = int(info.get('maxmemory', 0))

            memory_usage = used_memory / max_memory if max_memory > 0 else 0

            # Check connection count
            connected_clients = int(info.get('connected_clients', 0))
            max_clients = int(info.get('maxclients', 10000))

            client_usage = connected_clients / max_clients

            if ping_result and memory_usage < 0.8 and client_usage < 0.8:
                status = ReadinessStatus.READY
                message = "Cache connectivity and capacity healthy"
            elif ping_result and memory_usage < 0.9 and client_usage < 0.9:
                status = ReadinessStatus.WARNING
                message = "Cache resources approaching limits"
            else:
                status = ReadinessStatus.NOT_READY
                message = "Cache resources critical"
                self.critical_issues.append("Cache capacity critical")

            self.checks.append(ReadinessCheck(
                name="Cache Connectivity",
                category=CheckCategory.INFRASTRUCTURE,
                status=status,
                message=message,
                details={
                    'ping_success': ping_result,
                    'used_memory_bytes': used_memory,
                    'max_memory_bytes': max_memory,
                    'memory_usage_percent': memory_usage * 100,
                    'connected_clients': connected_clients,
                    'max_clients': max_clients,
                    'client_usage_percent': client_usage * 100
                },
                critical=True
            ))

        except Exception as e:
            self.checks.append(ReadinessCheck(
                name="Cache Connectivity",
                category=CheckCategory.INFRASTRUCTURE,
                status=ReadinessStatus.NOT_READY,
                message=f"Cache connection failed: {str(e)}",
                critical=False
            ))

    async def check_security_readiness(self):
        """Check security readiness"""
        logger.info("Checking security readiness...")

        # SSL/TLS certificates
        await self.check_ssl_certificates()

        # Security policies
        await self.check_security_policies()

        # Network policies
        await self.check_network_policies()

        # Secrets management
        await self.check_secrets_management()

        # Access controls
        await self.check_access_controls()

    async def check_ssl_certificates(self):
        """Check SSL/TLS certificates"""
        domains = self.config.get('security', {}).get('domains_to_check', [])

        if not domains:
            self.checks.append(ReadinessCheck(
                name="SSL/TLS Certificates",
                category=CheckCategory.SECURITY,
                status=ReadinessStatus.UNKNOWN,
                message="No domains configured for SSL check",
                critical=False
            ))
            return

        valid_certs = 0
        total_certs = len(domains)

        for domain in domains:
            try:
                # Check SSL certificate
                import ssl
                context = ssl.create_default_context()
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED

                with socket.create_connection((domain, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()

                        # Check expiration
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.utcnow()).days

                        if days_until_expiry > 30:
                            valid_certs += 1
                        else:
                            logger.warning(f"SSL certificate for {domain} expires in {days_until_expiry} days")

            except Exception as e:
                logger.warning(f"Failed to check SSL certificate for {domain}: {e}")

        if valid_certs == total_certs:
            status = ReadinessStatus.READY
            message = f"All SSL certificates valid ({valid_certs}/{total_certs})"
        elif valid_certs > 0:
            status = ReadinessStatus.WARNING
            message = f"Some SSL certificates need attention ({valid_certs}/{total_certs})"
        else:
            status = ReadinessStatus.NOT_READY
            message = f"No valid SSL certificates found ({valid_certs}/{total_certs})"
            self.critical_issues.append("SSL certificate issues")

        self.checks.append(ReadinessCheck(
            name="SSL/TLS Certificates",
            category=CheckCategory.SECURITY,
            status=status,
            message=message,
            details={
                'valid_certificates': valid_certs,
                'total_certificates': total_certs,
                'domains_checked': domains
            },
            critical=True
        ))

    async def check_security_policies(self):
        """Check security policies"""
        try:
            # Check Pod Security Policies
            psp_result = subprocess.run(
                ['kubectl', 'get', 'podsecuritypolicy', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if psp_result.returncode == 0:
                psps = json.loads(psp_result.stdout)
                psp_count = len(psps.get('items', []))

                # Check Network Policies
                np_result = subprocess.run(
                    ['kubectl', 'get', 'networkpolicy', '-A', '-o', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if np_result.returncode == 0:
                    nps = json.loads(np_result.stdout)
                    np_count = len(nps.get('items', []))

                    if psp_count > 0 and np_count > 0:
                        status = ReadinessStatus.READY
                        message = f"Security policies configured ({psp_count} PSPs, {np_count} Network Policies)"
                    else:
                        status = ReadinessStatus.WARNING
                        message = f"Limited security policies ({psp_count} PSPs, {np_count} Network Policies)"
                        self.recommendations.append("Configure additional security policies")
                else:
                    status = ReadinessStatus.WARNING
                    message = f"Pod Security Policies configured ({psp_count}), Network Policies check failed"
            else:
                status = ReadinessStatus.NOT_READY
                message = "Security policies not configured"
                self.critical_issues.append("Security policies not configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Security policies check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Security Policies",
            category=CheckCategory.SECURITY,
            status=status,
            message=message,
            critical=True
        ))

    async def check_network_policies(self):
        """Check network policies"""
        try:
            # Check existing network policies
            result = subprocess.run(
                ['kubectl', 'get', 'networkpolicy', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                policies = json.loads(result.stdout)
                policy_count = len(policies.get('items', []))

                # Check if policies are applied to production namespaces
                production_policies = 0
                for policy in policies.get('items', []):
                    namespace = policy.get('metadata', {}).get('namespace', '')
                    if 'production' in namespace.lower():
                        production_policies += 1

                if production_policies >= 3:  # Minimum 3 policies for production
                    status = ReadinessStatus.READY
                    message = f"Network policies properly configured ({production_policies} in production namespaces)"
                elif policy_count > 0:
                    status = ReadinessStatus.WARNING
                    message = f"Limited network policies ({production_policies} in production namespaces)"
                    self.recommendations.append("Add network policies to production namespaces")
                else:
                    status = ReadinessStatus.NOT_READY
                    message = "No network policies configured"
                    self.critical_issues.append("No network policies configured")
            else:
                status = ReadinessStatus.UNKNOWN
                message = "Failed to check network policies"

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Network policies check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Network Policies",
            category=CheckCategory.SECURITY,
            status=status,
            message=message,
            critical=True
        ))

    async def check_secrets_management(self):
        """Check secrets management"""
        try:
            # Check if secrets are properly managed
            result = subprocess.run(
                ['kubectl', 'get', 'secrets', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                secrets = json.loads(result.stdout)
                secret_items = secrets.get('items', [])

                # Check for base64 encoded secrets (not plaintext)
                plaintext_secrets = 0
                managed_secrets = 0

                for secret in secret_items:
                    secret_data = secret.get('data', {})
                    for key, value in secret_data.items():
                        try:
                            # Try to decode base64
                            import base64
                            decoded = base64.b64decode(value).decode('utf-8')
                            # Check if it looks like sensitive data
                            if any(sensitive in decoded.lower() for sensitive in ['password', 'key', 'secret', 'token']):
                                managed_secrets += 1
                        except:
                            plaintext_secrets += 1

                if plaintext_secrets == 0 and managed_secrets > 0:
                    status = ReadinessStatus.READY
                    message = f"Secrets properly managed ({managed_secrets} sensitive secrets)"
                elif plaintext_secrets == 0:
                    status = ReadinessStatus.WARNING
                    message = "No secrets found or all secrets are basic"
                else:
                    status = ReadinessStatus.NOT_READY
                    message = f"Potential plaintext secrets detected ({plaintext_secrets})"
                    self.critical_issues.append("Potential plaintext secrets")
            else:
                status = ReadinessStatus.UNKNOWN
                message = "Failed to check secrets management"

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Secrets management check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Secrets Management",
            category=CheckCategory.SECURITY,
            status=status,
            message=message,
            critical=True
        ))

    async def check_access_controls(self):
        """Check access controls"""
        try:
            # Check RBAC configuration
            result = subprocess.run(
                ['kubectl', 'get', 'rolebindings,clusterrolebindings', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                bindings = json.loads(result.stdout)
                binding_items = bindings.get('items', [])

                # Check for proper role bindings
                production_bindings = 0
                admin_bindings = 0

                for binding in binding_items:
                    binding_type = binding.get('kind', '')
                    namespace = binding.get('metadata', {}).get('namespace', '')

                    if 'production' in namespace.lower():
                        production_bindings += 1

                    subjects = binding.get('subjects', [])
                    for subject in subjects:
                        if subject.get('kind') == 'ServiceAccount':
                            role_ref = binding.get('roleRef', {})
                            if 'admin' in role_ref.get('name', '').lower():
                                admin_bindings += 1

                if production_bindings > 0 and admin_bindings == 0:
                    status = ReadinessStatus.READY
                    message = f"Access controls properly configured ({production_bindings} production bindings)"
                elif production_bindings > 0:
                    status = ReadinessStatus.WARNING
                    message = f"Access controls configured but admin privileges detected ({production_bindings} production bindings)"
                    self.recommendations.append("Review admin privileges in production")
                else:
                    status = ReadinessStatus.NOT_READY
                    message = "No access controls configured for production"
                    self.critical_issues.append("No access controls configured")
            else:
                status = ReadinessStatus.UNKNOWN
                message = "Failed to check access controls"

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Access controls check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Access Controls",
            category=CheckCategory.SECURITY,
            status=status,
            message=message,
            critical=True
        ))

    async def check_performance_readiness(self):
        """Check performance readiness"""
        logger.info("Checking performance readiness...")

        # Resource limits
        await self.check_resource_limits()

        # Performance monitoring
        await self.check_performance_monitoring()

        # Scaling configuration
        await self.check_scaling_configuration()

    async def check_resource_limits(self):
        """Check resource limits and requests"""
        try:
            # Check deployments for resource limits
            result = subprocess.run(
                ['kubectl', 'get', 'deployments', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                deployments = json.loads(result.stdout)
                deployment_items = deployments.get('items', [])

                deployments_with_limits = 0
                deployments_with_requests = 0
                total_deployments = 0

                for deployment in deployment_items:
                    namespace = deployment.get('metadata', {}).get('namespace', '')
                    if 'production' in namespace.lower():
                        total_deployments += 1

                        containers = deployment.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
                        has_limits = False
                        has_requests = False

                        for container in containers:
                            resources = container.get('resources', {})
                            if resources.get('limits'):
                                has_limits = True
                            if resources.get('requests'):
                                has_requests = True

                        if has_limits:
                            deployments_with_limits += 1
                        if has_requests:
                            deployments_with_requests += 1

                if total_deployments > 0:
                    limits_percentage = (deployments_with_limits / total_deployments) * 100
                    requests_percentage = (deployments_with_requests / total_deployments) * 100

                    if limits_percentage >= 90 and requests_percentage >= 90:
                        status = ReadinessStatus.READY
                        message = f"Resource limits properly configured ({limits_percentage:.1f}% with limits, {requests_percentage:.1f}% with requests)"
                    elif limits_percentage >= 70 and requests_percentage >= 70:
                        status = ReadinessStatus.WARNING
                        message = f"Resource limits partially configured ({limits_percentage:.1f}% with limits, {requests_percentage:.1f}% with requests)"
                        self.recommendations.append("Add resource limits to all production deployments")
                    else:
                        status = ReadinessStatus.NOT_READY
                        message = f"Resource limits not properly configured ({limits_percentage:.1f}% with limits, {requests_percentage:.1f}% with requests)"
                        self.critical_issues.append("Resource limits not configured")
                else:
                    status = ReadinessStatus.UNKNOWN
                    message = "No production deployments found"
            else:
                status = ReadinessStatus.UNKNOWN
                message = "Failed to check resource limits"

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Resource limits check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Resource Limits",
            category=CheckCategory.PERFORMANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_performance_monitoring(self):
        """Check performance monitoring configuration"""
        try:
            # Check for monitoring tools
            monitoring_checks = [
                ("Prometheus", "prometheus-server"),
                ("Grafana", "grafana"),
                ("Metrics Server", "metrics-server")
            ]

            available_tools = 0
            total_tools = len(monitoring_checks)

            for tool_name, deployment_name in monitoring_checks:
                result = subprocess.run(
                    ['kubectl', 'get', 'deployment', deployment_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    available_tools += 1

            if available_tools >= 2:  # At least Prometheus + Grafana
                status = ReadinessStatus.READY
                message = f"Performance monitoring configured ({available_tools}/{total_tools} tools)"
            elif available_tools >= 1:
                status = ReadinessStatus.WARNING
                message = f"Limited performance monitoring ({available_tools}/{total_tools} tools)"
                self.recommendations.append("Set up comprehensive monitoring stack")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No performance monitoring configured"
                self.critical_issues.append("No performance monitoring configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Performance monitoring check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Performance Monitoring",
            category=CheckCategory.PERFORMANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_scaling_configuration(self):
        """Check scaling configuration"""
        try:
            # Check for HPA configurations
            result = subprocess.run(
                ['kubectl', 'get', 'hpa', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                hpas = json.loads(result.stdout)
                hpa_items = hpas.get('items', [])

                production_hpas = 0
                for hpa in hpa_items:
                    namespace = hpa.get('metadata', {}).get('namespace', '')
                    if 'production' in namespace.lower():
                        production_hpas += 1

                if production_hpas >= 2:  # At least 2 HPAs for critical services
                    status = ReadinessStatus.READY
                    message = f"Scaling configured ({production_hpas} HPAs in production)"
                elif production_hpas >= 1:
                    status = ReadinessStatus.WARNING
                    message = f"Limited scaling configuration ({production_hpas} HPAs in production)"
                    self.recommendations.append("Configure HPA for more production services")
                else:
                    status = ReadinessStatus.NOT_READY
                    message = "No scaling configuration for production"
                    self.critical_issues.append("No scaling configuration")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No HPA configurations found"
                self.critical_issues.append("No HPA configurations found")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Scaling configuration check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Scaling Configuration",
            category=CheckCategory.PERFORMANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_monitoring_readiness(self):
        """Check monitoring readiness"""
        logger.info("Checking monitoring readiness...")

        # Alerting configuration
        await self.check_alerting_configuration()

        # Logging configuration
        await self.check_logging_configuration()

        # Metrics collection
        await self.check_metrics_collection()

    async def check_alerting_configuration(self):
        """Check alerting configuration"""
        try:
            # Check for Alertmanager
            result = subprocess.run(
                ['kubectl', 'get', 'deployment', 'alertmanager', '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Check for alert rules
                rules_result = subprocess.run(
                    ['kubectl', 'get', 'prometheusrules', '-A', '-o', 'json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if rules_result.returncode == 0:
                    rules = json.loads(rules_result.stdout)
                    rule_count = len(rules.get('items', []))

                    if rule_count >= 5:  # Minimum 5 alert rules
                        status = ReadinessStatus.READY
                        message = f"Alerting properly configured ({rule_count} alert rules)"
                    else:
                        status = ReadinessStatus.WARNING
                        message = f"Limited alerting configuration ({rule_count} alert rules)"
                        self.recommendations.append("Add more comprehensive alert rules")
                else:
                    status = ReadinessStatus.WARNING
                    message = "Alertmanager configured but no alert rules found"
                    self.recommendations.append("Configure alert rules")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No alerting system configured"
                self.critical_issues.append("No alerting system configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Alerting configuration check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Alerting Configuration",
            category=CheckCategory.MONITORING,
            status=status,
            message=message,
            critical=True
        ))

    async def check_logging_configuration(self):
        """Check logging configuration"""
        try:
            # Check for logging infrastructure
            logging_checks = [
                ("Fluentd", "fluentd"),
                ("Elasticsearch", "elasticsearch"),
                ("Kibana", "kibana")
            ]

            available_components = 0
            total_components = len(logging_checks)

            for component_name, deployment_name in logging_checks:
                result = subprocess.run(
                    ['kubectl', 'get', 'deployment', deployment_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    available_components += 1

            if available_components >= 2:  # At least Fluentd + Elasticsearch
                status = ReadinessStatus.READY
                message = f"Logging infrastructure configured ({available_components}/{total_components} components)"
            elif available_components >= 1:
                status = ReadinessStatus.WARNING
                message = f"Limited logging infrastructure ({available_components}/{total_components} components)"
                self.recommendations.append("Complete logging infrastructure setup")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No logging infrastructure configured"
                self.critical_issues.append("No logging infrastructure configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Logging configuration check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Logging Configuration",
            category=CheckCategory.MONITORING,
            status=status,
            message=message,
            critical=True
        ))

    async def check_metrics_collection(self):
        """Check metrics collection"""
        try:
            # Check for ServiceMonitor resources
            result = subprocess.run(
                ['kubectl', 'get', 'servicemonitor', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                monitors = json.loads(result.stdout)
                monitor_count = len(monitors.get('items', []))

                if monitor_count >= 3:  # Minimum 3 service monitors
                    status = ReadinessStatus.READY
                    message = f"Metrics collection configured ({monitor_count} service monitors)"
                else:
                    status = ReadinessStatus.WARNING
                    message = f"Limited metrics collection ({monitor_count} service monitors)"
                    self.recommendations.append("Configure more service monitors")
            else:
                status = ReadinessStatus.WARNING
                message = "No ServiceMonitor resources found"
                self.recommendations.append("Configure ServiceMonitor resources")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Metrics collection check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Metrics Collection",
            category=CheckCategory.MONITORING,
            status=status,
            message=message,
            critical=False
        ))

    async def check_backup_readiness(self):
        """Check backup readiness"""
        logger.info("Checking backup readiness...")

        # Backup configuration
        await self.check_backup_configuration()

        # Backup testing
        await self.check_backup_testing()

        # Disaster recovery
        await self.check_disaster_recovery()

    async def check_backup_configuration(self):
        """Check backup configuration"""
        try:
            # Check for backup tools (Velero, etc.)
            backup_checks = [
                ("Velero", "velero"),
                ("Restic", "restic")
            ]

            available_tools = 0
            total_tools = len(backup_checks)

            for tool_name, deployment_name in backup_checks:
                result = subprocess.run(
                    ['kubectl', 'get', 'deployment', deployment_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    available_tools += 1

            # Check for backup schedules
            schedule_result = subprocess.run(
                ['kubectl', 'get', 'schedule', '-A', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            backup_schedules = 0
            if schedule_result.returncode == 0:
                schedules = json.loads(schedule_result.stdout)
                backup_schedules = len(schedules.get('items', []))

            if available_tools > 0 and backup_schedules > 0:
                status = ReadinessStatus.READY
                message = f"Backup system configured ({available_tools} tools, {backup_schedules} schedules)"
            elif available_tools > 0:
                status = ReadinessStatus.WARNING
                message = f"Backup tools available but no schedules ({available_tools} tools, {backup_schedules} schedules)"
                self.recommendations.append("Configure backup schedules")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No backup system configured"
                self.critical_issues.append("No backup system configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Backup configuration check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Backup Configuration",
            category=CheckCategory.BACKUP,
            status=status,
            message=message,
            critical=True
        ))

    async def check_backup_testing(self):
        """Check backup testing procedures"""
        try:
            # Check for backup test procedures
            test_checks = [
                ("Backup Test Jobs", "backup-test"),
                ("Restore Test Jobs", "restore-test")
            ]

            test_procedures = 0
            total_tests = len(test_checks)

            for test_name, label_selector in test_checks:
                result = subprocess.run(
                    ['kubectl', 'get', 'job', '-l', f'app={label_selector}', '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    test_procedures += 1

            if test_procedures >= 2:
                status = ReadinessStatus.READY
                message = f"Backup testing procedures configured ({test_procedures}/{total_tests} procedures)"
            elif test_procedures >= 1:
                status = ReadinessStatus.WARNING
                message = f"Partial backup testing procedures ({test_procedures}/{total_tests} procedures)"
                self.recommendations.append("Complete backup testing setup")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No backup testing procedures configured"
                self.critical_issues.append("No backup testing procedures")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Backup testing check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Backup Testing",
            category=CheckCategory.BACKUP,
            status=status,
            message=message,
            critical=True
        ))

    async def check_disaster_recovery(self):
        """Check disaster recovery procedures"""
        try:
            # Check for disaster recovery documentation and procedures
            dr_checklist = [
                ("DR Documentation", "disaster-recovery-docs"),
                ("DR Playbooks", "dr-playbooks"),
                ("DR Contact List", "dr-contacts")
            ]

            dr_components = 0
            total_components = len(dr_checklist)

            for component_name, configmap_name in dr_checklist:
                result = subprocess.run(
                    ['kubectl', 'get', 'configmap', configmap_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    dr_components += 1

            if dr_components >= 2:
                status = ReadinessStatus.READY
                message = f"Disaster recovery procedures documented ({dr_components}/{total_components} components)"
            elif dr_components >= 1:
                status = ReadinessStatus.WARNING
                message = f"Partial disaster recovery documentation ({dr_components}/{total_components} components)"
                self.recommendations.append("Complete disaster recovery documentation")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No disaster recovery procedures documented"
                self.critical_issues.append("No disaster recovery procedures")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Disaster recovery check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Disaster Recovery",
            category=CheckCategory.BACKUP,
            status=status,
            message=message,
            critical=True
        ))

    async def check_compliance_readiness(self):
        """Check compliance readiness"""
        logger.info("Checking compliance readiness...")

        # Security compliance
        await self.check_security_compliance()

        # Data privacy
        await self.check_data_privacy()

        # Audit logging
        await self.check_audit_logging()

    async def check_security_compliance(self):
        """Check security compliance"""
        try:
            # Check for security scanning tools
            security_tools = [
                ("Trivy", "trivy"),
                ("Falco", "falco"),
                ("Kube-bench", "kube-bench")
            ]

            available_tools = 0
            total_tools = len(security_tools)

            for tool_name, deployment_name in security_tools:
                result = subprocess.run(
                    ['kubectl', 'get', 'deployment', deployment_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    available_tools += 1

            if available_tools >= 2:
                status = ReadinessStatus.READY
                message = f"Security compliance tools configured ({available_tools}/{total_tools} tools)"
            elif available_tools >= 1:
                status = ReadinessStatus.WARNING
                message = f"Limited security compliance tools ({available_tools}/{total_tools} tools)"
                self.recommendations.append("Add more security compliance tools")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No security compliance tools configured"
                self.critical_issues.append("No security compliance tools configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Security compliance check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Security Compliance",
            category=CheckCategory.COMPLIANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_data_privacy(self):
        """Check data privacy measures"""
        try:
            # Check for data privacy configurations
            privacy_checks = [
                ("Data Encryption", "data-encryption-config"),
                ("Data Retention", "data-retention-policy"),
                ("Access Logs", "access-logging")
            ]

            privacy_measures = 0
            total_measures = len(privacy_checks)

            for measure_name, configmap_name in privacy_checks:
                result = subprocess.run(
                    ['kubectl', 'get', 'configmap', configmap_name, '-A', '-o', 'jsonpath={.items[0].metadata.name}'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    privacy_measures += 1

            if privacy_measures >= 2:
                status = ReadinessStatus.READY
                message = f"Data privacy measures configured ({privacy_measures}/{total_measures} measures)"
            elif privacy_measures >= 1:
                status = ReadinessStatus.WARNING
                message = f"Limited data privacy measures ({privacy_measures}/{total_measures} measures)"
                self.recommendations.append("Complete data privacy configuration")
            else:
                status = ReadinessStatus.NOT_READY
                message = "No data privacy measures configured"
                self.critical_issues.append("No data privacy measures configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Data privacy check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Data Privacy",
            category=CheckCategory.COMPLIANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_audit_logging(self):
        """Check audit logging"""
        try:
            # Check for audit logging configuration
            audit_result = subprocess.run(
                ['kubectl', 'get', 'pod', '-l', 'component=kube-apiserver', '-n', 'kube-system', '-o', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if audit_result.returncode == 0:
                pods = json.loads(audit_result.stdout)
                api_server_pods = pods.get('items', [])

                # Check if audit logging is enabled
                audit_enabled = False
                for pod in api_server_pods:
                    spec = pod.get('spec', {})
                    containers = spec.get('containers', [])
                    for container in containers:
                        args = container.get('args', [])
                        if any('--audit-log-path' in arg for arg in args):
                            audit_enabled = True
                            break

                if audit_enabled:
                    status = ReadinessStatus.READY
                    message = "Kubernetes audit logging enabled"
                else:
                    status = ReadinessStatus.WARNING
                    message = "Kubernetes audit logging not enabled"
                    self.recommendations.append("Enable Kubernetes audit logging")
            else:
                status = ReadinessStatus.UNKNOWN
                message = "Failed to check audit logging configuration"

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Audit logging check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Audit Logging",
            category=CheckCategory.COMPLIANCE,
            status=status,
            message=message,
            critical=True
        ))

    async def check_deployment_readiness(self):
        """Check deployment readiness"""
        logger.info("Checking deployment readiness...")

        # Deployment manifests
        await self.check_deployment_manifests()

        # CI/CD pipeline
        await self.check_cicd_pipeline()

        # Rollback procedures
        await self.check_rollback_procedures()

    async def check_deployment_manifests(self):
        """Check deployment manifests"""
        try:
            # Check if production deployment manifests exist
            manifest_files = [
                'kubernetes/production/production-deployment.yaml',
                'kubernetes/production/values.yaml',
                'kubernetes/production/secrets.yaml'
            ]

            existing_manifests = 0
            total_manifests = len(manifest_files)

            for manifest_file in manifest_files:
                if os.path.exists(manifest_file):
                    existing_manifests += 1

            if existing_manifests == total_manifests:
                status = ReadinessStatus.READY
                message = f"All deployment manifests available ({existing_manifests}/{total_manifests})"
            elif existing_manifests >= 2:
                status = ReadinessStatus.WARNING
                message = f"Most deployment manifests available ({existing_manifests}/{total_manifests})"
                self.recommendations.append("Complete deployment manifests")
            else:
                status = ReadinessStatus.NOT_READY
                message = f"Missing deployment manifests ({existing_manifests}/{total_manifests})"
                self.critical_issues.append("Missing deployment manifests")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Deployment manifests check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Deployment Manifests",
            category=CheckCategory.DEPLOYMENT,
            status=status,
            message=message,
            critical=True
        ))

    async def check_cicd_pipeline(self):
        """Check CI/CD pipeline"""
        try:
            # Check for CI/CD pipeline configuration
            cicd_files = [
                '.github/workflows/ci.yml',
                '.github/workflows/cd.yml',
                '.github/workflows/agency-swarm-cicd.yml'
            ]

            existing_cicd = 0
            total_cicd = len(cicd_files)

            for cicd_file in cicd_files:
                if os.path.exists(cicd_file):
                    existing_cicd += 1

            if existing_cicd >= 2:
                status = ReadinessStatus.READY
                message = f"CI/CD pipeline configured ({existing_cicd}/{total_cicd} files)"
            elif existing_cicd >= 1:
                status = ReadinessStatus.WARNING
                message = f"Partial CI/CD pipeline ({existing_cicd}/{total_cicd} files)"
                self.recommendations.append("Complete CI/CD pipeline configuration")
            else:
                status = ReadinessStatus.NOT_READY
                message = f"No CI/CD pipeline configured ({existing_cicd}/{total_cicd} files)"
                self.critical_issues.append("No CI/CD pipeline configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"CI/CD pipeline check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="CI/CD Pipeline",
            category=CheckCategory.DEPLOYMENT,
            status=status,
            message=message,
            critical=True
        ))

    async def check_rollback_procedures(self):
        """Check rollback procedures"""
        try:
            # Check for rollback scripts and procedures
            rollback_files = [
                'scripts/deployment_scripts.py',
                'scripts/rollback_procedures.py',
                'kubernetes/rollback.yaml'
            ]

            existing_rollback = 0
            total_rollback = len(rollback_files)

            for rollback_file in rollback_files:
                if os.path.exists(rollback_file):
                    existing_rollback += 1

            if existing_rollback >= 2:
                status = ReadinessStatus.READY
                message = f"Rollback procedures configured ({existing_rollback}/{total_rollback} files)"
            elif existing_rollback >= 1:
                status = ReadinessStatus.WARNING
                message = f"Partial rollback procedures ({existing_rollback}/{total_rollback} files)"
                self.recommendations.append("Complete rollback procedures")
            else:
                status = ReadinessStatus.NOT_READY
                message = f"No rollback procedures configured ({existing_rollback}/{total_rollback} files)"
                self.critical_issues.append("No rollback procedures configured")

        except Exception as e:
            status = ReadinessStatus.UNKNOWN
            message = f"Rollback procedures check error: {str(e)}"

        self.checks.append(ReadinessCheck(
            name="Rollback Procedures",
            category=CheckCategory.DEPLOYMENT,
            status=status,
            message=message,
            critical=True
        ))

    def calculate_overall_readiness(self) -> Tuple[ReadinessStatus, float]:
        """Calculate overall readiness status and score"""
        if not self.checks:
            return ReadinessStatus.UNKNOWN, 0.0

        # Count checks by status
        status_counts = {status: 0 for status in ReadinessStatus}
        critical_checks = 0
        critical_ready = 0

        for check in self.checks:
            status_counts[check.status] += 1
            if check.critical:
                critical_checks += 1
                if check.status == ReadinessStatus.READY:
                    critical_ready += 1

        total_checks = len(self.checks)
        ready_checks = status_counts[ReadinessStatus.READY]
        warning_checks = status_counts[ReadinessStatus.WARNING]
        not_ready_checks = status_counts[ReadinessStatus.NOT_READY]

        # Calculate readiness score
        base_score = (ready_checks / total_checks) * 100
        warning_penalty = (warning_checks / total_checks) * 20
        not_ready_penalty = (not_ready_checks / total_checks) * 50

        score = max(0, base_score - warning_penalty - not_ready_penalty)

        # Critical readiness percentage
        critical_readiness = (critical_ready / critical_checks) * 100 if critical_checks > 0 else 0

        # Determine overall status
        if self.critical_issues:
            overall_status = ReadinessStatus.NOT_READY
        elif critical_readiness < 90:
            overall_status = ReadinessStatus.WARNING
        elif score >= 90:
            overall_status = ReadinessStatus.READY
        elif score >= 70:
            overall_status = ReadinessStatus.WARNING
        else:
            overall_status = ReadinessStatus.NOT_READY

        return overall_status, score

    async def generate_recommendations(self):
        """Generate recommendations based on check results"""
        # Add general recommendations
        warning_checks = [check for check in self.checks if check.status == ReadinessStatus.WARNING]
        not_ready_checks = [check for check in self.checks if check.status == ReadinessStatus.NOT_READY]

        if warning_checks:
            self.recommendations.append(f"Address {len(warning_checks)} warnings before production deployment")

        if not_ready_checks:
            self.recommendations.append(f"Fix {len(not_ready_checks)} critical issues before production deployment")

        # Category-specific recommendations
        categories_with_issues = set(check.category for check in self.checks if check.status != ReadinessStatus.READY)
        for category in categories_with_issues:
            category_issues = [check for check in self.checks if check.category == category and check.status != ReadinessStatus.READY]
            if len(category_issues) > 1:
                self.recommendations.append(f"Focus on {category.value} improvements ({len(category_issues)} issues)")

    async def generate_readiness_report(self, report: ProductionReadinessReport) -> str:
        """Generate detailed readiness report"""
        report_lines = []
        report_lines.append("=== Agency Swarm Production Readiness Report ===")
        report_lines.append("")
        report_lines.append(f"Deployment: {report.deployment_name}")
        report_lines.append(f"Environment: {report.environment}")
        report_lines.append(f"Overall Status: {report.overall_status.value.upper()}")
        report_lines.append(f"Readiness Score: {report.readiness_score:.1f}/100")
        report_lines.append(f"Timestamp: {report.timestamp.isoformat()}")
        report_lines.append(f"Generated by: {report.generated_by}")
        report_lines.append("")

        # Summary
        report_lines.append("=== Summary ===")
        status_counts = {status: 0 for status in ReadinessStatus}
        for check in report.checks:
            status_counts[check.status] += 1

        report_lines.append(f"Total Checks: {len(report.checks)}")
        report_lines.append(f"Ready: {status_counts[ReadinessStatus.READY]}")
        report_lines.append(f"Warnings: {status_counts[ReadinessStatus.WARNING]}")
        report_lines.append(f"Not Ready: {status_counts[ReadinessStatus.NOT_READY]}")
        report_lines.append(f"Unknown: {status_counts[ReadinessStatus.UNKNOWN]}")
        report_lines.append(f"Critical Issues: {len(report.critical_issues)}")
        report_lines.append("")

        # Critical Issues
        if report.critical_issues:
            report_lines.append("=== CRITICAL ISSUES ===")
            for i, issue in enumerate(report.critical_issues, 1):
                report_lines.append(f"{i}. {issue}")
            report_lines.append("")

        # Category Breakdown
        report_lines.append("=== Category Breakdown ===")
        for category in CheckCategory:
            category_checks = [check for check in report.checks if check.category == category]
            if category_checks:
                ready_count = len([c for c in category_checks if c.status == ReadinessStatus.READY])
                report_lines.append(f"{category.value.title()}: {ready_count}/{len(category_checks)} ready")
        report_lines.append("")

        # Detailed Results
        report_lines.append("=== Detailed Results ===")
        for category in CheckCategory:
            category_checks = [check for check in report.checks if check.category == category]
            if category_checks:
                report_lines.append(f"\n--- {category.value.title()} ---")
                for check in category_checks:
                    status_icon = {
                        ReadinessStatus.READY: "✅",
                        ReadinessStatus.WARNING: "⚠️",
                        ReadinessStatus.NOT_READY: "❌",
                        ReadinessStatus.UNKNOWN: "❓"
                    }[check.status]

                    report_lines.append(f"{status_icon} {check.name}")
                    report_lines.append(f"   Status: {check.status.value}")
                    report_lines.append(f"   Message: {check.message}")
                    if check.details:
                        report_lines.append(f"   Details: {len(check.details)} items")
                    report_lines.append("")

        # Recommendations
        if report.recommendations:
            report_lines.append("=== Recommendations ===")
            for i, rec in enumerate(report.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        # Next Steps
        if report.overall_status == ReadinessStatus.READY:
            report_lines.append("=== Next Steps ===")
            report_lines.append("✅ Production deployment is ready!")
            report_lines.append("1. Review all warnings and address if possible")
            report_lines.append("2. Schedule deployment during maintenance window")
            report_lines.append("3. Prepare rollback procedures")
            report_lines.append("4. Monitor deployment closely")
        elif report.overall_status == ReadinessStatus.WARNING:
            report_lines.append("=== Next Steps ===")
            report_lines.append("⚠️ Production deployment requires attention")
            report_lines.append("1. Address all warnings before deployment")
            report_lines.append("2. Test fixes in staging environment")
            report_lines.append("3. Update documentation")
            report_lines.append("4. Re-run readiness check")
        else:
            report_lines.append("=== Next Steps ===")
            report_lines.append("❌ Production deployment is not ready")
            report_lines.append("1. Fix all critical issues immediately")
            report_lines.append("2. Implement missing components")
            report_lines.append("3. Test all configurations")
            report_lines.append("4. Re-run readiness check")

        return "\n".join(report_lines)

    async def save_readiness_report(self, report: ProductionReadinessReport, filename: str = None):
        """Save readiness report to file"""
        if filename is None:
            timestamp = report.timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"production_readiness_report_{timestamp}.txt"

        report_text = await self.generate_readiness_report(report)

        try:
            with open(filename, 'w') as f:
                f.write(report_text)
            logger.info(f"Readiness report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving readiness report: {e}")

    async def shutdown(self):
        """Shutdown readiness checker"""
        logger.info("Shutting down production readiness checker...")

        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Production readiness checker shutdown complete")

# Configuration template
DEFAULT_CONFIG = {
    "deployment_name": "Agency Swarm",
    "environment": "production",
    "database": {
        "url": "postgresql://postgres:postgres@localhost:5432/archon",
        "max_size_gb": 100
    },
    "redis": {
        "host": "localhost",
        "port": 6379
    },
    "resource_requirements": {
        "cpu": 16,
        "memory_gb": 32,
        "storage_gb": 100
    },
    "security": {
        "domains_to_check": [
            "api.agency-swarm.com",
            "app.agency-swarm.com"
        ]
    }
}

async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Agency Swarm Production Readiness Check")
    parser.add_argument("--environment", default="production", help="Environment to check")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file for readiness report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and initialize readiness checker
    readiness_checker = ProductionReadinessChecker(config)
    await readiness_checker.initialize()

    try:
        # Run readiness check
        report = await readiness_checker.run_comprehensive_check()

        # Generate and display report
        report_text = await readiness_checker.generate_readiness_report(report)
        print(report_text)

        # Save report if output specified
        if args.output:
            await readiness_checker.save_readiness_report(report, args.output)

        # Exit with appropriate code
        if report.overall_status == ReadinessStatus.READY:
            exit_code = 0
        elif report.overall_status == ReadinessStatus.WARNING:
            exit_code = 1
        else:
            exit_code = 2

        exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error running readiness check: {e}")
        exit(3)
    finally:
        await readiness_checker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())