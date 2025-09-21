#!/usr/bin/env python3
"""
Deployment Tests for Agency Swarm System
Production deployment validation and configuration testing
"""

import asyncio
import aiohttp
import json
import logging
import subprocess
import docker
import kubernetes
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import yaml
import socket
import requests

logger = logging.getLogger(__name__)

class DeploymentTester:
    """Production deployment testing and validation system"""

    def __init__(self):
        self.services = {
            "frontend": "http://localhost:3737",
            "api": "http://localhost:8181",
            "mcp": "http://localhost:8051",
            "agents": "http://localhost:8052"
        }
        self.deployment_results = {}
        self.deployment_config = self.load_deployment_config()

    def load_deployment_config(self):
        """Load deployment configuration"""
        config_path = Path("../deployment/docker-compose.yml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    async def test_docker_deployment(self):
        """Test Docker deployment configuration"""
        logger.info("Testing Docker deployment configuration...")

        docker_tests = []

        try:
            # Initialize Docker client
            client = docker.from_env()

            # Test if Docker is running
            docker_tests.append({
                "test_name": "Docker Service Status",
                "status": "passed",
                "docker_running": True
            })

            # Test Docker Compose configuration
            try:
                # Parse docker-compose.yml
                if self.deployment_config:
                    services = self.deployment_config.get("services", {})
                    required_services = ["archon-frontend", "archon-api", "archon-mcp", "archon-agents"]

                    missing_services = [s for s in required_services if s not in services]
                    if not missing_services:
                        docker_tests.append({
                            "test_name": "Docker Compose Configuration",
                            "status": "passed",
                            "configured_services": list(services.keys()),
                            "missing_services": []
                        })
                    else:
                        docker_tests.append({
                            "test_name": "Docker Compose Configuration",
                            "status": "failed",
                            "configured_services": list(services.keys()),
                            "missing_services": missing_services
                        })
                else:
                    docker_tests.append({
                        "test_name": "Docker Compose Configuration",
                        "status": "failed",
                        "error": "docker-compose.yml not found"
                    })
            except Exception as e:
                docker_tests.append({
                    "test_name": "Docker Compose Configuration",
                    "status": "failed",
                    "error": str(e)
                })

            # Test container images
            if self.deployment_config:
                for service_name, service_config in self.deployment_config.get("services", {}).items():
                    try:
                        image_name = service_config.get("image", "")
                        if image_name:
                            # Try to pull image
                            image = client.images.pull(image_name)
                            docker_tests.append({
                                "test_name": f"Container Image - {service_name}",
                                "status": "passed",
                                "image": image_name,
                                "image_exists": True
                            })
                        else:
                            docker_tests.append({
                                "test_name": f"Container Image - {service_name}",
                                "status": "failed",
                                "error": "No image specified"
                            })
                    except Exception as e:
                        docker_tests.append({
                            "test_name": f"Container Image - {service_name}",
                            "status": "failed",
                            "error": str(e)
                        })

            # Test network configuration
            try:
                networks = client.networks.list()
                agency_swarm_network = any("archon" in net.name.lower() for net in networks)

                docker_tests.append({
                    "test_name": "Docker Network Configuration",
                    "status": "passed" if agency_swarm_network else "warning",
                    "network_exists": agency_swarm_network,
                    "available_networks": [net.name for net in networks]
                })
            except Exception as e:
                docker_tests.append({
                    "test_name": "Docker Network Configuration",
                    "status": "failed",
                    "error": str(e)
                })

            # Test volume configuration
            try:
                volumes = client.volumes.list()
                data_volumes = any("archon" in vol.name.lower() for vol in volumes)

                docker_tests.append({
                    "test_name": "Docker Volume Configuration",
                    "status": "passed" if data_volumes else "warning",
                    "volumes_exist": data_volumes,
                    "available_volumes": [vol.name for vol in volumes]
                })
            except Exception as e:
                docker_tests.append({
                    "test_name": "Docker Volume Configuration",
                    "status": "failed",
                    "error": str(e)
                })

        except Exception as e:
            docker_tests.append({
                "test_name": "Docker Deployment",
                "status": "failed",
                "error": f"Docker not available: {e}"
            })

        return {
            "test_name": "Docker Deployment",
            "tests": docker_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in docker_tests
            ) else "needs_attention"
        }

    async def test_kubernetes_deployment(self):
        """Test Kubernetes deployment configuration"""
        logger.info("Testing Kubernetes deployment configuration...")

        k8s_tests = []

        try:
            # Initialize Kubernetes client
            kubernetes.config.load_kube_config()
            k8s_client = kubernetes.client.CoreV1Api()
            apps_client = kubernetes.client.AppsV1Api()

            k8s_tests.append({
                "test_name": "Kubernetes Cluster Access",
                "status": "passed",
                "cluster_accessible": True
            })

            # Test namespace configuration
            try:
                namespaces = k8s_client.list_namespace()
                archon_namespace = any("archon" in ns.metadata.name.lower() for ns in namespaces.items)

                if archon_namespace:
                    k8s_tests.append({
                        "test_name": "Kubernetes Namespace",
                        "status": "passed",
                        "namespace_exists": True
                    })
                else:
                    k8s_tests.append({
                        "test_name": "Kubernetes Namespace",
                        "status": "warning",
                        "message": "Archon namespace not found (will be created)"
                    })
            except Exception as e:
                k8s_tests.append({
                    "test_name": "Kubernetes Namespace",
                    "status": "failed",
                    "error": str(e)
                })

            # Test deployment manifests
            deployment_files = [
                "kubernetes/deployment.yaml",
                "kubernetes/service.yaml",
                "kubernetes/configmap.yaml",
                "kubernetes/secret.yaml"
            ]

            for deployment_file in deployment_files:
                try:
                    file_path = Path(f"../deployment/{deployment_file}")
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            manifest = yaml.safe_load(f)

                        k8s_tests.append({
                            "test_name": f"Deployment Manifest - {deployment_file}",
                            "status": "passed",
                            "manifest_valid": True,
                            "manifest_type": manifest.get("kind", "Unknown")
                        })
                    else:
                        k8s_tests.append({
                            "test_name": f"Deployment Manifest - {deployment_file}",
                            "status": "warning",
                            "message": f"Manifest file not found: {deployment_file}"
                        })
                except Exception as e:
                    k8s_tests.append({
                        "test_name": f"Deployment Manifest - {deployment_file}",
                        "status": "failed",
                        "error": str(e)
                    })

            # Test pod resource limits
            try:
                # Load deployment manifest
                deployment_path = Path("../deployment/kubernetes/deployment.yaml")
                if deployment_path.exists():
                    with open(deployment_path, 'r') as f:
                        deployment = yaml.safe_load(f)

                    # Check resource limits
                    containers = deployment.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
                    resource_limits_configured = all(
                        container.get("resources", {}).get("limits") for container in containers
                    )

                    k8s_tests.append({
                        "test_name": "Pod Resource Limits",
                        "status": "passed" if resource_limits_configured else "failed",
                        "resource_limits_configured": resource_limits_configured
                    })
            except Exception as e:
                k8s_tests.append({
                    "test_name": "Pod Resource Limits",
                    "status": "failed",
                    "error": str(e)
                })

        except Exception as e:
            k8s_tests.append({
                "test_name": "Kubernetes Deployment",
                "status": "failed",
                "error": f"Kubernetes not available: {e}"
            })

        return {
            "test_name": "Kubernetes Deployment",
            "tests": k8s_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in k8s_tests
            ) else "needs_attention"
        }

    async def test_environment_configuration(self):
        """Test environment configuration and secrets"""
        logger.info("Testing environment configuration...")

        env_tests = []

        # Test required environment variables
        required_env_vars = [
            "SUPABASE_URL",
            "SUPABASE_SERVICE_KEY",
            "OPENAI_API_KEY",
            "LOG_LEVEL"
        ]

        for env_var in required_env_vars:
            try:
                # Check if environment variable is set
                import os
                env_value = os.getenv(env_var)

                if env_value:
                    env_tests.append({
                        "test_name": f"Environment Variable - {env_var}",
                        "status": "passed",
                        "variable_set": True,
                        "has_value": len(env_value) > 0
                    })
                else:
                    env_tests.append({
                        "test_name": f"Environment Variable - {env_var}",
                        "status": "failed",
                        "variable_set": False,
                        "message": f"Required environment variable {env_var} not set"
                    })
            except Exception as e:
                env_tests.append({
                    "test_name": f"Environment Variable - {env_var}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test configuration file validation
        config_files = [
            ".env",
            ".env.production",
            "config.yaml",
            "config/production.yaml"
        ]

        for config_file in config_files:
            try:
                config_path = Path(f"../{config_file}")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_content = f.read()

                    # Basic validation
                    if len(config_content) > 0:
                        env_tests.append({
                            "test_name": f"Configuration File - {config_file}",
                            "status": "passed",
                            "file_exists": True,
                            "file_size": len(config_content)
                        })
                    else:
                        env_tests.append({
                            "test_name": f"Configuration File - {config_file}",
                            "status": "failed",
                            "error": "Configuration file is empty"
                        })
                else:
                    env_tests.append({
                        "test_name": f"Configuration File - {config_file}",
                        "status": "warning",
                        "message": f"Configuration file not found: {config_file}"
                    })
            except Exception as e:
                env_tests.append({
                    "test_name": f"Configuration File - {config_file}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test secret management
        try:
            secret_tests = []
            secret_files = [
                "secrets.yaml",
                "kubernetes/secret.yaml",
                ".env.secrets"
            ]

            for secret_file in secret_files:
                secret_path = Path(f"../deployment/{secret_file}")
                if secret_path.exists():
                    with open(secret_path, 'r') as f:
                        secret_content = f.read()

                    # Check if secrets are encrypted or properly secured
                    if "ENC[AES256_GCM" in secret_content or "password" not in secret_content.lower():
                        secret_tests.append({
                            "file": secret_file,
                            "status": "passed",
                            "secrets_secured": True
                        })
                    else:
                        secret_tests.append({
                            "file": secret_file,
                            "status": "failed",
                            "secrets_secured": False,
                            "issue": "Secrets may be stored in plaintext"
                        })

            env_tests.append({
                "test_name": "Secret Management",
                "status": "passed" if all(test["status"] == "passed" for test in secret_tests) else "failed",
                "secret_tests": secret_tests
            })
        except Exception as e:
            env_tests.append({
                "test_name": "Secret Management",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Environment Configuration",
            "tests": env_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in env_tests
            ) else "needs_attention"
        }

    async def test_service_availability(self):
        """Test service availability after deployment"""
        logger.info("Testing service availability...")

        availability_tests = []

        for service_name, service_url in self.services.items():
            try:
                # Test HTTP/HTTPS availability
                response = requests.get(f"{service_url}/health", timeout=10)
                if response.status_code == 200:
                    availability_tests.append({
                        "service": service_name,
                        "status": "available",
                        "response_time": response.elapsed.total_seconds() * 1000,
                        "health_check": "passed"
                    })
                else:
                    availability_tests.append({
                        "service": service_name,
                        "status": "unavailable",
                        "error": f"HTTP {response.status_code}"
                    })
            except requests.exceptions.ConnectionError:
                availability_tests.append({
                    "service": service_name,
                    "status": "unavailable",
                    "error": "Connection refused"
                })
            except requests.exceptions.Timeout:
                availability_tests.append({
                    "service": service_name,
                    "status": "unavailable",
                    "error": "Request timeout"
                })
            except Exception as e:
                availability_tests.append({
                    "service": service_name,
                    "status": "unavailable",
                    "error": str(e)
                })

        # Test database connectivity
        try:
            import os
            supabase_url = os.getenv("SUPABASE_URL")
            if supabase_url:
                response = requests.get(f"{supabase_url}/health", timeout=5)
                if response.status_code == 200:
                    availability_tests.append({
                        "service": "database",
                        "status": "available",
                        "health_check": "passed"
                    })
                else:
                    availability_tests.append({
                        "service": "database",
                        "status": "unavailable",
                        "error": f"HTTP {response.status_code}"
                    })
            else:
                availability_tests.append({
                    "service": "database",
                    "status": "unavailable",
                    "error": "SUPABASE_URL not configured"
                })
        except Exception as e:
            availability_tests.append({
                "service": "database",
                "status": "unavailable",
                "error": str(e)
            })

        return {
            "test_name": "Service Availability",
            "tests": availability_tests,
            "overall_status": "passed" if all(
                test["status"] == "available" for test in availability_tests
            ) else "failed"
        }

    async def test_scaling_configuration(self):
        """Test scaling and load balancing configuration"""
        logger.info("Testing scaling configuration...")

        scaling_tests = []

        # Test Docker Compose scaling
        try:
            if self.deployment_config:
                services = self.deployment_config.get("services", {})
                scalable_services = []

                for service_name, service_config in services.items():
                    deploy_config = service_config.get("deploy", {})
                    replicas = deploy_config.get("replicas", 1)

                    if replicas > 1:
                        scalable_services.append({
                            "service": service_name,
                            "replicas": replicas
                        })

                scaling_tests.append({
                    "test_name": "Docker Scaling Configuration",
                    "status": "passed" if scalable_services else "warning",
                    "scalable_services": scalable_services,
                    "message": "No services configured for scaling" if not scalable_services else "Scaling configured"
                })
        except Exception as e:
            scaling_tests.append({
                "test_name": "Docker Scaling Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test Kubernetes Horizontal Pod Autoscaler
        try:
            hpa_path = Path("../deployment/kubernetes/hpa.yaml")
            if hpa_path.exists():
                with open(hpa_path, 'r') as f:
                    hpa_config = yaml.safe_load(f)

                scaling_tests.append({
                    "test_name": "Kubernetes HPA Configuration",
                    "status": "passed",
                    "hpa_configured": True,
                    "target_metrics": hpa_config.get("spec", {}).get("metrics", [])
                })
            else:
                scaling_tests.append({
                    "test_name": "Kubernetes HPA Configuration",
                    "status": "warning",
                    "hpa_configured": False,
                    "message": "HPA configuration not found"
                })
        except Exception as e:
            scaling_tests.append({
                "test_name": "Kubernetes HPA Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test load balancing
        try:
            lb_path = Path("../deployment/kubernetes/service.yaml")
            if lb_path.exists():
                with open(lb_path, 'r') as f:
                    service_config = yaml.safe_load(f)

                service_type = service_config.get("spec", {}).get("type", "")
                is_load_balanced = service_type == "LoadBalancer" or service_type == "NodePort"

                scaling_tests.append({
                    "test_name": "Load Balancing Configuration",
                    "status": "passed" if is_load_balanced else "warning",
                    "load_balanced": is_load_balanced,
                    "service_type": service_type
                })
            else:
                scaling_tests.append({
                    "test_name": "Load Balancing Configuration",
                    "status": "warning",
                    "load_balanced": False,
                    "message": "Service configuration not found"
                })
        except Exception as e:
            scaling_tests.append({
                "test_name": "Load Balancing Configuration",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Scaling Configuration",
            "tests": scaling_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in scaling_tests
            ) else "needs_attention"
        }

    async def test_backup_and_recovery(self):
        """Test backup and recovery configuration"""
        logger.info("Testing backup and recovery configuration...")

        backup_tests = []

        # Test database backup configuration
        try:
            backup_config_path = Path("../deployment/backup/backup-config.yaml")
            if backup_config_path.exists():
                with open(backup_config_path, 'r') as f:
                    backup_config = yaml.safe_load(f)

                schedule = backup_config.get("schedule", "")
                retention = backup_config.get("retention", "")

                backup_tests.append({
                    "test_name": "Database Backup Configuration",
                    "status": "passed" if schedule and retention else "failed",
                    "backup_scheduled": bool(schedule),
                    "retention_configured": bool(retention),
                    "schedule": schedule,
                    "retention_days": retention
                })
            else:
                backup_tests.append({
                    "test_name": "Database Backup Configuration",
                    "status": "warning",
                    "backup_configured": False,
                    "message": "Backup configuration not found"
                })
        except Exception as e:
            backup_tests.append({
                "test_name": "Database Backup Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test volume snapshot configuration
        try:
            snapshot_path = Path("../deployment/kubernetes/snapshot.yaml")
            if snapshot_path.exists():
                with open(snapshot_path, 'r') as f:
                    snapshot_config = yaml.safe_load(f)

                backup_tests.append({
                    "test_name": "Volume Snapshot Configuration",
                    "status": "passed",
                    "snapshots_configured": True,
                    "snapshot_class": snapshot_config.get("spec", {}).get("volumeSnapshotClassName")
                })
            else:
                backup_tests.append({
                    "test_name": "Volume Snapshot Configuration",
                    "status": "warning",
                    "snapshots_configured": False,
                    "message": "Snapshot configuration not found"
                })
        except Exception as e:
            backup_tests.append({
                "test_name": "Volume Snapshot Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test disaster recovery procedures
        try:
            dr_path = Path("../deployment/disaster-recovery.yaml")
            if dr_path.exists():
                with open(dr_path, 'r') as f:
                    dr_config = yaml.safe_load(f)

                rpo = dr_config.get("recovery_point_objective", "")
                rto = dr_config.get("recovery_time_objective", "")

                backup_tests.append({
                    "test_name": "Disaster Recovery Procedures",
                    "status": "passed" if rpo and rto else "failed",
                    "rpo_configured": bool(rpo),
                    "rto_configured": bool(rto),
                    "recovery_point_objective": rpo,
                    "recovery_time_objective": rto
                })
            else:
                backup_tests.append({
                    "test_name": "Disaster Recovery Procedures",
                    "status": "warning",
                    "dr_configured": False,
                    "message": "Disaster recovery configuration not found"
                })
        except Exception as e:
            backup_tests.append({
                "test_name": "Disaster Recovery Procedures",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Backup and Recovery",
            "tests": backup_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in backup_tests
            ) else "needs_attention"
        }

    async def test_monitoring_and_logging(self):
        """Test monitoring and logging configuration"""
        logger.info("Testing monitoring and logging configuration...")

        monitoring_tests = []

        # Test Prometheus configuration
        try:
            prometheus_path = Path("../deployment/monitoring/prometheus.yaml")
            if prometheus_path.exists():
                with open(prometheus_path, 'r') as f:
                    prometheus_config = yaml.safe_load(f)

                monitoring_tests.append({
                    "test_name": "Prometheus Configuration",
                    "status": "passed",
                    "prometheus_configured": True,
                    "scrape_configs": len(prometheus_config.get("scrape_configs", []))
                })
            else:
                monitoring_tests.append({
                    "test_name": "Prometheus Configuration",
                    "status": "warning",
                    "prometheus_configured": False,
                    "message": "Prometheus configuration not found"
                })
        except Exception as e:
            monitoring_tests.append({
                "test_name": "Prometheus Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test Grafana dashboards
        try:
            grafana_path = Path("../deployment/monitoring/grafana-dashboards")
            if grafana_path.exists():
                dashboard_files = list(grafana_path.glob("*.json"))
                monitoring_tests.append({
                    "test_name": "Grafana Dashboards",
                    "status": "passed",
                    "dashboards_configured": True,
                    "dashboard_count": len(dashboard_files)
                })
            else:
                monitoring_tests.append({
                    "test_name": "Grafana Dashboards",
                    "status": "warning",
                    "dashboards_configured": False,
                    "message": "Grafana dashboards not found"
                })
        except Exception as e:
            monitoring_tests.append({
                "test_name": "Grafana Dashboards",
                "status": "failed",
                "error": str(e)
            })

        # Test ELK stack configuration
        try:
            elk_path = Path("../deployment/logging/elk.yaml")
            if elk_path.exists():
                with open(elk_path, 'r') as f:
                    elk_config = yaml.safe_load(f)

                monitoring_tests.append({
                    "test_name": "ELK Stack Configuration",
                    "status": "passed",
                    "elk_configured": True,
                    "components": ["elasticsearch", "logstash", "kibana"]
                })
            else:
                monitoring_tests.append({
                    "test_name": "ELK Stack Configuration",
                    "status": "warning",
                    "elk_configured": False,
                    "message": "ELK stack configuration not found"
                })
        except Exception as e:
            monitoring_tests.append({
                "test_name": "ELK Stack Configuration",
                "status": "failed",
                "error": str(e)
            })

        # Test alerting configuration
        try:
            alerting_path = Path("../deployment/monitoring/alerting.yaml")
            if alerting_path.exists():
                with open(alerting_path, 'r') as f:
                    alerting_config = yaml.safe_load(f)

                alert_rules = alerting_config.get("groups", [])
                monitoring_tests.append({
                    "test_name": "Alerting Configuration",
                    "status": "passed",
                    "alerting_configured": True,
                    "alert_rules_count": len(alert_rules)
                })
            else:
                monitoring_tests.append({
                    "test_name": "Alerting Configuration",
                    "status": "warning",
                    "alerting_configured": False,
                    "message": "Alerting configuration not found"
                })
        except Exception as e:
            monitoring_tests.append({
                "test_name": "Alerting Configuration",
                "status": "failed",
                "error": str(e)
            })

        return {
            "test_name": "Monitoring and Logging",
            "tests": monitoring_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in monitoring_tests
            ) else "needs_attention"
        }

    async def test_deployment_automation(self):
        """Test deployment automation scripts"""
        logger.info("Testing deployment automation...")

        automation_tests = []

        # Test deployment scripts
        deployment_scripts = [
            "deploy.sh",
            "deploy.py",
            "deploy.ps1",
            "terraform/deploy.sh",
            "ansible/deploy.yml"
        ]

        for script in deployment_scripts:
            try:
                script_path = Path(f"../deployment/{script}")
                if script_path.exists():
                    # Check if script is executable
                    is_executable = script_path.stat().st_mode & 0o111

                    automation_tests.append({
                        "test_name": f"Deployment Script - {script}",
                        "status": "passed",
                        "script_exists": True,
                        "executable": is_executable
                    })
                else:
                    automation_tests.append({
                        "test_name": f"Deployment Script - {script}",
                        "status": "warning",
                        "script_exists": False,
                        "message": f"Deployment script not found: {script}"
                    })
            except Exception as e:
                automation_tests.append({
                    "test_name": f"Deployment Script - {script}",
                    "status": "failed",
                    "error": str(e)
                })

        # Test CI/CD pipeline configuration
        ci_cd_files = [
            ".github/workflows/deploy.yml",
            ".github/workflows/ci.yml",
            "jenkins/Jenkinsfile",
            "gitlab-ci.yml"
        ]

        for ci_cd_file in ci_cd_files:
            try:
                ci_cd_path = Path(f"../{ci_cd_file}")
                if ci_cd_path.exists():
                    with open(ci_cd_path, 'r') as f:
                        ci_cd_content = f.read()

                    automation_tests.append({
                        "test_name": f"CI/CD Configuration - {ci_cd_file}",
                        "status": "passed",
                        "ci_cd_configured": True,
                        "file_size": len(ci_cd_content)
                    })
                else:
                    automation_tests.append({
                        "test_name": f"CI/CD Configuration - {ci_cd_file}",
                        "status": "warning",
                        "ci_cd_configured": False,
                        "message": f"CI/CD configuration not found: {ci_cd_file}"
                    })
            except Exception as e:
                automation_tests.append({
                    "test_name": f"CI/CD Configuration - {ci_cd_file}",
                    "status": "failed",
                    "error": str(e)
                })

        return {
            "test_name": "Deployment Automation",
            "tests": automation_tests,
            "overall_status": "passed" if all(
                test.get("status") == "passed" for test in automation_tests
            ) else "needs_attention"
        }

    async def run_complete_deployment_test(self):
        """Run complete deployment test suite"""
        logger.info("Starting complete deployment test...")

        # Run all deployment tests
        test_functions = [
            self.test_docker_deployment,
            self.test_kubernetes_deployment,
            self.test_environment_configuration,
            self.test_service_availability,
            self.test_scaling_configuration,
            self.test_backup_and_recovery,
            self.test_monitoring_and_logging,
            self.test_deployment_automation
        ]

        for test_func in test_functions:
            try:
                result = await test_func()
                self.deployment_results[result["test_name"]] = result
                logger.info(f"✓ {test_func.__name__}: {result['overall_status']}")
            except Exception as e:
                logger.error(f"✗ {test_func.__name__} failed: {e}")

        # Generate final report
        return self.generate_deployment_report()

    def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        timestamp = datetime.now().isoformat()

        # Calculate deployment readiness score
        total_tests = len(self.deployment_results)
        passed_tests = sum(1 for result in self.deployment_results.values() if result["overall_status"] == "passed")
        needs_attention = sum(1 for result in self.deployment_results.values() if result["overall_status"] == "needs_attention")
        failed_tests = sum(1 for result in self.deployment_results.values() if result["overall_status"] == "failed")

        deployment_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        report = {
            "test_suite": "Agency Swarm Deployment Test",
            "timestamp": timestamp,
            "overall_assessment": {
                "deployment_score": deployment_score,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "needs_attention": needs_attention,
                "failed_tests": failed_tests,
                "deployment_ready": deployment_score >= 80,
                "deployment_grade": self.calculate_deployment_grade(deployment_score)
            },
            "deployment_results": self.deployment_results,
            "recommendations": self.generate_deployment_recommendations(),
            "deployment_checklist": self.generate_deployment_checklist()
        }

        # Save report
        report_path = Path("agency_swarm_deployment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Deployment report saved to {report_path}")
        return report

    def calculate_deployment_grade(self, score):
        """Calculate deployment grade based on score"""
        if score >= 90:
            return "A+ (Production Ready)"
        elif score >= 80:
            return "A (Ready for Production)"
        elif score >= 70:
            return "B (Ready with Minor Issues)"
        elif score >= 60:
            return "C (Needs Attention)"
        else:
            return "F (Not Ready)"

    def generate_deployment_recommendations(self):
        """Generate deployment recommendations"""
        recommendations = []

        for test_name, result in self.deployment_results.items():
            if result["overall_status"] == "failed":
                if "Docker" in test_name:
                    recommendations.append({
                        "priority": "High",
                        "category": "Docker Configuration",
                        "issue": "Docker deployment configuration failed",
                        "recommendation": "Fix Docker Compose configuration, container images, and networking"
                    })
                elif "Kubernetes" in test_name:
                    recommendations.append({
                        "priority": "High",
                        "category": "Kubernetes Configuration",
                        "issue": "Kubernetes deployment configuration failed",
                        "recommendation": "Fix Kubernetes manifests, namespace configuration, and resource limits"
                    })
                elif "Environment" in test_name:
                    recommendations.append({
                        "priority": "Critical",
                        "category": "Environment Configuration",
                        "issue": "Environment configuration failed",
                        "recommendation": "Configure required environment variables and secret management"
                    })
                elif "Service Availability" in test_name:
                    recommendations.append({
                        "priority": "Critical",
                        "category": "Service Availability",
                        "issue": "Services are not available",
                        "recommendation": "Start services and check health endpoints"
                    })

            elif result["overall_status"] == "needs_attention":
                recommendations.append({
                    "priority": "Medium",
                    "category": test_name,
                    "issue": "Configuration needs attention",
                    "recommendation": f"Review and improve {test_name} configuration"
                })

        return recommendations

    def generate_deployment_checklist(self):
        """Generate deployment checklist"""
        checklist = []

        # Docker checklist
        checklist.extend([
            {"category": "Docker", "item": "Docker service is running", "completed": False},
            {"category": "Docker", "item": "Docker Compose configuration is valid", "completed": False},
            {"category": "Docker", "item": "All container images are available", "completed": False},
            {"category": "Docker", "item": "Network configuration is correct", "completed": False},
            {"category": "Docker", "item": "Volume configuration is correct", "completed": False}
        ])

        # Kubernetes checklist
        checklist.extend([
            {"category": "Kubernetes", "item": "Cluster is accessible", "completed": False},
            {"category": "Kubernetes", "item": "Namespace exists or will be created", "completed": False},
            {"category": "Kubernetes", "item": "Deployment manifests are valid", "completed": False},
            {"category": "Kubernetes", "item": "Resource limits are configured", "completed": False},
            {"category": "Kubernetes", "item": "HPA is configured", "completed": False}
        ])

        # Configuration checklist
        checklist.extend([
            {"category": "Configuration", "item": "Required environment variables are set", "completed": False},
            {"category": "Configuration", "item": "Secrets are properly secured", "completed": False},
            {"category": "Configuration", "item": "Configuration files are valid", "completed": False}
        ])

        # Service checklist
        checklist.extend([
            {"category": "Services", "item": "Frontend service is available", "completed": False},
            {"category": "Services", "item": "API service is available", "completed": False},
            {"category": "Services", "item": "MCP service is available", "completed": False},
            {"category": "Services", "item": "Agents service is available", "completed": False},
            {"category": "Services", "item": "Database is accessible", "completed": False}
        ])

        # Update checklist based on test results
        for test_name, result in self.deployment_results.items():
            if result["overall_status"] == "passed":
                # Mark relevant checklist items as completed
                for item in checklist:
                    if (test_name in item["category"] or
                        any(keyword in item["item"].lower() for keyword in test_name.lower().split())):
                        item["completed"] = True

        return checklist

async def main():
    """Main function to run deployment tests"""
    tester = DeploymentTester()
    return await tester.run_complete_deployment_test()

if __name__ == "__main__":
    asyncio.run(main())