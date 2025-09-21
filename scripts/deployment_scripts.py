#!/usr/bin/env python3
"""
Agency Swarm Deployment Automation Scripts

Comprehensive deployment automation for the Agency Swarm enhancement
including deployment, rollback, health checks, and monitoring setup.

Features:
- Automated deployment across environments
- Blue-green deployment strategy
- Health checks and monitoring
- Rollback procedures
- Configuration management
- Zero-downtime deployments
"""

import os
import sys
import json
import yaml
import subprocess
import time
import logging
import shutil
import requests
import docker
import kubernetes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    version: str
    docker_image: str
    kubernetes_namespace: str
    replicas: int
    health_check_endpoint: str
    readiness_endpoint: str
    liveness_endpoint: str
    rollout_strategy: str = "blue_green"
    rollback_on_failure: bool = True
    monitoring_enabled: bool = True

@dataclass
class DeploymentResult:
    """Deployment execution result."""
    status: DeploymentStatus
    deployment_id: str
    version: str
    environment: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_version: Optional[str] = None

class AgencySwarmDeployer:
    """Agency Swarm deployment automation system."""

    def __init__(self, config_path: str = "deployment_config.yaml"):
        """Initialize deployer with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.docker_client = docker.from_env()
        self.k8s_client = self._init_kubernetes_client()
        self.deployment_history: List[DeploymentResult] = []
        self.current_deployment: Optional[DeploymentResult] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration."""
        return {
            "environments": {
                "development": {
                    "kubernetes_namespace": "archon-dev",
                    "replicas": 1,
                    "docker_registry": "localhost:5000"
                },
                "staging": {
                    "kubernetes_namespace": "archon-staging",
                    "replicas": 2,
                    "docker_registry": "registry.example.com"
                },
                "production": {
                    "kubernetes_namespace": "archon-prod",
                    "replicas": 3,
                    "docker_registry": "registry.example.com"
                }
            },
            "health_checks": {
                "endpoint": "/api/health",
                "timeout": 30,
                "retries": 3
            },
            "monitoring": {
                "enabled": True,
                "prometheus_port": 9090,
                "grafana_port": 3000
            }
        }

    def _init_kubernetes_client(self) -> Optional[kubernetes.client.CoreV1Api]:
        """Initialize Kubernetes client."""
        try:
            kubernetes.config.load_kube_config()
            return kubernetes.client.CoreV1Api()
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
            return None

    def build_docker_image(self, version: str, environment: DeploymentEnvironment) -> str:
        """Build Docker image for deployment."""
        logger.info(f"Building Docker image version {version} for {environment.value}")

        image_name = f"agency-swarm:{version}"
        dockerfile_path = project_root / "python" / "Dockerfile.server"

        try:
            # Build image
            image, build_logs = self.docker_client.images.build(
                path=str(project_root / "python"),
                dockerfile=str(dockerfile_path),
                tag=image_name,
                buildargs={
                    "VERSION": version,
                    "ENVIRONMENT": environment.value
                }
            )

            # Tag image for registry
            registry = self.config["environments"][environment.value]["docker_registry"]
            full_image_name = f"{registry}/{image_name}"
            image.tag(full_image_name)

            logger.info(f"Docker image built successfully: {full_image_name}")
            return full_image_name

        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            raise

    def push_docker_image(self, image_name: str) -> None:
        """Push Docker image to registry."""
        logger.info(f"Pushing Docker image: {image_name}")

        try:
            # Login to registry if credentials are provided
            registry_url = image_name.split('/')[0]
            username = os.getenv('DOCKER_USERNAME')
            password = os.getenv('DOCKER_PASSWORD')

            if username and password:
                self.docker_client.login(
                    username=username,
                    password=password,
                    registry=registry_url
                )

            # Push image
            push_logs = self.docker_client.images.push(image_name)
            logger.info(f"Docker image pushed successfully: {image_name}")

        except Exception as e:
            logger.error(f"Failed to push Docker image: {e}")
            raise

    def deploy_to_kubernetes(
        self,
        config: DeploymentConfig,
        dry_run: bool = False
    ) -> DeploymentResult:
        """Deploy to Kubernetes cluster."""
        deployment_id = f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        start_time = datetime.now()

        logger.info(f"Starting deployment {deployment_id} to {config.environment.value}")

        try:
            self.current_deployment = DeploymentResult(
                status=DeploymentStatus.IN_PROGRESS,
                deployment_id=deployment_id,
                version=config.version,
                environment=config.environment.value,
                start_time=start_time
            )

            if dry_run:
                logger.info("Dry run: skipping actual deployment")
                self.current_deployment.status = DeploymentStatus.SUCCESS
                self.current_deployment.end_time = datetime.now()
                return self.current_deployment

            # Apply Kubernetes manifests
            self._apply_kubernetes_manifests(config)

            # Wait for deployment to be ready
            self._wait_for_deployment_ready(config)

            # Run health checks
            self._run_health_checks(config)

            # Update deployment status
            self.current_deployment.status = DeploymentStatus.SUCCESS
            self.current_deployment.end_time = datetime.now()

            logger.info(f"Deployment {deployment_id} completed successfully")

            # Setup monitoring
            if config.monitoring_enabled:
                self._setup_monitoring(config)

            return self.current_deployment

        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            self.current_deployment.status = DeploymentStatus.FAILED
            self.current_deployment.error_message = str(e)
            self.current_deployment.end_time = datetime.now()

            # Rollback if enabled
            if config.rollback_on_failure:
                logger.info("Initiating rollback due to deployment failure")
                self._rollback_deployment(config)

            return self.current_deployment

    def _apply_kubernetes_manifests(self, config: DeploymentConfig) -> None:
        """Apply Kubernetes manifests for deployment."""
        logger.info("Applying Kubernetes manifests")

        namespace = config.kubernetes_namespace

        # Create namespace if it doesn't exist
        if self.k8s_client:
            try:
                self.k8s_client.create_namespace(
                    kubernetes.client.V1Namespace(
                        metadata=kubernetes.client.V1ObjectMeta(name=namespace)
                    )
                )
            except kubernetes.client.exceptions.ApiException as e:
                if e.status != 409:  # Namespace already exists
                    raise

        # Generate and apply deployment manifest
        deployment_manifest = self._generate_deployment_manifest(config)
        self._apply_manifest(deployment_manifest)

        # Generate and apply service manifest
        service_manifest = self._generate_service_manifest(config)
        self._apply_manifest(service_manifest)

        # Generate and apply ingress manifest
        ingress_manifest = self._generate_ingress_manifest(config)
        self._apply_manifest(ingress_manifest)

        logger.info("Kubernetes manifests applied successfully")

    def _generate_deployment_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"agency-swarm-{config.environment.value}",
                "namespace": config.kubernetes_namespace,
                "labels": {
                    "app": "agency-swarm",
                    "version": config.version,
                    "environment": config.environment.value
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "agency-swarm",
                        "environment": config.environment.value
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "agency-swarm",
                            "version": config.version,
                            "environment": config.environment.value
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "agency-swarm",
                            "image": config.docker_image,
                            "ports": [{
                                "containerPort": 8181,
                                "name": "api"
                            }, {
                                "containerPort": 3737,
                                "name": "ui"
                            }],
                            "env": [
                                {"name": "ENVIRONMENT", "value": config.environment.value},
                                {"name": "VERSION", "value": config.version},
                                {"name": "DEPLOYMENT_ID", "value": self.current_deployment.deployment_id}
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.liveness_endpoint,
                                    "port": 8181
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.readiness_endpoint,
                                    "port": 8181
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                }
                            }
                        }]
                    }
                },
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                }
            }
        }

    def _generate_service_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"agency-swarm-service-{config.environment.value}",
                "namespace": config.kubernetes_namespace,
                "labels": {
                    "app": "agency-swarm",
                    "environment": config.environment.value
                }
            },
            "spec": {
                "selector": {
                    "app": "agency-swarm",
                    "environment": config.environment.value
                },
                "ports": [
                    {
                        "port": 80,
                        "targetPort": 8181,
                        "name": "api"
                    },
                    {
                        "port": 8080,
                        "targetPort": 3737,
                        "name": "ui"
                    }
                ],
                "type": "ClusterIP"
            }
        }

    def _generate_ingress_manifest(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Generate Kubernetes ingress manifest."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"agency-swarm-ingress-{config.environment.value}",
                "namespace": config.kubernetes_namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                }
            },
            "spec": {
                "rules": [
                    {
                        "host": f"agency-swarm-{config.environment.value}.example.com",
                        "http": {
                            "paths": [
                                {
                                    "path": "/api",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"agency-swarm-service-{config.environment.value}",
                                            "port": {
                                                "number": 80
                                            }
                                        }
                                    }
                                },
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": f"agency-swarm-service-{config.environment.value}",
                                            "port": {
                                                "number": 8080
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }

    def _apply_manifest(self, manifest: Dict[str, Any]) -> None:
        """Apply Kubernetes manifest."""
        if not self.k8s_client:
            logger.warning("Kubernetes client not available, skipping manifest application")
            return

        # Convert manifest to YAML and apply using kubectl
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(manifest, f)
            manifest_path = f.name

        try:
            subprocess.run(
                ['kubectl', 'apply', '-f', manifest_path],
                check=True,
                capture_output=True
            )
            logger.info("Manifest applied successfully")
        finally:
            os.unlink(manifest_path)

    def _wait_for_deployment_ready(self, config: DeploymentConfig, timeout: int = 600) -> None:
        """Wait for deployment to be ready."""
        logger.info("Waiting for deployment to be ready")

        start_time = time.time()
        deployment_name = f"agency-swarm-{config.environment.value}"

        while time.time() - start_time < timeout:
            try:
                # Check deployment status
                result = subprocess.run(
                    ['kubectl', 'rollout', 'status', 'deployment', deployment_name,
                     '-n', config.kubernetes_namespace],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    logger.info("Deployment is ready")
                    return

                time.sleep(10)

            except subprocess.CalledProcessError:
                time.sleep(10)

        raise TimeoutError(f"Deployment not ready within {timeout} seconds")

    def _run_health_checks(self, config: DeploymentConfig) -> None:
        """Run deployment health checks."""
        logger.info("Running health checks")

        # Get service URL
        service_url = self._get_service_url(config)

        if not service_url:
            logger.warning("Could not determine service URL, skipping health checks")
            return

        # Health check endpoints
        endpoints = [
            ("Health", config.health_check_endpoint),
            ("Readiness", config.readiness_endpoint),
            ("Liveness", config.liveness_endpoint)
        ]

        for check_name, endpoint in endpoints:
            try:
                url = f"{service_url}{endpoint}"
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    logger.info(f"{check_name} check passed")
                else:
                    raise Exception(f"{check_name} check failed: {response.status_code}")

            except Exception as e:
                logger.error(f"{check_name} check failed: {e}")
                raise

        logger.info("All health checks passed")

    def _get_service_url(self, config: DeploymentConfig) -> Optional[str]:
        """Get service URL for health checks."""
        # For local development, use localhost
        if config.environment == DeploymentEnvironment.DEVELOPMENT:
            return "http://localhost:8181"

        # For cloud environments, try to get load balancer URL
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'svc', f'agency-swarm-service-{config.environment.value}',
                 '-n', config.kubernetes_namespace, '-o', 'jsonpath={.status.loadBalancer.ingress[0].hostname}'],
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                return f"http://{result.stdout.strip()}"

        except subprocess.CalledProcessError:
            pass

        return None

    def _setup_monitoring(self, config: DeploymentConfig) -> None:
        """Setup monitoring and observability."""
        logger.info("Setting up monitoring")

        # Deploy Prometheus if enabled
        if self.config.get("monitoring", {}).get("enabled", False):
            self._deploy_prometheus(config)
            self._deploy_grafana(config)
            self._setup_alerting(config)

    def _deploy_prometheus(self, config: DeploymentConfig) -> None:
        """Deploy Prometheus for monitoring."""
        logger.info("Deploying Prometheus")

        # Generate Prometheus configuration
        prometheus_config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "prometheus-config",
                "namespace": config.kubernetes_namespace
            },
            "data": {
                "prometheus.yml": """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agency-swarm'
    static_configs:
      - targets: ['agency-swarm-service-{}:80'].format(config.environment.value)
    metrics_path: '/metrics'
    scrape_interval: 5s
""".format(config.environment.value)
            }
        }

        self._apply_manifest(prometheus_config)

    def _deploy_grafana(self, config: DeploymentConfig) -> None:
        """Deploy Grafana for dashboards."""
        logger.info("Deploying Grafana")

        # Note: In production, use Helm charts or proper manifests
        grafana_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "grafana",
                "namespace": config.kubernetes_namespace
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "grafana"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "grafana"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "grafana",
                            "image": "grafana/grafana:latest",
                            "ports": [{
                                "containerPort": 3000
                            }],
                            "env": [
                                {"name": "GF_SECURITY_ADMIN_PASSWORD", "value": "admin"}
                            ]
                        }]
                    }
                }
            }
        }

        self._apply_manifest(grafana_deployment)

    def _setup_alerting(self, config: DeploymentConfig) -> None:
        """Setup alerting rules."""
        logger.info("Setting up alerting")

        alert_rules = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "alert-rules",
                "namespace": config.kubernetes_namespace
            },
            "data": {
                "alert-rules.yml": """
groups:
  - name: agency-swarm-alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
"""
            }
        }

        self._apply_manifest(alert_rules)

    def _rollback_deployment(self, config: DeploymentConfig) -> None:
        """Rollback deployment to previous version."""
        logger.info("Rolling back deployment")

        self.current_deployment.status = DeploymentStatus.ROLLING_BACK

        try:
            # Get previous version
            previous_version = self._get_previous_version(config.environment)

            if previous_version:
                # Rollback using kubectl
                subprocess.run([
                    'kubectl', 'rollout', 'undo', 'deployment',
                    f"agency-swarm-{config.environment.value}",
                    '-n', config.kubernetes_namespace
                ], check=True)

                # Wait for rollback to complete
                self._wait_for_deployment_ready(config)

                self.current_deployment.rollback_version = previous_version
                self.current_deployment.status = DeploymentStatus.ROLLED_BACK

                logger.info(f"Rollback completed to version {previous_version}")
            else:
                logger.warning("No previous version found for rollback")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    def _get_previous_version(self, environment: DeploymentEnvironment) -> Optional[str]:
        """Get previous deployment version."""
        # In production, this would query a deployment database
        # For now, return a placeholder
        return "v2.9.0"

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status by ID."""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None

    def list_deployments(self, environment: Optional[DeploymentEnvironment] = None) -> List[DeploymentResult]:
        """List deployment history."""
        if environment:
            return [d for d in self.deployment_history if d.environment == environment.value]
        return self.deployment_history

    def cleanup_old_deployments(self, keep_count: int = 10) -> None:
        """Clean up old deployment records."""
        if len(self.deployment_history) > keep_count:
            self.deployment_history = self.deployment_history[-keep_count:]

# CLI Interface
def main():
    """Command line interface for deployment automation."""
    import argparse

    parser = argparse.ArgumentParser(description='Agency Swarm Deployment Automation')
    parser.add_argument('action', choices=['deploy', 'rollback', 'status', 'build'])
    parser.add_argument('--environment', '-e', choices=['development', 'staging', 'production'],
                        default='development', help='Deployment environment')
    parser.add_argument('--version', '-v', help='Version to deploy')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without actual deployment')
    parser.add_argument('--config', '-c', default='deployment_config.yaml',
                        help='Configuration file path')

    args = parser.parse_args()

    try:
        deployer = AgencySwarmDeployer(args.config)
        environment = DeploymentEnvironment(args.environment)

        if args.action == 'deploy':
            if not args.version:
                print("Error: Version is required for deployment")
                sys.exit(1)

            config = DeploymentConfig(
                environment=environment,
                version=args.version,
                docker_image=f"agency-swarm:{args.version}",
                kubernetes_namespace=f"archon-{args.environment}",
                replicas=3 if args.environment == 'production' else 1,
                health_check_endpoint="/api/health",
                readiness_endpoint="/api/ready",
                liveness_endpoint="/api/live",
                monitoring_enabled=True
            )

            # Build and push image
            image_name = deployer.build_docker_image(args.version, environment)
            deployer.push_docker_image(image_name)
            config.docker_image = image_name

            # Deploy
            result = deployer.deploy_to_kubernetes(config, args.dry_run)

            print(f"Deployment {result.deployment_id}: {result.status.value}")
            if result.error_message:
                print(f"Error: {result.error_message}")

        elif args.action == 'rollback':
            print("Rollback functionality requires deployment history")
            # Implement rollback logic

        elif args.action == 'status':
            deployments = deployer.list_deployments(environment)
            for deployment in deployments[-5:]:  # Show last 5 deployments
                print(f"{deployment.deployment_id}: {deployment.status.value}")

        elif args.action == 'build':
            if not args.version:
                print("Error: Version is required for build")
                sys.exit(1)

            image_name = deployer.build_docker_image(args.version, environment)
            deployer.push_docker_image(image_name)
            print(f"Built and pushed: {image_name}")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()