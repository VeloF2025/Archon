#!/usr/bin/env python3
"""
Deployment Automation Script for Agency Swarm
Automated deployment to Kubernetes clusters
"""

import asyncio
import argparse
import subprocess
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
import kubernetes
from kubernetes import client, config
import docker

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Automated deployment management system"""

    def __init__(self, environment="development"):
        self.environment = environment
        self.k8s_configured = False
        self.docker_client = None
        self.k8s_client = None
        self.apps_client = None
        self.deployment_config = self.load_deployment_config()

    def load_deployment_config(self):
        """Load deployment configuration"""
        config_path = Path("config/deployment_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                "environments": {
                    "development": {
                        "kubernetes_context": "minikube",
                        "namespace": "archon-dev",
                        "replicas": {
                            "frontend": 1,
                            "api": 1,
                            "mcp": 1,
                            "agents": 1
                        }
                    },
                    "staging": {
                        "kubernetes_context": "staging-cluster",
                        "namespace": "archon-staging",
                        "replicas": {
                            "frontend": 2,
                            "api": 2,
                            "mcp": 2,
                            "agents": 2
                        }
                    },
                    "production": {
                        "kubernetes_context": "production-cluster",
                        "namespace": "archon",
                        "replicas": {
                            "frontend": 3,
                            "api": 3,
                            "mcp": 2,
                            "agents": 2
                        }
                    }
                },
                "deployment_strategies": {
                    "rolling_update": {
                        "max_unavailable": "25%",
                        "max_surge": "25%"
                    },
                    "blue_green": {
                        "preview_replicas": 1,
                        "switch_traffic": False
                    }
                },
                "health_checks": {
                    "initial_delay_seconds": 30,
                    "period_seconds": 10,
                    "timeout_seconds": 5,
                    "success_threshold": 1,
                    "failure_threshold": 3
                }
            }

    def setup_kubernetes(self):
        """Setup Kubernetes client"""
        try:
            config.load_kube_config(context=self.deployment_config["environments"][self.environment]["kubernetes_context"])
            self.k8s_client = client.CoreV1Api()
            self.apps_client = client.AppsV1Api()
            self.k8s_configured = True
            logger.info(f"Kubernetes client configured for {self.environment}")
        except Exception as e:
            logger.error(f"Failed to setup Kubernetes client: {e}")
            raise

    def setup_docker(self):
        """Setup Docker client"""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client configured")
        except Exception as e:
            logger.error(f"Failed to setup Docker client: {e}")
            raise

    def build_docker_images(self):
        """Build Docker images for all services"""
        logger.info("Building Docker images...")

        images_to_build = [
            {
                "name": "archon-frontend",
                "context": "../../archon-ui-main",
                "dockerfile": "Dockerfile"
            },
            {
                "name": "archon-api",
                "context": "../../python",
                "dockerfile": "Dockerfile.server"
            },
            {
                "name": "archon-mcp",
                "context": "../../python",
                "dockerfile": "Dockerfile.mcp"
            },
            {
                "name": "archon-agents",
                "context": "../../python",
                "dockerfile": "Dockerfile.agents"
            }
        ]

        built_images = []
        for image_config in images_to_build:
            try:
                logger.info(f"Building {image_config['name']}...")
                image, build_logs = self.docker_client.images.build(
                    path=image_config["context"],
                    dockerfile=image_config["dockerfile"],
                    tag=f"{image_config['name']}:latest"
                )

                # Tag with environment
                image.tag(f"{image_config['name']}:{self.environment}")

                built_images.append({
                    "name": image_config["name"],
                    "id": image.id,
                    "tags": image.tags
                })

                logger.info(f"Successfully built {image_config['name']}: {image.id[:12]}")

            except Exception as e:
                logger.error(f"Failed to build {image_config['name']}: {e}")
                raise

        return built_images

    def push_docker_images(self, images):
        """Push Docker images to registry"""
        logger.info("Pushing Docker images to registry...")

        for image_info in images:
            try:
                image = self.docker_client.images.get(image_info["id"])
                for tag in image_info["tags"]:
                    if self.environment in tag:
                        logger.info(f"Pushing {tag}...")
                        self.docker_client.images.push(tag)
                        logger.info(f"Successfully pushed {tag}")

            except Exception as e:
                logger.error(f"Failed to push {image_info['name']}: {e}")
                raise

    def create_namespace(self):
        """Create Kubernetes namespace"""
        namespace_name = self.deployment_config["environments"][self.environment]["namespace"]

        try:
            # Check if namespace exists
            self.k8s_client.read_namespace(namespace_name)
            logger.info(f"Namespace {namespace_name} already exists")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": namespace_name,
                        "labels": {
                            "name": namespace_name,
                            "environment": self.environment
                        }
                    }
                }

                self.k8s_client.create_namespace(namespace_manifest)
                logger.info(f"Created namespace {namespace_name}")
            else:
                raise

    def create_secrets(self):
        """Create Kubernetes secrets"""
        secrets_config = {
            "archon-secrets": {
                "SUPABASE_URL": "",
                "SUPABASE_SERVICE_KEY": "",
                "OPENAI_API_KEY": "",
                "POSTGRES_PASSWORD": "change-this-password",
                "GRAFANA_PASSWORD": "admin-password"
            }
        }

        for secret_name, secret_data in secrets_config.items():
            try:
                # Check if secret exists
                self.k8s_client.read_namespaced_secret(secret_name, self.deployment_config["environments"][self.environment]["namespace"])
                logger.info(f"Secret {secret_name} already exists")
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    # Create secret
                    secret_manifest = {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": {
                            "name": secret_name,
                            "namespace": self.deployment_config["environments"][self.environment]["namespace"]
                        },
                        "type": "Opaque",
                        "data": {
                            key: value.encode().hex() for key, value in secret_data.items()
                        }
                    }

                    self.k8s_client.create_namespaced_secret(
                        body=secret_manifest,
                        namespace=self.deployment_config["environments"][self.environment]["namespace"]
                    )
                    logger.info(f"Created secret {secret_name}")
                else:
                    raise

    def create_configmap(self):
        """Create Kubernetes configmap"""
        config_data = {
            "ENVIRONMENT": self.environment,
            "LOG_LEVEL": "INFO",
            "NODE_ENV": "production",
            "REACT_APP_API_URL": "http://archon-api-service:8000",
            "REACT_APP_WS_URL": "ws://archon-api-service:8000"
        }

        configmap_name = "archon-config"

        try:
            # Check if configmap exists
            self.k8s_client.read_namespaced_config_map(configmap_name, self.deployment_config["environments"][self.environment]["namespace"])
            logger.info(f"ConfigMap {configmap_name} already exists")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                # Create configmap
                configmap_manifest = {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": configmap_name,
                        "namespace": self.deployment_config["environments"][self.environment]["namespace"]
                    },
                    "data": config_data
                }

                self.k8s_client.create_namespaced_config_map(
                    body=configmap_manifest,
                    namespace=self.deployment_config["environments"][self.environment]["namespace"]
                )
                logger.info(f"Created configmap {configmap_name}")
            else:
                raise

    def deploy_database(self):
        """Deploy database services"""
        # Deploy PostgreSQL
        postgres_deployment = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {
                "name": "postgres",
                "namespace": self.deployment_config["environments"][self.environment]["namespace"]
            },
            "spec": {
                "serviceName": "postgres-service",
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "postgres"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "postgres"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "postgres",
                            "image": "postgres:15",
                            "ports": [{"containerPort": 5432}],
                            "env": [
                                {"name": "POSTGRES_DB", "value": "archon"},
                                {"name": "POSTGRES_USER", "value": "postgres"},
                                {"name": "POSTGRES_PASSWORD", "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "archon-secrets",
                                        "key": "POSTGRES_PASSWORD"
                                    }
                                }}
                            ],
                            "volumeMounts": [{
                                "name": "postgres-storage",
                                "mountPath": "/var/lib/postgresql/data"
                            }],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2000m", "memory": "4Gi"}
                            }
                        }]
                    }
                },
                "volumeClaimTemplates": [{
                    "metadata": {"name": "postgres-storage"},
                    "spec": {
                        "accessModes": ["ReadWriteOnce"],
                        "resources": {"requests": {"storage": "20Gi"}}
                    }
                }]
            }
        }

        try:
            self.apps_client.create_namespaced_stateful_set(
                body=postgres_deployment,
                namespace=self.deployment_config["environments"][self.environment]["namespace"]
            )
            logger.info("Deployed PostgreSQL StatefulSet")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                logger.info("PostgreSQL StatefulSet already exists")
            else:
                raise

    def deploy_services(self):
        """Deploy application services"""
        services = [
            "archon-frontend",
            "archon-api",
            "archon-mcp",
            "archon-agents"
        ]

        for service_name in services:
            # Read deployment manifest
            manifest_path = Path(f"../kubernetes/{service_name}-deployment.yaml")
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    deployment_manifest = yaml.safe_load(f)

                # Update namespace and replicas
                deployment_manifest["metadata"]["namespace"] = self.deployment_config["environments"][self.environment]["namespace"]
                if "replicas" in self.deployment_config["environments"][self.environment]["replicas"]:
                    service_key = service_name.replace("archon-", "")
                    if service_key in self.deployment_config["environments"][self.environment]["replicas"]:
                        deployment_manifest["spec"]["replicas"] = self.deployment_config["environments"][self.environment]["replicas"][service_key]

                # Update image tag
                if "spec" in deployment_manifest and "template" in deployment_manifest["spec"]:
                    if "spec" in deployment_manifest["spec"]["template"] and "containers" in deployment_manifest["spec"]["template"]["spec"]:
                        for container in deployment_manifest["spec"]["template"]["spec"]["containers"]:
                            container["image"] = f"{container['image'].split(':')[0]}:{self.environment}"

                try:
                    self.apps_client.create_namespaced_deployment(
                        body=deployment_manifest,
                        namespace=self.deployment_config["environments"][self.environment]["namespace"]
                    )
                    logger.info(f"Deployed {service_name}")
                except client.exceptions.ApiException as e:
                    if e.status == 409:
                        logger.info(f"{service_name} deployment already exists")
                    else:
                        raise

    def wait_for_deployment(self, timeout_minutes=10):
        """Wait for deployments to be ready"""
        logger.info("Waiting for deployments to be ready...")

        namespace = self.deployment_config["environments"][self.environment]["namespace"]
        timeout = timeout_minutes * 60
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_ready = True

            deployments = self.apps_client.list_namespaced_deployment(namespace)
            for deployment in deployments.items:
                if deployment.status.ready_replicas != deployment.status.replicas:
                    all_ready = False
                    break

            if all_ready:
                logger.info("All deployments are ready")
                return True

            time.sleep(10)

        logger.error(f"Timeout waiting for deployments to be ready after {timeout_minutes} minutes")
        return False

    def run_health_checks(self):
        """Run health checks on deployed services"""
        logger.info("Running health checks...")

        namespace = self.deployment_config["environments"][self.environment]["namespace"]
        health_results = {}

        # Check pod status
        pods = self.k8s_client.list_namespaced_pod(namespace)
        for pod in pods.items:
            pod_health = {
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "ready": all(container.ready for container in pod.status.container_statuses or []),
                "restarts": sum(container.restart_count for container in pod.status.container_statuses or [])
            }
            health_results[pod.metadata.name] = pod_health

        # Check service endpoints
        services = self.k8s_client.list_namespaced_service(namespace)
        for service in services.items:
            try:
                endpoints = self.k8s_client.read_namespaced_endpoints(service.metadata.name, namespace)
                health_results[f"service-{service.metadata.name}"] = {
                    "ready": len(endpoints.subsets) > 0,
                    "endpoints": len(endpoints.subsets[0].addresses) if endpoints.subsets else 0
                }
            except Exception as e:
                health_results[f"service-{service.metadata.name}"] = {
                    "ready": False,
                    "error": str(e)
                }

        return health_results

    def rollback_deployment(self, deployment_name=None):
        """Rollback deployment to previous version"""
        logger.info(f"Rolling back deployment {deployment_name or 'all'}...")

        namespace = self.deployment_config["environments"][self.environment]["namespace"]

        if deployment_name:
            try:
                # Rollback specific deployment
                self.apps_client.rollback_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace,
                    body={"rollbackTo": {"revision": 0}}
                )
                logger.info(f"Rolled back {deployment_name}")
            except Exception as e:
                logger.error(f"Failed to rollback {deployment_name}: {e}")
                raise
        else:
            # Rollback all deployments
            deployments = self.apps_client.list_namespaced_deployment(namespace)
            for deployment in deployments.items:
                try:
                    self.apps_client.rollback_namespaced_deployment(
                        name=deployment.metadata.name,
                        namespace=namespace,
                        body={"rollbackTo": {"revision": 0}}
                    )
                    logger.info(f"Rolled back {deployment.metadata.name}")
                except Exception as e:
                    logger.error(f"Failed to rollback {deployment.metadata.name}: {e}")

    def backup_deployment(self):
        """Create backup of current deployment"""
        logger.info("Creating deployment backup...")

        namespace = self.deployment_config["environments"][self.environment]["namespace"]
        backup_dir = Path(f"backups/{self.environment}/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Backup deployments
        deployments = self.apps_client.list_namespaced_deployment(namespace)
        for deployment in deployments.items:
            backup_file = backup_dir / f"{deployment.metadata.name}-deployment.yaml"
            with open(backup_file, 'w') as f:
                yaml.dump(deployment.to_dict(), f)

        # Backup services
        services = self.k8s_client.list_namespaced_service(namespace)
        for service in services.items:
            backup_file = backup_dir / f"{service.metadata.name}-service.yaml"
            with open(backup_file, 'w') as f:
                yaml.dump(service.to_dict(), f)

        # Backup configmaps and secrets
        configmaps = self.k8s_client.list_namespaced_config_map(namespace)
        for configmap in configmaps.items:
            backup_file = backup_dir / f"{configmap.metadata.name}-configmap.yaml"
            with open(backup_file, 'w') as f:
                yaml.dump(configmap.to_dict(), f)

        secrets = self.k8s_client.list_namespaced_secret(namespace)
        for secret in secrets.items:
            backup_file = backup_dir / f"{secret.metadata.name}-secret.yaml"
            with open(backup_file, 'w') as f:
                yaml.dump(secret.to_dict(), f)

        logger.info(f"Backup created at {backup_dir}")
        return str(backup_dir)

    def deploy(self, strategy="rolling_update"):
        """Deploy to Kubernetes cluster"""
        logger.info(f"Starting deployment to {self.environment}...")

        # Setup clients
        self.setup_kubernetes()
        self.setup_docker()

        deployment_steps = [
            ("Building Docker images", self.build_docker_images),
            ("Pushing Docker images", lambda: self.push_docker_images(self.build_docker_images())),
            ("Creating namespace", self.create_namespace),
            ("Creating secrets", self.create_secrets),
            ("Creating configmap", self.create_configmap),
            ("Deploying database", self.deploy_database),
            ("Deploying services", self.deploy_services),
            ("Waiting for deployment", self.wait_for_deployment),
            ("Running health checks", self.run_health_checks)
        ]

        for step_name, step_function in deployment_steps:
            try:
                logger.info(f"Executing: {step_name}")
                result = step_function()
                logger.info(f"Completed: {step_name}")
            except Exception as e:
                logger.error(f"Failed at {step_name}: {e}")
                logger.info("Rolling back deployment...")
                self.rollback_deployment()
                raise

        logger.info(f"Deployment to {self.environment} completed successfully!")

        # Create backup
        backup_path = self.backup_deployment()
        logger.info(f"Deployment backup created at {backup_path}")

        return True

    def scale_deployment(self, service_name, replicas):
        """Scale deployment to specified number of replicas"""
        logger.info(f"Scaling {service_name} to {replicas} replicas...")

        namespace = self.deployment_config["environments"][self.environment]["namespace"]

        try:
            # Get current deployment
            deployment = self.apps_client.read_namespaced_deployment(service_name, namespace)

            # Update replicas
            deployment.spec.replicas = replicas

            # Update deployment
            self.apps_client.patch_namespaced_deployment(
                name=service_name,
                namespace=namespace,
                body=deployment
            )

            logger.info(f"Scaled {service_name} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Failed to scale {service_name}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Agency Swarm Deployment Manager")
    parser.add_argument("--environment", choices=["development", "staging", "production"], default="development",
                        help="Target environment for deployment")
    parser.add_argument("--action", choices=["deploy", "scale", "rollback", "backup", "health"], default="deploy",
                        help="Action to perform")
    parser.add_argument("--service", help="Service name for scale/rollback actions")
    parser.add_argument("--replicas", type=int, help="Number of replicas for scale action")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    deployer = DeploymentManager(args.environment)

    try:
        if args.action == "deploy":
            success = deployer.deploy()
            print(f"Deployment {'succeeded' if success else 'failed'}")
        elif args.action == "scale":
            if args.service and args.replicas:
                success = deployer.scale_deployment(args.service, args.replicas)
                print(f"Scaling {'succeeded' if success else 'failed'}")
            else:
                print("Service name and replicas are required for scale action")
        elif args.action == "rollback":
            deployer.rollback_deployment(args.service)
            print("Rollback completed")
        elif args.action == "backup":
            backup_path = deployer.backup_deployment()
            print(f"Backup created at {backup_path}")
        elif args.action == "health":
            health_results = deployer.run_health_checks()
            print(json.dumps(health_results, indent=2))

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"Deployment failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import time
    exit(main())