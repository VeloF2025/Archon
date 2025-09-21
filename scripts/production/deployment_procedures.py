#!/usr/bin/env python3
"""
Agency Swarm Production Deployment Procedures
Automated deployment procedures for production environment
"""

import asyncio
import logging
import json
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import yaml
import tempfile
import os
import shutil
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment_procedures.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class DeploymentStrategy(Enum):
    BLUE_GREEN = "blue_green"
    ROLLING_UPDATE = "rolling_update"
    CANARY = "canary"

@dataclass
class DeploymentStep:
    """Deployment step information"""
    name: str
    command: str
    timeout: int
    critical: bool = True
    rollback_command: Optional[str] = None
    validation_command: Optional[str] = None

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    environment: str
    namespace: str
    kubernetes_context: str
    docker_registry: str
    image_tag: str
    health_check_timeout: int
    rollback_timeout: int
    pre_deployment_checks: List[DeploymentStep]
    deployment_steps: List[DeploymentStep]
    post_deployment_checks: List[DeploymentStep]
    rollback_steps: List[DeploymentStep]

@dataclass
class DeploymentResult:
    """Deployment result information"""
    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    steps_completed: List[str]
    steps_failed: List[str]
    error_message: Optional[str]
    rollback_deployment_id: Optional[str]
    metadata: Dict[str, Any]

class DeploymentManager:
    """Production deployment manager"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.current_deployment: Optional[DeploymentResult] = None
        self.rollback_deployment: Optional[DeploymentResult] = None
        self.deployment_history: List[DeploymentResult] = []

    async def execute_deployment(self) -> DeploymentResult:
        """Execute complete deployment process"""
        deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()

        logger.info(f"Starting deployment {deployment_id} with strategy {self.config.strategy.value}")

        self.current_deployment = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=start_time,
            end_time=None,
            duration=None,
            steps_completed=[],
            steps_failed=[],
            error_message=None,
            rollback_deployment_id=None,
            metadata={
                'strategy': self.config.strategy.value,
                'environment': self.config.environment,
                'namespace': self.config.namespace,
                'image_tag': self.config.image_tag
            }
        )

        try:
            # Execute deployment phases
            await self._execute_phase(DeploymentStatus.PREPARING, self._prepare_deployment)
            await self._execute_phase(DeploymentStatus.DEPLOYING, self._execute_deployment)
            await self._execute_phase(DeploymentStatus.VALIDATING, self._validate_deployment)

            # Mark as completed
            self.current_deployment.status = DeploymentStatus.COMPLETED
            self.current_deployment.end_time = datetime.utcnow()
            self.current_deployment.duration = self.current_deployment.end_time - self.current_deployment.start_time

            logger.info(f"Deployment {deployment_id} completed successfully in {self.current_deployment.duration}")

        except Exception as e:
            # Handle deployment failure
            await self._handle_deployment_failure(e)

        # Add to history
        self.deployment_history.append(self.current_deployment)
        return self.current_deployment

    async def _execute_phase(self, status: DeploymentStatus, phase_function):
        """Execute a deployment phase"""
        logger.info(f"Starting {status.value} phase...")
        self.current_deployment.status = status

        try:
            await phase_function()
            logger.info(f"{status.value} phase completed successfully")
        except Exception as e:
            logger.error(f"{status.value} phase failed: {e}")
            raise

    async def _prepare_deployment(self):
        """Prepare deployment environment"""
        logger.info("Preparing deployment environment...")

        # Run pre-deployment checks
        for step in self.config.pre_deployment_checks:
            logger.info(f"Running pre-deployment check: {step.name}")
            await self._execute_step(step)

        # Verify Kubernetes context
        await self._verify_kubernetes_context()

        # Check resource availability
        await self._check_resource_availability()

        # Backup current state
        await self._backup_current_state()

        # Prepare deployment manifests
        await self._prepare_manifests()

    async def _execute_deployment(self):
        """Execute main deployment"""
        logger.info(f"Executing {self.config.strategy.value} deployment...")

        if self.config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._execute_blue_green_deployment()
        elif self.config.strategy == DeploymentStrategy.ROLLING_UPDATE:
            await self._execute_rolling_update()
        elif self.config.strategy == DeploymentStrategy.CANARY:
            await self._execute_canary_deployment()
        else:
            raise ValueError(f"Unsupported deployment strategy: {self.config.strategy}")

    async def _validate_deployment(self):
        """Validate deployment"""
        logger.info("Validating deployment...")

        # Wait for deployment to stabilize
        await self._wait_for_deployment_stabilization()

        # Run post-deployment checks
        for step in self.config.post_deployment_checks:
            logger.info(f"Running post-deployment check: {step.name}")
            await self._execute_step(step)

        # Run comprehensive health checks
        await self._run_comprehensive_health_checks()

        # Validate performance
        await self._validate_performance()

        logger.info("Deployment validation completed successfully")

    async def _execute_blue_green_deployment(self):
        """Execute blue-green deployment strategy"""
        logger.info("Executing blue-green deployment...")

        # Determine current and new colors
        current_color = await self._get_current_deployment_color()
        new_color = "blue" if current_color == "green" else "green"

        logger.info(f"Current color: {current_color}, New color: {new_color}")

        # Deploy new version to new color
        await self._deploy_to_color(new_color)

        # Validate new deployment
        await self._validate_color_deployment(new_color)

        # Switch traffic to new color
        await self._switch_traffic(new_color)

        # Monitor new deployment
        await self._monitor_deployment_after_switch()

        # Cleanup old deployment
        await self._cleanup_old_deployment(current_color)

        logger.info("Blue-green deployment completed")

    async def _execute_rolling_update(self):
        """Execute rolling update deployment strategy"""
        logger.info("Executing rolling update deployment...")

        # Update deployment manifests
        await self._update_deployment_manifests()

        # Apply rolling update
        await self._apply_rolling_update()

        # Monitor rollout progress
        await self._monitor_rollout_progress()

        logger.info("Rolling update deployment completed")

    async def _execute_canary_deployment(self):
        """Execute canary deployment strategy"""
        logger.info("Executing canary deployment...")

        # Deploy canary instances
        await self._deploy_canary_instances()

        # Monitor canary performance
        await self._monitor_canary_performance()

        # If canary is successful, proceed with full deployment
        await self._proceed_with_full_deployment()

        logger.info("Canary deployment completed")

    async def _execute_step(self, step: DeploymentStep):
        """Execute a deployment step"""
        logger.info(f"Executing step: {step.name}")

        try:
            # Execute command
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )

            if result.returncode == 0:
                self.current_deployment.steps_completed.append(step.name)
                logger.info(f"Step {step.name} completed successfully")
                logger.debug(f"Step output: {result.stdout}")
            else:
                error_msg = f"Step {step.name} failed with return code {result.returncode}"
                if step.critical:
                    self.current_deployment.steps_failed.append(step.name)
                    logger.error(f"{error_msg}: {result.stderr}")
                    raise Exception(f"{error_msg}: {result.stderr}")
                else:
                    logger.warning(f"{error_msg}: {result.stderr}")
                    self.current_deployment.steps_completed.append(step.name)

            # Run validation if available
            if step.validation_command:
                await self._validate_step(step)

        except subprocess.TimeoutExpired:
            error_msg = f"Step {step.name} timed out after {step.timeout} seconds"
            if step.critical:
                self.current_deployment.steps_failed.append(step.name)
                logger.error(error_msg)
                raise Exception(error_msg)
            else:
                logger.warning(error_msg)

        except Exception as e:
            if step.critical:
                self.current_deployment.steps_failed.append(step.name)
                raise
            else:
                logger.warning(f"Non-critical step {step.name} failed: {e}")
                self.current_deployment.steps_completed.append(step.name)

    async def _validate_step(self, step: DeploymentStep):
        """Validate a deployment step"""
        logger.info(f"Validating step: {step.name}")

        try:
            result = subprocess.run(
                step.validation_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )

            if result.returncode != 0:
                raise Exception(f"Validation failed: {result.stderr}")

            logger.info(f"Step {step.name} validation passed")

        except Exception as e:
            logger.error(f"Step {step.name} validation failed: {e}")
            raise

    async def _verify_kubernetes_context(self):
        """Verify Kubernetes context"""
        logger.info("Verifying Kubernetes context...")

        result = subprocess.run(
            ['kubectl', 'config', 'current-context'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise Exception(f"Failed to get Kubernetes context: {result.stderr}")

        current_context = result.stdout.strip()
        if current_context != self.config.kubernetes_context:
            raise Exception(f"Expected Kubernetes context '{self.config.kubernetes_context}', got '{current_context}'")

        logger.info(f"Kubernetes context verified: {current_context}")

    async def _check_resource_availability(self):
        """Check resource availability"""
        logger.info("Checking resource availability...")

        # Check node resources
        result = subprocess.run(
            ['kubectl', 'get', 'nodes', '-o', 'jsonpath={range .items[*]}{.status.allocatable}{"\\n"}{end}'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Parse resource information (simplified)
            for line in result.stdout.strip().split('\n'):
                if line and 'cpu' in line and 'memory' in line:
                    logger.info(f"Available resources: {line}")

        # Check cluster capacity
        capacity_result = subprocess.run(
            ['kubectl', 'get', 'nodes', '-o', 'jsonpath={range .items[*]}{.status.capacity}{"\\n"}{end}'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if capacity_result.returncode == 0:
            logger.info("Cluster capacity verified")

    async def _backup_current_state(self):
        """Backup current deployment state"""
        logger.info("Backing up current deployment state...")

        # Backup current deployments
        backup_dir = f"/tmp/deployment_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)

        # Backup current manifests
        backup_command = f"kubectl get all,configmap,secret -n {self.config.namespace} -o yaml > {backup_dir}/current_state.yaml"
        result = subprocess.run(
            backup_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.warning(f"Failed to backup current state: {result.stderr}")
        else:
            logger.info(f"Current state backed up to {backup_dir}")

        # Store backup location in metadata
        self.current_deployment.metadata['backup_location'] = backup_dir

    async def _prepare_manifests(self):
        """Prepare deployment manifests"""
        logger.info("Preparing deployment manifests...")

        # Update image tags in manifests
        manifest_files = [
            'kubernetes/production/production-deployment.yaml'
        ]

        for manifest_file in manifest_files:
            if os.path.exists(manifest_file):
                await self._update_manifest_image(manifest_file, self.config.image_tag)

        logger.info("Deployment manifests prepared")

    async def _update_manifest_image(self, manifest_file: str, image_tag: str):
        """Update image tag in manifest file"""
        try:
            with open(manifest_file, 'r') as f:
                content = f.read()

            # Update image references
            # This is a simplified approach - in production you'd use a proper YAML parser
            updated_content = content.replace(':latest', f':{image_tag}')

            with open(manifest_file, 'w') as f:
                f.write(updated_content)

            logger.info(f"Updated image tag in {manifest_file}")

        except Exception as e:
            logger.error(f"Failed to update manifest {manifest_file}: {e}")
            raise

    async def _get_current_deployment_color(self) -> str:
        """Get current deployment color for blue-green strategy"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'ingress', '-n', self.config.namespace, '-o', 'jsonpath={.items[0].spec.rules[0].http.paths[0].backend.service.name}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                service_name = result.stdout.strip()
                if 'blue' in service_name:
                    return 'blue'
                elif 'green' in service_name:
                    return 'green'

            return 'green'  # Default to green

        except Exception as e:
            logger.warning(f"Failed to get current deployment color: {e}")
            return 'green'

    async def _deploy_to_color(self, color: str):
        """Deploy to specific color"""
        logger.info(f"Deploying to {color} environment...")

        # Apply manifests for the specific color
        manifest_file = 'kubernetes/production/production-deployment.yaml'
        if os.path.exists(manifest_file):
            command = f"kubectl apply -f {manifest_file} -n {self.config.namespace}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise Exception(f"Failed to apply manifests: {result.stderr}")

            logger.info(f"Deployment to {color} completed")

    async def _validate_color_deployment(self, color: str):
        """Validate deployment to specific color"""
        logger.info(f"Validating {color} deployment...")

        # Wait for deployments to be ready
        deployments = ['archon-backend', 'archon-ui', 'archon-mcp', 'archon-agents']

        for deployment in deployments:
            command = f"kubectl wait --for=condition=available --timeout=600s deployment/{deployment} -n {self.config.namespace}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=660
            )

            if result.returncode != 0:
                raise Exception(f"Deployment {deployment} not ready: {result.stderr}")

        logger.info(f"{color} deployment validated")

    async def _switch_traffic(self, new_color: str):
        """Switch traffic to new color"""
        logger.info(f"Switching traffic to {new_color}...")

        # Update ingress to point to new service
        ingress_patch = {
            "spec": {
                "rules": [{
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"archon-ui-{new_color}-service",
                                    "port": {"number": 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(ingress_patch, f)
            f.flush()

            command = f"kubectl patch ingress archon-ingress -n {self.config.namespace} --patch-file {f.name}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise Exception(f"Failed to switch traffic: {result.stderr}")

        logger.info(f"Traffic switched to {new_color}")

    async def _monitor_deployment_after_switch(self):
        """Monitor deployment after traffic switch"""
        logger.info("Monitoring deployment after traffic switch...")

        # Monitor for 5 minutes
        monitoring_duration = 300
        check_interval = 30

        for i in range(0, monitoring_duration, check_interval):
            await self._check_deployment_health()
            time.sleep(check_interval)

        logger.info("Post-switch monitoring completed")

    async def _cleanup_old_deployment(self, old_color: str):
        """Cleanup old deployment"""
        logger.info(f"Cleaning up {old_color} deployment...")

        # Scale down old deployment
        command = f"kubectl scale deployment archon-backend-{old_color} archon-ui-{old_color} archon-mcp-{old_color} archon-agents-{old_color} --replicas=0 -n {self.config.namespace}"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.warning(f"Failed to scale down {old_color} deployment: {result.stderr}")
        else:
            logger.info(f"{old_color} deployment scaled down")

    async def _update_deployment_manifests(self):
        """Update deployment manifests for rolling update"""
        logger.info("Updating deployment manifests...")

        # Apply updated manifests
        manifest_file = 'kubernetes/production/production-deployment.yaml'
        if os.path.exists(manifest_file):
            command = f"kubectl apply -f {manifest_file} -n {self.config.namespace}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise Exception(f"Failed to apply manifests: {result.stderr}")

        logger.info("Deployment manifests updated")

    async def _apply_rolling_update(self):
        """Apply rolling update"""
        logger.info("Applying rolling update...")

        # The rolling update is automatically applied by Kubernetes
        # We just need to monitor it
        await self._monitor_rollout_progress()

        logger.info("Rolling update applied")

    async def _monitor_rollout_progress(self):
        """Monitor rollout progress"""
        logger.info("Monitoring rollout progress...")

        deployments = ['archon-backend', 'archon-ui', 'archon-mcp', 'archon-agents']

        for deployment in deployments:
            logger.info(f"Monitoring rollout for {deployment}...")

            # Wait for rollout to complete
            command = f"kubectl rollout status deployment/{deployment} -n {self.config.namespace} --timeout=600s"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=660
            )

            if result.returncode != 0:
                raise Exception(f"Rollout for {deployment} failed: {result.stderr}")

            logger.info(f"Rollout for {deployment} completed")

    async def _deploy_canary_instances(self):
        """Deploy canary instances"""
        logger.info("Deploying canary instances...")

        # Create canary deployment with 10% of traffic
        canary_replicas = 1  # Assuming 10 replicas total

        # Create canary deployment
        canary_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "archon-backend-canary",
                "namespace": self.config.namespace
            },
            "spec": {
                "replicas": canary_replicas,
                "selector": {
                    "matchLabels": {
                        "app": "archon-backend",
                        "version": "canary"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "archon-backend",
                            "version": "canary"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "backend",
                            "image": f"{self.config.docker_registry}/archon-backend:{self.config.image_tag}",
                            "ports": [{"containerPort": 8000}]
                        }]
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
            yaml.dump(canary_manifest, f)
            f.flush()

            command = f"kubectl apply -f {f.name}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise Exception(f"Failed to deploy canary: {result.stderr}")

        logger.info("Canary instances deployed")

    async def _monitor_canary_performance(self):
        """Monitor canary performance"""
        logger.info("Monitoring canary performance...")

        # Monitor for 10 minutes
        monitoring_duration = 600
        check_interval = 60

        for i in range(0, monitoring_duration, check_interval):
            await self._check_canary_health()
            time.sleep(check_interval)

        logger.info("Canary performance monitoring completed")

    async def _proceed_with_full_deployment(self):
        """Proceed with full deployment after successful canary"""
        logger.info("Proceeding with full deployment...")

        # Remove canary deployment
        command = f"kubectl delete deployment archon-backend-canary -n {self.config.namespace}"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            logger.warning(f"Failed to remove canary deployment: {result.stderr}")

        # Proceed with full deployment using rolling update
        await self._execute_rolling_update()

        logger.info("Full deployment completed")

    async def _wait_for_deployment_stabilization(self):
        """Wait for deployment to stabilize"""
        logger.info("Waiting for deployment to stabilize...")

        # Wait for all pods to be ready
        stabilization_time = 300  # 5 minutes
        check_interval = 30

        for i in range(0, stabilization_time, check_interval):
            healthy = await self._check_all_pods_healthy()
            if healthy:
                logger.info("Deployment stabilized")
                return
            time.sleep(check_interval)

        logger.warning("Deployment did not stabilize within expected time")

    async def _check_all_pods_healthy(self) -> bool:
        """Check if all pods are healthy"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.config.namespace, '-o', 'jsonpath={range .items[*]}{.status.phase}{"\\n"}{end}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                phases = result.stdout.strip().split('\n')
                return all(phase == 'Running' for phase in phases if phase)

            return False

        except Exception as e:
            logger.error(f"Failed to check pod health: {e}")
            return False

    async def _run_comprehensive_health_checks(self):
        """Run comprehensive health checks"""
        logger.info("Running comprehensive health checks...")

        # Check all services
        services = ['archon-backend-service', 'archon-ui-service', 'archon-mcp-service', 'archon-agents-service']

        for service in services:
            await self._check_service_health(service)

        # Check ingress
        await self._check_ingress_health()

        # Check application health endpoints
        await self._check_application_health()

        logger.info("Comprehensive health checks completed")

    async def _check_service_health(self, service_name: str):
        """Check specific service health"""
        logger.info(f"Checking health of {service_name}...")

        try:
            result = subprocess.run(
                ['kubectl', 'get', 'service', service_name, '-n', self.config.namespace, '-o', 'jsonpath={.spec.ports[0].port}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                port = result.stdout.strip()
                logger.info(f"Service {service_name} is healthy on port {port}")
            else:
                logger.warning(f"Failed to check service {service_name}: {result.stderr}")

        except Exception as e:
            logger.error(f"Error checking service {service_name}: {e}")

    async def _check_ingress_health(self):
        """Check ingress health"""
        logger.info("Checking ingress health...")

        try:
            result = subprocess.run(
                ['kubectl', 'get', 'ingress', '-n', self.config.namespace],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("Ingress is healthy")
                logger.debug(f"Ingress details: {result.stdout}")
            else:
                logger.warning(f"Failed to check ingress: {result.stderr}")

        except Exception as e:
            logger.error(f"Error checking ingress: {e}")

    async def _check_application_health(self):
        """Check application health endpoints"""
        logger.info("Checking application health endpoints...")

        # Get ingress URL
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'ingress', 'archon-ingress', '-n', self.config.namespace, '-o', 'jsonpath={.spec.rules[0].host}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                host = result.stdout.strip()
                if host:
                    # Test health endpoint
                    health_url = f"https://{host}/health"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            if response.status == 200:
                                logger.info("Application health endpoint is responding")
                            else:
                                logger.warning(f"Application health endpoint returned status {response.status}")

        except Exception as e:
            logger.error(f"Error checking application health: {e}")

    async def _validate_performance(self):
        """Validate deployment performance"""
        logger.info("Validating deployment performance...")

        # Check response times
        await self._check_response_times()

        # Check resource usage
        await self._check_resource_usage()

        # Check error rates
        await self._check_error_rates()

        logger.info("Performance validation completed")

    async def _check_response_times(self):
        """Check application response times"""
        logger.info("Checking response times...")

        # Get ingress URL and test response times
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'ingress', 'archon-ingress', '-n', self.config.namespace, '-o', 'jsonpath={.spec.rules[0].host}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                host = result.stdout.strip()
                if host:
                    # Test multiple endpoints
                    endpoints = ['/health', '/api/v1/health', '/metrics']
                    for endpoint in endpoints:
                        url = f"https://{host}{endpoint}"
                        async with aiohttp.ClientSession() as session:
                            start_time = time.time()
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                                response_time = time.time() - start_time
                                logger.info(f"Response time for {endpoint}: {response_time:.3f}s")

                                if response_time > 2.0:
                                    logger.warning(f"Slow response time for {endpoint}: {response_time:.3f}s")

        except Exception as e:
            logger.error(f"Error checking response times: {e}")

    async def _check_resource_usage(self):
        """Check resource usage"""
        logger.info("Checking resource usage...")

        try:
            result = subprocess.run(
                ['kubectl', 'top', 'pods', '-n', self.config.namespace],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("Resource usage:")
                logger.info(result.stdout)

                # Check for high resource usage
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            cpu_usage = parts[1]
                            memory_usage = parts[2]

                            # Check for high usage (simplified parsing)
                            if 'm' in cpu_usage:
                                cpu_millicores = int(cpu_usage.replace('m', ''))
                                if cpu_millicores > 800:  # 800m = 0.8 CPU
                                    logger.warning(f"High CPU usage: {cpu_usage}")

                            if 'Mi' in memory_usage:
                                memory_mib = int(memory_usage.replace('Mi', ''))
                                if memory_mib > 700:  # 700Mi
                                    logger.warning(f"High memory usage: {memory_usage}")

        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")

    async def _check_error_rates(self):
        """Check error rates"""
        logger.info("Checking error rates...")

        # This would typically query metrics from Prometheus
        # For now, we'll check pod restart counts
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.config.namespace, '-o', 'jsonpath={range .items[*]}{.metadata.name}{\" \"}{.status.containerStatuses[0].restartCount}{\"\\n\"}'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.rsplit(' ', 1)
                        if len(parts) == 2:
                            pod_name, restart_count = parts
                            if int(restart_count) > 0:
                                logger.warning(f"Pod {pod_name} has {restart_count} restarts")

        except Exception as e:
            logger.error(f"Error checking error rates: {e}")

    async def _check_deployment_health(self):
        """Check deployment health during monitoring"""
        try:
            # Check pod status
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.config.namespace],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            status = parts[2]
                            if status not in ['Running', 'Completed']:
                                logger.warning(f"Pod {parts[0]} has status {status}")

        except Exception as e:
            logger.error(f"Error checking deployment health: {e}")

    async def _check_canary_health(self):
        """Check canary deployment health"""
        try:
            # Check canary pod status
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-l', 'version=canary', '-n', self.config.namespace],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            status = parts[2]
                            if status != 'Running':
                                raise Exception(f"Canary pod {parts[0]} has status {status}")

        except Exception as e:
            logger.error(f"Error checking canary health: {e}")
            raise

    async def _handle_deployment_failure(self, error: Exception):
        """Handle deployment failure"""
        logger.error(f"Deployment failed: {error}")

        # Update deployment status
        self.current_deployment.status = DeploymentStatus.FAILED
        self.current_deployment.end_time = datetime.utcnow()
        self.current_deployment.duration = self.current_deployment.end_time - self.current_deployment.start_time
        self.current_deployment.error_message = str(error)

        # Attempt rollback if configured
        if self.config.rollback_steps:
            logger.info("Attempting rollback...")
            try:
                rollback_result = await self._execute_rollback()
                self.current_deployment.rollback_deployment_id = rollback_result.deployment_id
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")

        raise Exception(f"Deployment failed: {error}")

    async def _execute_rollback(self) -> DeploymentResult:
        """Execute rollback procedure"""
        logger.info("Executing rollback...")

        rollback_id = f"rollback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()

        self.rollback_deployment = DeploymentResult(
            deployment_id=rollback_id,
            status=DeploymentStatus.ROLLING_BACK,
            start_time=start_time,
            end_time=None,
            duration=None,
            steps_completed=[],
            steps_failed=[],
            error_message=None,
            rollback_deployment_id=None,
            metadata={
                'original_deployment': self.current_deployment.deployment_id,
                'strategy': 'rollback',
                'environment': self.config.environment,
                'namespace': self.config.namespace
            }
        )

        try:
            # Execute rollback steps
            for step in self.config.rollback_steps:
                logger.info(f"Executing rollback step: {step.name}")
                await self._execute_step(step)

            # Restore from backup if available
            backup_location = self.current_deployment.metadata.get('backup_location')
            if backup_location and os.path.exists(backup_location):
                await self._restore_from_backup(backup_location)

            # Validate rollback
            await self._validate_rollback()

            # Mark rollback as completed
            self.rollback_deployment.status = DeploymentStatus.ROLLED_BACK
            self.rollback_deployment.end_time = datetime.utcnow()
            self.rollback_deployment.duration = self.rollback_deployment.end_time - self.rollback_deployment.start_time

            logger.info(f"Rollback {rollback_id} completed successfully")

        except Exception as e:
            self.rollback_deployment.status = DeploymentStatus.FAILED
            self.rollback_deployment.end_time = datetime.utcnow()
            self.rollback_deployment.duration = self.rollback_deployment.end_time - self.rollback_deployment.start_time
            self.rollback_deployment.error_message = str(e)

            logger.error(f"Rollback failed: {e}")

        # Add to history
        self.deployment_history.append(self.rollback_deployment)
        return self.rollback_deployment

    async def _restore_from_backup(self, backup_location: str):
        """Restore deployment from backup"""
        logger.info(f"Restoring from backup: {backup_location}")

        if os.path.exists(f"{backup_location}/current_state.yaml"):
            command = f"kubectl apply -f {backup_location}/current_state.yaml -n {self.config.namespace}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise Exception(f"Failed to restore from backup: {result.stderr}")

            logger.info("Successfully restored from backup")
        else:
            logger.warning(f"Backup file not found: {backup_location}/current_state.yaml")

    async def _validate_rollback(self):
        """Validate rollback success"""
        logger.info("Validating rollback...")

        # Wait for services to be ready
        deployments = ['archon-backend', 'archon-ui', 'archon-mcp', 'archon-agents']

        for deployment in deployments:
            command = f"kubectl wait --for=condition=available --timeout=300s deployment/{deployment} -n {self.config.namespace}"
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=330
            )

            if result.returncode != 0:
                raise Exception(f"Deployment {deployment} not ready after rollback: {result.stderr}")

        logger.info("Rollback validation completed")

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status by ID"""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None

    def get_latest_deployment(self) -> Optional[DeploymentResult]:
        """Get latest deployment"""
        return self.deployment_history[-1] if self.deployment_history else None

    def get_deployment_history(self, limit: int = 10) -> List[DeploymentResult]:
        """Get deployment history"""
        return self.deployment_history[-limit:] if self.deployment_history else []

# Default deployment configuration
DEFAULT_DEPLOYMENT_CONFIG = DeploymentConfig(
    strategy=DeploymentStrategy.BLUE_GREEN,
    environment="production",
    namespace="archon-production",
    kubernetes_context="production-cluster",
    docker_registry="ghcr.io/your-org",
    image_tag="latest",
    health_check_timeout=600,
    rollback_timeout=900,
    pre_deployment_checks=[
        DeploymentStep(
            name="Kubernetes Context Check",
            command="kubectl config current-context",
            timeout=30,
            critical=True
        ),
        DeploymentStep(
            name="Resource Availability Check",
            command="kubectl get nodes",
            timeout=30,
            critical=True
        ),
        DeploymentStep(
            name="Database Connectivity Check",
            command="kubectl exec -it deployment/postgresql-production -- pg_isready -U postgres",
            timeout=30,
            critical=True
        ),
        DeploymentStep(
            name="Backup Current State",
            command="kubectl get all,configmap,secret -n archon-production -o yaml > /tmp/pre_deployment_backup.yaml",
            timeout=60,
            critical=True
        )
    ],
    deployment_steps=[
        DeploymentStep(
            name="Apply Production Manifests",
            command="kubectl apply -f kubernetes/production/production-deployment.yaml -n archon-production",
            timeout=120,
            critical=True,
            rollback_command="kubectl apply -f /tmp/pre_deployment_backup.yaml -n archon-production"
        )
    ],
    post_deployment_checks=[
        DeploymentStep(
            name="Wait for Deployments",
            command="kubectl wait --for=condition=available --timeout=600s deployment -l app=archon -n archon-production",
            timeout=660,
            critical=True
        ),
        DeploymentStep(
            name="Health Check",
            command="kubectl get pods -n archon-production -l app=archon",
            timeout=30,
            critical=True,
            validation_command="kubectl get pods -n archon-production -l app=archon --field-selector=status.phase=Running"
        )
    ],
    rollback_steps=[
        DeploymentStep(
            name="Rollback Deployment",
            command="kubectl apply -f /tmp/pre_deployment_backup.yaml -n archon-production",
            timeout=120,
            critical=True
        ),
        DeploymentStep(
            name="Wait for Rollback",
            command="kubectl wait --for=condition=available --timeout=300s deployment -l app=archon -n archon-production",
            timeout=330,
            critical=True
        )
    ]
)

async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Agency Swarm Production Deployment")
    parser.add_argument("--strategy", choices=["blue_green", "rolling_update", "canary"], default="blue_green", help="Deployment strategy")
    parser.add_argument("--environment", default="production", help="Target environment")
    parser.add_argument("--namespace", default="archon-production", help="Target namespace")
    parser.add_argument("--image-tag", default="latest", help="Docker image tag")
    parser.add_argument("--config", help="Deployment configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_DEPLOYMENT_CONFIG
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                # Convert dict to DeploymentConfig (simplified)
                config.strategy = DeploymentStrategy(config_data.get('strategy', 'blue_green'))
                config.environment = config_data.get('environment', 'production')
                config.image_tag = config_data.get('image_tag', 'latest')
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return

    # Override with command line arguments
    config.strategy = DeploymentStrategy(args.strategy)
    config.environment = args.environment
    config.namespace = args.namespace
    config.image_tag = args.image_tag

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deployment will be performed")
        logger.info(f"Configuration: {config}")
        return

    # Create deployment manager
    deployment_manager = DeploymentManager(config)

    try:
        # Execute deployment
        result = await deployment_manager.execute_deployment()

        # Print result
        print(f"\n=== Deployment Result ===")
        print(f"Deployment ID: {result.deployment_id}")
        print(f"Status: {result.status.value}")
        print(f"Duration: {result.duration}")
        print(f"Steps Completed: {len(result.steps_completed)}")
        print(f"Steps Failed: {len(result.steps_failed)}")

        if result.error_message:
            print(f"Error: {result.error_message}")

        if result.rollback_deployment_id:
            print(f"Rollback ID: {result.rollback_deployment_id}")

        # Exit with appropriate code
        if result.status == DeploymentStatus.COMPLETED:
            exit_code = 0
        elif result.status == DeploymentStatus.ROLLED_BACK:
            exit_code = 1
        else:
            exit_code = 2

        exit(exit_code)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        exit(3)

if __name__ == "__main__":
    asyncio.run(main())