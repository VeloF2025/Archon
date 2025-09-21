"""
🚀 ARCHON ENHANCEMENT 2025 - PHASE 6: ADVANCED AI INTEGRATION
AI Model Management Platform - Complete MLOps and Model Lifecycle Management

This module provides a comprehensive AI model management platform with advanced capabilities
including model registry, version control, deployment, monitoring, A/B testing, and 
automated ML pipeline orchestration with enterprise-grade governance.
"""

import asyncio
import logging
import json
import pickle
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import hashlib
from collections import defaultdict, deque
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported AI model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDER = "recommender"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    MULTIMODAL = "multimodal"


class ModelStatus(Enum):
    """Model lifecycle status."""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class DeploymentTarget(Enum):
    """Model deployment targets."""
    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"
    MOBILE = "mobile"
    BATCH = "batch"
    STREAMING = "streaming"
    API_ENDPOINT = "api_endpoint"
    EMBEDDED = "embedded"


class MetricType(Enum):
    """Model performance metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DRIFT_SCORE = "drift_score"
    BIAS_SCORE = "bias_score"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    name: str
    version: str
    model_type: ModelType
    framework: str
    algorithm: str
    created_by: str
    created_at: datetime
    description: str = ""
    tags: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    feature_schema: Dict[str, Any] = field(default_factory=dict)
    target_schema: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    license: str = "proprietary"
    compliance_tags: List[str] = field(default_factory=list)


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    version_number: str
    parent_version: Optional[str] = None
    model_artifact_path: Optional[str] = None
    model_size_bytes: int = 0
    checksum: Optional[str] = None
    status: ModelStatus = ModelStatus.DRAFT
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    commit_message: str = ""
    experiment_id: Optional[str] = None


@dataclass
class ModelDeployment:
    """Model deployment configuration."""
    deployment_id: str
    model_version_id: str
    target: DeploymentTarget
    endpoint_url: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    status: str = "pending"
    deployed_at: Optional[datetime] = None
    health_check_url: Optional[str] = None
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics."""
    model_version_id: str
    deployment_id: Optional[str]
    metric_type: MetricType
    value: float
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    is_alerting: bool = False


@dataclass
class ExperimentRun:
    """ML experiment run information."""
    run_id: str
    experiment_name: str
    model_version_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    status: str = "running"
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    notes: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ABTestConfiguration:
    """A/B testing configuration for models."""
    test_id: str
    test_name: str
    model_a_version_id: str
    model_b_version_id: str
    traffic_split: Dict[str, float]  # e.g., {"A": 0.5, "B": 0.5}
    success_metrics: List[str]
    test_duration: timedelta
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
    results: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Optional[float] = None


class ModelRegistry:
    """Central registry for AI model management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self.artifacts: Dict[str, Any] = {}  # Model artifacts storage
        self.storage_path = Path(config.get('storage_path', './model_registry'))
        self.storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / 'models').mkdir(exist_ok=True)
        (self.storage_path / 'artifacts').mkdir(exist_ok=True)
        (self.storage_path / 'metadata').mkdir(exist_ok=True)
    
    async def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model in the registry."""
        model_id = f"{metadata.name}_{metadata.framework}_{int(metadata.created_at.timestamp())}"
        
        # Validate model metadata
        await self._validate_model_metadata(metadata)
        
        # Store model
        self.models[model_id] = metadata
        
        # Create initial version
        initial_version = ModelVersion(
            version_id=f"{model_id}_v1.0.0",
            version_number="1.0.0",
            status=ModelStatus.DRAFT,
            commit_message="Initial model registration"
        )
        
        self.versions[model_id]["1.0.0"] = initial_version
        
        # Persist to storage
        await self._persist_model_metadata(model_id, metadata)
        await self._persist_model_version(model_id, initial_version)
        
        logger.info(f"Registered model: {model_id} v1.0.0")
        return model_id
    
    async def create_model_version(
        self, 
        model_id: str, 
        version_number: str,
        model_artifact: Any = None,
        parent_version: Optional[str] = None,
        commit_message: str = "",
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Create a new version of an existing model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Generate version ID
        version_id = f"{model_id}_{version_number}"
        
        # Check if version already exists
        if version_number in self.versions[model_id]:
            raise ValueError(f"Version {version_number} already exists for model {model_id}")
        
        # Store model artifact if provided
        artifact_path = None
        model_size = 0
        checksum = None
        
        if model_artifact is not None:
            artifact_path, model_size, checksum = await self._store_model_artifact(
                version_id, model_artifact
            )
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            version_number=version_number,
            parent_version=parent_version,
            model_artifact_path=artifact_path,
            model_size_bytes=model_size,
            checksum=checksum,
            status=ModelStatus.DRAFT,
            metrics=metrics or {},
            commit_message=commit_message
        )
        
        self.versions[model_id][version_number] = version
        
        # Persist version
        await self._persist_model_version(model_id, version)
        
        logger.info(f"Created model version: {model_id} v{version_number}")
        return version_id
    
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata."""
        return self.models.get(model_id)
    
    async def get_model_version(self, model_id: str, version_number: str) -> Optional[ModelVersion]:
        """Retrieve specific model version."""
        return self.versions.get(model_id, {}).get(version_number)
    
    async def list_models(self, 
                         model_type: Optional[ModelType] = None,
                         tags: Optional[List[str]] = None,
                         status: Optional[ModelStatus] = None) -> List[Tuple[str, ModelMetadata]]:
        """List models with optional filtering."""
        results = []
        
        for model_id, metadata in self.models.items():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
                
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            if status:
                # Check latest version status
                latest_version = await self.get_latest_version(model_id)
                if not latest_version or latest_version.status != status:
                    continue
            
            results.append((model_id, metadata))
        
        return results
    
    async def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        if model_id not in self.versions:
            return None
        
        versions = self.versions[model_id]
        if not versions:
            return None
        
        # Sort versions by creation time
        sorted_versions = sorted(
            versions.values(), 
            key=lambda v: v.created_at, 
            reverse=True
        )
        
        return sorted_versions[0]
    
    async def update_model_status(self, model_id: str, version_number: str, status: ModelStatus) -> None:
        """Update model version status."""
        version = self.versions.get(model_id, {}).get(version_number)
        if not version:
            raise ValueError(f"Version {version_number} not found for model {model_id}")
        
        old_status = version.status
        version.status = status
        version.updated_at = datetime.now()
        
        # Persist changes
        await self._persist_model_version(model_id, version)
        
        logger.info(f"Updated model {model_id} v{version_number} status: {old_status} -> {status}")
    
    async def _validate_model_metadata(self, metadata: ModelMetadata) -> None:
        """Validate model metadata."""
        required_fields = ['name', 'model_type', 'framework', 'algorithm', 'created_by']
        
        for field in required_fields:
            if not getattr(metadata, field):
                raise ValueError(f"Required field '{field}' is missing or empty")
        
        # Validate model type
        if not isinstance(metadata.model_type, ModelType):
            raise ValueError(f"Invalid model type: {metadata.model_type}")
    
    async def _store_model_artifact(self, version_id: str, artifact: Any) -> Tuple[str, int, str]:
        """Store model artifact and return path, size, and checksum."""
        artifact_path = self.storage_path / 'artifacts' / f"{version_id}.pkl"
        
        # Serialize artifact
        with open(artifact_path, 'wb') as f:
            pickle.dump(artifact, f)
        
        # Calculate size and checksum
        size = artifact_path.stat().st_size
        
        with open(artifact_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.sha256(content).hexdigest()
        
        return str(artifact_path), size, checksum
    
    async def _persist_model_metadata(self, model_id: str, metadata: ModelMetadata) -> None:
        """Persist model metadata to storage."""
        metadata_path = self.storage_path / 'metadata' / f"{model_id}_metadata.json"
        
        # Convert dataclass to dict for JSON serialization
        metadata_dict = asdict(metadata)
        metadata_dict['model_type'] = metadata.model_type.value
        metadata_dict['created_at'] = metadata.created_at.isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    async def _persist_model_version(self, model_id: str, version: ModelVersion) -> None:
        """Persist model version to storage."""
        version_path = self.storage_path / 'metadata' / f"{model_id}_{version.version_number}_version.json"
        
        # Convert dataclass to dict for JSON serialization
        version_dict = asdict(version)
        version_dict['status'] = version.status.value
        version_dict['created_at'] = version.created_at.isoformat()
        version_dict['updated_at'] = version.updated_at.isoformat()
        
        with open(version_path, 'w') as f:
            json.dump(version_dict, f, indent=2)


class ModelDeploymentManager:
    """Manages model deployments across different targets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deployments: Dict[str, ModelDeployment] = {}
        self.active_deployments: Dict[str, List[str]] = defaultdict(list)  # model_version_id -> [deployment_ids]
        self.deployment_history: List[ModelDeployment] = []
    
    async def deploy_model(
        self, 
        model_version_id: str,
        target: DeploymentTarget,
        configuration: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy a model version to specified target."""
        deployment_id = f"dep_{uuid.uuid4().hex[:8]}"
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_version_id=model_version_id,
            target=target,
            configuration=configuration or {},
            status="deploying"
        )
        
        # Simulate deployment process
        await self._execute_deployment(deployment)
        
        self.deployments[deployment_id] = deployment
        self.active_deployments[model_version_id].append(deployment_id)
        self.deployment_history.append(deployment)
        
        logger.info(f"Deployed model {model_version_id} to {target.value} as {deployment_id}")
        return deployment_id
    
    async def _execute_deployment(self, deployment: ModelDeployment) -> None:
        """Execute the actual deployment (simulated)."""
        await asyncio.sleep(2.0)  # Simulate deployment time
        
        # Simulate deployment configuration based on target
        if deployment.target == DeploymentTarget.API_ENDPOINT:
            deployment.endpoint_url = f"https://api.models.company.com/v1/{deployment.deployment_id}"
            deployment.health_check_url = f"{deployment.endpoint_url}/health"
        
        elif deployment.target == DeploymentTarget.CLOUD:
            deployment.configuration.update({
                'instance_type': 'm5.large',
                'auto_scaling': True,
                'min_instances': 1,
                'max_instances': 10
            })
        
        elif deployment.target == DeploymentTarget.EDGE:
            deployment.resource_requirements = {
                'cpu_cores': 2,
                'memory_mb': 4096,
                'disk_mb': 10240
            }
        
        deployment.status = "active"
        deployment.deployed_at = datetime.now()
    
    async def undeploy_model(self, deployment_id: str) -> None:
        """Undeploy a model."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Simulate undeployment
        await asyncio.sleep(1.0)
        
        deployment.status = "terminated"
        
        # Remove from active deployments
        if deployment.model_version_id in self.active_deployments:
            self.active_deployments[deployment.model_version_id].remove(deployment_id)
        
        logger.info(f"Undeployed model deployment: {deployment_id}")
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[str]:
        """Get deployment status."""
        deployment = self.deployments.get(deployment_id)
        return deployment.status if deployment else None
    
    async def list_active_deployments(self, model_version_id: Optional[str] = None) -> List[ModelDeployment]:
        """List active deployments."""
        if model_version_id:
            deployment_ids = self.active_deployments.get(model_version_id, [])
            return [self.deployments[dep_id] for dep_id in deployment_ids if self.deployments[dep_id].status == "active"]
        
        return [dep for dep in self.deployments.values() if dep.status == "active"]
    
    async def scale_deployment(self, deployment_id: str, scale_config: Dict[str, Any]) -> None:
        """Scale a deployment."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        if deployment.status != "active":
            raise ValueError(f"Cannot scale non-active deployment: {deployment.status}")
        
        # Update scaling configuration
        deployment.scaling_config.update(scale_config)
        
        # Simulate scaling operation
        await asyncio.sleep(1.0)
        
        logger.info(f"Scaled deployment {deployment_id} with config: {scale_config}")


class ModelMonitoringSystem:
    """Advanced model monitoring and alerting system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.monitoring_active = {}  # deployment_id -> bool
    
    async def start_monitoring(self, deployment_id: str, metrics_config: Dict[str, Any]) -> None:
        """Start monitoring a model deployment."""
        self.monitoring_active[deployment_id] = True
        
        # Set up metric collection (simulated)
        asyncio.create_task(self._collect_metrics(deployment_id, metrics_config))
        
        logger.info(f"Started monitoring for deployment: {deployment_id}")
    
    async def stop_monitoring(self, deployment_id: str) -> None:
        """Stop monitoring a model deployment."""
        self.monitoring_active[deployment_id] = False
        logger.info(f"Stopped monitoring for deployment: {deployment_id}")
    
    async def _collect_metrics(self, deployment_id: str, config: Dict[str, Any]) -> None:
        """Collect metrics for a deployment (simulated)."""
        while self.monitoring_active.get(deployment_id, False):
            # Simulate metric collection
            await asyncio.sleep(config.get('collection_interval', 60))
            
            if not self.monitoring_active.get(deployment_id, False):
                break
            
            # Generate sample metrics
            metrics = await self._generate_sample_metrics(deployment_id)
            
            # Store metrics
            for metric_type, value in metrics.items():
                metric = ModelPerformanceMetrics(
                    model_version_id=f"model_for_{deployment_id}",
                    deployment_id=deployment_id,
                    metric_type=MetricType(metric_type),
                    value=value
                )
                
                self.metrics_history[f"{deployment_id}_{metric_type}"].append(metric)
            
            # Check alert rules
            await self._check_alert_rules(deployment_id, metrics)
    
    async def _generate_sample_metrics(self, deployment_id: str) -> Dict[str, float]:
        """Generate sample metrics for monitoring demonstration."""
        return {
            'accuracy': np.random.uniform(0.85, 0.95),
            'latency': np.random.uniform(50, 200),  # ms
            'throughput': np.random.uniform(100, 500),  # requests/sec
            'memory_usage': np.random.uniform(0.4, 0.8),  # 40-80%
            'cpu_usage': np.random.uniform(0.2, 0.6),  # 20-60%
            'drift_score': np.random.uniform(0.0, 0.3)  # Lower is better
        }
    
    async def add_alert_rule(
        self, 
        rule_name: str, 
        metric_type: MetricType,
        threshold: float,
        condition: str,  # 'greater_than', 'less_than', 'equals'
        deployments: Optional[List[str]] = None
    ) -> None:
        """Add an alert rule for monitoring."""
        self.alert_rules[rule_name] = {
            'metric_type': metric_type,
            'threshold': threshold,
            'condition': condition,
            'deployments': deployments or [],
            'created_at': datetime.now()
        }
        
        logger.info(f"Added alert rule: {rule_name}")
    
    async def _check_alert_rules(self, deployment_id: str, metrics: Dict[str, float]) -> None:
        """Check metrics against alert rules."""
        for rule_name, rule in self.alert_rules.items():
            # Check if rule applies to this deployment
            if rule['deployments'] and deployment_id not in rule['deployments']:
                continue
            
            metric_type = rule['metric_type'].value
            if metric_type not in metrics:
                continue
            
            value = metrics[metric_type]
            threshold = rule['threshold']
            condition = rule['condition']
            
            # Check condition
            alert_triggered = False
            if condition == 'greater_than' and value > threshold:
                alert_triggered = True
            elif condition == 'less_than' and value < threshold:
                alert_triggered = True
            elif condition == 'equals' and abs(value - threshold) < 0.001:
                alert_triggered = True
            
            if alert_triggered:
                await self._trigger_alert(rule_name, deployment_id, metric_type, value, threshold)
    
    async def _trigger_alert(
        self, 
        rule_name: str, 
        deployment_id: str, 
        metric_type: str, 
        value: float, 
        threshold: float
    ) -> None:
        """Trigger an alert."""
        alert = {
            'alert_id': f"alert_{uuid.uuid4().hex[:8]}",
            'rule_name': rule_name,
            'deployment_id': deployment_id,
            'metric_type': metric_type,
            'value': value,
            'threshold': threshold,
            'timestamp': datetime.now(),
            'status': 'active'
        }
        
        self.active_alerts.append(alert)
        
        logger.warning(f"ALERT: {rule_name} - {metric_type} = {value:.3f} (threshold: {threshold})")
    
    async def get_metrics_history(
        self, 
        deployment_id: str, 
        metric_type: MetricType,
        time_range: Optional[timedelta] = None
    ) -> List[ModelPerformanceMetrics]:
        """Get metrics history for a deployment."""
        key = f"{deployment_id}_{metric_type.value}"
        metrics = list(self.metrics_history.get(key, []))
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        return sorted(metrics, key=lambda m: m.timestamp)
    
    async def get_active_alerts(self, deployment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        if deployment_id:
            return [alert for alert in self.active_alerts if alert['deployment_id'] == deployment_id and alert['status'] == 'active']
        
        return [alert for alert in self.active_alerts if alert['status'] == 'active']


class ExperimentTracker:
    """MLOps experiment tracking and management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments: Dict[str, List[ExperimentRun]] = defaultdict(list)
        self.active_runs: Dict[str, ExperimentRun] = {}
    
    async def start_experiment_run(
        self, 
        experiment_name: str,
        parameters: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a new experiment run."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters=parameters,
            tags=tags or {},
            status="running"
        )
        
        self.experiments[experiment_name].append(run)
        self.active_runs[run_id] = run
        
        logger.info(f"Started experiment run: {experiment_name}/{run_id}")
        return run_id
    
    async def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        """Log metrics for an experiment run."""
        run = self.active_runs.get(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found or not active")
        
        run.metrics.update(metrics)
        logger.info(f"Logged metrics for {run_id}: {metrics}")
    
    async def log_artifact(self, run_id: str, artifact_name: str, artifact_path: str) -> None:
        """Log an artifact for an experiment run."""
        run = self.active_runs.get(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found or not active")
        
        run.artifacts.append(f"{artifact_name}:{artifact_path}")
        logger.info(f"Logged artifact for {run_id}: {artifact_name}")
    
    async def finish_experiment_run(self, run_id: str, status: str = "completed") -> None:
        """Finish an experiment run."""
        run = self.active_runs.get(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found or not active")
        
        run.status = status
        run.ended_at = datetime.now()
        
        # Remove from active runs
        del self.active_runs[run_id]
        
        logger.info(f"Finished experiment run: {run_id} with status {status}")
    
    async def get_experiment_runs(
        self, 
        experiment_name: str,
        status: Optional[str] = None
    ) -> List[ExperimentRun]:
        """Get experiment runs for an experiment."""
        runs = self.experiments.get(experiment_name, [])
        
        if status:
            runs = [run for run in runs if run.status == status]
        
        return sorted(runs, key=lambda r: r.started_at, reverse=True)
    
    async def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiment runs."""
        comparison = {
            'runs': {},
            'common_metrics': set(),
            'metric_comparison': {}
        }
        
        all_runs = []
        for exp_runs in self.experiments.values():
            all_runs.extend(exp_runs)
        
        # Find runs by ID
        target_runs = [run for run in all_runs if run.run_id in run_ids]
        
        if len(target_runs) != len(run_ids):
            missing = set(run_ids) - {run.run_id for run in target_runs}
            raise ValueError(f"Runs not found: {missing}")
        
        # Collect run data
        for run in target_runs:
            comparison['runs'][run.run_id] = {
                'experiment_name': run.experiment_name,
                'parameters': run.parameters,
                'metrics': run.metrics,
                'status': run.status,
                'duration': (run.ended_at - run.started_at).total_seconds() if run.ended_at else None
            }
        
        # Find common metrics
        if target_runs:
            comparison['common_metrics'] = set(target_runs[0].metrics.keys())
            for run in target_runs[1:]:
                comparison['common_metrics'] &= set(run.metrics.keys())
        
        # Compare metrics
        for metric in comparison['common_metrics']:
            values = [run.metrics[metric] for run in target_runs]
            comparison['metric_comparison'][metric] = {
                'values': dict(zip(run_ids, values)),
                'best_value': max(values),
                'worst_value': min(values),
                'average': np.mean(values),
                'std_dev': np.std(values)
            }
        
        return comparison


class ABTestManager:
    """A/B testing system for model comparison."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_tests: Dict[str, ABTestConfiguration] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    async def create_ab_test(
        self,
        test_name: str,
        model_a_version_id: str,
        model_b_version_id: str,
        traffic_split: Dict[str, float],
        success_metrics: List[str],
        test_duration: timedelta
    ) -> str:
        """Create a new A/B test."""
        test_id = f"abtest_{uuid.uuid4().hex[:8]}"
        
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")
        
        test_config = ABTestConfiguration(
            test_id=test_id,
            test_name=test_name,
            model_a_version_id=model_a_version_id,
            model_b_version_id=model_b_version_id,
            traffic_split=traffic_split,
            success_metrics=success_metrics,
            test_duration=test_duration
        )
        
        self.active_tests[test_id] = test_config
        
        # Start test execution
        asyncio.create_task(self._run_ab_test(test_config))
        
        logger.info(f"Created A/B test: {test_name} ({test_id})")
        return test_id
    
    async def _run_ab_test(self, test_config: ABTestConfiguration) -> None:
        """Run an A/B test (simulated)."""
        test_id = test_config.test_id
        
        # Simulate test execution
        end_time = test_config.started_at + test_config.test_duration
        
        while datetime.now() < end_time and test_config.status == "active":
            await asyncio.sleep(60)  # Check every minute
            
            # Simulate collecting metrics
            await self._collect_ab_test_metrics(test_config)
        
        # Test completed, analyze results
        if test_config.status == "active":
            await self._analyze_ab_test_results(test_config)
            test_config.status = "completed"
    
    async def _collect_ab_test_metrics(self, test_config: ABTestConfiguration) -> None:
        """Collect metrics for A/B test (simulated)."""
        test_id = test_config.test_id
        
        if test_id not in self.test_results:
            self.test_results[test_id] = {
                'A': {metric: [] for metric in test_config.success_metrics},
                'B': {metric: [] for metric in test_config.success_metrics}
            }
        
        # Simulate metric collection for both variants
        for variant in ['A', 'B']:
            for metric in test_config.success_metrics:
                # Generate sample metric values (model B slightly better)
                if variant == 'A':
                    value = np.random.normal(0.8, 0.1)  # Mean 0.8
                else:
                    value = np.random.normal(0.82, 0.1)  # Mean 0.82 (slightly better)
                
                value = max(0, min(1, value))  # Clamp to [0, 1]
                self.test_results[test_id][variant][metric].append(value)
    
    async def _analyze_ab_test_results(self, test_config: ABTestConfiguration) -> None:
        """Analyze A/B test results for statistical significance."""
        test_id = test_config.test_id
        results = self.test_results[test_id]
        
        analysis = {}
        
        for metric in test_config.success_metrics:
            values_a = results['A'][metric]
            values_b = results['B'][metric]
            
            if not values_a or not values_b:
                continue
            
            # Simple statistical analysis
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a)
            std_b = np.std(values_b)
            
            # Calculate effect size and confidence
            effect_size = (mean_b - mean_a) / np.sqrt((std_a**2 + std_b**2) / 2)
            
            # Simplified statistical significance (normally would use proper t-test)
            sample_size = min(len(values_a), len(values_b))
            confidence = min(0.99, abs(effect_size) * np.sqrt(sample_size) / 2.0)
            
            analysis[metric] = {
                'variant_a_mean': float(mean_a),
                'variant_b_mean': float(mean_b),
                'effect_size': float(effect_size),
                'confidence': float(confidence),
                'winner': 'B' if mean_b > mean_a else 'A',
                'improvement': float(abs(mean_b - mean_a) / mean_a * 100)  # Percentage improvement
            }
        
        test_config.results = analysis
        
        # Calculate overall statistical significance
        confidences = [analysis[metric]['confidence'] for metric in analysis]
        test_config.statistical_significance = np.mean(confidences) if confidences else 0.0
        
        logger.info(f"A/B test {test_id} completed. Statistical significance: {test_config.statistical_significance:.3f}")
    
    async def get_ab_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test status and results."""
        test = self.active_tests.get(test_id)
        if not test:
            return None
        
        return {
            'test_id': test.test_id,
            'test_name': test.test_name,
            'status': test.status,
            'started_at': test.started_at.isoformat(),
            'duration': test.test_duration.total_seconds(),
            'traffic_split': test.traffic_split,
            'results': test.results,
            'statistical_significance': test.statistical_significance
        }
    
    async def stop_ab_test(self, test_id: str) -> None:
        """Stop an active A/B test."""
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        test.status = "stopped"
        
        # Analyze current results
        await self._analyze_ab_test_results(test)
        
        logger.info(f"Stopped A/B test: {test_id}")


class AIModelManagementPlatform:
    """Complete AI model management platform orchestrator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.registry = ModelRegistry(config.get('registry', {}))
        self.deployment_manager = ModelDeploymentManager(config.get('deployment', {}))
        self.monitoring_system = ModelMonitoringSystem(config.get('monitoring', {}))
        self.experiment_tracker = ExperimentTracker(config.get('experiments', {}))
        self.ab_test_manager = ABTestManager(config.get('ab_testing', {}))
        
        # Platform statistics
        self.platform_stats = {
            'total_models': 0,
            'total_versions': 0,
            'active_deployments': 0,
            'total_experiments': 0,
            'active_ab_tests': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the AI model management platform."""
        logger.info("Initializing AI Model Management Platform...")
        
        # Set up default alert rules
        await self.monitoring_system.add_alert_rule(
            "high_latency", MetricType.LATENCY, 500.0, "greater_than"
        )
        await self.monitoring_system.add_alert_rule(
            "low_accuracy", MetricType.ACCURACY, 0.8, "less_than"
        )
        await self.monitoring_system.add_alert_rule(
            "high_drift", MetricType.DRIFT_SCORE, 0.3, "greater_than"
        )
        
        logger.info("AI Model Management Platform initialized successfully")
    
    async def create_model_pipeline(
        self,
        model_metadata: ModelMetadata,
        model_artifact: Any,
        deployment_target: DeploymentTarget,
        monitoring_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create complete model pipeline: register -> deploy -> monitor."""
        pipeline_result = {}
        
        # Step 1: Register model
        model_id = await self.registry.register_model(model_metadata)
        pipeline_result['model_id'] = model_id
        
        # Step 2: Create version with artifact
        version_id = await self.registry.create_model_version(
            model_id, "1.0.0", model_artifact, commit_message="Initial deployment version"
        )
        pipeline_result['version_id'] = version_id
        
        # Step 3: Deploy model
        deployment_id = await self.deployment_manager.deploy_model(
            version_id, deployment_target
        )
        pipeline_result['deployment_id'] = deployment_id
        
        # Step 4: Start monitoring
        if monitoring_config:
            await self.monitoring_system.start_monitoring(deployment_id, monitoring_config)
            pipeline_result['monitoring'] = "started"
        
        # Step 5: Update model status to production
        await self.registry.update_model_status(model_id, "1.0.0", ModelStatus.PRODUCTION)
        
        logger.info(f"Created complete model pipeline: {pipeline_result}")
        return pipeline_result
    
    async def promote_model_version(
        self,
        model_id: str,
        version_number: str,
        target_environment: str = "production"
    ) -> str:
        """Promote a model version through environments."""
        # Update status based on target environment
        status_mapping = {
            "staging": ModelStatus.STAGING,
            "production": ModelStatus.PRODUCTION,
            "testing": ModelStatus.TESTING
        }
        
        target_status = status_mapping.get(target_environment, ModelStatus.STAGING)
        await self.registry.update_model_status(model_id, version_number, target_status)
        
        # If promoting to production, handle existing production deployments
        if target_environment == "production":
            version_id = f"{model_id}_{version_number}"
            
            # Deploy to production target
            deployment_id = await self.deployment_manager.deploy_model(
                version_id, DeploymentTarget.CLOUD
            )
            
            # Start monitoring
            await self.monitoring_system.start_monitoring(deployment_id, {
                'collection_interval': 300,  # 5 minutes
                'metrics': ['accuracy', 'latency', 'throughput']
            })
            
            logger.info(f"Promoted {model_id} v{version_number} to production")
            return deployment_id
        
        return f"promoted_to_{target_environment}"
    
    async def compare_model_versions(
        self,
        model_id: str,
        version_a: str,
        version_b: str,
        test_duration_minutes: int = 60
    ) -> str:
        """Compare two model versions using A/B testing."""
        version_a_id = f"{model_id}_{version_a}"
        version_b_id = f"{model_id}_{version_b}"
        
        # Create A/B test
        test_id = await self.ab_test_manager.create_ab_test(
            test_name=f"{model_id}_v{version_a}_vs_v{version_b}",
            model_a_version_id=version_a_id,
            model_b_version_id=version_b_id,
            traffic_split={"A": 0.5, "B": 0.5},
            success_metrics=["accuracy", "latency"],
            test_duration=timedelta(minutes=test_duration_minutes)
        )
        
        logger.info(f"Started model version comparison: {test_id}")
        return test_id
    
    async def get_platform_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive platform dashboard data."""
        # Update statistics
        self.platform_stats['total_models'] = len(self.registry.models)
        self.platform_stats['total_versions'] = sum(len(versions) for versions in self.registry.versions.values())
        self.platform_stats['active_deployments'] = len([d for d in self.deployment_manager.deployments.values() if d.status == "active"])
        self.platform_stats['total_experiments'] = sum(len(runs) for runs in self.experiment_tracker.experiments.values())
        self.platform_stats['active_ab_tests'] = len([t for t in self.ab_test_manager.active_tests.values() if t.status == "active"])
        
        # Get recent activities
        recent_deployments = sorted(
            self.deployment_manager.deployment_history[-10:], 
            key=lambda d: d.deployed_at or datetime.min, 
            reverse=True
        )
        
        active_alerts = await self.monitoring_system.get_active_alerts()
        
        return {
            'platform_stats': self.platform_stats,
            'recent_deployments': [
                {
                    'deployment_id': d.deployment_id,
                    'model_version_id': d.model_version_id,
                    'target': d.target.value,
                    'status': d.status,
                    'deployed_at': d.deployed_at.isoformat() if d.deployed_at else None
                } for d in recent_deployments[:5]
            ],
            'active_alerts': active_alerts[:10],
            'system_health': {
                'registry_operational': True,
                'deployment_manager_operational': True,
                'monitoring_operational': True,
                'last_updated': datetime.now().isoformat()
            }
        }
    
    async def cleanup_old_resources(self, retention_days: int = 30) -> Dict[str, int]:
        """Cleanup old resources based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleanup_stats = {
            'archived_models': 0,
            'cleaned_deployments': 0,
            'purged_metrics': 0
        }
        
        # Archive old models
        for model_id, metadata in list(self.registry.models.items()):
            if metadata.created_at < cutoff_date:
                # Check if any versions are still in production
                has_production_version = False
                for version in self.registry.versions.get(model_id, {}).values():
                    if version.status == ModelStatus.PRODUCTION:
                        has_production_version = True
                        break
                
                if not has_production_version:
                    # Archive model (in real implementation, move to archive storage)
                    cleanup_stats['archived_models'] += 1
        
        # Clean up terminated deployments
        for deployment_id, deployment in list(self.deployment_manager.deployments.items()):
            if deployment.status == "terminated" and deployment.deployed_at and deployment.deployed_at < cutoff_date:
                # Remove old terminated deployments
                del self.deployment_manager.deployments[deployment_id]
                cleanup_stats['cleaned_deployments'] += 1
        
        # Purge old metrics (keep only recent data)
        for key, metrics_queue in self.monitoring_system.metrics_history.items():
            original_size = len(metrics_queue)
            # Filter metrics to keep only recent ones
            recent_metrics = [m for m in metrics_queue if m.timestamp >= cutoff_date]
            metrics_queue.clear()
            metrics_queue.extend(recent_metrics)
            cleanup_stats['purged_metrics'] += original_size - len(recent_metrics)
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    async def shutdown(self) -> None:
        """Shutdown the AI model management platform."""
        logger.info("Shutting down AI Model Management Platform...")
        
        # Stop all monitoring
        for deployment_id in list(self.monitoring_system.monitoring_active.keys()):
            await self.monitoring_system.stop_monitoring(deployment_id)
        
        # Stop all active A/B tests
        for test_id in list(self.ab_test_manager.active_tests.keys()):
            await self.ab_test_manager.stop_ab_test(test_id)
        
        logger.info("AI Model Management Platform shutdown complete")


# Example usage and testing
async def example_model_management():
    """Example of comprehensive AI model management."""
    config = {
        'registry': {
            'storage_path': './model_registry_demo'
        },
        'deployment': {},
        'monitoring': {},
        'experiments': {},
        'ab_testing': {}
    }
    
    # Initialize platform
    platform = AIModelManagementPlatform(config)
    await platform.initialize()
    
    # Create sample model metadata
    model_metadata = ModelMetadata(
        name="customer_churn_predictor",
        version="1.0.0",
        model_type=ModelType.CLASSIFICATION,
        framework="scikit-learn",
        algorithm="RandomForest",
        created_by="data_scientist@company.com",
        created_at=datetime.now(),
        description="Predicts customer churn based on usage patterns",
        tags=["churn", "classification", "business-critical"],
        hyperparameters={"n_estimators": 100, "max_depth": 10},
        training_config={"train_size": 0.8, "validation_size": 0.2}
    )
    
    # Create sample model artifact (simulated)
    model_artifact = {"model_type": "sklearn", "trained_model": "serialized_model_data"}
    
    # Create complete model pipeline
    pipeline_result = await platform.create_model_pipeline(
        model_metadata=model_metadata,
        model_artifact=model_artifact,
        deployment_target=DeploymentTarget.CLOUD,
        monitoring_config={'collection_interval': 60, 'metrics': ['accuracy', 'latency']}
    )
    
    logger.info(f"Model pipeline created: {pipeline_result}")
    
    # Start an experiment
    run_id = await platform.experiment_tracker.start_experiment_run(
        "hyperparameter_tuning",
        parameters={"n_estimators": 150, "max_depth": 15},
        tags={"experiment_type": "hyperparameter_optimization"}
    )
    
    # Log some metrics
    await platform.experiment_tracker.log_metrics(run_id, {
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.89
    })
    
    await platform.experiment_tracker.finish_experiment_run(run_id)
    
    # Wait a bit for monitoring to collect some data
    await asyncio.sleep(5)
    
    # Get platform dashboard
    dashboard = await platform.get_platform_dashboard()
    logger.info(f"Platform dashboard: {dashboard}")
    
    # Cleanup
    await platform.shutdown()


if __name__ == "__main__":
    asyncio.run(example_model_management())