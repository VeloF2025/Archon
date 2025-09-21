"""
Model Management System for Archon Enhancement 2025 Phase 4

Enterprise-grade model lifecycle management with:
- Model versioning and lineage tracking
- A/B testing and gradual rollouts
- Model performance monitoring and drift detection
- Automated model registry and metadata management
- Model approval workflows and governance
- Resource optimization and cost tracking
- Multi-environment deployment management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import uuid
from pathlib import Path
import shutil
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model lifecycle status"""
    REGISTERED = "registered"
    TRAINING = "training"
    VALIDATING = "validating"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"

class ModelType(Enum):
    """Model categorization"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE = "generative"
    ANOMALY_DETECTION = "anomaly_detection"

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    SHADOW = "shadow"

class ApprovalStatus(Enum):
    """Model approval workflow status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    CONDITIONALLY_APPROVED = "conditionally_approved"

class ModelDriftType(Enum):
    """Types of model drift detection"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    name: str
    description: str
    model_type: ModelType
    framework: str
    algorithm: str
    version: str
    author: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    target_variable: Optional[str] = None
    training_dataset: Optional[str] = None
    validation_dataset: Optional[str] = None
    test_dataset: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "framework": self.framework,
            "algorithm": self.algorithm,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "features": self.features,
            "target_variable": self.target_variable,
            "training_dataset": self.training_dataset,
            "validation_dataset": self.validation_dataset,
            "test_dataset": self.test_dataset,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "requirements": self.requirements,
            "license": self.license
        }

@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    status: ModelStatus
    metadata: ModelMetadata
    model_path: str
    checksum: str
    file_size: int
    parent_version: Optional[str] = None
    deployment_environments: Set[DeploymentEnvironment] = field(default_factory=set)
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approval_notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "model_path": self.model_path,
            "checksum": self.checksum,
            "file_size": self.file_size,
            "parent_version": self.parent_version,
            "deployment_environments": [env.value for env in self.deployment_environments],
            "approval_status": self.approval_status.value,
            "approval_notes": self.approval_notes,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ModelConfig:
    """Model deployment and configuration"""
    model_id: str
    version: str
    environment: DeploymentEnvironment
    replicas: int = 1
    resource_limits: Dict[str, str] = field(default_factory=lambda: {
        "cpu": "1",
        "memory": "2Gi",
        "gpu": "0"
    })
    scaling_config: Dict[str, Any] = field(default_factory=lambda: {
        "min_replicas": 1,
        "max_replicas": 10,
        "target_cpu_utilization": 70,
        "scale_down_stabilization": "5m"
    })
    monitoring_config: Dict[str, Any] = field(default_factory=lambda: {
        "enable_metrics": True,
        "enable_logging": True,
        "log_level": "INFO",
        "alert_thresholds": {
            "error_rate": 0.05,
            "latency_p99": 1000,
            "cpu_utilization": 80,
            "memory_utilization": 80
        }
    })
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "environment": self.environment.value,
            "replicas": self.replicas,
            "resource_limits": self.resource_limits,
            "scaling_config": self.scaling_config,
            "monitoring_config": self.monitoring_config,
            "feature_flags": self.feature_flags
        }

@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking"""
    model_id: str
    version: str
    environment: DeploymentEnvironment
    timestamp: datetime
    predictions_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_overall_score(self) -> float:
        """Calculate overall performance score"""
        # Weighted combination of key metrics
        performance_score = (0.3 * self.accuracy + 0.2 * self.f1_score + 
                           0.2 * (1 - self.error_rate) + 0.15 * (1000 / max(self.latency_p95, 1)) +
                           0.15 * (self.throughput / 100))
        return min(performance_score, 1.0)

@dataclass
class DriftDetectionResult:
    """Model drift detection results"""
    model_id: str
    version: str
    drift_type: ModelDriftType
    severity: float  # 0.0 = no drift, 1.0 = severe drift
    detected_at: datetime
    affected_features: List[str] = field(default_factory=list)
    statistical_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendation: str = ""
    alert_sent: bool = False
    
    def is_significant(self, threshold: float = 0.3) -> bool:
        """Check if drift is significant"""
        return self.severity >= threshold

class ModelManager:
    """
    Enterprise-grade model lifecycle management system with comprehensive
    versioning, deployment, monitoring, and governance capabilities.
    """
    
    def __init__(self, base_path: str = "./models", registry_backend: str = "local"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.registry_backend = registry_backend
        
        # Model registry storage
        self.models: Dict[str, List[ModelVersion]] = {}
        self.active_deployments: Dict[str, ModelConfig] = {}
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.drift_history: Dict[str, List[DriftDetectionResult]] = {}
        
        # Approval workflows
        self.pending_approvals: Dict[str, ModelVersion] = {}
        self.approval_rules: Dict[str, Any] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_tasks: Dict[str, Any] = {}
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        logger.info("Model Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the model manager"""
        try:
            await self._load_models()
            await self._setup_approval_rules()
            await self._start_monitoring_tasks()
            logger.info("Model Manager initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Model Manager: {e}")
            raise
    
    async def register_model(self, 
                           model_path: str,
                           metadata: ModelMetadata,
                           parent_version: Optional[str] = None) -> str:
        """Register a new model version in the registry"""
        try:
            model_id = f"{metadata.name}_{metadata.algorithm}"
            version_id = f"{model_id}_v{metadata.version}"
            
            # Calculate model checksum and size
            checksum = await self._calculate_checksum(model_path)
            file_size = Path(model_path).stat().st_size
            
            # Create model version
            model_version = ModelVersion(
                model_id=model_id,
                version=metadata.version,
                status=ModelStatus.REGISTERED,
                metadata=metadata,
                model_path=model_path,
                checksum=checksum,
                file_size=file_size,
                parent_version=parent_version
            )
            
            with self._lock:
                # Initialize model list if needed
                if model_id not in self.models:
                    self.models[model_id] = []
                    self.performance_history[model_id] = []
                    self.drift_history[model_id] = []
                
                # Check for version conflicts
                existing_versions = [v.version for v in self.models[model_id]]
                if metadata.version in existing_versions:
                    raise ValueError(f"Version {metadata.version} already exists for model {model_id}")
                
                # Add new version
                self.models[model_id].append(model_version)
                
                # Store model file in registry
                await self._store_model_file(model_version)
                
                # Queue for approval if rules exist
                if await self._requires_approval(model_version):
                    self.pending_approvals[version_id] = model_version
                    model_version.approval_status = ApprovalStatus.PENDING
                else:
                    model_version.approval_status = ApprovalStatus.APPROVED
                
                logger.info(f"Registered model: {model_id} v{metadata.version}")
                return version_id
                
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def approve_model(self, 
                          model_id: str,
                          version: str,
                          approver: str,
                          notes: str = "") -> bool:
        """Approve a model version for deployment"""
        try:
            version_id = f"{model_id}_v{version}"
            
            with self._lock:
                if version_id not in self.pending_approvals:
                    raise ValueError(f"No pending approval for {version_id}")
                
                model_version = self.pending_approvals[version_id]
                model_version.approval_status = ApprovalStatus.APPROVED
                model_version.approval_notes = f"Approved by {approver}: {notes}"
                
                # Remove from pending
                del self.pending_approvals[version_id]
                
                # Update status to staging
                model_version.status = ModelStatus.STAGING
                
                logger.info(f"Approved model: {version_id} by {approver}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to approve model {model_id} v{version}: {e}")
            return False
    
    async def deploy_model(self, 
                         model_id: str,
                         version: str,
                         environment: DeploymentEnvironment,
                         config: Optional[ModelConfig] = None) -> str:
        """Deploy a model version to specified environment"""
        try:
            # Find model version
            model_version = await self._find_model_version(model_id, version)
            if not model_version:
                raise ValueError(f"Model {model_id} v{version} not found")
            
            # Check approval status
            if model_version.approval_status != ApprovalStatus.APPROVED:
                raise ValueError(f"Model {model_id} v{version} is not approved for deployment")
            
            # Create deployment configuration
            if config is None:
                config = ModelConfig(
                    model_id=model_id,
                    version=version,
                    environment=environment
                )
            
            deployment_id = f"{model_id}_v{version}_{environment.value}"
            
            with self._lock:
                # Store deployment configuration
                self.active_deployments[deployment_id] = config
                
                # Update model version status
                model_version.deployment_environments.add(environment)
                
                if environment == DeploymentEnvironment.PRODUCTION:
                    model_version.status = ModelStatus.PRODUCTION
                elif environment == DeploymentEnvironment.STAGING:
                    model_version.status = ModelStatus.STAGING
                
                # Start monitoring for this deployment
                await self._start_deployment_monitoring(deployment_id, config)
                
                logger.info(f"Deployed model: {deployment_id}")
                return deployment_id
                
        except Exception as e:
            logger.error(f"Failed to deploy model {model_id} v{version}: {e}")
            raise
    
    async def create_ab_test(self,
                           test_name: str,
                           model_a: Tuple[str, str],  # (model_id, version)
                           model_b: Tuple[str, str],  # (model_id, version)
                           traffic_split: float = 0.5,
                           environment: DeploymentEnvironment = DeploymentEnvironment.STAGING,
                           duration_days: int = 7) -> str:
        """Create A/B test between two model versions"""
        try:
            test_id = f"ab_test_{test_name}_{uuid.uuid4().hex[:8]}"
            
            # Validate models exist and are approved
            model_a_version = await self._find_model_version(model_a[0], model_a[1])
            model_b_version = await self._find_model_version(model_b[0], model_b[1])
            
            if not model_a_version or not model_b_version:
                raise ValueError("One or both models not found")
            
            if (model_a_version.approval_status != ApprovalStatus.APPROVED or 
                model_b_version.approval_status != ApprovalStatus.APPROVED):
                raise ValueError("Both models must be approved for A/B testing")
            
            # Create A/B test configuration
            ab_test_config = {
                "test_id": test_id,
                "test_name": test_name,
                "model_a": {
                    "model_id": model_a[0],
                    "version": model_a[1],
                    "traffic_percentage": traffic_split * 100
                },
                "model_b": {
                    "model_id": model_b[0],
                    "version": model_b[1],
                    "traffic_percentage": (1 - traffic_split) * 100
                },
                "environment": environment.value,
                "start_date": datetime.utcnow(),
                "end_date": datetime.utcnow() + timedelta(days=duration_days),
                "status": "running",
                "metrics": {
                    "model_a": {"requests": 0, "errors": 0, "latency_sum": 0.0},
                    "model_b": {"requests": 0, "errors": 0, "latency_sum": 0.0}
                }
            }
            
            with self._lock:
                self.ab_tests[test_id] = ab_test_config
                
                logger.info(f"Created A/B test: {test_id}")
                return test_id
                
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise
    
    async def detect_model_drift(self,
                               model_id: str,
                               version: str,
                               reference_data: Any,
                               current_data: Any,
                               drift_threshold: float = 0.3) -> List[DriftDetectionResult]:
        """Detect various types of model drift"""
        try:
            drift_results = []
            
            # Data drift detection (feature distribution changes)
            data_drift = await self._detect_data_drift(
                model_id, version, reference_data, current_data
            )
            if data_drift.severity >= drift_threshold:
                drift_results.append(data_drift)
            
            # Performance drift detection (model accuracy degradation)
            performance_drift = await self._detect_performance_drift(
                model_id, version
            )
            if performance_drift and performance_drift.severity >= drift_threshold:
                drift_results.append(performance_drift)
            
            # Prediction drift detection (output distribution changes)
            prediction_drift = await self._detect_prediction_drift(
                model_id, version, reference_data, current_data
            )
            if prediction_drift.severity >= drift_threshold:
                drift_results.append(prediction_drift)
            
            # Store drift results
            with self._lock:
                self.drift_history[model_id].extend(drift_results)
            
            # Send alerts for significant drift
            for drift in drift_results:
                if drift.is_significant(drift_threshold):
                    await self._send_drift_alert(drift)
            
            logger.info(f"Detected {len(drift_results)} drift issues for {model_id} v{version}")
            return drift_results
            
        except Exception as e:
            logger.error(f"Failed to detect drift for {model_id} v{version}: {e}")
            raise
    
    async def rollback_deployment(self,
                                deployment_id: str,
                                target_version: Optional[str] = None) -> bool:
        """Rollback model deployment to previous version"""
        try:
            with self._lock:
                if deployment_id not in self.active_deployments:
                    raise ValueError(f"Deployment {deployment_id} not found")
                
                current_config = self.active_deployments[deployment_id]
                
                if target_version is None:
                    # Find previous version
                    target_version = await self._find_previous_version(
                        current_config.model_id, current_config.version
                    )
                
                if not target_version:
                    raise ValueError("No previous version available for rollback")
                
                # Create rollback configuration
                rollback_config = ModelConfig(
                    model_id=current_config.model_id,
                    version=target_version,
                    environment=current_config.environment,
                    replicas=current_config.replicas,
                    resource_limits=current_config.resource_limits,
                    scaling_config=current_config.scaling_config,
                    monitoring_config=current_config.monitoring_config,
                    feature_flags=current_config.feature_flags
                )
                
                # Execute rollback
                new_deployment_id = f"{current_config.model_id}_v{target_version}_{current_config.environment.value}"
                self.active_deployments[new_deployment_id] = rollback_config
                
                # Remove old deployment
                del self.active_deployments[deployment_id]
                
                logger.info(f"Rolled back {deployment_id} to version {target_version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback deployment {deployment_id}: {e}")
            return False
    
    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete lineage of a model including all versions and relationships"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            versions = self.models[model_id]
            lineage = {
                "model_id": model_id,
                "total_versions": len(versions),
                "versions": [],
                "lineage_graph": {}
            }
            
            # Build version information
            for version in sorted(versions, key=lambda v: v.created_at):
                version_info = {
                    "version": version.version,
                    "status": version.status.value,
                    "created_at": version.created_at.isoformat(),
                    "parent_version": version.parent_version,
                    "approval_status": version.approval_status.value,
                    "deployments": [env.value for env in version.deployment_environments],
                    "metrics": version.metadata.metrics,
                    "author": version.metadata.author
                }
                lineage["versions"].append(version_info)
                
                # Build lineage relationships
                if version.parent_version:
                    if version.parent_version not in lineage["lineage_graph"]:
                        lineage["lineage_graph"][version.parent_version] = []
                    lineage["lineage_graph"][version.parent_version].append(version.version)
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get lineage for {model_id}: {e}")
            raise
    
    async def get_model_performance_summary(self, 
                                          model_id: str,
                                          days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance summary for a model"""
        try:
            if model_id not in self.performance_history:
                raise ValueError(f"No performance history for {model_id}")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            recent_metrics = [
                m for m in self.performance_history[model_id]
                if m.timestamp >= cutoff_date
            ]
            
            if not recent_metrics:
                return {"model_id": model_id, "message": "No recent performance data"}
            
            # Calculate aggregated metrics
            total_predictions = sum(m.predictions_count for m in recent_metrics)
            avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
            avg_f1_score = np.mean([m.f1_score for m in recent_metrics])
            avg_latency = np.mean([m.latency_p95 for m in recent_metrics])
            avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
            
            # Performance trends
            accuracy_trend = self._calculate_trend([m.accuracy for m in recent_metrics[-7:]])
            latency_trend = self._calculate_trend([m.latency_p95 for m in recent_metrics[-7:]])
            
            return {
                "model_id": model_id,
                "time_period_days": days,
                "total_predictions": total_predictions,
                "average_accuracy": avg_accuracy,
                "average_f1_score": avg_f1_score,
                "average_latency_p95_ms": avg_latency,
                "average_error_rate": avg_error_rate,
                "performance_trends": {
                    "accuracy_trend": accuracy_trend,
                    "latency_trend": latency_trend
                },
                "environment_breakdown": self._get_environment_breakdown(recent_metrics),
                "drift_alerts": len([d for d in self.drift_history[model_id] 
                                   if d.detected_at >= cutoff_date]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary for {model_id}: {e}")
            raise
    
    async def list_models(self, 
                        status_filter: Optional[ModelStatus] = None,
                        model_type_filter: Optional[ModelType] = None,
                        environment_filter: Optional[DeploymentEnvironment] = None) -> List[Dict[str, Any]]:
        """List all models with optional filtering"""
        try:
            models_list = []
            
            for model_id, versions in self.models.items():
                latest_version = max(versions, key=lambda v: v.created_at)
                
                # Apply filters
                if status_filter and latest_version.status != status_filter:
                    continue
                if model_type_filter and latest_version.metadata.model_type != model_type_filter:
                    continue
                if environment_filter and environment_filter not in latest_version.deployment_environments:
                    continue
                
                # Get recent performance
                recent_performance = None
                if model_id in self.performance_history and self.performance_history[model_id]:
                    recent_performance = max(self.performance_history[model_id], 
                                          key=lambda p: p.timestamp)
                
                model_info = {
                    "model_id": model_id,
                    "name": latest_version.metadata.name,
                    "model_type": latest_version.metadata.model_type.value,
                    "framework": latest_version.metadata.framework,
                    "latest_version": latest_version.version,
                    "status": latest_version.status.value,
                    "approval_status": latest_version.approval_status.value,
                    "deployments": [env.value for env in latest_version.deployment_environments],
                    "total_versions": len(versions),
                    "author": latest_version.metadata.author,
                    "created_at": latest_version.created_at.isoformat(),
                    "performance": {
                        "accuracy": recent_performance.accuracy if recent_performance else None,
                        "f1_score": recent_performance.f1_score if recent_performance else None,
                        "overall_score": recent_performance.get_overall_score() if recent_performance else None
                    }
                }
                
                models_list.append(model_info)
            
            # Sort by overall performance score
            return sorted(models_list, 
                         key=lambda x: x["performance"]["overall_score"] or 0, 
                         reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get model manager performance metrics"""
        try:
            total_models = len(self.models)
            total_versions = sum(len(versions) for versions in self.models.values())
            active_deployments = len(self.active_deployments)
            pending_approvals = len(self.pending_approvals)
            active_ab_tests = len([t for t in self.ab_tests.values() if t["status"] == "running"])
            
            # Status distribution
            status_dist = {}
            for versions in self.models.values():
                for version in versions:
                    status = version.status.value
                    status_dist[status] = status_dist.get(status, 0) + 1
            
            # Environment distribution
            env_dist = {}
            for config in self.active_deployments.values():
                env = config.environment.value
                env_dist[env] = env_dist.get(env, 0) + 1
            
            # Recent drift alerts
            recent_drift = 0
            cutoff = datetime.utcnow() - timedelta(days=7)
            for drift_list in self.drift_history.values():
                recent_drift += len([d for d in drift_list if d.detected_at >= cutoff])
            
            return {
                "total_models": total_models,
                "total_versions": total_versions,
                "active_deployments": active_deployments,
                "pending_approvals": pending_approvals,
                "active_ab_tests": active_ab_tests,
                "status_distribution": status_dist,
                "environment_distribution": env_dist,
                "recent_drift_alerts": recent_drift,
                "registry_backend": self.registry_backend,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the model manager"""
        try:
            self._shutdown = True
            
            # Cancel monitoring tasks
            for task_id, task in self.monitoring_tasks.items():
                if hasattr(task, 'cancel'):
                    task.cancel()
                logger.info(f"Cancelled monitoring task: {task_id}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Save state
            await self._save_manager_state()
            
            logger.info("Model Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Model Manager shutdown: {e}")
    
    # Private helper methods
    
    async def _find_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Find specific model version"""
        if model_id not in self.models:
            return None
        
        for model_version in self.models[model_id]:
            if model_version.version == version:
                return model_version
        
        return None
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def _store_model_file(self, model_version: ModelVersion) -> None:
        """Store model file in registry storage"""
        registry_path = self.base_path / model_version.model_id / model_version.version
        registry_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model file to registry
        source_path = Path(model_version.model_path)
        target_path = registry_path / source_path.name
        shutil.copy2(source_path, target_path)
        
        # Update model path to registry location
        model_version.model_path = str(target_path)
        
        # Save metadata
        metadata_path = registry_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_version.to_dict(), f, indent=2)
    
    async def _requires_approval(self, model_version: ModelVersion) -> bool:
        """Check if model version requires approval"""
        # Simplified approval logic - can be expanded
        return (model_version.metadata.model_type in [ModelType.CLASSIFICATION, ModelType.NLP] or
                "production" in model_version.metadata.tags)
    
    async def _setup_approval_rules(self) -> None:
        """Setup model approval rules"""
        self.approval_rules = {
            "require_approval_for_types": [ModelType.CLASSIFICATION, ModelType.NLP],
            "require_approval_for_environments": [DeploymentEnvironment.PRODUCTION],
            "auto_approve_below_size": 100 * 1024 * 1024,  # 100MB
            "require_performance_threshold": 0.85
        }
    
    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks"""
        # Performance monitoring task
        monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        self.monitoring_tasks["performance_monitor"] = monitor_task
        
        # Drift detection task  
        drift_task = asyncio.create_task(self._drift_monitoring_loop())
        self.monitoring_tasks["drift_monitor"] = drift_task
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task for monitoring model performance"""
        while not self._shutdown:
            try:
                # Simulate performance monitoring
                await asyncio.sleep(60)  # Check every minute
                
                for deployment_id, config in self.active_deployments.items():
                    # Simulate collecting performance metrics
                    metrics = await self._collect_performance_metrics(deployment_id)
                    
                    if metrics:
                        with self._lock:
                            if config.model_id not in self.performance_history:
                                self.performance_history[config.model_id] = []
                            self.performance_history[config.model_id].append(metrics)
                            
                            # Keep only recent metrics (last 30 days)
                            cutoff = datetime.utcnow() - timedelta(days=30)
                            self.performance_history[config.model_id] = [
                                m for m in self.performance_history[config.model_id]
                                if m.timestamp >= cutoff
                            ]
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _drift_monitoring_loop(self) -> None:
        """Background task for monitoring model drift"""
        while not self._shutdown:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                # Simulate drift monitoring for active deployments
                for deployment_id, config in self.active_deployments.items():
                    if config.environment == DeploymentEnvironment.PRODUCTION:
                        # Simulate drift detection
                        drift_severity = np.random.random()
                        
                        if drift_severity > 0.3:  # Significant drift
                            drift_result = DriftDetectionResult(
                                model_id=config.model_id,
                                version=config.version,
                                drift_type=ModelDriftType.PERFORMANCE_DRIFT,
                                severity=drift_severity,
                                detected_at=datetime.utcnow(),
                                recommendation="Consider retraining or rolling back"
                            )
                            
                            with self._lock:
                                if config.model_id not in self.drift_history:
                                    self.drift_history[config.model_id] = []
                                self.drift_history[config.model_id].append(drift_result)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _detect_data_drift(self, model_id: str, version: str, 
                               reference_data: Any, current_data: Any) -> DriftDetectionResult:
        """Detect data drift between reference and current datasets"""
        # Simplified drift detection - in production would use statistical tests
        severity = np.random.uniform(0.0, 0.8)
        
        return DriftDetectionResult(
            model_id=model_id,
            version=version,
            drift_type=ModelDriftType.DATA_DRIFT,
            severity=severity,
            detected_at=datetime.utcnow(),
            statistical_tests={"ks_test": {"statistic": severity, "p_value": 1-severity}},
            recommendation="Check feature distributions" if severity > 0.3 else "No action needed"
        )
    
    async def _detect_performance_drift(self, model_id: str, version: str) -> Optional[DriftDetectionResult]:
        """Detect performance drift in model accuracy"""
        if model_id not in self.performance_history:
            return None
        
        recent_metrics = self.performance_history[model_id][-10:]  # Last 10 measurements
        if len(recent_metrics) < 5:
            return None
        
        # Calculate performance trend
        accuracies = [m.accuracy for m in recent_metrics]
        trend = self._calculate_trend(accuracies)
        
        if trend < -0.05:  # Significant decline
            return DriftDetectionResult(
                model_id=model_id,
                version=version,
                drift_type=ModelDriftType.PERFORMANCE_DRIFT,
                severity=abs(trend) * 10,  # Scale to 0-1
                detected_at=datetime.utcnow(),
                recommendation="Model performance declining - consider retraining"
            )
        
        return None
    
    async def _detect_prediction_drift(self, model_id: str, version: str,
                                     reference_data: Any, current_data: Any) -> DriftDetectionResult:
        """Detect drift in model predictions"""
        # Simplified prediction drift detection
        severity = np.random.uniform(0.0, 0.6)
        
        return DriftDetectionResult(
            model_id=model_id,
            version=version,
            drift_type=ModelDriftType.PREDICTION_DRIFT,
            severity=severity,
            detected_at=datetime.utcnow(),
            recommendation="Check prediction distributions" if severity > 0.3 else "No action needed"
        )
    
    async def _send_drift_alert(self, drift: DriftDetectionResult) -> None:
        """Send alert for detected drift"""
        logger.warning(f"DRIFT ALERT: {drift.drift_type.value} detected for "
                      f"{drift.model_id} v{drift.version} - Severity: {drift.severity:.3f}")
        drift.alert_sent = True
    
    async def _find_previous_version(self, model_id: str, current_version: str) -> Optional[str]:
        """Find previous version for rollback"""
        if model_id not in self.models:
            return None
        
        versions = sorted(self.models[model_id], key=lambda v: v.created_at, reverse=True)
        
        for i, version in enumerate(versions):
            if version.version == current_version and i + 1 < len(versions):
                return versions[i + 1].version
        
        return None
    
    async def _start_deployment_monitoring(self, deployment_id: str, config: ModelConfig) -> None:
        """Start monitoring for a specific deployment"""
        # Placeholder for deployment-specific monitoring setup
        logger.info(f"Started monitoring for deployment: {deployment_id}")
    
    async def _collect_performance_metrics(self, deployment_id: str) -> Optional[ModelPerformanceMetrics]:
        """Collect performance metrics for a deployment"""
        config = self.active_deployments.get(deployment_id)
        if not config:
            return None
        
        # Simulate metrics collection
        return ModelPerformanceMetrics(
            model_id=config.model_id,
            version=config.version,
            environment=config.environment,
            timestamp=datetime.utcnow(),
            predictions_count=np.random.randint(100, 1000),
            accuracy=np.random.uniform(0.75, 0.98),
            precision=np.random.uniform(0.70, 0.95),
            recall=np.random.uniform(0.70, 0.95),
            f1_score=np.random.uniform(0.70, 0.95),
            latency_p50=np.random.uniform(10, 50),
            latency_p95=np.random.uniform(50, 200),
            latency_p99=np.random.uniform(100, 500),
            throughput=np.random.uniform(100, 1000),
            error_rate=np.random.uniform(0.001, 0.05),
            cpu_utilization=np.random.uniform(20, 80),
            memory_utilization=np.random.uniform(30, 85)
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(x) == len(y) and len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return 0.0
    
    def _get_environment_breakdown(self, metrics: List[ModelPerformanceMetrics]) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by environment"""
        env_breakdown = {}
        
        for metric in metrics:
            env = metric.environment.value
            if env not in env_breakdown:
                env_breakdown[env] = {
                    "avg_accuracy": 0.0,
                    "avg_latency": 0.0,
                    "total_predictions": 0
                }
            
            env_breakdown[env]["avg_accuracy"] += metric.accuracy
            env_breakdown[env]["avg_latency"] += metric.latency_p95
            env_breakdown[env]["total_predictions"] += metric.predictions_count
        
        # Calculate averages
        for env_stats in env_breakdown.values():
            if env_stats["total_predictions"] > 0:
                env_stats["avg_accuracy"] /= env_stats["total_predictions"]
                env_stats["avg_latency"] /= env_stats["total_predictions"]
        
        return env_breakdown
    
    async def _load_models(self) -> None:
        """Load existing models from storage"""
        # Placeholder for loading models from registry
        logger.info("Loading existing models from registry")
    
    async def _save_manager_state(self) -> None:
        """Save manager state to storage"""
        state_file = self.base_path / "manager_state.json"
        
        state = {
            "models_count": len(self.models),
            "active_deployments_count": len(self.active_deployments),
            "pending_approvals_count": len(self.pending_approvals),
            "ab_tests_count": len(self.ab_tests),
            "last_saved": datetime.utcnow().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)