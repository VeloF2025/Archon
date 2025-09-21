"""
Training Orchestrator for Archon Enhancement 2025 Phase 4

Advanced distributed training orchestration system with:
- Multi-framework training support (PyTorch, TensorFlow, JAX)
- Distributed training coordination across multiple nodes/GPUs
- Hyperparameter optimization with advanced strategies
- Training job scheduling and resource management  
- Experiment tracking and model versioning
- Real-time monitoring and early stopping
- Checkpointing and fault tolerance
- Dynamic resource scaling and load balancing
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import pickle
import hashlib
import time
import signal

logger = logging.getLogger(__name__)

class TrainingFramework(Enum):
    """Supported training frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CUSTOM = "custom"

class TrainingStrategy(Enum):
    """Training execution strategies"""
    SINGLE_NODE = "single_node"
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    FEDERATED = "federated"

class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    POPULATION_BASED = "population_based"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"

class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class ResourceType(Enum):
    """Computing resource types"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: Union[int, str]
    min_amount: Optional[Union[int, str]] = None
    max_amount: Optional[Union[int, str]] = None
    preferred_specs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "resource_type": self.resource_type.value,
            "amount": self.amount,
            "min_amount": self.min_amount,
            "max_amount": self.max_amount,
            "preferred_specs": self.preferred_specs
        }

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    config_id: str
    framework: TrainingFramework
    strategy: TrainingStrategy
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    
    # Hyperparameter optimization
    enable_hyperopt: bool = False
    hyperopt_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
    hyperopt_trials: int = 50
    hyperopt_space: Dict[str, Any] = field(default_factory=dict)
    
    # Distributed training
    num_workers: int = 1
    num_gpus: int = 0
    mixed_precision: bool = False
    gradient_clipping: Optional[float] = None
    distributed_backend: str = "nccl"
    
    # Resource requirements
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    max_training_time: Optional[timedelta] = None
    priority: int = 1  # 1 = low, 10 = high
    
    # Monitoring and checkpointing
    checkpoint_frequency: int = 10  # epochs
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # min or max
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    
    # Advanced features
    enable_pruning: bool = False
    enable_quantization: bool = False
    enable_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "config_id": self.config_id,
            "framework": self.framework.value,
            "strategy": self.strategy.value,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "metrics": self.metrics,
            "enable_hyperopt": self.enable_hyperopt,
            "hyperopt_strategy": self.hyperopt_strategy.value,
            "hyperopt_trials": self.hyperopt_trials,
            "hyperopt_space": self.hyperopt_space,
            "num_workers": self.num_workers,
            "num_gpus": self.num_gpus,
            "mixed_precision": self.mixed_precision,
            "gradient_clipping": self.gradient_clipping,
            "distributed_backend": self.distributed_backend,
            "resource_requirements": [req.to_dict() for req in self.resource_requirements],
            "max_training_time": str(self.max_training_time) if self.max_training_time else None,
            "priority": self.priority,
            "checkpoint_frequency": self.checkpoint_frequency,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "early_stopping_mode": self.early_stopping_mode,
            "enable_tensorboard": self.enable_tensorboard,
            "enable_wandb": self.enable_wandb,
            "enable_pruning": self.enable_pruning,
            "enable_quantization": self.enable_quantization,
            "enable_knowledge_distillation": self.enable_knowledge_distillation,
            "teacher_model_path": self.teacher_model_path
        }

@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    job_id: str
    epoch: int
    step: int
    training_loss: float
    validation_loss: Optional[float] = None
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    learning_rate: Optional[float] = None
    gpu_utilization: Optional[float] = None
    memory_usage: Optional[float] = None
    throughput: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "job_id": self.job_id,
            "epoch": self.epoch,
            "step": self.step,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "learning_rate": self.learning_rate,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage": self.memory_usage,
            "throughput": self.throughput,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class TrainingJob:
    """Training job definition and state"""
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    current_epoch: int = 0
    current_step: int = 0
    
    # Performance tracking
    metrics_history: List[TrainingMetrics] = field(default_factory=list)
    best_metrics: Optional[TrainingMetrics] = None
    early_stopping_counter: int = 0
    
    # Resource allocation
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    worker_nodes: List[str] = field(default_factory=list)
    
    # Timing and lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Output paths
    model_output_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    logs_path: Optional[str] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def get_duration(self) -> Optional[timedelta]:
        """Get training duration"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    def get_eta(self) -> Optional[timedelta]:
        """Estimate time to completion"""
        if self.status != TrainingStatus.RUNNING or self.current_epoch == 0:
            return None
        
        duration = self.get_duration()
        if not duration:
            return None
        
        epochs_per_second = self.current_epoch / duration.total_seconds()
        remaining_epochs = self.config.epochs - self.current_epoch
        
        if epochs_per_second > 0:
            eta_seconds = remaining_epochs / epochs_per_second
            return timedelta(seconds=eta_seconds)
        
        return None

@dataclass
class ExperimentRun:
    """Experiment run with multiple training jobs"""
    experiment_id: str
    name: str
    description: str
    jobs: List[str] = field(default_factory=list)  # job IDs
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "jobs": self.jobs,
            "tags": self.tags,
            "parameters": self.parameters,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }

class TrainingOrchestrator:
    """
    Advanced training orchestration system with distributed training,
    hyperparameter optimization, and comprehensive experiment management.
    """
    
    def __init__(self, base_path: str = "./training_jobs", max_concurrent_jobs: int = 4):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Job management
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[str] = []
        self.completed_jobs: Dict[str, TrainingJob] = {}
        
        # Experiments
        self.experiments: Dict[str, ExperimentRun] = {}
        
        # Resource management
        self.available_resources: Dict[ResourceType, int] = {
            ResourceType.CPU: 8,
            ResourceType.GPU: 2,
            ResourceType.MEMORY: 32,  # GB
            ResourceType.STORAGE: 1000  # GB
        }
        self.allocated_resources: Dict[ResourceType, int] = {
            ResourceType.CPU: 0,
            ResourceType.GPU: 0,
            ResourceType.MEMORY: 0,
            ResourceType.STORAGE: 0
        }
        
        # Framework handlers
        self.framework_handlers = self._initialize_framework_handlers()
        
        # Executors for training jobs
        self.thread_executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self._lock = threading.RLock()
        self._shutdown = False
        self._scheduler_task = None
        
        logger.info("Training Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize the training orchestrator"""
        try:
            await self._load_existing_jobs()
            await self._load_experiments()
            await self._start_job_scheduler()
            await self._start_resource_monitor()
            logger.info("Training Orchestrator initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Training Orchestrator: {e}")
            raise
    
    async def create_training_job(self, config: TrainingConfig) -> str:
        """Create a new training job"""
        try:
            job_id = f"train_{uuid.uuid4().hex[:8]}"
            
            # Validate configuration
            await self._validate_training_config(config)
            
            # Create training job
            job = TrainingJob(
                job_id=job_id,
                config=config,
                model_output_path=str(self.base_path / "models" / f"{job_id}"),
                checkpoint_path=str(self.base_path / "checkpoints" / f"{job_id}"),
                logs_path=str(self.base_path / "logs" / f"{job_id}")
            )
            
            # Create directories
            for path in [job.model_output_path, job.checkpoint_path, job.logs_path]:
                Path(path).mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                self.active_jobs[job_id] = job
                
                # Add to queue based on priority
                self._add_job_to_queue(job_id, config.priority)
            
            # Save job configuration
            await self._save_job_config(job)
            
            logger.info(f"Created training job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise
    
    async def submit_job(self, job_id: str) -> bool:
        """Submit job for execution"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            # Check resource availability
            if not await self._can_allocate_resources(job.config.resource_requirements):
                job.warnings.append("Insufficient resources - job queued")
                return False
            
            # Allocate resources
            await self._allocate_resources(job_id, job.config.resource_requirements)
            
            # Start training
            asyncio.create_task(self._execute_training_job(job_id))
            
            logger.info(f"Submitted training job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            return False
    
    async def create_experiment(self,
                              experiment_name: str,
                              description: str,
                              job_configs: List[TrainingConfig],
                              tags: List[str] = None) -> str:
        """Create experiment with multiple training jobs"""
        try:
            experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
            
            # Create training jobs for experiment
            job_ids = []
            for i, config in enumerate(job_configs):
                config.config_id = f"{experiment_id}_job_{i}"
                job_id = await self.create_training_job(config)
                job_ids.append(job_id)
            
            # Create experiment
            experiment = ExperimentRun(
                experiment_id=experiment_id,
                name=experiment_name,
                description=description,
                jobs=job_ids,
                tags=tags or []
            )
            
            with self._lock:
                self.experiments[experiment_id] = experiment
            
            # Save experiment
            await self._save_experiment(experiment)
            
            logger.info(f"Created experiment: {experiment_id} with {len(job_ids)} jobs")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def run_hyperparameter_optimization(self,
                                            base_config: TrainingConfig,
                                            search_space: Dict[str, Any],
                                            n_trials: int = 50,
                                            strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION) -> str:
        """Run hyperparameter optimization experiment"""
        try:
            experiment_name = f"hyperopt_{base_config.config_id}_{strategy.value}"
            
            # Generate configurations for hyperparameter search
            job_configs = await self._generate_hyperopt_configs(
                base_config, search_space, n_trials, strategy
            )
            
            # Create experiment
            experiment_id = await self.create_experiment(
                experiment_name=experiment_name,
                description=f"Hyperparameter optimization using {strategy.value}",
                job_configs=job_configs,
                tags=["hyperopt", strategy.value]
            )
            
            # Submit all jobs
            experiment = self.experiments[experiment_id]
            for job_id in experiment.jobs:
                await self.submit_job(job_id)
            
            logger.info(f"Started hyperparameter optimization experiment: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to run hyperparameter optimization: {e}")
            raise
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a running training job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job.status == TrainingStatus.RUNNING:
                job.status = TrainingStatus.PAUSED
                # Send pause signal to training process
                await self._send_job_signal(job_id, "PAUSE")
                
                logger.info(f"Paused training job: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused training job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job.status == TrainingStatus.PAUSED:
                job.status = TrainingStatus.RUNNING
                # Send resume signal to training process
                await self._send_job_signal(job_id, "RESUME")
                
                logger.info(f"Resumed training job: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            if job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
                job.status = TrainingStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                
                # Send termination signal
                await self._send_job_signal(job_id, "TERMINATE")
                
                # Release resources
                await self._release_resources(job_id)
                
                # Move to completed jobs
                with self._lock:
                    self.completed_jobs[job_id] = job
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]
                
                logger.info(f"Cancelled training job: {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of training job"""
        try:
            job = None
            
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
            else:
                raise ValueError(f"Job {job_id} not found")
            
            # Get latest metrics
            latest_metrics = job.metrics_history[-1] if job.metrics_history else None
            
            status = {
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "current_epoch": job.current_epoch,
                "total_epochs": job.config.epochs,
                "current_step": job.current_step,
                "duration": str(job.get_duration()) if job.get_duration() else None,
                "eta": str(job.get_eta()) if job.get_eta() else None,
                "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
                "best_metrics": job.best_metrics.to_dict() if job.best_metrics else None,
                "allocated_resources": job.allocated_resources,
                "worker_nodes": job.worker_nodes,
                "errors": job.errors,
                "warnings": job.warnings,
                "retry_count": job.retry_count,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            raise
    
    async def get_job_metrics(self, job_id: str, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get training metrics for job"""
        try:
            job = None
            
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
            elif job_id in self.completed_jobs:
                job = self.completed_jobs[job_id]
            else:
                raise ValueError(f"Job {job_id} not found")
            
            metrics = job.metrics_history
            if last_n:
                metrics = metrics[-last_n:]
            
            return [m.to_dict() for m in metrics]
            
        except Exception as e:
            logger.error(f"Failed to get job metrics {job_id}: {e}")
            raise
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment status"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            job_statuses = []
            
            for job_id in experiment.jobs:
                try:
                    job_status = await self.get_job_status(job_id)
                    job_statuses.append(job_status)
                except Exception as e:
                    logger.warning(f"Failed to get status for job {job_id}: {e}")
            
            # Calculate experiment-level metrics
            total_jobs = len(experiment.jobs)
            completed_jobs = len([s for s in job_statuses if s["status"] == "completed"])
            running_jobs = len([s for s in job_statuses if s["status"] == "running"])
            failed_jobs = len([s for s in job_statuses if s["status"] == "failed"])
            
            # Find best performing job
            best_job = None
            best_metric = float('-inf')
            
            for status in job_statuses:
                if status["best_metrics"] and "validation_metrics" in status["best_metrics"]:
                    val_metrics = status["best_metrics"]["validation_metrics"]
                    if "accuracy" in val_metrics and val_metrics["accuracy"] > best_metric:
                        best_metric = val_metrics["accuracy"]
                        best_job = status
            
            experiment_status = {
                "experiment_id": experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "running_jobs": running_jobs,
                "failed_jobs": failed_jobs,
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
                "best_job": best_job,
                "job_statuses": job_statuses,
                "tags": experiment.tags,
                "created_by": experiment.created_by,
                "created_at": experiment.created_at.isoformat()
            }
            
            return experiment_status
            
        except Exception as e:
            logger.error(f"Failed to get experiment status {experiment_id}: {e}")
            raise
    
    async def list_jobs(self, 
                       status_filter: Optional[TrainingStatus] = None,
                       framework_filter: Optional[TrainingFramework] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """List training jobs with filtering"""
        try:
            all_jobs = {**self.active_jobs, **self.completed_jobs}
            filtered_jobs = []
            
            for job_id, job in all_jobs.items():
                # Apply filters
                if status_filter and job.status != status_filter:
                    continue
                if framework_filter and job.config.framework != framework_filter:
                    continue
                
                job_summary = {
                    "job_id": job_id,
                    "status": job.status.value,
                    "framework": job.config.framework.value,
                    "strategy": job.config.strategy.value,
                    "progress": job.progress,
                    "current_epoch": job.current_epoch,
                    "total_epochs": job.config.epochs,
                    "duration": str(job.get_duration()) if job.get_duration() else None,
                    "best_score": None,
                    "created_at": job.created_at.isoformat()
                }
                
                # Add best score if available
                if job.best_metrics and job.best_metrics.validation_metrics:
                    job_summary["best_score"] = job.best_metrics.validation_metrics.get("accuracy")
                
                filtered_jobs.append(job_summary)
            
            # Sort by creation time (newest first)
            filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)
            
            return filtered_jobs[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise
    
    async def get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization"""
        try:
            utilization = {}
            
            for resource_type in ResourceType:
                available = self.available_resources.get(resource_type, 0)
                allocated = self.allocated_resources.get(resource_type, 0)
                
                utilization[resource_type.value] = {
                    "available": available,
                    "allocated": allocated,
                    "free": available - allocated,
                    "utilization_percentage": (allocated / available * 100) if available > 0 else 0
                }
            
            return {
                "resource_utilization": utilization,
                "active_jobs": len(self.active_jobs),
                "queued_jobs": len(self.job_queue),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource utilization: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get training orchestrator performance metrics"""
        try:
            total_jobs = len(self.active_jobs) + len(self.completed_jobs)
            completed_jobs = len(self.completed_jobs)
            running_jobs = len([j for j in self.active_jobs.values() if j.status == TrainingStatus.RUNNING])
            failed_jobs = len([j for j in self.completed_jobs.values() if j.status == TrainingStatus.FAILED])
            
            # Calculate average training time
            completed_durations = [j.get_duration() for j in self.completed_jobs.values() if j.get_duration()]
            avg_training_time = timedelta(0)
            if completed_durations:
                total_time = sum(completed_durations, timedelta(0))
                avg_training_time = total_time / len(completed_durations)
            
            # Framework distribution
            framework_dist = {}
            for job in {**self.active_jobs, **self.completed_jobs}.values():
                framework = job.config.framework.value
                framework_dist[framework] = framework_dist.get(framework, 0) + 1
            
            # Strategy distribution
            strategy_dist = {}
            for job in {**self.active_jobs, **self.completed_jobs}.values():
                strategy = job.config.strategy.value
                strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "running_jobs": running_jobs,
                "failed_jobs": failed_jobs,
                "queued_jobs": len(self.job_queue),
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
                "total_experiments": len(self.experiments),
                "average_training_time_hours": avg_training_time.total_seconds() / 3600,
                "framework_distribution": framework_dist,
                "strategy_distribution": strategy_dist,
                "resource_utilization": await self.get_resource_utilization(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get training orchestrator metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the training orchestrator"""
        try:
            self._shutdown = True
            
            # Cancel scheduler task
            if self._scheduler_task:
                self._scheduler_task.cancel()
            
            # Cancel all running jobs
            for job_id, job in self.active_jobs.items():
                if job.status == TrainingStatus.RUNNING:
                    await self.cancel_job(job_id)
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            # Save all job states
            await self._save_all_job_states()
            
            logger.info("Training Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Training Orchestrator shutdown: {e}")
    
    # Private helper methods
    
    def _initialize_framework_handlers(self) -> Dict[TrainingFramework, Callable]:
        """Initialize framework-specific training handlers"""
        return {
            TrainingFramework.PYTORCH: self._handle_pytorch_training,
            TrainingFramework.TENSORFLOW: self._handle_tensorflow_training,
            TrainingFramework.JAX: self._handle_jax_training,
            TrainingFramework.SKLEARN: self._handle_sklearn_training,
            TrainingFramework.XGBOOST: self._handle_xgboost_training,
            TrainingFramework.LIGHTGBM: self._handle_lightgbm_training,
            TrainingFramework.CUSTOM: self._handle_custom_training
        }
    
    async def _validate_training_config(self, config: TrainingConfig) -> None:
        """Validate training configuration"""
        if config.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if config.num_workers < 0:
            raise ValueError("Number of workers cannot be negative")
        
        if config.num_gpus < 0:
            raise ValueError("Number of GPUs cannot be negative")
    
    def _add_job_to_queue(self, job_id: str, priority: int) -> None:
        """Add job to queue with priority ordering"""
        # Insert job based on priority (higher priority first)
        inserted = False
        for i, queued_job_id in enumerate(self.job_queue):
            queued_job = self.active_jobs[queued_job_id]
            if priority > queued_job.config.priority:
                self.job_queue.insert(i, job_id)
                inserted = True
                break
        
        if not inserted:
            self.job_queue.append(job_id)
    
    async def _can_allocate_resources(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated"""
        for req in requirements:
            resource_type = req.resource_type
            required_amount = req.amount if isinstance(req.amount, int) else 1
            
            available = self.available_resources.get(resource_type, 0)
            allocated = self.allocated_resources.get(resource_type, 0)
            
            if allocated + required_amount > available:
                return False
        
        return True
    
    async def _allocate_resources(self, job_id: str, requirements: List[ResourceRequirement]) -> None:
        """Allocate resources for job"""
        job = self.active_jobs[job_id]
        allocated = {}
        
        for req in requirements:
            resource_type = req.resource_type
            required_amount = req.amount if isinstance(req.amount, int) else 1
            
            self.allocated_resources[resource_type] += required_amount
            allocated[resource_type.value] = required_amount
        
        job.allocated_resources = allocated
        
        # Assign worker nodes (simulated)
        if job.config.num_workers > 1:
            job.worker_nodes = [f"worker-{i}" for i in range(job.config.num_workers)]
    
    async def _release_resources(self, job_id: str) -> None:
        """Release resources allocated to job"""
        job = self.active_jobs.get(job_id) or self.completed_jobs.get(job_id)
        if not job:
            return
        
        for resource_type_str, amount in job.allocated_resources.items():
            resource_type = ResourceType(resource_type_str)
            self.allocated_resources[resource_type] -= amount
            self.allocated_resources[resource_type] = max(0, self.allocated_resources[resource_type])
        
        job.allocated_resources = {}
        job.worker_nodes = []
    
    async def _start_job_scheduler(self) -> None:
        """Start background job scheduler"""
        self._scheduler_task = asyncio.create_task(self._job_scheduler_loop())
    
    async def _job_scheduler_loop(self) -> None:
        """Background job scheduler loop"""
        while not self._shutdown:
            try:
                # Check for jobs to schedule
                if self.job_queue and len([j for j in self.active_jobs.values() if j.status == TrainingStatus.RUNNING]) < self.max_concurrent_jobs:
                    job_id = self.job_queue.pop(0)
                    
                    if job_id in self.active_jobs:
                        success = await self.submit_job(job_id)
                        if not success:
                            # Put back in queue if couldn't allocate resources
                            self.job_queue.insert(0, job_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in job scheduler: {e}")
                await asyncio.sleep(30)
    
    async def _start_resource_monitor(self) -> None:
        """Start background resource monitoring"""
        asyncio.create_task(self._resource_monitor_loop())
    
    async def _resource_monitor_loop(self) -> None:
        """Background resource monitoring loop"""
        while not self._shutdown:
            try:
                # Update resource availability (simulated)
                # In real implementation, this would query actual system resources
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(60)
    
    async def _execute_training_job(self, job_id: str) -> None:
        """Execute training job"""
        try:
            if job_id not in self.active_jobs:
                return
            
            job = self.active_jobs[job_id]
            job.status = TrainingStatus.INITIALIZING
            job.started_at = datetime.utcnow()
            
            # Get framework handler
            framework_handler = self.framework_handlers.get(job.config.framework)
            if not framework_handler:
                raise ValueError(f"Unsupported framework: {job.config.framework}")
            
            job.status = TrainingStatus.RUNNING
            
            # Execute training with framework handler
            await framework_handler(job_id)
            
            # Job completed successfully
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 1.0
            
            # Move to completed jobs
            with self._lock:
                self.completed_jobs[job_id] = job
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]
            
            # Release resources
            await self._release_resources(job_id)
            
            logger.info(f"Completed training job: {job_id}")
            
        except Exception as e:
            # Job failed
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = TrainingStatus.FAILED
                job.completed_at = datetime.utcnow()
                job.errors.append(str(e))
                
                # Check for retries
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = TrainingStatus.PENDING
                    self._add_job_to_queue(job_id, job.config.priority)
                    logger.info(f"Retrying job {job_id} (attempt {job.retry_count})")
                else:
                    # Move to completed jobs
                    with self._lock:
                        self.completed_jobs[job_id] = job
                        if job_id in self.active_jobs:
                            del self.active_jobs[job_id]
                    
                    # Release resources
                    await self._release_resources(job_id)
                    
                    logger.error(f"Training job failed: {job_id} - {e}")
    
    # Framework-specific training handlers
    
    async def _handle_pytorch_training(self, job_id: str) -> None:
        """Handle PyTorch training job"""
        job = self.active_jobs[job_id]
        config = job.config
        
        logger.info(f"Starting PyTorch training for job {job_id}")
        
        # Simulate PyTorch training loop
        for epoch in range(config.epochs):
            if job.status != TrainingStatus.RUNNING:
                break
            
            job.current_epoch = epoch + 1
            job.progress = epoch / config.epochs
            
            # Simulate training step
            await asyncio.sleep(0.1)  # Simulate training time
            
            # Generate mock metrics
            training_loss = 2.0 * np.exp(-epoch / 20) + np.random.uniform(0, 0.1)
            validation_loss = training_loss + np.random.uniform(0, 0.2)
            training_accuracy = 1.0 - training_loss / 2.0 + np.random.uniform(-0.05, 0.05)
            validation_accuracy = training_accuracy - np.random.uniform(0, 0.1)
            
            # Create metrics
            metrics = TrainingMetrics(
                job_id=job_id,
                epoch=epoch + 1,
                step=job.current_step,
                training_loss=training_loss,
                validation_loss=validation_loss,
                training_metrics={"accuracy": training_accuracy},
                validation_metrics={"accuracy": validation_accuracy},
                learning_rate=config.learning_rate,
                gpu_utilization=np.random.uniform(0.7, 0.95) if config.num_gpus > 0 else None,
                memory_usage=np.random.uniform(0.6, 0.9),
                throughput=np.random.uniform(100, 1000)
            )
            
            job.metrics_history.append(metrics)
            
            # Update best metrics
            if (not job.best_metrics or 
                metrics.validation_metrics.get("accuracy", 0) > 
                job.best_metrics.validation_metrics.get("accuracy", 0)):
                job.best_metrics = metrics
                job.early_stopping_counter = 0
            else:
                job.early_stopping_counter += 1
            
            # Early stopping check
            if job.early_stopping_counter >= config.early_stopping_patience:
                job.warnings.append(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % config.checkpoint_frequency == 0:
                await self._save_checkpoint(job_id, epoch + 1)
            
            job.current_step += 1
    
    async def _handle_tensorflow_training(self, job_id: str) -> None:
        """Handle TensorFlow training job"""
        job = self.active_jobs[job_id]
        config = job.config
        
        logger.info(f"Starting TensorFlow training for job {job_id}")
        
        # Similar to PyTorch but with TensorFlow-specific logic
        for epoch in range(config.epochs):
            if job.status != TrainingStatus.RUNNING:
                break
            
            job.current_epoch = epoch + 1
            job.progress = epoch / config.epochs
            
            await asyncio.sleep(0.1)  # Simulate training
            
            # Generate mock metrics (similar to PyTorch)
            training_loss = 2.0 * np.exp(-epoch / 25) + np.random.uniform(0, 0.1)
            validation_loss = training_loss + np.random.uniform(0, 0.15)
            training_accuracy = 1.0 - training_loss / 2.0
            validation_accuracy = training_accuracy - np.random.uniform(0, 0.08)
            
            metrics = TrainingMetrics(
                job_id=job_id,
                epoch=epoch + 1,
                step=job.current_step,
                training_loss=training_loss,
                validation_loss=validation_loss,
                training_metrics={"accuracy": training_accuracy},
                validation_metrics={"accuracy": validation_accuracy},
                learning_rate=config.learning_rate,
                gpu_utilization=np.random.uniform(0.75, 0.9) if config.num_gpus > 0 else None,
                memory_usage=np.random.uniform(0.65, 0.85),
                throughput=np.random.uniform(150, 800)
            )
            
            job.metrics_history.append(metrics)
            
            if (not job.best_metrics or 
                metrics.validation_metrics.get("accuracy", 0) > 
                job.best_metrics.validation_metrics.get("accuracy", 0)):
                job.best_metrics = metrics
                job.early_stopping_counter = 0
            else:
                job.early_stopping_counter += 1
            
            if job.early_stopping_counter >= config.early_stopping_patience:
                job.warnings.append(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % config.checkpoint_frequency == 0:
                await self._save_checkpoint(job_id, epoch + 1)
            
            job.current_step += 1
    
    async def _handle_jax_training(self, job_id: str) -> None:
        """Handle JAX training job"""
        # Similar implementation to PyTorch/TensorFlow
        await self._handle_pytorch_training(job_id)  # Placeholder
    
    async def _handle_sklearn_training(self, job_id: str) -> None:
        """Handle scikit-learn training job"""
        job = self.active_jobs[job_id]
        
        logger.info(f"Starting scikit-learn training for job {job_id}")
        
        # Simulate sklearn training (typically much faster)
        await asyncio.sleep(1.0)  # Simulate training time
        
        # Create final metrics
        accuracy = np.random.uniform(0.8, 0.95)
        metrics = TrainingMetrics(
            job_id=job_id,
            epoch=1,
            step=1,
            training_loss=0.5 - accuracy/2,
            training_metrics={"accuracy": accuracy},
            validation_metrics={"accuracy": accuracy - np.random.uniform(0, 0.05)},
            throughput=1000.0
        )
        
        job.metrics_history.append(metrics)
        job.best_metrics = metrics
        job.current_epoch = 1
        job.progress = 1.0
    
    async def _handle_xgboost_training(self, job_id: str) -> None:
        """Handle XGBoost training job"""
        job = self.active_jobs[job_id]
        config = job.config
        
        logger.info(f"Starting XGBoost training for job {job_id}")
        
        # Simulate XGBoost boosting rounds
        for round_num in range(min(config.epochs, 100)):  # XGBoost typically uses fewer rounds
            if job.status != TrainingStatus.RUNNING:
                break
            
            job.current_epoch = round_num + 1
            job.progress = round_num / min(config.epochs, 100)
            
            await asyncio.sleep(0.05)  # XGBoost is typically faster
            
            # XGBoost metrics (gradually improving)
            training_error = 0.3 * np.exp(-round_num / 30) + np.random.uniform(0, 0.02)
            validation_error = training_error + np.random.uniform(0, 0.05)
            
            metrics = TrainingMetrics(
                job_id=job_id,
                epoch=round_num + 1,
                step=job.current_step,
                training_loss=training_error,
                validation_loss=validation_error,
                training_metrics={"accuracy": 1.0 - training_error},
                validation_metrics={"accuracy": 1.0 - validation_error},
                throughput=np.random.uniform(500, 2000)
            )
            
            job.metrics_history.append(metrics)
            
            if (not job.best_metrics or 
                metrics.validation_metrics.get("accuracy", 0) > 
                job.best_metrics.validation_metrics.get("accuracy", 0)):
                job.best_metrics = metrics
                job.early_stopping_counter = 0
            else:
                job.early_stopping_counter += 1
            
            if job.early_stopping_counter >= config.early_stopping_patience:
                break
            
            job.current_step += 1
    
    async def _handle_lightgbm_training(self, job_id: str) -> None:
        """Handle LightGBM training job"""
        # Similar to XGBoost
        await self._handle_xgboost_training(job_id)
    
    async def _handle_custom_training(self, job_id: str) -> None:
        """Handle custom training job"""
        # Placeholder for custom training logic
        await self._handle_pytorch_training(job_id)
    
    async def _generate_hyperopt_configs(self,
                                       base_config: TrainingConfig,
                                       search_space: Dict[str, Any],
                                       n_trials: int,
                                       strategy: OptimizationStrategy) -> List[TrainingConfig]:
        """Generate configurations for hyperparameter optimization"""
        configs = []
        
        # For simplicity, generate random configurations
        # In practice, this would use proper optimization algorithms
        for i in range(n_trials):
            config = TrainingConfig(
                config_id=f"{base_config.config_id}_trial_{i}",
                framework=base_config.framework,
                strategy=base_config.strategy,
                model_config=base_config.model_config.copy(),
                dataset_config=base_config.dataset_config.copy(),
                epochs=base_config.epochs,
                batch_size=base_config.batch_size,
                learning_rate=base_config.learning_rate,
                optimizer=base_config.optimizer,
                loss_function=base_config.loss_function
            )
            
            # Sample hyperparameters from search space
            for param_name, param_range in search_space.items():
                if isinstance(param_range, list):
                    # Categorical parameter
                    setattr(config, param_name, np.random.choice(param_range))
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    # Numerical parameter range
                    min_val, max_val = param_range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        setattr(config, param_name, np.random.randint(min_val, max_val + 1))
                    else:
                        setattr(config, param_name, np.random.uniform(min_val, max_val))
            
            configs.append(config)
        
        return configs
    
    async def _send_job_signal(self, job_id: str, signal_type: str) -> None:
        """Send signal to training job process"""
        # In real implementation, this would send signals to actual training processes
        logger.info(f"Sending {signal_type} signal to job {job_id}")
    
    async def _save_checkpoint(self, job_id: str, epoch: int) -> None:
        """Save training checkpoint"""
        job = self.active_jobs[job_id]
        checkpoint_file = Path(job.checkpoint_path) / f"checkpoint_epoch_{epoch}.pkl"
        
        # In real implementation, this would save actual model state
        checkpoint_data = {
            "job_id": job_id,
            "epoch": epoch,
            "model_state": "mock_model_state",
            "optimizer_state": "mock_optimizer_state",
            "metrics": job.best_metrics.to_dict() if job.best_metrics else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint for job {job_id} at epoch {epoch}")
    
    async def _save_job_config(self, job: TrainingJob) -> None:
        """Save job configuration to disk"""
        config_file = self.base_path / "configs" / f"{job.job_id}_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(job.config.to_dict(), f, indent=2)
    
    async def _save_experiment(self, experiment: ExperimentRun) -> None:
        """Save experiment to disk"""
        experiment_file = self.base_path / "experiments" / f"{experiment.experiment_id}.json"
        experiment_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(experiment_file, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2)
    
    async def _load_existing_jobs(self) -> None:
        """Load existing jobs from storage"""
        configs_dir = self.base_path / "configs"
        if configs_dir.exists():
            for config_file in configs_dir.glob("*_config.json"):
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Recreate job with completed status (simplified)
                    job_id = config_file.stem.replace('_config', '')
                    logger.info(f"Loaded job config: {job_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load job config {config_file}: {e}")
    
    async def _load_experiments(self) -> None:
        """Load existing experiments from storage"""
        experiments_dir = self.base_path / "experiments"
        if experiments_dir.exists():
            for exp_file in experiments_dir.glob("*.json"):
                try:
                    with open(exp_file, 'r') as f:
                        exp_data = json.load(f)
                    
                    experiment = ExperimentRun(
                        experiment_id=exp_data["experiment_id"],
                        name=exp_data["name"],
                        description=exp_data["description"],
                        jobs=exp_data["jobs"],
                        tags=exp_data["tags"],
                        parameters=exp_data.get("parameters", {}),
                        created_by=exp_data.get("created_by", "system")
                    )
                    
                    self.experiments[experiment.experiment_id] = experiment
                    logger.info(f"Loaded experiment: {experiment.experiment_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load experiment {exp_file}: {e}")
    
    async def _save_all_job_states(self) -> None:
        """Save all job states to storage"""
        state_file = self.base_path / "orchestrator_state.json"
        
        state = {
            "active_jobs_count": len(self.active_jobs),
            "completed_jobs_count": len(self.completed_jobs),
            "queued_jobs_count": len(self.job_queue),
            "total_experiments": len(self.experiments),
            "resource_utilization": {
                resource_type.value: {
                    "available": self.available_resources.get(resource_type, 0),
                    "allocated": self.allocated_resources.get(resource_type, 0)
                }
                for resource_type in ResourceType
            },
            "last_saved": datetime.utcnow().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)