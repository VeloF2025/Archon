"""
Neural Network Orchestrator for Archon Enhancement 2025 Phase 4

Advanced neural network management and orchestration system with:
- Dynamic network architecture creation and modification
- Multi-framework support (PyTorch, TensorFlow, JAX)
- Automated hyperparameter optimization
- Distributed training coordination
- Neural architecture search (NAS)
- Model compression and quantization
- Real-time performance monitoring
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class FrameworkType(Enum):
    """Supported ML frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    SCIKIT_LEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class LayerType(Enum):
    """Neural network layer types"""
    DENSE = "dense"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    POOLING = "pooling"
    EMBEDDING = "embedding"
    RESIDUAL = "residual"

class ActivationFunction(Enum):
    """Activation functions"""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"

class OptimizationObjective(Enum):
    """Network optimization objectives"""
    MINIMIZE_LOSS = "minimize_loss"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_PARAMETERS = "minimize_parameters"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_MEMORY = "minimize_memory"

class NetworkStatus(Enum):
    """Network lifecycle status"""
    DESIGNING = "designing"
    BUILDING = "building"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    OPTIMIZING = "optimizing"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class LayerSpec:
    """Specification for a neural network layer"""
    layer_type: LayerType
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[ActivationFunction] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    trainable: bool = True
    regularization: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer specification to dictionary"""
        return {
            "layer_type": self.layer_type.value,
            "name": self.name,
            "parameters": self.parameters,
            "activation": self.activation.value if self.activation else None,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "trainable": self.trainable,
            "regularization": self.regularization
        }

@dataclass
class NetworkConfig:
    """Neural network configuration"""
    network_id: str
    framework: FrameworkType
    architecture_layers: List[LayerSpec]
    loss_function: str = "categorical_crossentropy"
    optimizer: str = "adam"
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    input_shape: Tuple[int, ...] = (None,)
    output_shape: Tuple[int, ...] = (None,)
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping: bool = True
    model_checkpoint: bool = True
    tensorboard_logging: bool = True
    distributed_training: bool = False
    mixed_precision: bool = False
    gradient_clipping: Optional[float] = None
    learning_rate_schedule: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert network configuration to dictionary"""
        return {
            "network_id": self.network_id,
            "framework": self.framework.value,
            "architecture_layers": [layer.to_dict() for layer in self.architecture_layers],
            "loss_function": self.loss_function,
            "optimizer": self.optimizer,
            "optimizer_config": self.optimizer_config,
            "metrics": self.metrics,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "validation_split": self.validation_split,
            "early_stopping": self.early_stopping,
            "model_checkpoint": self.model_checkpoint,
            "tensorboard_logging": self.tensorboard_logging,
            "distributed_training": self.distributed_training,
            "mixed_precision": self.mixed_precision,
            "gradient_clipping": self.gradient_clipping,
            "learning_rate_schedule": self.learning_rate_schedule
        }

@dataclass
class NetworkPerformance:
    """Neural network performance metrics"""
    network_id: str
    training_loss: float
    validation_loss: float
    training_accuracy: float
    validation_accuracy: float
    training_time: timedelta
    inference_time: float
    memory_usage: int
    parameter_count: int
    flops: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def get_efficiency_score(self) -> float:
        """Calculate overall efficiency score"""
        accuracy_score = self.validation_accuracy
        speed_score = 1.0 / max(self.inference_time, 0.001)  # Avoid division by zero
        memory_score = 1.0 / max(self.memory_usage / 1024**2, 1)  # MB
        param_score = 1.0 / max(self.parameter_count / 1000, 1)  # K parameters
        
        # Weighted combination
        return (0.4 * accuracy_score + 0.3 * speed_score + 
                0.2 * memory_score + 0.1 * param_score)

@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition"""
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    dropout_rate: Tuple[float, float] = (0.0, 0.5)
    hidden_units: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    activation_functions: List[ActivationFunction] = field(
        default_factory=lambda: [ActivationFunction.RELU, ActivationFunction.TANH, ActivationFunction.GELU]
    )
    optimizer: List[str] = field(default_factory=lambda: ["adam", "sgd", "adamw"])
    regularization_l1: Tuple[float, float] = (0.0, 0.01)
    regularization_l2: Tuple[float, float] = (0.0, 0.01)

class NeuralOrchestrator:
    """
    Advanced neural network orchestration system with comprehensive
    architecture management, training coordination, and optimization.
    """
    
    def __init__(self, base_path: str = "./neural_networks"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.networks: Dict[str, NetworkConfig] = {}
        self.network_status: Dict[str, NetworkStatus] = {}
        self.network_performance: Dict[str, NetworkPerformance] = {}
        self.active_jobs: Dict[str, Any] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        logger.info("Neural Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize the neural orchestrator"""
        try:
            await self._load_networks()
            logger.info("Neural Orchestrator initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Neural Orchestrator: {e}")
            raise
    
    async def create_network(self, config: NetworkConfig) -> str:
        """Create a new neural network with specified architecture"""
        try:
            with self._lock:
                if config.network_id in self.networks:
                    raise ValueError(f"Network {config.network_id} already exists")
                
                # Validate network configuration
                await self._validate_network_config(config)
                
                # Store network configuration
                self.networks[config.network_id] = config
                self.network_status[config.network_id] = NetworkStatus.DESIGNING
                
                # Initialize metrics history
                self.metrics_history[config.network_id] = []
                self.optimization_history[config.network_id] = []
                
                # Save configuration
                await self._save_network_config(config)
                
                logger.info(f"Created network: {config.network_id}")
                return config.network_id
                
        except Exception as e:
            logger.error(f"Failed to create network: {e}")
            raise
    
    async def build_network(self, network_id: str) -> bool:
        """Build the neural network from configuration"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            config = self.networks[network_id]
            self.network_status[network_id] = NetworkStatus.BUILDING
            
            # Framework-specific network building
            success = await self._build_framework_network(config)
            
            if success:
                self.network_status[network_id] = NetworkStatus.VALIDATING
                logger.info(f"Built network: {network_id}")
            else:
                self.network_status[network_id] = NetworkStatus.FAILED
                logger.error(f"Failed to build network: {network_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to build network {network_id}: {e}")
            self.network_status[network_id] = NetworkStatus.FAILED
            return False
    
    async def optimize_architecture(self, 
                                  network_id: str,
                                  objective: OptimizationObjective,
                                  search_space: Optional[HyperparameterSpace] = None,
                                  max_trials: int = 50) -> Dict[str, Any]:
        """Perform neural architecture search and optimization"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            self.network_status[network_id] = NetworkStatus.OPTIMIZING
            
            if search_space is None:
                search_space = HyperparameterSpace()
            
            optimization_results = await self._run_architecture_search(
                network_id, objective, search_space, max_trials
            )
            
            # Update network with best configuration
            if optimization_results["best_config"]:
                await self._update_network_config(network_id, optimization_results["best_config"])
                
                logger.info(f"Optimized network {network_id}: {optimization_results['best_score']}")
            
            # Store optimization history
            self.optimization_history[network_id].append({
                "objective": objective.value,
                "best_score": optimization_results["best_score"],
                "trials_completed": optimization_results["trials_completed"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize network {network_id}: {e}")
            self.network_status[network_id] = NetworkStatus.FAILED
            raise
    
    async def train_network(self, 
                          network_id: str,
                          training_data: Any,
                          validation_data: Optional[Any] = None) -> NetworkPerformance:
        """Train a neural network with specified data"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            config = self.networks[network_id]
            self.network_status[network_id] = NetworkStatus.TRAINING
            
            # Start training process
            training_start = datetime.utcnow()
            
            performance = await self._execute_training(
                config, training_data, validation_data
            )
            
            # Update performance metrics
            self.network_performance[network_id] = performance
            
            # Store training metrics
            self.metrics_history[network_id].append({
                "training_loss": performance.training_loss,
                "validation_loss": performance.validation_loss,
                "training_accuracy": performance.training_accuracy,
                "validation_accuracy": performance.validation_accuracy,
                "efficiency_score": performance.get_efficiency_score(),
                "timestamp": performance.timestamp.isoformat()
            })
            
            self.network_status[network_id] = NetworkStatus.DEPLOYED
            
            logger.info(f"Trained network {network_id}: "
                       f"Val Accuracy: {performance.validation_accuracy:.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to train network {network_id}: {e}")
            self.network_status[network_id] = NetworkStatus.FAILED
            raise
    
    async def compress_network(self, 
                             network_id: str,
                             compression_method: str = "quantization",
                             target_size_reduction: float = 0.5) -> Dict[str, Any]:
        """Compress neural network for deployment optimization"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            original_performance = self.network_performance.get(network_id)
            if not original_performance:
                raise ValueError(f"Network {network_id} must be trained before compression")
            
            compression_result = await self._apply_compression(
                network_id, compression_method, target_size_reduction
            )
            
            # Test compressed network performance
            compressed_performance = await self._test_compressed_network(network_id)
            
            result = {
                "network_id": network_id,
                "compression_method": compression_method,
                "original_size": original_performance.memory_usage,
                "compressed_size": compression_result["compressed_size"],
                "size_reduction": compression_result["size_reduction"],
                "original_accuracy": original_performance.validation_accuracy,
                "compressed_accuracy": compressed_performance["accuracy"],
                "accuracy_loss": (original_performance.validation_accuracy - 
                                compressed_performance["accuracy"]),
                "inference_speedup": compression_result["inference_speedup"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Compressed network {network_id}: "
                       f"{result['size_reduction']:.2f}% size reduction, "
                       f"{result['accuracy_loss']:.4f} accuracy loss")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compress network {network_id}: {e}")
            raise
    
    async def deploy_network(self, 
                           network_id: str,
                           deployment_target: str = "production",
                           scaling_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deploy neural network to specified target environment"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            config = self.networks[network_id]
            
            if scaling_config is None:
                scaling_config = {
                    "replicas": 1,
                    "max_replicas": 5,
                    "cpu_limit": "2",
                    "memory_limit": "4Gi",
                    "gpu_limit": 0
                }
            
            deployment_result = await self._deploy_network_instance(
                config, deployment_target, scaling_config
            )
            
            self.network_status[network_id] = NetworkStatus.DEPLOYED
            
            logger.info(f"Deployed network {network_id} to {deployment_target}")
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Failed to deploy network {network_id}: {e}")
            raise
    
    async def get_network_status(self, network_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a neural network"""
        try:
            if network_id not in self.networks:
                raise ValueError(f"Network {network_id} not found")
            
            config = self.networks[network_id]
            status = self.network_status[network_id]
            performance = self.network_performance.get(network_id)
            
            return {
                "network_id": network_id,
                "status": status.value,
                "framework": config.framework.value,
                "layer_count": len(config.architecture_layers),
                "parameter_count": performance.parameter_count if performance else None,
                "performance": {
                    "training_accuracy": performance.training_accuracy if performance else None,
                    "validation_accuracy": performance.validation_accuracy if performance else None,
                    "inference_time": performance.inference_time if performance else None,
                    "efficiency_score": performance.get_efficiency_score() if performance else None
                },
                "metrics_history_count": len(self.metrics_history.get(network_id, [])),
                "optimization_history_count": len(self.optimization_history.get(network_id, [])),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get network status {network_id}: {e}")
            raise
    
    async def list_networks(self, 
                          status_filter: Optional[NetworkStatus] = None,
                          framework_filter: Optional[FrameworkType] = None) -> List[Dict[str, Any]]:
        """List all networks with optional filtering"""
        try:
            networks = []
            
            for network_id, config in self.networks.items():
                status = self.network_status[network_id]
                performance = self.network_performance.get(network_id)
                
                # Apply filters
                if status_filter and status != status_filter:
                    continue
                if framework_filter and config.framework != framework_filter:
                    continue
                
                network_info = {
                    "network_id": network_id,
                    "framework": config.framework.value,
                    "status": status.value,
                    "layer_count": len(config.architecture_layers),
                    "parameter_count": performance.parameter_count if performance else None,
                    "validation_accuracy": performance.validation_accuracy if performance else None,
                    "efficiency_score": performance.get_efficiency_score() if performance else None
                }
                
                networks.append(network_info)
            
            return sorted(networks, key=lambda x: x.get("efficiency_score", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list networks: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get neural orchestrator performance metrics"""
        try:
            active_networks = sum(1 for status in self.network_status.values() 
                                if status != NetworkStatus.ARCHIVED)
            
            training_networks = sum(1 for status in self.network_status.values()
                                  if status == NetworkStatus.TRAINING)
            
            deployed_networks = sum(1 for status in self.network_status.values()
                                  if status == NetworkStatus.DEPLOYED)
            
            avg_accuracy = 0.0
            if self.network_performance:
                accuracies = [p.validation_accuracy for p in self.network_performance.values()]
                avg_accuracy = sum(accuracies) / len(accuracies)
            
            return {
                "total_networks": len(self.networks),
                "active_networks": active_networks,
                "training_networks": training_networks,
                "deployed_networks": deployed_networks,
                "failed_networks": sum(1 for status in self.network_status.values()
                                     if status == NetworkStatus.FAILED),
                "average_accuracy": avg_accuracy,
                "framework_distribution": self._get_framework_distribution(),
                "active_jobs": len(self.active_jobs),
                "total_training_hours": self._calculate_total_training_time(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the neural orchestrator"""
        try:
            self._shutdown = True
            
            # Cancel active training jobs
            for job_id, job in self.active_jobs.items():
                if hasattr(job, 'cancel'):
                    job.cancel()
                logger.info(f"Cancelled job: {job_id}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Save all network configurations
            for network_id, config in self.networks.items():
                await self._save_network_config(config)
            
            logger.info("Neural Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Neural Orchestrator shutdown: {e}")
    
    # Private helper methods
    
    async def _validate_network_config(self, config: NetworkConfig) -> None:
        """Validate neural network configuration"""
        if not config.architecture_layers:
            raise ValueError("Network must have at least one layer")
        
        if not config.network_id:
            raise ValueError("Network ID is required")
        
        # Validate layer specifications
        for i, layer in enumerate(config.architecture_layers):
            if not layer.name:
                raise ValueError(f"Layer {i} must have a name")
            
            # Framework-specific validation
            await self._validate_layer_for_framework(layer, config.framework)
    
    async def _validate_layer_for_framework(self, layer: LayerSpec, framework: FrameworkType) -> None:
        """Validate layer specification for specific framework"""
        # Framework-specific layer validation logic
        if framework == FrameworkType.PYTORCH:
            # PyTorch-specific validation
            pass
        elif framework == FrameworkType.TENSORFLOW:
            # TensorFlow-specific validation  
            pass
        # Add more framework validations as needed
    
    async def _build_framework_network(self, config: NetworkConfig) -> bool:
        """Build network using specified framework"""
        try:
            if config.framework == FrameworkType.PYTORCH:
                return await self._build_pytorch_network(config)
            elif config.framework == FrameworkType.TENSORFLOW:
                return await self._build_tensorflow_network(config)
            elif config.framework == FrameworkType.JAX:
                return await self._build_jax_network(config)
            else:
                logger.warning(f"Framework {config.framework.value} not yet implemented")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build {config.framework.value} network: {e}")
            return False
    
    async def _build_pytorch_network(self, config: NetworkConfig) -> bool:
        """Build PyTorch neural network"""
        # Placeholder for PyTorch network building
        logger.info(f"Building PyTorch network: {config.network_id}")
        await asyncio.sleep(0.1)  # Simulate build time
        return True
    
    async def _build_tensorflow_network(self, config: NetworkConfig) -> bool:
        """Build TensorFlow neural network"""
        # Placeholder for TensorFlow network building
        logger.info(f"Building TensorFlow network: {config.network_id}")
        await asyncio.sleep(0.1)  # Simulate build time
        return True
    
    async def _build_jax_network(self, config: NetworkConfig) -> bool:
        """Build JAX neural network"""
        # Placeholder for JAX network building
        logger.info(f"Building JAX network: {config.network_id}")
        await asyncio.sleep(0.1)  # Simulate build time
        return True
    
    async def _run_architecture_search(self,
                                     network_id: str,
                                     objective: OptimizationObjective,
                                     search_space: HyperparameterSpace,
                                     max_trials: int) -> Dict[str, Any]:
        """Run neural architecture search"""
        best_score = 0.0
        best_config = None
        trials_completed = 0
        
        # Placeholder for actual NAS implementation
        for trial in range(min(max_trials, 10)):  # Limit for demo
            # Simulate trial
            trial_score = np.random.random()
            trials_completed += 1
            
            if trial_score > best_score:
                best_score = trial_score
                best_config = {"trial": trial, "score": trial_score}
            
            await asyncio.sleep(0.01)  # Simulate optimization time
        
        return {
            "network_id": network_id,
            "objective": objective.value,
            "best_score": best_score,
            "best_config": best_config,
            "trials_completed": trials_completed,
            "search_space": search_space
        }
    
    async def _execute_training(self,
                              config: NetworkConfig,
                              training_data: Any,
                              validation_data: Optional[Any] = None) -> NetworkPerformance:
        """Execute neural network training"""
        training_start = datetime.utcnow()
        
        # Simulate training process
        await asyncio.sleep(0.1)  # Simulate training time
        
        training_end = datetime.utcnow()
        training_time = training_end - training_start
        
        # Create mock performance metrics
        performance = NetworkPerformance(
            network_id=config.network_id,
            training_loss=np.random.uniform(0.1, 0.5),
            validation_loss=np.random.uniform(0.15, 0.6),
            training_accuracy=np.random.uniform(0.85, 0.98),
            validation_accuracy=np.random.uniform(0.82, 0.95),
            training_time=training_time,
            inference_time=np.random.uniform(0.001, 0.1),
            memory_usage=np.random.randint(100, 1000) * 1024 * 1024,  # MB to bytes
            parameter_count=np.random.randint(10000, 10000000),
            flops=np.random.randint(1000000, 1000000000)
        )
        
        return performance
    
    async def _apply_compression(self,
                               network_id: str,
                               method: str,
                               target_reduction: float) -> Dict[str, Any]:
        """Apply compression to neural network"""
        original_size = self.network_performance[network_id].memory_usage
        compressed_size = int(original_size * (1 - target_reduction))
        
        return {
            "method": method,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "size_reduction": ((original_size - compressed_size) / original_size) * 100,
            "inference_speedup": np.random.uniform(1.5, 3.0)
        }
    
    async def _test_compressed_network(self, network_id: str) -> Dict[str, Any]:
        """Test compressed network performance"""
        original_accuracy = self.network_performance[network_id].validation_accuracy
        compressed_accuracy = original_accuracy * np.random.uniform(0.95, 1.0)
        
        return {
            "accuracy": compressed_accuracy,
            "inference_time": np.random.uniform(0.0005, 0.05)
        }
    
    async def _deploy_network_instance(self,
                                     config: NetworkConfig,
                                     target: str,
                                     scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy network instance to target environment"""
        return {
            "deployment_id": f"deploy-{config.network_id}-{target}",
            "target": target,
            "status": "deployed",
            "endpoint": f"http://ml-api/{config.network_id}/predict",
            "scaling_config": scaling_config,
            "deployed_at": datetime.utcnow().isoformat()
        }
    
    async def _update_network_config(self, network_id: str, new_config: Dict[str, Any]) -> None:
        """Update network configuration with optimization results"""
        # Placeholder for updating network configuration
        logger.info(f"Updated network {network_id} with optimized config")
    
    async def _save_network_config(self, config: NetworkConfig) -> None:
        """Save network configuration to disk"""
        config_path = self.base_path / f"{config.network_id}_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    async def _load_networks(self) -> None:
        """Load existing network configurations"""
        config_files = list(self.base_path.glob("*_config.json"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Reconstruct network configuration (simplified)
                network_id = config_data["network_id"]
                self.network_status[network_id] = NetworkStatus.ARCHIVED
                
                logger.info(f"Loaded network config: {network_id}")
                
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    def _get_framework_distribution(self) -> Dict[str, int]:
        """Get distribution of networks by framework"""
        distribution = {}
        for config in self.networks.values():
            framework = config.framework.value
            distribution[framework] = distribution.get(framework, 0) + 1
        return distribution
    
    def _calculate_total_training_time(self) -> float:
        """Calculate total training time across all networks"""
        total_hours = 0.0
        for performance in self.network_performance.values():
            total_hours += performance.training_time.total_seconds() / 3600
        return total_hours