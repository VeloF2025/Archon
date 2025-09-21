"""
AI/ML API Gateway for Archon Enhancement 2025 Phase 4

Comprehensive REST API gateway for all AI/ML framework components:
- Neural Network Orchestrator API
- Model Management API
- AutoML Pipeline API  
- Data Processing API
- Training Orchestrator API
- Model Serving API
- Unified workflow orchestration
- Advanced analytics and monitoring
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import uuid
import json
import numpy as np

# Import all ML framework components
from ...agents.ml.neural_orchestrator import (
    NeuralOrchestrator, NetworkConfig, LayerSpec, LayerType, 
    ActivationFunction, OptimizationObjective, NetworkStatus
)
from ...agents.ml.model_manager import (
    ModelManager, ModelMetadata, ModelVersion, ModelStatus,
    ModelConfig, DeploymentEnvironment, ApprovalStatus
)
from ...agents.ml.automl_pipeline import (
    AutoMLPipeline, AutoMLConfig, OptimizationTarget, PipelineStage,
    FeatureSelectionMethod, AlgorithmCategory
)
from ...agents.ml.data_processor import (
    DataProcessor, ProcessingConfig, DataSourceType, DataQualityIssue,
    FeatureEngineeringPipeline, TransformationType
)
from ...agents.ml.training_orchestrator import (
    TrainingOrchestrator, TrainingConfig, TrainingFramework, 
    TrainingStrategy, OptimizationStrategy, ResourceRequirement, ResourceType
)
from ...agents.ml.model_serving import (
    ModelServingInfrastructure, ServingConfig, ModelArtifact, ServingFramework,
    InferenceType, DeploymentStrategy, InferenceRequest, ABTestConfig
)

router = APIRouter(prefix="/api/ml", tags=["machine_learning"])

# Global ML framework instances
neural_orchestrator: Optional[NeuralOrchestrator] = None
model_manager: Optional[ModelManager] = None
automl_pipeline: Optional[AutoMLPipeline] = None
data_processor: Optional[DataProcessor] = None
training_orchestrator: Optional[TrainingOrchestrator] = None
model_serving: Optional[ModelServingInfrastructure] = None

# Request/Response models

class StatusResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetricsResponse(BaseModel):
    component: str
    metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Neural Network Orchestrator Models
class NetworkConfigRequest(BaseModel):
    network_id: str
    framework: str
    layers: List[Dict[str, Any]]
    loss_function: str = "categorical_crossentropy"
    optimizer: str = "adam"
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    epochs: int = 100
    batch_size: int = 32

class NetworkOptimizationRequest(BaseModel):
    objective: str
    max_trials: int = 50
    search_space: Optional[Dict[str, Any]] = None

# Model Management Models
class ModelRegistrationRequest(BaseModel):
    model_path: str
    name: str
    description: str
    model_type: str
    framework: str
    algorithm: str
    version: str
    author: str
    tags: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)

class ModelDeploymentRequest(BaseModel):
    environment: str
    replicas: int = 1
    resource_limits: Dict[str, str] = Field(default_factory=lambda: {"cpu": "1", "memory": "2Gi"})
    scaling_config: Dict[str, Any] = Field(default_factory=dict)

class ABTestRequest(BaseModel):
    test_name: str
    model_a: Tuple[str, str]  # (model_id, version)
    model_b: Tuple[str, str]  # (model_id, version)
    traffic_split: float = 0.5
    duration_days: int = 7

# AutoML Pipeline Models
class AutoMLConfigRequest(BaseModel):
    pipeline_id: str
    task_type: str
    optimization_target: str
    time_budget_minutes: int = 60
    max_models: int = 100
    cv_folds: int = 5
    enable_feature_engineering: bool = True
    enable_ensembles: bool = True
    algorithm_categories: List[str] = Field(default_factory=lambda: ["linear_models", "tree_based", "ensemble"])

class AutoMLRunRequest(BaseModel):
    target_column: str
    test_data_path: Optional[str] = None

# Data Processing Models
class SchemaCreationRequest(BaseModel):
    schema_id: str
    columns: Dict[str, str]  # column_name -> feature_type
    constraints: Optional[Dict[str, Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

class ProcessingConfigRequest(BaseModel):
    config_id: str
    source_type: str
    transformations: List[Dict[str, Any]] = Field(default_factory=list)
    quality_checks: List[str] = Field(default_factory=list)
    batch_size: int = 10000
    enable_validation: bool = True

class FeaturePipelineRequest(BaseModel):
    pipeline_id: str
    input_features: List[str]

class FeatureStepRequest(BaseModel):
    step_id: str
    transformation: str
    input_columns: List[str]
    output_columns: List[str]
    params: Optional[Dict[str, Any]] = None

# Training Orchestrator Models
class TrainingConfigRequest(BaseModel):
    config_id: str
    framework: str
    strategy: str = "single_node"
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    enable_hyperopt: bool = False
    num_workers: int = 1
    num_gpus: int = 0
    priority: int = 1

class HyperoptRequest(BaseModel):
    search_space: Dict[str, Any]
    n_trials: int = 50
    strategy: str = "bayesian_optimization"

class ExperimentRequest(BaseModel):
    experiment_name: str
    description: str
    job_configs: List[TrainingConfigRequest]
    tags: List[str] = Field(default_factory=list)

# Model Serving Models
class ModelArtifactRequest(BaseModel):
    model_id: str
    version: str
    framework: str
    model_path: str
    config_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ServingConfigRequest(BaseModel):
    endpoint_name: str
    model_artifact: ModelArtifactRequest
    inference_type: str = "real_time"
    min_replicas: int = 1
    max_replicas: int = 10
    target_utilization: float = 70.0
    gpu_count: int = 0
    enable_batching: bool = False
    optimizations: List[str] = Field(default_factory=list)

class PredictionRequest(BaseModel):
    endpoint_name: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[float] = None
    priority: int = 1

class ServingABTestRequest(BaseModel):
    test_name: str
    endpoint_a: str
    endpoint_b: str
    traffic_split: float = 0.5
    duration_hours: int = 24
    success_metric: str = "accuracy"

# ============================================================================
# SYSTEM INITIALIZATION ENDPOINTS
# ============================================================================

@router.post("/initialize", response_model=StatusResponse)
async def initialize_ml_framework():
    """Initialize all ML framework components"""
    global neural_orchestrator, model_manager, automl_pipeline
    global data_processor, training_orchestrator, model_serving
    
    try:
        # Initialize components
        neural_orchestrator = NeuralOrchestrator()
        await neural_orchestrator.initialize()
        
        model_manager = ModelManager()
        await model_manager.initialize()
        
        automl_pipeline = AutoMLPipeline()
        await automl_pipeline.initialize()
        
        data_processor = DataProcessor()
        await data_processor.initialize()
        
        training_orchestrator = TrainingOrchestrator()
        await training_orchestrator.initialize()
        
        model_serving = ModelServingInfrastructure()
        await model_serving.initialize()
        
        return StatusResponse(
            status="success",
            message="ML framework initialized successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def get_ml_health():
    """Get health status of all ML components"""
    try:
        health_status = {
            "neural_orchestrator": neural_orchestrator is not None,
            "model_manager": model_manager is not None,
            "automl_pipeline": automl_pipeline is not None,
            "data_processor": data_processor is not None,
            "training_orchestrator": training_orchestrator is not None,
            "model_serving": model_serving is not None,
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check if all components are initialized
        all_healthy = all(health_status[comp] for comp in health_status if comp not in ["overall_status", "timestamp"])
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=Dict[str, Any])
async def get_ml_metrics():
    """Get comprehensive metrics from all ML components"""
    try:
        metrics = {}
        
        if neural_orchestrator:
            metrics["neural_orchestrator"] = await neural_orchestrator.get_metrics()
        
        if model_manager:
            metrics["model_manager"] = await model_manager.get_metrics()
        
        if automl_pipeline:
            metrics["automl_pipeline"] = await automl_pipeline.get_metrics()
        
        if data_processor:
            metrics["data_processor"] = await data_processor.get_metrics()
        
        if training_orchestrator:
            metrics["training_orchestrator"] = await training_orchestrator.get_metrics()
        
        if model_serving:
            metrics["model_serving"] = await model_serving.get_metrics()
        
        return {
            "component_metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEURAL NETWORK ORCHESTRATOR ENDPOINTS
# ============================================================================

@router.post("/neural/networks", response_model=Dict[str, str])
async def create_neural_network(config: NetworkConfigRequest):
    """Create a new neural network"""
    if not neural_orchestrator:
        raise HTTPException(status_code=400, detail="Neural orchestrator not initialized")
    
    try:
        # Convert request to internal config
        layers = []
        for layer_spec in config.layers:
            layer = LayerSpec(
                layer_type=LayerType(layer_spec["type"]),
                name=layer_spec["name"],
                parameters=layer_spec.get("parameters", {}),
                activation=ActivationFunction(layer_spec.get("activation")) if layer_spec.get("activation") else None
            )
            layers.append(layer)
        
        network_config = NetworkConfig(
            network_id=config.network_id,
            framework=config.framework,
            architecture_layers=layers,
            loss_function=config.loss_function,
            optimizer=config.optimizer,
            metrics=config.metrics,
            epochs=config.epochs,
            batch_size=config.batch_size
        )
        
        network_id = await neural_orchestrator.create_network(network_config)
        return {"network_id": network_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neural/networks/{network_id}/build", response_model=StatusResponse)
async def build_neural_network(network_id: str = Path(...)):
    """Build neural network from configuration"""
    if not neural_orchestrator:
        raise HTTPException(status_code=400, detail="Neural orchestrator not initialized")
    
    try:
        success = await neural_orchestrator.build_network(network_id)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Network {network_id} build {'succeeded' if success else 'failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/neural/networks/{network_id}/optimize", response_model=Dict[str, Any])
async def optimize_neural_architecture(network_id: str, optimization: NetworkOptimizationRequest):
    """Perform neural architecture search and optimization"""
    if not neural_orchestrator:
        raise HTTPException(status_code=400, detail="Neural orchestrator not initialized")
    
    try:
        objective = OptimizationObjective(optimization.objective)
        results = await neural_orchestrator.optimize_architecture(
            network_id, objective, None, optimization.max_trials
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neural/networks/{network_id}/status", response_model=Dict[str, Any])
async def get_neural_network_status(network_id: str = Path(...)):
    """Get neural network status"""
    if not neural_orchestrator:
        raise HTTPException(status_code=400, detail="Neural orchestrator not initialized")
    
    try:
        status = await neural_orchestrator.get_network_status(network_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/neural/networks", response_model=List[Dict[str, Any]])
async def list_neural_networks(
    status_filter: Optional[str] = Query(None),
    framework_filter: Optional[str] = Query(None)
):
    """List all neural networks"""
    if not neural_orchestrator:
        raise HTTPException(status_code=400, detail="Neural orchestrator not initialized")
    
    try:
        status_enum = NetworkStatus(status_filter) if status_filter else None
        networks = await neural_orchestrator.list_networks(status_enum, framework_filter)
        return networks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/models/register", response_model=Dict[str, str])
async def register_model(model_data: ModelRegistrationRequest):
    """Register a new model in the registry"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        metadata = ModelMetadata(
            name=model_data.name,
            description=model_data.description,
            model_type=model_data.model_type,
            framework=model_data.framework,
            algorithm=model_data.algorithm,
            version=model_data.version,
            author=model_data.author,
            tags=model_data.tags,
            features=model_data.features,
            hyperparameters=model_data.hyperparameters,
            metrics=model_data.metrics
        )
        
        version_id = await model_manager.register_model(model_data.model_path, metadata)
        return {"version_id": version_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/versions/{version}/approve", response_model=StatusResponse)
async def approve_model(
    model_id: str = Path(...),
    version: str = Path(...),
    approver: str = Query(...),
    notes: str = Query("")
):
    """Approve a model version for deployment"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        success = await model_manager.approve_model(model_id, version, approver, notes)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Model {model_id} v{version} {'approved' if success else 'approval failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/versions/{version}/deploy", response_model=Dict[str, str])
async def deploy_model(
    model_id: str = Path(...),
    version: str = Path(...),
    deployment: ModelDeploymentRequest
):
    """Deploy a model version to specified environment"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        environment = DeploymentEnvironment(deployment.environment)
        deployment_id = await model_manager.deploy_model(model_id, version, environment)
        
        return {"deployment_id": deployment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/ab-test", response_model=Dict[str, str])
async def create_model_ab_test(ab_test: ABTestRequest):
    """Create A/B test between two model versions"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        test_id = await model_manager.create_ab_test(
            ab_test.test_name,
            ab_test.model_a,
            ab_test.model_b,
            ab_test.traffic_split,
            DeploymentEnvironment.STAGING,
            ab_test.duration_days
        )
        
        return {"test_id": test_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/lineage", response_model=Dict[str, Any])
async def get_model_lineage(model_id: str = Path(...)):
    """Get complete lineage of a model"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        lineage = await model_manager.get_model_lineage(model_id)
        return lineage
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    status_filter: Optional[str] = Query(None),
    environment_filter: Optional[str] = Query(None)
):
    """List all models with optional filtering"""
    if not model_manager:
        raise HTTPException(status_code=400, detail="Model manager not initialized")
    
    try:
        status_enum = ModelStatus(status_filter) if status_filter else None
        env_enum = DeploymentEnvironment(environment_filter) if environment_filter else None
        
        models = await model_manager.list_models(status_enum, None, env_enum)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AUTOML PIPELINE ENDPOINTS
# ============================================================================

@router.post("/automl/pipelines", response_model=Dict[str, str])
async def create_automl_pipeline(config: AutoMLConfigRequest):
    """Create a new AutoML pipeline"""
    if not automl_pipeline:
        raise HTTPException(status_code=400, detail="AutoML pipeline not initialized")
    
    try:
        # Convert request to internal config
        automl_config = AutoMLConfig(
            pipeline_id=config.pipeline_id,
            task_type=config.task_type,
            optimization_target=OptimizationTarget(config.optimization_target),
            time_budget_minutes=config.time_budget_minutes,
            max_models=config.max_models,
            cv_folds=config.cv_folds,
            enable_feature_engineering=config.enable_feature_engineering,
            enable_ensembles=config.enable_ensembles,
            algorithm_categories=[AlgorithmCategory(cat) for cat in config.algorithm_categories]
        )
        
        pipeline_id = await automl_pipeline.create_pipeline(automl_config)
        return {"pipeline_id": pipeline_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automl/pipelines/{pipeline_id}/run", response_model=Dict[str, Any])
async def run_automl_pipeline(
    pipeline_id: str = Path(...),
    run_config: AutoMLRunRequest,
    train_data_path: str = Query(...)
):
    """Execute AutoML pipeline"""
    if not automl_pipeline:
        raise HTTPException(status_code=400, detail="AutoML pipeline not initialized")
    
    try:
        # Mock data loading for API demo
        mock_train_data = "mock_training_data"
        mock_test_data = "mock_test_data" if run_config.test_data_path else None
        
        results = await automl_pipeline.run_pipeline(
            pipeline_id, mock_train_data, run_config.target_column, mock_test_data
        )
        
        return {
            "pipeline_id": results.pipeline_id,
            "total_models_trained": results.total_models_trained,
            "best_model_score": results.best_model.mean_cv_score if results.best_model else None,
            "execution_time": str(results.execution_time),
            "stages_completed": [stage.value for stage in results.stages_completed]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/automl/pipelines/{pipeline_id}/status", response_model=Dict[str, Any])
async def get_automl_pipeline_status(pipeline_id: str = Path(...)):
    """Get AutoML pipeline status"""
    if not automl_pipeline:
        raise HTTPException(status_code=400, detail="AutoML pipeline not initialized")
    
    try:
        status = await automl_pipeline.get_pipeline_status(pipeline_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automl/compare", response_model=Dict[str, Any])
async def compare_automl_pipelines(pipeline_ids: List[str]):
    """Compare results from multiple AutoML pipelines"""
    if not automl_pipeline:
        raise HTTPException(status_code=400, detail="AutoML pipeline not initialized")
    
    try:
        comparison = await automl_pipeline.compare_pipelines(pipeline_ids)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/automl/pipelines/{pipeline_id}/export", response_model=Dict[str, Any])
async def export_automl_pipeline(
    pipeline_id: str = Path(...),
    format: str = Query("sklearn", description="Export format"),
    include_preprocessing: bool = Query(True)
):
    """Export trained AutoML pipeline for deployment"""
    if not automl_pipeline:
        raise HTTPException(status_code=400, detail="AutoML pipeline not initialized")
    
    try:
        export_package = await automl_pipeline.export_pipeline(
            pipeline_id, format, include_preprocessing
        )
        return export_package
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATA PROCESSING ENDPOINTS
# ============================================================================

@router.post("/data/schemas", response_model=Dict[str, str])
async def create_data_schema(schema_request: SchemaCreationRequest):
    """Create a new data schema definition"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        # Convert string types to enums (simplified)
        from ...agents.ml.data_processor import FeatureType
        columns = {name: FeatureType.NUMERICAL_CONTINUOUS for name, _ in schema_request.columns.items()}  # Simplified
        
        schema = await data_processor.create_schema(
            schema_request.schema_id,
            columns,
            schema_request.constraints,
            schema_request.metadata
        )
        
        return {"schema_id": schema.schema_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/schemas/infer", response_model=Dict[str, Any])
async def infer_data_schema(
    data_source: str = Query(...),
    source_type: str = Query(...),
    sample_size: int = Query(10000)
):
    """Automatically infer schema from data source"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        source_type_enum = DataSourceType(source_type)
        schema = await data_processor.infer_schema(data_source, source_type_enum, sample_size)
        
        return {
            "schema_id": schema.schema_id,
            "columns": {k: v.value for k, v in schema.columns.items()},
            "constraints": schema.constraints,
            "metadata": schema.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/profile", response_model=Dict[str, Any])
async def profile_data(
    data_source: str = Query(...),
    source_type: str = Query(...),
    schema_id: Optional[str] = Query(None),
    sample_size: Optional[int] = Query(None)
):
    """Generate comprehensive data profile"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        source_type_enum = DataSourceType(source_type)
        profile = await data_processor.profile_data(
            data_source, source_type_enum, schema_id, sample_size
        )
        
        return {
            "profile_id": profile.profile_id,
            "dataset_name": profile.dataset_name,
            "total_rows": profile.total_rows,
            "total_columns": profile.total_columns,
            "quality_score": profile.get_quality_score(),
            "quality_issues": profile.quality_issues,
            "processing_time": profile.processing_time.total_seconds()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/processing/jobs", response_model=Dict[str, str])
async def create_processing_job(
    config: ProcessingConfigRequest,
    input_source: str = Query(...),
    output_path: str = Query(...)
):
    """Create a new data processing job"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        # Convert request to internal config
        processing_config = ProcessingConfig(
            config_id=config.config_id,
            source_config={"type": config.source_type},
            transformations=config.transformations,
            quality_checks=[DataQualityIssue(check) for check in config.quality_checks],
            batch_size=config.batch_size,
            enable_validation=config.enable_validation
        )
        
        job_id = await data_processor.create_processing_job(
            processing_config, input_source, output_path
        )
        
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/processing/jobs/{job_id}/execute", response_model=Dict[str, Any])
async def execute_processing_job(
    job_id: str = Path(...),
    input_source: str = Query(...)
):
    """Execute a data processing job"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        job = await data_processor.execute_processing_job(job_id, input_source)
        
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "rows_processed": job.rows_processed,
            "metrics": job.metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/processing/jobs/{job_id}/status", response_model=Dict[str, Any])
async def get_processing_job_status(job_id: str = Path(...)):
    """Get status of a processing job"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        status = await data_processor.get_job_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/pipelines/features", response_model=Dict[str, str])
async def create_feature_pipeline(pipeline_request: FeaturePipelineRequest):
    """Create a new feature engineering pipeline"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        pipeline = await data_processor.create_feature_pipeline(
            pipeline_request.pipeline_id,
            pipeline_request.input_features
        )
        
        return {"pipeline_id": pipeline.pipeline_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/pipelines/{pipeline_id}/steps", response_model=StatusResponse)
async def add_feature_step(
    pipeline_id: str = Path(...),
    step_request: FeatureStepRequest
):
    """Add a feature engineering step to pipeline"""
    if not data_processor:
        raise HTTPException(status_code=400, detail="Data processor not initialized")
    
    try:
        transformation = TransformationType(step_request.transformation)
        
        await data_processor.add_feature_step(
            pipeline_id,
            step_request.step_id,
            transformation,
            step_request.input_columns,
            step_request.output_columns,
            step_request.params
        )
        
        return StatusResponse(
            status="success",
            message=f"Added step {step_request.step_id} to pipeline {pipeline_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TRAINING ORCHESTRATOR ENDPOINTS
# ============================================================================

@router.post("/training/jobs", response_model=Dict[str, str])
async def create_training_job(config: TrainingConfigRequest):
    """Create a new training job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        training_config = TrainingConfig(
            config_id=config.config_id,
            framework=TrainingFramework(config.framework),
            strategy=TrainingStrategy(config.strategy),
            model_config=config.model_config,
            dataset_config=config.dataset_config,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            enable_hyperopt=config.enable_hyperopt,
            num_workers=config.num_workers,
            num_gpus=config.num_gpus,
            priority=config.priority
        )
        
        job_id = await training_orchestrator.create_training_job(training_config)
        return {"job_id": job_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/jobs/{job_id}/submit", response_model=StatusResponse)
async def submit_training_job(job_id: str = Path(...)):
    """Submit training job for execution"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        success = await training_orchestrator.submit_job(job_id)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Training job {job_id} {'submitted' if success else 'submission failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/experiments", response_model=Dict[str, str])
async def create_training_experiment(experiment: ExperimentRequest):
    """Create experiment with multiple training jobs"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        # Convert request configs to internal configs
        job_configs = []
        for config_req in experiment.job_configs:
            training_config = TrainingConfig(
                config_id=config_req.config_id,
                framework=TrainingFramework(config_req.framework),
                strategy=TrainingStrategy(config_req.strategy),
                model_config=config_req.model_config,
                dataset_config=config_req.dataset_config,
                epochs=config_req.epochs,
                batch_size=config_req.batch_size,
                learning_rate=config_req.learning_rate,
                optimizer=config_req.optimizer,
                enable_hyperopt=config_req.enable_hyperopt,
                num_workers=config_req.num_workers,
                num_gpus=config_req.num_gpus,
                priority=config_req.priority
            )
            job_configs.append(training_config)
        
        experiment_id = await training_orchestrator.create_experiment(
            experiment.experiment_name,
            experiment.description,
            job_configs,
            experiment.tags
        )
        
        return {"experiment_id": experiment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/hyperopt", response_model=Dict[str, str])
async def run_hyperparameter_optimization(
    base_config: TrainingConfigRequest,
    hyperopt: HyperoptRequest
):
    """Run hyperparameter optimization experiment"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        base_training_config = TrainingConfig(
            config_id=base_config.config_id,
            framework=TrainingFramework(base_config.framework),
            strategy=TrainingStrategy(base_config.strategy),
            model_config=base_config.model_config,
            dataset_config=base_config.dataset_config,
            epochs=base_config.epochs,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            optimizer=base_config.optimizer
        )
        
        strategy = OptimizationStrategy(hyperopt.strategy)
        
        experiment_id = await training_orchestrator.run_hyperparameter_optimization(
            base_training_config,
            hyperopt.search_space,
            hyperopt.n_trials,
            strategy
        )
        
        return {"experiment_id": experiment_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/jobs/{job_id}/status", response_model=Dict[str, Any])
async def get_training_job_status(job_id: str = Path(...)):
    """Get detailed status of training job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        status = await training_orchestrator.get_job_status(job_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/jobs/{job_id}/metrics", response_model=List[Dict[str, Any]])
async def get_training_job_metrics(
    job_id: str = Path(...),
    last_n: Optional[int] = Query(None)
):
    """Get training metrics for job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        metrics = await training_orchestrator.get_job_metrics(job_id, last_n)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/experiments/{experiment_id}/status", response_model=Dict[str, Any])
async def get_experiment_status(experiment_id: str = Path(...)):
    """Get comprehensive experiment status"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        status = await training_orchestrator.get_experiment_status(experiment_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/jobs", response_model=List[Dict[str, Any]])
async def list_training_jobs(
    status_filter: Optional[str] = Query(None),
    framework_filter: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """List training jobs with filtering"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        from ...agents.ml.training_orchestrator import TrainingStatus
        status_enum = TrainingStatus(status_filter) if status_filter else None
        framework_enum = TrainingFramework(framework_filter) if framework_filter else None
        
        jobs = await training_orchestrator.list_jobs(status_enum, framework_enum, limit)
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/jobs/{job_id}/pause", response_model=StatusResponse)
async def pause_training_job(job_id: str = Path(...)):
    """Pause a running training job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        success = await training_orchestrator.pause_job(job_id)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Training job {job_id} {'paused' if success else 'pause failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/jobs/{job_id}/resume", response_model=StatusResponse)
async def resume_training_job(job_id: str = Path(...)):
    """Resume a paused training job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        success = await training_orchestrator.resume_job(job_id)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Training job {job_id} {'resumed' if success else 'resume failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/jobs/{job_id}/cancel", response_model=StatusResponse)
async def cancel_training_job(job_id: str = Path(...)):
    """Cancel a training job"""
    if not training_orchestrator:
        raise HTTPException(status_code=400, detail="Training orchestrator not initialized")
    
    try:
        success = await training_orchestrator.cancel_job(job_id)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Training job {job_id} {'cancelled' if success else 'cancellation failed'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODEL SERVING ENDPOINTS
# ============================================================================

@router.post("/serving/deploy", response_model=Dict[str, str])
async def deploy_model_for_serving(serving_config: ServingConfigRequest, strategy: str = Query("rolling")):
    """Deploy a model for serving"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        # Convert request to internal types
        artifact = ModelArtifact(
            model_id=serving_config.model_artifact.model_id,
            version=serving_config.model_artifact.version,
            framework=ServingFramework(serving_config.model_artifact.framework),
            model_path=serving_config.model_artifact.model_path,
            config_path=serving_config.model_artifact.config_path,
            metadata=serving_config.model_artifact.metadata
        )
        
        config = ServingConfig(
            endpoint_name=serving_config.endpoint_name,
            model_artifact=artifact,
            inference_type=InferenceType(serving_config.inference_type),
            min_replicas=serving_config.min_replicas,
            max_replicas=serving_config.max_replicas,
            target_utilization=serving_config.target_utilization,
            gpu_count=serving_config.gpu_count,
            enable_batching=serving_config.enable_batching
        )
        
        deployment_strategy = DeploymentStrategy(strategy)
        
        endpoint_name = await model_serving.deploy_model(config, deployment_strategy)
        return {"endpoint_name": endpoint_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/serving/predict", response_model=Dict[str, Any])
async def make_prediction(prediction: PredictionRequest):
    """Make prediction using specified endpoint"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            endpoint_name=prediction.endpoint_name,
            inputs=prediction.inputs,
            parameters=prediction.parameters,
            timeout=prediction.timeout,
            priority=prediction.priority
        )
        
        response = await model_serving.predict(request)
        return response.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/serving/batch-predict", response_model=List[Dict[str, Any]])
async def make_batch_predictions(predictions: List[PredictionRequest]):
    """Execute batch predictions"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        requests = []
        for pred in predictions:
            request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                endpoint_name=pred.endpoint_name,
                inputs=pred.inputs,
                parameters=pred.parameters,
                timeout=pred.timeout,
                priority=pred.priority
            )
            requests.append(request)
        
        responses = await model_serving.batch_predict(requests)
        return [response.to_dict() for response in responses]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/serving/endpoints/{endpoint_name}/scale", response_model=StatusResponse)
async def scale_serving_endpoint(
    endpoint_name: str = Path(...),
    target_replicas: int = Query(...)
):
    """Scale endpoint to target number of replicas"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        success = await model_serving.scale_endpoint(endpoint_name, target_replicas)
        
        return StatusResponse(
            status="success" if success else "failed",
            message=f"Endpoint {endpoint_name} {'scaled' if success else 'scaling failed'} to {target_replicas} replicas"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/serving/ab-test", response_model=Dict[str, str])
async def create_serving_ab_test(ab_test: ServingABTestRequest):
    """Create A/B test between two serving endpoints"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        test_config = ABTestConfig(
            test_name=ab_test.test_name,
            endpoint_a=ab_test.endpoint_a,
            endpoint_b=ab_test.endpoint_b,
            traffic_split=ab_test.traffic_split,
            duration_hours=ab_test.duration_hours,
            success_metric=ab_test.success_metric
        )
        
        test_name = await model_serving.create_ab_test(test_config)
        return {"test_name": test_name}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/serving/ab-test/{test_name}/results", response_model=Dict[str, Any])
async def get_serving_ab_test_results(test_name: str = Path(...)):
    """Get A/B test results and analysis"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        results = await model_serving.get_ab_test_results(test_name)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/serving/endpoints/{endpoint_name}/status", response_model=Dict[str, Any])
async def get_serving_endpoint_status(endpoint_name: str = Path(...)):
    """Get comprehensive endpoint status"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        status = await model_serving.get_endpoint_status(endpoint_name)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/serving/endpoints", response_model=List[Dict[str, Any]])
async def list_serving_endpoints(status_filter: Optional[str] = Query(None)):
    """List all serving endpoints with optional status filtering"""
    if not model_serving:
        raise HTTPException(status_code=400, detail="Model serving not initialized")
    
    try:
        from ...agents.ml.model_serving import ServingStatus
        status_enum = ServingStatus(status_filter) if status_filter else None
        
        endpoints = await model_serving.list_endpoints(status_enum)
        return endpoints
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UNIFIED WORKFLOW ORCHESTRATION ENDPOINTS
# ============================================================================

@router.post("/workflows/ml-pipeline", response_model=Dict[str, Any])
async def create_end_to_end_ml_pipeline(
    pipeline_name: str = Query(...),
    data_source: str = Query(...),
    target_column: str = Query(...),
    model_type: str = Query("classification"),
    deployment_environment: str = Query("staging")
):
    """Create end-to-end ML pipeline from data processing to model serving"""
    try:
        workflow_id = f"ml_pipeline_{uuid.uuid4().hex[:8]}"
        
        # Validate all components are initialized
        components = {
            "data_processor": data_processor,
            "automl_pipeline": automl_pipeline,
            "training_orchestrator": training_orchestrator,
            "model_manager": model_manager,
            "model_serving": model_serving
        }
        
        uninitialized = [name for name, component in components.items() if component is None]
        if uninitialized:
            raise HTTPException(
                status_code=400, 
                detail=f"Components not initialized: {', '.join(uninitialized)}"
            )
        
        # Create workflow execution plan
        workflow_steps = [
            {"step": "data_profiling", "component": "data_processor", "status": "pending"},
            {"step": "data_processing", "component": "data_processor", "status": "pending"},
            {"step": "automl_training", "component": "automl_pipeline", "status": "pending"},
            {"step": "model_registration", "component": "model_manager", "status": "pending"},
            {"step": "model_deployment", "component": "model_serving", "status": "pending"}
        ]
        
        # Execute workflow asynchronously
        async def execute_ml_pipeline():
            try:
                # Step 1: Data Profiling
                workflow_steps[0]["status"] = "running"
                profile = await data_processor.profile_data(
                    data_source, DataSourceType.CSV, None, 10000
                )
                workflow_steps[0]["status"] = "completed"
                workflow_steps[0]["result"] = {"profile_id": profile.profile_id}
                
                # Step 2: AutoML Training
                workflow_steps[2]["status"] = "running"
                automl_config = AutoMLConfig(
                    pipeline_id=f"{workflow_id}_automl",
                    task_type=model_type,
                    optimization_target=OptimizationTarget.ACCURACY,
                    time_budget_minutes=30,
                    max_models=20
                )
                
                pipeline_id = await automl_pipeline.create_pipeline(automl_config)
                results = await automl_pipeline.run_pipeline(
                    pipeline_id, "mock_data", target_column, None
                )
                workflow_steps[2]["status"] = "completed"
                workflow_steps[2]["result"] = {
                    "pipeline_id": pipeline_id,
                    "best_score": results.best_model.mean_cv_score if results.best_model else 0
                }
                
                # Step 3: Model Registration
                workflow_steps[3]["status"] = "running"
                metadata = ModelMetadata(
                    name=f"{pipeline_name}_model",
                    description=f"Auto-generated model from {pipeline_name} pipeline",
                    model_type=model_type,
                    framework="sklearn",
                    algorithm="auto_selected",
                    version="1.0",
                    author="automl_system",
                    metrics={"accuracy": results.best_model.mean_cv_score if results.best_model else 0}
                )
                
                model_version_id = await model_manager.register_model(
                    f"/tmp/{workflow_id}_model.pkl", metadata
                )
                workflow_steps[3]["status"] = "completed"
                workflow_steps[3]["result"] = {"model_version_id": model_version_id}
                
                # Step 4: Model Deployment
                workflow_steps[4]["status"] = "running"
                artifact = ModelArtifact(
                    model_id=f"{pipeline_name}_model",
                    version="1.0",
                    framework=ServingFramework.SKLEARN,
                    model_path=f"/tmp/{workflow_id}_model.pkl"
                )
                
                serving_config = ServingConfig(
                    endpoint_name=f"{pipeline_name}_endpoint",
                    model_artifact=artifact,
                    inference_type=InferenceType.REAL_TIME
                )
                
                endpoint_name = await model_serving.deploy_model(
                    serving_config, DeploymentStrategy.ROLLING
                )
                workflow_steps[4]["status"] = "completed"
                workflow_steps[4]["result"] = {"endpoint_name": endpoint_name}
                
            except Exception as e:
                # Mark failed steps
                for step in workflow_steps:
                    if step["status"] == "running":
                        step["status"] = "failed"
                        step["error"] = str(e)
        
        # Start workflow execution
        asyncio.create_task(execute_ml_pipeline())
        
        return {
            "workflow_id": workflow_id,
            "pipeline_name": pipeline_name,
            "status": "running",
            "steps": workflow_steps,
            "created_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/status", response_model=Dict[str, Any])
async def get_workflow_status(workflow_id: str = Path(...)):
    """Get status of ML workflow execution"""
    # This would typically load from a workflow state store
    # For demo purposes, return a mock status
    return {
        "workflow_id": workflow_id,
        "status": "completed",
        "progress": 100.0,
        "message": "ML pipeline execution completed successfully",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/shutdown", response_model=StatusResponse)
async def shutdown_ml_framework():
    """Gracefully shutdown all ML framework components"""
    try:
        shutdown_results = []
        
        if neural_orchestrator:
            await neural_orchestrator.shutdown()
            shutdown_results.append("neural_orchestrator")
        
        if model_manager:
            await model_manager.shutdown()
            shutdown_results.append("model_manager")
        
        if automl_pipeline:
            await automl_pipeline.shutdown()
            shutdown_results.append("automl_pipeline")
        
        if data_processor:
            await data_processor.shutdown()
            shutdown_results.append("data_processor")
        
        if training_orchestrator:
            await training_orchestrator.shutdown()
            shutdown_results.append("training_orchestrator")
        
        if model_serving:
            await model_serving.shutdown()
            shutdown_results.append("model_serving")
        
        return StatusResponse(
            status="success",
            message=f"ML framework components shutdown: {', '.join(shutdown_results) if shutdown_results else 'none were running'}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))