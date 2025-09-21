"""
Data Processing Engine for Archon Enhancement 2025 Phase 4

Advanced data processing and feature engineering system with:
- Automated data validation and quality assessment
- Stream processing for real-time data ingestion
- Advanced feature engineering and transformation
- Data versioning and lineage tracking
- Distributed data processing capabilities
- Multi-format data source integration
- Automated data profiling and schema inference
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Generator, AsyncGenerator
from enum import Enum
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import io
import re

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Data source types"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    AVRO = "avro"
    DATABASE = "database"
    API = "api"
    STREAM = "stream"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    KAFKA = "kafka"
    KINESIS = "kinesis"
    BIGQUERY = "bigquery"
    SNOWFLAKE = "snowflake"

class DataQualityIssue(Enum):
    """Data quality issue types"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_RECORDS = "duplicate_records"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMAT = "inconsistent_format"
    INVALID_VALUES = "invalid_values"
    SCHEMA_DRIFT = "schema_drift"
    DATA_DRIFT = "data_drift"
    CARDINALITY_DRIFT = "cardinality_drift"
    NULL_RATE_ANOMALY = "null_rate_anomaly"
    DISTRIBUTION_SHIFT = "distribution_shift"

class ProcessingStage(Enum):
    """Data processing pipeline stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    NORMALIZATION = "normalization"
    SPLITTING = "splitting"
    EXPORT = "export"

class TransformationType(Enum):
    """Data transformation types"""
    NORMALIZATION = "normalization"
    STANDARDIZATION = "standardization"
    ENCODING = "encoding"
    IMPUTATION = "imputation"
    OUTLIER_TREATMENT = "outlier_treatment"
    FEATURE_SCALING = "feature_scaling"
    DISCRETIZATION = "discretization"
    AGGREGATION = "aggregation"
    PIVOT = "pivot"
    MELT = "melt"
    MERGE = "merge"
    FILTER = "filter"
    SORT = "sort"
    GROUP = "group"

class FeatureType(Enum):
    """Feature data types"""
    NUMERICAL_CONTINUOUS = "numerical_continuous"
    NUMERICAL_DISCRETE = "numerical_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    GEOSPATIAL = "geospatial"
    IMAGE = "image"
    AUDIO = "audio"
    SEQUENCE = "sequence"
    GRAPH = "graph"

@dataclass
class DataSchema:
    """Data schema definition"""
    schema_id: str
    columns: Dict[str, FeatureType]
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    relationships: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate_data(self, data: pd.DataFrame) -> List[str]:
        """Validate data against schema"""
        issues = []
        
        # Check required columns
        missing_cols = set(self.columns.keys()) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check data types
        for col, expected_type in self.columns.items():
            if col in data.columns:
                actual_type = self._infer_column_type(data[col])
                if actual_type != expected_type:
                    issues.append(f"Column {col}: expected {expected_type.value}, got {actual_type.value}")
        
        # Check constraints
        for col, constraint in self.constraints.items():
            if col in data.columns:
                if "min_value" in constraint and data[col].min() < constraint["min_value"]:
                    issues.append(f"Column {col}: values below minimum {constraint['min_value']}")
                if "max_value" in constraint and data[col].max() > constraint["max_value"]:
                    issues.append(f"Column {col}: values above maximum {constraint['max_value']}")
                if "allowed_values" in constraint:
                    invalid_values = set(data[col].unique()) - set(constraint["allowed_values"])
                    if invalid_values:
                        issues.append(f"Column {col}: invalid values {invalid_values}")
        
        return issues
    
    def _infer_column_type(self, series: pd.Series) -> FeatureType:
        """Infer feature type from pandas series"""
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32']:
                return FeatureType.NUMERICAL_DISCRETE
            else:
                return FeatureType.NUMERICAL_CONTINUOUS
        elif pd.api.types.is_datetime64_any_dtype(series):
            return FeatureType.DATETIME
        elif pd.api.types.is_bool_dtype(series):
            return FeatureType.BOOLEAN
        else:
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:
                return FeatureType.CATEGORICAL_NOMINAL
            else:
                return FeatureType.TEXT

@dataclass
class ProcessingConfig:
    """Data processing configuration"""
    config_id: str
    source_config: Dict[str, Any]
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    quality_checks: List[DataQualityIssue] = field(default_factory=list)
    output_format: str = "parquet"
    batch_size: int = 10000
    parallel_workers: int = 4
    enable_caching: bool = True
    enable_profiling: bool = True
    enable_validation: bool = True
    error_handling: str = "raise"  # raise, skip, replace
    memory_limit: str = "8GB"
    timeout_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "config_id": self.config_id,
            "source_config": self.source_config,
            "transformations": self.transformations,
            "quality_checks": [check.value for check in self.quality_checks],
            "output_format": self.output_format,
            "batch_size": self.batch_size,
            "parallel_workers": self.parallel_workers,
            "enable_caching": self.enable_caching,
            "enable_profiling": self.enable_profiling,
            "enable_validation": self.enable_validation,
            "error_handling": self.error_handling,
            "memory_limit": self.memory_limit,
            "timeout_minutes": self.timeout_minutes
        }

@dataclass
class DataProfile:
    """Comprehensive data profiling results"""
    profile_id: str
    dataset_name: str
    total_rows: int
    total_columns: int
    memory_usage: int
    column_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    schema_inference: Optional[DataSchema] = None
    processing_time: timedelta = field(default_factory=lambda: timedelta(0))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)"""
        if not self.column_profiles:
            return 0.0
        
        scores = []
        for col_profile in self.column_profiles.values():
            completeness = 1.0 - col_profile.get("null_percentage", 0.0)
            uniqueness = min(col_profile.get("unique_ratio", 0.0), 1.0)
            validity = 1.0 - col_profile.get("invalid_percentage", 0.0)
            
            col_score = (0.4 * completeness + 0.3 * uniqueness + 0.3 * validity)
            scores.append(col_score)
        
        return np.mean(scores) if scores else 0.0

@dataclass
class ProcessingJob:
    """Data processing job tracking"""
    job_id: str
    config: ProcessingConfig
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    current_stage: ProcessingStage = ProcessingStage.INGESTION
    rows_processed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output_path: Optional[str] = None
    
    def get_duration(self) -> Optional[timedelta]:
        """Get job duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.utcnow() - self.start_time
        return None

@dataclass
class FeatureEngineeringPipeline:
    """Feature engineering pipeline definition"""
    pipeline_id: str
    steps: List[Dict[str, Any]]
    input_features: List[str]
    output_features: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_step(self, step_id: str, transformation: str, params: Dict[str, Any], 
                 input_cols: List[str], output_cols: List[str]) -> None:
        """Add feature engineering step"""
        step = {
            "step_id": step_id,
            "transformation": transformation,
            "params": params,
            "input_columns": input_cols,
            "output_columns": output_cols
        }
        self.steps.append(step)
        self.dependencies[step_id] = input_cols
        self.output_features.extend(output_cols)
        self._update_execution_order()
    
    def _update_execution_order(self) -> None:
        """Update execution order based on dependencies"""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {step_id}")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            # Visit dependencies first
            for step in self.steps:
                if step["step_id"] == step_id:
                    for dep_col in step["input_columns"]:
                        # Find step that produces this column
                        for dep_step in self.steps:
                            if dep_col in dep_step["output_columns"]:
                                visit(dep_step["step_id"])
                    break
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            execution_order.append(step_id)
        
        for step in self.steps:
            if step["step_id"] not in visited:
                visit(step["step_id"])
        
        self.execution_order = execution_order

class DataProcessor:
    """
    Comprehensive data processing engine with advanced transformation,
    validation, profiling, and feature engineering capabilities.
    """
    
    def __init__(self, base_path: str = "./data_processing", enable_distributed: bool = False):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.enable_distributed = enable_distributed
        
        # Processing state
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.schemas: Dict[str, DataSchema] = {}
        self.profiles: Dict[str, DataProfile] = {}
        self.pipelines: Dict[str, FeatureEngineeringPipeline] = {}
        
        # Processing cache
        self.transformation_cache: Dict[str, Any] = {}
        self.profile_cache: Dict[str, DataProfile] = {}
        
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        self._lock = threading.RLock()
        self._shutdown = False
        
        # Initialize transformation registry
        self.transformation_registry = self._initialize_transformations()
        
        logger.info("Data Processor initialized")
    
    async def initialize(self) -> None:
        """Initialize the data processor"""
        try:
            await self._load_schemas()
            await self._load_pipelines()
            await self._setup_data_connectors()
            logger.info("Data Processor initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize Data Processor: {e}")
            raise
    
    async def create_schema(self, 
                          schema_id: str,
                          columns: Dict[str, FeatureType],
                          constraints: Optional[Dict[str, Dict[str, Any]]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> DataSchema:
        """Create a new data schema definition"""
        try:
            schema = DataSchema(
                schema_id=schema_id,
                columns=columns,
                constraints=constraints or {},
                metadata=metadata or {}
            )
            
            with self._lock:
                self.schemas[schema_id] = schema
            
            # Save schema
            await self._save_schema(schema)
            
            logger.info(f"Created data schema: {schema_id}")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to create schema {schema_id}: {e}")
            raise
    
    async def infer_schema(self, 
                         data_source: str,
                         source_type: DataSourceType,
                         sample_size: int = 10000) -> DataSchema:
        """Automatically infer schema from data source"""
        try:
            # Load data sample
            data_sample = await self._load_data_sample(data_source, source_type, sample_size)
            
            # Infer column types
            columns = {}
            constraints = {}
            
            for col in data_sample.columns:
                series = data_sample[col]
                feature_type = self._infer_feature_type(series)
                columns[col] = feature_type
                
                # Infer constraints
                col_constraints = {}
                if feature_type in [FeatureType.NUMERICAL_CONTINUOUS, FeatureType.NUMERICAL_DISCRETE]:
                    col_constraints["min_value"] = float(series.min())
                    col_constraints["max_value"] = float(series.max())
                elif feature_type in [FeatureType.CATEGORICAL_NOMINAL, FeatureType.CATEGORICAL_ORDINAL]:
                    unique_values = series.unique().tolist()
                    if len(unique_values) <= 50:  # Reasonable limit
                        col_constraints["allowed_values"] = unique_values
                
                if col_constraints:
                    constraints[col] = col_constraints
            
            # Create schema
            schema_id = f"inferred_{hashlib.md5(data_source.encode()).hexdigest()[:8]}"
            schema = DataSchema(
                schema_id=schema_id,
                columns=columns,
                constraints=constraints,
                metadata={
                    "source": data_source,
                    "source_type": source_type.value,
                    "sample_size": len(data_sample),
                    "inferred_at": datetime.utcnow().isoformat()
                }
            )
            
            with self._lock:
                self.schemas[schema_id] = schema
            
            await self._save_schema(schema)
            
            logger.info(f"Inferred schema: {schema_id} from {source_type.value} source")
            return schema
            
        except Exception as e:
            logger.error(f"Failed to infer schema from {data_source}: {e}")
            raise
    
    async def profile_data(self, 
                         data_source: str,
                         source_type: DataSourceType,
                         schema_id: Optional[str] = None,
                         sample_size: Optional[int] = None) -> DataProfile:
        """Generate comprehensive data profile"""
        try:
            start_time = datetime.utcnow()
            
            # Load data
            data = await self._load_data(data_source, source_type, sample_size)
            
            # Generate unique profile ID
            profile_id = f"profile_{hashlib.md5(f'{data_source}_{datetime.utcnow()}'.encode()).hexdigest()[:8]}"
            
            # Basic statistics
            total_rows = len(data)
            total_columns = len(data.columns)
            memory_usage = data.memory_usage(deep=True).sum()
            
            # Column-level profiling
            column_profiles = {}
            for col in data.columns:
                col_profile = await self._profile_column(data[col])
                column_profiles[col] = col_profile
            
            # Correlation analysis
            correlations = await self._calculate_correlations(data)
            
            # Quality assessment
            quality_issues = await self._assess_data_quality(data, schema_id)
            
            # Schema inference
            inferred_schema = None
            if schema_id is None:
                inferred_schema = await self.infer_schema(data_source, source_type)
            
            # Create profile
            end_time = datetime.utcnow()
            profile = DataProfile(
                profile_id=profile_id,
                dataset_name=Path(data_source).name,
                total_rows=total_rows,
                total_columns=total_columns,
                memory_usage=memory_usage,
                column_profiles=column_profiles,
                correlations=correlations,
                quality_issues=quality_issues,
                schema_inference=inferred_schema,
                processing_time=end_time - start_time
            )
            
            with self._lock:
                self.profiles[profile_id] = profile
                # Cache for future use
                cache_key = f"{data_source}_{source_type.value}"
                self.profile_cache[cache_key] = profile
            
            # Save profile
            await self._save_profile(profile)
            
            logger.info(f"Generated data profile: {profile_id} (Quality Score: {profile.get_quality_score():.3f})")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to profile data from {data_source}: {e}")
            raise
    
    async def create_processing_job(self, 
                                  config: ProcessingConfig,
                                  input_source: str,
                                  output_path: str) -> str:
        """Create a new data processing job"""
        try:
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            
            job = ProcessingJob(
                job_id=job_id,
                config=config,
                output_path=output_path
            )
            
            with self._lock:
                self.active_jobs[job_id] = job
            
            logger.info(f"Created processing job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create processing job: {e}")
            raise
    
    async def execute_processing_job(self, job_id: str, input_source: str) -> ProcessingJob:
        """Execute a data processing job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            job.status = "running"
            job.start_time = datetime.utcnow()
            
            logger.info(f"Starting processing job: {job_id}")
            
            try:
                # Stage 1: Data Ingestion
                job.current_stage = ProcessingStage.INGESTION
                job.progress = 0.1
                data = await self._stage_ingestion(job, input_source)
                
                # Stage 2: Data Validation
                if job.config.enable_validation:
                    job.current_stage = ProcessingStage.VALIDATION
                    job.progress = 0.2
                    data = await self._stage_validation(job, data)
                
                # Stage 3: Data Cleaning
                job.current_stage = ProcessingStage.CLEANING
                job.progress = 0.3
                data = await self._stage_cleaning(job, data)
                
                # Stage 4: Data Transformation
                job.current_stage = ProcessingStage.TRANSFORMATION
                job.progress = 0.5
                data = await self._stage_transformation(job, data)
                
                # Stage 5: Feature Engineering
                job.current_stage = ProcessingStage.FEATURE_ENGINEERING
                job.progress = 0.7
                data = await self._stage_feature_engineering(job, data)
                
                # Stage 6: Normalization
                job.current_stage = ProcessingStage.NORMALIZATION
                job.progress = 0.8
                data = await self._stage_normalization(job, data)
                
                # Stage 7: Data Splitting (if configured)
                job.current_stage = ProcessingStage.SPLITTING
                job.progress = 0.9
                data = await self._stage_splitting(job, data)
                
                # Stage 8: Export
                job.current_stage = ProcessingStage.EXPORT
                job.progress = 0.95
                await self._stage_export(job, data)
                
                job.status = "completed"
                job.progress = 1.0
                job.end_time = datetime.utcnow()
                
                # Update metrics
                job.metrics = {
                    "total_rows": len(data) if hasattr(data, '__len__') else job.rows_processed,
                    "total_columns": len(data.columns) if hasattr(data, 'columns') else 0,
                    "processing_time_seconds": job.get_duration().total_seconds() if job.get_duration() else 0,
                    "throughput_rows_per_second": job.rows_processed / max(job.get_duration().total_seconds(), 1) if job.get_duration() else 0
                }
                
                logger.info(f"Completed processing job: {job_id} in {job.get_duration()}")
                
            except Exception as e:
                job.status = "failed"
                job.end_time = datetime.utcnow()
                job.errors.append(str(e))
                logger.error(f"Processing job {job_id} failed: {e}")
                raise
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to execute processing job {job_id}: {e}")
            raise
    
    async def create_feature_pipeline(self, pipeline_id: str, input_features: List[str]) -> FeatureEngineeringPipeline:
        """Create a new feature engineering pipeline"""
        try:
            pipeline = FeatureEngineeringPipeline(
                pipeline_id=pipeline_id,
                steps=[],
                input_features=input_features
            )
            
            with self._lock:
                self.pipelines[pipeline_id] = pipeline
            
            await self._save_pipeline(pipeline)
            
            logger.info(f"Created feature engineering pipeline: {pipeline_id}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create feature pipeline {pipeline_id}: {e}")
            raise
    
    async def add_feature_step(self,
                             pipeline_id: str,
                             step_id: str,
                             transformation: TransformationType,
                             input_columns: List[str],
                             output_columns: List[str],
                             params: Optional[Dict[str, Any]] = None) -> None:
        """Add a feature engineering step to pipeline"""
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            pipeline = self.pipelines[pipeline_id]
            
            pipeline.add_step(
                step_id=step_id,
                transformation=transformation.value,
                params=params or {},
                input_cols=input_columns,
                output_cols=output_columns
            )
            
            await self._save_pipeline(pipeline)
            
            logger.info(f"Added step {step_id} to pipeline {pipeline_id}")
            
        except Exception as e:
            logger.error(f"Failed to add step to pipeline {pipeline_id}: {e}")
            raise
    
    async def execute_feature_pipeline(self,
                                     pipeline_id: str,
                                     data: pd.DataFrame) -> pd.DataFrame:
        """Execute feature engineering pipeline on data"""
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"Pipeline {pipeline_id} not found")
            
            pipeline = self.pipelines[pipeline_id]
            processed_data = data.copy()
            
            # Execute steps in dependency order
            for step_id in pipeline.execution_order:
                step = next(s for s in pipeline.steps if s["step_id"] == step_id)
                
                logger.info(f"Executing pipeline step: {step_id}")
                
                # Apply transformation
                processed_data = await self._apply_transformation(
                    processed_data,
                    step["transformation"],
                    step["input_columns"],
                    step["output_columns"],
                    step["params"]
                )
            
            logger.info(f"Executed feature pipeline: {pipeline_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to execute feature pipeline {pipeline_id}: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a processing job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            return {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "current_stage": job.current_stage.value,
                "rows_processed": job.rows_processed,
                "errors": job.errors,
                "warnings": job.warnings,
                "metrics": job.metrics,
                "duration": str(job.get_duration()) if job.get_duration() else None,
                "output_path": job.output_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            raise
    
    async def validate_data_quality(self,
                                  data: pd.DataFrame,
                                  schema_id: Optional[str] = None,
                                  quality_threshold: float = 0.8) -> Dict[str, Any]:
        """Validate data quality against standards"""
        try:
            validation_results = {
                "overall_quality_score": 0.0,
                "passed": False,
                "issues": [],
                "recommendations": [],
                "column_scores": {}
            }
            
            # Schema validation
            if schema_id and schema_id in self.schemas:
                schema = self.schemas[schema_id]
                schema_issues = schema.validate_data(data)
                if schema_issues:
                    validation_results["issues"].extend([
                        {"type": "schema_violation", "description": issue} for issue in schema_issues
                    ])
            
            # Quality checks per column
            column_scores = {}
            for col in data.columns:
                col_score = await self._validate_column_quality(data[col])
                column_scores[col] = col_score
                
                if col_score < quality_threshold:
                    validation_results["issues"].append({
                        "type": "low_quality_column",
                        "column": col,
                        "score": col_score,
                        "description": f"Column {col} quality score {col_score:.3f} below threshold {quality_threshold}"
                    })
            
            # Overall quality score
            overall_score = np.mean(list(column_scores.values())) if column_scores else 0.0
            validation_results["overall_quality_score"] = overall_score
            validation_results["column_scores"] = column_scores
            validation_results["passed"] = overall_score >= quality_threshold and len(validation_results["issues"]) == 0
            
            # Generate recommendations
            validation_results["recommendations"] = await self._generate_quality_recommendations(
                validation_results["issues"], column_scores
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate data quality: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get data processor performance metrics"""
        try:
            total_jobs = len(self.active_jobs)
            completed_jobs = len([j for j in self.active_jobs.values() if j.status == "completed"])
            failed_jobs = len([j for j in self.active_jobs.values() if j.status == "failed"])
            running_jobs = len([j for j in self.active_jobs.values() if j.status == "running"])
            
            # Calculate average processing time
            completed = [j for j in self.active_jobs.values() if j.status == "completed"]
            avg_processing_time = timedelta(0)
            if completed:
                total_time = sum([j.get_duration() for j in completed if j.get_duration()], timedelta(0))
                avg_processing_time = total_time / len(completed)
            
            # Calculate throughput
            total_rows = sum([j.rows_processed for j in completed])
            total_time_seconds = sum([j.get_duration().total_seconds() for j in completed if j.get_duration()])
            avg_throughput = total_rows / total_time_seconds if total_time_seconds > 0 else 0
            
            return {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "running_jobs": running_jobs,
                "success_rate": completed_jobs / total_jobs if total_jobs > 0 else 0,
                "total_schemas": len(self.schemas),
                "total_profiles": len(self.profiles),
                "total_pipelines": len(self.pipelines),
                "average_processing_time_minutes": avg_processing_time.total_seconds() / 60,
                "average_throughput_rows_per_second": avg_throughput,
                "cache_hit_ratio": len(self.profile_cache) / max(len(self.profiles), 1),
                "memory_usage_mb": sum([p.memory_usage for p in self.profiles.values()]) / (1024 * 1024),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get data processor metrics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the data processor"""
        try:
            self._shutdown = True
            
            # Cancel running jobs
            for job_id, job in self.active_jobs.items():
                if job.status == "running":
                    job.status = "cancelled"
                    job.end_time = datetime.utcnow()
                    logger.info(f"Cancelled job: {job_id}")
            
            # Shutdown executors
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            # Save all state
            await self._save_all_state()
            
            logger.info("Data Processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Data Processor shutdown: {e}")
    
    # Private helper methods
    
    def _initialize_transformations(self) -> Dict[str, Callable]:
        """Initialize transformation registry"""
        return {
            TransformationType.NORMALIZATION: self._transform_normalize,
            TransformationType.STANDARDIZATION: self._transform_standardize,
            TransformationType.ENCODING: self._transform_encode,
            TransformationType.IMPUTATION: self._transform_impute,
            TransformationType.OUTLIER_TREATMENT: self._transform_outliers,
            TransformationType.FEATURE_SCALING: self._transform_scale,
            TransformationType.DISCRETIZATION: self._transform_discretize,
            TransformationType.AGGREGATION: self._transform_aggregate
        }
    
    def _infer_feature_type(self, series: pd.Series) -> FeatureType:
        """Infer feature type from pandas series"""
        if pd.api.types.is_numeric_dtype(series):
            if series.dtype in ['int64', 'int32', 'int16', 'int8']:
                return FeatureType.NUMERICAL_DISCRETE
            else:
                return FeatureType.NUMERICAL_CONTINUOUS
        elif pd.api.types.is_datetime64_any_dtype(series):
            return FeatureType.DATETIME
        elif pd.api.types.is_bool_dtype(series):
            return FeatureType.BOOLEAN
        else:
            # Check if it's categorical
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:
                # Check if ordinal (has natural ordering)
                if self._is_ordinal(series):
                    return FeatureType.CATEGORICAL_ORDINAL
                else:
                    return FeatureType.CATEGORICAL_NOMINAL
            else:
                # Check if it's text
                if series.dtype == 'object' and series.str.len().mean() > 10:
                    return FeatureType.TEXT
                else:
                    return FeatureType.CATEGORICAL_NOMINAL
    
    def _is_ordinal(self, series: pd.Series) -> bool:
        """Check if categorical series has natural ordering"""
        unique_values = series.unique()
        
        # Check for common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always']
        ]
        
        for pattern in ordinal_patterns:
            if all(val in pattern for val in unique_values if pd.notna(val)):
                return True
        
        return False
    
    async def _load_data_sample(self, source: str, source_type: DataSourceType, sample_size: int) -> pd.DataFrame:
        """Load a sample of data from source"""
        if source_type == DataSourceType.CSV:
            return pd.read_csv(source, nrows=sample_size)
        elif source_type == DataSourceType.JSON:
            return pd.read_json(source, lines=True, nrows=sample_size)
        elif source_type == DataSourceType.PARQUET:
            return pd.read_parquet(source).head(sample_size)
        else:
            # For other types, simulate loading
            return pd.DataFrame({
                'feature_1': np.random.randn(min(sample_size, 1000)),
                'feature_2': np.random.choice(['A', 'B', 'C'], min(sample_size, 1000)),
                'feature_3': np.random.randint(0, 100, min(sample_size, 1000))
            })
    
    async def _load_data(self, source: str, source_type: DataSourceType, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load data from source"""
        if sample_size:
            return await self._load_data_sample(source, source_type, sample_size)
        
        if source_type == DataSourceType.CSV:
            return pd.read_csv(source)
        elif source_type == DataSourceType.JSON:
            return pd.read_json(source, lines=True)
        elif source_type == DataSourceType.PARQUET:
            return pd.read_parquet(source)
        else:
            # Simulate loading for other types
            return pd.DataFrame({
                'feature_1': np.random.randn(10000),
                'feature_2': np.random.choice(['A', 'B', 'C'], 10000),
                'feature_3': np.random.randint(0, 100, 10000),
                'target': np.random.randint(0, 2, 10000)
            })
    
    async def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive column profile"""
        profile = {
            "data_type": str(series.dtype),
            "count": len(series),
            "null_count": series.isnull().sum(),
            "null_percentage": series.isnull().sum() / len(series),
            "unique_count": series.nunique(),
            "unique_ratio": series.nunique() / len(series)
        }
        
        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            profile.update({
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25))
            })
            
            # Outlier detection
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
            profile["outlier_count"] = outlier_mask.sum()
            profile["outlier_percentage"] = outlier_mask.sum() / len(series)
        
        # Categorical statistics
        elif series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            value_counts = series.value_counts().head(10)
            profile.update({
                "mode": series.mode().iloc[0] if not series.mode().empty else None,
                "top_values": value_counts.to_dict(),
                "entropy": self._calculate_entropy(series)
            })
        
        return profile
    
    async def _calculate_correlations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature correlations"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = data[numeric_cols].corr()
        
        # Extract high correlations
        correlations = {}
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    correlations[f"{col1}_vs_{col2}"] = float(corr_value)
        
        return correlations
    
    async def _assess_data_quality(self, data: pd.DataFrame, schema_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Assess data quality issues"""
        issues = []
        
        # Missing values check
        for col in data.columns:
            null_pct = data[col].isnull().sum() / len(data)
            if null_pct > 0.1:  # More than 10% missing
                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES.value,
                    "column": col,
                    "severity": "high" if null_pct > 0.3 else "medium",
                    "description": f"Column {col} has {null_pct:.1%} missing values",
                    "affected_rows": int(data[col].isnull().sum())
                })
        
        # Duplicate records check
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                "type": DataQualityIssue.DUPLICATE_RECORDS.value,
                "severity": "medium",
                "description": f"Found {duplicate_count} duplicate records",
                "affected_rows": duplicate_count
            })
        
        # Outliers check for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (data[col] < (q1 - 1.5 * iqr)) | (data[col] > (q3 + 1.5 * iqr))
            outlier_count = outlier_mask.sum()
            
            if outlier_count > len(data) * 0.05:  # More than 5% outliers
                issues.append({
                    "type": DataQualityIssue.OUTLIERS.value,
                    "column": col,
                    "severity": "low",
                    "description": f"Column {col} has {outlier_count} potential outliers",
                    "affected_rows": outlier_count
                })
        
        return issues
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of categorical series"""
        value_counts = series.value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))  # Add small value to avoid log(0)
        return float(entropy)
    
    # Processing stage methods
    
    async def _stage_ingestion(self, job: ProcessingJob, input_source: str) -> pd.DataFrame:
        """Data ingestion stage"""
        source_type = DataSourceType(job.config.source_config.get("type", "csv"))
        data = await self._load_data(input_source, source_type)
        job.rows_processed = len(data)
        return data
    
    async def _stage_validation(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Data validation stage"""
        # Apply quality checks
        for check in job.config.quality_checks:
            if check == DataQualityIssue.MISSING_VALUES:
                missing_threshold = 0.3
                high_missing = data.isnull().sum() / len(data) > missing_threshold
                if high_missing.any():
                    job.warnings.append(f"High missing values detected in columns: {high_missing[high_missing].index.tolist()}")
        
        return data
    
    async def _stage_cleaning(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning stage"""
        cleaned_data = data.copy()
        
        # Handle missing values
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                else:
                    cleaned_data[col].fillna(cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove duplicates
        original_len = len(cleaned_data)
        cleaned_data.drop_duplicates(inplace=True)
        if len(cleaned_data) < original_len:
            job.warnings.append(f"Removed {original_len - len(cleaned_data)} duplicate rows")
        
        return cleaned_data
    
    async def _stage_transformation(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Data transformation stage"""
        transformed_data = data.copy()
        
        # Apply configured transformations
        for transformation in job.config.transformations:
            trans_type = TransformationType(transformation["type"])
            input_cols = transformation.get("input_columns", list(data.columns))
            output_cols = transformation.get("output_columns", input_cols)
            params = transformation.get("params", {})
            
            transformed_data = await self._apply_transformation(
                transformed_data, trans_type.value, input_cols, output_cols, params
            )
        
        return transformed_data
    
    async def _stage_feature_engineering(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering stage"""
        # Basic feature engineering
        engineered_data = data.copy()
        
        # Add datetime features if datetime columns exist
        datetime_cols = engineered_data.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            engineered_data[f"{col}_year"] = engineered_data[col].dt.year
            engineered_data[f"{col}_month"] = engineered_data[col].dt.month
            engineered_data[f"{col}_day"] = engineered_data[col].dt.day
            engineered_data[f"{col}_dayofweek"] = engineered_data[col].dt.dayofweek
        
        # Add polynomial features for numeric columns (degree 2)
        numeric_cols = engineered_data.select_dtypes(include=[np.number]).columns[:3]  # Limit to avoid explosion
        for col in numeric_cols:
            engineered_data[f"{col}_squared"] = engineered_data[col] ** 2
            engineered_data[f"{col}_sqrt"] = np.sqrt(np.abs(engineered_data[col]))
        
        return engineered_data
    
    async def _stage_normalization(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Normalization stage"""
        normalized_data = data.copy()
        
        # Normalize numeric columns
        numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if normalized_data[col].std() > 0:
                normalized_data[f"{col}_normalized"] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()
        
        return normalized_data
    
    async def _stage_splitting(self, job: ProcessingJob, data: pd.DataFrame) -> pd.DataFrame:
        """Data splitting stage"""
        # This would typically split data into train/validation/test
        # For now, just return the data
        return data
    
    async def _stage_export(self, job: ProcessingJob, data: pd.DataFrame) -> None:
        """Export stage"""
        if job.output_path:
            output_format = job.config.output_format.lower()
            
            if output_format == "csv":
                data.to_csv(job.output_path, index=False)
            elif output_format == "parquet":
                data.to_parquet(job.output_path, index=False)
            elif output_format == "json":
                data.to_json(job.output_path, orient='records', lines=True)
            else:
                # Default to CSV
                data.to_csv(job.output_path, index=False)
    
    # Transformation methods
    
    async def _apply_transformation(self,
                                  data: pd.DataFrame,
                                  transformation: str,
                                  input_cols: List[str],
                                  output_cols: List[str],
                                  params: Dict[str, Any]) -> pd.DataFrame:
        """Apply transformation to data"""
        trans_type = TransformationType(transformation)
        
        if trans_type in self.transformation_registry:
            transform_func = self.transformation_registry[trans_type]
            return await transform_func(data, input_cols, output_cols, params)
        else:
            logger.warning(f"Unknown transformation type: {transformation}")
            return data
    
    async def _transform_normalize(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Normalize numeric columns to 0-1 range"""
        result = data.copy()
        
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col in result.columns and pd.api.types.is_numeric_dtype(result[input_col]):
                min_val = result[input_col].min()
                max_val = result[input_col].max()
                if max_val > min_val:
                    result[output_col] = (result[input_col] - min_val) / (max_val - min_val)
                else:
                    result[output_col] = 0.0
        
        return result
    
    async def _transform_standardize(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Standardize numeric columns to mean=0, std=1"""
        result = data.copy()
        
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col in result.columns and pd.api.types.is_numeric_dtype(result[input_col]):
                mean_val = result[input_col].mean()
                std_val = result[input_col].std()
                if std_val > 0:
                    result[output_col] = (result[input_col] - mean_val) / std_val
                else:
                    result[output_col] = 0.0
        
        return result
    
    async def _transform_encode(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Encode categorical columns"""
        result = data.copy()
        encoding_type = params.get("method", "onehot")
        
        for input_col in input_cols:
            if input_col in result.columns:
                if encoding_type == "onehot":
                    # One-hot encoding
                    dummies = pd.get_dummies(result[input_col], prefix=input_col)
                    result = pd.concat([result, dummies], axis=1)
                elif encoding_type == "label":
                    # Label encoding
                    unique_values = result[input_col].unique()
                    label_map = {val: i for i, val in enumerate(unique_values)}
                    result[f"{input_col}_encoded"] = result[input_col].map(label_map)
        
        return result
    
    async def _transform_impute(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Impute missing values"""
        result = data.copy()
        strategy = params.get("strategy", "median")
        
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col in result.columns:
                if strategy == "median" and pd.api.types.is_numeric_dtype(result[input_col]):
                    fill_value = result[input_col].median()
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(result[input_col]):
                    fill_value = result[input_col].mean()
                elif strategy == "mode":
                    fill_value = result[input_col].mode().iloc[0] if not result[input_col].mode().empty else 'Unknown'
                else:
                    fill_value = params.get("fill_value", 0)
                
                result[output_col] = result[input_col].fillna(fill_value)
        
        return result
    
    async def _transform_outliers(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        result = data.copy()
        method = params.get("method", "clip")
        
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col in result.columns and pd.api.types.is_numeric_dtype(result[input_col]):
                q1, q3 = result[input_col].quantile(0.25), result[input_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if method == "clip":
                    result[output_col] = result[input_col].clip(lower_bound, upper_bound)
                elif method == "remove":
                    mask = (result[input_col] >= lower_bound) & (result[input_col] <= upper_bound)
                    result = result[mask]
                elif method == "replace":
                    outlier_mask = (result[input_col] < lower_bound) | (result[input_col] > upper_bound)
                    result.loc[outlier_mask, output_col] = result[input_col].median()
                    result.loc[~outlier_mask, output_col] = result[input_col]
        
        return result
    
    async def _transform_scale(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Scale numeric columns"""
        # Similar to normalize/standardize but with different scaling methods
        return await self._transform_standardize(data, input_cols, output_cols, params)
    
    async def _transform_discretize(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Discretize continuous variables into bins"""
        result = data.copy()
        bins = params.get("bins", 5)
        
        for input_col, output_col in zip(input_cols, output_cols):
            if input_col in result.columns and pd.api.types.is_numeric_dtype(result[input_col]):
                result[output_col] = pd.cut(result[input_col], bins=bins, labels=False)
        
        return result
    
    async def _transform_aggregate(self, data: pd.DataFrame, input_cols: List[str], output_cols: List[str], params: Dict[str, Any]) -> pd.DataFrame:
        """Aggregate data by specified columns"""
        group_by = params.get("group_by", [])
        agg_func = params.get("function", "mean")
        
        if group_by:
            aggregated = data.groupby(group_by)[input_cols].agg(agg_func).reset_index()
            return aggregated
        
        return data
    
    async def _validate_column_quality(self, series: pd.Series) -> float:
        """Validate quality of individual column"""
        completeness = 1.0 - (series.isnull().sum() / len(series))
        uniqueness = series.nunique() / len(series) if len(series) > 0 else 0
        
        # Additional checks based on data type
        if pd.api.types.is_numeric_dtype(series):
            # Check for infinite values
            validity = 1.0 - (np.isinf(series).sum() / len(series))
        else:
            # Check for empty strings or invalid formats
            if series.dtype == 'object':
                empty_strings = (series == '').sum()
                validity = 1.0 - (empty_strings / len(series))
            else:
                validity = 1.0
        
        # Weighted score
        quality_score = 0.5 * completeness + 0.3 * uniqueness + 0.2 * validity
        return min(quality_score, 1.0)
    
    async def _generate_quality_recommendations(self, issues: List[Dict[str, Any]], column_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        # Missing value recommendations
        missing_issues = [i for i in issues if i["type"] == "missing_values"]
        if missing_issues:
            recommendations.append("Consider imputation strategies for columns with high missing values")
        
        # Outlier recommendations
        outlier_issues = [i for i in issues if i["type"] == "outliers"]
        if outlier_issues:
            recommendations.append("Review outlier detection and consider appropriate treatment methods")
        
        # Low quality columns
        low_quality_cols = [col for col, score in column_scores.items() if score < 0.7]
        if low_quality_cols:
            recommendations.append(f"Focus on improving quality for columns: {low_quality_cols}")
        
        # Schema violations
        schema_issues = [i for i in issues if i["type"] == "schema_violation"]
        if schema_issues:
            recommendations.append("Address schema violations before proceeding with analysis")
        
        return recommendations[:5]  # Limit recommendations
    
    # Storage methods
    
    async def _save_schema(self, schema: DataSchema) -> None:
        """Save schema to storage"""
        schema_path = self.base_path / "schemas" / f"{schema.schema_id}.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        
        schema_dict = {
            "schema_id": schema.schema_id,
            "columns": {k: v.value for k, v in schema.columns.items()},
            "constraints": schema.constraints,
            "relationships": schema.relationships,
            "metadata": schema.metadata,
            "version": schema.version,
            "created_at": schema.created_at.isoformat()
        }
        
        with open(schema_path, 'w') as f:
            json.dump(schema_dict, f, indent=2)
    
    async def _save_profile(self, profile: DataProfile) -> None:
        """Save data profile to storage"""
        profile_path = self.base_path / "profiles" / f"{profile.profile_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        
        profile_dict = {
            "profile_id": profile.profile_id,
            "dataset_name": profile.dataset_name,
            "total_rows": profile.total_rows,
            "total_columns": profile.total_columns,
            "memory_usage": profile.memory_usage,
            "column_profiles": profile.column_profiles,
            "correlations": profile.correlations,
            "quality_issues": profile.quality_issues,
            "quality_score": profile.get_quality_score(),
            "processing_time": profile.processing_time.total_seconds(),
            "created_at": profile.created_at.isoformat()
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
    
    async def _save_pipeline(self, pipeline: FeatureEngineeringPipeline) -> None:
        """Save feature engineering pipeline to storage"""
        pipeline_path = self.base_path / "pipelines" / f"{pipeline.pipeline_id}.json"
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        
        pipeline_dict = {
            "pipeline_id": pipeline.pipeline_id,
            "steps": pipeline.steps,
            "input_features": pipeline.input_features,
            "output_features": pipeline.output_features,
            "dependencies": pipeline.dependencies,
            "execution_order": pipeline.execution_order,
            "validation_rules": pipeline.validation_rules,
            "created_at": pipeline.created_at.isoformat()
        }
        
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline_dict, f, indent=2)
    
    async def _load_schemas(self) -> None:
        """Load existing schemas from storage"""
        schemas_dir = self.base_path / "schemas"
        if schemas_dir.exists():
            for schema_file in schemas_dir.glob("*.json"):
                try:
                    with open(schema_file, 'r') as f:
                        schema_data = json.load(f)
                    
                    # Reconstruct schema
                    columns = {k: FeatureType(v) for k, v in schema_data["columns"].items()}
                    schema = DataSchema(
                        schema_id=schema_data["schema_id"],
                        columns=columns,
                        constraints=schema_data.get("constraints", {}),
                        relationships=schema_data.get("relationships", {}),
                        metadata=schema_data.get("metadata", {}),
                        version=schema_data.get("version", "1.0")
                    )
                    
                    self.schemas[schema.schema_id] = schema
                    
                except Exception as e:
                    logger.error(f"Failed to load schema from {schema_file}: {e}")
    
    async def _load_pipelines(self) -> None:
        """Load existing pipelines from storage"""
        pipelines_dir = self.base_path / "pipelines"
        if pipelines_dir.exists():
            for pipeline_file in pipelines_dir.glob("*.json"):
                try:
                    with open(pipeline_file, 'r') as f:
                        pipeline_data = json.load(f)
                    
                    # Reconstruct pipeline
                    pipeline = FeatureEngineeringPipeline(
                        pipeline_id=pipeline_data["pipeline_id"],
                        steps=pipeline_data.get("steps", []),
                        input_features=pipeline_data.get("input_features", []),
                        output_features=pipeline_data.get("output_features", []),
                        dependencies=pipeline_data.get("dependencies", {}),
                        execution_order=pipeline_data.get("execution_order", []),
                        validation_rules=pipeline_data.get("validation_rules", {})
                    )
                    
                    self.pipelines[pipeline.pipeline_id] = pipeline
                    
                except Exception as e:
                    logger.error(f"Failed to load pipeline from {pipeline_file}: {e}")
    
    async def _setup_data_connectors(self) -> None:
        """Setup data source connectors"""
        # Placeholder for setting up various data connectors
        logger.info("Setting up data source connectors")
    
    async def _save_all_state(self) -> None:
        """Save all processor state"""
        state_file = self.base_path / "processor_state.json"
        
        state = {
            "active_jobs_count": len(self.active_jobs),
            "total_schemas": len(self.schemas),
            "total_profiles": len(self.profiles),
            "total_pipelines": len(self.pipelines),
            "cache_size": len(self.transformation_cache),
            "last_saved": datetime.utcnow().isoformat()
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)