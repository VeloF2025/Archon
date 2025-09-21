"""
Anomaly Detector Module
Machine learning-based anomaly detection for metrics and events
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .analytics_engine import AnalyticsEngine


class AnomalyType(Enum):
    """Types of anomalies"""
    POINT = "point"  # Single outlier point
    CONTEXTUAL = "contextual"  # Anomaly in specific context
    COLLECTIVE = "collective"  # Group of points forming anomaly
    SEASONAL = "seasonal"  # Deviation from seasonal pattern
    TREND = "trend"  # Deviation from expected trend
    LEVEL_SHIFT = "level_shift"  # Sudden change in level
    VARIANCE_CHANGE = "variance_change"  # Change in variance
    FREQUENCY = "frequency"  # Unusual frequency/rate
    CORRELATION = "correlation"  # Broken correlation pattern
    MULTIVARIATE = "multivariate"  # Anomaly across multiple metrics


class DetectionMethod(Enum):
    """Anomaly detection methods"""
    STATISTICAL = "statistical"  # Z-score, IQR, etc.
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    DBSCAN = "dbscan"
    AUTOENCODER = "autoencoder"
    LSTM = "lstm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class SeverityLevel(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    timestamp: datetime
    metric_name: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float  # 0 to 1
    value: float
    expected_value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None
    deviation: float = 0.0
    detection_method: DetectionMethod = DetectionMethod.STATISTICAL
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    related_anomalies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPattern:
    """Recurring anomaly pattern"""
    pattern_id: str
    pattern_name: str
    frequency: int  # Number of occurrences
    metrics_involved: Set[str]
    typical_duration: timedelta
    typical_magnitude: float
    conditions: Dict[str, Any]  # Conditions when pattern occurs
    examples: List[Anomaly]
    first_seen: datetime
    last_seen: datetime


@dataclass
class DetectionModel:
    """Anomaly detection model"""
    model_id: str
    model_type: DetectionMethod
    metric_name: str
    model_object: Any
    scaler: Optional[Any] = None
    trained_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """Anomaly detection report"""
    report_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    total_anomalies: int
    anomalies_by_type: Dict[AnomalyType, int]
    anomalies_by_severity: Dict[SeverityLevel, int]
    top_affected_metrics: List[Tuple[str, int]]
    patterns_detected: List[AnomalyPattern]
    false_positive_rate: float
    detection_accuracy: float
    recommendations: List[str]


class AnomalyDetector:
    """
    Advanced anomaly detection system using multiple ML techniques
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.anomalies: List[Anomaly] = []
        self.models: Dict[str, DetectionModel] = {}
        self.patterns: Dict[str, AnomalyPattern] = {}
        self.alerts_triggered: Set[str] = set()
        
        # Detection parameters
        self.sensitivity = 0.95  # Higher = more sensitive
        self.min_confidence = 0.7
        self.pattern_min_frequency = 3
        self.ensemble_threshold = 0.6  # Percentage of models agreeing
        
        # Model parameters
        self.isolation_forest_params = {
            'contamination': 0.1,
            'random_state': 42
        }
        self.lof_params = {
            'n_neighbors': 20,
            'contamination': 0.1
        }
        self.dbscan_params = {
            'eps': 0.5,
            'min_samples': 5
        }
        
        self._start_detection()
    
    def _start_detection(self):
        """Start anomaly detection background tasks"""
        asyncio.create_task(self._continuous_detection())
        asyncio.create_task(self._pattern_detection())
        asyncio.create_task(self._model_training())
    
    async def _continuous_detection(self):
        """Continuously detect anomalies"""
        while True:
            try:
                # Get all available metrics
                metrics = await self.analytics_engine.get_metric_names()
                
                for metric_name in metrics:
                    await self.detect_anomalies(metric_name)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"Anomaly detection error: {e}")
                await asyncio.sleep(300)
    
    async def _pattern_detection(self):
        """Detect recurring anomaly patterns"""
        while True:
            try:
                await self._identify_patterns()
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                print(f"Pattern detection error: {e}")
                await asyncio.sleep(3600)
    
    async def _model_training(self):
        """Train and update detection models"""
        while True:
            try:
                metrics = await self.analytics_engine.get_metric_names()
                
                for metric_name in metrics:
                    await self._train_model(metric_name)
                
                await asyncio.sleep(86400)  # Train daily
                
            except Exception as e:
                print(f"Model training error: {e}")
                await asyncio.sleep(86400)
    
    async def detect_anomalies(self, metric_name: str,
                             method: Optional[DetectionMethod] = None,
                             time_range: Optional[timedelta] = None) -> List[Anomaly]:
        """Detect anomalies in a metric"""
        import uuid
        
        # Get metric data
        if time_range is None:
            time_range = timedelta(hours=24)
        
        metrics = await self.analytics_engine.get_metrics(
            metric_names=[metric_name],
            time_range=time_range
        )
        
        if len(metrics) < 10:
            return []
        
        # Prepare data
        timestamps = [m["timestamp"] for m in metrics]
        values = np.array([m["value"] for m in metrics])
        
        # Use ensemble method if not specified
        if method is None:
            anomalies = await self._ensemble_detection(
                metric_name, timestamps, values
            )
        else:
            anomalies = await self._detect_with_method(
                metric_name, timestamps, values, method
            )
        
        # Filter by confidence
        anomalies = [a for a in anomalies if a.confidence >= self.min_confidence]
        
        # Store anomalies
        self.anomalies.extend(anomalies)
        
        # Trigger alerts for critical anomalies
        for anomaly in anomalies:
            if anomaly.severity == SeverityLevel.CRITICAL:
                await self._trigger_alert(anomaly)
        
        return anomalies
    
    async def _ensemble_detection(self, metric_name: str,
                                 timestamps: List[datetime],
                                 values: np.ndarray) -> List[Anomaly]:
        """Use ensemble of methods for detection"""
        all_anomalies = []
        methods_results = {}
        
        # Run multiple detection methods
        methods = [
            DetectionMethod.STATISTICAL,
            DetectionMethod.ISOLATION_FOREST,
            DetectionMethod.LOCAL_OUTLIER_FACTOR
        ]
        
        for method in methods:
            try:
                anomalies = await self._detect_with_method(
                    metric_name, timestamps, values, method
                )
                methods_results[method] = set(
                    i for i, a in enumerate(anomalies) 
                    for j, t in enumerate(timestamps) 
                    if a.timestamp == t
                )
            except Exception as e:
                print(f"Method {method.value} failed: {e}")
        
        # Find consensus anomalies
        n_methods = len(methods_results)
        min_agreement = int(n_methods * self.ensemble_threshold)
        
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            agreement_count = sum(
                1 for method_indices in methods_results.values()
                if i in method_indices
            )
            
            if agreement_count >= min_agreement:
                import uuid
                
                # Calculate severity based on deviation
                deviation = self._calculate_deviation(value, values)
                severity = self._determine_severity(deviation)
                
                anomaly = Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.POINT,
                    severity=severity,
                    confidence=agreement_count / n_methods,
                    value=value,
                    deviation=deviation,
                    detection_method=DetectionMethod.ENSEMBLE,
                    description=f"Detected by {agreement_count}/{n_methods} methods"
                )
                
                all_anomalies.append(anomaly)
        
        return all_anomalies
    
    async def _detect_with_method(self, metric_name: str,
                                 timestamps: List[datetime],
                                 values: np.ndarray,
                                 method: DetectionMethod) -> List[Anomaly]:
        """Detect anomalies using specific method"""
        anomalies = []
        
        if method == DetectionMethod.STATISTICAL:
            anomalies = self._statistical_detection(
                metric_name, timestamps, values
            )
        elif method == DetectionMethod.ISOLATION_FOREST:
            anomalies = self._isolation_forest_detection(
                metric_name, timestamps, values
            )
        elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
            anomalies = self._lof_detection(
                metric_name, timestamps, values
            )
        elif method == DetectionMethod.DBSCAN:
            anomalies = self._dbscan_detection(
                metric_name, timestamps, values
            )
        
        return anomalies
    
    def _statistical_detection(self, metric_name: str,
                              timestamps: List[datetime],
                              values: np.ndarray) -> List[Anomaly]:
        """Statistical anomaly detection (Z-score and IQR)"""
        import uuid
        anomalies = []
        
        # Z-score method
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
        
        # IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i, (timestamp, value) in enumerate(zip(timestamps, values)):
            is_anomaly = False
            confidence = 0.0
            
            # Check Z-score
            if z_scores[i] > 3:
                is_anomaly = True
                confidence = min(z_scores[i] / 4, 1.0)
            
            # Check IQR
            if value < lower_bound or value > upper_bound:
                is_anomaly = True
                iqr_confidence = min(
                    abs(value - lower_bound) / iqr if value < lower_bound else
                    abs(value - upper_bound) / iqr,
                    1.0
                )
                confidence = max(confidence, iqr_confidence)
            
            if is_anomaly:
                severity = self._determine_severity(z_scores[i])
                
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.POINT,
                    severity=severity,
                    confidence=confidence,
                    value=value,
                    expected_value=mean,
                    expected_range=(lower_bound, upper_bound),
                    deviation=z_scores[i],
                    detection_method=DetectionMethod.STATISTICAL,
                    description=f"Z-score: {z_scores[i]:.2f}"
                ))
        
        return anomalies
    
    def _isolation_forest_detection(self, metric_name: str,
                                   timestamps: List[datetime],
                                   values: np.ndarray) -> List[Anomaly]:
        """Isolation Forest anomaly detection"""
        import uuid
        anomalies = []
        
        # Reshape for sklearn
        X = values.reshape(-1, 1)
        
        # Train or get model
        model_key = f"{metric_name}_isolation"
        if model_key in self.models:
            model = self.models[model_key].model_object
        else:
            model = IsolationForest(**self.isolation_forest_params)
            model.fit(X)
        
        # Predict anomalies
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        for i, (timestamp, value, pred, score) in enumerate(
            zip(timestamps, values, predictions, scores)
        ):
            if pred == -1:  # Anomaly
                confidence = min(abs(score), 1.0)
                severity = self._determine_severity(abs(score))
                
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.POINT,
                    severity=severity,
                    confidence=confidence,
                    value=value,
                    deviation=abs(score),
                    detection_method=DetectionMethod.ISOLATION_FOREST,
                    description=f"Isolation score: {score:.3f}"
                ))
        
        return anomalies
    
    def _lof_detection(self, metric_name: str,
                      timestamps: List[datetime],
                      values: np.ndarray) -> List[Anomaly]:
        """Local Outlier Factor anomaly detection"""
        import uuid
        anomalies = []
        
        if len(values) < 20:
            return anomalies
        
        # Reshape for sklearn
        X = values.reshape(-1, 1)
        
        # Train LOF
        lof = LocalOutlierFactor(**self.lof_params)
        predictions = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        
        for i, (timestamp, value, pred, score) in enumerate(
            zip(timestamps, values, predictions, scores)
        ):
            if pred == -1:  # Anomaly
                confidence = min(abs(score - (-1)), 1.0)
                severity = self._determine_severity(abs(score))
                
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.CONTEXTUAL,
                    severity=severity,
                    confidence=confidence,
                    value=value,
                    deviation=abs(score),
                    detection_method=DetectionMethod.LOCAL_OUTLIER_FACTOR,
                    description=f"LOF score: {score:.3f}"
                ))
        
        return anomalies
    
    def _dbscan_detection(self, metric_name: str,
                         timestamps: List[datetime],
                         values: np.ndarray) -> List[Anomaly]:
        """DBSCAN clustering for anomaly detection"""
        import uuid
        anomalies = []
        
        # Prepare features (value and time index)
        X = np.column_stack([
            values,
            np.arange(len(values))
        ])
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        dbscan = DBSCAN(**self.dbscan_params)
        labels = dbscan.fit_predict(X_scaled)
        
        # Points with label -1 are outliers
        for i, (timestamp, value, label) in enumerate(
            zip(timestamps, values, labels)
        ):
            if label == -1:
                # Calculate distance to nearest cluster
                distances = []
                for cluster_label in set(labels):
                    if cluster_label != -1:
                        cluster_points = X_scaled[labels == cluster_label]
                        if len(cluster_points) > 0:
                            dist = np.min(np.linalg.norm(
                                cluster_points - X_scaled[i], axis=1
                            ))
                            distances.append(dist)
                
                confidence = min(np.mean(distances) / 2, 1.0) if distances else 0.8
                severity = self._determine_severity(confidence * 3)
                
                anomalies.append(Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=timestamp,
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.COLLECTIVE,
                    severity=severity,
                    confidence=confidence,
                    value=value,
                    detection_method=DetectionMethod.DBSCAN,
                    description="Outlier cluster"
                ))
        
        return anomalies
    
    async def detect_multivariate_anomalies(self,
                                          metric_names: List[str],
                                          time_range: Optional[timedelta] = None) -> List[Anomaly]:
        """Detect anomalies across multiple metrics"""
        import uuid
        
        if time_range is None:
            time_range = timedelta(hours=24)
        
        # Get data for all metrics
        all_data = []
        timestamps = None
        
        for metric_name in metric_names:
            metrics = await self.analytics_engine.get_metrics(
                metric_names=[metric_name],
                time_range=time_range
            )
            
            if not timestamps:
                timestamps = [m["timestamp"] for m in metrics]
            
            values = [m["value"] for m in metrics]
            all_data.append(values)
        
        if not all_data or not timestamps:
            return []
        
        # Create feature matrix
        X = np.column_stack(all_data)
        
        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use Isolation Forest for multivariate detection
        model = IsolationForest(contamination=0.1, random_state=42)
        predictions = model.fit_predict(X_scaled)
        scores = model.score_samples(X_scaled)
        
        anomalies = []
        for i, (timestamp, pred, score) in enumerate(
            zip(timestamps, predictions, scores)
        ):
            if pred == -1:
                affected_metrics = []
                for j, metric_name in enumerate(metric_names):
                    if abs(X_scaled[i, j]) > 2:  # Significant deviation
                        affected_metrics.append(metric_name)
                
                if affected_metrics:
                    anomalies.append(Anomaly(
                        anomaly_id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        metric_name=",".join(affected_metrics),
                        anomaly_type=AnomalyType.MULTIVARIATE,
                        severity=self._determine_severity(abs(score)),
                        confidence=min(abs(score), 1.0),
                        value=score,
                        detection_method=DetectionMethod.ISOLATION_FOREST,
                        description=f"Multivariate anomaly affecting {len(affected_metrics)} metrics",
                        context={"affected_metrics": affected_metrics}
                    ))
        
        return anomalies
    
    async def _train_model(self, metric_name: str):
        """Train anomaly detection model for a metric"""
        import uuid
        
        # Get training data
        metrics = await self.analytics_engine.get_metrics(
            metric_names=[metric_name],
            time_range=timedelta(days=30)
        )
        
        if len(metrics) < 100:
            return
        
        values = np.array([m["value"] for m in metrics])
        X = values.reshape(-1, 1)
        
        # Train Isolation Forest
        model = IsolationForest(**self.isolation_forest_params)
        model.fit(X)
        
        # Store model
        self.models[f"{metric_name}_isolation"] = DetectionModel(
            model_id=str(uuid.uuid4()),
            model_type=DetectionMethod.ISOLATION_FOREST,
            metric_name=metric_name,
            model_object=model,
            parameters=self.isolation_forest_params
        )
    
    async def _identify_patterns(self):
        """Identify recurring anomaly patterns"""
        import uuid
        
        # Group anomalies by metric and type
        grouped = {}
        for anomaly in self.anomalies:
            key = (anomaly.metric_name, anomaly.anomaly_type)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(anomaly)
        
        # Find patterns
        for (metric_name, anomaly_type), anomaly_group in grouped.items():
            if len(anomaly_group) >= self.pattern_min_frequency:
                # Analyze pattern characteristics
                timestamps = [a.timestamp for a in anomaly_group]
                values = [a.value for a in anomaly_group]
                
                # Calculate typical duration between occurrences
                if len(timestamps) > 1:
                    deltas = [timestamps[i+1] - timestamps[i] 
                             for i in range(len(timestamps)-1)]
                    typical_duration = timedelta(
                        seconds=np.median([d.total_seconds() for d in deltas])
                    )
                else:
                    typical_duration = timedelta(hours=1)
                
                pattern_id = str(uuid.uuid4())
                
                pattern = AnomalyPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"{metric_name}_{anomaly_type.value}_pattern",
                    frequency=len(anomaly_group),
                    metrics_involved={metric_name},
                    typical_duration=typical_duration,
                    typical_magnitude=np.mean(values),
                    conditions={
                        "anomaly_type": anomaly_type.value,
                        "metric": metric_name
                    },
                    examples=anomaly_group[:5],  # Keep first 5 examples
                    first_seen=min(timestamps),
                    last_seen=max(timestamps)
                )
                
                self.patterns[pattern_id] = pattern
    
    def _calculate_deviation(self, value: float, values: np.ndarray) -> float:
        """Calculate deviation of value from normal"""
        mean = np.mean(values)
        std = np.std(values)
        
        if std > 0:
            return abs((value - mean) / std)
        else:
            return abs(value - mean)
    
    def _determine_severity(self, deviation: float) -> SeverityLevel:
        """Determine anomaly severity based on deviation"""
        if deviation < 2:
            return SeverityLevel.LOW
        elif deviation < 3:
            return SeverityLevel.MEDIUM
        elif deviation < 4:
            return SeverityLevel.HIGH
        else:
            return SeverityLevel.CRITICAL
    
    async def _trigger_alert(self, anomaly: Anomaly):
        """Trigger alert for critical anomaly"""
        alert_key = f"{anomaly.metric_name}_{anomaly.timestamp.isoformat()}"
        
        if alert_key not in self.alerts_triggered:
            self.alerts_triggered.add(alert_key)
            
            await self.analytics_engine.create_alert(
                name=f"anomaly_{anomaly.anomaly_id}",
                condition=f"{anomaly.metric_name} anomaly detected",
                message=f"Critical {anomaly.anomaly_type.value} anomaly: {anomaly.description}",
                severity="critical"
            )
    
    async def generate_report(self, start_date: datetime,
                            end_date: datetime) -> AnomalyReport:
        """Generate anomaly detection report"""
        import uuid
        
        # Filter anomalies for period
        period_anomalies = [
            a for a in self.anomalies
            if start_date <= a.timestamp <= end_date
        ]
        
        # Calculate statistics
        anomalies_by_type = {}
        anomalies_by_severity = {}
        metric_counts = {}
        
        for anomaly in period_anomalies:
            # By type
            if anomaly.anomaly_type not in anomalies_by_type:
                anomalies_by_type[anomaly.anomaly_type] = 0
            anomalies_by_type[anomaly.anomaly_type] += 1
            
            # By severity
            if anomaly.severity not in anomalies_by_severity:
                anomalies_by_severity[anomaly.severity] = 0
            anomalies_by_severity[anomaly.severity] += 1
            
            # By metric
            if anomaly.metric_name not in metric_counts:
                metric_counts[anomaly.metric_name] = 0
            metric_counts[anomaly.metric_name] += 1
        
        # Top affected metrics
        top_metrics = sorted(metric_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        
        # Find patterns in period
        period_patterns = [
            p for p in self.patterns.values()
            if start_date <= p.last_seen <= end_date
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            period_anomalies, period_patterns
        )
        
        return AnomalyReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            total_anomalies=len(period_anomalies),
            anomalies_by_type=anomalies_by_type,
            anomalies_by_severity=anomalies_by_severity,
            top_affected_metrics=top_metrics,
            patterns_detected=period_patterns,
            false_positive_rate=0.1,  # Would calculate from feedback
            detection_accuracy=0.9,  # Would calculate from validation
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, anomalies: List[Anomaly],
                                patterns: List[AnomalyPattern]) -> List[str]:
        """Generate recommendations based on anomalies"""
        recommendations = []
        
        # Check for critical anomalies
        critical = [a for a in anomalies if a.severity == SeverityLevel.CRITICAL]
        if critical:
            recommendations.append(
                f"Investigate {len(critical)} critical anomalies immediately"
            )
        
        # Check for recurring patterns
        if patterns:
            recommendations.append(
                f"Address {len(patterns)} recurring anomaly patterns"
            )
        
        # Check for multivariate anomalies
        multivariate = [a for a in anomalies 
                       if a.anomaly_type == AnomalyType.MULTIVARIATE]
        if multivariate:
            recommendations.append(
                "Review system-wide issues causing multivariate anomalies"
            )
        
        # Sensitivity adjustment
        if len(anomalies) > 100:
            recommendations.append(
                "Consider reducing detection sensitivity to reduce false positives"
            )
        
        return recommendations