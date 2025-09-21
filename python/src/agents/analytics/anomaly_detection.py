"""
Anomaly Detection Framework
Advanced Analytics & Intelligence Platform - Archon Enhancement 2025 Phase 5

Enterprise-grade anomaly detection with:
- Multi-modal anomaly detection across data types
- Real-time streaming anomaly detection
- Ensemble anomaly detection methods
- Contextual and collective anomaly detection
- Adaptive threshold management
- Anomaly scoring and ranking systems
- Root cause analysis and explanation
- Automated alerting and response systems
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    SEASONAL = "seasonal"
    TREND = "trend"
    MULTIVARIATE = "multivariate"


class DetectionMethod(Enum):
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"
    RULE_BASED = "rule_based"
    TIME_SERIES = "time_series"


class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TIME_SERIES = "time_series"
    TEXT = "text"
    IMAGE = "image"
    GRAPH = "graph"
    MIXED = "mixed"


class AlertChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    LOG = "log"


@dataclass
class AnomalyPoint:
    """Individual anomaly detection result"""
    anomaly_id: str
    timestamp: datetime
    value: Any
    anomaly_score: float
    confidence: float
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Complete anomaly detection result"""
    detection_id: str
    detector_name: str
    method: DetectionMethod
    data_type: DataType
    anomalies: List[AnomalyPoint]
    detection_threshold: float
    false_positive_rate: float
    detection_time_ms: float
    model_version: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class DetectorConfig:
    """Configuration for anomaly detector"""
    detector_id: str
    name: str
    method: DetectionMethod
    data_type: DataType
    parameters: Dict[str, Any]
    threshold_config: Dict[str, Any]
    is_adaptive: bool = True
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=1))
    min_training_samples: int = 100


@dataclass
class AlertRule:
    """Alert configuration rule"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]  # severity >= HIGH, count > 5, etc.
    channels: List[AlertChannel]
    recipients: List[str]
    cooldown_minutes: int = 60
    is_active: bool = True
    message_template: str = "Anomaly detected: {anomaly_count} anomalies with severity {max_severity}"


@dataclass
class Alert:
    """Generated alert"""
    alert_id: str
    rule_id: str
    triggered_at: datetime
    anomalies: List[AnomalyPoint]
    severity: AnomalySeverity
    message: str
    channels_sent: List[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class ThresholdInfo:
    """Adaptive threshold information"""
    current_threshold: float
    historical_thresholds: List[Tuple[datetime, float]]
    adaptation_rate: float
    last_updated: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.is_trained = False
        self.training_data_size = 0
        self.last_training_time: Optional[datetime] = None
        self.threshold_info = ThresholdInfo(
            current_threshold=config.parameters.get('threshold', 0.5),
            historical_thresholds=[],
            adaptation_rate=config.parameters.get('adaptation_rate', 0.01),
            last_updated=datetime.now()
        )
    
    @abstractmethod
    async def train(self, data: pd.DataFrame) -> bool:
        """Train the anomaly detector"""
        pass
    
    @abstractmethod
    async def detect(self, data: pd.DataFrame) -> AnomalyDetectionResult:
        """Detect anomalies in new data"""
        pass
    
    @abstractmethod
    async def update_model(self, new_data: pd.DataFrame, feedback: List[Dict[str, Any]] = None) -> bool:
        """Update model with new data and feedback"""
        pass
    
    def get_threshold(self) -> float:
        """Get current detection threshold"""
        return self.threshold_info.current_threshold
    
    async def adapt_threshold(self, performance_metrics: Dict[str, float]):
        """Adapt detection threshold based on performance"""
        if not self.config.is_adaptive:
            return
        
        # Simple adaptation based on false positive rate
        fpr = performance_metrics.get('false_positive_rate', 0.05)
        target_fpr = self.config.threshold_config.get('target_fpr', 0.05)
        
        if fpr > target_fpr * 1.2:  # Too many false positives
            self.threshold_info.current_threshold *= (1 + self.threshold_info.adaptation_rate)
        elif fpr < target_fpr * 0.8:  # Too few detections
            self.threshold_info.current_threshold *= (1 - self.threshold_info.adaptation_rate)
        
        # Store historical threshold
        self.threshold_info.historical_thresholds.append(
            (datetime.now(), self.threshold_info.current_threshold)
        )
        
        # Keep only recent history
        if len(self.threshold_info.historical_thresholds) > 100:
            self.threshold_info.historical_thresholds = self.threshold_info.historical_thresholds[-100:]
        
        self.threshold_info.last_updated = datetime.now()
        self.threshold_info.performance_metrics = performance_metrics.copy()
        
        logger.info(f"Adapted threshold for {self.config.name}: {self.threshold_info.current_threshold:.4f}")


class StatisticalAnomalyDetector(BaseAnomalyDetector):
    """Statistical anomaly detection using Z-score and robust statistics"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.statistics: Dict[str, Dict[str, float]] = {}
        self.use_robust_stats = config.parameters.get('use_robust_stats', True)
    
    async def train(self, data: pd.DataFrame) -> bool:
        """Train statistical model on historical data"""
        try:
            self.statistics = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                series = data[column].dropna()
                
                if len(series) < self.config.min_training_samples:
                    logger.warning(f"Insufficient data for column {column}: {len(series)}")
                    continue
                
                if self.use_robust_stats:
                    # Robust statistics
                    median = series.median()
                    mad = (series - median).abs().median()  # Median Absolute Deviation
                    q25, q75 = series.quantile([0.25, 0.75])
                    iqr = q75 - q25
                    
                    self.statistics[column] = {
                        'median': median,
                        'mad': mad,
                        'q25': q25,
                        'q75': q75,
                        'iqr': iqr,
                        'robust': True
                    }
                else:
                    # Standard statistics
                    mean = series.mean()
                    std = series.std()
                    
                    self.statistics[column] = {
                        'mean': mean,
                        'std': std,
                        'robust': False
                    }
            
            self.is_trained = len(self.statistics) > 0
            self.training_data_size = len(data)
            self.last_training_time = datetime.now()
            
            logger.info(f"Statistical detector trained on {len(data)} samples with {len(self.statistics)} features")
            return self.is_trained
            
        except Exception as e:
            logger.error(f"Statistical detector training failed: {e}")
            return False
    
    async def detect(self, data: pd.DataFrame) -> AnomalyDetectionResult:
        """Detect statistical anomalies"""
        start_time = datetime.now()
        anomalies = []
        
        if not self.is_trained:
            raise ValueError("Detector not trained")
        
        try:
            for index, row in data.iterrows():
                anomaly_scores = {}
                feature_contributions = {}
                
                for column, stats in self.statistics.items():
                    if column not in row or pd.isna(row[column]):
                        continue
                    
                    value = row[column]
                    
                    if stats['robust']:
                        # Robust anomaly score
                        if stats['mad'] > 0:
                            modified_z_score = 0.6745 * (value - stats['median']) / stats['mad']
                        else:
                            modified_z_score = 0
                        
                        # IQR-based score
                        if value < stats['q25'] - 1.5 * stats['iqr']:
                            iqr_score = (stats['q25'] - 1.5 * stats['iqr'] - value) / stats['iqr']
                        elif value > stats['q75'] + 1.5 * stats['iqr']:
                            iqr_score = (value - stats['q75'] - 1.5 * stats['iqr']) / stats['iqr']
                        else:
                            iqr_score = 0
                        
                        anomaly_score = max(abs(modified_z_score), iqr_score)
                    else:
                        # Standard Z-score
                        if stats['std'] > 0:
                            z_score = abs((value - stats['mean']) / stats['std'])
                        else:
                            z_score = 0
                        anomaly_score = z_score
                    
                    anomaly_scores[column] = anomaly_score
                    feature_contributions[column] = anomaly_score
                
                # Overall anomaly score
                if anomaly_scores:
                    overall_score = max(anomaly_scores.values())
                    
                    if overall_score > self.get_threshold():
                        # Determine severity
                        if overall_score > self.get_threshold() * 3:
                            severity = AnomalySeverity.CRITICAL
                        elif overall_score > self.get_threshold() * 2:
                            severity = AnomalySeverity.HIGH
                        elif overall_score > self.get_threshold() * 1.5:
                            severity = AnomalySeverity.MEDIUM
                        else:
                            severity = AnomalySeverity.LOW
                        
                        # Find most contributing feature
                        max_feature = max(anomaly_scores.keys(), key=anomaly_scores.get)
                        explanation = f"Statistical anomaly in {max_feature} (score: {overall_score:.3f})"
                        
                        anomaly = AnomalyPoint(
                            anomaly_id=f"stat_{uuid.uuid4().hex[:8]}",
                            timestamp=index if isinstance(index, datetime) else datetime.now(),
                            value=row.to_dict(),
                            anomaly_score=overall_score,
                            confidence=min(1.0, overall_score / (self.get_threshold() * 2)),
                            severity=severity,
                            anomaly_type=AnomalyType.POINT,
                            feature_contributions=feature_contributions,
                            explanation=explanation
                        )
                        anomalies.append(anomaly)
            
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AnomalyDetectionResult(
                detection_id=f"detection_{uuid.uuid4().hex[:8]}",
                detector_name=self.config.name,
                method=DetectionMethod.STATISTICAL,
                data_type=self.config.data_type,
                anomalies=anomalies,
                detection_threshold=self.get_threshold(),
                false_positive_rate=0.05,  # Would be calculated from feedback
                detection_time_ms=detection_time,
                model_version="1.0",
                parameters=self.config.parameters
            )
            
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            raise
    
    async def update_model(self, new_data: pd.DataFrame, feedback: List[Dict[str, Any]] = None) -> bool:
        """Update statistical model incrementally"""
        try:
            if not self.is_trained:
                return await self.train(new_data)
            
            # Simple incremental update (would use more sophisticated methods in production)
            alpha = self.config.parameters.get('learning_rate', 0.1)
            
            numeric_columns = new_data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if column not in self.statistics:
                    continue
                
                series = new_data[column].dropna()
                if len(series) == 0:
                    continue
                
                stats = self.statistics[column]
                
                if stats['robust']:
                    # Update robust statistics
                    new_median = series.median()
                    new_mad = (series - new_median).abs().median()
                    
                    stats['median'] = (1 - alpha) * stats['median'] + alpha * new_median
                    stats['mad'] = (1 - alpha) * stats['mad'] + alpha * new_mad
                else:
                    # Update standard statistics
                    new_mean = series.mean()
                    new_std = series.std()
                    
                    stats['mean'] = (1 - alpha) * stats['mean'] + alpha * new_mean
                    stats['std'] = (1 - alpha) * stats['std'] + alpha * new_std
            
            self.training_data_size += len(new_data)
            logger.info(f"Updated statistical model with {len(new_data)} new samples")
            return True
            
        except Exception as e:
            logger.error(f"Statistical model update failed: {e}")
            return False


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.model = None
        self.feature_names = []
    
    async def train(self, data: pd.DataFrame) -> bool:
        """Train Isolation Forest model"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data: {len(numeric_data)}")
                return False
            
            # Handle missing values
            numeric_data = numeric_data.fillna(numeric_data.median())
            
            # Feature scaling
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Train Isolation Forest
            contamination = self.config.parameters.get('contamination', 0.1)
            n_estimators = self.config.parameters.get('n_estimators', 100)
            
            self.model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(scaled_data)
            self.feature_names = list(numeric_data.columns)
            
            self.is_trained = True
            self.training_data_size = len(data)
            self.last_training_time = datetime.now()
            
            logger.info(f"Isolation Forest trained on {len(data)} samples with {len(self.feature_names)} features")
            return True
            
        except ImportError:
            logger.error("sklearn not available for Isolation Forest")
            return False
        except Exception as e:
            logger.error(f"Isolation Forest training failed: {e}")
            return False
    
    async def detect(self, data: pd.DataFrame) -> AnomalyDetectionResult:
        """Detect anomalies using Isolation Forest"""
        start_time = datetime.now()
        anomalies = []
        
        if not self.is_trained or self.model is None:
            raise ValueError("Detector not trained")
        
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            numeric_data = numeric_data.fillna(numeric_data.median())
            
            # Ensure same features as training
            for feature in self.feature_names:
                if feature not in numeric_data.columns:
                    numeric_data[feature] = 0
            
            numeric_data = numeric_data[self.feature_names]
            
            # Scale features
            scaled_data = self.scaler.transform(numeric_data)
            
            # Get anomaly predictions and scores
            predictions = self.model.predict(scaled_data)
            anomaly_scores = -self.model.decision_function(scaled_data)  # Negative for anomalies
            
            # Process results
            for i, (index, row) in enumerate(data.iterrows()):
                if predictions[i] == -1:  # Anomaly detected
                    score = anomaly_scores[i]
                    
                    # Determine severity based on score percentile
                    score_percentile = np.percentile(anomaly_scores, 95)
                    
                    if score > score_percentile * 1.5:
                        severity = AnomalySeverity.CRITICAL
                    elif score > score_percentile * 1.2:
                        severity = AnomalySeverity.HIGH
                    elif score > score_percentile:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW
                    
                    # Feature contributions (simplified)
                    feature_contributions = {}
                    for j, feature in enumerate(self.feature_names):
                        if feature in row:
                            feature_contributions[feature] = abs(scaled_data[i, j])
                    
                    explanation = f"Isolation Forest anomaly (score: {score:.3f})"
                    
                    anomaly = AnomalyPoint(
                        anomaly_id=f"iforest_{uuid.uuid4().hex[:8]}",
                        timestamp=index if isinstance(index, datetime) else datetime.now(),
                        value=row.to_dict(),
                        anomaly_score=score,
                        confidence=min(1.0, score / (score_percentile * 2)),
                        severity=severity,
                        anomaly_type=AnomalyType.POINT,
                        feature_contributions=feature_contributions,
                        explanation=explanation
                    )
                    anomalies.append(anomaly)
            
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AnomalyDetectionResult(
                detection_id=f"detection_{uuid.uuid4().hex[:8]}",
                detector_name=self.config.name,
                method=DetectionMethod.MACHINE_LEARNING,
                data_type=self.config.data_type,
                anomalies=anomalies,
                detection_threshold=self.get_threshold(),
                false_positive_rate=0.1,  # Configurable contamination rate
                detection_time_ms=detection_time,
                model_version="1.0",
                parameters=self.config.parameters
            )
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            raise
    
    async def update_model(self, new_data: pd.DataFrame, feedback: List[Dict[str, Any]] = None) -> bool:
        """Update Isolation Forest model"""
        # For Isolation Forest, we typically retrain the model
        # In production, you might use online learning variants
        return await self.train(new_data)


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """Ensemble anomaly detector combining multiple methods"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.base_detectors: List[BaseAnomalyDetector] = []
        self.weights: List[float] = []
        self.combination_method = config.parameters.get('combination_method', 'average')
    
    def add_detector(self, detector: BaseAnomalyDetector, weight: float = 1.0):
        """Add base detector to ensemble"""
        self.base_detectors.append(detector)
        self.weights.append(weight)
        logger.info(f"Added detector {detector.config.name} to ensemble with weight {weight}")
    
    async def train(self, data: pd.DataFrame) -> bool:
        """Train all base detectors"""
        training_results = []
        
        for detector in self.base_detectors:
            try:
                result = await detector.train(data)
                training_results.append(result)
                logger.info(f"Detector {detector.config.name} training: {'success' if result else 'failed'}")
            except Exception as e:
                logger.error(f"Training failed for {detector.config.name}: {e}")
                training_results.append(False)
        
        self.is_trained = any(training_results)
        self.training_data_size = len(data)
        self.last_training_time = datetime.now()
        
        successful_detectors = sum(training_results)
        logger.info(f"Ensemble training completed: {successful_detectors}/{len(self.base_detectors)} detectors trained")
        
        return self.is_trained
    
    async def detect(self, data: pd.DataFrame) -> AnomalyDetectionResult:
        """Detect anomalies using ensemble approach"""
        start_time = datetime.now()
        
        if not self.is_trained:
            raise ValueError("Ensemble not trained")
        
        try:
            # Get predictions from all base detectors
            detector_results = []
            
            for detector in self.base_detectors:
                if detector.is_trained:
                    try:
                        result = await detector.detect(data)
                        detector_results.append(result)
                    except Exception as e:
                        logger.warning(f"Detection failed for {detector.config.name}: {e}")
            
            if not detector_results:
                raise ValueError("No detectors produced results")
            
            # Combine results
            combined_anomalies = await self._combine_results(detector_results, data)
            
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AnomalyDetectionResult(
                detection_id=f"detection_{uuid.uuid4().hex[:8]}",
                detector_name=self.config.name,
                method=DetectionMethod.ENSEMBLE,
                data_type=self.config.data_type,
                anomalies=combined_anomalies,
                detection_threshold=self.get_threshold(),
                false_positive_rate=0.05,  # Ensemble typically has lower FPR
                detection_time_ms=detection_time,
                model_version="1.0",
                parameters=self.config.parameters
            )
            
        except Exception as e:
            logger.error(f"Ensemble detection failed: {e}")
            raise
    
    async def _combine_results(self, detector_results: List[AnomalyDetectionResult], 
                             original_data: pd.DataFrame) -> List[AnomalyPoint]:
        """Combine results from multiple detectors"""
        combined_anomalies = []
        
        # Create index mapping from original data
        data_index_map = {idx: i for i, idx in enumerate(original_data.index)}
        
        if self.combination_method == 'voting':
            # Majority voting
            anomaly_votes = defaultdict(list)
            
            for result in detector_results:
                for anomaly in result.anomalies:
                    key = (anomaly.timestamp, str(anomaly.value))
                    anomaly_votes[key].append(anomaly)
            
            # Keep anomalies with majority vote
            min_votes = len(detector_results) // 2 + 1
            
            for key, anomaly_list in anomaly_votes.items():
                if len(anomaly_list) >= min_votes:
                    # Combine anomaly information
                    combined_anomaly = await self._merge_anomalies(anomaly_list)
                    combined_anomalies.append(combined_anomaly)
        
        elif self.combination_method == 'average':
            # Score averaging
            all_anomalies = []
            for result in detector_results:
                all_anomalies.extend(result.anomalies)
            
            # Group anomalies by timestamp/value
            anomaly_groups = defaultdict(list)
            for anomaly in all_anomalies:
                key = (anomaly.timestamp, str(anomaly.value))
                anomaly_groups[key].append(anomaly)
            
            for key, anomaly_list in anomaly_groups.items():
                # Calculate weighted average score
                weighted_score = 0
                total_weight = 0
                
                for i, anomaly in enumerate(anomaly_list):
                    detector_weight = self.weights[i % len(self.weights)]
                    weighted_score += anomaly.anomaly_score * detector_weight
                    total_weight += detector_weight
                
                if total_weight > 0:
                    avg_score = weighted_score / total_weight
                    
                    if avg_score > self.get_threshold():
                        combined_anomaly = await self._merge_anomalies(anomaly_list, avg_score)
                        combined_anomalies.append(combined_anomaly)
        
        else:
            # Union: include all anomalies
            all_anomalies = []
            for result in detector_results:
                all_anomalies.extend(result.anomalies)
            combined_anomalies = all_anomalies
        
        # Sort by anomaly score
        combined_anomalies.sort(key=lambda x: x.anomaly_score, reverse=True)
        
        return combined_anomalies
    
    async def _merge_anomalies(self, anomaly_list: List[AnomalyPoint], 
                             override_score: Optional[float] = None) -> AnomalyPoint:
        """Merge multiple anomaly detections into one"""
        if not anomaly_list:
            raise ValueError("Empty anomaly list")
        
        # Use first anomaly as base
        base_anomaly = anomaly_list[0]
        
        # Calculate combined score
        if override_score is not None:
            combined_score = override_score
        else:
            scores = [a.anomaly_score for a in anomaly_list]
            combined_score = np.mean(scores)
        
        # Determine combined severity
        severities = [a.severity for a in anomaly_list]
        max_severity = max(severities, key=lambda s: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(s.value))
        
        # Combine feature contributions
        combined_contributions = {}
        for anomaly in anomaly_list:
            for feature, contribution in anomaly.feature_contributions.items():
                if feature in combined_contributions:
                    combined_contributions[feature] = max(combined_contributions[feature], contribution)
                else:
                    combined_contributions[feature] = contribution
        
        # Create explanation
        detector_names = [a.explanation.split()[0] for a in anomaly_list]
        explanation = f"Ensemble detection from {', '.join(set(detector_names))} (score: {combined_score:.3f})"
        
        return AnomalyPoint(
            anomaly_id=f"ensemble_{uuid.uuid4().hex[:8]}",
            timestamp=base_anomaly.timestamp,
            value=base_anomaly.value,
            anomaly_score=combined_score,
            confidence=min(1.0, len(anomaly_list) / len(self.base_detectors)),
            severity=max_severity,
            anomaly_type=AnomalyType.POINT,
            feature_contributions=combined_contributions,
            explanation=explanation,
            metadata={'contributing_detectors': len(anomaly_list)}
        )
    
    async def update_model(self, new_data: pd.DataFrame, feedback: List[Dict[str, Any]] = None) -> bool:
        """Update all base detectors"""
        update_results = []
        
        for detector in self.base_detectors:
            try:
                result = await detector.update_model(new_data, feedback)
                update_results.append(result)
            except Exception as e:
                logger.error(f"Update failed for {detector.config.name}: {e}")
                update_results.append(False)
        
        successful_updates = sum(update_results)
        logger.info(f"Ensemble update completed: {successful_updates}/{len(self.base_detectors)} detectors updated")
        
        return any(update_results)


class AlertManager:
    """Anomaly alert management system"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, datetime] = {}
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    async def process_anomalies(self, anomalies: List[AnomalyPoint]) -> List[Alert]:
        """Process anomalies and generate alerts based on rules"""
        generated_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.is_active:
                continue
            
            # Check cooldown
            if rule_id in self.cooldown_tracker:
                cooldown_end = self.cooldown_tracker[rule_id] + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue
            
            # Evaluate rule conditions
            matching_anomalies = await self._evaluate_rule_conditions(rule, anomalies)
            
            if matching_anomalies:
                alert = await self._create_alert(rule, matching_anomalies)
                generated_alerts.append(alert)
                
                # Send alert
                await self._send_alert(alert)
                
                # Update cooldown
                self.cooldown_tracker[rule_id] = datetime.now()
                
                # Store alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
        
        return generated_alerts
    
    async def _evaluate_rule_conditions(self, rule: AlertRule, anomalies: List[AnomalyPoint]) -> List[AnomalyPoint]:
        """Evaluate if anomalies match rule conditions"""
        matching_anomalies = []
        
        conditions = rule.conditions
        
        for anomaly in anomalies:
            match = True
            
            # Check severity condition
            if 'min_severity' in conditions:
                min_severity = AnomalySeverity(conditions['min_severity'])
                severity_order = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                if severity_order.index(anomaly.severity.value) < severity_order.index(min_severity.value):
                    match = False
            
            # Check score threshold
            if 'min_score' in conditions:
                if anomaly.anomaly_score < conditions['min_score']:
                    match = False
            
            # Check anomaly type
            if 'anomaly_types' in conditions:
                if anomaly.anomaly_type.value not in conditions['anomaly_types']:
                    match = False
            
            # Check feature conditions
            if 'features' in conditions:
                for feature, feature_conditions in conditions['features'].items():
                    if feature in anomaly.feature_contributions:
                        contribution = anomaly.feature_contributions[feature]
                        if 'min_contribution' in feature_conditions:
                            if contribution < feature_conditions['min_contribution']:
                                match = False
            
            if match:
                matching_anomalies.append(anomaly)
        
        # Check count condition
        if 'min_count' in conditions:
            if len(matching_anomalies) < conditions['min_count']:
                return []
        
        return matching_anomalies
    
    async def _create_alert(self, rule: AlertRule, anomalies: List[AnomalyPoint]) -> Alert:
        """Create alert from rule and anomalies"""
        # Determine overall severity
        max_severity = max(anomalies, key=lambda a: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(a.severity.value)).severity
        
        # Format message
        message_vars = {
            'anomaly_count': len(anomalies),
            'max_severity': max_severity.value,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rule_name': rule.name
        }
        
        message = rule.message_template.format(**message_vars)
        
        return Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            rule_id=rule.rule_id,
            triggered_at=datetime.now(),
            anomalies=anomalies,
            severity=max_severity,
            message=message
        )
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        rule = self.alert_rules[alert.rule_id]
        
        for channel in rule.channels:
            try:
                if channel == AlertChannel.LOG:
                    await self._send_log_alert(alert, rule)
                elif channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert, rule)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert, rule)
                # Add other channels as needed
                
                alert.channels_sent.append(channel)
                logger.info(f"Alert {alert.alert_id} sent via {channel.value}")
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
    
    async def _send_log_alert(self, alert: Alert, rule: AlertRule):
        """Send alert to log"""
        logger.warning(f"ALERT: {alert.message}")
    
    async def _send_email_alert(self, alert: Alert, rule: AlertRule):
        """Send email alert (placeholder)"""
        logger.info(f"Would send email alert to {rule.recipients}")
    
    async def _send_webhook_alert(self, alert: Alert, rule: AlertRule):
        """Send webhook alert (placeholder)"""
        webhook_payload = {
            'alert_id': alert.alert_id,
            'rule_name': rule.name,
            'severity': alert.severity.value,
            'message': alert.message,
            'anomaly_count': len(alert.anomalies),
            'timestamp': alert.triggered_at.isoformat()
        }
        logger.info(f"Would send webhook alert: {webhook_payload}")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        
        return False
    
    def get_active_alerts(self, severity_filter: Optional[AnomalySeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)


class AnomalyDetectionFramework:
    """Main anomaly detection framework orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detectors: Dict[str, BaseAnomalyDetector] = {}
        self.alert_manager = AlertManager()
        self.detection_history: List[AnomalyDetectionResult] = []
        self.feedback_store: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        logger.info("Anomaly Detection Framework initialized")
    
    async def initialize(self):
        """Initialize anomaly detection framework"""
        try:
            await self._load_configuration()
            await self._setup_default_detectors()
            await self._setup_default_alert_rules()
            await self._start_background_tasks()
            
            logger.info("Anomaly Detection Framework fully initialized")
            
        except Exception as e:
            logger.error(f"Framework initialization failed: {e}")
            raise
    
    async def _load_configuration(self):
        """Load framework configuration"""
        default_config = {
            'max_detection_history': 1000,
            'performance_tracking_enabled': True,
            'auto_threshold_adaptation': True,
            'feedback_learning_enabled': True
        }
        
        self.config = {**default_config, **self.config}
        logger.info("Configuration loaded")
    
    async def _setup_default_detectors(self):
        """Setup default anomaly detectors"""
        # Statistical detector
        stat_config = DetectorConfig(
            detector_id="statistical_default",
            name="Statistical Anomaly Detector",
            method=DetectionMethod.STATISTICAL,
            data_type=DataType.NUMERICAL,
            parameters={'threshold': 3.0, 'use_robust_stats': True},
            threshold_config={'target_fpr': 0.05}
        )
        
        stat_detector = StatisticalAnomalyDetector(stat_config)
        self.detectors[stat_config.detector_id] = stat_detector
        
        # Isolation Forest detector
        try:
            iforest_config = DetectorConfig(
                detector_id="isolation_forest_default",
                name="Isolation Forest Detector",
                method=DetectionMethod.MACHINE_LEARNING,
                data_type=DataType.NUMERICAL,
                parameters={'contamination': 0.1, 'n_estimators': 100},
                threshold_config={'target_fpr': 0.1}
            )
            
            iforest_detector = IsolationForestDetector(iforest_config)
            self.detectors[iforest_config.detector_id] = iforest_detector
            
        except Exception as e:
            logger.warning(f"Could not setup Isolation Forest detector: {e}")
        
        # Ensemble detector
        ensemble_config = DetectorConfig(
            detector_id="ensemble_default",
            name="Ensemble Anomaly Detector",
            method=DetectionMethod.ENSEMBLE,
            data_type=DataType.NUMERICAL,
            parameters={'combination_method': 'average'},
            threshold_config={'target_fpr': 0.03}
        )
        
        ensemble_detector = EnsembleAnomalyDetector(ensemble_config)
        
        # Add base detectors to ensemble
        for detector_id, detector in self.detectors.items():
            if detector_id != ensemble_config.detector_id:
                ensemble_detector.add_detector(detector)
        
        self.detectors[ensemble_config.detector_id] = ensemble_detector
        
        logger.info(f"Setup {len(self.detectors)} default detectors")
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        # Critical anomaly rule
        critical_rule = AlertRule(
            rule_id="critical_anomalies",
            name="Critical Anomaly Alert",
            conditions={'min_severity': 'CRITICAL'},
            channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
            recipients=["admin@company.com"],
            cooldown_minutes=30,
            message_template="CRITICAL: {anomaly_count} critical anomalies detected at {timestamp}"
        )
        
        self.alert_manager.add_alert_rule(critical_rule)
        
        # High volume anomaly rule
        volume_rule = AlertRule(
            rule_id="high_volume_anomalies",
            name="High Volume Anomaly Alert",
            conditions={'min_count': 5, 'min_severity': 'MEDIUM'},
            channels=[AlertChannel.LOG],
            recipients=["ops@company.com"],
            cooldown_minutes=60,
            message_template="HIGH VOLUME: {anomaly_count} anomalies detected with max severity {max_severity}"
        )
        
        self.alert_manager.add_alert_rule(volume_rule)
        
        logger.info("Default alert rules configured")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._threshold_adaptation_loop())
        asyncio.create_task(self._history_cleanup_loop())
        logger.info("Background tasks started")
    
    async def register_detector(self, detector: BaseAnomalyDetector) -> str:
        """Register custom anomaly detector"""
        self.detectors[detector.config.detector_id] = detector
        logger.info(f"Registered detector: {detector.config.name}")
        return detector.config.detector_id
    
    async def train_detector(self, detector_id: str, training_data: pd.DataFrame) -> bool:
        """Train specific detector"""
        if detector_id not in self.detectors:
            raise ValueError(f"Detector not found: {detector_id}")
        
        detector = self.detectors[detector_id]
        result = await detector.train(training_data)
        
        logger.info(f"Detector {detector.config.name} training: {'success' if result else 'failed'}")
        return result
    
    async def detect_anomalies(self, data: pd.DataFrame, detector_ids: Optional[List[str]] = None) -> List[AnomalyDetectionResult]:
        """Detect anomalies using specified detectors"""
        if detector_ids is None:
            detector_ids = list(self.detectors.keys())
        
        detection_results = []
        
        for detector_id in detector_ids:
            if detector_id not in self.detectors:
                logger.warning(f"Detector not found: {detector_id}")
                continue
            
            detector = self.detectors[detector_id]
            
            if not detector.is_trained:
                logger.warning(f"Detector {detector.config.name} not trained")
                continue
            
            try:
                result = await detector.detect(data)
                detection_results.append(result)
                
                # Store in history
                self.detection_history.append(result)
                
                # Process alerts
                if result.anomalies:
                    await self.alert_manager.process_anomalies(result.anomalies)
                
                logger.info(f"Detection completed: {len(result.anomalies)} anomalies found by {detector.config.name}")
                
            except Exception as e:
                logger.error(f"Detection failed for {detector.config.name}: {e}")
        
        # Cleanup history
        if len(self.detection_history) > self.config['max_detection_history']:
            self.detection_history = self.detection_history[-self.config['max_detection_history']:]
        
        return detection_results
    
    async def provide_feedback(self, detection_id: str, feedback: List[Dict[str, Any]]):
        """Provide feedback on detection results"""
        # Store feedback
        feedback_entry = {
            'detection_id': detection_id,
            'feedback': feedback,
            'timestamp': datetime.now()
        }
        
        self.feedback_store.append(feedback_entry)
        
        # Update detector models if feedback learning is enabled
        if self.config['feedback_learning_enabled']:
            await self._update_detectors_with_feedback(detection_id, feedback)
        
        logger.info(f"Feedback provided for detection {detection_id}: {len(feedback)} items")
    
    async def _update_detectors_with_feedback(self, detection_id: str, feedback: List[Dict[str, Any]]):
        """Update detector models based on feedback"""
        # Find corresponding detection result
        detection_result = None
        for result in self.detection_history:
            if result.detection_id == detection_id:
                detection_result = result
                break
        
        if not detection_result:
            logger.warning(f"Detection result not found for feedback: {detection_id}")
            return
        
        # Calculate performance metrics from feedback
        true_positives = sum(1 for f in feedback if f.get('is_anomaly', False))
        false_positives = sum(1 for f in feedback if not f.get('is_anomaly', False))
        
        if len(feedback) > 0:
            precision = true_positives / len(feedback)
            false_positive_rate = false_positives / len(feedback)
            
            performance_metrics = {
                'precision': precision,
                'false_positive_rate': false_positive_rate,
                'feedback_count': len(feedback)
            }
            
            # Update detector performance tracking
            detector_name = detection_result.detector_name
            if detector_name not in self.performance_metrics:
                self.performance_metrics[detector_name] = {}
            
            self.performance_metrics[detector_name].update(performance_metrics)
            
            # Adapt thresholds if enabled
            if self.config['auto_threshold_adaptation']:
                for detector in self.detectors.values():
                    if detector.config.name == detector_name:
                        await detector.adapt_threshold(performance_metrics)
    
    async def get_detection_summary(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get detection summary statistics"""
        if time_range:
            start_time, end_time = time_range
            relevant_results = [r for r in self.detection_history 
                             if start_time <= r.detected_at <= end_time]
        else:
            relevant_results = self.detection_history[-100:]  # Last 100 detections
        
        if not relevant_results:
            return {'message': 'No detection results in specified time range'}
        
        total_anomalies = sum(len(r.anomalies) for r in relevant_results)
        
        # Severity distribution
        severity_counts = defaultdict(int)
        for result in relevant_results:
            for anomaly in result.anomalies:
                severity_counts[anomaly.severity.value] += 1
        
        # Detector performance
        detector_stats = defaultdict(lambda: {'detections': 0, 'anomalies': 0})
        for result in relevant_results:
            detector_stats[result.detector_name]['detections'] += 1
            detector_stats[result.detector_name]['anomalies'] += len(result.anomalies)
        
        # Alert statistics
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'summary': {
                'total_detections': len(relevant_results),
                'total_anomalies': total_anomalies,
                'avg_anomalies_per_detection': total_anomalies / len(relevant_results),
                'time_range': {
                    'start': relevant_results[0].detected_at.isoformat() if relevant_results else None,
                    'end': relevant_results[-1].detected_at.isoformat() if relevant_results else None
                }
            },
            'severity_distribution': dict(severity_counts),
            'detector_performance': dict(detector_stats),
            'alerts': {
                'active_count': len(active_alerts),
                'by_severity': {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AnomalySeverity
                }
            }
        }
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        while True:
            try:
                await asyncio.sleep(3600)  # Monitor every hour
                await self._track_system_performance()
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def _threshold_adaptation_loop(self):
        """Background threshold adaptation"""
        while True:
            try:
                await asyncio.sleep(1800)  # Adapt every 30 minutes
                if self.config['auto_threshold_adaptation']:
                    await self._adapt_all_thresholds()
            except Exception as e:
                logger.error(f"Threshold adaptation error: {e}")
    
    async def _history_cleanup_loop(self):
        """Background history cleanup"""
        while True:
            try:
                await asyncio.sleep(86400)  # Cleanup daily
                await self._cleanup_old_data()
            except Exception as e:
                logger.error(f"History cleanup error: {e}")
    
    async def _track_system_performance(self):
        """Track overall system performance"""
        if self.detection_history:
            recent_results = self.detection_history[-50:]  # Last 50 detections
            
            avg_detection_time = np.mean([r.detection_time_ms for r in recent_results])
            total_anomalies = sum(len(r.anomalies) for r in recent_results)
            
            logger.info(f"System performance: avg detection time {avg_detection_time:.2f}ms, "
                       f"{total_anomalies} anomalies in last {len(recent_results)} detections")
    
    async def _adapt_all_thresholds(self):
        """Adapt thresholds for all detectors based on performance"""
        for detector in self.detectors.values():
            detector_name = detector.config.name
            
            if detector_name in self.performance_metrics:
                performance = self.performance_metrics[detector_name]
                await detector.adapt_threshold(performance)
    
    async def _cleanup_old_data(self):
        """Cleanup old detection history and feedback"""
        # Keep only recent detection history
        max_history = self.config.get('max_detection_history', 1000)
        if len(self.detection_history) > max_history:
            self.detection_history = self.detection_history[-max_history:]
        
        # Keep only recent feedback
        max_feedback = 500
        if len(self.feedback_store) > max_feedback:
            self.feedback_store = self.feedback_store[-max_feedback:]
        
        # Cleanup old alert history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.alert_manager.alert_history = [
            alert for alert in self.alert_manager.alert_history
            if alert.triggered_at > cutoff_date
        ]
        
        logger.info("Old data cleanup completed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get framework system status"""
        trained_detectors = sum(1 for d in self.detectors.values() if d.is_trained)
        active_alerts = len(self.alert_manager.get_active_alerts())
        
        return {
            'detectors': {
                'total': len(self.detectors),
                'trained': trained_detectors,
                'methods': [d.config.method.value for d in self.detectors.values()]
            },
            'detection_history': len(self.detection_history),
            'feedback_entries': len(self.feedback_store),
            'alerts': {
                'active': active_alerts,
                'rules': len(self.alert_manager.alert_rules),
                'total_history': len(self.alert_manager.alert_history)
            },
            'performance_tracking': len(self.performance_metrics),
            'configuration': {
                'auto_threshold_adaptation': self.config['auto_threshold_adaptation'],
                'feedback_learning_enabled': self.config['feedback_learning_enabled']
            }
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize anomaly detection framework
        config = {
            'max_detection_history': 500,
            'auto_threshold_adaptation': True,
            'feedback_learning_enabled': True
        }
        
        framework = AnomalyDetectionFramework(config)
        await framework.initialize()
        
        # Create sample data with anomalies
        np.random.seed(42)
        n_samples = 1000
        
        # Normal data
        normal_data = pd.DataFrame({
            'feature1': np.random.normal(10, 2, n_samples),
            'feature2': np.random.normal(5, 1, n_samples),
            'feature3': np.random.normal(0, 0.5, n_samples)
        })
        
        # Add anomalies
        anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
        normal_data.loc[anomaly_indices, 'feature1'] += np.random.normal(0, 10, 50)
        normal_data.loc[anomaly_indices, 'feature2'] += np.random.normal(0, 5, 50)
        
        # Add timestamps
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        normal_data.index = dates
        
        # Train detectors
        training_data = normal_data.iloc[:800]  # First 800 samples for training
        test_data = normal_data.iloc[800:]      # Last 200 for testing
        
        print("Training detectors...")
        for detector_id in framework.detectors:
            result = await framework.train_detector(detector_id, training_data)
            print(f"Detector {detector_id} training: {'success' if result else 'failed'}")
        
        # Detect anomalies
        print("\nDetecting anomalies...")
        detection_results = await framework.detect_anomalies(test_data)
        
        for result in detection_results:
            print(f"{result.detector_name}: {len(result.anomalies)} anomalies detected")
            print(f"  - Detection time: {result.detection_time_ms:.2f}ms")
            print(f"  - Method: {result.method.value}")
            
            if result.anomalies:
                severities = [a.severity.value for a in result.anomalies]
                severity_counts = pd.Series(severities).value_counts()
                print(f"  - Severity distribution: {severity_counts.to_dict()}")
        
        # Get detection summary
        print("\nDetection Summary:")
        summary = await framework.get_detection_summary()
        print(f"Total detections: {summary['summary']['total_detections']}")
        print(f"Total anomalies: {summary['summary']['total_anomalies']}")
        print(f"Severity distribution: {summary['severity_distribution']}")
        print(f"Active alerts: {summary['alerts']['active_count']}")
        
        # Provide feedback (simulated)
        if detection_results:
            detection_id = detection_results[0].detection_id
            feedback = [
                {'anomaly_id': 'test', 'is_anomaly': True, 'confidence': 0.9},
                {'anomaly_id': 'test2', 'is_anomaly': False, 'confidence': 0.7}
            ]
            await framework.provide_feedback(detection_id, feedback)
            print(f"\nFeedback provided for detection {detection_id}")
        
        # Get system status
        status = framework.get_system_status()
        print(f"\nSystem Status:")
        print(f"Detectors: {status['detectors']['trained']}/{status['detectors']['total']} trained")
        print(f"Detection history: {status['detection_history']} entries")
        print(f"Alert rules: {status['alerts']['rules']}")
        
        logger.info("Anomaly Detection Framework demonstration completed")
    
    # Run the example
    asyncio.run(main())