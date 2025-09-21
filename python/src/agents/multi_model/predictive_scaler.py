"""
Predictive Intelligence Scaling System
ML-powered system to predict agent demand and pre-spawn resources for sub-second response times.
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import pandas as pd
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

from .model_ensemble import ModelEnsemble, TaskType, TaskRequest
from ..monitoring.metrics import websocket_connections_active

logger = logging.getLogger(__name__)


@dataclass
class DemandMetrics:
    """Metrics for demand prediction."""
    timestamp: datetime
    active_connections: int
    pending_tasks: int
    completed_tasks_last_hour: int
    avg_response_time: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    task_type_distribution: Dict[str, int] = field(default_factory=dict)
    hour_of_day: int = 0
    day_of_week: int = 0
    is_business_hours: bool = False


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning."""
    action: str  # scale_up, scale_down, maintain
    target_agents: int
    confidence: float
    reasoning: str
    predicted_demand: float
    current_capacity: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentPool:
    """Pool of pre-spawned agents."""
    model_id: str
    active_agents: int
    idle_agents: int
    spawning_agents: int
    max_agents: int
    min_agents: int
    last_scaling: datetime = field(default_factory=datetime.now)
    warmup_time: float = 2.0  # Time to spawn new agent


class PredictiveAgentScaler:
    """
    Predictive scaling system using ML to anticipate demand.
    """
    
    def __init__(self, model_ensemble: ModelEnsemble, redis_client=None):
        self.ensemble = model_ensemble
        self.redis_client = redis_client
        
        # Agent pools for each model
        self.agent_pools: Dict[str, AgentPool] = {}
        
        # Historical data for ML training
        self.demand_history: deque = deque(maxlen=10000)  # Keep last 10k data points
        self.prediction_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Real-time metrics
        self.current_metrics = DemandMetrics(
            timestamp=datetime.now(),
            active_connections=0,
            pending_tasks=0,
            completed_tasks_last_hour=0,
            avg_response_time=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            error_rate=0.0
        )
        
        # Scaling parameters
        self.scaling_config = {
            "prediction_interval": 60,  # Predict every 60 seconds
            "scaling_interval": 30,     # Scale every 30 seconds
            "min_confidence": 0.7,      # Minimum confidence to scale
            "scale_up_threshold": 0.8,  # Scale up when capacity > 80%
            "scale_down_threshold": 0.3, # Scale down when capacity < 30%
            "max_scale_up": 3,          # Max agents to scale up at once
            "max_scale_down": 2,        # Max agents to scale down at once
            "cooldown_period": 300,     # 5 minutes between scaling actions
        }
        
        # Feature engineering parameters
        self.lookback_window = 12  # 12 time periods for trend analysis
        
        # Initialize agent pools
        self._initialize_agent_pools()
        
        # Start background tasks
        self.running = True
        self.prediction_task = None
        self.scaling_task = None
        self.metrics_task = None
    
    def _initialize_agent_pools(self):
        """Initialize agent pools for each model."""
        for model_id in self.ensemble.model_configs.keys():
            self.agent_pools[model_id] = AgentPool(
                model_id=model_id,
                active_agents=0,
                idle_agents=1,  # Start with 1 idle agent
                spawning_agents=0,
                max_agents=10,
                min_agents=1
            )
    
    async def start(self):
        """Start the predictive scaling system."""
        logger.info("Starting predictive intelligence scaling system")
        
        # Load historical data and models
        await self._load_historical_data()
        await self._load_prediction_models()
        
        # Start background tasks
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Predictive scaling system started")
    
    async def stop(self):
        """Stop the predictive scaling system."""
        logger.info("Stopping predictive scaling system")
        
        self.running = False
        
        if self.prediction_task:
            self.prediction_task.cancel()
        if self.scaling_task:
            self.scaling_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # Save models and data
        await self._save_prediction_models()
        await self._save_historical_data()
        
        logger.info("Predictive scaling system stopped")
    
    async def _metrics_collection_loop(self):
        """Collect metrics for demand prediction."""
        while self.running:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(10)
    
    async def _collect_current_metrics(self):
        """Collect current system metrics."""
        now = datetime.now()
        
        # Get current metrics from various sources
        # This would integrate with your monitoring system
        active_connections = await self._get_active_connections()
        pending_tasks = await self._get_pending_tasks()
        completed_tasks = await self._get_completed_tasks_last_hour()
        avg_response_time = await self._get_avg_response_time()
        cpu_usage = await self._get_cpu_usage()
        memory_usage = await self._get_memory_usage()
        error_rate = await self._get_error_rate()
        
        # Task type distribution
        task_distribution = await self._get_task_type_distribution()
        
        self.current_metrics = DemandMetrics(
            timestamp=now,
            active_connections=active_connections,
            pending_tasks=pending_tasks,
            completed_tasks_last_hour=completed_tasks,
            avg_response_time=avg_response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            error_rate=error_rate,
            task_type_distribution=task_distribution,
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_business_hours=9 <= now.hour <= 17 and now.weekday() < 5
        )
        
        # Add to history for ML training
        self.demand_history.append(self.current_metrics)
        
        # Retrain models periodically
        if len(self.demand_history) % 100 == 0:
            await self._retrain_models()
    
    async def _prediction_loop(self):
        """Main prediction loop."""
        while self.running:
            try:
                # Make predictions for each model type
                predictions = await self._predict_demand()
                
                # Store predictions for scaling decisions
                await self._store_predictions(predictions)
                
                await asyncio.sleep(self.scaling_config["prediction_interval"])
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(30)
    
    async def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                # Get scaling decisions for each model
                scaling_decisions = await self._make_scaling_decisions()
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    await self._execute_scaling_decision(decision)
                
                await asyncio.sleep(self.scaling_config["scaling_interval"])
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(30)
    
    async def _predict_demand(self) -> Dict[str, float]:
        """Predict demand for each task type."""
        if len(self.demand_history) < 10:
            # Not enough data for prediction
            return {}
        
        predictions = {}
        
        for task_type in TaskType:
            try:
                prediction = await self._predict_task_type_demand(task_type)
                predictions[task_type.value] = prediction
            except Exception as e:
                logger.warning(f"Prediction failed for {task_type.value}: {e}")
                # Use historical average as fallback
                recent_demand = [
                    m.task_type_distribution.get(task_type.value, 0)
                    for m in list(self.demand_history)[-10:]
                ]
                predictions[task_type.value] = np.mean(recent_demand) if recent_demand else 1
        
        logger.debug(f"Demand predictions: {predictions}")
        return predictions
    
    async def _predict_task_type_demand(self, task_type: TaskType) -> float:
        """Predict demand for specific task type using ML."""
        # Prepare feature matrix
        features = self._prepare_features_for_prediction()
        
        if features is None or len(features) < 5:
            # Fallback to simple trend analysis
            return self._simple_trend_prediction(task_type)
        
        model_key = f"demand_{task_type.value}"
        
        if model_key not in self.prediction_models:
            # Train new model
            await self._train_demand_model(task_type)
        
        if model_key not in self.prediction_models:
            # Training failed, use fallback
            return self._simple_trend_prediction(task_type)
        
        model = self.prediction_models[model_key]
        scaler = self.scalers[model_key]
        
        # Scale features and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        return max(prediction, 0)  # Ensure non-negative
    
    def _prepare_features_for_prediction(self) -> Optional[List[float]]:
        """Prepare feature vector for prediction."""
        if len(self.demand_history) < self.lookback_window:
            return None
        
        recent_metrics = list(self.demand_history)[-self.lookback_window:]
        
        features = []
        
        # Current metrics
        current = self.current_metrics
        features.extend([
            current.active_connections,
            current.pending_tasks,
            current.completed_tasks_last_hour,
            current.avg_response_time,
            current.cpu_usage,
            current.memory_usage,
            current.error_rate,
            current.hour_of_day,
            current.day_of_week,
            int(current.is_business_hours),
        ])
        
        # Trend features (last 5 periods)
        trend_window = recent_metrics[-5:]
        
        # Connection trend
        connections = [m.active_connections for m in trend_window]
        features.extend([
            np.mean(connections),
            np.std(connections) if len(connections) > 1 else 0,
            connections[-1] - connections[0] if len(connections) > 1 else 0  # Slope
        ])
        
        # Response time trend
        response_times = [m.avg_response_time for m in trend_window]
        features.extend([
            np.mean(response_times),
            np.std(response_times) if len(response_times) > 1 else 0,
            response_times[-1] - response_times[0] if len(response_times) > 1 else 0
        ])
        
        # Task completion trend
        completed = [m.completed_tasks_last_hour for m in trend_window]
        features.extend([
            np.mean(completed),
            np.std(completed) if len(completed) > 1 else 0,
            completed[-1] - completed[0] if len(completed) > 1 else 0
        ])
        
        return features
    
    def _simple_trend_prediction(self, task_type: TaskType) -> float:
        """Simple trend-based prediction as fallback."""
        recent_metrics = list(self.demand_history)[-5:]
        
        demands = [
            m.task_type_distribution.get(task_type.value, 0)
            for m in recent_metrics
        ]
        
        if not demands:
            return 1.0
        
        # Simple linear trend
        if len(demands) >= 3:
            # Use last 3 points for trend
            x = np.array([0, 1, 2])
            y = np.array(demands[-3:])
            
            # Linear regression
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x))**2)
            intercept = np.mean(y) - slope * np.mean(x)
            
            # Predict next point
            prediction = slope * 3 + intercept
            return max(prediction, np.mean(demands))
        else:
            return np.mean(demands)
    
    async def _train_demand_model(self, task_type: TaskType):
        """Train ML model for demand prediction."""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(task_type)
            
            if len(X) < 10:  # Need minimum data for training
                return
            
            model_key = f"demand_{task_type.value}"
            
            # Create and train model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = model.predict(X_scaled)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            logger.info(f"Trained model for {task_type.value}: MAE={mae:.3f}, R2={r2:.3f}")
            
            # Store model and scaler
            self.prediction_models[model_key] = model
            self.scalers[model_key] = scaler
            
        except Exception as e:
            logger.error(f"Failed to train model for {task_type.value}: {e}")
    
    def _prepare_training_data(self, task_type: TaskType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for demand prediction."""
        if len(self.demand_history) < self.lookback_window + 1:
            return np.array([]), np.array([])
        
        X = []
        y = []
        
        history_list = list(self.demand_history)
        
        # Create sliding window of features and targets
        for i in range(self.lookback_window, len(history_list)):
            # Features from current and past observations
            features = self._extract_features_from_window(
                history_list[i-self.lookback_window:i]
            )
            
            if features is None:
                continue
            
            # Target: demand in next period
            target = history_list[i].task_type_distribution.get(task_type.value, 0)
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _extract_features_from_window(self, window: List[DemandMetrics]) -> Optional[List[float]]:
        """Extract features from a window of metrics."""
        if not window:
            return None
        
        features = []
        
        # Current point features
        current = window[-1]
        features.extend([
            current.active_connections,
            current.pending_tasks,
            current.completed_tasks_last_hour,
            current.avg_response_time,
            current.cpu_usage,
            current.memory_usage,
            current.error_rate,
            current.hour_of_day,
            current.day_of_week,
            int(current.is_business_hours),
        ])
        
        # Statistical features over window
        connections = [m.active_connections for m in window]
        features.extend([
            np.mean(connections),
            np.std(connections) if len(connections) > 1 else 0,
            max(connections) - min(connections),  # Range
        ])
        
        response_times = [m.avg_response_time for m in window]
        features.extend([
            np.mean(response_times),
            np.std(response_times) if len(response_times) > 1 else 0,
            max(response_times) - min(response_times),
        ])
        
        return features
    
    async def _make_scaling_decisions(self) -> List[ScalingDecision]:
        """Make scaling decisions based on predictions."""
        decisions = []
        
        # Get current predictions
        predictions = await self._get_stored_predictions()
        
        if not predictions:
            return decisions
        
        for model_id, pool in self.agent_pools.items():
            try:
                decision = await self._make_model_scaling_decision(
                    model_id, pool, predictions
                )
                if decision:
                    decisions.append(decision)
            except Exception as e:
                logger.error(f"Scaling decision failed for {model_id}: {e}")
        
        return decisions
    
    async def _make_model_scaling_decision(
        self,
        model_id: str,
        pool: AgentPool,
        predictions: Dict[str, float]
    ) -> Optional[ScalingDecision]:
        """Make scaling decision for specific model."""
        
        # Calculate predicted demand for this model's task types
        model_config = self.ensemble.model_configs[model_id]
        relevant_predictions = []
        
        for task_type in model_config.strengths:
            if task_type.value in predictions:
                relevant_predictions.append(predictions[task_type.value])
        
        if not relevant_predictions:
            return None
        
        predicted_demand = sum(relevant_predictions)
        current_capacity = pool.active_agents + pool.idle_agents
        utilization = predicted_demand / max(current_capacity, 1)
        
        # Check cooldown period
        time_since_scaling = (datetime.now() - pool.last_scaling).total_seconds()
        if time_since_scaling < self.scaling_config["cooldown_period"]:
            return None
        
        # Make scaling decision
        if utilization > self.scaling_config["scale_up_threshold"]:
            # Scale up
            target_agents = min(
                current_capacity + self.scaling_config["max_scale_up"],
                pool.max_agents
            )
            
            if target_agents > current_capacity:
                return ScalingDecision(
                    action="scale_up",
                    target_agents=target_agents,
                    confidence=min(utilization, 1.0),
                    reasoning=f"High utilization: {utilization:.2f}",
                    predicted_demand=predicted_demand,
                    current_capacity=current_capacity
                )
        
        elif utilization < self.scaling_config["scale_down_threshold"]:
            # Scale down
            target_agents = max(
                current_capacity - self.scaling_config["max_scale_down"],
                pool.min_agents
            )
            
            if target_agents < current_capacity:
                return ScalingDecision(
                    action="scale_down",
                    target_agents=target_agents,
                    confidence=1.0 - utilization,
                    reasoning=f"Low utilization: {utilization:.2f}",
                    predicted_demand=predicted_demand,
                    current_capacity=current_capacity
                )
        
        return ScalingDecision(
            action="maintain",
            target_agents=current_capacity,
            confidence=0.8,
            reasoning=f"Optimal utilization: {utilization:.2f}",
            predicted_demand=predicted_demand,
            current_capacity=current_capacity
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        # This would integrate with your agent spawning system
        logger.info(
            f"Scaling decision: {decision.action} to {decision.target_agents} agents "
            f"(confidence: {decision.confidence:.3f}, reason: {decision.reasoning})"
        )
        
        # For now, just update the pool tracking
        # In practice, this would spawn/terminate actual agent processes
        
        # Track scaling decisions
        await self._track_scaling_decision(decision)
    
    # Placeholder methods for metric collection
    # These would integrate with your actual monitoring system
    
    async def _get_active_connections(self) -> int:
        """Get number of active WebSocket connections."""
        # This would integrate with your WebSocket monitoring
        return 10  # Placeholder
    
    async def _get_pending_tasks(self) -> int:
        """Get number of pending tasks."""
        return 5  # Placeholder
    
    async def _get_completed_tasks_last_hour(self) -> int:
        """Get number of completed tasks in last hour."""
        return 50  # Placeholder
    
    async def _get_avg_response_time(self) -> float:
        """Get average response time."""
        return 1.5  # Placeholder
    
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return 45.0  # Placeholder
    
    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        return 60.0  # Placeholder
    
    async def _get_error_rate(self) -> float:
        """Get error rate percentage."""
        return 2.0  # Placeholder
    
    async def _get_task_type_distribution(self) -> Dict[str, int]:
        """Get distribution of task types."""
        return {
            "coding": 15,
            "analysis": 10,
            "simple_query": 20,
            "creative_writing": 5
        }  # Placeholder
    
    async def _store_predictions(self, predictions: Dict[str, float]):
        """Store predictions for scaling decisions."""
        if self.redis_client:
            try:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "predictions": predictions
                }
                self.redis_client.setex(
                    "demand_predictions",
                    300,  # 5 minutes TTL
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Failed to store predictions: {e}")
    
    async def _get_stored_predictions(self) -> Dict[str, float]:
        """Get stored predictions."""
        if self.redis_client:
            try:
                data = self.redis_client.get("demand_predictions")
                if data:
                    parsed = json.loads(data)
                    return parsed.get("predictions", {})
            except Exception as e:
                logger.warning(f"Failed to get predictions: {e}")
        return {}
    
    async def _track_scaling_decision(self, decision: ScalingDecision):
        """Track scaling decisions for analysis."""
        # This would store scaling decisions for later analysis
        pass
    
    async def _retrain_models(self):
        """Periodically retrain prediction models."""
        logger.info("Retraining demand prediction models")
        
        for task_type in TaskType:
            try:
                await self._train_demand_model(task_type)
            except Exception as e:
                logger.error(f"Retraining failed for {task_type.value}: {e}")
    
    async def _load_historical_data(self):
        """Load historical data from storage."""
        # This would load from persistent storage
        pass
    
    async def _save_historical_data(self):
        """Save historical data to storage."""
        # This would save to persistent storage
        pass
    
    async def _load_prediction_models(self):
        """Load trained models from storage."""
        # This would load from persistent storage
        pass
    
    async def _save_prediction_models(self):
        """Save trained models to storage."""
        # This would save to persistent storage
        pass
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        predictions = await self._get_stored_predictions()
        
        status = {
            "current_metrics": {
                "active_connections": self.current_metrics.active_connections,
                "pending_tasks": self.current_metrics.pending_tasks,
                "avg_response_time": self.current_metrics.avg_response_time,
                "cpu_usage": self.current_metrics.cpu_usage,
                "memory_usage": self.current_metrics.memory_usage,
                "error_rate": self.current_metrics.error_rate,
            },
            "predictions": predictions,
            "agent_pools": {},
            "models_trained": len(self.prediction_models),
            "data_points": len(self.demand_history)
        }
        
        for model_id, pool in self.agent_pools.items():
            status["agent_pools"][model_id] = {
                "active_agents": pool.active_agents,
                "idle_agents": pool.idle_agents,
                "spawning_agents": pool.spawning_agents,
                "max_agents": pool.max_agents,
                "min_agents": pool.min_agents,
                "last_scaling": pool.last_scaling.isoformat()
            }
        
        return status