"""
Trend Analyzer Module
Time series analysis and trend detection for metrics
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats, signal
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

from .analytics_engine import AnalyticsEngine


class TrendType(Enum):
    """Types of trends"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    SEASONAL = "seasonal"
    CYCLICAL = "cyclical"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POLYNOMIAL = "polynomial"


class SeasonalityType(Enum):
    """Types of seasonality patterns"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"


class ChangePointType(Enum):
    """Types of change points"""
    LEVEL_SHIFT = "level_shift"
    TREND_CHANGE = "trend_change"
    VARIANCE_CHANGE = "variance_change"
    SEASONAL_CHANGE = "seasonal_change"
    OUTLIER = "outlier"


@dataclass
class TrendInfo:
    """Trend information for a metric"""
    metric_name: str
    trend_type: TrendType
    direction: float  # -1 to 1, negative is decreasing
    strength: float  # 0 to 1, confidence in trend
    slope: float  # Rate of change
    r_squared: float  # Goodness of fit
    p_value: float  # Statistical significance
    start_value: float
    end_value: float
    percent_change: float
    forecast_next: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class Seasonality:
    """Seasonality pattern information"""
    seasonality_type: SeasonalityType
    period: int  # Period in data points
    amplitude: float  # Strength of seasonal effect
    phase: float  # Phase shift
    significance: float  # Statistical significance
    seasonal_factors: List[float] = field(default_factory=list)


@dataclass
class ChangePoint:
    """Detected change point in time series"""
    change_id: str
    timestamp: datetime
    change_type: ChangePointType
    magnitude: float  # Size of change
    confidence: float  # Detection confidence
    before_value: float
    after_value: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Correlation:
    """Correlation between two metrics"""
    metric1: str
    metric2: str
    correlation: float  # -1 to 1
    lag: int  # Optimal lag for correlation
    p_value: float  # Statistical significance
    relationship: str  # linear, non-linear, none
    causality_score: float  # 0 to 1, likelihood of causation


@dataclass
class TrendAnalysis:
    """Complete trend analysis results"""
    analysis_id: str
    timestamp: datetime
    metric_name: str
    time_range: Tuple[datetime, datetime]
    data_points: int
    trend: TrendInfo
    seasonality: Optional[Seasonality] = None
    change_points: List[ChangePoint] = field(default_factory=list)
    forecast: Dict[str, Any] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    correlations: List[Correlation] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


class TrendAnalyzer:
    """
    Advanced trend analysis and pattern detection system
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        self.analytics_engine = analytics_engine
        self.analyses: Dict[str, TrendAnalysis] = {}
        self.metric_history: Dict[str, pd.Series] = {}
        self.correlations: List[Correlation] = []
        self.change_points: Dict[str, List[ChangePoint]] = {}
        
        # Analysis parameters
        self.min_data_points = 30
        self.seasonality_threshold = 0.3
        self.trend_threshold = 0.1
        self.correlation_threshold = 0.5
        
        self._start_analysis()
    
    def _start_analysis(self):
        """Start background analysis tasks"""
        asyncio.create_task(self._continuous_trend_analysis())
        asyncio.create_task(self._detect_correlations())
        asyncio.create_task(self._monitor_change_points())
    
    async def _continuous_trend_analysis(self):
        """Continuously analyze trends"""
        while True:
            try:
                # Get all available metrics
                metrics = await self.analytics_engine.get_metric_names()
                
                for metric_name in metrics:
                    await self.analyze_trend(metric_name)
                
                await asyncio.sleep(3600)  # Analyze hourly
                
            except Exception as e:
                print(f"Trend analysis error: {e}")
                await asyncio.sleep(3600)
    
    async def _detect_correlations(self):
        """Detect correlations between metrics"""
        while True:
            try:
                await self._analyze_metric_correlations()
                await asyncio.sleep(7200)  # Every 2 hours
                
            except Exception as e:
                print(f"Correlation detection error: {e}")
                await asyncio.sleep(7200)
    
    async def _monitor_change_points(self):
        """Monitor for change points in metrics"""
        while True:
            try:
                for metric_name in self.metric_history.keys():
                    await self._detect_change_points(metric_name)
                
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except Exception as e:
                print(f"Change point detection error: {e}")
                await asyncio.sleep(1800)
    
    async def analyze_trend(self, metric_name: str,
                          time_range: Optional[timedelta] = None) -> TrendAnalysis:
        """Analyze trend for a metric"""
        import uuid
        
        # Get metric data
        if time_range is None:
            time_range = timedelta(days=30)
        
        metrics = await self.analytics_engine.get_metrics(
            metric_names=[metric_name],
            time_range=time_range
        )
        
        if len(metrics) < self.min_data_points:
            raise ValueError(f"Insufficient data points for trend analysis: {len(metrics)}")
        
        # Convert to pandas Series
        timestamps = [m["timestamp"] for m in metrics]
        values = [m["value"] for m in metrics]
        ts = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        ts = ts.sort_index()
        
        # Store in history
        self.metric_history[metric_name] = ts
        
        # Analyze trend
        trend_info = self._analyze_trend_type(ts)
        
        # Detect seasonality
        seasonality = self._detect_seasonality(ts)
        
        # Detect change points
        change_points = await self._detect_change_points_in_series(ts)
        
        # Generate forecast
        forecast = self._generate_forecast(ts, trend_info, seasonality)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(ts)
        
        # Find correlations
        correlations = await self._find_metric_correlations(metric_name)
        
        # Generate insights
        insights = self._generate_insights(trend_info, seasonality, 
                                          change_points, anomalies)
        
        # Create analysis
        analysis = TrendAnalysis(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            metric_name=metric_name,
            time_range=(timestamps[0], timestamps[-1]),
            data_points=len(metrics),
            trend=trend_info,
            seasonality=seasonality,
            change_points=change_points,
            forecast=forecast,
            anomalies=anomalies,
            correlations=correlations,
            insights=insights
        )
        
        self.analyses[analysis.analysis_id] = analysis
        return analysis
    
    def _analyze_trend_type(self, ts: pd.Series) -> TrendInfo:
        """Analyze the type and characteristics of a trend"""
        # Prepare data
        x = np.arange(len(ts))
        y = ts.values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate trend metrics
        start_value = y[0]
        end_value = y[-1]
        percent_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else 0
        
        # Determine trend type
        if abs(slope) < self.trend_threshold:
            trend_type = TrendType.STABLE
        elif self._is_exponential(x, y):
            trend_type = TrendType.EXPONENTIAL
        elif self._is_logarithmic(x, y):
            trend_type = TrendType.LOGARITHMIC
        elif self._is_polynomial(x, y):
            trend_type = TrendType.POLYNOMIAL
        elif np.std(y) / np.mean(y) > 0.5:  # High coefficient of variation
            trend_type = TrendType.VOLATILE
        elif slope > 0:
            trend_type = TrendType.INCREASING
        else:
            trend_type = TrendType.DECREASING
        
        # Calculate direction and strength
        direction = np.sign(slope)
        strength = min(abs(r_value), 1.0)
        
        # Forecast next value
        forecast_next = intercept + slope * len(ts)
        
        # Confidence interval
        std_error = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (len(y) - 2))
        confidence_interval = (
            forecast_next - 1.96 * std_error,
            forecast_next + 1.96 * std_error
        )
        
        return TrendInfo(
            metric_name=ts.name if ts.name else "unknown",
            trend_type=trend_type,
            direction=direction,
            strength=strength,
            slope=slope,
            r_squared=r_value ** 2,
            p_value=p_value,
            start_value=start_value,
            end_value=end_value,
            percent_change=percent_change,
            forecast_next=forecast_next,
            confidence_interval=confidence_interval
        )
    
    def _detect_seasonality(self, ts: pd.Series) -> Optional[Seasonality]:
        """Detect seasonality in time series"""
        if len(ts) < 2 * 7:  # Need at least 2 periods for detection
            return None
        
        # Detrend the series
        detrended = signal.detrend(ts.values)
        
        # Try different seasonal periods
        periods = {
            SeasonalityType.DAILY: 1,
            SeasonalityType.WEEKLY: 7,
            SeasonalityType.MONTHLY: 30,
            SeasonalityType.QUARTERLY: 90,
            SeasonalityType.YEARLY: 365
        }
        
        best_seasonality = None
        best_score = 0
        
        for season_type, period in periods.items():
            if len(ts) < 2 * period:
                continue
            
            # Calculate autocorrelation at this lag
            if period < len(detrended):
                autocorr = np.corrcoef(detrended[:-period], detrended[period:])[0, 1]
                
                if abs(autocorr) > self.seasonality_threshold and abs(autocorr) > best_score:
                    best_score = abs(autocorr)
                    
                    # Calculate seasonal factors
                    seasonal_factors = []
                    for i in range(period):
                        indices = np.arange(i, len(detrended), period)
                        if len(indices) > 0:
                            seasonal_factors.append(np.mean(detrended[indices]))
                    
                    # Calculate amplitude
                    amplitude = np.std(seasonal_factors)
                    
                    best_seasonality = Seasonality(
                        seasonality_type=season_type,
                        period=period,
                        amplitude=amplitude,
                        phase=0,  # Would calculate phase shift
                        significance=best_score,
                        seasonal_factors=seasonal_factors
                    )
        
        return best_seasonality
    
    async def _detect_change_points_in_series(self, ts: pd.Series) -> List[ChangePoint]:
        """Detect change points in time series"""
        import uuid
        
        change_points = []
        values = ts.values
        
        if len(values) < 10:
            return change_points
        
        # Use CUSUM for change detection
        cumsum = np.cumsum(values - np.mean(values))
        
        # Find peaks in CUSUM
        peaks, properties = signal.find_peaks(np.abs(cumsum), 
                                             height=np.std(cumsum) * 2)
        
        for peak_idx in peaks:
            if peak_idx > 0 and peak_idx < len(values) - 1:
                before_mean = np.mean(values[:peak_idx])
                after_mean = np.mean(values[peak_idx:])
                magnitude = after_mean - before_mean
                
                # Determine change type
                if abs(magnitude) > np.std(values):
                    change_type = ChangePointType.LEVEL_SHIFT
                else:
                    # Check for trend change
                    before_slope = np.polyfit(range(peak_idx), values[:peak_idx], 1)[0]
                    after_slope = np.polyfit(range(len(values) - peak_idx), 
                                           values[peak_idx:], 1)[0]
                    
                    if np.sign(before_slope) != np.sign(after_slope):
                        change_type = ChangePointType.TREND_CHANGE
                    else:
                        change_type = ChangePointType.VARIANCE_CHANGE
                
                change_points.append(ChangePoint(
                    change_id=str(uuid.uuid4()),
                    timestamp=ts.index[peak_idx],
                    change_type=change_type,
                    magnitude=magnitude,
                    confidence=min(abs(cumsum[peak_idx]) / np.max(np.abs(cumsum)), 1.0),
                    before_value=before_mean,
                    after_value=after_mean,
                    description=f"{change_type.value} detected at index {peak_idx}"
                ))
        
        return change_points
    
    def _generate_forecast(self, ts: pd.Series, trend: TrendInfo,
                         seasonality: Optional[Seasonality]) -> Dict[str, Any]:
        """Generate forecast for time series"""
        forecast_horizon = min(len(ts) // 4, 30)  # Forecast 25% ahead or 30 points max
        
        # Simple forecast based on trend
        x = np.arange(len(ts))
        x_future = np.arange(len(ts), len(ts) + forecast_horizon)
        
        if trend.trend_type == TrendType.EXPONENTIAL:
            # Fit exponential model
            try:
                z = np.polyfit(x, np.log(ts.values + 1), 1)
                forecast_values = np.exp(np.poly1d(z)(x_future)) - 1
            except:
                forecast_values = [trend.forecast_next] * forecast_horizon
        elif trend.trend_type == TrendType.POLYNOMIAL:
            # Fit polynomial model
            z = np.polyfit(x, ts.values, 2)
            forecast_values = np.poly1d(z)(x_future)
        else:
            # Linear forecast
            forecast_values = trend.slope * x_future + (trend.end_value - trend.slope * (len(ts) - 1))
        
        # Add seasonality if detected
        if seasonality and seasonality.seasonal_factors:
            for i in range(forecast_horizon):
                seasonal_index = i % len(seasonality.seasonal_factors)
                forecast_values[i] += seasonality.seasonal_factors[seasonal_index]
        
        # Calculate prediction intervals
        std_error = np.std(ts.values)
        lower_bound = forecast_values - 1.96 * std_error
        upper_bound = forecast_values + 1.96 * std_error
        
        return {
            "horizon": forecast_horizon,
            "values": forecast_values.tolist(),
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "confidence_level": 0.95
        }
    
    def _detect_anomalies(self, ts: pd.Series) -> List[Dict[str, Any]]:
        """Detect anomalies in time series"""
        anomalies = []
        values = ts.values
        
        # Use IQR method
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Also use z-score method
        mean = np.mean(values)
        std = np.std(values)
        
        for i, value in enumerate(values):
            is_outlier = False
            anomaly_score = 0
            
            # Check IQR
            if value < lower_bound or value > upper_bound:
                is_outlier = True
                anomaly_score = max(
                    abs(value - lower_bound) / iqr if value < lower_bound else 0,
                    abs(value - upper_bound) / iqr if value > upper_bound else 0
                )
            
            # Check z-score
            z_score = abs((value - mean) / std) if std > 0 else 0
            if z_score > 3:
                is_outlier = True
                anomaly_score = max(anomaly_score, z_score / 3)
            
            if is_outlier:
                anomalies.append({
                    "timestamp": ts.index[i].isoformat() if hasattr(ts.index[i], 'isoformat') else str(ts.index[i]),
                    "value": value,
                    "expected_range": (lower_bound, upper_bound),
                    "z_score": z_score,
                    "anomaly_score": min(anomaly_score, 1.0),
                    "type": "high" if value > upper_bound else "low"
                })
        
        return anomalies
    
    async def _find_metric_correlations(self, metric_name: str) -> List[Correlation]:
        """Find correlations with other metrics"""
        correlations = []
        
        if metric_name not in self.metric_history:
            return correlations
        
        ts1 = self.metric_history[metric_name]
        
        for other_metric, ts2 in self.metric_history.items():
            if other_metric == metric_name:
                continue
            
            # Align time series
            aligned = pd.DataFrame({'ts1': ts1, 'ts2': ts2}).dropna()
            
            if len(aligned) < self.min_data_points:
                continue
            
            # Calculate correlation
            corr, p_value = stats.pearsonr(aligned['ts1'], aligned['ts2'])
            
            if abs(corr) > self.correlation_threshold:
                # Check for lagged correlation
                best_lag = 0
                best_corr = corr
                
                for lag in range(-5, 6):
                    if lag == 0:
                        continue
                    
                    if lag > 0:
                        lagged_corr, _ = stats.pearsonr(
                            aligned['ts1'][:-lag],
                            aligned['ts2'][lag:]
                        )
                    else:
                        lagged_corr, _ = stats.pearsonr(
                            aligned['ts1'][-lag:],
                            aligned['ts2'][:lag]
                        )
                    
                    if abs(lagged_corr) > abs(best_corr):
                        best_corr = lagged_corr
                        best_lag = lag
                
                # Determine relationship type
                if abs(best_corr) > 0.9:
                    relationship = "strong_linear"
                elif abs(best_corr) > 0.7:
                    relationship = "linear"
                else:
                    relationship = "weak_linear"
                
                correlations.append(Correlation(
                    metric1=metric_name,
                    metric2=other_metric,
                    correlation=best_corr,
                    lag=best_lag,
                    p_value=p_value,
                    relationship=relationship,
                    causality_score=abs(best_corr) * (1 if best_lag > 0 else 0.5)
                ))
        
        return correlations
    
    def _generate_insights(self, trend: TrendInfo, seasonality: Optional[Seasonality],
                         change_points: List[ChangePoint],
                         anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from analysis"""
        insights = []
        
        # Trend insights
        if trend.trend_type == TrendType.INCREASING:
            insights.append(f"Metric showing {trend.percent_change:.1f}% increase with {trend.strength:.1%} confidence")
        elif trend.trend_type == TrendType.DECREASING:
            insights.append(f"Metric declining by {abs(trend.percent_change):.1f}% with slope {trend.slope:.2f}")
        elif trend.trend_type == TrendType.VOLATILE:
            insights.append("High volatility detected - consider smoothing or aggregation")
        elif trend.trend_type == TrendType.EXPONENTIAL:
            insights.append("Exponential growth pattern detected - may indicate scaling issues")
        
        # Seasonality insights
        if seasonality:
            insights.append(f"{seasonality.seasonality_type.value} seasonality detected with period {seasonality.period}")
        
        # Change point insights
        if change_points:
            recent_changes = [cp for cp in change_points 
                            if (datetime.now() - cp.timestamp).days < 7]
            if recent_changes:
                insights.append(f"{len(recent_changes)} recent change points detected in last 7 days")
        
        # Anomaly insights
        if anomalies:
            recent_anomalies = len([a for a in anomalies 
                                  if a.get('anomaly_score', 0) > 0.8])
            if recent_anomalies > 0:
                insights.append(f"{recent_anomalies} significant anomalies detected")
        
        # Forecast insights
        if trend.forecast_next:
            if trend.confidence_interval:
                lower, upper = trend.confidence_interval
                insights.append(f"Next value forecast: {trend.forecast_next:.2f} [{lower:.2f}, {upper:.2f}]")
        
        return insights
    
    def _is_exponential(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if data follows exponential pattern"""
        try:
            # Fit exponential: y = a * exp(b * x)
            log_y = np.log(y + 1)  # Add 1 to handle zeros
            slope, intercept, r_value, _, _ = stats.linregress(x, log_y)
            return abs(r_value) > 0.9 and slope > 0.1
        except:
            return False
    
    def _is_logarithmic(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if data follows logarithmic pattern"""
        try:
            # Fit logarithmic: y = a * log(x) + b
            log_x = np.log(x + 1)
            slope, intercept, r_value, _, _ = stats.linregress(log_x, y)
            return abs(r_value) > 0.9
        except:
            return False
    
    def _is_polynomial(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if data follows polynomial pattern"""
        try:
            # Fit polynomial of degree 2
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            y_pred = p(x)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Check if polynomial fits better than linear
            linear_r = stats.linregress(x, y)[2] ** 2
            
            return r_squared > 0.9 and r_squared > linear_r + 0.1
        except:
            return False
    
    async def _analyze_metric_correlations(self):
        """Analyze correlations between all metrics"""
        self.correlations.clear()
        
        metric_names = list(self.metric_history.keys())
        
        for i, metric1 in enumerate(metric_names):
            for metric2 in metric_names[i+1:]:
                correlations = await self._find_metric_correlations(metric1)
                self.correlations.extend([c for c in correlations if c.metric2 == metric2])
    
    async def _detect_change_points(self, metric_name: str):
        """Detect change points for a metric"""
        if metric_name not in self.metric_history:
            return
        
        ts = self.metric_history[metric_name]
        change_points = await self._detect_change_points_in_series(ts)
        
        self.change_points[metric_name] = change_points
    
    async def get_metric_trends(self, 
                              metrics: Optional[List[str]] = None) -> Dict[str, TrendInfo]:
        """Get current trends for metrics"""
        if metrics is None:
            metrics = list(self.metric_history.keys())
        
        trends = {}
        for metric in metrics:
            if metric in self.metric_history:
                ts = self.metric_history[metric]
                trends[metric] = self._analyze_trend_type(ts)
        
        return trends
    
    async def get_correlated_metrics(self, metric_name: str,
                                   min_correlation: float = 0.5) -> List[Correlation]:
        """Get metrics correlated with given metric"""
        return [c for c in self.correlations
                if (c.metric1 == metric_name or c.metric2 == metric_name) and
                   abs(c.correlation) >= min_correlation]