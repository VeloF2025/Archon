"""
Visualization Engine
Advanced data visualization and chart generation system
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import base64
import io

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts available"""
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    PIE = "pie"
    DONUT = "donut"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    FUNNEL = "funnel"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    CANDLESTICK = "candlestick"
    RADAR = "radar"
    BUBBLE = "bubble"
    WATERFALL = "waterfall"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    GANTT = "gantt"
    NETWORK = "network"
    TIMELINE = "timeline"


class ColorScheme(Enum):
    """Available color schemes"""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COLORBLIND = "colorblind"
    MONOCHROME = "monochrome"
    VIBRANT = "vibrant"
    PASTEL = "pastel"
    GRADIENT = "gradient"


class InteractionMode(Enum):
    """Chart interaction modes"""
    STATIC = "static"
    INTERACTIVE = "interactive"
    ANIMATED = "animated"
    REAL_TIME = "real_time"


@dataclass
class ChartConfig:
    """Configuration for chart rendering"""
    chart_type: ChartType
    title: str
    subtitle: Optional[str] = None
    width: int = 800
    height: int = 400
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    interaction_mode: InteractionMode = InteractionMode.INTERACTIVE
    show_legend: bool = True
    show_grid: bool = True
    show_axes: bool = True
    show_tooltips: bool = True
    show_data_labels: bool = False
    animation_duration: int = 500  # milliseconds
    custom_colors: List[str] = field(default_factory=list)
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    margin: Dict[str, int] = field(default_factory=lambda: {"top": 50, "right": 50, "bottom": 50, "left": 50})


@dataclass
class DataSeries:
    """Data series for visualization"""
    name: str
    data: List[Union[float, int, Dict[str, Any]]]
    labels: Optional[List[str]] = None
    color: Optional[str] = None
    type: Optional[str] = None  # For mixed charts
    y_axis: int = 0  # For multiple y-axes
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartAxis:
    """Axis configuration"""
    title: str
    type: str = "linear"  # linear, logarithmic, category, datetime
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    tick_interval: Optional[float] = None
    format: Optional[str] = None  # Format string for labels
    position: str = "bottom"  # bottom, top, left, right
    visible: bool = True
    grid_lines: bool = True


@dataclass
class Annotation:
    """Chart annotation"""
    type: str  # text, line, rect, circle
    text: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    x2: Optional[float] = None  # For lines and rectangles
    y2: Optional[float] = None
    color: str = "#000000"
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartDefinition:
    """Complete chart definition"""
    config: ChartConfig
    series: List[DataSeries]
    x_axis: Optional[ChartAxis] = None
    y_axis: Optional[ChartAxis] = None
    y2_axis: Optional[ChartAxis] = None  # Secondary y-axis
    annotations: List[Annotation] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VisualizationTemplate:
    """Reusable visualization template"""
    template_id: str
    name: str
    description: str
    chart_type: ChartType
    default_config: ChartConfig
    required_fields: List[str]
    optional_fields: List[str]
    transformations: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]


class DataTransformer:
    """Transform data for visualization"""
    
    @staticmethod
    def pivot_data(data: List[Dict[str, Any]], index: str, columns: str, 
                   values: str, aggfunc: str = "mean") -> pd.DataFrame:
        """Pivot data for visualization"""
        df = pd.DataFrame(data)
        return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    @staticmethod
    def aggregate_time_series(data: List[Dict[str, Any]], time_field: str,
                            value_field: str, interval: str = "1H") -> pd.DataFrame:
        """Aggregate time series data"""
        df = pd.DataFrame(data)
        df[time_field] = pd.to_datetime(df[time_field])
        df.set_index(time_field, inplace=True)
        return df[value_field].resample(interval).mean()
    
    @staticmethod
    def normalize_data(data: List[float], method: str = "minmax") -> List[float]:
        """Normalize data values"""
        if not data:
            return []
        
        if method == "minmax":
            min_val = min(data)
            max_val = max(data)
            if max_val == min_val:
                return [0.5] * len(data)
            return [(x - min_val) / (max_val - min_val) for x in data]
        
        elif method == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return [0] * len(data)
            return [(x - mean) / std for x in data]
        
        else:
            return data
    
    @staticmethod
    def calculate_moving_average(data: List[float], window: int = 5) -> List[float]:
        """Calculate moving average"""
        if len(data) < window:
            return data
        
        result = []
        for i in range(len(data)):
            if i < window - 1:
                result.append(np.mean(data[:i+1]))
            else:
                result.append(np.mean(data[i-window+1:i+1]))
        
        return result
    
    @staticmethod
    def detect_outliers(data: List[float], threshold: float = 3.0) -> List[int]:
        """Detect outliers using z-score"""
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return []
        
        outliers = []
        for i, value in enumerate(data):
            z_score = abs((value - mean) / std)
            if z_score > threshold:
                outliers.append(i)
        
        return outliers


class VisualizationEngine:
    """Advanced visualization engine for enterprise analytics"""
    
    def __init__(self):
        self.charts: Dict[str, ChartDefinition] = {}
        self.templates: Dict[str, VisualizationTemplate] = {}
        self.color_palettes = self._initialize_color_palettes()
        self.transformer = DataTransformer()
        
        # Chart rendering cache
        self.render_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Real-time chart updates
        self.real_time_charts: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "charts_created": 0,
            "charts_rendered": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "templates_used": 0
        }
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _initialize_color_palettes(self) -> Dict[ColorScheme, List[str]]:
        """Initialize color palettes for different schemes"""
        return {
            ColorScheme.DEFAULT: [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ],
            ColorScheme.DARK: [
                "#00d9ff", "#00ff88", "#ff00ff", "#ffaa00", "#ff0055",
                "#0099ff", "#66ff00", "#ff0099", "#ffff00", "#00ffff"
            ],
            ColorScheme.LIGHT: [
                "#a8dadc", "#f1e5d1", "#e8c9a0", "#d4a5a5", "#c9ada7",
                "#b8bedd", "#b8e6b8", "#f4acb7", "#ffd6a5", "#caffbf"
            ],
            ColorScheme.COLORBLIND: [
                "#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ece133",
                "#56b4e9", "#a6761d", "#e5601e", "#55a868", "#b4436c"
            ],
            ColorScheme.VIBRANT: [
                "#ff006e", "#fb5607", "#ffbe0b", "#8338ec", "#3a86ff",
                "#06eb00", "#ff4365", "#00d9ff", "#ffd60a", "#7209b7"
            ],
            ColorScheme.PASTEL: [
                "#ffb3ba", "#ffdfba", "#ffffba", "#baffc9", "#bae1ff",
                "#e0bbe4", "#fec8d8", "#d4f1ee", "#fff5b5", "#e5d4ff"
            ]
        }
    
    def _initialize_default_templates(self) -> None:
        """Initialize default visualization templates"""
        # Time series template
        self.templates["time_series"] = VisualizationTemplate(
            template_id="time_series",
            name="Time Series Chart",
            description="Standard time series visualization",
            chart_type=ChartType.LINE,
            default_config=ChartConfig(
                chart_type=ChartType.LINE,
                title="Time Series",
                show_grid=True,
                interaction_mode=InteractionMode.INTERACTIVE
            ),
            required_fields=["timestamp", "value"],
            optional_fields=["series_name", "category"],
            transformations=[
                {"type": "sort", "field": "timestamp"},
                {"type": "aggregate", "interval": "auto"}
            ],
            examples=[]
        )
        
        # Performance dashboard template
        self.templates["performance_dashboard"] = VisualizationTemplate(
            template_id="performance_dashboard",
            name="Performance Dashboard",
            description="Multi-metric performance visualization",
            chart_type=ChartType.LINE,
            default_config=ChartConfig(
                chart_type=ChartType.LINE,
                title="Performance Metrics",
                height=600,
                show_data_labels=False
            ),
            required_fields=["metric_name", "timestamp", "value"],
            optional_fields=["threshold", "target"],
            transformations=[
                {"type": "normalize", "method": "minmax"},
                {"type": "moving_average", "window": 5}
            ],
            examples=[]
        )
        
        # Distribution template
        self.templates["distribution"] = VisualizationTemplate(
            template_id="distribution",
            name="Distribution Chart",
            description="Data distribution visualization",
            chart_type=ChartType.HISTOGRAM,
            default_config=ChartConfig(
                chart_type=ChartType.HISTOGRAM,
                title="Distribution",
                show_data_labels=True
            ),
            required_fields=["values"],
            optional_fields=["bins", "labels"],
            transformations=[
                {"type": "histogram", "bins": "auto"}
            ],
            examples=[]
        )
    
    async def create_chart(self, chart_type: ChartType, title: str,
                          data: Union[List[DataSeries], pd.DataFrame, List[Dict[str, Any]]],
                          config: Optional[ChartConfig] = None) -> ChartDefinition:
        """Create a new chart"""
        try:
            # Create default config if not provided
            if not config:
                config = ChartConfig(chart_type=chart_type, title=title)
            
            # Convert data to DataSeries format
            if isinstance(data, pd.DataFrame):
                series = self._dataframe_to_series(data)
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                series = self._dict_list_to_series(data)
            else:
                series = data
            
            # Create chart definition
            chart = ChartDefinition(
                config=config,
                series=series,
                x_axis=ChartAxis(title="X Axis", type="linear"),
                y_axis=ChartAxis(title="Y Axis", type="linear")
            )
            
            # Store chart
            chart_id = f"chart_{datetime.now().timestamp()}"
            self.charts[chart_id] = chart
            self.stats["charts_created"] += 1
            
            logger.info(f"Created chart: {title} ({chart_type.value})")
            
            return chart
            
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            raise
    
    def _dataframe_to_series(self, df: pd.DataFrame) -> List[DataSeries]:
        """Convert DataFrame to DataSeries"""
        series_list = []
        
        for column in df.columns:
            if column != df.index.name:
                series = DataSeries(
                    name=column,
                    data=df[column].tolist(),
                    labels=[str(idx) for idx in df.index]
                )
                series_list.append(series)
        
        return series_list
    
    def _dict_list_to_series(self, data: List[Dict[str, Any]]) -> List[DataSeries]:
        """Convert list of dictionaries to DataSeries"""
        if not data:
            return []
        
        # Extract keys
        keys = list(data[0].keys())
        
        # Assume first key is x-axis
        x_key = keys[0]
        labels = [str(item[x_key]) for item in data]
        
        series_list = []
        for key in keys[1:]:
            values = [item.get(key, 0) for item in data]
            series = DataSeries(
                name=key,
                data=values,
                labels=labels
            )
            series_list.append(series)
        
        return series_list
    
    async def render_chart(self, chart: ChartDefinition, 
                          format: str = "json") -> Union[Dict[str, Any], str]:
        """Render chart to specified format"""
        try:
            self.stats["charts_rendered"] += 1
            
            # Check cache
            cache_key = f"{id(chart)}_{format}"
            if cache_key in self.render_cache:
                cached = self.render_cache[cache_key]
                if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return cached["data"]
            
            self.stats["cache_misses"] += 1
            
            # Render based on format
            if format == "json":
                rendered = await self._render_to_json(chart)
            elif format == "html":
                rendered = await self._render_to_html(chart)
            elif format == "svg":
                rendered = await self._render_to_svg(chart)
            elif format == "png":
                rendered = await self._render_to_png(chart)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Cache result
            self.render_cache[cache_key] = {
                "data": rendered,
                "timestamp": datetime.now()
            }
            
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering chart: {e}")
            raise
    
    async def _render_to_json(self, chart: ChartDefinition) -> Dict[str, Any]:
        """Render chart to JSON format"""
        colors = self._get_colors(chart.config.color_scheme, len(chart.series))
        
        # Build chart specification
        spec = {
            "type": chart.config.chart_type.value,
            "title": {
                "text": chart.config.title,
                "subtext": chart.config.subtitle
            },
            "legend": {
                "show": chart.config.show_legend
            },
            "grid": {
                "show": chart.config.show_grid,
                "top": chart.config.margin["top"],
                "right": chart.config.margin["right"],
                "bottom": chart.config.margin["bottom"],
                "left": chart.config.margin["left"]
            },
            "tooltip": {
                "show": chart.config.show_tooltips
            },
            "animation": {
                "enabled": chart.config.interaction_mode in [InteractionMode.ANIMATED, InteractionMode.REAL_TIME],
                "duration": chart.config.animation_duration
            },
            "series": [],
            "xAxis": {},
            "yAxis": {},
            "colors": colors
        }
        
        # Add series data
        for i, series in enumerate(chart.series):
            series_spec = {
                "name": series.name,
                "type": series.type or chart.config.chart_type.value,
                "data": series.data,
                "color": series.color or colors[i % len(colors)],
                "visible": series.visible
            }
            
            if series.labels:
                series_spec["labels"] = series.labels
            
            spec["series"].append(series_spec)
        
        # Configure axes
        if chart.x_axis:
            spec["xAxis"] = {
                "title": {"text": chart.x_axis.title},
                "type": chart.x_axis.type,
                "min": chart.x_axis.min_value,
                "max": chart.x_axis.max_value,
                "gridLines": {"show": chart.x_axis.grid_lines}
            }
        
        if chart.y_axis:
            spec["yAxis"] = {
                "title": {"text": chart.y_axis.title},
                "type": chart.y_axis.type,
                "min": chart.y_axis.min_value,
                "max": chart.y_axis.max_value,
                "gridLines": {"show": chart.y_axis.grid_lines}
            }
        
        # Add annotations
        if chart.annotations:
            spec["annotations"] = [
                {
                    "type": ann.type,
                    "text": ann.text,
                    "x": ann.x,
                    "y": ann.y,
                    "x2": ann.x2,
                    "y2": ann.y2,
                    "color": ann.color,
                    "style": ann.style
                }
                for ann in chart.annotations
            ]
        
        return spec
    
    async def _render_to_html(self, chart: ChartDefinition) -> str:
        """Render chart to HTML with embedded JavaScript"""
        json_spec = await self._render_to_json(chart)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{chart.config.title}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
            <style>
                body {{ font-family: {chart.config.font_family}; }}
                #chart {{ width: {chart.config.width}px; height: {chart.config.height}px; }}
            </style>
        </head>
        <body>
            <div id="chart"></div>
            <script>
                const chartSpec = {json.dumps(json_spec)};
                const container = document.getElementById('chart');
                const chart = echarts.init(container);
                
                // Convert spec to ECharts format
                const option = {{
                    title: chartSpec.title,
                    legend: chartSpec.legend,
                    grid: chartSpec.grid,
                    tooltip: chartSpec.tooltip,
                    xAxis: chartSpec.xAxis,
                    yAxis: chartSpec.yAxis,
                    series: chartSpec.series,
                    color: chartSpec.colors
                }};
                
                chart.setOption(option);
                
                // Handle resize
                window.addEventListener('resize', () => {{
                    chart.resize();
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    async def _render_to_svg(self, chart: ChartDefinition) -> str:
        """Render chart to SVG format"""
        # Simplified SVG rendering (would use proper library in production)
        width = chart.config.width
        height = chart.config.height
        margin = chart.config.margin
        
        svg = f"""
        <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="{width}" height="{height}" fill="white"/>
            <text x="{width/2}" y="{margin['top']/2}" text-anchor="middle" 
                  font-size="{chart.config.font_size + 4}" font-weight="bold">
                {chart.config.title}
            </text>
        """
        
        # Add basic chart elements based on type
        if chart.config.chart_type == ChartType.LINE:
            svg += await self._render_line_chart_svg(chart, width, height, margin)
        elif chart.config.chart_type == ChartType.BAR:
            svg += await self._render_bar_chart_svg(chart, width, height, margin)
        
        svg += "</svg>"
        
        return svg
    
    async def _render_line_chart_svg(self, chart: ChartDefinition, 
                                    width: int, height: int, 
                                    margin: Dict[str, int]) -> str:
        """Render line chart as SVG elements"""
        svg_elements = ""
        
        chart_width = width - margin["left"] - margin["right"]
        chart_height = height - margin["top"] - margin["bottom"]
        
        colors = self._get_colors(chart.config.color_scheme, len(chart.series))
        
        for series_idx, series in enumerate(chart.series):
            if not series.visible or not series.data:
                continue
            
            color = series.color or colors[series_idx % len(colors)]
            
            # Calculate points
            points = []
            for i, value in enumerate(series.data):
                if isinstance(value, (int, float)):
                    x = margin["left"] + (i / (len(series.data) - 1)) * chart_width
                    y = margin["top"] + chart_height - (value / 100) * chart_height  # Simplified scaling
                    points.append(f"{x},{y}")
            
            if points:
                svg_elements += f"""
                <polyline points="{' '.join(points)}" 
                         fill="none" stroke="{color}" stroke-width="2"/>
                """
        
        return svg_elements
    
    async def _render_bar_chart_svg(self, chart: ChartDefinition,
                                   width: int, height: int,
                                   margin: Dict[str, int]) -> str:
        """Render bar chart as SVG elements"""
        svg_elements = ""
        
        chart_width = width - margin["left"] - margin["right"]
        chart_height = height - margin["top"] - margin["bottom"]
        
        colors = self._get_colors(chart.config.color_scheme, len(chart.series))
        
        # Calculate bar dimensions
        total_bars = sum(len(s.data) for s in chart.series if s.visible)
        if total_bars == 0:
            return ""
        
        bar_width = chart_width / total_bars * 0.8
        bar_spacing = chart_width / total_bars * 0.2
        
        x_position = margin["left"]
        
        for series_idx, series in enumerate(chart.series):
            if not series.visible:
                continue
            
            color = series.color or colors[series_idx % len(colors)]
            
            for value in series.data:
                if isinstance(value, (int, float)):
                    bar_height = (value / 100) * chart_height  # Simplified scaling
                    y_position = margin["top"] + chart_height - bar_height
                    
                    svg_elements += f"""
                    <rect x="{x_position}" y="{y_position}" 
                          width="{bar_width}" height="{bar_height}"
                          fill="{color}"/>
                    """
                    
                    x_position += bar_width + bar_spacing
        
        return svg_elements
    
    async def _render_to_png(self, chart: ChartDefinition) -> str:
        """Render chart to PNG format (base64 encoded)"""
        # This would use a proper rendering library like matplotlib or Pillow
        # For now, return a placeholder
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def _get_colors(self, scheme: ColorScheme, count: int) -> List[str]:
        """Get colors for chart based on scheme"""
        palette = self.color_palettes.get(scheme, self.color_palettes[ColorScheme.DEFAULT])
        
        if count <= len(palette):
            return palette[:count]
        
        # Repeat colors if needed
        colors = []
        for i in range(count):
            colors.append(palette[i % len(palette)])
        
        return colors
    
    async def create_from_template(self, template_id: str, 
                                  data: Union[pd.DataFrame, List[Dict[str, Any]]],
                                  customizations: Dict[str, Any] = None) -> ChartDefinition:
        """Create chart from template"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            self.stats["templates_used"] += 1
            
            # Apply template transformations to data
            transformed_data = await self._apply_transformations(data, template.transformations)
            
            # Create config from template
            config = template.default_config
            
            # Apply customizations
            if customizations:
                if "title" in customizations:
                    config.title = customizations["title"]
                if "subtitle" in customizations:
                    config.subtitle = customizations["subtitle"]
                if "color_scheme" in customizations:
                    config.color_scheme = ColorScheme(customizations["color_scheme"])
            
            # Create chart
            return await self.create_chart(
                chart_type=template.chart_type,
                title=config.title,
                data=transformed_data,
                config=config
            )
            
        except Exception as e:
            logger.error(f"Error creating chart from template: {e}")
            raise
    
    async def _apply_transformations(self, data: Union[pd.DataFrame, List[Dict[str, Any]]],
                                    transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply data transformations"""
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        for transform in transformations:
            transform_type = transform.get("type")
            
            if transform_type == "sort":
                field = transform.get("field")
                if field in df.columns:
                    df = df.sort_values(by=field)
            
            elif transform_type == "aggregate":
                interval = transform.get("interval", "auto")
                if interval == "auto":
                    # Determine interval based on data range
                    interval = "1H"
                # Aggregate would be implemented here
            
            elif transform_type == "normalize":
                method = transform.get("method", "minmax")
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[col] = self.transformer.normalize_data(df[col].tolist(), method)
            
            elif transform_type == "moving_average":
                window = transform.get("window", 5)
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[f"{col}_ma"] = self.transformer.calculate_moving_average(df[col].tolist(), window)
            
            elif transform_type == "histogram":
                bins = transform.get("bins", "auto")
                # Histogram transformation would be implemented here
        
        return df
    
    async def create_dashboard(self, title: str, 
                             charts: List[ChartDefinition],
                             layout: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a dashboard with multiple charts"""
        try:
            dashboard = {
                "id": f"dashboard_{datetime.now().timestamp()}",
                "title": title,
                "charts": [],
                "layout": layout or {
                    "type": "grid",
                    "columns": 2,
                    "row_height": 400
                },
                "created_at": datetime.now().isoformat()
            }
            
            # Render all charts
            for chart in charts:
                rendered = await self.render_chart(chart, format="json")
                dashboard["charts"].append(rendered)
            
            logger.info(f"Created dashboard: {title} with {len(charts)} charts")
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    async def create_real_time_chart(self, chart: ChartDefinition,
                                    update_interval: int = 1000) -> str:
        """Create real-time updating chart"""
        try:
            chart_id = f"realtime_{datetime.now().timestamp()}"
            
            # Create update task
            async def update_chart():
                while chart_id in self.real_time_charts:
                    try:
                        # Update chart data (would fetch new data here)
                        await asyncio.sleep(update_interval / 1000)
                        
                        # Emit update event (would use WebSocket in production)
                        logger.debug(f"Updated real-time chart: {chart_id}")
                        
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Error updating real-time chart: {e}")
            
            # Start update task
            task = asyncio.create_task(update_chart())
            self.real_time_charts[chart_id] = task
            
            # Store chart
            self.charts[chart_id] = chart
            
            logger.info(f"Created real-time chart: {chart_id}")
            
            return chart_id
            
        except Exception as e:
            logger.error(f"Error creating real-time chart: {e}")
            raise
    
    async def stop_real_time_chart(self, chart_id: str) -> bool:
        """Stop real-time chart updates"""
        try:
            if chart_id in self.real_time_charts:
                task = self.real_time_charts[chart_id]
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                del self.real_time_charts[chart_id]
                logger.info(f"Stopped real-time chart: {chart_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error stopping real-time chart: {e}")
            return False
    
    def add_annotation(self, chart: ChartDefinition, annotation: Annotation) -> None:
        """Add annotation to chart"""
        chart.annotations.append(annotation)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get visualization engine statistics"""
        return {
            **self.stats,
            "active_charts": len(self.charts),
            "active_templates": len(self.templates),
            "real_time_charts": len(self.real_time_charts),
            "cache_size": len(self.render_cache)
        }
    
    async def export_chart(self, chart: ChartDefinition, 
                         filename: str, format: str = "png") -> bool:
        """Export chart to file"""
        try:
            rendered = await self.render_chart(chart, format)
            
            if format in ["json", "html", "svg"]:
                # Text formats
                with open(filename, "w") as f:
                    if format == "json":
                        json.dump(rendered, f, indent=2)
                    else:
                        f.write(rendered)
            
            elif format == "png":
                # Binary format (base64 decoded)
                if rendered.startswith("data:image/png;base64,"):
                    image_data = base64.b64decode(rendered.split(",")[1])
                    with open(filename, "wb") as f:
                        f.write(image_data)
            
            logger.info(f"Exported chart to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown visualization engine"""
        logger.info("Shutting down visualization engine...")
        
        # Stop all real-time charts
        for chart_id in list(self.real_time_charts.keys()):
            await self.stop_real_time_chart(chart_id)
        
        # Clear cache
        self.render_cache.clear()
        
        logger.info("Visualization engine shutdown complete")