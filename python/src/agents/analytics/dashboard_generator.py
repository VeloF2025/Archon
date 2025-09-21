"""
Dashboard Generator Module
Multi-widget dashboard composition and management
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict

from .analytics_engine import AnalyticsEngine
from .visualization_engine import VisualizationEngine, ChartType, ChartConfig


class WidgetType(Enum):
    """Dashboard widget types"""
    CHART = "chart"
    METRIC = "metric"
    TABLE = "table"
    GAUGE = "gauge"
    MAP = "map"
    ALERT = "alert"
    TEXT = "text"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    PROGRESS = "progress"
    STATUS = "status"
    LOG = "log"
    TERMINAL = "terminal"
    MARKDOWN = "markdown"
    CODE = "code"


class LayoutType(Enum):
    """Dashboard layout types"""
    GRID = "grid"
    FLEX = "flex"
    ABSOLUTE = "absolute"
    RESPONSIVE = "responsive"
    MASONRY = "masonry"
    CAROUSEL = "carousel"
    TABS = "tabs"
    ACCORDION = "accordion"


class RefreshStrategy(Enum):
    """Widget refresh strategies"""
    MANUAL = "manual"
    INTERVAL = "interval"
    REALTIME = "realtime"
    ON_DEMAND = "on_demand"
    EVENT_DRIVEN = "event_driven"


@dataclass
class WidgetPosition:
    """Widget position in dashboard"""
    x: int
    y: int
    width: int
    height: int
    z_index: int = 0
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    resizable: bool = True
    draggable: bool = True


@dataclass
class WidgetConfig:
    """Widget configuration"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: WidgetPosition
    refresh_strategy: RefreshStrategy = RefreshStrategy.INTERVAL
    refresh_interval: int = 60000  # milliseconds
    data_source: Optional[str] = None
    query: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    styling: Dict[str, Any] = field(default_factory=dict)
    interactions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardTemplate:
    """Pre-defined dashboard template"""
    template_id: str
    name: str
    description: str
    category: str
    widgets: List[WidgetConfig]
    layout_type: LayoutType
    variables: Dict[str, Any]
    tags: List[str]
    thumbnail: Optional[str] = None
    preview_url: Optional[str] = None


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    owner: str
    created_at: datetime
    updated_at: datetime
    widgets: List[WidgetConfig]
    layout_type: LayoutType = LayoutType.GRID
    theme: str = "default"
    variables: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    auto_refresh: bool = True
    refresh_interval: int = 30000
    tags: List[str] = field(default_factory=list)
    version: int = 1
    is_public: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DashboardGenerator:
    """
    Advanced dashboard generation and management system
    """
    
    def __init__(self, analytics_engine: AnalyticsEngine, 
                 visualization_engine: VisualizationEngine):
        self.analytics_engine = analytics_engine
        self.visualization_engine = visualization_engine
        self.dashboards: Dict[str, Dashboard] = {}
        self.templates: Dict[str, DashboardTemplate] = {}
        self.widget_registry: Dict[str, WidgetConfig] = {}
        self.active_sessions: Dict[str, Set[str]] = defaultdict(set)
        self.widget_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        self._init_default_templates()
        
    def _init_default_templates(self):
        """Initialize default dashboard templates"""
        # System Performance Dashboard
        self.templates["system_performance"] = DashboardTemplate(
            template_id="system_performance",
            name="System Performance",
            description="Monitor system metrics and performance",
            category="monitoring",
            widgets=[
                WidgetConfig(
                    widget_id="cpu_usage",
                    widget_type=WidgetType.GAUGE,
                    title="CPU Usage",
                    position=WidgetPosition(0, 0, 4, 2),
                    data_source="system.cpu",
                    refresh_strategy=RefreshStrategy.REALTIME
                ),
                WidgetConfig(
                    widget_id="memory_usage",
                    widget_type=WidgetType.GAUGE,
                    title="Memory Usage",
                    position=WidgetPosition(4, 0, 4, 2),
                    data_source="system.memory",
                    refresh_strategy=RefreshStrategy.REALTIME
                ),
                WidgetConfig(
                    widget_id="disk_io",
                    widget_type=WidgetType.CHART,
                    title="Disk I/O",
                    position=WidgetPosition(8, 0, 4, 2),
                    data_source="system.disk_io",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="network_traffic",
                    widget_type=WidgetType.CHART,
                    title="Network Traffic",
                    position=WidgetPosition(0, 2, 6, 3),
                    data_source="system.network",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="process_list",
                    widget_type=WidgetType.TABLE,
                    title="Top Processes",
                    position=WidgetPosition(6, 2, 6, 3),
                    data_source="system.processes",
                    refresh_strategy=RefreshStrategy.INTERVAL
                )
            ],
            layout_type=LayoutType.GRID,
            variables={},
            tags=["system", "monitoring", "performance"]
        )
        
        # AI Agent Performance Dashboard
        self.templates["agent_performance"] = DashboardTemplate(
            template_id="agent_performance",
            name="AI Agent Performance",
            description="Monitor AI agent execution and performance",
            category="ai",
            widgets=[
                WidgetConfig(
                    widget_id="agent_status",
                    widget_type=WidgetType.STATUS,
                    title="Agent Status",
                    position=WidgetPosition(0, 0, 12, 1),
                    data_source="agents.status",
                    refresh_strategy=RefreshStrategy.REALTIME
                ),
                WidgetConfig(
                    widget_id="execution_timeline",
                    widget_type=WidgetType.TIMELINE,
                    title="Execution Timeline",
                    position=WidgetPosition(0, 1, 8, 3),
                    data_source="agents.executions",
                    refresh_strategy=RefreshStrategy.REALTIME
                ),
                WidgetConfig(
                    widget_id="success_rate",
                    widget_type=WidgetType.METRIC,
                    title="Success Rate",
                    position=WidgetPosition(8, 1, 4, 1),
                    data_source="agents.success_rate",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="avg_duration",
                    widget_type=WidgetType.METRIC,
                    title="Avg Duration",
                    position=WidgetPosition(8, 2, 4, 1),
                    data_source="agents.duration",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="error_log",
                    widget_type=WidgetType.LOG,
                    title="Error Log",
                    position=WidgetPosition(0, 4, 12, 2),
                    data_source="agents.errors",
                    refresh_strategy=RefreshStrategy.REALTIME
                )
            ],
            layout_type=LayoutType.GRID,
            variables={},
            tags=["ai", "agents", "monitoring"]
        )
        
        # Code Quality Dashboard
        self.templates["code_quality"] = DashboardTemplate(
            template_id="code_quality",
            name="Code Quality Metrics",
            description="Track code quality and technical debt",
            category="development",
            widgets=[
                WidgetConfig(
                    widget_id="coverage_trend",
                    widget_type=WidgetType.CHART,
                    title="Test Coverage Trend",
                    position=WidgetPosition(0, 0, 6, 3),
                    data_source="quality.coverage",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="complexity_heatmap",
                    widget_type=WidgetType.HEATMAP,
                    title="Code Complexity",
                    position=WidgetPosition(6, 0, 6, 3),
                    data_source="quality.complexity",
                    refresh_strategy=RefreshStrategy.ON_DEMAND
                ),
                WidgetConfig(
                    widget_id="tech_debt",
                    widget_type=WidgetType.PROGRESS,
                    title="Technical Debt",
                    position=WidgetPosition(0, 3, 4, 2),
                    data_source="quality.debt",
                    refresh_strategy=RefreshStrategy.INTERVAL
                ),
                WidgetConfig(
                    widget_id="code_smells",
                    widget_type=WidgetType.TABLE,
                    title="Code Smells",
                    position=WidgetPosition(4, 3, 8, 2),
                    data_source="quality.smells",
                    refresh_strategy=RefreshStrategy.ON_DEMAND
                )
            ],
            layout_type=LayoutType.GRID,
            variables={},
            tags=["quality", "development", "metrics"]
        )
    
    async def create_dashboard(self, name: str, description: str, owner: str,
                             template_id: Optional[str] = None,
                             layout_type: LayoutType = LayoutType.GRID) -> Dashboard:
        """Create a new dashboard"""
        dashboard_id = str(uuid.uuid4())
        
        # Use template if provided
        widgets = []
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            widgets = template.widgets.copy()
            layout_type = template.layout_type
        
        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            owner=owner,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            widgets=widgets,
            layout_type=layout_type
        )
        
        self.dashboards[dashboard_id] = dashboard
        return dashboard
    
    async def add_widget(self, dashboard_id: str, widget_config: WidgetConfig) -> bool:
        """Add widget to dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        # Check for position conflicts
        for existing_widget in dashboard.widgets:
            if self._widgets_overlap(widget_config.position, existing_widget.position):
                # Adjust position to avoid overlap
                widget_config.position = self._find_free_position(
                    dashboard, widget_config.position
                )
        
        dashboard.widgets.append(widget_config)
        dashboard.updated_at = datetime.now()
        dashboard.version += 1
        
        # Register widget
        self.widget_registry[widget_config.widget_id] = widget_config
        
        return True
    
    async def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """Remove widget from dashboard"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
        dashboard.updated_at = datetime.now()
        dashboard.version += 1
        
        # Unregister widget
        if widget_id in self.widget_registry:
            del self.widget_registry[widget_id]
        
        return True
    
    async def update_widget(self, dashboard_id: str, widget_id: str,
                          updates: Dict[str, Any]) -> bool:
        """Update widget configuration"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        for widget in dashboard.widgets:
            if widget.widget_id == widget_id:
                # Update widget properties
                for key, value in updates.items():
                    if hasattr(widget, key):
                        setattr(widget, key, value)
                
                dashboard.updated_at = datetime.now()
                dashboard.version += 1
                return True
        
        return False
    
    async def render_dashboard(self, dashboard_id: str,
                             variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Render complete dashboard with all widgets"""
        if dashboard_id not in self.dashboards:
            return {}
        
        dashboard = self.dashboards[dashboard_id]
        rendered_widgets = []
        
        # Merge variables
        effective_variables = {**dashboard.variables, **(variables or {})}
        
        # Render each widget
        for widget in dashboard.widgets:
            widget_data = await self._render_widget(widget, effective_variables)
            rendered_widgets.append({
                "widget_id": widget.widget_id,
                "type": widget.widget_type.value,
                "title": widget.title,
                "position": {
                    "x": widget.position.x,
                    "y": widget.position.y,
                    "width": widget.position.width,
                    "height": widget.position.height
                },
                "data": widget_data,
                "styling": widget.styling,
                "interactions": widget.interactions
            })
        
        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "layout_type": dashboard.layout_type.value,
            "theme": dashboard.theme,
            "widgets": rendered_widgets,
            "updated_at": dashboard.updated_at.isoformat(),
            "version": dashboard.version
        }
    
    async def _render_widget(self, widget: WidgetConfig,
                           variables: Dict[str, Any]) -> Any:
        """Render individual widget"""
        # Check cache
        cache_key = self._get_widget_cache_key(widget, variables)
        if cache_key in self.widget_cache:
            cached_data, timestamp = self.widget_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        # Fetch data based on widget type
        widget_data = None
        
        if widget.widget_type == WidgetType.CHART:
            # Generate chart using visualization engine
            chart_config = ChartConfig(
                chart_type=ChartType.LINE,  # Default, should be configurable
                title=widget.title,
                data=[],  # Fetch from data source
                options=widget.styling
            )
            widget_data = await self.visualization_engine.create_chart(
                "line", chart_config
            )
        
        elif widget.widget_type == WidgetType.METRIC:
            # Fetch single metric value
            if widget.data_source:
                metrics = await self.analytics_engine.get_metrics(
                    metric_names=[widget.data_source],
                    limit=1
                )
                if metrics:
                    widget_data = {
                        "value": metrics[0]["value"],
                        "unit": metrics[0].get("unit", ""),
                        "trend": self._calculate_trend(widget.data_source)
                    }
        
        elif widget.widget_type == WidgetType.TABLE:
            # Fetch tabular data
            if widget.query:
                widget_data = await self._execute_query(widget.query, variables)
        
        elif widget.widget_type == WidgetType.GAUGE:
            # Fetch gauge data
            if widget.data_source:
                metrics = await self.analytics_engine.get_metrics(
                    metric_names=[widget.data_source],
                    limit=1
                )
                if metrics:
                    widget_data = {
                        "value": metrics[0]["value"],
                        "min": widget.metadata.get("min", 0),
                        "max": widget.metadata.get("max", 100),
                        "thresholds": widget.metadata.get("thresholds", [])
                    }
        
        elif widget.widget_type == WidgetType.HEATMAP:
            # Generate heatmap data
            if widget.data_source:
                widget_data = await self._generate_heatmap_data(
                    widget.data_source, variables
                )
        
        elif widget.widget_type == WidgetType.TIMELINE:
            # Fetch timeline events
            if widget.data_source:
                widget_data = await self._fetch_timeline_events(
                    widget.data_source, variables
                )
        
        elif widget.widget_type == WidgetType.STATUS:
            # Fetch status information
            if widget.data_source:
                widget_data = await self._fetch_status(
                    widget.data_source, variables
                )
        
        elif widget.widget_type == WidgetType.LOG:
            # Fetch log entries
            if widget.data_source:
                widget_data = await self._fetch_logs(
                    widget.data_source, 
                    widget.metadata.get("limit", 100)
                )
        
        # Cache the result
        self.widget_cache[cache_key] = (widget_data, datetime.now())
        
        return widget_data
    
    def _widgets_overlap(self, pos1: WidgetPosition, pos2: WidgetPosition) -> bool:
        """Check if two widgets overlap"""
        return not (
            pos1.x + pos1.width <= pos2.x or
            pos2.x + pos2.width <= pos1.x or
            pos1.y + pos1.height <= pos2.y or
            pos2.y + pos2.height <= pos1.y
        )
    
    def _find_free_position(self, dashboard: Dashboard,
                          desired_pos: WidgetPosition) -> WidgetPosition:
        """Find free position for widget"""
        # Simple strategy: move to the right until free space found
        new_pos = WidgetPosition(
            x=desired_pos.x,
            y=desired_pos.y,
            width=desired_pos.width,
            height=desired_pos.height
        )
        
        while any(self._widgets_overlap(new_pos, w.position) 
                 for w in dashboard.widgets):
            new_pos.x += desired_pos.width
            if new_pos.x + new_pos.width > 12:  # Assuming 12-column grid
                new_pos.x = 0
                new_pos.y += desired_pos.height
        
        return new_pos
    
    def _get_widget_cache_key(self, widget: WidgetConfig,
                            variables: Dict[str, Any]) -> str:
        """Generate cache key for widget"""
        key_data = {
            "widget_id": widget.widget_id,
            "data_source": widget.data_source,
            "query": widget.query,
            "filters": widget.filters,
            "variables": variables
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate metric trend"""
        # Simplified trend calculation
        # In production, would analyze historical data
        return "up"  # or "down", "stable"
    
    async def _execute_query(self, query: str,
                           variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute data query"""
        # Placeholder for query execution
        # Would integrate with actual data sources
        return []
    
    async def _generate_heatmap_data(self, data_source: str,
                                   variables: Dict[str, Any]) -> Dict[str, Any]:
        """Generate heatmap data"""
        # Placeholder for heatmap data generation
        return {
            "data": [],
            "x_labels": [],
            "y_labels": []
        }
    
    async def _fetch_timeline_events(self, data_source: str,
                                   variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch timeline events"""
        # Placeholder for timeline event fetching
        return []
    
    async def _fetch_status(self, data_source: str,
                         variables: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch status information"""
        # Placeholder for status fetching
        return {
            "status": "healthy",
            "components": []
        }
    
    async def _fetch_logs(self, data_source: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch log entries"""
        # Placeholder for log fetching
        return []
    
    async def export_dashboard(self, dashboard_id: str,
                             format: str = "json") -> str:
        """Export dashboard configuration"""
        if dashboard_id not in self.dashboards:
            return ""
        
        dashboard = self.dashboards[dashboard_id]
        
        if format == "json":
            export_data = {
                "name": dashboard.name,
                "description": dashboard.description,
                "layout_type": dashboard.layout_type.value,
                "theme": dashboard.theme,
                "widgets": [
                    {
                        "widget_id": w.widget_id,
                        "widget_type": w.widget_type.value,
                        "title": w.title,
                        "position": {
                            "x": w.position.x,
                            "y": w.position.y,
                            "width": w.position.width,
                            "height": w.position.height
                        },
                        "data_source": w.data_source,
                        "query": w.query,
                        "filters": w.filters,
                        "styling": w.styling
                    }
                    for w in dashboard.widgets
                ],
                "variables": dashboard.variables,
                "tags": dashboard.tags
            }
            return json.dumps(export_data, indent=2)
        
        return ""
    
    async def import_dashboard(self, config: str, owner: str) -> Optional[Dashboard]:
        """Import dashboard from configuration"""
        try:
            data = json.loads(config)
            
            dashboard = await self.create_dashboard(
                name=data["name"],
                description=data["description"],
                owner=owner,
                layout_type=LayoutType(data["layout_type"])
            )
            
            # Add widgets
            for widget_data in data["widgets"]:
                widget = WidgetConfig(
                    widget_id=widget_data["widget_id"],
                    widget_type=WidgetType(widget_data["widget_type"]),
                    title=widget_data["title"],
                    position=WidgetPosition(**widget_data["position"]),
                    data_source=widget_data.get("data_source"),
                    query=widget_data.get("query"),
                    filters=widget_data.get("filters", {}),
                    styling=widget_data.get("styling", {})
                )
                await self.add_widget(dashboard.dashboard_id, widget)
            
            # Set variables and tags
            dashboard.variables = data.get("variables", {})
            dashboard.tags = data.get("tags", [])
            dashboard.theme = data.get("theme", "default")
            
            return dashboard
            
        except Exception as e:
            print(f"Failed to import dashboard: {e}")
            return None
    
    async def share_dashboard(self, dashboard_id: str, users: List[str],
                           permissions: List[str]) -> bool:
        """Share dashboard with users"""
        if dashboard_id not in self.dashboards:
            return False
        
        dashboard = self.dashboards[dashboard_id]
        
        for user in users:
            dashboard.permissions[user] = permissions
        
        dashboard.updated_at = datetime.now()
        return True
    
    async def clone_dashboard(self, dashboard_id: str, new_owner: str,
                            new_name: Optional[str] = None) -> Optional[Dashboard]:
        """Clone an existing dashboard"""
        if dashboard_id not in self.dashboards:
            return None
        
        original = self.dashboards[dashboard_id]
        
        new_dashboard = await self.create_dashboard(
            name=new_name or f"{original.name} (Copy)",
            description=original.description,
            owner=new_owner,
            layout_type=original.layout_type
        )
        
        # Copy widgets
        for widget in original.widgets:
            new_widget = WidgetConfig(
                widget_id=str(uuid.uuid4()),
                widget_type=widget.widget_type,
                title=widget.title,
                position=WidgetPosition(
                    x=widget.position.x,
                    y=widget.position.y,
                    width=widget.position.width,
                    height=widget.position.height
                ),
                refresh_strategy=widget.refresh_strategy,
                refresh_interval=widget.refresh_interval,
                data_source=widget.data_source,
                query=widget.query,
                filters=widget.filters.copy(),
                styling=widget.styling.copy(),
                interactions=widget.interactions.copy(),
                metadata=widget.metadata.copy()
            )
            await self.add_widget(new_dashboard.dashboard_id, new_widget)
        
        # Copy other properties
        new_dashboard.theme = original.theme
        new_dashboard.variables = original.variables.copy()
        new_dashboard.tags = original.tags.copy()
        new_dashboard.auto_refresh = original.auto_refresh
        new_dashboard.refresh_interval = original.refresh_interval
        
        return new_dashboard
    
    def get_dashboard_list(self, owner: Optional[str] = None,
                         tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get list of dashboards"""
        dashboards = []
        
        for dashboard in self.dashboards.values():
            # Filter by owner if specified
            if owner and dashboard.owner != owner and owner not in dashboard.permissions:
                continue
            
            # Filter by tags if specified
            if tags and not any(tag in dashboard.tags for tag in tags):
                continue
            
            dashboards.append({
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "owner": dashboard.owner,
                "created_at": dashboard.created_at.isoformat(),
                "updated_at": dashboard.updated_at.isoformat(),
                "widget_count": len(dashboard.widgets),
                "tags": dashboard.tags,
                "is_public": dashboard.is_public
            })
        
        return dashboards
    
    def get_template_list(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of dashboard templates"""
        templates = []
        
        for template in self.templates.values():
            if category and template.category != category:
                continue
            
            templates.append({
                "template_id": template.template_id,
                "name": template.name,
                "description": template.description,
                "category": template.category,
                "widget_count": len(template.widgets),
                "tags": template.tags,
                "thumbnail": template.thumbnail,
                "preview_url": template.preview_url
            })
        
        return templates