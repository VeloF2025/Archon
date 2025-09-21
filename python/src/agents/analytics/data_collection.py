"""
Data Collection Pipeline
Real-time data collection for conversation analytics from various sources
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import asyncpg
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Types of data sources"""
    CONVERSATION_EVENTS = "conversation_events"
    AGENT_INTERACTIONS = "agent_interactions"
    SYSTEM_METRICS = "system_metrics"
    PERFORMANCE_LOGS = "performance_logs"
    KNOWLEDGE_BASE = "knowledge_base"
    EXTERNAL_APIS = "external_apis"
    DATABASE_QUERIES = "database_queries"
    FILE_SYSTEM = "file_system"
    MESSAGE_QUEUES = "message_queues"
    WEB_SOCKETS = "web_sockets"


class DataCollectionMode(Enum):
    """Collection modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"


class DataFormat(Enum):
    """Data formats"""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    XML = "xml"
    PROTOBUF = "protobuf"
    AVRO = "avro"


@dataclass
class DataCollectionConfig:
    """Configuration for data collection"""
    source_type: DataSource
    collection_mode: DataCollectionMode
    endpoint_url: Optional[str] = None
    database_config: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    api_key: Optional[str] = None
    collection_interval: int = 60  # seconds
    batch_size: int = 1000
    max_retries: int = 3
    timeout: int = 30
    format: DataFormat = DataFormat.JSON
    filters: Dict[str, Any] = field(default_factory=dict)
    transformations: List[str] = field(default_factory=list)


@dataclass
class CollectedData:
    """Represents collected data"""
    data_id: str
    source_type: DataSource
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    processing_time_ms: float = 0.0


@dataclass
class CollectionMetrics:
    """Metrics for data collection"""
    collector_id: str
    total_records_collected: int = 0
    collection_success_rate: float = 1.0
    avg_processing_time_ms: float = 0.0
    last_collection_time: Optional[datetime] = None
    error_count: int = 0
    data_quality_score: float = 1.0


class ConversationDataCollector:
    """
    Real-time data collection for conversation analytics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.collectors: Dict[str, 'DataCollector'] = {}
        self.data_buffer = deque(maxlen=10000)
        self.metrics: Dict[str, CollectionMetrics] = {}
        self.collection_configs: Dict[str, DataCollectionConfig] = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.session_pool = None

    async def initialize(self) -> None:
        """Initialize data collection system"""
        try:
            # Initialize database connection pool if configured
            if self.config.get('database'):
                self.session_pool = await asyncpg.create_pool(
                    **self.config['database'],
                    min_size=2,
                    max_size=10
                )

            # Initialize HTTP session
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            logger.info("Data collection system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing data collection system: {e}")
            raise

    async def register_collector(self, collector_id: str, config: DataCollectionConfig) -> None:
        """Register a new data collector"""
        try:
            collector = DataCollector(collector_id, config, self.http_session, self.session_pool)
            await collector.initialize()
            self.collectors[collector_id] = collector
            self.collection_configs[collector_id] = config
            self.metrics[collector_id] = CollectionMetrics(collector_id=collector_id)

            logger.info(f"Registered data collector: {collector_id}")

        except Exception as e:
            logger.error(f"Error registering collector {collector_id}: {e}")
            raise

    async def start_collection(self, collector_ids: Optional[List[str]] = None) -> None:
        """Start data collection for specified collectors"""
        if collector_ids is None:
            collector_ids = list(self.collectors.keys())

        self.is_running = True

        # Start collection tasks for each collector
        collection_tasks = []
        for collector_id in collector_ids:
            if collector_id in self.collectors:
                task = asyncio.create_task(self._run_collector(collector_id))
                collection_tasks.append(task)

        logger.info(f"Started data collection for {len(collection_tasks)} collectors")

        # Wait for all collection tasks
        await asyncio.gather(*collection_tasks, return_exceptions=True)

    async def stop_collection(self) -> None:
        """Stop all data collection"""
        self.is_running = False

        # Close HTTP session
        if hasattr(self, 'http_session'):
            await self.http_session.close()

        # Close database pool
        if self.session_pool:
            await self.session_pool.close()

        logger.info("Data collection stopped")

    async def _run_collector(self, collector_id: str) -> None:
        """Run a specific data collector"""
        collector = self.collectors.get(collector_id)
        if not collector:
            logger.error(f"Collector {collector_id} not found")
            return

        config = self.collection_configs[collector_id]
        metrics = self.metrics[collector_id]

        while self.is_running:
            try:
                # Collect data
                start_time = datetime.utcnow()
                collected_data = await collector.collect()

                # Process collected data
                for data in collected_data:
                    # Add to buffer
                    self.data_buffer.append(data)

                    # Update metrics
                    metrics.total_records_collected += 1
                    metrics.last_collection_time = datetime.utcnow()

                # Update metrics
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                metrics.avg_processing_time_ms = (
                    (metrics.avg_processing_time_ms * (metrics.total_records_collected - len(collected_data)) +
                     processing_time * len(collected_data)) / metrics.total_records_collected
                )

                # Calculate success rate
                successful_collections = len(collected_data)
                total_attempts = len(collected_data) + metrics.error_count
                metrics.collection_success_rate = successful_collections / max(total_attempts, 1)

                # Sleep based on collection interval
                await asyncio.sleep(config.collection_interval)

            except Exception as e:
                metrics.error_count += 1
                logger.error(f"Error in collector {collector_id}: {e}")
                await asyncio.sleep(config.collection_interval)

    async def get_recent_data(self, source_type: Optional[DataSource] = None,
                             time_range: Optional[Tuple[datetime, datetime]] = None,
                             limit: int = 100) -> List[CollectedData]:
        """Get recently collected data"""
        try:
            # Filter data buffer
            filtered_data = list(self.data_buffer)

            # Filter by source type
            if source_type:
                filtered_data = [d for d in filtered_data if d.source_type == source_type]

            # Filter by time range
            if time_range:
                start_time, end_time = time_range
                filtered_data = [d for d in filtered_data if start_time <= d.timestamp <= end_time]

            # Sort by timestamp and limit
            filtered_data.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_data[:limit]

        except Exception as e:
            logger.error(f"Error getting recent data: {e}")
            return []

    async def get_collection_metrics(self) -> Dict[str, CollectionMetrics]:
        """Get collection metrics for all collectors"""
        return self.metrics.copy()

    async def export_data(self, file_path: str, format: DataFormat = DataFormat.JSON,
                         filters: Optional[Dict[str, Any]] = None) -> bool:
        """Export collected data to file"""
        try:
            # Get data to export
            if filters:
                data = [d for d in self.data_buffer if self._matches_filters(d, filters)]
            else:
                data = list(self.data_buffer)

            if not data:
                logger.warning("No data to export")
                return False

            # Convert to DataFrame for easier export
            df_data = []
            for item in data:
                row = {
                    'data_id': item.data_id,
                    'source_type': item.source_type.value,
                    'timestamp': item.timestamp.isoformat(),
                    'data': json.dumps(item.data),
                    'quality_score': item.quality_score,
                    'processing_time_ms': item.processing_time_ms
                }
                row.update(item.metadata)
                df_data.append(row)

            df = pd.DataFrame(df_data)

            # Export based on format
            file_path_obj = Path(file_path)
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)

            if format == DataFormat.CSV:
                df.to_csv(file_path, index=False)
            elif format == DataFormat.JSON:
                df.to_json(file_path, orient='records', indent=2)
            elif format == DataFormat.PARQUET:
                df.to_parquet(file_path, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported {len(data)} records to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

    def _matches_filters(self, data: CollectedData, filters: Dict[str, Any]) -> bool:
        """Check if data matches filters"""
        for key, value in filters.items():
            if key == 'source_type' and data.source_type != value:
                return False
            elif key == 'min_quality_score' and data.quality_score < value:
                return False
            elif key in data.metadata and data.metadata[key] != value:
                return False

        return True


class DataCollector:
    """Individual data collector for a specific source"""

    def __init__(self, collector_id: str, config: DataCollectionConfig,
                 http_session: Optional[aiohttp.ClientSession] = None,
                 db_pool: Optional[asyncpg.Pool] = None):
        self.collector_id = collector_id
        self.config = config
        self.http_session = http_session
        self.db_pool = db_pool
        self.last_collection_time = None

    async def initialize(self) -> None:
        """Initialize the collector"""
        # Perform any collector-specific initialization
        pass

    async def collect(self) -> List[CollectedData]:
        """Collect data from the configured source"""
        start_time = datetime.utcnow()

        try:
            if self.config.source_type == DataSource.CONVERSATION_EVENTS:
                data = await self._collect_conversation_events()
            elif self.config.source_type == DataSource.AGENT_INTERACTIONS:
                data = await self._collect_agent_interactions()
            elif self.config.source_type == DataSource.SYSTEM_METRICS:
                data = await self._collect_system_metrics()
            elif self.config.source_type == DataSource.DATABASE_QUERIES:
                data = await self._collect_database_data()
            elif self.config.source_type == DataSource.EXTERNAL_APIS:
                data = await self._collect_api_data()
            elif self.config.source_type == DataSource.FILE_SYSTEM:
                data = await self._collect_file_data()
            elif self.config.source_type == DataSource.MESSAGE_QUEUES:
                data = await self._collect_queue_data()
            elif self.config.source_type == DataSource.WEB_SOCKETS:
                data = await self._collect_websocket_data()
            else:
                logger.warning(f"Unsupported source type: {self.config.source_type}")
                data = []

            # Apply transformations
            data = await self._apply_transformations(data)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            for item in data:
                item.processing_time_ms = processing_time

            self.last_collection_time = start_time
            return data

        except Exception as e:
            logger.error(f"Error collecting data for {self.collector_id}: {e}")
            return []

    async def _collect_conversation_events(self) -> List[CollectedData]:
        """Collect conversation events from various sources"""
        data = []

        try:
            # Simulate collecting conversation events
            # In a real implementation, this would connect to actual event sources
            sample_events = [
                {
                    'event_type': 'message',
                    'agent_id': 'agent_001',
                    'target_agent_id': 'agent_002',
                    'message': 'Working on the task now',
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': 'session_123'
                },
                {
                    'event_type': 'handoff',
                    'agent_id': 'agent_002',
                    'target_agent_id': 'agent_003',
                    'success': True,
                    'timestamp': datetime.utcnow().isoformat(),
                    'session_id': 'session_123'
                }
            ]

            for event in sample_events:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.CONVERSATION_EVENTS,
                    timestamp=datetime.utcnow(),
                    data=event,
                    metadata={'collection_method': 'simulation'},
                    quality_score=0.95
                ))

        except Exception as e:
            logger.error(f"Error collecting conversation events: {e}")

        return data

    async def _collect_agent_interactions(self) -> List[CollectedData]:
        """Collect agent interaction data"""
        data = []

        try:
            # Simulate agent interaction collection
            sample_interactions = [
                {
                    'interaction_type': 'collaboration',
                    'agent_a': 'agent_001',
                    'agent_b': 'agent_002',
                    'duration': 45.2,
                    'success': True,
                    'quality_score': 0.85
                }
            ]

            for interaction in sample_interactions:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.AGENT_INTERACTIONS,
                    timestamp=datetime.utcnow(),
                    data=interaction,
                    metadata={'collection_method': 'simulation'},
                    quality_score=0.90
                ))

        except Exception as e:
            logger.error(f"Error collecting agent interactions: {e}")

        return data

    async def _collect_system_metrics(self) -> List[CollectedData]:
        """Collect system metrics"""
        data = []

        try:
            # Simulate system metrics collection
            sample_metrics = [
                {
                    'metric_name': 'cpu_usage',
                    'value': 65.5,
                    'unit': 'percent',
                    'timestamp': datetime.utcnow().isoformat()
                },
                {
                    'metric_name': 'memory_usage',
                    'value': 2048,
                    'unit': 'mb',
                    'timestamp': datetime.utcnow().isoformat()
                }
            ]

            for metric in sample_metrics:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.SYSTEM_METRICS,
                    timestamp=datetime.utcnow(),
                    data=metric,
                    metadata={'collection_method': 'simulation'},
                    quality_score=0.98
                ))

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return data

    async def _collect_database_data(self) -> List[CollectedData]:
        """Collect data from database queries"""
        data = []

        try:
            if not self.db_pool:
                logger.warning("No database pool available for collector")
                return data

            # Execute database query
            async with self.db_pool.acquire() as conn:
                # This would be a real query in production
                query = """
                SELECT agent_id, session_id, event_type, timestamp,
                       extract(epoch from (now() - timestamp)) as age_seconds
                FROM conversation_events
                WHERE timestamp > now() - interval '1 hour'
                LIMIT 100
                """
                rows = await conn.fetch(query)

                for row in rows:
                    data.append(CollectedData(
                        data_id=str(uuid.uuid4()),
                        source_type=DataSource.DATABASE_QUERIES,
                        timestamp=datetime.utcnow(),
                        data=dict(row),
                        metadata={'query': query, 'collection_method': 'database'},
                        quality_score=0.95
                    ))

        except Exception as e:
            logger.error(f"Error collecting database data: {e}")

        return data

    async def _collect_api_data(self) -> List[CollectedData]:
        """Collect data from external APIs"""
        data = []

        try:
            if not self.http_session or not self.config.endpoint_url:
                logger.warning("No HTTP session or endpoint configured for API collector")
                return data

            # Make API request
            headers = {}
            if self.config.api_key:
                headers['Authorization'] = f"Bearer {self.config.api_key}"

            async with self.http_session.get(
                self.config.endpoint_url,
                headers=headers,
                timeout=self.config.timeout
            ) as response:
                if response.status == 200:
                    api_data = await response.json()

                    # Handle different response formats
                    if isinstance(api_data, list):
                        for item in api_data:
                            data.append(CollectedData(
                                data_id=str(uuid.uuid4()),
                                source_type=DataSource.EXTERNAL_APIS,
                                timestamp=datetime.utcnow(),
                                data=item,
                                metadata={'endpoint': self.config.endpoint_url, 'status_code': response.status},
                                quality_score=0.90
                            ))
                    else:
                        data.append(CollectedData(
                            data_id=str(uuid.uuid4()),
                            source_type=DataSource.EXTERNAL_APIS,
                            timestamp=datetime.utcnow(),
                            data=api_data,
                            metadata={'endpoint': self.config.endpoint_url, 'status_code': response.status},
                            quality_score=0.90
                        ))
                else:
                    logger.error(f"API request failed with status {response.status}")

        except Exception as e:
            logger.error(f"Error collecting API data: {e}")

        return data

    async def _collect_file_data(self) -> List[CollectedData]:
        """Collect data from file system"""
        data = []

        try:
            if not self.config.file_path:
                logger.warning("No file path configured for file collector")
                return data

            file_path = Path(self.config.file_path)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return data

            # Read file based on format
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    file_data = json.load(f)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                file_data = df.to_dict('records')
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return data

            # Create collected data items
            if isinstance(file_data, list):
                for item in file_data:
                    data.append(CollectedData(
                        data_id=str(uuid.uuid4()),
                        source_type=DataSource.FILE_SYSTEM,
                        timestamp=datetime.utcnow(),
                        data=item,
                        metadata={'file_path': str(file_path), 'file_size': file_path.stat().st_size},
                        quality_score=0.95
                    ))
            else:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.FILE_SYSTEM,
                    timestamp=datetime.utcnow(),
                    data=file_data,
                    metadata={'file_path': str(file_path), 'file_size': file_path.stat().st_size},
                    quality_score=0.95
                ))

        except Exception as e:
            logger.error(f"Error collecting file data: {e}")

        return data

    async def _collect_queue_data(self) -> List[CollectedData]:
        """Collect data from message queues"""
        data = []

        try:
            # Simulate message queue data collection
            sample_queue_data = [
                {
                    'queue_name': 'conversation_events',
                    'message_count': 1250,
                    'consumer_count': 5,
                    'avg_processing_time': 45.2
                }
            ]

            for queue_data in sample_queue_data:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.MESSAGE_QUEUES,
                    timestamp=datetime.utcnow(),
                    data=queue_data,
                    metadata={'collection_method': 'simulation'},
                    quality_score=0.92
                ))

        except Exception as e:
            logger.error(f"Error collecting queue data: {e}")

        return data

    async def _collect_websocket_data(self) -> List[CollectedData]:
        """Collect data from WebSocket connections"""
        data = []

        try:
            # Simulate WebSocket data collection
            sample_ws_data = [
                {
                    'connection_id': 'ws_123',
                    'event_type': 'message',
                    'payload': {'type': 'conversation_update', 'status': 'active'},
                    'timestamp': datetime.utcnow().isoformat()
                }
            ]

            for ws_data in sample_ws_data:
                data.append(CollectedData(
                    data_id=str(uuid.uuid4()),
                    source_type=DataSource.WEB_SOCKETS,
                    timestamp=datetime.utcnow(),
                    data=ws_data,
                    metadata={'collection_method': 'simulation'},
                    quality_score=0.88
                ))

        except Exception as e:
            logger.error(f"Error collecting WebSocket data: {e}")

        return data

    async def _apply_transformations(self, data: List[CollectedData]) -> List[CollectedData]:
        """Apply configured transformations to collected data"""
        try:
            for transformation in self.config.transformations:
                if transformation == 'normalize_timestamps':
                    data = self._normalize_timestamps(data)
                elif transformation == 'filter_low_quality':
                    data = self._filter_low_quality(data)
                elif transformation == 'enrich_metadata':
                    data = self._enrich_metadata(data)
                elif transformation == 'deduplicate':
                    data = self._deduplicate_data(data)

            return data

        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return data

    def _normalize_timestamps(self, data: List[CollectedData]) -> List[CollectedData]:
        """Normalize timestamps in data"""
        for item in data:
            if 'timestamp' in item.data and isinstance(item.data['timestamp'], str):
                try:
                    item.data['timestamp'] = datetime.fromisoformat(item.data['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    pass

        return data

    def _filter_low_quality(self, data: List[CollectedData]) -> List[CollectedData]:
        """Filter out low quality data"""
        quality_threshold = self.config.filters.get('min_quality_score', 0.5)
        return [item for item in data if item.quality_score >= quality_threshold]

    def _enrich_metadata(self, data: List[CollectedData]) -> List[CollectedData]:
        """Enrich metadata with additional information"""
        for item in data:
            item.metadata.update({
                'collector_id': self.collector_id,
                'collection_timestamp': datetime.utcnow().isoformat(),
                'data_size_bytes': len(json.dumps(item.data).encode('utf-8'))
            })

        return data

    def _deduplicate_data(self, data: List[CollectedData]) -> List[CollectedData]:
        """Remove duplicate data entries"""
        seen_content = set()
        unique_data = []

        for item in data:
            content_hash = hash(json.dumps(item.data, sort_keys=True))
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_data.append(item)

        return unique_data


class DataQualityAssessor:
    """Assess and improve data quality"""

    @staticmethod
    def assess_data_quality(data: CollectedData) -> float:
        """Assess the quality of collected data"""
        quality_score = 1.0
        deductions = 0.0

        # Check for missing required fields
        required_fields = ['timestamp', 'source_type']
        for field in required_fields:
            if field not in data.data:
                deductions += 0.2

        # Check data completeness
        if not data.data:
            deductions += 0.5

        # Check timestamp validity
        if 'timestamp' in data.data:
            try:
                if isinstance(data.data['timestamp'], str):
                    datetime.fromisoformat(data.data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                deductions += 0.3

        # Check data structure
        if not isinstance(data.data, dict):
            deductions += 0.4

        return max(0.0, quality_score - deductions)

    @staticmethod
    def validate_data_schema(data: CollectedData, schema: Dict[str, Any]) -> bool:
        """Validate data against schema"""
        try:
            # Simple schema validation
            for field, field_type in schema.items():
                if field not in data.data:
                    return False

                if not isinstance(data.data[field], field_type):
                    return False

            return True

        except Exception:
            return False

    @staticmethod
    def detect_anomalies(data: List[CollectedData]) -> List[CollectedData]:
        """Detect anomalous data points"""
        anomalies = []

        if len(data) < 10:
            return anomalies

        # Simple anomaly detection based on data size
        data_sizes = [len(json.dumps(item.data).encode('utf-8')) for item in data]
        mean_size = np.mean(data_sizes)
        std_size = np.std(data_sizes)

        for item in data:
            item_size = len(json.dumps(item.data).encode('utf-8'))
            if abs(item_size - mean_size) > 3 * std_size:  # 3 standard deviations
                anomalies.append(item)

        return anomalies


# Factory functions
def create_data_collector(config: Optional[Dict[str, Any]] = None) -> ConversationDataCollector:
    """Factory function to create data collector"""
    return ConversationDataCollector(config)


def create_collection_config(source_type: DataSource, **kwargs) -> DataCollectionConfig:
    """Factory function to create collection configuration"""
    return DataCollectionConfig(source_type=source_type, **kwargs)