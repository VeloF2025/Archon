#!/usr/bin/env python3
"""
Real Environment Interfaces Module

This module provides interfaces to real-world environments including IoT devices,
sensors, APIs, databases, file systems, network resources, and external services.
It bridges the gap between simulated environments and production deployments.

Created: 2025-01-09
Author: Archon Enhancement System
Version: 7.1.0
"""

import asyncio
import json
import uuid
import time
import os
import subprocess
import psutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import aiofiles
import aiohttp
import socket
import threading
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Types of environment interfaces"""
    IOT_DEVICE = auto()
    SENSOR_NETWORK = auto()
    DATABASE = auto()
    FILE_SYSTEM = auto()
    WEB_SERVICE = auto()
    SYSTEM_RESOURCES = auto()
    NETWORK = auto()
    CLOUD_SERVICE = auto()
    CONTAINER = auto()
    PROCESS = auto()
    HARDWARE = auto()
    EXTERNAL_API = auto()


class InterfaceProtocol(Enum):
    """Communication protocols"""
    HTTP = auto()
    HTTPS = auto()
    MQTT = auto()
    WEBSOCKET = auto()
    TCP = auto()
    UDP = auto()
    SERIAL = auto()
    I2C = auto()
    SPI = auto()
    MODBUS = auto()
    REST_API = auto()
    GRAPHQL = auto()
    GRPC = auto()
    SSH = auto()
    FTP = auto()
    SFTP = auto()


class DataFormat(Enum):
    """Data formats for environment communication"""
    JSON = auto()
    XML = auto()
    CSV = auto()
    BINARY = auto()
    PROTOBUF = auto()
    MSGPACK = auto()
    YAML = auto()
    TEXT = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


@dataclass
class EnvironmentConfig:
    """Configuration for environment interfaces"""
    interface_id: str
    environment_type: EnvironmentType
    protocol: InterfaceProtocol
    connection_params: Dict[str, Any] = field(default_factory=dict)
    data_format: DataFormat = DataFormat.JSON
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    authentication: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentData:
    """Data from environment interfaces"""
    interface_id: str
    timestamp: datetime
    data_type: str
    raw_data: Any
    processed_data: Optional[Any] = None
    quality_score: float = 1.0
    source_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentCommand:
    """Command to send to environment"""
    command_id: str
    interface_id: str
    command_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_response: Optional[str] = None
    timeout: float = 30.0
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentMetrics:
    """Metrics for environment interfaces"""
    interface_id: str
    connection_status: str = "disconnected"
    last_connection: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    data_points_collected: int = 0
    error_rate: float = 0.0
    uptime_percentage: float = 0.0
    last_error: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


class BaseEnvironmentInterface(ABC):
    """Abstract base class for environment interfaces"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.connection = None
        self.is_connected = False
        self.metrics = EnvironmentMetrics(interface_id=config.interface_id)
        self.data_cache: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to environment"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to environment"""
        pass
    
    @abstractmethod
    async def read_data(self, data_type: str = None) -> Optional[EnvironmentData]:
        """Read data from environment"""
        pass
    
    @abstractmethod
    async def write_data(self, data: Any, data_type: str = None) -> bool:
        """Write data to environment"""
        pass
    
    @abstractmethod
    async def execute_command(self, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute command in environment"""
        pass
    
    def update_metrics(self, success: bool, response_time: float, error: str = None):
        """Update interface metrics"""
        with self.lock:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_requests += 1
                # Update average response time
                total_success = self.metrics.successful_requests
                current_avg = self.metrics.average_response_time
                self.metrics.average_response_time = (
                    (current_avg * (total_success - 1) + response_time) / total_success
                )
            else:
                self.metrics.failed_requests += 1
                self.metrics.last_error = error
            
            # Update error rate
            self.metrics.error_rate = (
                self.metrics.failed_requests / self.metrics.total_requests
            ) * 100
            
            self.metrics.last_updated = datetime.now()


class WebServiceInterface(BaseEnvironmentInterface):
    """Interface for web services and APIs"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = config.connection_params.get('base_url', '')
        self.api_key = config.authentication.get('api_key') if config.authentication else None
    
    async def connect(self) -> bool:
        """Connect to web service"""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            
            headers = self.config.headers.copy()
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers
            )
            
            # Test connection with a simple GET request
            if self.base_url:
                try:
                    async with self.session.get(f"{self.base_url}/health") as response:
                        self.is_connected = response.status < 400
                except:
                    # If /health endpoint doesn't exist, assume connected if we can reach the base URL
                    try:
                        async with self.session.get(self.base_url) as response:
                            self.is_connected = response.status < 500
                    except:
                        self.is_connected = True  # Assume connected, will fail on actual requests
            else:
                self.is_connected = True
            
            if self.is_connected:
                self.metrics.connection_status = "connected"
                self.metrics.last_connection = datetime.now()
                logger.info(f"Connected to web service: {self.config.interface_id}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Web service connection failed: {e}")
            self.metrics.last_error = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from web service"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.is_connected = False
            self.metrics.connection_status = "disconnected"
            
            logger.info(f"Disconnected from web service: {self.config.interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"Web service disconnect failed: {e}")
            return False
    
    async def read_data(self, data_type: str = None) -> Optional[EnvironmentData]:
        """Read data from web service"""
        try:
            if not self.is_connected or not self.session:
                await self.connect()
            
            start_time = time.time()
            endpoint = self.config.connection_params.get('read_endpoint', '/data')
            url = f"{self.base_url}{endpoint}"
            
            params = {}
            if data_type:
                params['type'] = data_type
            
            async with self.session.get(url, params=params) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    if self.config.data_format == DataFormat.JSON:
                        raw_data = await response.json()
                    elif self.config.data_format == DataFormat.TEXT:
                        raw_data = await response.text()
                    elif self.config.data_format == DataFormat.BINARY:
                        raw_data = await response.read()
                    else:
                        raw_data = await response.text()
                    
                    env_data = EnvironmentData(
                        interface_id=self.config.interface_id,
                        timestamp=datetime.now(),
                        data_type=data_type or "web_data",
                        raw_data=raw_data,
                        source_info={'url': url, 'status': response.status}
                    )
                    
                    self.data_cache.append(env_data)
                    self.metrics.data_points_collected += 1
                    self.update_metrics(True, response_time)
                    
                    return env_data
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    self.update_metrics(False, response_time, error_msg)
                    logger.error(f"Web service read failed: {error_msg}")
                    return None
                    
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Web service read failed: {e}")
            return None
    
    async def write_data(self, data: Any, data_type: str = None) -> bool:
        """Write data to web service"""
        try:
            if not self.is_connected or not self.session:
                await self.connect()
            
            start_time = time.time()
            endpoint = self.config.connection_params.get('write_endpoint', '/data')
            url = f"{self.base_url}{endpoint}"
            
            # Prepare data based on format
            if self.config.data_format == DataFormat.JSON:
                payload = json.dumps(data) if not isinstance(data, str) else data
                headers = {'Content-Type': 'application/json'}
            else:
                payload = str(data)
                headers = {'Content-Type': 'text/plain'}
            
            async with self.session.post(url, data=payload, headers=headers) as response:
                response_time = time.time() - start_time
                success = response.status < 400
                
                if not success:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    self.update_metrics(False, response_time, error_msg)
                else:
                    self.update_metrics(True, response_time)
                
                return success
                
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Web service write failed: {e}")
            return False
    
    async def execute_command(self, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute command via web service"""
        try:
            if not self.is_connected or not self.session:
                await self.connect()
            
            start_time = time.time()
            endpoint = self.config.connection_params.get('command_endpoint', '/command')
            url = f"{self.base_url}{endpoint}"
            
            payload = {
                'command_id': command.command_id,
                'command_type': command.command_type,
                'parameters': command.parameters
            }
            
            async with self.session.post(url, json=payload) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    self.update_metrics(True, response_time)
                    return result
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    self.update_metrics(False, response_time, error_msg)
                    return None
                    
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Web service command failed: {e}")
            return None


class FileSystemInterface(BaseEnvironmentInterface):
    """Interface for file system operations"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.base_path = Path(config.connection_params.get('base_path', '.'))
        self.allowed_extensions = set(config.connection_params.get('allowed_extensions', ['.txt', '.json', '.csv', '.log']))
    
    async def connect(self) -> bool:
        """Connect to file system"""
        try:
            # Ensure base path exists and is accessible
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Test read/write access
            test_file = self.base_path / f".test_{uuid.uuid4().hex[:8]}.tmp"
            test_file.write_text("test")
            test_content = test_file.read_text()
            test_file.unlink()
            
            self.is_connected = test_content == "test"
            
            if self.is_connected:
                self.metrics.connection_status = "connected"
                self.metrics.last_connection = datetime.now()
                logger.info(f"Connected to file system: {self.base_path}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"File system connection failed: {e}")
            self.metrics.last_error = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from file system"""
        self.is_connected = False
        self.metrics.connection_status = "disconnected"
        logger.info(f"Disconnected from file system: {self.config.interface_id}")
        return True
    
    async def read_data(self, data_type: str = None) -> Optional[EnvironmentData]:
        """Read data from file system"""
        try:
            start_time = time.time()
            
            # Determine file to read
            if data_type:
                file_path = self.base_path / data_type
            else:
                # Find most recent file
                files = [f for f in self.base_path.iterdir() if f.is_file() and f.suffix in self.allowed_extensions]
                if not files:
                    return None
                file_path = max(files, key=lambda f: f.stat().st_mtime)
            
            if not file_path.exists():
                return None
            
            # Read file content
            if file_path.suffix == '.json':
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    raw_data = json.loads(content)
            else:
                async with aiofiles.open(file_path, 'r') as f:
                    raw_data = await f.read()
            
            response_time = time.time() - start_time
            
            env_data = EnvironmentData(
                interface_id=self.config.interface_id,
                timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
                data_type=data_type or file_path.name,
                raw_data=raw_data,
                source_info={
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime
                }
            )
            
            self.data_cache.append(env_data)
            self.metrics.data_points_collected += 1
            self.update_metrics(True, response_time)
            
            return env_data
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"File system read failed: {e}")
            return None
    
    async def write_data(self, data: Any, data_type: str = None) -> bool:
        """Write data to file system"""
        try:
            start_time = time.time()
            
            # Determine file path
            if data_type:
                file_path = self.base_path / data_type
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = self.base_path / f"data_{timestamp}.json"
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data
            if isinstance(data, dict) or isinstance(data, list):
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
            else:
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(str(data))
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            
            return True
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"File system write failed: {e}")
            return False
    
    async def execute_command(self, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute file system command"""
        try:
            start_time = time.time()
            command_type = command.command_type
            
            if command_type == "list_files":
                pattern = command.parameters.get('pattern', '*')
                files = list(self.base_path.glob(pattern))
                result = {
                    'files': [str(f.relative_to(self.base_path)) for f in files if f.is_file()],
                    'count': len(files)
                }
                
            elif command_type == "delete_file":
                file_path = self.base_path / command.parameters.get('filename', '')
                if file_path.exists() and file_path.is_file():
                    file_path.unlink()
                    result = {'deleted': True, 'file': str(file_path)}
                else:
                    result = {'deleted': False, 'error': 'File not found'}
                    
            elif command_type == "create_directory":
                dir_path = self.base_path / command.parameters.get('dirname', '')
                dir_path.mkdir(parents=True, exist_ok=True)
                result = {'created': True, 'directory': str(dir_path)}
                
            elif command_type == "get_stats":
                total_files = len([f for f in self.base_path.rglob('*') if f.is_file()])
                total_size = sum(f.stat().st_size for f in self.base_path.rglob('*') if f.is_file())
                result = {
                    'total_files': total_files,
                    'total_size_bytes': total_size,
                    'total_size_mb': total_size / (1024 * 1024)
                }
            else:
                result = {'error': f'Unknown command: {command_type}'}
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"File system command failed: {e}")
            return {'error': str(e)}


class SystemResourceInterface(BaseEnvironmentInterface):
    """Interface for system resources (CPU, memory, disk, etc.)"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.monitoring_interval = config.connection_params.get('monitoring_interval', 1.0)
        self.psutil_available = self._check_psutil()
    
    def _check_psutil(self) -> bool:
        """Check if psutil is available"""
        try:
            import psutil
            return True
        except ImportError:
            logger.warning("psutil not available - using limited system monitoring")
            return False
    
    async def connect(self) -> bool:
        """Connect to system resources"""
        try:
            # Test basic system access
            if self.psutil_available:
                import psutil
                # Test access to system metrics
                _ = psutil.cpu_percent()
                _ = psutil.virtual_memory()
                _ = psutil.disk_usage('/')
            else:
                # Fallback: check if we can at least access basic system info
                import platform
                _ = platform.system()
            
            self.is_connected = True
            self.metrics.connection_status = "connected"
            self.metrics.last_connection = datetime.now()
            
            logger.info(f"Connected to system resources: {self.config.interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"System resource connection failed: {e}")
            self.metrics.last_error = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from system resources"""
        self.is_connected = False
        self.metrics.connection_status = "disconnected"
        logger.info(f"Disconnected from system resources: {self.config.interface_id}")
        return True
    
    async def read_data(self, data_type: str = None) -> Optional[EnvironmentData]:
        """Read system resource data"""
        try:
            start_time = time.time()
            
            if self.psutil_available:
                import psutil
                
                if data_type == "cpu":
                    raw_data = {
                        'cpu_percent': psutil.cpu_percent(interval=0.1),
                        'cpu_count': psutil.cpu_count(),
                        'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                        'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
                    }
                elif data_type == "memory":
                    mem = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    raw_data = {
                        'virtual_memory': mem._asdict(),
                        'swap_memory': swap._asdict()
                    }
                elif data_type == "disk":
                    disk_usage = psutil.disk_usage('/')
                    disk_io = psutil.disk_io_counters()
                    raw_data = {
                        'disk_usage': disk_usage._asdict(),
                        'disk_io': disk_io._asdict() if disk_io else None,
                        'disk_partitions': [p._asdict() for p in psutil.disk_partitions()]
                    }
                elif data_type == "network":
                    net_io = psutil.net_io_counters()
                    net_connections = psutil.net_connections(kind='inet')
                    raw_data = {
                        'net_io': net_io._asdict() if net_io else None,
                        'connections_count': len(net_connections),
                        'network_interfaces': list(psutil.net_if_addrs().keys())
                    }
                elif data_type == "processes":
                    processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                        try:
                            processes.append(proc.info)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    raw_data = {
                        'process_count': len(processes),
                        'top_processes': sorted(processes, key=lambda p: p.get('cpu_percent', 0), reverse=True)[:10]
                    }
                else:
                    # All system data
                    raw_data = {
                        'cpu_percent': psutil.cpu_percent(),
                        'memory': psutil.virtual_memory()._asdict(),
                        'disk': psutil.disk_usage('/')._asdict(),
                        'network': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else None,
                        'boot_time': psutil.boot_time(),
                        'users': [u._asdict() for u in psutil.users()]
                    }
            else:
                # Fallback system info
                import platform
                import shutil
                
                raw_data = {
                    'platform': platform.system(),
                    'platform_version': platform.version(),
                    'architecture': platform.architecture(),
                    'processor': platform.processor(),
                    'hostname': platform.node(),
                    'disk_free': shutil.disk_usage('.').free if hasattr(shutil, 'disk_usage') else None,
                    'python_version': platform.python_version()
                }
            
            response_time = time.time() - start_time
            
            env_data = EnvironmentData(
                interface_id=self.config.interface_id,
                timestamp=datetime.now(),
                data_type=data_type or "system",
                raw_data=raw_data,
                source_info={'psutil_available': self.psutil_available}
            )
            
            self.data_cache.append(env_data)
            self.metrics.data_points_collected += 1
            self.update_metrics(True, response_time)
            
            return env_data
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"System resource read failed: {e}")
            return None
    
    async def write_data(self, data: Any, data_type: str = None) -> bool:
        """Write system configuration (limited operations)"""
        try:
            # System resources are mostly read-only, but we can set some configurations
            if data_type == "environment_variable":
                if isinstance(data, dict):
                    for key, value in data.items():
                        os.environ[key] = str(value)
                    return True
            
            # Most system resources are read-only
            logger.warning(f"Write operation not supported for system resources: {data_type}")
            return False
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"System resource write failed: {e}")
            return False
    
    async def execute_command(self, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute system command"""
        try:
            start_time = time.time()
            command_type = command.command_type
            
            if command_type == "system_info":
                if self.psutil_available:
                    import psutil
                    result = {
                        'cpu_count': psutil.cpu_count(),
                        'memory_total': psutil.virtual_memory().total,
                        'disk_total': psutil.disk_usage('/').total,
                        'boot_time': psutil.boot_time(),
                        'platform': platform.system()
                    }
                else:
                    import platform
                    result = {
                        'platform': platform.system(),
                        'version': platform.version(),
                        'architecture': platform.architecture(),
                        'hostname': platform.node()
                    }
                    
            elif command_type == "kill_process":
                if self.psutil_available:
                    import psutil
                    pid = command.parameters.get('pid')
                    if pid:
                        try:
                            process = psutil.Process(pid)
                            process.terminate()
                            result = {'killed': True, 'pid': pid}
                        except psutil.NoSuchProcess:
                            result = {'killed': False, 'error': 'Process not found'}
                    else:
                        result = {'killed': False, 'error': 'No PID specified'}
                else:
                    result = {'killed': False, 'error': 'psutil not available'}
                    
            elif command_type == "execute_shell":
                shell_command = command.parameters.get('command', '')
                if shell_command:
                    try:
                        result_process = subprocess.run(
                            shell_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=command.timeout
                        )
                        result = {
                            'return_code': result_process.returncode,
                            'stdout': result_process.stdout,
                            'stderr': result_process.stderr,
                            'success': result_process.returncode == 0
                        }
                    except subprocess.TimeoutExpired:
                        result = {'error': 'Command timed out'}
                else:
                    result = {'error': 'No command specified'}
                    
            else:
                result = {'error': f'Unknown command: {command_type}'}
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"System command failed: {e}")
            return {'error': str(e)}


class NetworkInterface(BaseEnvironmentInterface):
    """Interface for network operations"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        self.host = config.connection_params.get('host', 'localhost')
        self.port = config.connection_params.get('port', 80)
        self.socket_type = config.connection_params.get('socket_type', 'tcp')
    
    async def connect(self) -> bool:
        """Test network connectivity"""
        try:
            if self.socket_type.lower() == 'tcp':
                # Test TCP connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.timeout)
                result = sock.connect_ex((self.host, self.port))
                sock.close()
                self.is_connected = result == 0
            else:
                # For UDP, just assume connected (connectionless)
                self.is_connected = True
            
            if self.is_connected:
                self.metrics.connection_status = "connected"
                self.metrics.last_connection = datetime.now()
                logger.info(f"Network connection established: {self.host}:{self.port}")
            
            return self.is_connected
            
        except Exception as e:
            logger.error(f"Network connection failed: {e}")
            self.metrics.last_error = str(e)
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from network"""
        self.is_connected = False
        self.metrics.connection_status = "disconnected"
        logger.info(f"Disconnected from network: {self.config.interface_id}")
        return True
    
    async def read_data(self, data_type: str = None) -> Optional[EnvironmentData]:
        """Read network data"""
        try:
            start_time = time.time()
            
            if data_type == "ping":
                # Ping test
                ping_result = await self._ping_test()
                raw_data = ping_result
            elif data_type == "port_scan":
                # Simple port scan
                ports = self.config.connection_params.get('scan_ports', [80, 443, 22, 21])
                scan_result = await self._port_scan(ports)
                raw_data = scan_result
            else:
                # Network interface stats
                if hasattr(psutil, 'net_io_counters'):
                    import psutil
                    net_stats = psutil.net_io_counters()
                    raw_data = net_stats._asdict() if net_stats else {}
                else:
                    raw_data = {'error': 'Network stats not available'}
            
            response_time = time.time() - start_time
            
            env_data = EnvironmentData(
                interface_id=self.config.interface_id,
                timestamp=datetime.now(),
                data_type=data_type or "network",
                raw_data=raw_data,
                source_info={'host': self.host, 'port': self.port}
            )
            
            self.data_cache.append(env_data)
            self.metrics.data_points_collected += 1
            self.update_metrics(True, response_time)
            
            return env_data
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Network read failed: {e}")
            return None
    
    async def write_data(self, data: Any, data_type: str = None) -> bool:
        """Send network data"""
        try:
            start_time = time.time()
            
            if self.socket_type.lower() == 'tcp':
                # TCP send
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.timeout)
                sock.connect((self.host, self.port))
                
                message = str(data).encode('utf-8')
                sock.sendall(message)
                sock.close()
                
            elif self.socket_type.lower() == 'udp':
                # UDP send
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                message = str(data).encode('utf-8')
                sock.sendto(message, (self.host, self.port))
                sock.close()
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            
            return True
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Network write failed: {e}")
            return False
    
    async def execute_command(self, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute network command"""
        try:
            start_time = time.time()
            command_type = command.command_type
            
            if command_type == "ping":
                result = await self._ping_test()
            elif command_type == "port_scan":
                ports = command.parameters.get('ports', [80, 443, 22])
                result = await self._port_scan(ports)
            elif command_type == "dns_lookup":
                hostname = command.parameters.get('hostname', self.host)
                try:
                    import socket
                    ip_address = socket.gethostbyname(hostname)
                    result = {'hostname': hostname, 'ip_address': ip_address, 'success': True}
                except socket.gaierror as e:
                    result = {'hostname': hostname, 'error': str(e), 'success': False}
            else:
                result = {'error': f'Unknown command: {command_type}'}
            
            response_time = time.time() - start_time
            self.update_metrics(True, response_time)
            
            return result
            
        except Exception as e:
            self.update_metrics(False, 0, str(e))
            logger.error(f"Network command failed: {e}")
            return {'error': str(e)}
    
    async def _ping_test(self) -> Dict[str, Any]:
        """Perform ping test"""
        try:
            import subprocess
            import platform
            
            # Determine ping command based on OS
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', '4', self.host]
            else:
                cmd = ['ping', '-c', '4', self.host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            return {
                'host': self.host,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            
        except Exception as e:
            return {'host': self.host, 'success': False, 'error': str(e)}
    
    async def _port_scan(self, ports: List[int]) -> Dict[str, Any]:
        """Perform simple port scan"""
        try:
            open_ports = []
            closed_ports = []
            
            for port in ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                result = sock.connect_ex((self.host, port))
                sock.close()
                
                if result == 0:
                    open_ports.append(port)
                else:
                    closed_ports.append(port)
            
            return {
                'host': self.host,
                'scanned_ports': ports,
                'open_ports': open_ports,
                'closed_ports': closed_ports,
                'total_open': len(open_ports)
            }
            
        except Exception as e:
            return {'host': self.host, 'error': str(e)}


class RealEnvironmentManager:
    """Main manager for real environment interfaces"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.manager_id = f"env_mgr_{uuid.uuid4().hex[:8]}"
        
        # Interface registry
        self.interfaces: Dict[str, BaseEnvironmentInterface] = {}
        self.interface_configs: Dict[str, EnvironmentConfig] = {}
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Data aggregation
        self.aggregated_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    async def start(self) -> None:
        """Start the environment manager"""
        try:
            self.is_running = True
            
            # Start background tasks
            self.background_tasks.add(
                asyncio.create_task(self._health_monitor())
            )
            
            self.background_tasks.add(
                asyncio.create_task(self._data_aggregator())
            )
            
            logger.info(f"Environment manager {self.manager_id} started")
            
        except Exception as e:
            logger.error(f"Environment manager start failed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the environment manager"""
        try:
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.background_tasks.clear()
            
            # Disconnect all interfaces
            for interface in self.interfaces.values():
                try:
                    await interface.disconnect()
                except Exception as e:
                    logger.error(f"Interface disconnect failed: {e}")
            
            logger.info(f"Environment manager {self.manager_id} stopped")
            
        except Exception as e:
            logger.error(f"Environment manager stop failed: {e}")
    
    async def create_interface(self, config: EnvironmentConfig) -> bool:
        """Create an environment interface"""
        try:
            interface_id = config.interface_id
            
            # Create interface based on type
            if config.environment_type == EnvironmentType.WEB_SERVICE:
                interface = WebServiceInterface(config)
            elif config.environment_type == EnvironmentType.FILE_SYSTEM:
                interface = FileSystemInterface(config)
            elif config.environment_type == EnvironmentType.SYSTEM_RESOURCES:
                interface = SystemResourceInterface(config)
            elif config.environment_type == EnvironmentType.NETWORK:
                interface = NetworkInterface(config)
            else:
                raise ValueError(f"Unsupported environment type: {config.environment_type}")
            
            # Connect interface
            success = await interface.connect()
            if not success:
                logger.error(f"Failed to connect interface: {interface_id}")
                return False
            
            # Store interface
            self.interfaces[interface_id] = interface
            self.interface_configs[interface_id] = config
            
            logger.info(f"Created interface: {interface_id} ({config.environment_type.name})")
            return True
            
        except Exception as e:
            logger.error(f"Interface creation failed: {e}")
            return False
    
    async def remove_interface(self, interface_id: str) -> bool:
        """Remove an environment interface"""
        try:
            if interface_id not in self.interfaces:
                return False
            
            interface = self.interfaces[interface_id]
            await interface.disconnect()
            
            del self.interfaces[interface_id]
            del self.interface_configs[interface_id]
            
            logger.info(f"Removed interface: {interface_id}")
            return True
            
        except Exception as e:
            logger.error(f"Interface removal failed: {e}")
            return False
    
    async def read_from_interface(self, interface_id: str, data_type: str = None) -> Optional[EnvironmentData]:
        """Read data from an interface"""
        try:
            if interface_id not in self.interfaces:
                logger.error(f"Interface {interface_id} not found")
                return None
            
            interface = self.interfaces[interface_id]
            data = await interface.read_data(data_type)
            
            if data:
                self.aggregated_data[interface_id].append(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Interface read failed: {e}")
            return None
    
    async def write_to_interface(self, interface_id: str, data: Any, data_type: str = None) -> bool:
        """Write data to an interface"""
        try:
            if interface_id not in self.interfaces:
                logger.error(f"Interface {interface_id} not found")
                return False
            
            interface = self.interfaces[interface_id]
            return await interface.write_data(data, data_type)
            
        except Exception as e:
            logger.error(f"Interface write failed: {e}")
            return False
    
    async def execute_interface_command(self, interface_id: str, command: EnvironmentCommand) -> Optional[Dict[str, Any]]:
        """Execute command on an interface"""
        try:
            if interface_id not in self.interfaces:
                logger.error(f"Interface {interface_id} not found")
                return None
            
            interface = self.interfaces[interface_id]
            return await interface.execute_command(command)
            
        except Exception as e:
            logger.error(f"Interface command failed: {e}")
            return None
    
    def get_interface_status(self, interface_id: str = None) -> Dict[str, Any]:
        """Get status of interfaces"""
        if interface_id:
            if interface_id in self.interfaces:
                interface = self.interfaces[interface_id]
                return {
                    'interface_id': interface_id,
                    'type': self.interface_configs[interface_id].environment_type.name,
                    'connected': interface.is_connected,
                    'metrics': interface.metrics.__dict__
                }
            else:
                return {}
        else:
            # All interfaces
            return {
                iid: {
                    'interface_id': iid,
                    'type': self.interface_configs[iid].environment_type.name,
                    'connected': interface.is_connected,
                    'metrics': interface.metrics.__dict__
                }
                for iid, interface in self.interfaces.items()
            }
    
    def list_interfaces(self) -> List[Dict[str, Any]]:
        """List all interfaces"""
        return [
            {
                'interface_id': interface_id,
                'environment_type': config.environment_type.name,
                'protocol': config.protocol.name,
                'connected': self.interfaces[interface_id].is_connected,
                'data_points': self.interfaces[interface_id].metrics.data_points_collected
            }
            for interface_id, config in self.interface_configs.items()
        ]
    
    def get_aggregated_data(self, interface_id: str = None, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get aggregated data from interfaces"""
        if interface_id:
            if interface_id in self.aggregated_data:
                data_list = list(self.aggregated_data[interface_id])[-limit:]
                return {interface_id: [d.__dict__ for d in data_list]}
            else:
                return {}
        else:
            result = {}
            for iid, data_deque in self.aggregated_data.items():
                data_list = list(data_deque)[-limit:]
                result[iid] = [d.__dict__ for d in data_list]
            return result
    
    async def _health_monitor(self) -> None:
        """Background task for monitoring interface health"""
        while self.is_running:
            try:
                for interface_id, interface in self.interfaces.items():
                    if not interface.is_connected:
                        logger.warning(f"Interface {interface_id} disconnected, attempting reconnect")
                        try:
                            await interface.connect()
                        except Exception as e:
                            logger.error(f"Reconnect failed for {interface_id}: {e}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
                await asyncio.sleep(60)
    
    async def _data_aggregator(self) -> None:
        """Background task for data aggregation"""
        while self.is_running:
            try:
                # Periodic data collection from all interfaces
                for interface_id in list(self.interfaces.keys()):
                    try:
                        await self.read_from_interface(interface_id)
                    except Exception as e:
                        logger.error(f"Background data collection failed for {interface_id}: {e}")
                
                # Clean up old data
                for interface_id in self.aggregated_data:
                    # Data is automatically limited by deque maxlen
                    pass
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Data aggregation failed: {e}")
                await asyncio.sleep(30)


async def example_real_environment_interfaces_usage():
    """Comprehensive example of real environment interfaces usage"""
    
    print("\n🌐 Real Environment Interfaces Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'monitoring_interval': 30,
        'data_retention': 1000
    }
    
    # Initialize environment manager
    env_manager = RealEnvironmentManager(config)
    await env_manager.start()
    
    print(f"✅ Environment manager {env_manager.manager_id} started")
    
    try:
        # Example 1: File System Interface
        print("\n1. File System Interface")
        print("-" * 40)
        
        fs_config = EnvironmentConfig(
            interface_id="filesystem_1",
            environment_type=EnvironmentType.FILE_SYSTEM,
            protocol=InterfaceProtocol.REST_API,  # Not applicable for FS
            connection_params={
                'base_path': './test_data',
                'allowed_extensions': ['.txt', '.json', '.csv', '.log']
            }
        )
        
        success = await env_manager.create_interface(fs_config)
        print(f"✅ File system interface created: {success}")
        
        # Write test data
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'sensor_readings': [23.5, 24.1, 23.8, 24.3],
            'status': 'operational'
        }
        
        write_success = await env_manager.write_to_interface('filesystem_1', test_data, 'sensor_data.json')
        print(f"✅ Data written to file system: {write_success}")
        
        # Read data back
        read_data = await env_manager.read_from_interface('filesystem_1', 'sensor_data.json')
        if read_data:
            print(f"✅ Data read from file system: {len(str(read_data.raw_data))} chars")
        
        # Execute file system command
        list_command = EnvironmentCommand(
            command_id="list_1",
            interface_id="filesystem_1",
            command_type="list_files",
            parameters={'pattern': '*.json'}
        )
        
        command_result = await env_manager.execute_interface_command('filesystem_1', list_command)
        if command_result:
            print(f"✅ File listing: {command_result.get('count', 0)} files found")
        
        # Example 2: System Resources Interface
        print("\n2. System Resources Interface")
        print("-" * 40)
        
        sys_config = EnvironmentConfig(
            interface_id="system_1",
            environment_type=EnvironmentType.SYSTEM_RESOURCES,
            protocol=InterfaceProtocol.REST_API,  # Not applicable
            connection_params={'monitoring_interval': 1.0}
        )
        
        success = await env_manager.create_interface(sys_config)
        print(f"✅ System resources interface created: {success}")
        
        # Read CPU data
        cpu_data = await env_manager.read_from_interface('system_1', 'cpu')
        if cpu_data:
            cpu_percent = cpu_data.raw_data.get('cpu_percent', 'N/A')
            cpu_count = cpu_data.raw_data.get('cpu_count', 'N/A')
            print(f"✅ CPU data: {cpu_percent}% usage, {cpu_count} cores")
        
        # Read memory data
        memory_data = await env_manager.read_from_interface('system_1', 'memory')
        if memory_data:
            vm = memory_data.raw_data.get('virtual_memory', {})
            total_gb = vm.get('total', 0) / (1024**3) if vm else 0
            used_percent = vm.get('percent', 0) if vm else 0
            print(f"✅ Memory data: {total_gb:.1f}GB total, {used_percent}% used")
        
        # Execute system command
        info_command = EnvironmentCommand(
            command_id="sysinfo_1",
            interface_id="system_1",
            command_type="system_info"
        )
        
        sys_info = await env_manager.execute_interface_command('system_1', info_command)
        if sys_info:
            platform_name = sys_info.get('platform', 'Unknown')
            print(f"✅ System info: Platform {platform_name}")
        
        # Example 3: Network Interface
        print("\n3. Network Interface")
        print("-" * 40)
        
        net_config = EnvironmentConfig(
            interface_id="network_1",
            environment_type=EnvironmentType.NETWORK,
            protocol=InterfaceProtocol.TCP,
            connection_params={
                'host': 'google.com',
                'port': 80,
                'socket_type': 'tcp'
            }
        )
        
        success = await env_manager.create_interface(net_config)
        print(f"✅ Network interface created: {success}")
        
        # Test network connectivity
        ping_command = EnvironmentCommand(
            command_id="ping_1",
            interface_id="network_1",
            command_type="ping"
        )
        
        ping_result = await env_manager.execute_interface_command('network_1', ping_command)
        if ping_result:
            ping_success = ping_result.get('success', False)
            print(f"✅ Ping test: {'Success' if ping_success else 'Failed'}")
        
        # DNS lookup
        dns_command = EnvironmentCommand(
            command_id="dns_1",
            interface_id="network_1",
            command_type="dns_lookup",
            parameters={'hostname': 'google.com'}
        )
        
        dns_result = await env_manager.execute_interface_command('network_1', dns_command)
        if dns_result:
            ip_address = dns_result.get('ip_address', 'Unknown')
            print(f"✅ DNS lookup: google.com -> {ip_address}")
        
        # Example 4: Web Service Interface (Mock)
        print("\n4. Web Service Interface")
        print("-" * 40)
        
        web_config = EnvironmentConfig(
            interface_id="webservice_1",
            environment_type=EnvironmentType.WEB_SERVICE,
            protocol=InterfaceProtocol.HTTPS,
            connection_params={
                'base_url': 'https://httpbin.org',
                'read_endpoint': '/get',
                'write_endpoint': '/post',
                'command_endpoint': '/status/200'
            },
            headers={'User-Agent': 'Archon-Environment-Interface/1.0'}
        )
        
        success = await env_manager.create_interface(web_config)
        print(f"✅ Web service interface created: {success}")
        
        # Read from web service
        try:
            web_data = await env_manager.read_from_interface('webservice_1')
            if web_data:
                origin = web_data.raw_data.get('origin', 'Unknown') if isinstance(web_data.raw_data, dict) else 'Data received'
                print(f"✅ Web service data: {origin}")
        except Exception as e:
            print(f"⚠️ Web service read: {str(e)[:50]}...")
        
        # Write to web service
        try:
            post_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
            write_success = await env_manager.write_to_interface('webservice_1', post_data)
            print(f"✅ Web service write: {write_success}")
        except Exception as e:
            print(f"⚠️ Web service write: {str(e)[:50]}...")
        
        # Example 5: Interface Status and Monitoring
        print("\n5. Interface Status and Monitoring")
        print("-" * 40)
        
        # List all interfaces
        interfaces = env_manager.list_interfaces()
        print(f"✅ Active interfaces: {len(interfaces)}")
        
        for interface_info in interfaces:
            print(f"   - {interface_info['interface_id']}: {interface_info['environment_type']}")
            print(f"     Protocol: {interface_info['protocol']}")
            print(f"     Connected: {interface_info['connected']}")
            print(f"     Data points: {interface_info['data_points']}")
        
        # Get detailed status
        status = env_manager.get_interface_status()
        print(f"\n✅ Detailed Interface Status:")
        for iid, info in status.items():
            metrics = info['metrics']
            print(f"   {iid} ({info['type']}):")
            print(f"     Connection: {info['connected']}")
            print(f"     Total requests: {metrics['total_requests']}")
            print(f"     Success rate: {((metrics['successful_requests'] / max(metrics['total_requests'], 1)) * 100):.1f}%")
            print(f"     Avg response time: {metrics['average_response_time']:.3f}s")
        
        # Example 6: Aggregated Data
        print("\n6. Aggregated Data Collection")
        print("-" * 40)
        
        # Let background data collection run
        await asyncio.sleep(2)
        
        aggregated = env_manager.get_aggregated_data(limit=5)
        print(f"✅ Aggregated data from {len(aggregated)} interfaces:")
        
        for interface_id, data_list in aggregated.items():
            if data_list:
                print(f"   {interface_id}: {len(data_list)} data points")
                latest = data_list[-1] if data_list else {}
                print(f"     Latest: {latest.get('data_type', 'unknown')} at {latest.get('timestamp', 'unknown')}")
        
        # Allow background tasks to run
        await asyncio.sleep(2)
        
    finally:
        # Cleanup
        await env_manager.stop()
        print(f"\n✅ Real environment interfaces system stopped successfully")


if __name__ == "__main__":
    asyncio.run(example_real_environment_interfaces_usage())