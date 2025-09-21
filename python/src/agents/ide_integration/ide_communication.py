"""
IDE Communication Protocol
Handles communication between Archon and various IDEs using multiple protocols
"""

import asyncio
import json
import websockets
import aiohttp
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import ssl
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in IDE communication"""
    REQUEST = "request"
    RESPONSE = "response" 
    NOTIFICATION = "notification"
    EVENT = "event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class CommunicationMessage:
    """Base communication message structure"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.REQUEST
    protocol: str = "archon"
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    target: str = ""
    method: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class ConnectionConfig:
    """Configuration for IDE connections"""
    host: str = "localhost"
    port: int = 0
    use_ssl: bool = False
    ssl_context: Optional[ssl.SSLContext] = None
    timeout_seconds: int = 30
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 30.0
    message_buffer_size: int = 1000
    compression: bool = False
    authentication: Optional[Dict[str, str]] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)


class CommunicationChannel(ABC):
    """Abstract base class for communication channels"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.message_handlers: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._connection = None
        self._last_heartbeat: Optional[datetime] = None
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.message_buffer_size)
        self._response_futures: Dict[str, asyncio.Future] = {}
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to IDE"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from IDE"""
        pass
    
    @abstractmethod
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message to IDE"""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncGenerator[CommunicationMessage, None]:
        """Receive messages from IDE"""
        pass
    
    def register_message_handler(self, method: str, handler: Callable) -> None:
        """Register handler for specific message method"""
        self.message_handlers[method] = handler
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register handler for specific events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request and wait for response"""
        message = CommunicationMessage(
            message_type=MessageType.REQUEST,
            method=method,
            params=params,
            source="archon",
            target="ide"
        )
        
        # Create future for response
        future = asyncio.create_future()
        self._response_futures[message.message_id] = future
        
        try:
            await self.send_message(message)
            response = await asyncio.wait_for(future, timeout=self.config.timeout_seconds)
            return response
        except asyncio.TimeoutError:
            self._response_futures.pop(message.message_id, None)
            raise TimeoutError(f"Request {method} timed out")
        except Exception as e:
            self._response_futures.pop(message.message_id, None)
            raise e
    
    async def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send notification (no response expected)"""
        message = CommunicationMessage(
            message_type=MessageType.NOTIFICATION,
            method=method,
            params=params,
            source="archon",
            target="ide"
        )
        await self.send_message(message)
    
    async def handle_incoming_message(self, message: CommunicationMessage) -> None:
        """Handle incoming message from IDE"""
        try:
            if message.message_type == MessageType.RESPONSE:
                # Handle response to our request
                if message.message_id in self._response_futures:
                    future = self._response_futures.pop(message.message_id)
                    if not future.done():
                        if message.error_code:
                            future.set_exception(Exception(f"Error {message.error_code}: {message.error_message}"))
                        else:
                            future.set_result(message.data)
            
            elif message.message_type == MessageType.REQUEST:
                # Handle incoming request
                if message.method and message.method in self.message_handlers:
                    handler = self.message_handlers[message.method]
                    result = await handler(message)
                    
                    # Send response
                    response = CommunicationMessage(
                        message_id=message.message_id,
                        message_type=MessageType.RESPONSE,
                        source="archon",
                        target="ide",
                        data=result if result else {}
                    )
                    await self.send_message(response)
            
            elif message.message_type == MessageType.NOTIFICATION:
                # Handle notification
                if message.method and message.method in self.message_handlers:
                    handler = self.message_handlers[message.method]
                    await handler(message)
            
            elif message.message_type == MessageType.EVENT:
                # Handle events
                event_type = message.params.get("event_type", "unknown")
                if event_type in self.event_handlers:
                    for handler in self.event_handlers[event_type]:
                        await handler(message)
                        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            
            # Send error response if it was a request
            if message.message_type == MessageType.REQUEST:
                error_response = CommunicationMessage(
                    message_id=message.message_id,
                    message_type=MessageType.ERROR,
                    source="archon",
                    target="ide",
                    error_code=500,
                    error_message=str(e)
                )
                await self.send_message(error_response)


class WebSocketChannel(CommunicationChannel):
    """WebSocket-based communication channel"""
    
    async def connect(self) -> bool:
        """Connect via WebSocket"""
        try:
            self.state = ConnectionState.CONNECTING
            
            uri = f"{'wss' if self.config.use_ssl else 'ws'}://{self.config.host}:{self.config.port}"
            extra_headers = self.config.custom_headers.copy()
            
            if self.config.authentication:
                auth_header = self._create_auth_header(self.config.authentication)
                extra_headers.update(auth_header)
            
            self._connection = await websockets.connect(
                uri,
                ssl=self.config.ssl_context if self.config.use_ssl else None,
                extra_headers=extra_headers,
                compression="deflate" if self.config.compression else None,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.timeout_seconds
            )
            
            self.state = ConnectionState.CONNECTED
            self._last_heartbeat = datetime.now()
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect WebSocket"""
        try:
            if self._connection:
                await self._connection.close()
                self._connection = None
            self.state = ConnectionState.DISCONNECTED
            return True
        except Exception as e:
            logger.error(f"WebSocket disconnect failed: {e}")
            return False
    
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message via WebSocket"""
        try:
            if not self._connection or self.state != ConnectionState.CONNECTED:
                return False
            
            message_json = json.dumps({
                "id": message.message_id,
                "type": message.message_type.value,
                "protocol": message.protocol,
                "timestamp": message.timestamp.isoformat(),
                "source": message.source,
                "target": message.target,
                "method": message.method,
                "params": message.params,
                "data": message.data,
                "error_code": message.error_code,
                "error_message": message.error_message
            })
            
            await self._connection.send(message_json)
            return True
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    async def receive_messages(self) -> AsyncGenerator[CommunicationMessage, None]:
        """Receive messages via WebSocket"""
        try:
            async for raw_message in self._connection:
                try:
                    data = json.loads(raw_message)
                    message = self._parse_message(data)
                    yield message
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.state = ConnectionState.DISCONNECTED
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            self.state = ConnectionState.ERROR
    
    async def _process_messages(self) -> None:
        """Process incoming messages"""
        async for message in self.receive_messages():
            await self.handle_incoming_message(message)
    
    def _create_auth_header(self, auth: Dict[str, str]) -> Dict[str, str]:
        """Create authentication header"""
        if "token" in auth:
            return {"Authorization": f"Bearer {auth['token']}"}
        elif "username" in auth and "password" in auth:
            import base64
            credentials = base64.b64encode(f"{auth['username']}:{auth['password']}".encode()).decode()
            return {"Authorization": f"Basic {credentials}"}
        return {}
    
    def _parse_message(self, data: Dict[str, Any]) -> CommunicationMessage:
        """Parse raw message data into CommunicationMessage"""
        return CommunicationMessage(
            message_id=data.get("id", str(uuid.uuid4())),
            message_type=MessageType(data.get("type", "request")),
            protocol=data.get("protocol", "archon"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            source=data.get("source", ""),
            target=data.get("target", ""),
            method=data.get("method"),
            params=data.get("params", {}),
            data=data.get("data", {}),
            error_code=data.get("error_code"),
            error_message=data.get("error_message")
        )


class HTTPChannel(CommunicationChannel):
    """HTTP-based communication channel"""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Establish HTTP session"""
        try:
            self.state = ConnectionState.CONNECTING
            
            connector = aiohttp.TCPConnector(
                ssl=self.config.ssl_context if self.config.use_ssl else False
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.config.custom_headers
            )
            
            self.state = ConnectionState.CONNECTED
            return True
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            self.state = ConnectionState.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """Close HTTP session"""
        try:
            if self._session:
                await self._session.close()
                self._session = None
            self.state = ConnectionState.DISCONNECTED
            return True
        except Exception as e:
            logger.error(f"HTTP disconnect failed: {e}")
            return False
    
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message via HTTP POST"""
        try:
            if not self._session or self.state != ConnectionState.CONNECTED:
                return False
            
            url = f"{'https' if self.config.use_ssl else 'http'}://{self.config.host}:{self.config.port}/archon/message"
            
            payload = {
                "id": message.message_id,
                "type": message.message_type.value,
                "protocol": message.protocol,
                "timestamp": message.timestamp.isoformat(),
                "source": message.source,
                "target": message.target,
                "method": message.method,
                "params": message.params,
                "data": message.data,
                "error_code": message.error_code,
                "error_message": message.error_message
            }
            
            async with self._session.post(url, json=payload) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send HTTP message: {e}")
            return False
    
    async def receive_messages(self) -> AsyncGenerator[CommunicationMessage, None]:
        """Receive messages via HTTP polling"""
        while self.state == ConnectionState.CONNECTED:
            try:
                if not self._session:
                    break
                
                url = f"{'https' if self.config.use_ssl else 'http'}://{self.config.host}:{self.config.port}/archon/poll"
                
                async with self._session.get(url) as response:
                    if response.status == 200:
                        messages = await response.json()
                        for msg_data in messages:
                            message = self._parse_message(msg_data)
                            yield message
                    
                # Polling delay
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"HTTP polling error: {e}")
                await asyncio.sleep(5.0)
    
    def _parse_message(self, data: Dict[str, Any]) -> CommunicationMessage:
        """Parse raw message data into CommunicationMessage"""
        return CommunicationMessage(
            message_id=data.get("id", str(uuid.uuid4())),
            message_type=MessageType(data.get("type", "request")),
            protocol=data.get("protocol", "archon"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            source=data.get("source", ""),
            target=data.get("target", ""),
            method=data.get("method"),
            params=data.get("params", {}),
            data=data.get("data", {}),
            error_code=data.get("error_code"),
            error_message=data.get("error_message")
        )


class IDECommunicationProtocol:
    """Main communication protocol manager"""
    
    def __init__(self):
        self._channels: Dict[str, CommunicationChannel] = {}
        self._active_connections: Dict[str, str] = {}  # ide_id -> channel_name
        self._message_router: Dict[str, Callable] = {}
        
    def register_channel(self, name: str, channel: CommunicationChannel) -> None:
        """Register a communication channel"""
        self._channels[name] = channel
    
    def get_channel(self, name: str) -> Optional[CommunicationChannel]:
        """Get communication channel by name"""
        return self._channels.get(name)
    
    async def connect_to_ide(self, ide_id: str, channel_name: str, config: ConnectionConfig) -> bool:
        """Connect to IDE using specified channel"""
        try:
            if channel_name not in self._channels:
                # Create channel based on name
                if channel_name == "websocket":
                    channel = WebSocketChannel(config)
                elif channel_name == "http":
                    channel = HTTPChannel(config)
                else:
                    logger.error(f"Unknown channel type: {channel_name}")
                    return False
                
                self._channels[channel_name] = channel
            else:
                channel = self._channels[channel_name]
            
            success = await channel.connect()
            if success:
                self._active_connections[ide_id] = channel_name
                logger.info(f"Connected to IDE {ide_id} via {channel_name}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to connect to IDE {ide_id}: {e}")
            return False
    
    async def disconnect_from_ide(self, ide_id: str) -> bool:
        """Disconnect from IDE"""
        try:
            if ide_id not in self._active_connections:
                return True
            
            channel_name = self._active_connections[ide_id]
            channel = self._channels.get(channel_name)
            
            if channel:
                success = await channel.disconnect()
                if success:
                    del self._active_connections[ide_id]
                    logger.info(f"Disconnected from IDE {ide_id}")
                return success
            
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from IDE {ide_id}: {e}")
            return False
    
    async def send_to_ide(self, ide_id: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to IDE and wait for response"""
        if ide_id not in self._active_connections:
            raise ConnectionError(f"Not connected to IDE {ide_id}")
        
        channel_name = self._active_connections[ide_id]
        channel = self._channels[channel_name]
        
        return await channel.send_request(method, params)
    
    async def notify_ide(self, ide_id: str, method: str, params: Dict[str, Any]) -> None:
        """Send notification to IDE"""
        if ide_id not in self._active_connections:
            raise ConnectionError(f"Not connected to IDE {ide_id}")
        
        channel_name = self._active_connections[ide_id]
        channel = self._channels[channel_name]
        
        await channel.send_notification(method, params)
    
    def register_message_handler(self, method: str, handler: Callable) -> None:
        """Register global message handler"""
        self._message_router[method] = handler
        
        # Register with all channels
        for channel in self._channels.values():
            channel.register_message_handler(method, handler)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        status = {
            "total_connections": len(self._active_connections),
            "connections": {},
            "channels": {}
        }
        
        for ide_id, channel_name in self._active_connections.items():
            channel = self._channels[channel_name]
            status["connections"][ide_id] = {
                "channel": channel_name,
                "state": channel.state.value,
                "last_heartbeat": channel._last_heartbeat.isoformat() if channel._last_heartbeat else None
            }
        
        for name, channel in self._channels.items():
            status["channels"][name] = {
                "state": channel.state.value,
                "message_queue_size": channel._message_queue.qsize()
            }
        
        return status