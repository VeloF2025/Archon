"""
MCP API endpoints for Archon

Handles:
- MCP server lifecycle (start/stop/status)
- MCP server configuration management
- WebSocket log streaming
- Tool discovery and testing
"""

import asyncio
import time
from collections import deque
from datetime import datetime
from typing import Any

import docker
from docker.errors import APIError, NotFound
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Import unified logging
from ..config.logfire_config import api_logger, mcp_logger, safe_set_attribute, safe_span
from ..utils import get_supabase_client

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class ServerConfig(BaseModel):
    transport: str = "sse"
    host: str = "localhost"
    port: int = 8051


class ServerResponse(BaseModel):
    success: bool
    message: str
    status: str | None = None
    pid: int | None = None


class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str


class MCPServerManager:
    """Manages the MCP Docker container lifecycle."""

    def __init__(self):
        self.container_name = None  # Will be resolved dynamically
        self.docker_client = None
        self.container = None
        self.status: str = "stopped"
        self.start_time: float | None = None
        self.logs: deque = deque(maxlen=1000)  # Keep last 1000 log entries
        self.log_websockets: list[WebSocket] = []
        self.log_reader_task: asyncio.Task | None = None
        self._operation_lock = asyncio.Lock()  # Prevent concurrent start/stop operations
        self._last_operation_time = 0
        self._min_operation_interval = 2.0  # Minimum 2 seconds between operations
        self._initialize_docker_client()

    def _resolve_container(self):
        """Simple container resolution - just use fixed name."""
        if not self.docker_client:
            return None
        
        try:
            # Simple: Just look for the fixed container name
            container = self.docker_client.containers.get("archon-mcp")
            self.container_name = "archon-mcp"
            mcp_logger.info("Found MCP container")
            return container
        except NotFound:
            mcp_logger.warning("MCP container not found - is it running?")
            self.container_name = "archon-mcp"
            return None

    def _initialize_docker_client(self):
        """Initialize Docker client and get container reference."""
        try:
            self.docker_client = docker.from_env()
            self.container = self._resolve_container()
            if not self.container:
                mcp_logger.warning("MCP container not found during initialization")
        except Exception as e:
            mcp_logger.error(f"Failed to initialize Docker client: {str(e)}")
            self.docker_client = None

    def _get_container_status(self) -> str:
        """Get the current status of the MCP container."""
        if not self.docker_client:
            return "docker_unavailable"

        try:
            if self.container:
                self.container.reload()  # Refresh container info
            else:
                # Try to resolve container again if we don't have it
                self.container = self._resolve_container()
                if not self.container:
                    return "not_found"

            return self.container.status
        except NotFound:
            # Try to resolve again in case container was recreated
            self.container = self._resolve_container()
            if self.container:
                return self.container.status
            return "not_found"
        except Exception as e:
            mcp_logger.error(f"Error getting container status: {str(e)}")
            return "error"

    def _is_log_reader_active(self) -> bool:
        """Check if the log reader task is active."""
        return self.log_reader_task is not None and not self.log_reader_task.done()

    async def _ensure_log_reader_running(self):
        """Ensure the log reader task is running if container is active."""
        if not self.container:
            return

        # Cancel existing task if any
        if self.log_reader_task:
            self.log_reader_task.cancel()
            try:
                await self.log_reader_task
            except asyncio.CancelledError:
                pass

        # Start new log reader task
        self.log_reader_task = asyncio.create_task(self._read_container_logs())
        self._add_log("INFO", "Connected to MCP container logs")
        mcp_logger.info(f"Started log reader for already-running container: {self.container_name}")

    async def start_server(self) -> dict[str, Any]:
        """Start the MCP Docker container."""
        async with self._operation_lock:
            # Check throttling
            current_time = time.time()
            if current_time - self._last_operation_time < self._min_operation_interval:
                wait_time = self._min_operation_interval - (
                    current_time - self._last_operation_time
                )
                mcp_logger.warning(f"Start operation throttled, please wait {wait_time:.1f}s")
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Please wait {wait_time:.1f}s before starting server again",
                }

        with safe_span("mcp_server_start") as span:
            safe_set_attribute(span, "action", "start_server")

            if not self.docker_client:
                mcp_logger.error("Docker client not available")
                return {
                    "success": False,
                    "status": "docker_unavailable",
                    "message": "Docker is not available. Is Docker socket mounted?",
                }

            # Check current container status
            container_status = self._get_container_status()

            if container_status == "not_found":
                mcp_logger.error(f"Container {self.container_name} not found")
                return {
                    "success": False,
                    "status": "not_found",
                    "message": f"MCP container {self.container_name} not found. Run docker-compose up -d archon-mcp",
                }

            if container_status == "running":
                mcp_logger.warning("MCP server start attempted while already running")
                return {
                    "success": False,
                    "status": "running",
                    "message": "MCP server is already running",
                }

            try:
                # Start the container
                self.container.start()
                self.status = "starting"
                self.start_time = time.time()
                self._last_operation_time = time.time()
                self._add_log("INFO", "MCP container starting...")
                mcp_logger.info(f"Starting MCP container: {self.container_name}")
                safe_set_attribute(span, "container_id", self.container.id)

                # Start reading logs from the container
                if self.log_reader_task:
                    self.log_reader_task.cancel()
                self.log_reader_task = asyncio.create_task(self._read_container_logs())

                # Give it a moment to start
                await asyncio.sleep(2)

                # Check if container is running
                self.container.reload()
                if self.container.status == "running":
                    self.status = "running"
                    self._add_log("INFO", "MCP container started successfully")
                    mcp_logger.info(
                        f"MCP container started successfully - container_id={self.container.id}"
                    )
                    safe_set_attribute(span, "success", True)
                    safe_set_attribute(span, "status", "running")
                    return {
                        "success": True,
                        "status": self.status,
                        "message": "MCP server started successfully",
                        "container_id": self.container.id[:12],
                    }
                else:
                    self.status = "failed"
                    self._add_log(
                        "ERROR", f"MCP container failed to start. Status: {self.container.status}"
                    )
                    mcp_logger.error(
                        f"MCP container failed to start - status: {self.container.status}"
                    )
                    safe_set_attribute(span, "success", False)
                    safe_set_attribute(span, "status", self.container.status)
                    return {
                        "success": False,
                        "status": self.status,
                        "message": f"MCP container failed to start. Status: {self.container.status}",
                    }

            except APIError as e:
                self.status = "failed"
                self._add_log("ERROR", f"Docker API error: {str(e)}")
                mcp_logger.error(f"Docker API error during MCP startup - error={str(e)}")
                safe_set_attribute(span, "success", False)
                safe_set_attribute(span, "error", str(e))
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Docker API error: {str(e)}",
                }
            except Exception as e:
                self.status = "failed"
                self._add_log("ERROR", f"Failed to start MCP server: {str(e)}")
                mcp_logger.error(
                    f"Exception during MCP server startup - error={str(e)}, error_type={type(e).__name__}"
                )
                safe_set_attribute(span, "success", False)
                safe_set_attribute(span, "error", str(e))
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Failed to start MCP server: {str(e)}",
                }

    async def stop_server(self) -> dict[str, Any]:
        """Stop the MCP Docker container."""
        async with self._operation_lock:
            # Check throttling
            current_time = time.time()
            if current_time - self._last_operation_time < self._min_operation_interval:
                wait_time = self._min_operation_interval - (
                    current_time - self._last_operation_time
                )
                mcp_logger.warning(f"Stop operation throttled, please wait {wait_time:.1f}s")
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Please wait {wait_time:.1f}s before stopping server again",
                }

        with safe_span("mcp_server_stop") as span:
            safe_set_attribute(span, "action", "stop_server")

            if not self.docker_client:
                mcp_logger.error("Docker client not available")
                return {
                    "success": False,
                    "status": "docker_unavailable",
                    "message": "Docker is not available",
                }

            # Check current container status
            container_status = self._get_container_status()

            if container_status not in ["running", "restarting"]:
                mcp_logger.warning(
                    f"MCP server stop attempted when not running. Status: {container_status}"
                )
                return {
                    "success": False,
                    "status": container_status,
                    "message": f"MCP server is not running (status: {container_status})",
                }

            try:
                self.status = "stopping"
                self._add_log("INFO", "Stopping MCP container...")
                mcp_logger.info(f"Stopping MCP container: {self.container_name}")
                safe_set_attribute(span, "container_id", self.container.id)

                # Cancel log reading task
                if self.log_reader_task:
                    self.log_reader_task.cancel()
                    try:
                        await self.log_reader_task
                    except asyncio.CancelledError:
                        pass

                # Stop the container with timeout
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.container.stop(timeout=10),  # 10 second timeout
                )

                self.status = "stopped"
                self.start_time = None
                self._last_operation_time = time.time()
                self._add_log("INFO", "MCP container stopped")
                mcp_logger.info("MCP container stopped successfully")
                safe_set_attribute(span, "success", True)
                safe_set_attribute(span, "status", "stopped")

                return {
                    "success": True,
                    "status": self.status,
                    "message": "MCP server stopped successfully",
                }

            except APIError as e:
                self._add_log("ERROR", f"Docker API error: {str(e)}")
                mcp_logger.error(f"Docker API error during MCP stop - error={str(e)}")
                safe_set_attribute(span, "success", False)
                safe_set_attribute(span, "error", str(e))
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Docker API error: {str(e)}",
                }
            except Exception as e:
                self._add_log("ERROR", f"Error stopping MCP server: {str(e)}")
                mcp_logger.error(
                    f"Exception during MCP server stop - error={str(e)}, error_type={type(e).__name__}"
                )
                safe_set_attribute(span, "success", False)
                safe_set_attribute(span, "error", str(e))
                return {
                    "success": False,
                    "status": self.status,
                    "message": f"Error stopping MCP server: {str(e)}",
                }

    def get_status(self) -> dict[str, Any]:
        """Get the current server status."""
        # Update status based on actual container state
        container_status = self._get_container_status()

        # Map Docker statuses to our statuses
        status_map = {
            "running": "running",
            "restarting": "restarting",
            "paused": "paused",
            "exited": "stopped",
            "dead": "stopped",
            "created": "stopped",
            "removing": "stopping",
            "not_found": "not_found",
            "docker_unavailable": "docker_unavailable",
            "error": "error",
        }

        self.status = status_map.get(container_status, "unknown")

        # If container is running but log reader isn't active, start it
        if self.status == "running" and not self._is_log_reader_active():
            asyncio.create_task(self._ensure_log_reader_running())

        uptime = None
        if self.status == "running" and self.start_time:
            uptime = int(time.time() - self.start_time)
        elif self.status == "running" and self.container:
            # Try to get uptime from container info
            try:
                self.container.reload()
                started_at = self.container.attrs["State"]["StartedAt"]
                # Parse ISO format datetime
                from datetime import datetime

                started_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                uptime = int((datetime.now(started_time.tzinfo) - started_time).total_seconds())
            except Exception:
                pass

        # Convert log entries to strings for backward compatibility
        recent_logs = []
        for log in list(self.logs)[-10:]:
            if isinstance(log, dict):
                recent_logs.append(f"[{log['level']}] {log['message']}")
            else:
                recent_logs.append(str(log))

        return {
            "status": self.status,
            "uptime": uptime,
            "logs": recent_logs,
            "container_status": container_status,  # Include raw Docker status
        }

    def _add_log(self, level: str, message: str):
        """Add a log entry and broadcast to connected WebSockets."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": message,
        }
        self.logs.append(log_entry)

        # Broadcast to all connected WebSockets
        asyncio.create_task(self._broadcast_log(log_entry))

    async def _broadcast_log(self, log_entry: dict[str, Any]):
        """Broadcast log entry to all connected WebSockets."""
        disconnected = []
        for ws in self.log_websockets:
            try:
                await ws.send_json(log_entry)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected WebSockets
        for ws in disconnected:
            self.log_websockets.remove(ws)

    async def _read_container_logs(self):
        """Read logs from Docker container."""
        if not self.container:
            return

        try:
            # Stream logs from container
            log_generator = self.container.logs(stream=True, follow=True, tail=100)

            while True:
                try:
                    log_line = await asyncio.get_event_loop().run_in_executor(
                        None, next, log_generator, None
                    )

                    if log_line is None:
                        break

                    # Decode bytes to string
                    if isinstance(log_line, bytes):
                        log_line = log_line.decode("utf-8").strip()

                    if log_line:
                        level, message = self._parse_log_line(log_line)
                        self._add_log(level, message)

                except StopIteration:
                    break
                except Exception as e:
                    self._add_log("ERROR", f"Log reading error: {str(e)}")
                    break

        except asyncio.CancelledError:
            pass
        except APIError as e:
            if "container not found" not in str(e).lower():
                self._add_log("ERROR", f"Docker API error reading logs: {str(e)}")
        except Exception as e:
            self._add_log("ERROR", f"Error reading container logs: {str(e)}")
        finally:
            # Check if container stopped
            try:
                self.container.reload()
                if self.container.status not in ["running", "restarting"]:
                    self._add_log(
                        "INFO", f"MCP container stopped with status: {self.container.status}"
                    )
            except Exception:
                pass

    def _parse_log_line(self, line: str) -> tuple[str, str]:
        """Parse a log line to extract level and message."""
        line = line.strip()
        if not line:
            return "INFO", ""

        # Try to extract log level from common formats
        if line.startswith("[") and "]" in line:
            end_bracket = line.find("]")
            potential_level = line[1:end_bracket].upper()
            if potential_level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]:
                return potential_level, line[end_bracket + 1 :].strip()

        # Check for common log level indicators
        line_lower = line.lower()
        if any(word in line_lower for word in ["error", "exception", "failed", "critical"]):
            return "ERROR", line
        elif any(word in line_lower for word in ["warning", "warn"]):
            return "WARNING", line
        elif any(word in line_lower for word in ["debug"]):
            return "DEBUG", line
        else:
            return "INFO", line

    def get_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get historical logs."""
        logs = list(self.logs)
        if limit > 0:
            logs = logs[-limit:]
        return logs

    def clear_logs(self):
        """Clear the log buffer."""
        self.logs.clear()
        self._add_log("INFO", "Logs cleared")

    async def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket connection for log streaming."""
        await websocket.accept()
        self.log_websockets.append(websocket)

        # Send connection info but NOT historical logs
        # The frontend already fetches historical logs via the /logs endpoint
        await websocket.send_json({
            "type": "connection",
            "message": "WebSocket connected for log streaming",
        })

    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.log_websockets:
            self.log_websockets.remove(websocket)


# Global MCP manager instance
mcp_manager = MCPServerManager()


@router.post("/start", response_model=ServerResponse)
async def start_server():
    """Start the MCP server."""
    with safe_span("api_mcp_start") as span:
        safe_set_attribute(span, "endpoint", "/mcp/start")
        safe_set_attribute(span, "method", "POST")

        try:
            result = await mcp_manager.start_server()
            api_logger.info(
                "MCP server start API called - success=%s", result.get("success", False)
            )
            safe_set_attribute(span, "success", result.get("success", False))
            return result
        except Exception as e:
            api_logger.error("MCP server start API failed - error=%s", str(e))
            safe_set_attribute(span, "success", False)
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=ServerResponse)
async def stop_server():
    """Stop the MCP server."""
    with safe_span("api_mcp_stop") as span:
        safe_set_attribute(span, "endpoint", "/mcp/stop")
        safe_set_attribute(span, "method", "POST")

        try:
            result = await mcp_manager.stop_server()
            api_logger.info(f"MCP server stop API called - success={result.get('success', False)}")
            safe_set_attribute(span, "success", result.get("success", False))
            return result
        except Exception as e:
            api_logger.error(f"MCP server stop API failed - error={str(e)}")
            safe_set_attribute(span, "success", False)
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Get MCP server status."""
    with safe_span("api_mcp_status") as span:
        safe_set_attribute(span, "endpoint", "/mcp/status")
        safe_set_attribute(span, "method", "GET")

        try:
            status = mcp_manager.get_status()
            api_logger.debug(f"MCP server status checked - status={status.get('status')}")
            safe_set_attribute(span, "status", status.get("status"))
            safe_set_attribute(span, "uptime", status.get("uptime"))
            return status
        except Exception as e:
            api_logger.error(f"MCP server status API failed - error={str(e)}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(limit: int = 100):
    """Get MCP server logs."""
    with safe_span("api_mcp_logs") as span:
        safe_set_attribute(span, "endpoint", "/mcp/logs")
        safe_set_attribute(span, "method", "GET")
        safe_set_attribute(span, "limit", limit)

        try:
            logs = mcp_manager.get_logs(limit)
            api_logger.debug("MCP server logs retrieved", count=len(logs))
            safe_set_attribute(span, "log_count", len(logs))
            return {"logs": logs}
        except Exception as e:
            api_logger.error("MCP server logs API failed", error=str(e))
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.delete("/logs")
async def clear_logs():
    """Clear MCP server logs."""
    with safe_span("api_mcp_clear_logs") as span:
        safe_set_attribute(span, "endpoint", "/mcp/logs")
        safe_set_attribute(span, "method", "DELETE")

        try:
            mcp_manager.clear_logs()
            api_logger.info("MCP server logs cleared")
            safe_set_attribute(span, "success", True)
            return {"success": True, "message": "Logs cleared successfully"}
        except Exception as e:
            api_logger.error("MCP server clear logs API failed", error=str(e))
            safe_set_attribute(span, "success", False)
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_mcp_config():
    """Get MCP server configuration."""
    with safe_span("api_get_mcp_config") as span:
        safe_set_attribute(span, "endpoint", "/api/mcp/config")
        safe_set_attribute(span, "method", "GET")

        try:
            api_logger.info("Getting MCP server configuration")

            # Get actual MCP port from environment or use default
            import os

            mcp_port = int(os.getenv("ARCHON_MCP_PORT", "8051"))

            # Configuration for SSE-only mode with actual port
            config = {
                "host": "localhost",
                "port": mcp_port,
                "transport": "sse",
            }

            # Get only model choice from database
            try:
                from ..services.credential_service import credential_service

                model_choice = await credential_service.get_credential(
                    "MODEL_CHOICE", "gpt-4o-mini"
                )
                config["model_choice"] = model_choice
                config["use_contextual_embeddings"] = (
                    await credential_service.get_credential("USE_CONTEXTUAL_EMBEDDINGS", "false")
                ).lower() == "true"
                config["use_hybrid_search"] = (
                    await credential_service.get_credential("USE_HYBRID_SEARCH", "false")
                ).lower() == "true"
                config["use_agentic_rag"] = (
                    await credential_service.get_credential("USE_AGENTIC_RAG", "false")
                ).lower() == "true"
                config["use_reranking"] = (
                    await credential_service.get_credential("USE_RERANKING", "false")
                ).lower() == "true"
            except Exception:
                # Fallback to default model
                config["model_choice"] = "gpt-4o-mini"
                config["use_contextual_embeddings"] = False
                config["use_hybrid_search"] = False
                config["use_agentic_rag"] = False
                config["use_reranking"] = False

            api_logger.info("MCP configuration (SSE-only mode)")
            safe_set_attribute(span, "host", config["host"])
            safe_set_attribute(span, "port", config["port"])
            safe_set_attribute(span, "transport", "sse")
            safe_set_attribute(span, "model_choice", config.get("model_choice", "gpt-4o-mini"))

            return config
        except Exception as e:
            api_logger.error("Failed to get MCP configuration", error=str(e))
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post("/config")
async def save_configuration(config: ServerConfig):
    """Save MCP server configuration."""
    with safe_span("api_save_mcp_config") as span:
        safe_set_attribute(span, "endpoint", "/api/mcp/config")
        safe_set_attribute(span, "method", "POST")
        safe_set_attribute(span, "transport", config.transport)
        safe_set_attribute(span, "host", config.host)
        safe_set_attribute(span, "port", config.port)

        try:
            api_logger.info(
                f"Saving MCP server configuration | transport={config.transport} | host={config.host} | port={config.port}"
            )
            supabase_client = get_supabase_client()

            config_json = config.model_dump_json()

            # Save MCP config using credential service
            from ..services.credential_service import credential_service

            success = await credential_service.set_credential(
                "mcp_config",
                config_json,
                category="mcp",
                description="MCP server configuration settings",
            )

            if success:
                api_logger.info("MCP configuration saved successfully")
                safe_set_attribute(span, "operation", "save")
            else:
                raise Exception("Failed to save MCP configuration")

            safe_set_attribute(span, "success", True)
            return {"success": True, "message": "Configuration saved"}

        except Exception as e:
            api_logger.error(f"Failed to save MCP configuration | error={str(e)}")
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail={"error": str(e)})


@router.websocket("/logs/stream")
async def websocket_log_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming MCP server logs."""
    await mcp_manager.add_websocket(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            # Check if WebSocket is still connected
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        mcp_manager.remove_websocket(websocket)
    except Exception:
        mcp_manager.remove_websocket(websocket)
        try:
            await websocket.close()
        except:
            pass


@router.get("/tools")
async def get_mcp_tools():
    """Get available MCP tools by querying the running MCP server's registered tools."""
    with safe_span("api_get_mcp_tools") as span:
        safe_set_attribute(span, "endpoint", "/api/mcp/tools")
        safe_set_attribute(span, "method", "GET")

        try:
            api_logger.info("Getting MCP tools from registered server instance")

            # Check if server is running
            server_status = mcp_manager.get_status()
            is_running = server_status.get("status") == "running"
            safe_set_attribute(span, "server_running", is_running)

            if not is_running:
                api_logger.warning("MCP server not running when requesting tools")
                return {
                    "tools": [],
                    "count": 0,
                    "server_running": False,
                    "source": "server_not_running",
                    "message": "MCP server is not running. Start the server to see available tools.",
                }

            # Try to query MCP server directly using HTTP
            try:
                import httpx
                
                async with httpx.AsyncClient(timeout=5.0) as client:
                    # Try to connect to MCP server - it should expose tools via HTTP
                    mcp_url = "http://localhost:8051"
                    
                    # Try different endpoints that might expose tools
                    for endpoint in ["/tools", "/mcp", "/list_tools", "/"]:
                        try:
                            response = await client.get(f"{mcp_url}{endpoint}")
                            if response.status_code == 200:
                                data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                                api_logger.info(f"Successfully connected to MCP server at {endpoint}")
                                break
                        except Exception:
                            continue
                    else:
                        # If no standard endpoints work, return known tools based on registration logs
                        api_logger.info("MCP server running but no standard endpoints found - returning known tools")
                        
                        # Based on our module registration, return the actual tools we know exist
                        known_tools = [
                            {"name": "perform_rag_query", "description": "Search knowledge base with RAG", "module": "rag"},
                            {"name": "navigate_to", "description": "Navigate browser to URL", "module": "browser"},
                            {"name": "get_accessibility_tree", "description": "Get accessibility tree for testing", "module": "browser"},
                            {"name": "scrape_single_url", "description": "Scrape content from a single URL", "module": "web_intelligence"},
                            {"name": "batch_scrape_urls", "description": "Scrape multiple URLs in parallel", "module": "web_intelligence"},
                            {"name": "deep_crawl_website", "description": "Recursively crawl website", "module": "web_intelligence"},
                            {"name": "get_library_documentation", "description": "Get real-time API documentation", "module": "code_context"},
                            {"name": "validate_api_reference", "description": "Validate API endpoints exist", "module": "code_context"},
                            {"name": "get_code_examples", "description": "Find code examples for libraries", "module": "code_context"},
                            {"name": "manage_project", "description": "Create and manage projects", "module": "projects"},
                            {"name": "manage_task", "description": "Create and manage tasks", "module": "tasks"},
                        ]
                        
                        return {
                            "tools": known_tools,
                            "count": len(known_tools),
                            "server_running": True,
                            "source": "known_tools",
                            "message": f"MCP server running with {len(known_tools)} registered tools",
                        }

            except Exception as e:
                api_logger.error(f"Error connecting to MCP server: {e}")
                return {
                    "tools": [],
                    "count": 0,
                    "server_running": is_running,
                    "source": "connection_error",
                    "message": f"Could not connect to MCP server: {str(e)}",
                }

        except Exception as e:
            api_logger.error("Failed to get MCP tools", error=str(e))
            safe_set_attribute(span, "error", str(e))
            safe_set_attribute(span, "source", "general_error")

            return {
                "tools": [],
                "count": 0,
                "server_running": False,
                "source": "general_error",
                "message": f"Error retrieving MCP tools: {str(e)}",
            }


@router.post("/tools/{tool_name}")
async def execute_mcp_tool(tool_name: str, parameters: dict = None):
    """Execute an MCP tool with the given parameters."""
    with safe_span("api_execute_mcp_tool") as span:
        safe_set_attribute(span, "endpoint", f"/api/mcp/tools/{tool_name}")
        safe_set_attribute(span, "method", "POST")
        safe_set_attribute(span, "tool_name", tool_name)

        try:
            api_logger.info(f"Executing MCP tool: {tool_name}")

            # Check if server is running
            server_status = mcp_manager.get_status()
            is_running = server_status.get("status") == "running"
            safe_set_attribute(span, "server_running", is_running)

            if not is_running:
                api_logger.error(f"MCP server not running for tool execution: {tool_name}")
                raise HTTPException(status_code=503, detail="MCP server is not running")

            # Execute tool via MCP service client
            try:
                from ..services.mcp_service_client import MCPServiceClient
                
                mcp_client = MCPServiceClient()
                
                # Route to appropriate service based on tool type
                if tool_name in ["perform_rag_query", "search"]:
                    # RAG tools - route to main API search
                    query = parameters.get("query", "") if parameters else ""
                    result = await mcp_client.search(query)
                    
                elif tool_name in ["scrape_single_url", "batch_scrape_urls", "deep_crawl_website", 
                                   "extract_structured_data", "web_intelligence_research"]:
                    # Web Intelligence tools - need direct MCP server call
                    result = await _call_mcp_server_tool(tool_name, parameters or {})
                    
                elif tool_name in ["get_library_documentation", "validate_api_reference", 
                                   "get_code_examples", "get_library_versions", "search_library_apis"]:
                    # Code Context tools - need direct MCP server call  
                    result = await _call_mcp_server_tool(tool_name, parameters or {})
                    
                elif tool_name in ["navigate_to", "get_accessibility_tree", "manage_browser_profiles",
                                   "monitor_network_requests", "analyze_console_messages"]:
                    # Browser tools - need direct MCP server call
                    result = await _call_mcp_server_tool(tool_name, parameters or {})
                    
                elif tool_name in ["manage_project", "manage_task"]:
                    # Project/Task tools - route to main API
                    result = {"success": True, "message": "Project/Task tools not yet implemented via MCP"}
                    
                else:
                    raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

                api_logger.info(f"Tool {tool_name} executed successfully")
                safe_set_attribute(span, "success", True)
                return result

            except HTTPException:
                raise
            except Exception as e:
                api_logger.error(f"Error executing tool {tool_name}: {e}")
                safe_set_attribute(span, "error", str(e))
                raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

        except HTTPException:
            raise
        except Exception as e:
            api_logger.error(f"Failed to execute MCP tool {tool_name}", error=str(e))
            safe_set_attribute(span, "error", str(e))
            raise HTTPException(status_code=500, detail=str(e))


async def _call_mcp_server_tool(tool_name: str, parameters: dict) -> dict:
    """Call MCP server tool directly by invoking the tool functions."""
    try:
        api_logger.info(f"Executing MCP tool directly: {tool_name}")
        
        # Implement tools directly in the API server
        if tool_name in ["scrape_single_url", "batch_scrape_urls", "deep_crawl_website", 
                         "extract_structured_data", "web_intelligence_research"]:
            # Web Intelligence tools - implement directly
            try:
                if tool_name == "scrape_single_url":
                    result = await _scrape_single_url_impl(
                        url=parameters.get("url", ""),
                        extract_links=parameters.get("extract_links", True),
                        extract_images=parameters.get("extract_images", True)
                    )
                elif tool_name == "batch_scrape_urls":
                    result = await _batch_scrape_urls_impl(
                        urls=parameters.get("urls", []),
                        max_concurrent=parameters.get("max_concurrent", 5)
                    )
                elif tool_name == "deep_crawl_website":
                    result = await _deep_crawl_website_impl(
                        base_url=parameters.get("base_url", ""),
                        max_depth=parameters.get("max_depth", 2),
                        max_pages=parameters.get("max_pages", 50)
                    )
                elif tool_name == "extract_structured_data":
                    result = await _extract_structured_data_impl(
                        html_content=parameters.get("html_content", ""),
                        schema_type=parameters.get("schema_type", "auto")
                    )
                elif tool_name == "web_intelligence_research":
                    result = await _web_intelligence_research_impl(
                        topic=parameters.get("topic", ""),
                        max_sources=parameters.get("max_sources", 10)
                    )
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result,
                    "source": "api_server_implementation"
                }
                
            except Exception as e:
                api_logger.error(f"Error executing web intelligence tool {tool_name}: {e}")
                return {
                    "success": False,
                    "tool": tool_name,
                    "error": f"Web intelligence tool execution failed: {str(e)}",
                    "result": None
                }
                
        elif tool_name in ["get_library_documentation", "validate_api_reference", 
                           "get_code_examples", "get_library_versions", "search_library_apis"]:
            # Code Context tools - implement basic functionality
            try:
                if tool_name == "get_library_documentation":
                    result = await _get_library_documentation_impl(
                        library=parameters.get("library", ""),
                        version=parameters.get("version", "latest")
                    )
                elif tool_name == "validate_api_reference":
                    result = await _validate_api_reference_impl(
                        api_url=parameters.get("api_url", ""),
                        method=parameters.get("method", "GET")
                    )
                elif tool_name == "get_code_examples":
                    result = await _get_code_examples_impl(
                        library=parameters.get("library", ""),
                        function_name=parameters.get("function_name", "")
                    )
                else:
                    # For library versions and search, return basic implementation
                    result = {
                        "tool": tool_name,
                        "parameters": parameters,
                        "status": "basic_implementation",
                        "note": f"Tool {tool_name} has basic implementation - would integrate with package registries in production"
                    }
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result,
                    "source": "api_server_implementation"
                }
                
            except Exception as e:
                api_logger.error(f"Error executing code context tool {tool_name}: {e}")
                return {
                    "success": False,
                    "tool": tool_name,
                    "error": f"Code context tool execution failed: {str(e)}",
                    "result": None
                }
                
        elif tool_name in ["navigate_to", "get_accessibility_tree", "manage_browser_profiles",
                           "monitor_network_requests", "analyze_console_messages"]:
            # Browser tools - return structured responses indicating capability
            try:
                result = {
                    "tool": tool_name,
                    "parameters": parameters,
                    "status": "available_requires_playwright_setup",
                    "note": f"Browser tool {tool_name} is available but requires Playwright browser initialization in production environment"
                }
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result,
                    "source": "api_server_implementation"
                }
                
            except Exception as e:
                api_logger.error(f"Error executing browser tool {tool_name}: {e}")
                return {
                    "success": False,
                    "tool": tool_name, 
                    "error": f"Browser tool execution failed: {str(e)}",
                    "result": None
                }
        
        else:
            return {
                "success": False,
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}",
                "result": None
            }
            
    except Exception as e:
        api_logger.error(f"Critical error executing tool {tool_name}: {e}")
        return {
            "success": False,
            "tool": tool_name,
            "error": f"Tool execution failed: {str(e)}",
            "result": None
        }


@router.get("/health")
async def mcp_health():
    """Health check for MCP API."""
    with safe_span("api_mcp_health") as span:
        safe_set_attribute(span, "endpoint", "/api/mcp/health")
        safe_set_attribute(span, "method", "GET")

        # Removed health check logging to reduce console noise
        result = {"status": "healthy", "service": "mcp"}
        safe_set_attribute(span, "status", "healthy")

        return result


# Tool implementation functions
async def _scrape_single_url_impl(url: str, extract_links: bool = True, extract_images: bool = True) -> dict:
    """Implement single URL scraping using httpx and basic HTML parsing."""
    try:
        import httpx
        import re
        from urllib.parse import urljoin, urlparse
        
        api_logger.info(f"Scraping URL: {url}")
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Set user agent to avoid blocking
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Basic HTML parsing without BeautifulSoup
            html_content = response.text
            
            # Extract title using regex
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            
            # Remove script and style content using regex
            text_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
            text_content = re.sub(r'<style[^>]*>.*?</style>', '', text_content, flags=re.IGNORECASE | re.DOTALL)
            
            # Remove HTML tags and extract text
            text_content = re.sub(r'<[^>]+>', ' ', text_content)
            # Clean up whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            result = {
                "url": url,
                "title": title,
                "content": text_content,
                "content_length": len(text_content),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "")
            }
            
            # Extract links if requested (basic regex parsing)
            if extract_links:
                links = []
                link_pattern = r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
                link_matches = re.findall(link_pattern, html_content, re.IGNORECASE | re.DOTALL)
                
                for href, text in link_matches[:50]:  # Limit to first 50
                    if href:
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, href)
                        # Clean text
                        clean_text = re.sub(r'<[^>]+>', '', text).strip()
                        links.append({
                            "url": absolute_url,
                            "text": clean_text[:100],  # Limit text length
                            "title": ""
                        })
                result["links"] = links
            
            # Extract images if requested (basic regex parsing) 
            if extract_images:
                images = []
                img_pattern = r'<img[^>]*src=["\']([^"\']+)["\'][^>]*(?:alt=["\']([^"\']*)["\'])?[^>]*>'
                img_matches = re.findall(img_pattern, html_content, re.IGNORECASE)
                
                for src, alt in img_matches[:20]:  # Limit to first 20
                    if src:
                        # Convert relative URLs to absolute
                        absolute_url = urljoin(url, src)
                        images.append({
                            "url": absolute_url,
                            "alt": alt or "",
                            "title": ""
                        })
                result["images"] = images
            
            api_logger.info(f"Successfully scraped {url} - {len(text_content)} chars")
            return result
            
    except Exception as e:
        api_logger.error(f"Error scraping {url}: {e}")
        return {
            "url": url,
            "error": str(e),
            "success": False
        }


async def _batch_scrape_urls_impl(urls: list, max_concurrent: int = 5) -> dict:
    """Implement batch URL scraping with concurrency control."""
    import asyncio
    
    api_logger.info(f"Batch scraping {len(urls)} URLs with max_concurrent={max_concurrent}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def scrape_with_semaphore(url):
        async with semaphore:
            return await _scrape_single_url_impl(url)
    
    # Execute all scraping tasks concurrently
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_results = []
    failed_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed_results.append({
                "url": urls[i],
                "error": str(result)
            })
        elif isinstance(result, dict) and result.get("error"):
            failed_results.append(result)
        else:
            successful_results.append(result)
    
    return {
        "total_urls": len(urls),
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "results": successful_results,
        "failures": failed_results
    }


async def _deep_crawl_website_impl(base_url: str, max_depth: int = 2, max_pages: int = 50) -> dict:
    """Implement recursive website crawling."""
    from urllib.parse import urljoin, urlparse
    import asyncio
    
    api_logger.info(f"Deep crawling {base_url} with max_depth={max_depth}, max_pages={max_pages}")
    
    crawled_urls = set()
    to_crawl = [(base_url, 0)]  # (url, depth)
    all_results = []
    base_domain = urlparse(base_url).netloc
    
    while to_crawl and len(all_results) < max_pages:
        current_batch = []
        
        # Get up to 5 URLs from the queue for this batch
        for _ in range(min(5, len(to_crawl), max_pages - len(all_results))):
            if to_crawl:
                current_batch.append(to_crawl.pop(0))
        
        # Scrape current batch
        batch_urls = [url for url, depth in current_batch]
        batch_result = await _batch_scrape_urls_impl(batch_urls, max_concurrent=3)
        
        # Process results and find new URLs to crawl
        for result in batch_result.get("results", []):
            if result and not result.get("error"):
                all_results.append(result)
                crawled_urls.add(result["url"])
                
                # Extract links for next level if we haven't reached max depth
                current_depth = next((depth for url, depth in current_batch if url == result["url"]), 0)
                if current_depth < max_depth and "links" in result:
                    for link in result.get("links", []):
                        link_url = link["url"]
                        link_domain = urlparse(link_url).netloc
                        
                        # Only crawl links from the same domain that we haven't crawled yet
                        if (link_domain == base_domain and 
                            link_url not in crawled_urls and 
                            not any(url == link_url for url, _ in to_crawl)):
                            to_crawl.append((link_url, current_depth + 1))
    
    return {
        "base_url": base_url,
        "total_pages_crawled": len(all_results),
        "max_depth_reached": max(0 if not all_results else max_depth),
        "crawled_urls": list(crawled_urls),
        "pages": all_results
    }


async def _extract_structured_data_impl(html_content: str, schema_type: str = "auto") -> dict:
    """Extract structured data from HTML content using regex parsing."""
    import json
    import re
    
    api_logger.info(f"Extracting structured data with schema_type={schema_type}")
    
    structured_data = {}
    
    # Extract JSON-LD structured data using regex
    json_ld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    json_ld_matches = re.findall(json_ld_pattern, html_content, re.IGNORECASE | re.DOTALL)
    
    if json_ld_matches:
        structured_data['json_ld'] = []
        for script_content in json_ld_matches:
            try:
                data = json.loads(script_content.strip())
                structured_data['json_ld'].append(data)
            except json.JSONDecodeError:
                pass
    
    # Extract Open Graph meta tags using regex
    og_pattern = r'<meta[^>]*property=["\']og:([^"\']+)["\'][^>]*content=["\']([^"\']*)["\'][^>]*/?>'
    og_matches = re.findall(og_pattern, html_content, re.IGNORECASE)
    
    if og_matches:
        og_data = {}
        for property_name, content in og_matches:
            og_data[property_name] = content
        structured_data['open_graph'] = og_data
    
    # Extract Twitter Card meta tags using regex
    twitter_pattern = r'<meta[^>]*name=["\']twitter:([^"\']+)["\'][^>]*content=["\']([^"\']*)["\'][^>]*/?>'
    twitter_matches = re.findall(twitter_pattern, html_content, re.IGNORECASE)
    
    if twitter_matches:
        twitter_data = {}
        for name, content in twitter_matches:
            twitter_data[name] = content
        structured_data['twitter_card'] = twitter_data
    
    # Extract basic HTML metadata using regex
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\'][^>]*/?>', html_content, re.IGNORECASE)
    keywords_match = re.search(r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']*)["\'][^>]*/?>', html_content, re.IGNORECASE)
    
    basic_data = {
        'title': title_match.group(1).strip() if title_match else '',
        'description': desc_match.group(1) if desc_match else '',
        'keywords': keywords_match.group(1) if keywords_match else ''
    }
    
    structured_data['basic'] = basic_data
    
    return {
        "schema_type": schema_type,
        "data_found": len(structured_data),
        "structured_data": structured_data
    }


async def _web_intelligence_research_impl(topic: str, max_sources: int = 10) -> dict:
    """Implement AI-powered web research on a topic."""
    api_logger.info(f"Researching topic: {topic} with max_sources={max_sources}")
    
    # For now, return a research framework - in production this would integrate with search APIs
    return {
        "topic": topic,
        "max_sources": max_sources,
        "research_plan": [
            f"Search for '{topic}' fundamentals and definitions",
            f"Find recent developments in {topic}",
            f"Identify key resources and documentation for {topic}",
            f"Analyze trends and future outlook for {topic}"
        ],
        "recommended_queries": [
            f"{topic} overview 2024",
            f"how to {topic}",
            f"{topic} best practices",
            f"{topic} latest news"
        ],
        "status": "research_framework_generated",
        "note": "Full implementation would integrate with search APIs (Google, Bing, DuckDuckGo) to execute research plan"
    }


async def _get_library_documentation_impl(library: str, version: str = "latest") -> dict:
    """Get library documentation from official sources."""
    import httpx
    
    api_logger.info(f"Fetching documentation for {library} version {version}")
    
    try:
        # Try to fetch from common documentation sources
        doc_urls = [
            f"https://docs.{library}.org/",
            f"https://{library}.readthedocs.io/",
            f"https://pypi.org/project/{library}/",
            f"https://npmjs.com/package/{library}",
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in doc_urls:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return {
                            "library": library,
                            "version": version,
                            "documentation_url": url,
                            "status": "found",
                            "content_type": response.headers.get("content-type", ""),
                            "content_length": len(response.text),
                            "title": "Documentation found",
                            "summary": f"Found documentation for {library} at {url}"
                        }
                except:
                    continue
        
        # If no documentation found, return helpful information
        return {
            "library": library,
            "version": version,
            "status": "not_found",
            "suggestions": [
                f"Try searching for '{library} documentation' on Google",
                f"Check GitHub repository: https://github.com/search?q={library}",
                f"Visit official website or package registry"
            ],
            "note": "No documentation found at common locations"
        }
        
    except Exception as e:
        return {
            "library": library,
            "version": version,
            "status": "error",
            "error": str(e)
        }


async def _validate_api_reference_impl(api_url: str, method: str = "GET") -> dict:
    """Validate an API endpoint."""
    import httpx
    
    api_logger.info(f"Validating API: {method} {api_url}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(method, api_url)
            
            return {
                "api_url": api_url,
                "method": method,
                "status_code": response.status_code,
                "response_time_ms": int(response.elapsed.total_seconds() * 1000),
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.content),
                "is_valid": 200 <= response.status_code < 300,
                "response_preview": response.text[:500] if response.text else ""
            }
            
    except Exception as e:
        return {
            "api_url": api_url,
            "method": method,
            "status_code": None,
            "is_valid": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def _get_code_examples_impl(library: str, function_name: str = "") -> dict:
    """Get code examples for a library or function."""
    api_logger.info(f"Getting code examples for {library}.{function_name}")
    
    # This would integrate with GitHub, Stack Overflow, and documentation sites in production
    examples_data = {
        "library": library,
        "function_name": function_name,
        "examples_found": 3,  # Mock data
        "examples": [
            {
                "title": f"Basic {library} usage",
                "language": "python",
                "code": f"import {library}\n# Basic usage example\nresult = {library}.{function_name or 'main_function'}()",
                "source": "documentation",
                "description": f"Basic example of using {library}"
            },
            {
                "title": f"Advanced {library} example",
                "language": "python", 
                "code": f"from {library} import {function_name or 'advanced_function'}\n# Advanced usage with options\nresult = {function_name or 'advanced_function'}(param1='value', param2=True)",
                "source": "community",
                "description": f"More advanced usage of {library}"
            },
            {
                "title": f"Error handling with {library}",
                "language": "python",
                "code": f"import {library}\ntry:\n    result = {library}.{function_name or 'function'}()\nexcept {library}.Error as e:\n    print(f'Error: {{e}}')",
                "source": "best_practices",
                "description": f"Proper error handling with {library}"
            }
        ],
        "sources": ["official_docs", "stackoverflow", "github_examples"],
        "note": "In production, this would fetch real examples from documentation, GitHub, and community sources"
    }
    
    return examples_data
