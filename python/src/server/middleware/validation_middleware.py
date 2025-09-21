"""
Active Validation Middleware for DGTS and NLNH Protocols

This middleware actively enforces validation rules in real-time:
- DGTS: Prevents gaming the system
- NLNH: Ensures no lies or hallucinations
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...agents.validation.dgts_validator import DGTSValidator
from ...agents.validation.agent_behavior_monitor import AgentBehaviorMonitor
from ..socketio_app import get_socketio_instance

logger = logging.getLogger(__name__)


class NLNHValidator:
    """No Lies, No Hallucination validator"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.hallucination_patterns = [
            "I think", "probably", "might be", "could be",
            "I'm not sure but", "I believe", "it seems"
        ]
        self.lie_indicators = [
            "definitely works", "100% certain", "guaranteed to work",
            "no issues", "perfect solution", "bug-free"
        ]
    
    def validate_response(self, response_data: Any) -> Dict[str, Any]:
        """
        Validate response for lies and hallucinations
        Returns validation result with confidence score
        """
        validation_result = {
            "valid": True,
            "confidence": 1.0,
            "issues": [],
            "corrections": []
        }
        
        # Convert response to string for analysis
        response_str = json.dumps(response_data) if isinstance(response_data, dict) else str(response_data)
        response_lower = response_str.lower()
        
        # Check for hallucination patterns
        for pattern in self.hallucination_patterns:
            if pattern in response_lower:
                validation_result["confidence"] *= 0.8
                validation_result["issues"].append(f"Uncertain language detected: '{pattern}'")
                validation_result["corrections"].append("Use definitive language or explicitly state uncertainty")
        
        # Check for potential lies (overconfident statements)
        for indicator in self.lie_indicators:
            if indicator in response_lower:
                validation_result["confidence"] *= 0.6
                validation_result["issues"].append(f"Overconfident claim detected: '{indicator}'")
                validation_result["corrections"].append("Provide realistic assessment with known limitations")
        
        # Check for missing error handling acknowledgment
        if "error" not in response_lower and "exception" not in response_lower:
            if any(word in response_lower for word in ["success", "completed", "done"]):
                validation_result["confidence"] *= 0.9
                validation_result["issues"].append("Success claimed without error handling mention")
                validation_result["corrections"].append("Include error handling and edge cases")
        
        # Determine if response is valid based on confidence
        if validation_result["confidence"] < self.confidence_threshold:
            validation_result["valid"] = False
        
        return validation_result


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to actively enforce DGTS and NLNH validation
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.dgts_validator = DGTSValidator(project_path=".")  # Use current directory as project path
        self.nlnh_validator = NLNHValidator()
        self.behavior_monitor = AgentBehaviorMonitor(project_path=".")  # Use current directory as project path
        self.socketio = None
        self.validation_enabled = True
        self.monitoring_enabled = True
        
        # Paths that should be validated
        self.validated_paths = [
            "/api/agents/",
            "/api/tasks/",
            "/api/projects/",
            "/api/knowledge/",
            "/api/confidence/"
        ]
        
        # Paths to exclude from validation
        self.excluded_paths = [
            "/api/validation/",
            "/health",
            "/docs",
            "/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Intercept requests and responses for validation
        """
        # Skip validation for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Check if this path should be validated
        should_validate = any(request.url.path.startswith(path) for path in self.validated_paths)
        
        if not should_validate or not self.validation_enabled:
            return await call_next(request)
        
        # Initialize Socket.IO if not already done
        if self.socketio is None:
            self.socketio = get_socketio_instance()
        
        # Extract request data for validation
        request_data = {
            "path": request.url.path,
            "method": request.method,
            "timestamp": time.time()
        }
        
        # For POST/PUT requests, validate the request body
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body carefully to avoid issues
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        request_data["body"] = json.loads(body_bytes.decode())
                    except json.JSONDecodeError:
                        # Not JSON, skip validation
                        pass
                    
                    # Check for gaming patterns in request
                    if "body" in request_data and isinstance(request_data["body"], dict):
                        if "code" in request_data["body"] or "content" in request_data["body"]:
                            code_content = request_data["body"].get("code") or request_data["body"].get("content", "")
                            gaming_score = self.dgts_validator.calculate_gaming_score(code_content)
                            
                            if gaming_score > 0.3:
                                # Gaming detected - block request
                                await self._emit_validation_event("dgts_violation", {
                                    "path": request.url.path,
                                    "gaming_score": gaming_score,
                                    "action": "request_blocked",
                                    "timestamp": time.time()
                                })
                                
                                return JSONResponse(
                                    status_code=400,
                                    content={
                                        "error": "DGTS Validation Failed",
                                        "message": "Gaming patterns detected in request",
                                        "gaming_score": gaming_score,
                                        "details": "Request blocked due to attempts to game the system"
                                    }
                                )
            except Exception as e:
                logger.warning(f"Could not parse request body: {e}")
        
        # Monitor agent behavior
        if self.monitoring_enabled and "/agents/" in request.url.path:
            agent_id = self._extract_agent_id(request.url.path)
            if agent_id:
                # Check if agent is blocked
                if self.behavior_monitor.is_agent_blocked(agent_id):
                    blocked_until = self.behavior_monitor.blocked_agents.get(agent_id, {}).get("until", 0)
                    
                    await self._emit_validation_event("agent_blocked", {
                        "agent_id": agent_id,
                        "blocked_until": blocked_until,
                        "timestamp": time.time()
                    })
                    
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "Agent Blocked",
                            "message": f"Agent {agent_id} is blocked due to gaming behavior",
                            "blocked_until": blocked_until,
                            "details": "Agent must wait for block to expire or be manually unblocked"
                        }
                    )
                
                # Track agent action
                action = {
                    "timestamp": time.time(),
                    "action_type": request.method,
                    "target": request.url.path,
                    "details": request_data
                }
                
                suspicious = self.behavior_monitor.track_agent_action(agent_id, action)
                if suspicious:
                    await self._emit_validation_event("suspicious_behavior", {
                        "agent_id": agent_id,
                        "action": action,
                        "timestamp": time.time()
                    })
        
        # Process the request
        response = await call_next(request)
        
        # For successful responses, validate the response content
        if 200 <= response.status_code < 300:
            try:
                # Read response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                
                # Parse and validate response
                if response_body:
                    response_data = json.loads(response_body.decode())
                    
                    # NLNH validation
                    nlnh_result = self.nlnh_validator.validate_response(response_data)
                    
                    if not nlnh_result["valid"]:
                        # NLNH violation detected
                        await self._emit_validation_event("nlnh_violation", {
                            "path": request.url.path,
                            "confidence": nlnh_result["confidence"],
                            "issues": nlnh_result["issues"],
                            "corrections": nlnh_result["corrections"],
                            "timestamp": time.time()
                        })
                        
                        # Modify response to include validation warning
                        response_data["_validation_warning"] = {
                            "type": "NLNH",
                            "confidence": nlnh_result["confidence"],
                            "issues": nlnh_result["issues"],
                            "corrections": nlnh_result["corrections"]
                        }
                        
                        response_body = json.dumps(response_data).encode()
                    
                    # Create new response with validated body
                    return Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                
            except Exception as e:
                logger.error(f"Error validating response: {e}")
        
        return response
    
    def _extract_agent_id(self, path: str) -> Optional[str]:
        """Extract agent ID from path"""
        parts = path.split("/")
        if "agents" in parts:
            idx = parts.index("agents")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return None
    
    async def _emit_validation_event(self, event_type: str, data: Dict[str, Any]):
        """Emit validation event via Socket.IO"""
        if self.socketio:
            try:
                await self.socketio.emit(f"validation_{event_type}", data)
                logger.info(f"Emitted validation event: {event_type}")
            except Exception as e:
                logger.error(f"Failed to emit validation event: {e}")


def create_validation_middleware(app: ASGIApp) -> ValidationMiddleware:
    """Factory function to create validation middleware"""
    return ValidationMiddleware(app)