"""
MCP (Model Context Protocol) integration for External Validator
"""

import json
import logging
from typing import Dict, Any, Optional
import httpx

from .validation_engine import ValidationEngine
from .models import ValidationRequest, ConfigureValidatorRequest

logger = logging.getLogger(__name__)


class MCPIntegration:
    """Handles MCP tool registration and communication"""
    
    def __init__(self, validation_engine: ValidationEngine):
        self.validation_engine = validation_engine
        self.mcp_server_url = "http://localhost:8051"  # Archon MCP server
        self.tools_registered = False
    
    async def register_tools(self):
        """Register validator tools with MCP server"""
        
        tools = [
            {
                "name": "validate",
                "description": "Validate content using External Validator",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output": {
                            "type": "string",
                            "description": "Content to validate"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Original prompt (optional)"
                        },
                        "context": {
                            "type": "object",
                            "description": "Validation context (optional)"
                        },
                        "validation_type": {
                            "type": "string",
                            "enum": ["code", "documentation", "prompt", "output", "full"],
                            "description": "Type of validation"
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 0.2,
                            "description": "Temperature override (optional)"
                        }
                    },
                    "required": ["output"]
                }
            },
            {
                "name": "configure_validator",
                "description": "Configure External Validator settings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "provider": {
                            "type": "string",
                            "enum": ["deepseek", "openai"],
                            "description": "LLM provider"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use"
                        },
                        "temperature": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 0.2,
                            "description": "Temperature setting"
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence threshold"
                        }
                    }
                }
            },
            {
                "name": "validator_health",
                "description": "Check External Validator health status",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
        
        try:
            # Register each tool with MCP server
            async with httpx.AsyncClient() as client:
                for tool in tools:
                    response = await client.post(
                        f"{self.mcp_server_url}/register_tool",
                        json={
                            "tool": tool,
                            "handler_url": "http://localhost:8053/mcp/handle"
                        }
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Registered MCP tool: {tool['name']}")
                    else:
                        logger.warning(f"Failed to register tool {tool['name']}: {response.text}")
                
                self.tools_registered = True
                logger.info("All validator MCP tools registered")
                
        except Exception as e:
            logger.error(f"Failed to register MCP tools: {e}")
            self.tools_registered = False
    
    async def handle_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        
        try:
            if tool_name == "validate":
                return await self._handle_validate(parameters)
            
            elif tool_name == "configure_validator":
                return await self._handle_configure(parameters)
            
            elif tool_name == "validator_health":
                return await self._handle_health()
            
            else:
                return {
                    "error": f"Unknown tool: {tool_name}"
                }
                
        except Exception as e:
            logger.error(f"Tool call error: {e}", exc_info=True)
            return {
                "error": str(e)
            }
    
    async def _handle_validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validate tool call"""
        
        # Create validation request
        request = ValidationRequest(
            output=parameters.get("output", ""),
            prompt=parameters.get("prompt"),
            context=parameters.get("context"),
            validation_type=parameters.get("validation_type", "full"),
            temperature_override=parameters.get("temperature")
        )
        
        # Perform validation
        response = await self.validation_engine.validate(request)
        
        # Convert to dict for MCP response
        return response.model_dump()
    
    async def _handle_configure(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configure_validator tool call"""
        
        # Create configuration request
        request = ConfigureValidatorRequest(**parameters)
        
        # Update configuration
        config = self.validation_engine.config
        
        if request.provider:
            config.llm_config.provider = request.provider
        if request.model:
            config.llm_config.model = request.model
        if request.temperature is not None:
            config.llm_config.temperature = request.temperature
        if request.confidence_threshold is not None:
            config.validation_config.confidence_threshold = request.confidence_threshold
        
        # Save configuration
        config.save_config()
        
        # Reinitialize validation engine
        await self.validation_engine.initialize()
        
        return {
            "status": "success",
            "message": "Validator configured successfully",
            "config": {
                "provider": config.llm_config.provider,
                "model": config.llm_config.model,
                "temperature": config.llm_config.temperature,
                "confidence_threshold": config.validation_config.confidence_threshold
            }
        }
    
    async def _handle_health(self) -> Dict[str, Any]:
        """Handle validator_health tool call"""
        
        llm_connected = await self.validation_engine.check_llm_connection()
        deterministic_available = self.validation_engine.check_deterministic_tools()
        
        return {
            "status": "healthy" if llm_connected and deterministic_available else "degraded",
            "llm_connected": llm_connected,
            "deterministic_available": deterministic_available,
            "tools_registered": self.tools_registered,
            "config": {
                "provider": self.validation_engine.config.llm_config.provider,
                "model": self.validation_engine.config.llm_config.model,
                "temperature": self.validation_engine.config.llm_config.temperature
            }
        }
    
    async def notify_archon(self, event_type: str, data: Dict[str, Any]):
        """Send notifications to Archon system"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.mcp_server_url}/notify",
                    json={
                        "event_type": f"validator_{event_type}",
                        "data": data
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to notify Archon: {response.text}")
                    
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")