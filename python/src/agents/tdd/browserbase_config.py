#!/usr/bin/env python3
"""
Browserbase Configuration - Cloud Testing Infrastructure Setup

This module handles configuration and connection management for Browserbase
cloud testing infrastructure, providing scalable browser automation for
TDD enforcement and Stagehand test execution.
"""

import os
import logging
import aiohttp
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BrowserbaseConfig:
    """Browserbase API configuration"""
    api_key: str
    project_id: str
    base_url: str = "https://www.browserbase.com/v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_environment(cls) -> 'BrowserbaseConfig':
        """Create configuration from environment variables"""
        api_key = os.getenv("BROWSERBASE_API_KEY")
        project_id = os.getenv("BROWSERBASE_PROJECT_ID")
        
        if not api_key:
            raise ValueError("BROWSERBASE_API_KEY environment variable required")
        if not project_id:
            raise ValueError("BROWSERBASE_PROJECT_ID environment variable required")
        
        return cls(
            api_key=api_key,
            project_id=project_id,
            base_url=os.getenv("BROWSERBASE_BASE_URL", "https://www.browserbase.com/v1"),
            timeout_seconds=int(os.getenv("BROWSERBASE_TIMEOUT", "30")),
            max_retries=int(os.getenv("BROWSERBASE_MAX_RETRIES", "3"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API usage"""
        return {
            "api_key": self.api_key,
            "project_id": self.project_id,
            "base_url": self.base_url,
            "timeout": self.timeout_seconds,
            "max_retries": self.max_retries
        }

class BrowserbaseClient:
    """Client for Browserbase API operations"""
    
    def __init__(self, config: BrowserbaseConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def check_connection(self) -> bool:
        """Check if connection to Browserbase is working"""
        try:
            if not self.session:
                raise RuntimeError("Client session not initialized")
            
            # Test connection with project info endpoint
            url = f"{self.config.base_url}/projects/{self.config.project_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ Browserbase connection verified - Project: {data.get('name', 'Unknown')}")
                    return True
                elif response.status == 401:
                    logger.error("❌ Browserbase authentication failed - check API key")
                    return False
                elif response.status == 404:
                    logger.error("❌ Browserbase project not found - check project ID")
                    return False
                else:
                    logger.error(f"❌ Browserbase connection failed - Status: {response.status}")
                    return False
                    
        except aiohttp.ClientError as e:
            logger.error(f"❌ Browserbase network error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Browserbase connection check failed: {str(e)}")
            return False
    
    async def get_project_info(self) -> Dict[str, Any]:
        """Get project information from Browserbase"""
        if not self.session:
            raise RuntimeError("Client session not initialized")
        
        url = f"{self.config.base_url}/projects/{self.config.project_id}"
        
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def create_session(self, browser_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new browser session"""
        if not self.session:
            raise RuntimeError("Client session not initialized")
        
        default_settings = {
            "projectId": self.config.project_id,
            "browserSettings": {
                "viewport": {"width": 1920, "height": 1080},
                "headless": True,
                "browser": "chromium"
            }
        }
        
        if browser_settings:
            default_settings["browserSettings"].update(browser_settings)
        
        url = f"{self.config.base_url}/sessions"
        
        async with self.session.post(url, json=default_settings) as response:
            response.raise_for_status()
            return await response.json()

# Global configuration instance
_browserbase_config: Optional[BrowserbaseConfig] = None

def get_browserbase_config() -> BrowserbaseConfig:
    """Get or create global Browserbase configuration"""
    global _browserbase_config
    
    if _browserbase_config is None:
        _browserbase_config = BrowserbaseConfig.from_environment()
    
    return _browserbase_config

async def test_browserbase_connection() -> Dict[str, Any]:
    """Test connection to Browserbase API"""
    try:
        config = get_browserbase_config()
        
        async with BrowserbaseClient(config) as client:
            connected = await client.check_connection()
            
            result = {
                "connected": connected,
                "api_key_configured": bool(config.api_key),
                "project_id_configured": bool(config.project_id),
                "base_url": config.base_url,
                "status": "connected" if connected else "failed"
            }
            
            if connected:
                try:
                    project_info = await client.get_project_info()
                    result.update({
                        "project_name": project_info.get("name"),
                        "project_created": project_info.get("createdAt"),
                        "sessions_limit": project_info.get("sessionsLimit", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"Could not fetch project info: {str(e)}")
            
            return result
            
    except ValueError as e:
        # Configuration error
        return {
            "connected": False,
            "error": str(e),
            "status": "configuration_error",
            "api_key_configured": bool(os.getenv("BROWSERBASE_API_KEY")),
            "project_id_configured": bool(os.getenv("BROWSERBASE_PROJECT_ID"))
        }
    except Exception as e:
        logger.error(f"Browserbase connection test failed: {str(e)}")
        return {
            "connected": False,
            "error": str(e),
            "status": "error",
            "api_key_configured": bool(os.getenv("BROWSERBASE_API_KEY")),
            "project_id_configured": bool(os.getenv("BROWSERBASE_PROJECT_ID"))
        }

def setup_browserbase_environment():
    """Setup Browserbase environment variables with defaults"""
    
    # Set default values if not configured
    if not os.getenv("BROWSERBASE_API_KEY"):
        logger.warning("⚠️  BROWSERBASE_API_KEY not configured - TDD cloud testing disabled")
    
    if not os.getenv("BROWSERBASE_PROJECT_ID"):
        logger.warning("⚠️  BROWSERBASE_PROJECT_ID not configured - TDD cloud testing disabled")
    
    # Set default timeout if not specified
    if not os.getenv("BROWSERBASE_TIMEOUT"):
        os.environ["BROWSERBASE_TIMEOUT"] = "30"
    
    # Set default retry count if not specified  
    if not os.getenv("BROWSERBASE_MAX_RETRIES"):
        os.environ["BROWSERBASE_MAX_RETRIES"] = "3"
    
    logger.info("🔧 Browserbase environment configuration loaded")

# Initialize on import
setup_browserbase_environment()