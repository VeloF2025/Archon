"""
Database integration for External Validator
Fetches API keys marked with useAsValidator from Archon database
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class DatabaseIntegration:
    """Integrates with Archon database to fetch validator API keys"""
    
    def __init__(self):
        # Use Docker service name when running in container
        if os.path.exists("/.dockerenv"):
            self.archon_server_url = "http://archon-server:8181"
        else:
            self.archon_server_url = os.getenv("ARCHON_SERVER_URL", "http://localhost:8181")
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def get_validator_api_key(self) -> Optional[Dict[str, Any]]:
        """
        Fetch API key marked for validator use from database
        
        Returns:
            Dict with provider, api_key, and model if found
            None if no validator API key configured
        """
        try:
            # Call Archon server to get credentials
            response = await self.client.get(
                f"{self.archon_server_url}/api/credentials",
                headers={"Accept": "application/json"}
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch credentials: {response.status_code}")
                return None
            
            credentials = response.json()
            
            # Find credential marked with useAsValidator
            for cred in credentials:
                metadata = cred.get("metadata", {})
                if metadata.get("useAsValidator") == True:
                    key_name = cred.get("key", "")
                    value = cred.get("value", "")
                    
                    if not value:
                        continue
                    
                    # Determine provider based on key name or content
                    provider_config = self._detect_provider(key_name, value)
                    if provider_config:
                        return provider_config
            
            logger.info("No API key marked for validator use in database")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching validator API key from database: {e}")
            return None
    
    def _detect_provider(self, key_name: str, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Detect provider based on key name or API key format
        
        Supports:
        - DeepSeek (sk-... format)
        - OpenAI (sk-proj-... format)
        - Anthropic (sk-ant-... format)
        - Groq (gsk_... format)
        - Google (AIza... format)
        - Mistral (... format)
        """
        key_upper = key_name.upper()
        
        # Check key name first (highest priority)
        if "DEEPSEEK" in key_upper:
            return {
                "provider": "deepseek",
                "api_key": api_key,
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com"
            }
        elif "OPENAI" in key_upper or "GPT" in key_upper:
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": "gpt-4o",
                "base_url": None
            }
        elif "ANTHROPIC" in key_upper or "CLAUDE" in key_upper:
            return {
                "provider": "anthropic",
                "api_key": api_key,
                "model": "claude-3-5-sonnet-20241022",
                "base_url": None
            }
        elif "GROQ" in key_upper:
            return {
                "provider": "groq",
                "api_key": api_key,
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1"
            }
        elif "GOOGLE" in key_upper or "GEMINI" in key_upper:
            return {
                "provider": "google",
                "api_key": api_key,
                "model": "gemini-1.5-pro",
                "base_url": None
            }
        elif "MISTRAL" in key_upper:
            return {
                "provider": "mistral",
                "api_key": api_key,
                "model": "mistral-large-latest",
                "base_url": "https://api.mistral.ai"
            }
        
        # Try to infer from API key format
        if api_key.startswith("sk-proj-"):
            return {
                "provider": "openai",
                "api_key": api_key,
                "model": "gpt-4o",
                "base_url": None
            }
        elif api_key.startswith("sk-ant-"):
            return {
                "provider": "anthropic",
                "api_key": api_key,
                "model": "claude-3-5-sonnet-20241022",
                "base_url": None
            }
        elif api_key.startswith("gsk_"):
            return {
                "provider": "groq",
                "api_key": api_key,
                "model": "llama-3.3-70b-versatile",
                "base_url": "https://api.groq.com/openai/v1"
            }
        elif api_key.startswith("AIza"):
            return {
                "provider": "google",
                "api_key": api_key,
                "model": "gemini-1.5-pro",
                "base_url": None
            }
        else:
            # Default to DeepSeek for unknown keys
            logger.info("Unknown API key format, defaulting to DeepSeek")
            return {
                "provider": "deepseek",
                "api_key": api_key,
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com"
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()