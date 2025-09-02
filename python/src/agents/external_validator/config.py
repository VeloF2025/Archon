"""
Configuration management for External Validator
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, SecretStr
from pathlib import Path
from dotenv import load_dotenv
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class LLMConfig(BaseModel):
    """LLM configuration for validation"""
    
    provider: Literal["deepseek", "openai", "anthropic", "groq", "google", "mistral"] = Field(
        default="deepseek",
        description="LLM provider to use"
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the LLM provider"
    )
    model: str = Field(
        default="deepseek-chat",
        description="Model to use for validation"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=0.2,
        description="Temperature for deterministic responses"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for response"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for API (for DeepSeek compatibility)"
    )


class ValidationConfig(BaseModel):
    """Validation engine configuration"""
    
    enable_deterministic: bool = Field(
        default=True,
        description="Enable deterministic checks (pytest, ruff, mypy)"
    )
    enable_cross_check: bool = Field(
        default=True,
        description="Enable cross-checking with context"
    )
    confidence_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="DeepConf confidence threshold"
    )
    max_context_tokens: int = Field(
        default=5000,
        description="Maximum tokens for PRP context"
    )
    enable_proactive_triggers: bool = Field(
        default=True,
        description="Enable automatic validation triggers"
    )


class ValidatorConfig:
    """Main configuration manager for External Validator"""
    
    _instance = None
    _config_file = Path(__file__).parent / "validator_config.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.llm_config = LLMConfig()
        self.validation_config = ValidationConfig()
        self._db_config_loaded = False  # Initialize before loading config
        self._load_config()
        self._initialized = True
    
    def _load_config(self):
        """Load configuration from file and environment"""
        
        # Load from config file if exists
        if self._config_file.exists():
            try:
                with open(self._config_file, 'r') as f:
                    config_data = json.load(f)
                    
                if "llm" in config_data:
                    self.llm_config = LLMConfig(**config_data["llm"])
                if "validation" in config_data:
                    self.validation_config = ValidationConfig(**config_data["validation"])
                    
                logger.info(f"Loaded configuration from {self._config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        # Override with environment variables
        self._load_from_env()
        
        # Database config will be loaded async later in main.py
    
    def _load_from_env(self):
        """Load configuration from environment variables - ONLY AS FALLBACK"""
        
        # ONLY load from env if database config is not loaded
        # This ensures database config takes priority
        
        # LLM configuration - SKIP if database loaded
        if not self._db_config_loaded:
            if os.getenv("VALIDATOR_LLM_PROVIDER"):
                self.llm_config.provider = os.getenv("VALIDATOR_LLM_PROVIDER")
                logger.warning("Using VALIDATOR_LLM_PROVIDER from env - should use database instead")
            
            if os.getenv("DEEPSEEK_API_KEY"):
                self.llm_config.api_key = SecretStr(os.getenv("DEEPSEEK_API_KEY"))
                self.llm_config.base_url = "https://api.deepseek.com"
                logger.warning("Using DEEPSEEK_API_KEY from env - should use database instead")
            elif os.getenv("OPENAI_API_KEY"):
                self.llm_config.api_key = SecretStr(os.getenv("OPENAI_API_KEY"))
                self.llm_config.provider = "openai"
                logger.warning("Using OPENAI_API_KEY from env - should use database instead")
            
            if os.getenv("VALIDATOR_MODEL"):
                self.llm_config.model = os.getenv("VALIDATOR_MODEL")
                logger.warning("Using VALIDATOR_MODEL from env - should use database instead")
            
            if os.getenv("VALIDATOR_TEMPERATURE"):
                self.llm_config.temperature = float(os.getenv("VALIDATOR_TEMPERATURE"))
                logger.warning("Using VALIDATOR_TEMPERATURE from env - should use database instead")
        
        # Validation configuration
        if os.getenv("VALIDATOR_CONFIDENCE_THRESHOLD"):
            self.validation_config.confidence_threshold = float(
                os.getenv("VALIDATOR_CONFIDENCE_THRESHOLD")
            )
    
    def save_config(self):
        """Save current configuration to file"""
        
        config_data = {
            "llm": self.llm_config.model_dump(exclude={"api_key"}),
            "validation": self.validation_config.model_dump()
        }
        
        with open(self._config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration to {self._config_file}")
    
    def update_llm_config(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """Update LLM configuration"""
        
        if provider:
            self.llm_config.provider = provider
        if api_key:
            self.llm_config.api_key = SecretStr(api_key)
        if model:
            self.llm_config.model = model
        if temperature is not None:
            self.llm_config.temperature = temperature
        
        self.save_config()
        logger.info("Updated LLM configuration")
    
    async def load_from_database(self):
        """Load API key configuration from database - HIGHEST PRIORITY"""
        
        if self._db_config_loaded:
            return
        
        try:
            from .database_integration import DatabaseIntegration
            
            db = DatabaseIntegration()
            db_config = await db.get_validator_api_key()
            await db.cleanup()
            
            if db_config:
                logger.info(f"Loading validator API key from database: {db_config['provider']}")
                
                # Update LLM config with database values - OVERRIDES ENV VARS
                self.llm_config.provider = db_config["provider"]
                self.llm_config.api_key = SecretStr(db_config["api_key"])
                self.llm_config.model = db_config["model"]
                
                if db_config.get("base_url"):
                    self.llm_config.base_url = db_config["base_url"]
                
                self._db_config_loaded = True
                logger.info(f"Successfully loaded {db_config['provider']} API key from database (overrides env vars)")
            else:
                logger.info("No validator API key found in database, using environment/config file")
                
        except Exception as e:
            logger.warning(f"Failed to load API key from database: {e}")
    
    def get_llm_client_config(self) -> dict:
        """Get configuration for LLM client"""
        
        config = {
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens
        }
        
        if self.llm_config.api_key:
            config["api_key"] = self.llm_config.api_key.get_secret_value()
        
        if self.llm_config.base_url:
            config["base_url"] = self.llm_config.base_url
        
        return config