"""
Load credentials from database into environment variables for agents.
This ensures agents can access API keys stored via the UI.
"""

import asyncio
import os
import logging
import sys

# Add parent directory to path to import server modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.services.credential_service import credential_service

logger = logging.getLogger(__name__)


async def load_credentials_to_env():
    """
    Load credentials from database and set them as environment variables.
    This is called at startup to ensure agents have access to API keys.
    """
    try:
        # Initialize the credential service
        await credential_service.initialize()
        
        # Load API keys that agents need
        api_keys = [
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
        ]
        
        loaded_keys = []
        for key_name in api_keys:
            try:
                # Get the key from database
                value = await credential_service.get_credential(key_name, decrypt=True)
                if value:
                    # Set it as environment variable
                    os.environ[key_name] = str(value)
                    loaded_keys.append(key_name)
                    logger.info(f"Loaded {key_name} from database")
            except Exception as e:
                logger.debug(f"Could not load {key_name}: {e}")
        
        # Also load model configurations
        model_configs = [
            "MODEL_CHOICE",
            "DOCUMENT_AGENT_MODEL",
            "RAG_AGENT_MODEL",
            "TASK_AGENT_MODEL",
            "VALIDATOR_LLM_PROVIDER",
            "VALIDATOR_MODEL",
        ]
        
        for config_name in model_configs:
            try:
                value = await credential_service.get_credential(config_name, decrypt=True)
                if value:
                    os.environ[config_name] = str(value)
                    logger.info(f"Loaded {config_name} from database")
            except Exception as e:
                logger.debug(f"Could not load {config_name}: {e}")
        
        logger.info(f"Successfully loaded {len(loaded_keys)} API keys from database")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load credentials from database: {e}")
        logger.info("Falling back to environment variables from .env file")
        return False


def load_credentials_sync():
    """Synchronous wrapper for loading credentials."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(load_credentials_to_env())
    finally:
        loop.close()


if __name__ == "__main__":
    # Test the credential loading
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if load_credentials_sync():
        print("✅ Credentials loaded successfully")
        print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
        print(f"DEEPSEEK_API_KEY: {'Set' if os.getenv('DEEPSEEK_API_KEY') else 'Not set'}")
    else:
        print("❌ Failed to load credentials")