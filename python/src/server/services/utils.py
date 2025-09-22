"""
Utilities service module

Common utility functions for server services.
"""

# Import from the common utils module
from ...utils import get_supabase_client
from ..config.config import get_config

__all__ = [
    "get_supabase_client",
    "get_config",
]