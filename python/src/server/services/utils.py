"""
Utilities service module

Common utility functions for server services.
"""

from .client_manager import get_supabase_client
from ..config.config import get_config

__all__ = [
    "get_supabase_client",
    "get_config",
]