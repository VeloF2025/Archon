"""
Common utilities for the Archon application
"""

import os
from typing import Optional

from supabase import create_client

def get_supabase_client():
    """Get Supabase client instance."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables are required")

    return create_client(supabase_url, supabase_key)

__all__ = ["get_supabase_client"]