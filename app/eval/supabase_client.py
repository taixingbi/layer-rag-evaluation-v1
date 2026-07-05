"""Supabase Python client (v2) for eval persistence."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_supabase_service_key, get_supabase_url
from supabase import Client, create_client


@lru_cache(maxsize=8)
def get_supabase_admin_client(env: str = "dev") -> Client:
    """Return service-role Supabase client for eval env label (``dev``, ``prod``, …)."""
    url = get_supabase_url(required=True, env=env)
    key = get_supabase_service_key(required=True, env=env)
    return create_client(url, key)
