"""Server-side Supabase clients for Lyric Atlas.

Supabase is the query layer for structured metadata/stats.
R2 keys in Supabase point to heavy artifacts in Cloudflare R2.
"""

from __future__ import annotations

from supabase import Client, create_client

from src.atlas_env import get_supabase_private_key, get_supabase_url


def create_supabase_private_client() -> Client:
    """Server-side Supabase client using the private key."""
    return create_client(get_supabase_url(), get_supabase_private_key())
