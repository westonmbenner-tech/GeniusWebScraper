"""Environment helpers for Lyric Atlas storage architecture.

Supabase is the query layer, R2 is the artifact layer.
Keep R2 credentials and Supabase service role server-side only.
"""

from __future__ import annotations

import os


def _req(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_r2_account_id() -> str:
    return _req("R2_ACCOUNT_ID")


def get_r2_access_key_id() -> str:
    return _req("R2_ACCESS_KEY_ID")


def get_r2_secret_access_key() -> str:
    return _req("R2_SECRET_ACCESS_KEY")


def get_r2_bucket_name() -> str:
    return _req("R2_BUCKET_NAME")


def get_r2_endpoint() -> str:
    endpoint = _req("R2_ENDPOINT")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    raise RuntimeError("R2_ENDPOINT must include http:// or https://")


def get_supabase_url() -> str:
    return _req("NEXT_PUBLIC_SUPABASE_URL")


def get_supabase_private_key() -> str:
    return _req("SUPABASE_PRIVATE_KEY")
