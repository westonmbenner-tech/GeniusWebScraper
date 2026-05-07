"""Server-side Cloudflare R2 client.

R2 is the artifact layer for large JSON/lyrics blobs.
Do not call this module from browser/client code.
"""

from __future__ import annotations

import boto3

from src.atlas_env import (
    get_r2_access_key_id,
    get_r2_endpoint,
    get_r2_secret_access_key,
)


def build_r2_client():
    """Build an S3-compatible client for Cloudflare R2."""
    return boto3.client(
        "s3",
        endpoint_url=get_r2_endpoint(),
        aws_access_key_id=get_r2_access_key_id(),
        aws_secret_access_key=get_r2_secret_access_key(),
        region_name="auto",
    )
