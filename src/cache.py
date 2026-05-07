"""Small helpers shared across Lyric Atlas modules."""

from __future__ import annotations

import re


def slugify_artist_name(artist_name: str) -> str:
    """Create a stable slug from an artist name (e.g. for R2 keys)."""
    slug = artist_name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "unknown-artist"
