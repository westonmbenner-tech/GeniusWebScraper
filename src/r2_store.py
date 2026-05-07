"""Cloudflare R2 storage helpers for cache-first Lyric Atlas workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.r2_json import get_json_from_r2, object_exists, put_json_to_r2
from src.r2_keys import artist_raw_import_key


def _safe_artist_token(artist_name: str) -> str:
    return "-".join("".join(ch for ch in part.lower() if ch.isalnum()) for part in artist_name.split() if part).strip("-")


def build_artist_corpus_key(artist_metadata: dict[str, Any]) -> str:
    """Preferred key by Genius artist id, fallback by normalized artist name."""
    genius_artist_id = artist_metadata.get("artist_id") or artist_metadata.get("genius_artist_id")
    if genius_artist_id:
        return f"artists/{genius_artist_id}/corpus_v1.json"
    fallback = _safe_artist_token(str(artist_metadata.get("artist_name", ""))) or "unknown-artist"
    return f"artists/{fallback}/corpus_v1.json"


def get_r2_corpus_key(artist_metadata: dict[str, Any]) -> str:
    return build_artist_corpus_key(artist_metadata)


def r2_object_exists(key: str) -> bool:
    return object_exists(key)


def download_json_from_r2(key: str) -> dict[str, Any] | None:
    try:
        value = get_json_from_r2(key)
    except FileNotFoundError:
        return None
    if isinstance(value, dict):
        return value
    raise ValueError(f"R2 object exists but is not a JSON object: {key}")


def upload_json_to_r2(key: str, data: dict[str, Any]) -> str:
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    return put_json_to_r2(key, data)


def get_corpus_summary(corpus: dict[str, Any]) -> dict[str, Any]:
    songs = list(corpus.get("songs", [])) if isinstance(corpus.get("songs", []), list) else []
    with_lyrics = [
        s
        for s in songs
        if len(str(s.get("lyrics", "") or "").strip()) > 0 or int(s.get("lyrics_char_count", 0) or 0) > 0
    ]
    return {
        "song_count": len(songs),
        "songs_with_lyrics": len(with_lyrics),
        "updated_at": corpus.get("updated_at"),
        "version": corpus.get("version", "v1"),
    }


def make_empty_corpus(artist_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "v1",
        "artist_metadata": artist_metadata,
        "songs": [],
        "source": "r2_cache",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
