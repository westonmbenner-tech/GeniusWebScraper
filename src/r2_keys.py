"""Deterministic R2 key builders for Lyric Atlas."""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_slug(value: str, field_name: str) -> str:
    v = value.strip()
    if not _SLUG_RE.match(v):
        raise ValueError(f"{field_name} must be lowercase URL-safe slug (letters/numbers/hyphens): {value!r}")
    return v


def _validate_safe(value: str, field_name: str) -> str:
    v = value.strip()
    if not _SAFE_RE.match(v):
        raise ValueError(f"{field_name} contains unsafe characters: {value!r}")
    return v


def _join_key(*parts: str) -> str:
    key = "/".join(part.strip("/") for part in parts)
    if key.startswith("/"):
        raise ValueError("R2 key must not start with /")
    if " " in key:
        raise ValueError("R2 key must not contain spaces")
    return key


def artist_raw_import_key(artist_slug: str, import_id: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "raw", "imports", _validate_safe(import_id, "import_id"), "corpus.json")


def artist_song_lyrics_key(artist_slug: str, song_slug: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "raw", "songs", f"{_validate_slug(song_slug, 'song_slug')}.json")


def artist_run_full_analysis_key(artist_slug: str, run_id: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "runs", _validate_safe(run_id, "run_id"), "full-analysis.json")


def artist_run_per_song_analysis_key(artist_slug: str, run_id: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "runs", _validate_safe(run_id, "run_id"), "per-song-analysis.json")


def artist_run_debug_key(artist_slug: str, run_id: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "runs", _validate_safe(run_id, "run_id"), "debug.json")


def artist_export_report_key(artist_slug: str, run_id: str) -> str:
    return _join_key("artists", _validate_slug(artist_slug, "artist_slug"), "exports", _validate_safe(run_id, "run_id"), "report.json")
