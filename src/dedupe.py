"""Song title normalization and deduplication logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


VERSION_PHRASES = [
    "taylor's version",
    "from the vault",
    "live",
    "remix",
    "demo",
    "acoustic",
    "sped up",
    "slowed",
    "instrumental",
    "karaoke",
    "translation",
    "romanized",
    "edit",
    "mix",
    "version",
]

HARD_EXCLUDE_TERMS = ["remix", "demo", "acoustic", "translation", "romanized"]
HARD_EXCLUDE_PATTERNS = [
    r"\blive\b",
    r"\(live[^)]*\)",
    r"\blive[\.\-:)]",
    r"\bremix\b",
    r"\bdemo\b",
    r"\bacoustic\b",
    r"\btranslation\b",
    r"\bromanized\b",
]


@dataclass
class DeduplicationResult:
    included: list[dict[str, Any]]
    excluded: list[dict[str, Any]]


def normalize_title(title: str) -> str:
    """Normalize title for dedupe matching."""
    cleaned = title.lower()
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", cleaned)
    for phrase in VERSION_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def should_exclude_version(title: str, canonical_artist_name: str | None = None) -> bool:
    """Check if title appears to be an excluded version."""
    lowered = title.lower()
    if any(re.search(pattern, lowered) for pattern in HARD_EXCLUDE_PATTERNS):
        return True
    if any(term in lowered for term in HARD_EXCLUDE_TERMS):
        return True
    if canonical_artist_name:
        artist_lower = canonical_artist_name.strip().lower()
        if artist_lower and artist_lower in lowered and re.search(r"\bon\b", lowered):
            if re.search(rf"\b{re.escape(artist_lower)}\b\s+on\b", lowered) or re.search(
                rf"\bon\b.*\b{re.escape(artist_lower)}\b",
                lowered,
            ):
                return True
            if any(term in lowered for term in ["podcast", "interview"]):
                return True
    return False


def dedupe_songs(
    songs: list[dict[str, Any]],
    mode: str,
    exclude_versions: bool,
    canonical_artist_name: str | None = None,
) -> DeduplicationResult:
    """Deduplicate songs according to selected mode."""
    if mode == "Keep everything":
        if not exclude_versions:
            return DeduplicationResult(included=songs, excluded=[])
        included = []
        excluded = []
        for song in songs:
            artist_for_filter = canonical_artist_name or str(song.get("artist", "") or "")
            if should_exclude_version(song.get("title", ""), canonical_artist_name=artist_for_filter):
                excluded.append({**song, "exclude_reason": "Excluded version term"})
            else:
                included.append(song)
        return DeduplicationResult(included=included, excluded=excluded)

    seen: dict[str, dict[str, Any]] = {}
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for song in songs:
        title = song.get("title", "")
        norm = normalize_title(title)
        artist_for_filter = canonical_artist_name or str(song.get("artist", "") or "")

        if exclude_versions and should_exclude_version(title, canonical_artist_name=artist_for_filter):
            excluded.append({**song, "exclude_reason": "Excluded version term"})
            continue

        if mode == "Keep major alternate versions":
            if norm in seen:
                excluded.append({**song, "exclude_reason": "Duplicate normalized title"})
            else:
                seen[norm] = song
                included.append(song)
            continue

        # Strict canonical songs only
        if norm in seen:
            excluded.append({**song, "exclude_reason": "Strict duplicate"})
        else:
            seen[norm] = song
            included.append(song)

    return DeduplicationResult(included=included, excluded=excluded)
