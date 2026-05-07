"""Official Genius API client helpers (metadata only)."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OFFICIAL_GENIUS_API_BASE = "https://api.genius.com"


class MissingGeniusTokenError(ValueError):
    """Raised when Genius token is missing."""


class GeniusCloudflareChallengeError(RuntimeError):
    """Raised when Genius responds with Cloudflare human verification."""


class GeniusRequestError(RuntimeError):
    """Raised when Genius repeatedly fails for retryable reasons."""


class ArtistNotFoundError(ValueError):
    """Raised when Genius returns no songs for the requested artist name."""

    def __init__(self, typed_name: str, suggestions: list[str]) -> None:
        self.typed_name = typed_name
        self.suggestions = suggestions
        super().__init__(typed_name)


def _extract_song_metadata(song: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized song metadata from official Genius API response."""
    primary_artist = song.get("primary_artist") or {}
    return {
        "title": song.get("title"),
        "artist": primary_artist.get("name"),
        "genius_song_id": song.get("id"),
        "url": song.get("url"),
        "album": (song.get("album") or {}).get("name"),
        "release_date": song.get("release_date_for_display") or song.get("release_date"),
        # Official API mode is metadata-only by default.
        "lyrics": "",
    }


def _build_official_session(token: str) -> requests.Session:
    """Construct a requests session for official Genius API calls."""
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        backoff_factor=1.0,
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _is_cloudflare_challenge(content: str) -> bool:
    text = content.lower()
    return "cloudflare" in text and ("challenge" in text or "human" in text or "verify" in text)


def _request_with_backoff(
    session: requests.Session,
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute official Genius API request with retry/backoff and robust errors."""
    url = f"{OFFICIAL_GENIUS_API_BASE}{path}"
    last_error: Exception | None = None

    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=20)
            content_type = response.headers.get("Content-Type", "")
            response_text = response.text or ""

            if response.status_code == 403 and (
                "text/html" in content_type or _is_cloudflare_challenge(response_text)
            ):
                raise GeniusCloudflareChallengeError(
                    "Genius blocked this web request with a Cloudflare challenge. "
                    "This usually affects unofficial Genius web endpoints, not necessarily your API key. "
                    "Try official metadata mode or CSV upload mode."
                )

            if response.status_code in {403, 429, 500, 502, 503, 504}:
                if attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise GeniusRequestError(
                    f"Genius API returned status {response.status_code} after retries for {path}"
                )

            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if isinstance(exc, GeniusCloudflareChallengeError):
                raise
            if attempt == 4:
                break
            time.sleep(2**attempt)

    raise GeniusRequestError(f"Genius API request failed after retries: {last_error}") from last_error


def suggest_similar_artists(query: str, max_suggestions: int = 5) -> list[str]:
    """Return distinct primary-artist names from Genius search hits (for typo help)."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        return []
    session = _build_official_session(token)
    try:
        payload = _request_with_backoff(session, "/search", params={"q": query})
    except (GeniusRequestError, GeniusCloudflareChallengeError, MissingGeniusTokenError):
        return []
    hits = payload.get("response", {}).get("hits", [])
    seen: set[str] = set()
    names: list[str] = []
    for hit in hits:
        result = hit.get("result") or {}
        artist = result.get("primary_artist") or {}
        name = str(artist.get("name", "")).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= max_suggestions:
            break
    return names


def search_artist_official(query: str) -> dict[str, Any] | None:
    """Search for the best matching artist using official Genius API."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise MissingGeniusTokenError("GENIUS_ACCESS_TOKEN is not set.")
    session = _build_official_session(token)
    payload = _request_with_backoff(session, "/search", params={"q": query})
    hits = payload.get("response", {}).get("hits", [])
    normalized_query = query.strip().lower()
    exact_match: dict[str, Any] | None = None
    partial_match: dict[str, Any] | None = None

    for hit in hits:
        result = hit.get("result") or {}
        artist = result.get("primary_artist") or {}
        artist_id = artist.get("id")
        artist_name = str(artist.get("name", "")).strip()
        if not artist_id:
            continue
        lowered = artist_name.lower()
        candidate = {"id": artist_id, "name": artist_name}
        if lowered == normalized_query:
            exact_match = candidate
            break
        if normalized_query in lowered and partial_match is None:
            partial_match = candidate

    return exact_match or partial_match or None


def get_song_metadata_official(song_id: int) -> dict[str, Any]:
    """Fetch song metadata by song id from official Genius API."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise MissingGeniusTokenError("GENIUS_ACCESS_TOKEN is not set.")
    session = _build_official_session(token)
    payload = _request_with_backoff(session, f"/songs/{song_id}")
    song = payload.get("response", {}).get("song") or {}
    return _extract_song_metadata(song)


def get_artist_songs_official(
    artist_id: int,
    max_songs: int,
    include_features: bool,
    canonical_artist_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch official Genius metadata for an artist's songs with optional feature filtering."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise MissingGeniusTokenError("GENIUS_ACCESS_TOKEN is not set.")
    session = _build_official_session(token)
    songs: list[dict[str, Any]] = []
    page = 1

    while len(songs) < max_songs:
        payload = _request_with_backoff(
            session,
            f"/artists/{artist_id}/songs",
            params={"per_page": 50, "page": page, "sort": "popularity"},
        )
        response = payload.get("response", {})
        raw_songs = response.get("songs", [])
        if not raw_songs:
            break

        for song in raw_songs:
            if not include_features:
                raw_primary = song.get("primary_artist") or {}
                raw_primary_id = raw_primary.get("id")
                if raw_primary_id is not None and int(raw_primary_id) != artist_id:
                    continue
            extracted = _extract_song_metadata(song)
            if not include_features:
                # Fallback check if primary artist id is unavailable.
                expected = str(canonical_artist_name or "").strip().lower()
                actual = str(extracted.get("artist", "")).strip().lower()
                if expected and actual != expected:
                    continue
            songs.append(extracted)
            if len(songs) >= max_songs:
                break

        next_page = response.get("next_page")
        if not next_page:
            break
        page = int(next_page)

    return songs[:max_songs]


def fetch_artist_songs(
    artist_name: str,
    max_songs: int,
    include_features: bool,
) -> dict[str, Any]:
    """Fetch artist songs from the official Genius API."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise MissingGeniusTokenError("GENIUS_ACCESS_TOKEN is not set.")

    artist = search_artist_official(artist_name)

    if artist is None:
        return {"artist_name": artist_name, "songs": [], "fetched_at": datetime.now(timezone.utc).isoformat()}

    songs = get_artist_songs_official(
        int(artist["id"]),
        max_songs=max_songs,
        include_features=include_features,
        canonical_artist_name=str(artist.get("name", "")),
    )

    return {
        "artist_name": artist.get("name"),
        "artist_id": artist.get("id"),
        "max_songs": max_songs,
        "include_features": include_features,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "songs": songs,
    }
