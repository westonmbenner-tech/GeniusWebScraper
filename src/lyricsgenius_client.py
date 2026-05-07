"""Optional lyricsgenius lyric fetching helpers with diagnostics."""

from __future__ import annotations

import os
import time
from typing import Any

import lyricsgenius
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

EXCLUDED_TERMS = [
    "(Remix)",
    "(Live)",
    "(Demo)",
    "(Acoustic)",
    "(Instrumental)",
    "(Karaoke)",
    "(Cover)",
    "(Translation)",
    "(Romanized)",
    "Tracklist",
    "Album Art",
    "Liner Notes",
    "Credits",
    "Interview",
    "Skit",
    "Setlist",
]


def _is_blocked_text(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "cloudflare",
        "human",
        "captcha",
        "403",
        "make sure you're a human",
        "make sure you are a human",
        "cf-mitigated",
        "human verification",
    ]
    return any(marker in lowered for marker in markers)


def _build_client(token: str) -> lyricsgenius.Genius:
    genius = lyricsgenius.Genius(
        token,
        timeout=20,
        sleep_time=2.0,
        retries=3,
        user_agent=USER_AGENT,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=EXCLUDED_TERMS,
    )
    session = getattr(genius, "session", None) or getattr(genius, "_session", None)
    if session is not None:
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
            }
        )
    return genius


def fetch_lyrics_with_diagnostics(title: str, artist: str) -> dict[str, Any]:
    """Fetch lyrics and return verbose diagnostics payload."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        return {
            "called": True,
            "input_title": title,
            "input_artist": artist,
            "ok": False,
            "lyrics": "",
            "found_song": False,
            "returned_title": None,
            "returned_artist": None,
            "returned_url": None,
            "lyrics_type": "none",
            "lyrics_char_count": 0,
            "error_type": "MissingTokenError",
            "error_message": "Missing GENIUS_ACCESS_TOKEN",
            "blocked": False,
        }

    genius = _build_client(token)
    print(f"Calling lyricsgenius.search_song(title={title!r}, artist={artist!r})")
    time.sleep(0.5)
    try:
        song = genius.search_song(title=title, artist=artist)
        if song is None:
            return {
                "called": True,
                "input_title": title,
                "input_artist": artist,
                "ok": False,
                "lyrics": "",
                "found_song": False,
                "returned_title": None,
                "returned_artist": None,
                "returned_url": None,
                "lyrics_type": "none",
                "lyrics_char_count": 0,
                "error_type": None,
                "error_message": None,
                "blocked": False,
            }

        lyrics = getattr(song, "lyrics", None)
        if lyrics is None:
            lyrics_type = "none"
            lyrics_char_count = 0
        elif lyrics == "":
            lyrics_type = "empty_string"
            lyrics_char_count = 0
        elif isinstance(lyrics, str) and len(lyrics.strip()) > 0:
            lyrics_type = "string"
            lyrics_char_count = len(lyrics)
        elif isinstance(lyrics, str):
            lyrics_type = "empty_string"
            lyrics_char_count = len(lyrics)
        else:
            lyrics_type = "other"
            lyrics_char_count = 0
        blocked = _is_blocked_text(str(lyrics if isinstance(lyrics, str) else ""))
        return {
            "called": True,
            "input_title": title,
            "input_artist": artist,
            "ok": lyrics_char_count > 0,
            "lyrics": lyrics if isinstance(lyrics, str) else "",
            "found_song": True,
            "returned_title": getattr(song, "title", None),
            "returned_artist": getattr(song, "artist", None),
            "returned_url": getattr(song, "url", None),
            "lyrics_type": lyrics_type,
            "lyrics_char_count": lyrics_char_count,
            "error_type": None,
            "error_message": None,
            "blocked": blocked,
        }
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        blocked = _is_blocked_text(error_message)
        return {
            "called": True,
            "input_title": title,
            "input_artist": artist,
            "ok": False,
            "lyrics": "",
            "found_song": False,
            "returned_title": None,
            "returned_artist": None,
            "returned_url": None,
            "lyrics_type": "none",
            "lyrics_char_count": 0,
            "error_type": exc.__class__.__name__,
            "error_message": error_message,
            "blocked": blocked,
        }


def fetch_lyrics_for_song(title: str, artist: str) -> dict[str, Any]:
    """Fetch lyrics for one song in a safe, failure-tolerant format."""
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        return {
            "ok": False,
            "lyrics": "",
            "source": "lyricsgenius",
            "error": "Missing GENIUS_ACCESS_TOKEN",
            "blocked": False,
        }
    genius = _build_client(token)
    time.sleep(0.5)
    try:
        song = genius.search_song(title=title, artist=artist)
        if song is None:
            return {"ok": False, "lyrics": "", "source": "lyricsgenius", "error": "not_found", "blocked": False}
        lyrics = getattr(song, "lyrics", None)
        if not isinstance(lyrics, str) or not lyrics.strip():
            return {"ok": False, "lyrics": "", "source": "lyricsgenius", "error": "missing", "blocked": False}
        blocked = _is_blocked_text(lyrics)
        return {
            "ok": not blocked,
            "lyrics": "" if blocked else lyrics,
            "source": "lyricsgenius",
            "error": "blocked" if blocked else None,
            "blocked": blocked,
        }
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        blocked = _is_blocked_text(msg)
        return {
            "ok": False,
            "lyrics": "",
            "source": "lyricsgenius",
            "error": msg,
            "blocked": blocked,
        }
