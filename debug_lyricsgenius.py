"""Standalone lyricsgenius debug script."""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import lyricsgenius
from dotenv import load_dotenv

ENV_LOCAL_PATH = Path(__file__).resolve().parent / ".env.local"
load_dotenv(dotenv_path=ENV_LOCAL_PATH, override=True)

token = os.getenv("GENIUS_ACCESS_TOKEN")

print("lyricsgenius module:", lyricsgenius)
print("lyricsgenius version:", getattr(lyricsgenius, "__version__", "unknown"))
print("Token loaded:", "yes" if token else "no", (token[:6] + "...") if token else "")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

try:
    genius = lyricsgenius.Genius(
        token,
        timeout=20,
        sleep_time=2.0,
        retries=3,
        user_agent=USER_AGENT,
        remove_section_headers=True,
        skip_non_songs=True,
        excluded_terms=[
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
        ],
    )

    title = "Stick Season"
    artist = "Noah Kahan"

    print(f"Calling genius.search_song(title={title!r}, artist={artist!r})")
    song = genius.search_song(title=title, artist=artist)

    print("Returned object type:", type(song))

    if song is None:
        print("No song returned.")
    else:
        lyrics = getattr(song, "lyrics", None)
        print("Returned title:", getattr(song, "title", None))
        print("Returned artist:", getattr(song, "artist", None))
        print("Returned URL:", getattr(song, "url", None))
        print("Lyrics is None?:", lyrics is None)
        print("Lyrics char count:", len(lyrics) if isinstance(lyrics, str) else 0)
        print("Lyrics preview (first 500 chars):")
        print((lyrics or "")[:500])

except Exception:
    print("Exception occurred:")
    traceback.print_exc()
