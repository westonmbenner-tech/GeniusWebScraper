"""Python backend scaffolding for Lyric Atlas v2.

Architecture:
- Supabase: query layer (metadata, counts, ranks, pointers)
- R2: artifact layer (raw lyrics/full analysis JSON)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.r2_json import get_json_from_r2, put_json_to_r2
from src.r2_keys import (
    artist_raw_import_key,
    artist_run_full_analysis_key,
    artist_run_per_song_analysis_key,
    artist_song_lyrics_key,
)
from src.supabase_client import create_supabase_private_client


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]


def import_artist_corpus(
    *,
    artist_name: str,
    artist_slug: str,
    songs: list[dict[str, Any]],
    import_id: str | None = None,
) -> dict[str, Any]:
    """Scaffold equivalent of POST /api/artists/import."""
    supabase = create_supabase_private_client()
    import_id = import_id or _utc_run_id()
    corpus_key = artist_raw_import_key(artist_slug, import_id)
    put_json_to_r2(corpus_key, {"artist_name": artist_name, "artist_slug": artist_slug, "songs": songs})

    artist_resp = (
        supabase.table("artists")
        .upsert({"name": artist_name, "slug": artist_slug}, on_conflict="slug")
        .execute()
    )
    if not artist_resp.data:
        raise RuntimeError("Failed to upsert artist into Supabase.")
    artist = artist_resp.data[0]
    artist_id = artist["id"]

    upsert_rows: list[dict[str, Any]] = []
    for song in songs:
        song_slug = str(song["slug"])
        lyrics_key = artist_song_lyrics_key(artist_slug, song_slug)
        put_json_to_r2(
            lyrics_key,
            {
                "artist_name": artist_name,
                "artist_slug": artist_slug,
                "title": song.get("title"),
                "slug": song_slug,
                "lyrics_text": song.get("lyrics_text", ""),
            },
        )
        upsert_rows.append(
            {
                "artist_id": artist_id,
                "title": song.get("title"),
                "slug": song_slug,
                "album": song.get("album"),
                "release_year": song.get("release_year"),
                "source_url": song.get("source_url"),
                "r2_lyrics_key": lyrics_key,
            }
        )
    supabase.table("songs").upsert(upsert_rows, on_conflict="artist_id,slug").execute()
    return {"artist": artist, "song_count": len(songs), "r2_raw_corpus_key": corpus_key}


def save_analysis_run(
    *,
    artist_slug: str,
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    """Scaffold equivalent of POST /api/artists/[artistSlug]/analysis/save."""
    supabase = create_supabase_private_client()
    artist_resp = supabase.table("artists").select("*").eq("slug", artist_slug).limit(1).execute()
    if not artist_resp.data:
        raise LookupError(f"Artist not found: {artist_slug}")
    artist = artist_resp.data[0]
    artist_id = artist["id"]
    run_id = _utc_run_id()

    full_key = artist_run_full_analysis_key(artist_slug, run_id)
    put_json_to_r2(full_key, analysis_payload)
    per_song_payload = analysis_payload.get("per_song_analysis")
    per_song_key = None
    if per_song_payload is not None:
        per_song_key = artist_run_per_song_analysis_key(artist_slug, run_id)
        put_json_to_r2(per_song_key, per_song_payload)

    run_resp = (
        supabase.table("analysis_runs")
        .insert(
            {
                "artist_id": artist_id,
                "run_id": run_id,
                "status": "completed",
                "song_count": analysis_payload.get("song_count"),
                "total_words": analysis_payload.get("total_words"),
                "r2_full_analysis_key": full_key,
                "r2_per_song_analysis_key": per_song_key,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .execute()
    )
    if not run_resp.data:
        raise RuntimeError("Failed to insert analysis run.")
    return {"run": run_resp.data[0], "r2_full_analysis_key": full_key, "r2_per_song_analysis_key": per_song_key}


def get_artist_profile_bundle(artist_slug: str) -> dict[str, Any]:
    """Scaffold equivalent of GET /api/artists/[artistSlug]/profile."""
    supabase = create_supabase_private_client()
    artist_resp = supabase.table("artists").select("*").eq("slug", artist_slug).limit(1).execute()
    if not artist_resp.data:
        raise LookupError(f"Artist not found: {artist_slug}")
    artist = artist_resp.data[0]
    run_resp = (
        supabase.table("analysis_runs")
        .select("*")
        .eq("artist_id", artist["id"])
        .eq("status", "completed")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    latest_run = run_resp.data[0] if run_resp.data else None
    if not latest_run:
        return {"artist": artist, "latest_run": None}

    run_id = latest_run["id"]
    profile = (
        supabase.table("artist_profiles").select("*").eq("analysis_run_id", run_id).limit(1).execute().data or [None]
    )[0]
    words = (
        supabase.table("word_stats").select("*").eq("analysis_run_id", run_id).order("rank").limit(50).execute().data
        or []
    )
    phrases = (
        supabase.table("phrase_stats").select("*").eq("analysis_run_id", run_id).order("rank").limit(50).execute().data
        or []
    )
    themes = (
        supabase.table("theme_stats").select("*").eq("analysis_run_id", run_id).order("rank").limit(50).execute().data
        or []
    )
    songs = supabase.table("songs").select("*").eq("artist_id", artist["id"]).order("title").execute().data or []
    summaries = (
        supabase.table("song_analysis_summaries").select("*").eq("analysis_run_id", run_id).execute().data or []
    )
    return {
        "artist": artist,
        "latest_run": latest_run,
        "artist_profile": profile,
        "top_words": words,
        "top_phrases": phrases,
        "themes": themes,
        "songs": songs,
        "song_summaries": summaries,
    }


def get_song_lyrics(song_id: str) -> dict[str, Any]:
    """Scaffold equivalent of GET /api/songs/[songId]/lyrics."""
    supabase = create_supabase_private_client()
    song_resp = supabase.table("songs").select("*").eq("id", song_id).limit(1).execute()
    if not song_resp.data:
        raise LookupError(f"Song not found: {song_id}")
    song = song_resp.data[0]
    return {"song": song, "lyrics_blob": get_json_from_r2(song["r2_lyrics_key"])}


def get_full_analysis_by_run_id(run_id: str) -> dict[str, Any]:
    """Scaffold equivalent of GET /api/analysis-runs/[runId]/full."""
    supabase = create_supabase_private_client()
    run_resp = supabase.table("analysis_runs").select("*").eq("run_id", run_id).limit(1).execute()
    if not run_resp.data:
        raise LookupError(f"Analysis run not found: {run_id}")
    run = run_resp.data[0]
    full_key = run.get("r2_full_analysis_key")
    if not full_key:
        raise FileNotFoundError("Analysis run has no r2_full_analysis_key")
    return {"analysis_run": run, "full_analysis": get_json_from_r2(full_key)}
