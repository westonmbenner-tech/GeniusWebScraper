"""Supabase archive metadata + analysis persistence helpers."""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client


def get_supabase_client() -> Client:
    # Match .env.example names exactly:
    # - NEXT_PUBLIC_SUPABASE_URL
    # - SUPABASE_PRIVATE_KEY
    url = (os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_PRIVATE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError(
            "Supabase env not configured. Set NEXT_PUBLIC_SUPABASE_URL and SUPABASE_PRIVATE_KEY "
            "(for example via environment variables or a .env file loaded before startup)."
        )
    return create_client(url, key)


def _row_settings_payload(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("settings_json") or {})


def _archive_compact_identity_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identity for de-dupe: filter + dedupe + features + source (not requested_song_count)."""
    sj = _row_settings_payload(row)
    return (
        str(sj.get("filtering_strictness") or row.get("filtering_strictness") or ""),
        str(sj.get("dedupe_mode") or row.get("dedupe_mode") or ""),
        bool(sj.get("include_features")),
        str(sj.get("source_used") or ""),
    )


def _archive_compact_identity_from_settings(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(settings.get("filtering_strictness") or ""),
        str(settings.get("dedupe_mode") or ""),
        bool(settings.get("include_features")),
        str(settings.get("source_used") or ""),
    )


def settings_match(row: dict[str, Any], settings: dict[str, Any]) -> bool:
    """Match filter, dedupe, and features. Optionally match source_used when the query includes it.

    ``requested_song_count`` is ignored so a single “best” archive can cover multiple request sizes.
    """
    sj = _row_settings_payload(row)
    if str(sj.get("filtering_strictness") or row.get("filtering_strictness") or "") != str(
        settings.get("filtering_strictness") or ""
    ):
        return False
    if str(sj.get("dedupe_mode") or row.get("dedupe_mode") or "") != str(settings.get("dedupe_mode") or ""):
        return False
    if bool(sj.get("include_features")) != bool(settings.get("include_features")):
        return False
    if "source_used" in settings:
        if str(sj.get("source_used") or "") != str(settings.get("source_used") or ""):
            return False
    return True


def prune_redundant_archives_for_artist(client: Client, genius_artist_id: str) -> None:
    """For each distinct archive identity, keep the row with the most songs_analyzed; delete the rest."""
    if not genius_artist_id.strip():
        return
    rows = list_archived_analyses_for_artist(genius_artist_id)
    if len(rows) <= 1:
        return
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[_archive_compact_identity_from_row(r)].append(r)
    tbl = client.table("lyric_analysis_archives")
    for group in buckets.values():
        if len(group) <= 1:
            continue
        best = max(
            group,
            key=lambda r: (
                int(r.get("songs_analyzed") or 0),
                str(r.get("created_at") or ""),
            ),
        )
        keep_id = str(best.get("analysis_run_id") or "")
        for r in group:
            rid = str(r.get("analysis_run_id") or "")
            if rid and rid != keep_id:
                tbl.delete().eq("analysis_run_id", rid).execute()


def list_archived_analyses_for_artist(genius_artist_id: str) -> list[dict[str, Any]]:
    client = get_supabase_client()
    resp = (
        client.table("lyric_analysis_archives")
        .select("*")
        .eq("genius_artist_id", genius_artist_id)
        .order("created_at", desc=True)
        .execute()
    )
    return list(resp.data or [])


def get_archived_analysis_for_artist(genius_artist_id: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | None:
    rows = list_archived_analyses_for_artist(genius_artist_id)
    if not rows:
        return None
    if not settings:
        return rows[0]
    matching = [r for r in rows if settings_match(r, settings)]
    if not matching:
        return rows[0]
    return max(
        matching,
        key=lambda r: (
            int(r.get("songs_analyzed") or 0),
            str(r.get("created_at") or ""),
        ),
    )


def search_archives_by_artist_name(substring: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Case-insensitive substring match on stored artist_name."""
    client = get_supabase_client()
    q = substring.strip()
    if not q:
        return []
    resp = (
        client.table("lyric_analysis_archives")
        .select("*")
        .ilike("artist_name", f"%{q}%")
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 200)))
        .execute()
    )
    return list(resp.data or [])


def list_recent_archives(*, limit: int = 20) -> list[dict[str, Any]]:
    client = get_supabase_client()
    resp = (
        client.table("lyric_analysis_archives")
        .select("*")
        .order("created_at", desc=True)
        .limit(max(1, min(limit, 100)))
        .execute()
    )
    return list(resp.data or [])


def load_analysis_from_supabase(analysis_run_id: str) -> dict[str, Any] | None:
    client = get_supabase_client()
    resp = client.table("lyric_analysis_archives").select("*").eq("analysis_run_id", analysis_run_id).limit(1).execute()
    data = list(resp.data or [])
    return data[0] if data else None


def save_analysis_to_supabase(
    artist_metadata: dict[str, Any],
    analysis_results: dict[str, Any],
    settings: dict[str, Any],
    r2_corpus_key: str,
) -> dict[str, Any]:
    client = get_supabase_client()
    aid = str(artist_metadata.get("artist_id") or artist_metadata.get("genius_artist_id") or "").strip()
    new_n = int(analysis_results.get("songs_analyzed", 0))
    identity = _archive_compact_identity_from_settings(settings)

    if aid:
        try:
            prune_redundant_archives_for_artist(client, aid)
        except Exception:
            pass
        peers = [
            r
            for r in list_archived_analyses_for_artist(aid)
            if _archive_compact_identity_from_row(r) == identity
        ]
        if peers:
            best = max(
                peers,
                key=lambda r: (int(r.get("songs_analyzed") or 0), str(r.get("created_at") or "")),
            )
            best_n = int(best.get("songs_analyzed") or 0)
            if best_n > new_n:
                retained = dict(best)
                retained["__lyric_atlas_save_action__"] = "retained_existing"
                return retained

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    payload = {
        "analysis_run_id": run_id,
        "genius_artist_id": aid,
        "artist_name": str(artist_metadata.get("artist_name") or artist_metadata.get("artist") or ""),
        "songs_analyzed": new_n,
        "total_meaningful_tokens": int(analysis_results.get("total_meaningful_tokens", 0)),
        "unique_word_count": int(analysis_results.get("unique_word_count", 0)),
        "lexical_diversity": float(analysis_results.get("lexical_diversity", 0.0)),
        "filtering_strictness": settings.get("filtering_strictness"),
        "dedupe_mode": settings.get("dedupe_mode"),
        "categories_exist": bool(analysis_results.get("categories")),
        "r2_corpus_key": r2_corpus_key,
        "settings_json": settings,
        "top_words_json": analysis_results.get("top_words"),
        "bigrams_json": analysis_results.get("top_bigrams"),
        "categories_json": analysis_results.get("categories"),
        "summary_json": analysis_results.get("summary"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tbl = client.table("lyric_analysis_archives")
    # PostgREST often returns an empty body for write operations unless representation
    # is requested; supabase-py supports returning= on newer postgrest builds.
    try:
        resp = tbl.upsert(payload, on_conflict="analysis_run_id", returning="representation").execute()
    except TypeError:
        resp = tbl.upsert(payload, on_conflict="analysis_run_id").execute()

    rows: list[dict[str, Any]] = list(resp.data or [])
    if not rows:
        verify = tbl.select("*").eq("analysis_run_id", run_id).limit(1).execute()
        rows = list(verify.data or [])

    if not rows:
        raise RuntimeError(
            "Supabase write did not return a row and follow-up select found nothing. "
            "Confirm the `lyric_analysis_archives` table exists, migrations were applied, "
            "and SUPABASE_PRIVATE_KEY is the server secret (not the publishable key)."
        )
    saved = rows[0]
    if aid:
        try:
            prune_redundant_archives_for_artist(client, aid)
        except Exception:
            # Best-effort cleanup; save already succeeded
            pass
    return saved
